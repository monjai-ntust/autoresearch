"""Canonical CODE-ACCORD span-NER and relation training.

This private module is invoked only by ``pipeline.py _train-encoder`` with
arguments derived from the validated run configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utils.encoder import data as code_accord
from utils.encoder.network import BertKGExtractor
from utils.common.artifact_io import (
    DataContractError,
    _load_manifest,
    _quarantine_run_artifact,
    _require_fields,
    _same_run_seal_path,
    _validate_recovery_provenance,
    _validate_same_run_seal,
    _write_recovery_provenance,
    _write_same_run_seal,
    _next_recovery_path,
    iter_jsonl,
    sha256_file,
)
from utils.common.config import PipelineConfig, load_pipeline_config
from utils.common.paths import RunLayout, discover_source_root
from utils.common.records import Candidate
from utils.encoder.model import (
    canonical_trainer_arguments,
    generate_candidates,
    plan_training,
    validate_prediction_cache,
)
from utils.evaluation.publication import assemble_seed_candidates
from utils.rag.runner import require_consistent_encoder_identity


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--model-local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--max-span-width", type=int, required=True)
    parser.add_argument("--re-weight", type=float, required=True)
    parser.add_argument("--re-no-rel-weight", type=float, required=True)
    parser.add_argument("--neg-sample-ratio", type=float, required=True)
    parser.add_argument("--focal-gamma", type=float, required=True)
    parser.add_argument("--label-smoothing", type=float, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--re-comparison-boost", type=float, required=True)
    parser.add_argument("--re-boost-adaptive-steps", type=int, required=True)
    parser.add_argument("--re-boost-adaptive-threshold", type=float, required=True)
    parser.add_argument("--re-boost-adaptive-threshold2", type=float, required=True)
    parser.add_argument("--re-boost-mid", type=float, required=True)
    parser.add_argument("--re-boost-end", type=float, required=True)
    parser.add_argument("--save-best-to", required=True)
    parser.add_argument("--save-last-to", required=True)
    parser.add_argument("--progress-log", required=True)
    parser.add_argument("--run-summary-out", required=True)
    parser.add_argument("--resume-from")
    return parser.parse_args(argv)


def _total_memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _append_progress(args, message):
    path = Path(args.progress_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(value, partial)
    partial.replace(path)


def _atomic_json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _save_best_checkpoint(args, model, step, metrics):
    _atomic_torch_save(
        {"encoder": model.state_dict(), "step": step, "metrics": metrics},
        args.save_best_to,
    )


def _canonical_sampler(train_loader):
    sampler = getattr(train_loader, "sampler", None)
    if not hasattr(sampler, "state_dict") or not hasattr(sampler, "load_state_dict"):
        raise ValueError("canonical training requires a resumable train sampler")
    return sampler


def _restart_payload(
    args,
    model,
    optimizer,
    scheduler,
    train_loader,
    *,
    next_step,
    best_metrics,
    best_step,
    boost_adaptive_triggered,
    boost_adaptive_switched,
):
    return {
        "format_version": "train-span-restart-2.0",
        "seed": args.seed,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "max_steps": args.max_steps,
        "encoder": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "next_step": next_step,
        "best_metrics": dict(best_metrics),
        "best_step": best_step,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "train_sampler_state": _canonical_sampler(train_loader).state_dict(),
        "boost_adaptive_triggered": boost_adaptive_triggered,
        "boost_adaptive_switched": boost_adaptive_switched,
    }


def _load_restart(args, model, optimizer, scheduler, train_loader, _device=None):
    """Restore a complete deterministic training state, including v1 states."""

    state = torch.load(args.resume_from, map_location="cpu", weights_only=False)
    common = {
        "format_version",
        "seed",
        "model_name",
        "model_revision",
        "max_steps",
        "encoder",
        "optimizer",
        "scheduler",
        "next_step",
        "best_metrics",
        "best_step",
        "python_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "train_sampler_state",
        "boost_adaptive_triggered",
        "boost_adaptive_switched",
    }
    if not isinstance(state, dict):
        raise ValueError("restart state has an invalid field contract")
    version = state.get("format_version")
    required = common | ({"re_head_finetune_active"} if version == "train-span-restart-1.0" else set())
    if version not in {"train-span-restart-1.0", "train-span-restart-2.0"} or set(state) != required:
        raise ValueError("restart state has an invalid field contract")
    if state.get("re_head_finetune_active") is True:
        raise ValueError("restart state belongs to a removed noncanonical training branch")
    expected = {
        "seed": args.seed,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "max_steps": args.max_steps,
    }
    for field, value in expected.items():
        if state[field] != value:
            raise ValueError(
                f"restart state {field} mismatch: expected {value!r}, got {state[field]!r}"
            )
    model.load_state_dict(state["encoder"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _canonical_sampler(train_loader).load_state_dict(state["train_sampler_state"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return state


def focal_loss(logits, targets, gamma=2.0, weights=None, label_smoothing=0.0):
    cross_entropy = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    loss = (1 - torch.exp(-cross_entropy)) ** gamma * cross_entropy
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def _build_span_labels(gold_entities, max_span_width, entity_type2id):
    return {
        (start, end): entity_type2id[entity_type]
        for start, end, entity_type in gold_entities
        if end - start + 1 <= max_span_width and entity_type in entity_type2id
    }


def compute_span_loss(
    model,
    batch,
    device,
    entity_type2id,
    *,
    re_weight,
    neg_sample_ratio,
    max_span_width,
    focal_gamma,
    label_smoothing,
    re_comparison_boost,
    re_no_rel_weight,
):
    """Compute the approved span-NER plus relation cross-entropy objective."""

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_ids_list = batch["word_ids"]
    gold_entities_list = batch["gold_entities"]
    gold_relations_list = batch["gold_relations"]
    num_words_list = batch["num_words"]

    hidden = model.encode(input_ids=input_ids, attention_mask=attention_mask)
    span_losses = []
    relation_losses = []

    for batch_index in range(input_ids.size(0)):
        gold_entities = gold_entities_list[batch_index]
        gold_relations = gold_relations_list[batch_index]
        span_logits, candidates = model.forward_span_ner(
            hidden[batch_index],
            word_ids_list[batch_index],
            num_words_list[batch_index],
            max_span_width,
        )
        if not candidates:
            continue

        gold_labels = _build_span_labels(
            gold_entities,
            max_span_width,
            entity_type2id,
        )
        targets = torch.tensor(
            [gold_labels.get(candidate, 0) for candidate in candidates],
            device=device,
            dtype=torch.long,
        )
        positive_mask = targets > 0
        negative_indices = (targets == 0).nonzero(as_tuple=True)[0]
        negative_count = max(int(positive_mask.sum().item() * neg_sample_ratio), 1)
        if len(negative_indices) > negative_count:
            permutation = torch.randperm(len(negative_indices), device=device)[:negative_count]
            negative_indices = negative_indices[permutation]
        kept = torch.cat([positive_mask.nonzero(as_tuple=True)[0], negative_indices])
        if len(kept):
            span_losses.append(
                focal_loss(
                    span_logits[kept],
                    targets[kept],
                    gamma=focal_gamma,
                    label_smoothing=label_smoothing,
                )
            )

        gold_spans = {(start, end) for start, end, _ in gold_entities}
        with torch.no_grad():
            predicted_types = span_logits.argmax(dim=-1).tolist()
            predicted_confidences = (
                torch.softmax(span_logits, dim=-1).max(dim=-1).values.tolist()
            )
        predicted_spans = {
            span
            for span, entity_type, confidence in zip(
                candidates, predicted_types, predicted_confidences
            )
            if entity_type > 0 and confidence >= 0.5
        }
        relation_spans = list(predicted_spans | gold_spans)
        if len(relation_spans) < 2:
            continue
        pairs = [
            (head, tail)
            for head in relation_spans
            for tail in relation_spans
            if head != tail
        ]
        relation_lookup = {
            (head, tail): relation_id
            for head, tail, relation_id in gold_relations
        }
        pair_targets = torch.tensor(
            [relation_lookup.get(pair, code_accord.NO_REL_ID) for pair in pairs],
            device=device,
            dtype=torch.long,
        )
        relation_logits = model.forward_re(
            hidden[batch_index],
            word_ids_list[batch_index],
            pairs,
        )
        class_weights = None
        if re_comparison_boost > 1.0 or re_no_rel_weight != 1.0:
            class_weights = relation_logits.new_ones(relation_logits.size(-1))
            class_weights[code_accord.NO_REL_ID] = re_no_rel_weight
            if re_comparison_boost > 1.0:
                for relation_id in code_accord.COMPARISON_REL_IDS:
                    class_weights[relation_id] = re_comparison_boost
        relation_losses.append(
            F.cross_entropy(relation_logits, pair_targets, weight=class_weights)
        )

    ner_loss = torch.stack(span_losses).mean() if span_losses else hidden.new_tensor(0.0)
    re_loss = (
        torch.stack(relation_losses).mean()
        if relation_losses
        else hidden.new_tensor(0.0)
    )
    return ner_loss + re_weight * re_loss, ner_loss.detach(), re_loss.detach()


def _prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_span(
    model,
    dataloader,
    device,
    entity_type2id,
    id2entity_type,
    *,
    max_span_width,
    verbose=False,
):
    """Evaluate strict span and end-to-end triple F1 on development data."""

    del entity_type2id
    model.eval()
    ner_tp = ner_fp = ner_fn = 0
    triple_tp = triple_fp = triple_fn = 0
    n_examples = 0
    total_predicted_entities = total_gold_entities = 0
    total_predicted_relations = total_gold_relations = total_relation_pairs = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden = model.encode(input_ids=input_ids, attention_mask=attention_mask)
            for batch_index in range(input_ids.size(0)):
                n_examples += 1
                gold_entities = batch["gold_entities"][batch_index]
                gold_relations = batch["gold_relations"][batch_index]
                span_logits, candidates = model.forward_span_ner(
                    hidden[batch_index],
                    batch["word_ids"][batch_index],
                    batch["num_words"][batch_index],
                    max_span_width,
                )
                gold_triples = set(gold_relations)
                total_gold_entities += len(gold_entities)
                total_gold_relations += len(gold_triples)
                if not candidates:
                    ner_fn += len(gold_entities)
                    triple_fn += len(gold_triples)
                    continue

                probabilities = torch.softmax(span_logits, dim=-1)
                predicted = [
                    (start, end, id2entity_type[entity_type], confidence)
                    for (start, end), entity_type, confidence in zip(
                        candidates,
                        span_logits.argmax(dim=-1).tolist(),
                        probabilities.max(dim=-1).values.tolist(),
                    )
                    if entity_type > 0 and confidence >= 0.5
                ]
                taken = set()
                filtered = []
                for start, end, entity_type, _confidence in sorted(
                    predicted, key=lambda item: -item[3]
                ):
                    if not any(not (end < other_start or other_end < start) for other_start, other_end in taken):
                        filtered.append((start, end, entity_type))
                        taken.add((start, end))
                total_predicted_entities += len(filtered)

                predicted_entities = set(filtered)
                gold_entity_set = set(gold_entities)
                ner_tp += len(predicted_entities & gold_entity_set)
                ner_fp += len(predicted_entities - gold_entity_set)
                ner_fn += len(gold_entity_set - predicted_entities)

                spans = [(start, end) for start, end, _ in filtered]
                pairs = [
                    (head, tail)
                    for head in spans
                    for tail in spans
                    if head != tail
                ]
                total_relation_pairs += len(pairs)
                if pairs:
                    relation_ids = model.forward_re(
                        hidden[batch_index],
                        batch["word_ids"][batch_index],
                        pairs,
                    ).argmax(dim=-1).tolist()
                    predicted_triples = {
                        (head, tail, relation_id)
                        for (head, tail), relation_id in zip(pairs, relation_ids)
                        if relation_id != code_accord.NO_REL_ID
                    }
                else:
                    predicted_triples = set()
                total_predicted_relations += len(predicted_triples)
                triple_tp += len(predicted_triples & gold_triples)
                triple_fp += len(predicted_triples - gold_triples)
                triple_fn += len(gold_triples - predicted_triples)

    _, _, ner_f1 = _prf(ner_tp, ner_fp, ner_fn)
    _, _, triple_f1 = _prf(triple_tp, triple_fp, triple_fn)
    if verbose:
        ner_precision, ner_recall, _ = _prf(ner_tp, ner_fp, ner_fn)
        triple_precision, triple_recall, _ = _prf(triple_tp, triple_fp, triple_fn)
        no_relation_fraction = 1.0 - total_predicted_relations / max(total_relation_pairs, 1)
        print(
            f"    [diag] pred_ents={total_predicted_entities} gold_ents={total_gold_entities} | "
            f"pred_rels={total_predicted_relations} gold_rels={total_gold_relations} | "
            f"re_pairs={total_relation_pairs} NO_REL%={no_relation_fraction:.2f}"
        )
        print(
            f"    [diag] NER  P={ner_precision:.3f} R={ner_recall:.3f} F1={ner_f1:.3f} | "
            f"Triple P={triple_precision:.3f} R={triple_recall:.3f} F1={triple_f1:.3f}"
        )
        print(
            f"    [diag] triple_tp={triple_tp} triple_fp={triple_fp} triple_fn={triple_fn}"
        )
    return {"ner_f1": ner_f1, "triple_f1": triple_f1, "n_examples": n_examples}


def cycle(loader):
    while True:
        yield from loader


def _validate_paths(args):
    prepared_root = Path(args.prepared_dir).resolve()
    if Path(args.model_cache_dir).resolve() != prepared_root.parent / "inputs" / "huggingface":
        raise ValueError(
            "canonical training requires the run-local inputs/huggingface model cache"
        )


def main(argv=None):
    args = parse_args(argv)
    _validate_paths(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_kwargs = {
        "add_prefix_space": True,
        "revision": args.model_revision,
        "cache_dir": args.model_cache_dir,
    }
    if args.model_local_files_only:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    train_loader, development_loader, test_loader = code_accord.build_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        prepared_dir=args.prepared_dir,
    )
    if test_loader is not None:
        raise ValueError("canonical data adapter unexpectedly loaded the test split")

    entity_type2id = {
        entity_type: index + 1
        for index, entity_type in enumerate(code_accord.ENTITY_TYPES)
    }
    id2entity_type = {value: key for key, value in entity_type2id.items()}
    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=code_accord.NUM_BIO_TAGS,
        num_relations=code_accord.NUM_RELATIONS,
        num_entity_types=len(code_accord.ENTITY_TYPES),
        max_span_width=args.max_span_width,
        model_revision=args.model_revision,
        model_cache_dir=args.model_cache_dir,
        model_local_files_only=args.model_local_files_only,
    ).to(device)

    # Preserve the historical initialization sequence: construct the unused
    # 2H head above, then replace it with the approved 3H context head.
    model.re_context_span = True
    hidden_size = model.backbone.hidden_size
    model.re_head = torch.nn.Sequential(
        torch.nn.Linear(hidden_size * 3, hidden_size),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_size, code_accord.NUM_RELATIONS),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    best_metrics = {"triple_f1": -1.0}
    best_step = -1
    boost_adaptive_triggered = False
    boost_adaptive_switched = False
    step = 0
    resumed_from = None
    if args.resume_from:
        restart = _load_restart(
            args, model, optimizer, scheduler, train_loader, device
        )
        step = restart["next_step"]
        best_metrics = dict(restart["best_metrics"])
        best_step = restart["best_step"]
        boost_adaptive_triggered = restart["boost_adaptive_triggered"]
        boost_adaptive_switched = restart["boost_adaptive_switched"]
        resumed_from = str(Path(args.resume_from))

    train_iterator = cycle(train_loader)
    model.train()
    started = time.time()
    while step < args.max_steps:
        optimizer.zero_grad()
        batch = next(train_iterator)

        progress = step / max(args.max_steps - 1, 1)
        boost = args.re_comparison_boost - (
            args.re_comparison_boost - args.re_boost_end
        ) * progress
        if boost_adaptive_switched:
            boost = args.re_boost_end
        if (
            not boost_adaptive_triggered
            and step == args.re_boost_adaptive_steps
        ):
            boost_adaptive_triggered = True
            model.eval()
            with torch.no_grad():
                adaptive_metrics = evaluate_span(
                    model,
                    development_loader,
                    device,
                    entity_type2id,
                    id2entity_type,
                    max_span_width=args.max_span_width,
                )
            model.train()
            adaptive_f1 = adaptive_metrics["triple_f1"]
            if adaptive_f1 >= args.re_boost_adaptive_threshold2:
                boost_adaptive_switched = True
                boost = args.re_boost_end
            elif adaptive_f1 >= args.re_boost_adaptive_threshold:
                boost_adaptive_switched = True
                boost = args.re_boost_mid

        loss, ner_loss, relation_loss = compute_span_loss(
            model,
            batch,
            device,
            entity_type2id,
            re_weight=args.re_weight,
            neg_sample_ratio=args.neg_sample_ratio,
            max_span_width=args.max_span_width,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            re_comparison_boost=boost,
            re_no_rel_weight=args.re_no_rel_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            elapsed_ms = (time.time() - started) * 1000 / max(step, 1)
            message = (
                f"[Step {step:04d}] L={loss.item():.4f} NER={ner_loss.item():.4f} "
                f"RE={relation_loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed_ms:.0f}ms/step"
            )
            print(message)
            sys.stdout.flush()
            _append_progress(args, message)

        if step > 0 and step % args.eval_every == 0:
            metrics = evaluate_span(
                model,
                development_loader,
                device,
                entity_type2id,
                id2entity_type,
                max_span_width=args.max_span_width,
                verbose=True,
            )
            selected = metrics["triple_f1"] > best_metrics["triple_f1"]
            if selected:
                best_metrics = dict(metrics)
                best_step = step
                _save_best_checkpoint(args, model, step, metrics)
            message = (
                f"[Eval @ {step}] NER={metrics['ner_f1']:.4f} "
                f"Triple={metrics['triple_f1']:.4f}{' *' if selected else ''}"
            )
            print(message)
            sys.stdout.flush()
            _append_progress(args, message)
            model.train()
            _atomic_torch_save(
                _restart_payload(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    next_step=step + 1,
                    best_metrics=best_metrics,
                    best_step=best_step,
                    boost_adaptive_triggered=boost_adaptive_triggered,
                    boost_adaptive_switched=boost_adaptive_switched,
                ),
                args.save_last_to,
            )
        step += 1

    metrics = evaluate_span(
        model,
        development_loader,
        device,
        entity_type2id,
        id2entity_type,
        max_span_width=args.max_span_width,
    )
    if metrics["triple_f1"] > best_metrics["triple_f1"]:
        best_metrics = dict(metrics)
        best_step = step
        _save_best_checkpoint(args, model, step, metrics)
    _atomic_torch_save(
        _restart_payload(
            args,
            model,
            optimizer,
            scheduler,
            train_loader,
            next_step=step,
            best_metrics=best_metrics,
            best_step=best_step,
            boost_adaptive_triggered=boost_adaptive_triggered,
            boost_adaptive_switched=boost_adaptive_switched,
        ),
        args.save_last_to,
    )

    elapsed_seconds = time.time() - started
    final_message = (
        f"=== BEST DEV (step {best_step}) ===\n"
        f"  NER={best_metrics['ner_f1']:.4f} "
        f"Triple={best_metrics['triple_f1']:.4f}\n"
        f"  time={elapsed_seconds:.1f}s"
    )
    print(final_message)
    sys.stdout.flush()
    _append_progress(args, final_message)
    summary = {
        "schema_version": "train-span-summary-1.0",
        "status": "completed",
        "canonical_mode": True,
        "dataset": "accord",
        "seed": args.seed,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "completed_steps": step,
        "selected_step": best_step,
        "selection_metric": "triple_f1",
        "selected_metrics": best_metrics,
        "test_evaluated": False,
        "resumed_from": resumed_from,
        "elapsed_seconds": elapsed_seconds,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "total_memory_bytes": _total_memory_bytes(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
            "cuda_device_memory_bytes": (
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                if torch.cuda.is_available()
                else None
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }
    _atomic_json_write(summary, args.run_summary_out)
    return summary


def train_and_generate_development(
    layout: RunLayout,
    config: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, object], Path]:
    """Train all seeds and assemble authenticated development candidates."""

    encoder_identities = []
    training_hardware: dict[str, object] = {}
    for seed in config.value["training_seeds"]:
        dry_manifest = layout.resolve(f"manifests/model-train-dry-run-seed-{seed}.json")
        if not dry_manifest.is_file():
            plan_training(
                layout, config, execution_mode="dry-run", training_seed=seed
            )
        _validate_training_stage(layout, config, seed, execution_mode="dry-run")

        live_manifest = layout.resolve(f"manifests/model-train-live-seed-{seed}.json")
        if not live_manifest.is_file():
            plan_training(layout, config, execution_mode="live", training_seed=seed)
        training = _validate_training_stage(
            layout, config, seed, execution_mode="live"
        )
        checkpoint_identity = _load_manifest(
            layout.resolve(
                f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True
            ),
            "checkpoint manifest",
        )
        encoder_identities.append(
            {
                "base_model": checkpoint_identity["base_model"],
                "base_model_revision": checkpoint_identity["base_model_revision"],
            }
        )
        if not training_hardware and isinstance(training.get("environment"), dict):
            training_hardware = training["environment"]
        _resume_generation_stage(layout, config, seed=seed, split="development")

    encoder_identity = require_consistent_encoder_identity(encoder_identities)
    development_candidates = layout.resolve(
        "predictions/dev/development-candidates.jsonl"
    )
    if not development_candidates.is_file():
        assemble_seed_candidates(layout, config, split="development")
    _validate_candidate_assembly(layout, config, "development")
    return encoder_identity, training_hardware, development_candidates


def generate_test_candidates(
    layout: RunLayout,
    config: PipelineConfig,
) -> Path:
    """Generate and assemble authenticated final-test candidates."""

    for seed in config.value["training_seeds"]:
        _resume_generation_stage(layout, config, seed=seed, split="test")
    test_candidates = layout.resolve("predictions/test/candidates.jsonl")
    if not test_candidates.is_file():
        assemble_seed_candidates(layout, config, split="test")
    _validate_candidate_assembly(layout, config, "test")
    return test_candidates


def _validate_private_training_inputs(layout: RunLayout, config) -> None:
    """Authenticate the run-local files consumed by the internal trainer."""

    acquisition_path = layout.resolve(
        "manifests/02-input-acquisition-manifest.json", must_exist=True
    )
    preparation_path = layout.resolve(
        "manifests/03-data-preparation-manifest.json", must_exist=True
    )
    acquisition = _load_manifest(acquisition_path, "acquisition manifest")
    preparation = _load_manifest(preparation_path, "preparation manifest")
    archive = acquisition.get("archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("path"), str):
        raise DataContractError("training acquisition manifest lacks archive identity")
    archive_path = layout.resolve(archive["path"], must_exist=True)
    if (
        acquisition.get("dataset_id") != config.value["dataset"]["dataset_id"]
        or archive.get("sha256") != sha256_file(archive_path)
        or preparation.get("protocol_id") != config.value["protocol_id"]
        or preparation.get("dataset_id") != config.value["dataset"]["dataset_id"]
        or preparation.get("byte_identical_independent_materializations") is not True
        or preparation.get("acquisition_manifest_sha256")
        != sha256_file(acquisition_path)
        or preparation.get("archive_sha256") != archive.get("sha256")
    ):
        raise DataContractError(
            "private trainer inputs are not bound to one acquisition/preparation"
        )
    artifacts = preparation.get("artifacts")
    if not isinstance(artifacts, list):
        raise DataContractError("preparation manifest lacks its artifact ledger")
    by_path = {
        item.get("path"): item
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for relative in ("train.jsonl", "development.jsonl", "split-manifest.json"):
        record = by_path.get(relative)
        path = layout.resolve(f"data-prepared/{relative}", must_exist=True)
        if (
            not isinstance(record, dict)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise DataContractError(
                f"private trainer input differs from preparation: {relative}"
            )
    split = _load_manifest(
        layout.resolve("data-prepared/split-manifest.json", must_exist=True),
        "split manifest",
    )
    expected_split = config.value["split"]
    if (
        split.get("split_id") != expected_split["split_id"]
        or split.get("seed") != expected_split["seed"]
        or split.get("train_count") != expected_split["train_sentences"]
        or split.get("development_count")
        != expected_split["development_sentences"]
        or split.get("test_count") != expected_split["test_sentences"]
    ):
        raise DataContractError("private trainer split differs from the selected config")


def _validate_training_stage(
    layout: RunLayout, config, seed: int, *, execution_mode: str
) -> dict:
    manifest_path = layout.resolve(
        f"manifests/model-train-{execution_mode}-seed-{seed}.json", must_exist=True
    )
    manifest = _load_manifest(manifest_path, "training manifest")
    _require_fields(
        manifest,
        {
            "stage": "model-train",
            "execution_mode": execution_mode,
            "status": "planned" if execution_mode == "dry-run" else "completed",
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
            "expected_checkpoint_dir": f"checkpoints/seed-{seed}",
        },
        f"seed-{seed} training manifest",
    )
    if manifest.get("recipe") != config.value["training"]:
        raise DataContractError(f"seed-{seed} training recipe differs from this run")
    if execution_mode == "dry-run":
        return manifest

    checkpoint_relative = f"checkpoints/seed-{seed}/checkpoint.pt"
    checkpoint_manifest_relative = f"checkpoints/seed-{seed}/checkpoint-manifest.json"
    expected_outputs = {
        "checkpoint": checkpoint_relative,
        "checkpoint_manifest": checkpoint_manifest_relative,
        "restart_state": f"checkpoints/seed-{seed}/restart-state.pt",
        "training_summary": f"checkpoints/seed-{seed}/training-summary.json",
        "progress_log": f"logs/model-train-seed-{seed}.log",
        "dataset_compatibility_report": "audit/model-training-dataset-compatibility.json",
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or any(
        outputs.get(field) != relative for field, relative in expected_outputs.items()
    ):
        raise DataContractError(f"seed-{seed} training outputs use unexpected paths")
    for relative in outputs.values():
        if isinstance(relative, str):
            layout.resolve(relative, must_exist=True)

    checkpoint_path = layout.resolve(checkpoint_relative, must_exist=True)
    identity_path = layout.resolve(checkpoint_manifest_relative, must_exist=True)
    identity = _load_manifest(identity_path, "checkpoint manifest")
    if identity.get("schema_version") not in {
        "phase-b-model-checkpoint-manifest-2.0",
        "phase-b-model-checkpoint-manifest-3.0",
        "phase-b-model-checkpoint-manifest-4.0",
    }:
        raise DataContractError(f"seed-{seed} checkpoint schema is unsupported")
    _require_fields(
        identity,
        {
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
            "base_model": config.value["training"]["base_model"],
            "base_model_revision": config.value["training"]["base_model_revision"],
            "max_span_width": config.value["training"]["max_span_width"],
            "context_between_spans": config.value["training"]["context_between_spans"],
        },
        f"seed-{seed} checkpoint manifest",
    )
    revision = identity.get("base_model_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise DataContractError(f"seed-{seed} encoder revision is not immutable")
    digest = identity.get("checkpoint_sha256")
    if not isinstance(digest, str) or sha256_file(checkpoint_path) != digest:
        raise DataContractError(f"seed-{seed} checkpoint bytes differ from their manifest")

    known_inputs = {
        "checkout_manifest_sha256": "manifests/00-checkout-manifest.json",
        "acquisition_manifest_sha256": "manifests/02-input-acquisition-manifest.json",
        "preparation_manifest_sha256": "manifests/03-data-preparation-manifest.json",
        "split_manifest_sha256": "data-prepared/split-manifest.json",
        "train_jsonl_sha256": "data-prepared/train.jsonl",
        "development_jsonl_sha256": "data-prepared/development.jsonl",
    }
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise DataContractError(f"seed-{seed} training manifest lacks input hashes")
    for field, relative in known_inputs.items():
        path = layout.resolve(relative, must_exist=True)
        if inputs.get(field) != sha256_file(path):
            raise DataContractError(f"seed-{seed} training input changed: {relative}")
    for field, relative in {
        "split_manifest_sha256": "data-prepared/split-manifest.json",
        "acquisition_manifest_sha256": "manifests/02-input-acquisition-manifest.json",
        "train_jsonl_sha256": "data-prepared/train.jsonl",
        "development_jsonl_sha256": "data-prepared/development.jsonl",
    }.items():
        if field in identity and identity[field] != sha256_file(
            layout.resolve(relative, must_exist=True)
        ):
            raise DataContractError(f"seed-{seed} checkpoint identity changed: {relative}")
    restart = layout.resolve(expected_outputs["restart_state"], must_exist=True)
    if manifest.get("resume", {}).get("restart_state_sha256") != sha256_file(restart):
        raise DataContractError(f"seed-{seed} restart state differs from training manifest")
    compatibility = layout.resolve(
        expected_outputs["dataset_compatibility_report"], must_exist=True
    )
    if (
        identity.get("dataset_compatibility_sha256") is not None
        and identity.get("dataset_compatibility_sha256") != sha256_file(compatibility)
    ):
        raise DataContractError(f"seed-{seed} compatibility report changed")
    summary = _load_manifest(
        layout.resolve(expected_outputs["training_summary"], must_exist=True),
        "training summary",
    )
    if (
        summary.get("status") != "completed"
        or summary.get("canonical_mode") is not True
        or summary.get("seed") != seed
        or summary.get("test_evaluated") is not False
    ):
        raise DataContractError(f"seed-{seed} training summary is not publication-safe")
    return manifest


def _validate_generation_stage(
    layout: RunLayout,
    config,
    seed: int,
    split: str,
    *,
    require_seal: bool = True,
) -> dict:
    manifest_path = layout.resolve(
        f"manifests/model-generate-candidates-live-seed-{seed}-{split}.json",
        must_exist=True,
    )
    manifest = _load_manifest(manifest_path, "candidate-generation manifest")
    _require_fields(
        manifest,
        {
            "stage": "model-generate-candidates",
            "status": "completed",
            "execution_mode": "live",
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
        },
        f"seed-{seed} {split} generation manifest",
    )
    directory = "dev" if split == "development" else "test"
    candidates_relative = f"predictions/{directory}/seed-{seed}-candidates.jsonl"
    checkpoint_relative = f"checkpoints/seed-{seed}/checkpoint-manifest.json"
    prepared_relative = f"data-prepared/{split}.jsonl"
    checkpoint_path = layout.resolve(checkpoint_relative, must_exist=True)
    prepared_path = layout.resolve(prepared_relative, must_exist=True)
    inputs = manifest.get("inputs", {})
    checkpoint_input = inputs.get("checkpoint_manifest", {})
    prepared_input = inputs.get("prepared_sentences", {})
    if (
        manifest.get("candidates_output") != candidates_relative
        or checkpoint_input.get("path") != checkpoint_relative
        or checkpoint_input.get("sha256") != sha256_file(checkpoint_path)
        or prepared_input.get("path") != prepared_relative
        or prepared_input.get("sha256") != sha256_file(prepared_path)
    ):
        raise DataContractError(
            f"seed-{seed} {split} candidates are not bound to same-run inputs"
        )
    checkpoint = _load_manifest(checkpoint_path, "checkpoint manifest")
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint.get(
        "checkpoint_sha256"
    ):
        raise DataContractError(f"seed-{seed} {split} checkpoint identity differs")
    if require_seal:
        _validate_same_run_seal(layout, manifest_path, manifest)
    return manifest


def _validate_candidate_assembly(layout: RunLayout, config, split: str) -> None:
    directory = "dev" if split == "development" else "test"
    combined_relative = (
        "predictions/dev/development-candidates.jsonl"
        if split == "development"
        else "predictions/test/candidates.jsonl"
    )
    combined = layout.resolve(combined_relative, must_exist=True)
    rows: list[dict] = []
    expected_bytes = b""
    files = []
    for seed in config.value["training_seeds"]:
        relative = f"predictions/{directory}/seed-{seed}-candidates.jsonl"
        path = layout.resolve(relative, must_exist=True)
        seed_rows = [value for _, value in iter_jsonl(path)]
        if any(row.get("training_seed") != seed for row in seed_rows):
            raise DataContractError(f"{relative} contains another training seed")
        rows.extend(seed_rows)
        expected_bytes += path.read_bytes()
        files.append(
            {
                "training_seed": seed,
                "path": relative,
                "sha256": sha256_file(path),
                "candidate_count": len(seed_rows),
                "candidates": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "example_id": row.get("example_id"),
                    }
                    for row in seed_rows
                ],
            }
        )
    if combined.read_bytes() != expected_bytes:
        raise DataContractError(
            f"{combined_relative} is not the exact ordered same-run seed assembly"
        )
    if split == "development":
        index = _load_manifest(
            layout.resolve("predictions/dev/candidate-index.json", must_exist=True),
            "development candidate index",
        )
        expected = {
            "schema_version": "phase-b-candidate-index-1.0",
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "split_id": "CODE-SPLIT-1:development",
            "split_manifest_sha256": sha256_file(
                layout.resolve("data-prepared/split-manifest.json", must_exist=True)
            ),
            "files": files,
            "candidate_count": len(rows),
        }
        if index != expected:
            raise DataContractError("development candidate index differs from seed ledgers")


def _resume_generation_stage(
    layout: RunLayout, config, *, seed: int, split: str
) -> dict:
    manifest_path = layout.resolve(
        f"manifests/model-generate-candidates-live-seed-{seed}-{split}.json"
    )
    seal_path = _same_run_seal_path(manifest_path)
    if manifest_path.is_file():
        manifest = _validate_generation_stage(
            layout, config, seed, split, require_seal=False
        )
        if not seal_path.is_file():
            _write_same_run_seal(layout, manifest_path, manifest)
        return _validate_generation_stage(layout, config, seed, split)

    directory = "dev" if split == "development" else "test"
    prepared = layout.resolve(f"data-prepared/{split}.jsonl", must_exist=True)
    checkpoint_manifest = layout.resolve(
        f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True
    )
    checkpoint_blob = layout.resolve(
        f"checkpoints/seed-{seed}/checkpoint.pt", must_exist=True
    )
    candidates = layout.resolve(f"predictions/{directory}/seed-{seed}-candidates.jsonl")
    live_ledger = layout.resolve(
        f"predictions/{directory}/seed-{seed}-prediction-ledger.jsonl"
    )
    bindings = {"prepared": prepared, "checkpoint_manifest": checkpoint_manifest}
    identity: dict[str, object] = {"training_seed": seed, "split": split}
    cache: Path | None = None
    if live_ledger.is_file():
        try:
            validate_prediction_cache(
                layout,
                config,
                sentences_path=prepared,
                checkpoint_manifest_path=checkpoint_manifest,
                cache_ledger_path=live_ledger,
            )
        except DataContractError:
            _quarantine_run_artifact(
                layout, live_ledger, f"generation-seed-{seed}-{split}"
            )
        else:
            cache = _next_recovery_path(
                layout,
                f"generation-seed-{seed}-{split}",
                "prediction-ledger.jsonl",
            )
            os.replace(live_ledger, cache)
            _write_recovery_provenance(
                layout,
                cache,
                kind="prediction-ledger",
                identity=identity,
                bindings=bindings,
            )
    if cache is None:
        directory_path = layout.resolve(
            f"inputs/recovery/generation-seed-{seed}-{split}"
        )
        if directory_path.is_dir():
            for recovery_candidate in sorted(
                directory_path.glob("*-prediction-ledger.jsonl"), reverse=True
            ):
                try:
                    _validate_recovery_provenance(
                        layout,
                        recovery_candidate,
                        kind="prediction-ledger",
                        identity=identity,
                        bindings=bindings,
                    )
                    validate_prediction_cache(
                        layout,
                        config,
                        sentences_path=prepared,
                        checkpoint_manifest_path=checkpoint_manifest,
                        cache_ledger_path=recovery_candidate,
                    )
                except DataContractError:
                    continue
                cache = recovery_candidate
                break
    for partial in (candidates, seal_path):
        _quarantine_run_artifact(
            layout, partial, f"generation-seed-{seed}-{split}"
        )
    manifest = generate_candidates(
        layout,
        config,
        execution_mode="live",
        sentences_path=prepared,
        checkpoint_manifest_path=checkpoint_manifest,
        candidates_out_path=candidates,
        prediction_ledger_path=None,
        cache_ledger_path=cache,
        checkpoint_blob_path=checkpoint_blob,
        base_model=None,
        device=None,
    )
    _write_same_run_seal(layout, manifest_path, manifest)
    return _validate_generation_stage(layout, config, seed, split)


def run_private_trainer(
    arguments: list[str], *, default_config: str
) -> int:
    """Resolve the trainer only from a run ID, approved seed, and tracked config."""

    parser = argparse.ArgumentParser(prog="pipeline.py _train-encoder")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--config", default=default_config)
    args = parser.parse_args(arguments)
    source_root = discover_source_root()
    config = load_pipeline_config(source_root, args.config)
    layout = RunLayout(source_root=source_root, run_id=args.run_id)
    layout.require_existing()
    if args.training_seed not in config.value["training_seeds"]:
        raise DataContractError(
            f"training seed {args.training_seed} is not approved by the run config"
        )

    from stages.preparation import _validate_existing_checkout

    _validate_existing_checkout(layout, config)
    _validate_training_stage(
        layout, config, args.training_seed, execution_mode="dry-run"
    )
    _validate_private_training_inputs(layout, config)
    # Import resolution occurs only after the narrow namespace contract passes.
    # The trainer never sees user-supplied artifact paths.
    main(canonical_trainer_arguments(layout, config, args.training_seed))
    return 0
