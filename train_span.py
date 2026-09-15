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
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import code_accord
from models.bert_kg_encoder import BertKGExtractor


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
