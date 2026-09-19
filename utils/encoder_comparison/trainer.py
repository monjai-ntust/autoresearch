"""Historical-derived Phase G trainer and prediction-ledger producer.

The loss, span enumeration, ordered relation pairs, optimizer, development
selection, and effective A20 schedule come from historical ``train_span.py``
blob ``104ca6e4afcb5122228fff5c73227c6b3b3edb40`` at commit
``9feafa4029e65ab48ecfb2f452f4b0fabbff0826``. Run-local paths, immutable
model loading, exact restart state, and omission of test access are explicit
adapters to the current artifact contract.
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
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utils.common.constants import PROTOCOL_ID
from utils.encoder_comparison import data as code_accord
from utils.encoder_comparison.network import HistoricalKGExtractor


HISTORICAL_SOURCE_COMMIT = "9feafa4029e65ab48ecfb2f452f4b0fabbff0826"
HISTORICAL_TRAINER_BLOB = "104ca6e4afcb5122228fff5c73227c6b3b3edb40"
CONFIDENCE_PRECISION = 6


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("train", "predict"), required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected-hidden-size", type=int, required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--model-local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-span-width", type=int, required=True)
    parser.add_argument("--context-between-spans", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--re-weight", type=float)
    parser.add_argument("--re-no-rel-weight", type=float)
    parser.add_argument("--neg-sample-ratio", type=float)
    parser.add_argument("--focal-gamma", type=float)
    parser.add_argument("--bio-loss-weight", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument(
        "--comparison-boost-schedule",
        choices=("disabled", "pipeline-a20"),
    )
    parser.add_argument("--re-comparison-boost", type=float)
    parser.add_argument("--re-boost-adaptive-steps", type=int)
    parser.add_argument("--re-boost-adaptive-threshold", type=float)
    parser.add_argument("--re-boost-adaptive-threshold2", type=float)
    parser.add_argument("--re-boost-mid", type=float)
    parser.add_argument("--re-boost-end", type=float)
    parser.add_argument("--save-best-to")
    parser.add_argument("--save-last-to")
    parser.add_argument("--progress-log")
    parser.add_argument("--run-summary-out")
    parser.add_argument("--resume-from")
    parser.add_argument("--checkpoint")
    parser.add_argument("--sentences")
    parser.add_argument("--prediction-ledger-out")
    args = parser.parse_args(argv)
    required_by_mode = {
        "train": (
            "lr",
            "max_steps",
            "warmup_steps",
            "re_weight",
            "re_no_rel_weight",
            "neg_sample_ratio",
            "focal_gamma",
            "bio_loss_weight",
            "label_smoothing",
            "eval_every",
            "comparison_boost_schedule",
            "re_comparison_boost",
            "re_boost_adaptive_steps",
            "re_boost_adaptive_threshold",
            "re_boost_adaptive_threshold2",
            "re_boost_mid",
            "re_boost_end",
            "save_best_to",
            "save_last_to",
            "progress_log",
            "run_summary_out",
        ),
        "predict": ("checkpoint", "sentences", "prediction_ledger_out"),
    }
    missing = [name for name in required_by_mode[args.mode] if getattr(args, name) is None]
    if missing:
        parser.error(f"{args.mode} mode requires: {', '.join(missing)}")
    return args


def _total_memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _atomic_torch_save(value, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    torch.save(value, partial)
    partial.replace(target)


def _atomic_json_write(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    partial.replace(target)


def _atomic_jsonl_write(values: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
    partial.replace(target)


def _append_progress(args, message: str) -> None:
    path = Path(args.progress_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _canonical_sampler(train_loader):
    sampler = getattr(train_loader, "sampler", None)
    if not hasattr(sampler, "state_dict") or not hasattr(sampler, "load_state_dict"):
        raise ValueError("comparison training requires the shared resumable sampler")
    return sampler


def _restart_payload(
    args,
    model,
    optimizer,
    scheduler,
    train_loader,
    *,
    next_step: int,
    best_metrics: dict[str, Any],
    best_step: int,
    boost_adaptive_triggered: bool,
    boost_adaptive_switched: bool,
) -> dict[str, Any]:
    return {
        "format_version": "phase-g-historical-trainer-restart-1.0",
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


def _load_restart(args, model, optimizer, scheduler, train_loader):
    state = torch.load(args.resume_from, map_location="cpu", weights_only=False)
    required = {
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
    if (
        not isinstance(state, dict)
        or set(state) != required
        or state.get("format_version") != "phase-g-historical-trainer-restart-1.0"
    ):
        raise ValueError("comparison restart state has an invalid field contract")
    expected = {
        "seed": args.seed,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "max_steps": args.max_steps,
    }
    for field, expected_value in expected.items():
        if state[field] != expected_value:
            raise ValueError(f"comparison restart {field} mismatch")
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
    bio_loss_weight,
    label_smoothing,
    re_comparison_boost,
    re_no_rel_weight,
):
    """Historical active span-NER, optional BIO, and RE objective."""

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    hidden = model.encode(input_ids=input_ids, attention_mask=attention_mask)
    bio_loss = hidden.new_tensor(0.0)
    if bio_loss_weight > 0:
        bio_logits = model.forward_ner(hidden)
        bio_loss = F.cross_entropy(
            bio_logits.view(-1, bio_logits.size(-1)),
            batch["ner_labels"].to(device).view(-1),
            ignore_index=-100,
        )

    span_losses = []
    relation_losses = []
    for batch_index in range(input_ids.size(0)):
        gold_entities = batch["gold_entities"][batch_index]
        gold_relations = batch["gold_relations"][batch_index]
        span_logits, candidates = model.forward_span_ner(
            hidden[batch_index],
            batch["word_ids"][batch_index],
            batch["num_words"][batch_index],
            max_span_width,
        )
        if not candidates:
            continue
        labels = _build_span_labels(gold_entities, max_span_width, entity_type2id)
        targets = torch.tensor(
            [labels.get(candidate, 0) for candidate in candidates],
            device=device,
            dtype=torch.long,
        )
        positives = (targets > 0).nonzero(as_tuple=True)[0]
        negatives = (targets == 0).nonzero(as_tuple=True)[0]
        negative_count = max(int(len(positives) * neg_sample_ratio), 1)
        if len(negatives) > negative_count:
            permutation = torch.randperm(len(negatives), device=device)[:negative_count]
            negatives = negatives[permutation]
        kept = torch.cat([positives, negatives])
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
        pairs = [
            (head, tail)
            for head in relation_spans
            for tail in relation_spans
            if head != tail
        ]
        if not pairs:
            continue
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
            hidden[batch_index], batch["word_ids"][batch_index], pairs
        )
        class_weights = relation_logits.new_ones(relation_logits.size(-1))
        class_weights[code_accord.NO_REL_ID] = re_no_rel_weight
        for relation_id in code_accord.COMPARISON_REL_IDS:
            class_weights[relation_id] = re_comparison_boost
        relation_losses.append(
            F.cross_entropy(relation_logits, pair_targets, weight=class_weights)
        )

    ner_loss = torch.stack(span_losses).mean() if span_losses else hidden.new_tensor(0.0)
    relation_loss = (
        torch.stack(relation_losses).mean()
        if relation_losses
        else hidden.new_tensor(0.0)
    )
    total = ner_loss + re_weight * relation_loss + bio_loss_weight * bio_loss
    return total, ner_loss.detach(), relation_loss.detach(), bio_loss.detach()


def _prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _selected_entities(model, hidden, word_ids, num_words, max_span_width, id2entity):
    span_logits, candidates = model.forward_span_ner(
        hidden, word_ids, num_words, max_span_width
    )
    if not candidates:
        return span_logits, [], []
    probabilities = torch.softmax(span_logits, dim=-1)
    predicted = [
        (start, end, id2entity[entity_type], float(confidence))
        for (start, end), entity_type, confidence in zip(
            candidates,
            span_logits.argmax(dim=-1).tolist(),
            probabilities.max(dim=-1).values.tolist(),
        )
        if entity_type > 0 and confidence >= 0.5
    ]
    taken: set[tuple[int, int]] = set()
    selected = []
    for item in sorted(predicted, key=lambda value: (-value[3], value[0], value[1], value[2])):
        start, end = item[0], item[1]
        if not any(
            not (end < other_start or other_end < start)
            for other_start, other_end in taken
        ):
            selected.append(item)
            taken.add((start, end))
    return span_logits, candidates, selected


def evaluate_span(
    model,
    dataloader,
    device,
    id2entity,
    *,
    max_span_width,
):
    model.eval()
    ner_tp = ner_fp = ner_fn = 0
    triple_tp = triple_fp = triple_fn = 0
    example_count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden = model.encode(input_ids=input_ids, attention_mask=attention_mask)
            for batch_index in range(input_ids.size(0)):
                example_count += 1
                gold_entities = set(batch["gold_entities"][batch_index])
                gold_triples = set(batch["gold_relations"][batch_index])
                _logits, _candidates, selected = _selected_entities(
                    model,
                    hidden[batch_index],
                    batch["word_ids"][batch_index],
                    batch["num_words"][batch_index],
                    max_span_width,
                    id2entity,
                )
                predicted_entities = {
                    (start, end, entity_type)
                    for start, end, entity_type, _confidence in selected
                }
                ner_tp += len(predicted_entities & gold_entities)
                ner_fp += len(predicted_entities - gold_entities)
                ner_fn += len(gold_entities - predicted_entities)
                spans = [(start, end) for start, end, _type, _confidence in selected]
                pairs = [
                    (head, tail)
                    for head in spans
                    for tail in spans
                    if head != tail
                ]
                if pairs:
                    relation_ids = model.forward_re(
                        hidden[batch_index], batch["word_ids"][batch_index], pairs
                    ).argmax(dim=-1).tolist()
                    predicted_triples = {
                        (head, tail, relation_id)
                        for (head, tail), relation_id in zip(pairs, relation_ids)
                        if relation_id != code_accord.NO_REL_ID
                    }
                else:
                    predicted_triples = set()
                triple_tp += len(predicted_triples & gold_triples)
                triple_fp += len(predicted_triples - gold_triples)
                triple_fn += len(gold_triples - predicted_triples)
    ner_precision, ner_recall, ner_f1 = _prf(ner_tp, ner_fp, ner_fn)
    triple_precision, triple_recall, triple_f1 = _prf(
        triple_tp, triple_fp, triple_fn
    )
    return {
        "ner_precision": ner_precision,
        "ner_recall": ner_recall,
        "ner_f1": ner_f1,
        "triple_precision": triple_precision,
        "triple_recall": triple_recall,
        "triple_f1": triple_f1,
        "triple_tp": triple_tp,
        "triple_fp": triple_fp,
        "triple_fn": triple_fn,
        "n_examples": example_count,
    }


def _cycle(loader):
    while True:
        yield from loader


def _tokenizer(args):
    tokenizer_kwargs = {}
    if "deberta" in args.model_name.lower():
        tokenizer_kwargs["add_prefix_space"] = True
    return AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        cache_dir=args.model_cache_dir,
        local_files_only=args.model_local_files_only,
        **tokenizer_kwargs,
    )


def _model(args, device):
    return HistoricalKGExtractor(
        args.model_name,
        model_revision=args.model_revision,
        model_cache_dir=args.model_cache_dir,
        model_local_files_only=args.model_local_files_only,
        expected_hidden_size=args.expected_hidden_size,
        num_bio_tags=code_accord.NUM_BIO_TAGS,
        num_relations=code_accord.NUM_RELATIONS,
        num_entity_types=len(code_accord.ENTITY_TYPES),
        max_span_width=args.max_span_width,
        context_between_spans=args.context_between_spans,
    ).to(device)


def pipeline_comparison_boost(
    *,
    step: int,
    max_steps: int,
    initial: float,
    end: float,
    adaptive_step: int,
    threshold_low: float,
    threshold_high: float,
    middle: float,
    adaptive_triggered: bool,
    adaptive_switched: bool,
    adaptive_triple_f1: float | None = None,
) -> tuple[float, bool, bool]:
    """Mirror the effective A20 behavior in the current canonical pipeline.

    The historical implementation first computes linear ``initial -> end``
    decay. At the adaptive step, a middle-threshold result sets ``middle`` for
    that batch and also marks the gate switched. Consequently the next batch
    uses ``end``. This one-batch middle value is intentional provenance, not a
    repaired staircase. The comparison path keeps this pure mirror so the
    original pipeline flow remains byte-for-byte outside Phase G additions.
    """

    progress = step / max(max_steps - 1, 1)
    boost = initial - (initial - end) * progress
    if adaptive_switched:
        boost = end
    if not adaptive_triggered and step == adaptive_step:
        if adaptive_triple_f1 is None:
            raise ValueError("adaptive-step boost requires development Triple F1")
        adaptive_triggered = True
        if adaptive_triple_f1 >= threshold_high:
            adaptive_switched = True
            boost = end
        elif adaptive_triple_f1 >= threshold_low:
            adaptive_switched = True
            boost = middle
    return boost, adaptive_triggered, adaptive_switched


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _tokenizer(args)
    train_loader, development_loader, test_loader = code_accord.build_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        prepared_dir=args.prepared_dir,
    )
    if test_loader is not None:
        raise ValueError("comparison data adapter unexpectedly loaded test")
    entity_type2id = {
        entity_type: index + 1
        for index, entity_type in enumerate(code_accord.ENTITY_TYPES)
    }
    id2entity = {value: key for key, value in entity_type2id.items()}
    model = _model(args, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    best_metrics = {"triple_f1": -1.0}
    best_step = -1
    adaptive_triggered = False
    adaptive_switched = False
    step = 0
    resumed_from = None
    if args.resume_from:
        state = _load_restart(args, model, optimizer, scheduler, train_loader)
        step = state["next_step"]
        best_metrics = dict(state["best_metrics"])
        best_step = state["best_step"]
        adaptive_triggered = state["boost_adaptive_triggered"]
        adaptive_switched = state["boost_adaptive_switched"]
        resumed_from = str(Path(args.resume_from))

    iterator = _cycle(train_loader)
    started = time.time()
    model.train()
    while step < args.max_steps:
        optimizer.zero_grad()
        batch = next(iterator)
        adaptive_triple_f1 = None
        if (
            args.comparison_boost_schedule == "pipeline-a20"
            and not adaptive_triggered
            and step == args.re_boost_adaptive_steps
        ):
            metrics = evaluate_span(
                model,
                development_loader,
                device,
                id2entity,
                max_span_width=args.max_span_width,
            )
            model.train()
            adaptive_triple_f1 = metrics["triple_f1"]
        if args.comparison_boost_schedule == "pipeline-a20":
            boost, adaptive_triggered, adaptive_switched = pipeline_comparison_boost(
                step=step,
                max_steps=args.max_steps,
                initial=args.re_comparison_boost,
                end=args.re_boost_end,
                adaptive_step=args.re_boost_adaptive_steps,
                threshold_low=args.re_boost_adaptive_threshold,
                threshold_high=args.re_boost_adaptive_threshold2,
                middle=args.re_boost_mid,
                adaptive_triggered=adaptive_triggered,
                adaptive_switched=adaptive_switched,
                adaptive_triple_f1=adaptive_triple_f1,
            )
        else:
            boost = args.re_comparison_boost

        loss, ner_loss, relation_loss, bio_loss = compute_span_loss(
            model,
            batch,
            device,
            entity_type2id,
            re_weight=args.re_weight,
            neg_sample_ratio=args.neg_sample_ratio,
            max_span_width=args.max_span_width,
            focal_gamma=args.focal_gamma,
            bio_loss_weight=args.bio_loss_weight,
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
                f"[Step {step:04d}] L={loss.item():.4f} "
                f"NER={ner_loss.item():.4f} RE={relation_loss.item():.4f} "
                f"BIO={bio_loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed_ms:.0f}ms/step"
            )
            print(message, flush=True)
            _append_progress(args, message)
        if step > 0 and step % args.eval_every == 0:
            metrics = evaluate_span(
                model,
                development_loader,
                device,
                id2entity,
                max_span_width=args.max_span_width,
            )
            selected = metrics["triple_f1"] > best_metrics["triple_f1"]
            if selected:
                best_metrics = dict(metrics)
                best_step = step
                _atomic_torch_save(
                    {"encoder": model.state_dict(), "step": step, "metrics": metrics},
                    args.save_best_to,
                )
            message = (
                f"[Eval @ {step}] NER={metrics['ner_f1']:.4f} "
                f"Triple={metrics['triple_f1']:.4f}{' *' if selected else ''}"
            )
            print(message, flush=True)
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
                    boost_adaptive_triggered=adaptive_triggered,
                    boost_adaptive_switched=adaptive_switched,
                ),
                args.save_last_to,
            )
        step += 1

    metrics = evaluate_span(
        model,
        development_loader,
        device,
        id2entity,
        max_span_width=args.max_span_width,
    )
    if metrics["triple_f1"] > best_metrics["triple_f1"]:
        best_metrics = dict(metrics)
        best_step = step
        _atomic_torch_save(
            {"encoder": model.state_dict(), "step": step, "metrics": metrics},
            args.save_best_to,
        )
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
            boost_adaptive_triggered=adaptive_triggered,
            boost_adaptive_switched=adaptive_switched,
        ),
        args.save_last_to,
    )
    summary = {
        "schema_version": "phase-g-historical-trainer-summary-1.0",
        "status": "completed",
        "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
        "historical_trainer_blob": HISTORICAL_TRAINER_BLOB,
        "seed": args.seed,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "completed_steps": step,
        "selected_step": best_step,
        "selection_metric": "development_strict_triple_f1",
        "selected_metrics": best_metrics,
        "test_evaluated": False,
        "resumed_from": resumed_from,
        "elapsed_seconds": time.time() - started,
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
        },
    }
    _atomic_json_write(summary, args.run_summary_out)
    return summary


def _read_sentences(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            records.append(value)
    return records


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _tokenizer(args)
    model = _model(args, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint:
        raise ValueError("comparison checkpoint lacks the historical encoder payload")
    model.load_state_dict(checkpoint["encoder"], strict=True)
    model.eval()
    id2entity = {
        index + 1: entity_type
        for index, entity_type in enumerate(code_accord.ENTITY_TYPES)
    }
    records = []
    for sentence in sorted(_read_sentences(Path(args.sentences)), key=lambda item: item["example_id"]):
        words = sentence["words"]
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=args.max_length,
            return_tensors=None,
        )
        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long, device=device
        )
        word_ids = encoding.word_ids()
        with torch.no_grad():
            hidden = model.encode(input_ids=input_ids, attention_mask=attention_mask)
            span_logits, candidate_spans = model.forward_span_ner(
                hidden[0], word_ids, len(words), args.max_span_width
            )
        predicted_spans = []
        if candidate_spans:
            probabilities = torch.softmax(span_logits, dim=-1)
            for (start, end), type_id, confidence in zip(
                candidate_spans,
                span_logits.argmax(dim=-1).tolist(),
                probabilities.max(dim=-1).values.tolist(),
            ):
                if type_id > 0 and end < len(words):
                    predicted_spans.append(
                        {
                            "start": int(start),
                            "end": int(end),
                            "type": id2entity[type_id],
                            "confidence": round(float(confidence), CONFIDENCE_PRECISION),
                        }
                    )
        predicted_spans.sort(key=lambda item: (item["start"], item["end"], item["type"]))
        selected = []
        occupied: set[tuple[int, int]] = set()
        for span in sorted(
            predicted_spans,
            key=lambda item: (-item["confidence"], item["start"], item["end"], item["type"]),
        ):
            if not any(
                not (span["end"] < start or end < span["start"])
                for start, end in occupied
            ):
                selected.append(span)
                occupied.add((span["start"], span["end"]))
        pairs = [
            ((head["start"], head["end"]), (tail["start"], tail["end"]))
            for head in selected
            for tail in selected
            if head is not tail
        ]
        predicted_relations = []
        if pairs:
            with torch.no_grad():
                relation_logits = model.forward_re(hidden[0], word_ids, pairs)
            relation_probabilities = torch.softmax(relation_logits, dim=-1)
            for (head, tail), relation_id, confidence in zip(
                pairs,
                relation_logits.argmax(dim=-1).tolist(),
                relation_probabilities.max(dim=-1).values.tolist(),
            ):
                if relation_id != code_accord.NO_REL_ID:
                    predicted_relations.append(
                        {
                            "head": {"start": head[0], "end": head[1]},
                            "tail": {"start": tail[0], "end": tail[1]},
                            "relation": code_accord.ID2REL[relation_id],
                            "re_confidence": round(
                                float(confidence), CONFIDENCE_PRECISION
                            ),
                        }
                    )
        records.append(
            {
                "protocol_id": PROTOCOL_ID,
                "example_id": sentence["example_id"],
                "training_seed": args.seed,
                "predicted_spans": predicted_spans,
                "predicted_relations": predicted_relations,
            }
        )
    _atomic_jsonl_write(records, args.prediction_ledger_out)
    return records


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    prepared_root = Path(args.prepared_dir).resolve()
    cache_root = Path(args.model_cache_dir).resolve()
    if cache_root != prepared_root.parent / "inputs" / "huggingface":
        raise ValueError("comparison trainer requires the same run-local cache")
    return train(args) if args.mode == "train" else predict(args)
