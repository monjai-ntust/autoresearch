"""encoder candidates responsibilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from utils.common.constants import ENTITY_TYPES, PROTOCOL_ID, RELATION_TYPES
from utils.common.artifact_io import DataContractError, iter_jsonl
from utils.common.records import Candidate, EntitySpan, StrictTriple, candidate_id_for
from utils.verifier.requests import PreparedSentence, _source_text

_SPLIT_ID = "CODE-SPLIT-1"

_CONFIDENCE_PRECISION = 6

def _exact_keys(
    value: dict[str, Any], required: set[str], label: str, optional: set[str] | None = None
) -> None:
    optional = optional or set()
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be an object")
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing:
        raise DataContractError(f"{label} is missing required fields: {', '.join(missing)}")
    if extra:
        raise DataContractError(f"{label} contains unsupported fields: {', '.join(extra)}")

def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise DataContractError(f"{label} must be >= {minimum}")
    return value

def _unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataContractError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise DataContractError(f"{label} must be finite")
    if not 0.0 <= number <= 1.0:
        raise DataContractError(f"{label} must be in [0, 1]")
    return number

@dataclass(frozen=True)
class ScoredSpan:
    start: int
    end: int
    entity_type: str
    confidence: float

@dataclass(frozen=True)
class PredictionLedgerIdentity:
    """Minimal immutable checkpoint identity needed for offline ledger replay."""

    training_seed: int
    max_span_width: int

def _scored_span(value: Any, label: str, max_span_width: int) -> ScoredSpan:
    _exact_keys(value, {"start", "end", "type", "confidence"}, label)
    start = _integer(value["start"], f"{label}.start", minimum=0)
    end = _integer(value["end"], f"{label}.end", minimum=0)
    if end < start:
        raise DataContractError(f"{label}.end must be >= {label}.start")
    if end - start + 1 > max_span_width:
        raise DataContractError(f"{label} width exceeds the recipe max_span_width")
    entity_type = value["type"]
    if entity_type not in ENTITY_TYPES:
        raise DataContractError(f"{label}.type must be one of {list(ENTITY_TYPES)}")
    return ScoredSpan(
        start=start,
        end=end,
        entity_type=entity_type,
        confidence=_unit_float(value["confidence"], f"{label}.confidence"),
    )

def _endpoint(value: Any, label: str) -> tuple[int, int]:
    _exact_keys(value, {"start", "end"}, label)
    start = _integer(value["start"], f"{label}.start", minimum=0)
    end = _integer(value["end"], f"{label}.end", minimum=0)
    if end < start:
        raise DataContractError(f"{label}.end must be >= {label}.start")
    return start, end

def _select_non_overlapping(spans: list[ScoredSpan]) -> dict[tuple[int, int], ScoredSpan]:
    """Greedy highest-confidence non-overlapping selection (inference_kg parity).

    ``spans`` is required to be ordered by ``(start, end, type)`` so that the
    stable descending-confidence sort resolves ties deterministically, matching
    the historical script's stable ``sort(key=-confidence)`` over an ordered
    candidate list.
    """

    ordered = sorted(range(len(spans)), key=lambda index: -spans[index].confidence)
    taken: list[tuple[int, int]] = []
    selected: dict[tuple[int, int], ScoredSpan] = {}
    for index in ordered:
        span = spans[index]
        overlaps = any(not (span.end < start or end < span.start) for start, end in taken)
        if overlaps:
            continue
        taken.append((span.start, span.end))
        selected[(span.start, span.end)] = span
    return selected

def _sentence_candidates(
    sentence: PreparedSentence,
    scored_spans: list[ScoredSpan],
    relations: list[dict[str, Any]],
    seed: int,
    input_hashes: dict[str, str],
    label: str,
) -> list[dict[str, Any]]:
    selected = _select_non_overlapping(scored_spans)
    records: list[dict[str, Any]] = []
    for index, relation in enumerate(relations):
        relation_label = f"{label}.predicted_relations[{index}]"
        _exact_keys(relation, {"head", "tail", "relation", "re_confidence"}, relation_label)
        head_key = _endpoint(relation["head"], f"{relation_label}.head")
        tail_key = _endpoint(relation["tail"], f"{relation_label}.tail")
        if head_key == tail_key:
            raise DataContractError(f"{relation_label} head and tail are identical spans")
        relation_type = relation["relation"]
        if relation_type not in RELATION_TYPES:
            raise DataContractError(
                f"{relation_label}.relation must be one of {list(RELATION_TYPES)}"
            )
        head_span = selected.get(head_key)
        tail_span = selected.get(tail_key)
        if head_span is None or tail_span is None:
            raise DataContractError(
                f"{relation_label} references a span that was not a selected entity"
            )
        re_confidence = _unit_float(relation["re_confidence"], f"{relation_label}.re_confidence")
        confidence = round(
            min(head_span.confidence, tail_span.confidence) * re_confidence,
            _CONFIDENCE_PRECISION,
        )
        head = EntitySpan(
            head_span.start,
            head_span.end,
            head_span.entity_type,
            _source_text(
                sentence,
                EntitySpan(head_span.start, head_span.end, head_span.entity_type),
                f"{relation_label}.head",
            ),
        )
        tail = EntitySpan(
            tail_span.start,
            tail_span.end,
            tail_span.entity_type,
            _source_text(
                sentence,
                EntitySpan(tail_span.start, tail_span.end, tail_span.entity_type),
                f"{relation_label}.tail",
            ),
        )
        triple = StrictTriple(sentence.example_id, head, relation_type, tail)
        record = {
            "protocol_id": PROTOCOL_ID,
            "split_id": f"{_SPLIT_ID}:{sentence.split}",
            "training_seed": seed,
            "example_id": sentence.example_id,
            "source_document_id": sentence.source_document_id,
            "candidate_id": candidate_id_for(seed, triple),
            "head": head.to_mapping(include_text=True),
            "relation": relation_type,
            "tail": tail.to_mapping(include_text=True),
            "triple_confidence": confidence,
            "input_hashes": dict(input_hashes),
        }
        # Re-validate through the shared contract so the emitted file always
        # parses under the same rules the verifier and scorer apply.
        Candidate.from_mapping(record, relation_label)
        records.append(record)
    return records

def _load_prediction_ledger(
    path: Path,
    sentences: dict[str, PreparedSentence],
    checkpoint: CheckpointIdentity | PredictionLedgerIdentity,
    input_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], int, int, int]:
    seen: set[str] = set()
    observed_order: list[str] = []
    collected: dict[str, dict[str, Any]] = {}
    duplicate_collapsed = 0
    selected_total = 0
    for line_number, value in iter_jsonl(path):
        label = f"{path.name}:{line_number}"
        _exact_keys(
            value,
            {"protocol_id", "example_id", "training_seed", "predicted_spans", "predicted_relations"},
            label,
        )
        if value["protocol_id"] != PROTOCOL_ID:
            raise DataContractError(f"{label}.protocol_id differs from {PROTOCOL_ID}")
        example_id = value["example_id"]
        if not isinstance(example_id, str) or not example_id:
            raise DataContractError(f"{label}.example_id must be a nonempty string")
        if example_id in seen:
            raise DataContractError(f"{label} duplicates example_id {example_id}")
        seen.add(example_id)
        observed_order.append(example_id)
        sentence = sentences.get(example_id)
        if sentence is None:
            raise DataContractError(f"{label} refers to an unknown prepared sentence")
        seed = _integer(value["training_seed"], f"{label}.training_seed")
        if seed != checkpoint.training_seed:
            raise DataContractError(f"{label}.training_seed differs from the checkpoint identity")
        raw_spans = value["predicted_spans"]
        if not isinstance(raw_spans, list):
            raise DataContractError(f"{label}.predicted_spans must be an array")
        scored_spans = [
            _scored_span(item, f"{label}.predicted_spans[{index}]", checkpoint.max_span_width)
            for index, item in enumerate(raw_spans)
        ]
        span_order = [(span.start, span.end, span.entity_type) for span in scored_spans]
        if span_order != sorted(span_order):
            raise DataContractError(
                f"{label}.predicted_spans must be sorted by start, end, type"
            )
        if len(set(span_order)) != len(span_order):
            raise DataContractError(f"{label}.predicted_spans contains a duplicate span")
        for span in scored_spans:
            _source_text(
                sentence,
                EntitySpan(span.start, span.end, span.entity_type),
                f"{label}.predicted_spans",
            )
        raw_relations = value["predicted_relations"]
        if not isinstance(raw_relations, list):
            raise DataContractError(f"{label}.predicted_relations must be an array")
        selected_total += len(_select_non_overlapping(scored_spans))
        for record in _sentence_candidates(
            sentence, scored_spans, raw_relations, seed, input_hashes, label
        ):
            candidate_id = record["candidate_id"]
            existing = collected.get(candidate_id)
            if existing is None:
                collected[candidate_id] = record
            else:
                duplicate_collapsed += 1
                if record["triple_confidence"] > existing["triple_confidence"]:
                    collected[candidate_id] = record
    if observed_order != sorted(observed_order):
        raise DataContractError(f"{path.name} must be sorted by example_id")
    if seen != set(sentences):
        missing = sorted(set(sentences) - seen)
        extra = sorted(seen - set(sentences))
        raise DataContractError(
            "prediction ledger coverage mismatch; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    candidates = sorted(
        collected.values(),
        key=lambda record: (record["training_seed"], record["example_id"], record["candidate_id"]),
    )
    return candidates, len(observed_order), selected_total, duplicate_collapsed
