"""Minimal canonical graph construction used by the table pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from records import Candidate, GoldRecord, StrictTriple, Verdict


StrictKey = tuple[str, int, int, str, str, int, int, str]
CANONICAL_VERSION = "graph-rag-canonical-1.0"
PRODUCER_VERSION = "C-04-PHASE-B-REGIME-D-1.0"


class GraphConstructionError(ValueError):
    """Raised when same-run records cannot produce a canonical graph."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fingerprint(namespace: str, *parts: Any) -> str:
    return content_sha256({"namespace": namespace, "parts": list(parts)})


@dataclass(frozen=True)
class PreparedExample:
    example_id: str
    source_document_id: str
    content: str
    words: tuple[str, ...]
    word_char_spans: tuple[tuple[int, int], ...]

    def span_text(self, start: int, end: int) -> str:
        if not 0 <= start <= end < len(self.words):
            raise GraphConstructionError(
                f"{self.example_id}: strict span {start}:{end} escapes prepared words"
            )
        return " ".join(self.words[start : end + 1])

    def span_char_range(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start <= end < len(self.word_char_spans):
            raise GraphConstructionError(
                f"{self.example_id}: strict span {start}:{end} escapes prepared words"
            )
        return self.word_char_spans[start][0], self.word_char_spans[end][1]


def load_prepared(rows: Iterable[dict[str, Any]]) -> dict[str, PreparedExample]:
    result: dict[str, PreparedExample] = {}
    observed: list[str] = []
    for index, row in enumerate(rows, 1):
        example_id = row.get("example_id")
        source_id = row.get("source_document_id")
        content = row.get("content")
        words = row.get("words")
        if (
            not isinstance(example_id, str)
            or not example_id
            or not isinstance(source_id, str)
            or not source_id
            or not isinstance(content, str)
            or not isinstance(words, list)
            or not all(isinstance(word, str) and word for word in words)
        ):
            raise GraphConstructionError(f"prepared test row {index} is malformed")
        if example_id in result:
            raise GraphConstructionError(f"duplicate prepared example: {example_id}")
        cursor = 0
        spans = []
        for word in words:
            start = content.find(word, cursor)
            if start < 0:
                raise GraphConstructionError(
                    f"prepared words do not align to content: {example_id}"
                )
            end = start + len(word)
            spans.append((start, end))
            cursor = end
        result[example_id] = PreparedExample(
            example_id,
            source_id,
            content,
            tuple(words),
            tuple(spans),
        )
        observed.append(example_id)
    if not result or observed != sorted(observed):
        raise GraphConstructionError("prepared test examples are not stably sorted")
    return result


def _chunk_id(example_id: str) -> str:
    return "chunk-" + fingerprint("phase-b-example-chunk-v1", example_id)


def _entity_id(key: StrictKey, endpoint: str) -> str:
    payload = key[0:4] if endpoint == "head" else (key[0], *key[5:8])
    return "entity-" + fingerprint("phase-b-strict-entity-v1", payload)


def _relation_id(label: str) -> str:
    return "relation-" + fingerprint("phase-b-strict-relation-v1", label)


def _triple_id(key: StrictKey) -> str:
    return "triple-" + fingerprint("phase-b-strict-triple-v1", key)


def _provenance(
    *,
    run_id: str,
    dataset_id: str,
    stage: str,
    payload: Any,
    document_id: str | None = None,
    chunk_id: str | None = None,
    span: tuple[int, int] | None = None,
    parents: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "content_sha256": content_sha256(payload),
        "dataset_id": dataset_id,
        "document_id": document_id,
        "parent_ids": list(parents),
        "producer_version": PRODUCER_VERSION,
        "snapshot_id": f"{run_id}:CODE-SPLIT-1:test",
        "span_end": span[1] if span else None,
        "span_start": span[0] if span else None,
        "stage": stage,
        "timestamp": "deterministic",
    }


def _chunks(
    prepared: dict[str, PreparedExample], run_id: str, dataset_id: str
) -> tuple[dict[str, Any], ...]:
    rows = []
    for example_id, item in sorted(prepared.items()):
        chunk_id = _chunk_id(example_id)
        text = item.content
        rows.append({
            "char_end": len(text),
            "char_start": 0,
            "chunk_id": chunk_id,
            "chunker_id": "phase-b-prepared-sentence-v1",
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "dataset_id": dataset_id,
            "document_id": item.source_document_id,
            "provenance": _provenance(
                run_id=run_id,
                dataset_id=dataset_id,
                stage="phase-b-prepared-test",
                payload={"example_id": example_id, "text": text},
                document_id=item.source_document_id,
                chunk_id=chunk_id,
                parents=(example_id,),
            ),
            "schema_version": CANONICAL_VERSION,
            "text": text,
            "token_end": len(item.words),
            "token_start": 0,
        })
    return tuple(sorted(rows, key=lambda row: row["chunk_id"]))


def _strict_from_key(
    key: StrictKey, prepared: dict[str, PreparedExample]
) -> StrictTriple:
    example_id, hs, he, ht, relation, ts, te, tt = key
    item = prepared.get(example_id)
    if item is None:
        raise GraphConstructionError(f"strict key references unknown example: {example_id}")
    return StrictTriple.from_mapping(
        {
            "head": {"start": hs, "end": he, "type": ht, "text": item.span_text(hs, he)},
            "relation": relation,
            "tail": {"start": ts, "end": te, "type": tt, "text": item.span_text(ts, te)},
        },
        example_id=example_id,
        label="derived strict triple",
    )


def _graph_from_keys(
    *,
    keys: Iterable[StrictKey],
    prepared: dict[str, PreparedExample],
    chunks: tuple[dict[str, Any], ...],
    run_id: str,
    dataset_id: str,
    condition: str,
    extractor_identity: Any,
    input_identity: dict[str, Any],
    recipe: str,
) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[str, dict[str, Any]] = {}
    triples = []
    for key in sorted(set(keys)):
        strict = _strict_from_key(key, prepared)
        item = prepared[strict.example_id]
        chunk_id = _chunk_id(strict.example_id)
        head_id = _entity_id(key, "head")
        tail_id = _entity_id(key, "tail")
        relation_id = _relation_id(strict.relation)
        for entity_id, entity_span in ((head_id, strict.head), (tail_id, strict.tail)):
            char_span = item.span_char_range(entity_span.start, entity_span.end)
            label = entity_span.text or item.span_text(entity_span.start, entity_span.end)
            candidate = {
                "canonical_label": label,
                "dataset_id": dataset_id,
                "entity_id": entity_id,
                "entity_type": entity_span.entity_type,
                "provenance": _provenance(
                    run_id=run_id,
                    dataset_id=dataset_id,
                    stage="phase-b-strict-entity",
                    payload=(strict.example_id, entity_span.key()),
                    document_id=item.source_document_id,
                    chunk_id=chunk_id,
                    span=char_span,
                    parents=(strict.example_id,),
                ),
                "schema_version": CANONICAL_VERSION,
                "source_spans": [[chunk_id, char_span[0], char_span[1]]],
                "surface_forms": [label],
            }
            previous = entities.get(entity_id)
            if previous is not None and canonical_json(previous) != canonical_json(candidate):
                raise GraphConstructionError("entity identity collision during graph construction")
            entities[entity_id] = candidate
        if relation_id not in relations:
            relations[relation_id] = {
                "dataset_id": dataset_id,
                "description": None,
                "direction": "directed",
                "label": strict.relation,
                "provenance": _provenance(
                    run_id=run_id,
                    dataset_id=dataset_id,
                    stage="phase-b-strict-relation-vocabulary",
                    payload=strict.relation,
                ),
                "relation_id": relation_id,
                "schema_version": CANONICAL_VERSION,
            }
        triples.append({
            "chunk_ids": [chunk_id],
            "confidence": None,
            "dataset_id": dataset_id,
            "head_id": head_id,
            "provenance": _provenance(
                run_id=run_id,
                dataset_id=dataset_id,
                stage="phase-b-condition-emission",
                payload={"condition": condition, "strict_key": key},
                document_id=item.source_document_id,
                chunk_id=chunk_id,
                parents=(strict.example_id,),
            ),
            "relation_id": relation_id,
            "schema_version": CANONICAL_VERSION,
            "tail_id": tail_id,
            "triple_id": _triple_id(key),
        })

    entity_rows = tuple(sorted(entities.values(), key=lambda row: row["entity_id"]))
    relation_rows = tuple(sorted(relations.values(), key=lambda row: row["relation_id"]))
    triple_rows = tuple(sorted(triples, key=lambda row: row["triple_id"]))
    descriptor_hash = content_sha256({
        "dataset_id": dataset_id,
        "run_id": run_id,
        "split": "CODE-SPLIT-1:test",
        **input_identity,
    })
    entity_hash = content_sha256(entity_rows)
    relation_hash = content_sha256(relation_rows)
    triple_hash = content_sha256(triple_rows)
    identity = {
        "dataset_id": dataset_id,
        "condition": condition,
        "descriptor_sha256": descriptor_hash,
        "corpus_sha256": content_sha256(chunks),
        "adapter_sha256": fingerprint("phase-b-output-adapter-v1", input_identity),
        "extractor_sha256": fingerprint("phase-b-condition-extractor-v1", extractor_identity),
        "entity_sha256": entity_hash,
        "relation_sha256": relation_hash,
        "triple_sha256": triple_hash,
        "construction_recipe": recipe,
        "parent_graph_id": None,
        "corruption_recipe": None,
    }
    return {
        "adapter_sha256": identity["adapter_sha256"],
        "condition": condition,
        "construction_recipe": recipe,
        "construction_time": "deterministic",
        "corpus_sha256": identity["corpus_sha256"],
        "corruption_recipe": None,
        "dataset_id": dataset_id,
        "descriptor_sha256": descriptor_hash,
        "entities": list(entity_rows),
        "entity_sha256": entity_hash,
        "extractor_sha256": identity["extractor_sha256"],
        "graph_id": content_sha256(identity),
        "parent_graph_id": None,
        "relation_sha256": relation_hash,
        "relations": list(relation_rows),
        "schema_version": "rag-graph-snapshot-1.0",
        "triple_sha256": triple_hash,
        "triples": list(triple_rows),
    }


def _load_gold(
    rows: Iterable[dict[str, Any]], prepared: dict[str, PreparedExample]
) -> dict[str, GoldRecord]:
    result = {}
    observed = []
    for index, row in enumerate(rows, 1):
        record = GoldRecord.from_mapping(row, f"private gold:{index}")
        item = prepared.get(record.example_id)
        if item is None or item.source_document_id != record.source_document_id:
            raise GraphConstructionError("private gold differs from prepared test identity")
        result[record.example_id] = record
        observed.append(record.example_id)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise GraphConstructionError("private gold must be uniquely and stably sorted")
    if set(result) != set(prepared):
        raise GraphConstructionError("prepared test and private gold example sets differ")
    return result


def _load_candidates(
    rows: Iterable[dict[str, Any]], gold: dict[str, GoldRecord], prepared_sha256: str
) -> tuple[Candidate, ...]:
    result = []
    identifiers = set()
    for index, row in enumerate(rows, 1):
        candidate = Candidate.from_mapping(row, f"test candidates:{index}")
        target = gold.get(candidate.triple.example_id)
        if target is None or target.source_document_id != candidate.source_document_id:
            raise GraphConstructionError("candidate differs from private-gold identity")
        if candidate.input_hashes.get("prepared_sentences") != prepared_sha256:
            raise GraphConstructionError("candidate is not bound to prepared test text")
        if candidate.candidate_id in identifiers:
            raise GraphConstructionError(f"duplicate candidate ID: {candidate.candidate_id}")
        identifiers.add(candidate.candidate_id)
        result.append(candidate)
    if [row.sort_key() for row in result] != sorted(row.sort_key() for row in result):
        raise GraphConstructionError("test candidate ledger is not stably sorted")
    return tuple(result)


def _load_verdicts(
    rows: Iterable[dict[str, Any]], candidates: dict[str, Candidate], condition: str
) -> dict[str, Verdict]:
    result = {}
    order = []
    for index, row in enumerate(rows, 1):
        candidate = candidates.get(row.get("candidate_id"))
        if candidate is None:
            raise GraphConstructionError(f"{condition} verdict references an unknown candidate")
        verdict = Verdict.from_mapping(row, f"{condition} verdicts:{index}", candidate)
        if verdict.condition_id != condition or verdict.candidate_id in result:
            raise GraphConstructionError(f"invalid or duplicate {condition} verdict")
        result[verdict.candidate_id] = verdict
        order.append((verdict.training_seed, verdict.candidate_id))
    if order != sorted(order) or set(result) != set(candidates):
        raise GraphConstructionError(f"{condition} verdict coverage/order differs from candidates")
    return result


def build_table_graphs(
    *,
    run_id: str,
    prepared_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    corrective_rows: list[dict[str, Any]],
    selected_threshold: float,
    seed: int,
    prepared_sha256: str,
    split_sha256: str,
    gold_sha256: str,
    candidate_sha256: str,
    corrective_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Construct the three canonical snapshots consumed by Table 2."""
    prepared = load_prepared(prepared_rows)
    gold = _load_gold(gold_rows, prepared)
    candidates = _load_candidates(candidate_rows, gold, prepared_sha256)
    candidate_by_id = {row.candidate_id: row for row in candidates}
    corrective = _load_verdicts(corrective_rows, candidate_by_id, "VER-CORRECTIVE")
    dataset_ids = {row.get("dataset_id") for row in prepared_rows}
    if len(dataset_ids) != 1 or not isinstance(next(iter(dataset_ids)), str):
        raise GraphConstructionError("prepared rows do not carry one dataset identity")
    dataset_id = next(iter(dataset_ids))
    selected = [row for row in candidates if row.training_seed == seed]
    if not selected:
        raise GraphConstructionError(f"candidate ledger has no seed-{seed} records")

    confidence_keys = {
        row.triple.key() for row in selected if row.triple_confidence >= selected_threshold
    }
    corrective_keys: set[StrictKey] = set()
    for candidate in selected:
        verdict = corrective[candidate.candidate_id]
        if verdict.response_status != "valid_response":
            continue
        if verdict.action == "KEEP":
            corrective_keys.add(candidate.triple.key())
        elif (
            verdict.action == "CORRECT"
            and verdict.correction_validation_status == "valid"
            and verdict.corrected is not None
        ):
            corrective_keys.add(verdict.corrected.key())
    gold_keys = {triple.key() for row in gold.values() for triple in row.triples}
    chunks = _chunks(prepared, run_id, dataset_id)
    input_identity = {
        "prepared": {"path": "data-prepared/test.jsonl", "sha256": prepared_sha256},
        "split_identity": {"path": "data-prepared/split-manifest.json", "sha256": split_sha256},
    }
    common = {
        "prepared": prepared,
        "chunks": chunks,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "input_identity": input_identity,
    }
    return {
        "confidence": _graph_from_keys(
            keys=confidence_keys,
            condition=f"confidence_filtered:seed-{seed}",
            extractor_identity={
                "phase_b_condition": "VER-CONFIDENCE",
                "training_seed": seed,
                "candidate_sha256": candidate_sha256,
                "verdict_sha256": None,
                "threshold": selected_threshold,
            },
            recipe="phase-b-confidence-threshold-emissions-v1",
            **common,
        ),
        "corrective": _graph_from_keys(
            keys=corrective_keys,
            condition=f"corrective_verifier:seed-{seed}",
            extractor_identity={
                "phase_b_condition": "VER-CORRECTIVE",
                "training_seed": seed,
                "candidate_sha256": candidate_sha256,
                "verdict_sha256": corrective_sha256,
                "threshold": selected_threshold,
            },
            recipe="phase-b-corrective-verdict-emissions-v1",
            **common,
        ),
        "gold": _graph_from_keys(
            keys=gold_keys,
            condition="gold_graph_oracle",
            extractor_identity={"private_gold_sha256": gold_sha256},
            recipe="private-phase-b-gold-oracle-v1",
            **common,
        ),
    }


def project_graph(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Project a canonical snapshot to the frozen evaluator's legacy shape."""
    entity_labels = {
        row["entity_id"]: row["canonical_label"] for row in snapshot["entities"]
    }
    relation_labels = {
        row["relation_id"]: row["label"] for row in snapshot["relations"]
    }
    return {
        "nodes": [
            {
                "id": row["canonical_label"],
                "entity_id": row["entity_id"],
                "type": row["entity_type"],
            }
            for row in snapshot["entities"]
        ],
        "edges": [
            {
                "head": entity_labels[row["head_id"]],
                "relation": relation_labels[row["relation_id"]],
                "tail": entity_labels[row["tail_id"]],
                "triple_id": row["triple_id"],
            }
            for row in snapshot["triples"]
        ],
    }
