"""preparation annotations responsibilities."""
from __future__ import annotations

import ast
import csv
import hashlib
import re
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from utils.common.constants import ENTITY_TYPES, PROTOCOL_ID, RELATION_TYPES
from utils.common.artifact_io import DataContractError

ENTITY_HEADER = ("example_id", "content", "processed_content", "label", "metadata")

RELATION_HEADER = (
    "example_id",
    "content",
    "metadata",
    "tagged_sentence",
    "relation_type",
)

LOWER_ENTITY_TYPES = {value.lower(): value for value in ENTITY_TYPES}

TAG_RE = re.compile(r"</?e[12]>")

@dataclass(frozen=True)
class DatasetContract:
    dataset_id: str
    archive_bytes: int
    archive_md5: str
    annotation_files: dict[str, dict[str, Any]]
    counts: dict[str, int]
    relation_split_audit: dict[str, int] | None
    repaired_uuid: str

@dataclass(frozen=True)
class EntityRecord:
    example_id: str
    content: str
    processed_content: str
    words: tuple[str, ...]
    labels: tuple[str, ...]
    metadata: str
    source_document_id: str
    source_country: str
    entities: tuple[dict[str, Any], ...]
    source_row: int

@dataclass(frozen=True)
class RelationRecord:
    example_id: str
    relation: str
    head: dict[str, Any]
    tail: dict[str, Any]
    source_rows: tuple[int, ...]
    head_alignment: str
    tail_alignment: str

def _read_csv(path: Path, expected_header: tuple[str, ...]) -> list[tuple[int, dict[str, str]]]:
    records: list[tuple[int, dict[str, str]]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise DataContractError(f"empty annotation CSV: {path.name}") from exc
            if tuple(header) != expected_header:
                raise DataContractError(
                    f"unexpected CSV header in {path.name}: {header!r}"
                )
            for record_number, values in enumerate(reader, start=2):
                if len(values) != len(expected_header):
                    raise DataContractError(
                        f"{path.name} record {record_number} has {len(values)} fields; "
                        f"expected {len(expected_header)}"
                    )
                if any("\x00" in value for value in values):
                    raise DataContractError(
                        f"{path.name} record {record_number} contains a NUL byte"
                    )
                records.append((record_number, dict(zip(expected_header, values))))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataContractError(f"invalid UTF-8 CSV {path.name}: {exc}") from exc
    return records

def _canonical_uuid(value: str) -> str | None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return None
    return value if str(parsed) == value else None

def _metadata_identity(raw: str, *, label: str) -> tuple[str, str]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise DataContractError(f"{label} metadata is not a Python-literal object") from exc
    if not isinstance(value, dict) or not isinstance(value.get("ID"), str) or not value["ID"]:
        raise DataContractError(f"{label} metadata lacks a nonempty ID")
    document_id = value["ID"]
    tokens = document_id.split("_")
    if "UK" in tokens:
        country = "UK"
    elif "Finnish" in tokens:
        country = "Finland"
    else:
        raise DataContractError(f"{label} metadata has an unknown source country: {document_id}")
    return document_id, country

def _decode_bio(words: tuple[str, ...], labels: tuple[str, ...], *, label: str) -> tuple[dict[str, Any], ...]:
    if len(words) != len(labels):
        raise DataContractError(
            f"{label} has {len(words)} processed words but {len(labels)} BIO labels"
        )
    entities: list[dict[str, Any]] = []
    current_start: int | None = None
    current_type: str | None = None

    def close(end: int) -> None:
        nonlocal current_start, current_type
        if current_start is not None and current_type is not None:
            entities.append(
                {
                    "start": current_start,
                    "end": end,
                    "type": current_type,
                    "text": " ".join(words[current_start : end + 1]),
                }
            )
        current_start = None
        current_type = None

    for index, tag in enumerate(labels):
        if tag == "O":
            close(index - 1)
            continue
        if "-" not in tag:
            raise DataContractError(f"{label} has invalid BIO tag {tag!r}")
        prefix, raw_type = tag.split("-", 1)
        entity_type = LOWER_ENTITY_TYPES.get(raw_type)
        if entity_type is None or prefix not in {"B", "I"}:
            raise DataContractError(f"{label} has invalid BIO tag {tag!r}")
        if prefix == "B":
            close(index - 1)
            current_start, current_type = index, entity_type
        elif current_start is None or current_type != entity_type:
            raise DataContractError(
                f"{label} has an I-tag without a matching B-tag at word {index}"
            )
    close(len(words) - 1)
    return tuple(entities)

def _load_entities(path: Path) -> dict[str, EntityRecord]:
    entities: dict[str, EntityRecord] = {}
    for row_number, row in _read_csv(path, ENTITY_HEADER):
        example_id = row["example_id"]
        if _canonical_uuid(example_id) is None:
            raise DataContractError(f"{path.name} record {row_number} has a noncanonical UUID")
        if example_id in entities:
            raise DataContractError(f"{path.name} duplicates sentence ID {example_id}")
        words = tuple(row["processed_content"].split())
        labels = tuple(row["label"].split())
        if not words:
            raise DataContractError(f"{path.name} record {row_number} has no processed words")
        source_document_id, country = _metadata_identity(
            row["metadata"], label=f"{path.name} record {row_number}"
        )
        decoded = _decode_bio(
            words, labels, label=f"{path.name} record {row_number}"
        )
        entities[example_id] = EntityRecord(
            example_id=example_id,
            content=row["content"],
            processed_content=row["processed_content"],
            words=words,
            labels=labels,
            metadata=row["metadata"],
            source_document_id=source_document_id,
            source_country=country,
            entities=decoded,
            source_row=row_number,
        )
    return entities

def _load_entity_boundary(path: Path) -> dict[str, dict[str, str]]:
    """Load task-split IDs without treating their stale labels as authoritative."""

    rows: dict[str, dict[str, str]] = {}
    allowed_tags = {
        "O",
        *(f"{prefix}-{entity_type.lower()}" for entity_type in ENTITY_TYPES for prefix in ("B", "I")),
    }
    for row_number, row in _read_csv(path, ENTITY_HEADER):
        example_id = row["example_id"]
        if _canonical_uuid(example_id) is None:
            raise DataContractError(f"{path.name} record {row_number} has a noncanonical UUID")
        if example_id in rows:
            raise DataContractError(f"{path.name} duplicates sentence ID {example_id}")
        words = row["processed_content"].split()
        labels = row["label"].split()
        if not words or len(words) != len(labels) or any(tag not in allowed_tags for tag in labels):
            raise DataContractError(
                f"{path.name} record {row_number} violates the task-split BIO field schema"
            )
        _metadata_identity(row["metadata"], label=f"{path.name} record {row_number}")
        rows[example_id] = row
    return rows

def _repair_relation_id(
    row: dict[str, str],
    *,
    row_number: int,
    file_label: str,
    entities: dict[str, EntityRecord],
) -> tuple[str, dict[str, Any] | None]:
    raw = row["example_id"]
    canonical = _canonical_uuid(raw)
    if canonical is not None:
        return canonical, None

    parts = raw.split("\t")
    candidate = parts[0].strip() if parts else ""
    embedded = parts[1:]
    normal_values = [row[field] for field in RELATION_HEADER[1:]]
    if (
        _canonical_uuid(candidate) is None
        or len(embedded) < 2
        or len(embedded) > len(normal_values)
        or embedded != normal_values[: len(embedded)]
    ):
        raise DataContractError(
            f"{file_label} record {row_number} has an unauditable non-UUID example_id"
        )
    entity = entities.get(candidate)
    if entity is None or row["content"] != entity.content or row["metadata"] != entity.metadata:
        raise DataContractError(
            f"{file_label} record {row_number} repair fields do not match the entity row"
        )
    before_bytes = raw.encode("utf-8")
    after_bytes = candidate.encode("utf-8")
    ledger = {
        "file": file_label,
        "csv_record_number": row_number,
        "reason": "tab-expanded duplicate fields in non-UUID example_id",
        "before": raw,
        "after": candidate,
        "before_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "after_sha256": hashlib.sha256(after_bytes).hexdigest(),
        "embedded_field_count": len(embedded),
        "content_sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
        "metadata_sha256": hashlib.sha256(row["metadata"].encode("utf-8")).hexdigest(),
    }
    return candidate, ledger

def _canonical_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())

def _parse_markers(tagged: str, *, label: str) -> tuple[str, dict[str, tuple[int, int, str]]]:
    clean_parts: list[str] = []
    clean_length = 0
    cursor = 0
    active: str | None = None
    starts: dict[str, int] = {}
    raw_spans: dict[str, tuple[int, int]] = {}
    counts = Counter()
    for match in TAG_RE.finditer(tagged):
        segment = tagged[cursor : match.start()]
        clean_parts.append(segment)
        clean_length += len(segment)
        token = match.group(0)
        marker = "1" if "e1" in token else "2"
        closing = token.startswith("</")
        if closing:
            if active != marker or marker not in starts:
                raise DataContractError(f"{label} has malformed or nested entity markers")
            raw_spans[marker] = (starts[marker], clean_length)
            active = None
        else:
            if active is not None or marker in starts:
                raise DataContractError(f"{label} has duplicate or nested entity markers")
            starts[marker] = clean_length
            counts[marker] += 1
            active = marker
        cursor = match.end()
    clean_parts.append(tagged[cursor:])
    clean = "".join(clean_parts)
    if active is not None or set(raw_spans) != {"1", "2"} or counts != Counter({"1": 1, "2": 1}):
        raise DataContractError(f"{label} must contain exactly one closed e1 and e2 marker")
    if re.search(r"</?e[12](?:\s|>|/)", clean, flags=re.IGNORECASE):
        raise DataContractError(f"{label} contains an unrecognized marker variant")
    markers: dict[str, tuple[int, int, str]] = {}
    for marker, (start, end) in raw_spans.items():
        markers[marker] = (
            len(_canonical_surface(clean[:start])),
            len(_canonical_surface(clean[:end])),
            clean[start:end],
        )
    return clean, markers

def _alignment_choice(
    entity: EntityRecord, marker: tuple[int, int, str]
) -> tuple[dict[str, Any] | None, str, dict[str, list[dict[str, Any]]]]:
    """Diagnose one marker without hiding later official-data failures."""

    diagnosis = _marker_alignment_diagnosis(entity, marker)
    exact_candidates = diagnosis["exact"]
    containing_candidates = diagnosis["contained_by_one_span"]
    if len(exact_candidates) == 1:
        return dict(exact_candidates[0]), "exact_span", diagnosis
    if not exact_candidates and len(containing_candidates) == 1:
        return (
            dict(containing_candidates[0]),
            "marker_contained_in_typed_span",
            diagnosis,
        )
    return None, "unresolved", diagnosis

def _alignment_issue(
    *,
    row_number: int,
    example_id: str,
    partition: str,
    relation: str,
    role: str,
    marker: tuple[int, int, str],
    diagnosis: dict[str, list[dict[str, Any]]],
    entity: EntityRecord,
    tagged_sentence: str,
) -> dict[str, Any]:
    marker_start, marker_end, marker_text = marker
    overlapping = [_span_for_output(span) for span in diagnosis["overlapping"]]
    if len(overlapping) > 1:
        category = "multiple_bio_spans_overlap"
    elif len(overlapping) == 1:
        category = "partial_single_bio_span_overlap"
    else:
        category = "no_bio_span_overlap"
    return {
        "annotation_row": row_number,
        "example_id": example_id,
        "official_entity_partition": partition,
        "entity_annotation_row": entity.source_row,
        "relation": relation,
        "marker_role": role,
        "marker_text": marker_text,
        "canonical_character_start": marker_start,
        "canonical_character_end_exclusive": marker_end,
        "category": category,
        "overlapping_typed_bio_spans": overlapping,
        "authoritative_typed_bio_spans": [
            _span_for_output(span) for span in entity.entities
        ],
        "tagged_sentence": tagged_sentence,
    }

def _marker_alignment_diagnosis(
    entity: EntityRecord, marker: tuple[int, int, str]
) -> dict[str, list[dict[str, Any]]]:
    """Return exact/containing/overlap candidates without choosing a repair policy."""

    marker_start, marker_end, marker_text = marker
    exact_candidates: list[dict[str, Any]] = []
    containing_candidates: list[dict[str, Any]] = []
    overlapping_candidates: list[dict[str, Any]] = []
    marker_canonical = _canonical_surface(marker_text)
    for span in entity.entities:
        start = len(_canonical_surface(" ".join(entity.words[: span["start"]])))
        end = len(_canonical_surface(" ".join(entity.words[: span["end"] + 1])))
        span_canonical = _canonical_surface(span["text"])
        if start < marker_end and marker_start < end:
            overlapping_candidates.append(span)
        if (
            start == marker_start
            and end == marker_end
            and span_canonical == marker_canonical
        ):
            exact_candidates.append(span)
        elif start <= marker_start and marker_end <= end:
            relative_start = marker_start - start
            relative_end = relative_start + len(marker_canonical)
            if span_canonical[relative_start:relative_end] == marker_canonical:
                containing_candidates.append(span)
    return {
        "exact": exact_candidates,
        "contained_by_one_span": containing_candidates,
        "overlapping": overlapping_candidates,
    }

def _load_relations_all(
    path: Path,
    entities: dict[str, EntityRecord],
    *,
    expected_repaired_uuid: str,
    partition_by_id: dict[str, str] | None = None,
) -> tuple[
    dict[str, list[RelationRecord]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, Any],
    list[dict[str, Any]],
]:
    relations: dict[str, list[RelationRecord]] = {}
    raw_relation_ids: set[str] = set()
    repair_ledger: list[dict[str, Any]] = []
    row_count = positive_count = none_count = 0
    strict_keys: dict[tuple[Any, ...], RelationRecord] = {}
    alignment_counts = Counter()
    alignment_issues: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for row_number, row in _read_csv(path, RELATION_HEADER):
        row_count += 1
        example_id, ledger = _repair_relation_id(
            row,
            row_number=row_number,
            file_label="annotated_data/relations/all.csv",
            entities=entities,
        )
        raw_relation_ids.add(example_id)
        if ledger is not None:
            repair_ledger.append(ledger)
        entity = entities.get(example_id)
        if entity is None:
            raise DataContractError(
                f"relations/all.csv record {row_number} references unknown sentence {example_id}"
            )
        if _canonical_surface(row["content"]) != _canonical_surface(entity.content):
            raise DataContractError(
                f"relations/all.csv record {row_number} content differs from entities/all.csv"
            )
        relation = row["relation_type"]
        raw_row = {
            "annotation_row": row_number,
            "example_id": example_id,
            "source_example_id": row["example_id"],
            "content": entity.content,
            "metadata": entity.metadata,
            "tagged_sentence": row["tagged_sentence"],
            "relation": relation,
            "uuid_repair_applied": ledger is not None,
        }
        if relation == "none":
            none_count += 1
            raw_rows.append(raw_row)
            continue
        if relation not in RELATION_TYPES:
            raise DataContractError(
                f"relations/all.csv record {row_number} has unknown relation {relation!r}"
            )
        positive_count += 1
        clean, markers = _parse_markers(
            row["tagged_sentence"], label=f"relations/all.csv record {row_number}"
        )
        canonical_clean = _canonical_surface(clean)
        if canonical_clean not in {
            _canonical_surface(entity.content),
            _canonical_surface(entity.processed_content),
        }:
            raise DataContractError(
                f"relations/all.csv record {row_number} tagged sentence differs from entity text"
            )
        aligned: dict[str, tuple[dict[str, Any] | None, str]] = {}
        for marker_id, role in (("1", "head"), ("2", "tail")):
            span, mode, diagnosis = _alignment_choice(entity, markers[marker_id])
            alignment_counts[mode] += 1
            aligned[marker_id] = (span, mode)
            if span is None:
                partition = (
                    partition_by_id.get(example_id, "unknown")
                    if partition_by_id is not None
                    else "unknown"
                )
                alignment_issues.append(
                    _alignment_issue(
                        row_number=row_number,
                        example_id=example_id,
                        partition=partition,
                        relation=relation,
                        role=role,
                        marker=markers[marker_id],
                        diagnosis=diagnosis,
                        entity=entity,
                        tagged_sentence=row["tagged_sentence"],
                    )
                )
        head, head_alignment = aligned["1"]
        tail, tail_alignment = aligned["2"]
        raw_row["head_alignment"] = head_alignment
        raw_row["tail_alignment"] = tail_alignment
        raw_row["strict_eligible"] = head is not None and tail is not None
        raw_rows.append(raw_row)
        if head is None or tail is None:
            continue
        key = (
            example_id,
            head["start"],
            head["end"],
            head["type"],
            relation,
            tail["start"],
            tail["end"],
            tail["type"],
        )
        if key in strict_keys:
            existing = strict_keys[key]
            replacement = replace(
                existing, source_rows=(*existing.source_rows, row_number)
            )
            values = relations[example_id]
            values[values.index(existing)] = replacement
            strict_keys[key] = replacement
            continue
        record = RelationRecord(
            example_id,
            relation,
            head,
            tail,
            (row_number,),
            head_alignment,
            tail_alignment,
        )
        strict_keys[key] = record
        relations.setdefault(example_id, []).append(record)
    if len(repair_ledger) != 1 or repair_ledger[0]["after"] != expected_repaired_uuid:
        raise DataContractError(
            "relations/all.csv must contain exactly the approved one-row UUID repair"
        )
    affected_rows = sorted({issue["annotation_row"] for issue in alignment_issues})
    affected_ids = sorted({issue["example_id"] for issue in alignment_issues})
    categories = Counter(issue["category"] for issue in alignment_issues)
    partitions = Counter(issue["official_entity_partition"] for issue in alignment_issues)
    audit = {
        "schema_version": "phase-b-gold-alignment-audit-2.0",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": "CODE-ACCORD-v1.0.0",
        "status": "raw_preserved_typed_strict_materialized",
        "policy": "unique-typed-bio-span-by-canonical-position-1.0",
        "summary": {
            "relation_rows_scanned": row_count,
            "positive_relation_rows_scanned": positive_count,
            "none_relation_rows_scanned": none_count,
            "accepted_uuid_repair_rows": len(repair_ledger),
            "marker_arguments_scanned": positive_count * 2,
            "exact_span_alignments": alignment_counts["exact_span"],
            "marker_contained_in_typed_span_alignments": alignment_counts["marker_contained_in_typed_span"],
            "unresolved_marker_arguments": len(alignment_issues),
            "affected_positive_relation_rows": len(affected_rows),
            "affected_sentence_ids": len(affected_ids),
            "affected_rows": affected_rows,
            "affected_ids": affected_ids,
            "unresolved_categories": dict(sorted(categories.items())),
            "official_entity_partitions": dict(sorted(partitions.items())),
            "typed_strict_eligible_positive_rows": positive_count - len(affected_rows),
            "typed_strict_unique_triples": sum(len(values) for values in relations.values()),
        },
        "decision": (
            "Raw provenance preserves every official relation row. The nine rows with "
            "unresolved marker arguments remain auditable but are ineligible for typed "
            "strict gold; no endpoint is projected, expanded, or manually typed."
        ),
        "issues": alignment_issues,
    }
    for values in relations.values():
        values.sort(
            key=lambda item: (
                item.head["start"],
                item.head["end"],
                item.head["type"],
                item.relation,
                item.tail["start"],
                item.tail["end"],
                item.tail["type"],
                item.head_alignment,
                item.tail_alignment,
                item.source_rows,
            )
        )
    return relations, repair_ledger, {
        "relation_rows": row_count,
        "positive_relation_rows": positive_count,
        "none_relation_rows": none_count,
        # These are immutable-corpus counts, not typed-strict eligibility counts.
        # A sentence whose relation markers are all ineligible remains relation-covered
        # in raw provenance and must not make the source-identity check drift.
        "relation_covered_sentences": len(raw_relation_ids),
        "entity_only_sentences": len(set(entities) - raw_relation_ids),
    }, audit, raw_rows

def _relation_task_ids(
    path: Path, entities: dict[str, EntityRecord], file_label: str
) -> tuple[int, set[str], int, int]:
    ids: set[str] = set()
    raw_ids: set[str] = set()
    repairs = 0
    rows = _read_csv(path, RELATION_HEADER)
    for row_number, row in rows:
        raw_ids.add(row["example_id"])
        example_id, ledger = _repair_relation_id(
            row,
            row_number=row_number,
            file_label=file_label,
            entities=entities,
        )
        if ledger is not None:
            repairs += 1
        entity = entities.get(example_id)
        if entity is None or _canonical_surface(row["content"]) != _canonical_surface(
            entity.content
        ):
            raise DataContractError(f"{file_label} record {row_number} does not match entities/all.csv")
        if row["relation_type"] not in {*RELATION_TYPES, "none"}:
            raise DataContractError(f"{file_label} record {row_number} has an unknown relation")
        ids.add(example_id)
    return len(rows), ids, repairs, len(raw_ids)

def _span_for_output(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": span["start"],
        "end": span["end"],
        "type": span["type"],
        "text": span["text"],
    }
