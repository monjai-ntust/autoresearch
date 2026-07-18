"""Strict, leakage-safe CODE-ACCORD reconstruction for the Path A workflow."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import unicodedata
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from config import PipelineConfig
from constants import (
    CODE_ACCORD_ANNOTATION_FILES,
    CODE_ACCORD_COUNTS,
    CODE_ACCORD_RELATION_SPLIT_AUDIT,
    CODE_ACCORD_REPAIRED_UUID,
    ENTITY_TYPES,
    PROTOCOL_ID,
    RELATION_TYPES,
    TREE_HASH_REVISION,
)
from phase_b_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_bytes,
    load_json,
    md5_file,
    sha256_file,
)
from paths import RunLayout
from split import SplitItem, build_official_code_split, official_split_manifest


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


def dataset_contract_from_config(config: PipelineConfig) -> DatasetContract:
    dataset = config.value["dataset"]
    return DatasetContract(
        dataset_id=dataset["dataset_id"],
        archive_bytes=dataset["archive_bytes"],
        archive_md5=dataset["archive_md5"],
        annotation_files=dataset["annotation_files"],
        counts=dataset["expected_counts"],
        relation_split_audit=dataset["relation_split_audit"],
        repaired_uuid=dataset["repaired_uuid"],
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def extract_zip_safely(
    archive: Path,
    destination: Path,
    *,
    include_suffixes: tuple[str, ...] | None = None,
) -> Path:
    """Extract selected immutable ZIP bytes without links, traversal, or overwrite."""

    if destination.exists():
        if not destination.is_dir():
            raise DataContractError("archive extraction destination is not a directory")
        return destination
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise DataContractError(
            f"a retained partial extraction blocks overwrite: {staging.name}"
        )
    staging.mkdir(parents=True)
    staging_root = staging.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = member.filename
                if not name or "\\" in name or "\x00" in name:
                    raise DataContractError(f"unsafe ZIP member name: {name!r}")
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts or any(
                    ":" in part for part in pure.parts
                ):
                    raise DataContractError(f"ZIP member escapes extraction root: {name!r}")
                mode = (member.external_attr >> 16) & 0xFFFF
                if mode and stat.S_ISLNK(mode):
                    raise DataContractError(f"ZIP symbolic links are forbidden: {name!r}")
                selected = include_suffixes is None or any(
                    pure.parts[-len(PurePosixPath(suffix).parts) :]
                    == PurePosixPath(suffix).parts
                    for suffix in include_suffixes
                )
                if not selected:
                    continue
                target = staging.joinpath(*pure.parts)
                resolved = target.resolve(strict=False)
                if not _is_within(resolved, staging_root):
                    raise DataContractError(f"ZIP member resolves outside extraction root: {name!r}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise DataContractError(f"duplicate ZIP output path: {name!r}")
                with bundle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
        os.replace(staging, destination)
    except BaseException:
        # The partial tree is evidence of the failed extraction and is not erased.
        raise
    return destination


def _locate_annotation_files(
    extracted_root: Path, contract: DatasetContract
) -> dict[str, Path]:
    files = [path for path in extracted_root.rglob("*") if path.is_file()]
    located: dict[str, Path] = {}
    for relative, expected in contract.annotation_files.items():
        suffix = PurePosixPath(relative).parts
        matches = [path for path in files if path.parts[-len(suffix) :] == suffix]
        if len(matches) != 1:
            raise DataContractError(
                f"expected exactly one extracted {relative}, found {len(matches)}"
            )
        path = matches[0]
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected["bytes"] or actual_sha256 != expected["sha256"]:
            raise DataContractError(
                f"annotation identity mismatch for {relative}: "
                f"bytes={actual_bytes}, sha256={actual_sha256}"
            )
        located[relative] = path
    return located


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


def _annotation_bundle_sha256(contract: DatasetContract) -> str:
    digest = hashlib.sha256()
    for path, identity in sorted(contract.annotation_files.items()):
        digest.update(
            f"{path}\0{identity['bytes']}\0{identity['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha256 = sha256_file(path)
        rows.append({"path": relative, "bytes": size, "sha256": file_sha256})
        digest.update(f"{relative}\0{size}\0{file_sha256}\n".encode("utf-8"))
    return rows, digest.hexdigest()


def _tsv_bytes(rows: Iterable[Iterable[Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    for row in rows:
        writer.writerow(list(row))
    return output.getvalue()


def _span_for_output(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": span["start"],
        "end": span["end"],
        "type": span["type"],
        "text": span["text"],
    }


def _materialize_dataset(
    destination: Path,
    *,
    contract: DatasetContract,
    entities: dict[str, EntityRecord],
    relations: dict[str, list[RelationRecord]],
    repair_ledger: list[dict[str, Any]],
    alignment_audit: dict[str, Any],
    raw_relation_rows: list[dict[str, Any]],
    split: Any,
    relation_split_audit: dict[str, int],
) -> None:
    destination.mkdir()
    annotation_bundle = _annotation_bundle_sha256(contract)
    split_by_id = {**split.assignments, **{example_id: "test" for example_id in split.test_ids}}

    records: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "development": [],
        "test": [],
    }
    for example_id in sorted(entities):
        entity = entities[example_id]
        relation_rows = relations.get(example_id, [])
        relation_values = [
            {
                "head": _span_for_output(item.head),
                "relation": item.relation,
                "tail": _span_for_output(item.tail),
                "annotation_rows": list(item.source_rows),
                "head_alignment": item.head_alignment,
                "tail_alignment": item.tail_alignment,
            }
            for item in relation_rows
        ]
        record = {
            "protocol_id": PROTOCOL_ID,
            "dataset_id": contract.dataset_id,
            "example_id": example_id,
            "source_document_id": entity.source_document_id,
            "source_country": entity.source_country,
            "split": split_by_id[example_id],
            "content": entity.content,
            "processed_content": entity.processed_content,
            "words": list(entity.words),
            "entities": [_span_for_output(span) for span in entity.entities],
            "relations": relation_values,
        }
        records.append(record)
        by_split[record["split"]].append(record)

    atomic_write_jsonl(destination / "sentences.jsonl", records)
    for name, values in by_split.items():
        atomic_write_jsonl(destination / f"{name}.jsonl", values)
    atomic_write_text(destination / "train_ids.txt", "".join(f"{value}\n" for value in split.train_ids))
    atomic_write_text(
        destination / "development_ids.txt",
        "".join(f"{value}\n" for value in split.development_ids),
    )
    atomic_write_text(destination / "test_ids.txt", "".join(f"{value}\n" for value in split.test_ids))

    strata_rows: list[list[Any]] = [
        ["example_id", "split", "source_country", "entity_types", "relation_types", "labels"]
    ]
    for record in records:
        entity_types = sorted({span["type"] for span in record["entities"]})
        relation_types = sorted({item["relation"] for item in record["relations"]})
        labels = [
            *(f"entity:{value}" for value in entity_types),
            *(f"relation:{value}" for value in relation_types),
            f"country:{record['source_country']}",
        ]
        strata_rows.append(
            [
                record["example_id"],
                record["split"],
                record["source_country"],
                "|".join(entity_types),
                "|".join(relation_types),
                "|".join(sorted(labels)),
            ]
        )
    atomic_write_text(destination / "strata.tsv", _tsv_bytes(strata_rows))

    manifest = official_split_manifest(split)
    manifest["labels"] = [
        "entity_type_presence",
        "positive_relation_type_presence",
        "source_country",
    ]
    manifest["tie_break"] = "seeded_generator_then_ascending_uuid"
    atomic_write_json(destination / "split-manifest.json", manifest)
    atomic_write_jsonl(destination / "repair-ledger.jsonl", repair_ledger)
    atomic_write_jsonl(destination / "raw-relation-provenance.jsonl", raw_relation_rows)
    atomic_write_json(destination / "gold-alignment-audit.json", alignment_audit)
    atomic_write_jsonl(
        destination / "typed-strict-eligibility.jsonl",
        [
            {
                "annotation_row": row["annotation_row"],
                "example_id": row["example_id"],
                "relation": row["relation"],
                "strict_eligible": row.get("strict_eligible", False),
                "head_alignment": row.get("head_alignment"),
                "tail_alignment": row.get("tail_alignment"),
            }
            for row in raw_relation_rows
            if row["relation"] != "none"
        ],
    )

    for split_name in ("development", "test"):
        gold_records: list[dict[str, Any]] = []
        for record in by_split[split_name]:
            gold_records.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "split_id": f"CODE-SPLIT-1:{split_name}",
                    "example_id": record["example_id"],
                    "source_document_id": record["source_document_id"],
                    "gold_triples": [
                        {
                            "head": item["head"],
                            "relation": item["relation"],
                            "tail": item["tail"],
                        }
                        for item in record["relations"]
                    ],
                    "input_hashes": {"annotation_bundle": annotation_bundle},
                }
            )
        atomic_write_jsonl(destination / f"{split_name}-gold.jsonl", gold_records)

    distributions: dict[str, Any] = {}
    for split_name, values in by_split.items():
        distributions[split_name] = {
            "sentences": len(values),
            "countries": dict(sorted(Counter(value["source_country"] for value in values).items())),
            "entity_spans": dict(
                sorted(Counter(span["type"] for value in values for span in value["entities"]).items())
            ),
            "positive_relations": dict(
                sorted(
                    Counter(item["relation"] for value in values for item in value["relations"]).items()
                )
            ),
        }
    atomic_write_json(destination / "distributions.json", distributions)
    atomic_write_json(
        destination / "attribution.json",
        {
            "dataset_id": contract.dataset_id,
            "title": "CODE-ACCORD",
            "doi": "10.5281/zenodo.10210022",
            "descriptor_doi": "10.1038/s41597-024-04320-x",
            "license": "CC-BY-4.0",
            "required_attribution_retained": True,
        },
    )

    pre_manifest_rows, _ = _tree_inventory(destination)
    atomic_write_json(
        destination / "data-manifest.json",
        {
            "schema_version": "phase-b-code-accord-data-manifest-1.0",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": contract.dataset_id,
            "annotation_bundle_sha256": annotation_bundle,
            "annotation_files": contract.annotation_files,
            "counts": contract.counts,
            "relation_split_audit": relation_split_audit,
            "repair": {
                "accepted_rows": 1,
                "repaired_uuid": contract.repaired_uuid,
                "policy": "audited-tab-expanded-duplicate-fields-1.0",
            },
            "alignment": {
                "policy": "unique-typed-bio-span-by-canonical-position-1.0",
                "unresolved": alignment_audit["summary"]["unresolved_marker_arguments"],
                "ineligible_positive_rows": alignment_audit["summary"]["affected_positive_relation_rows"],
                "typed_strict_eligible_positive_rows": alignment_audit["summary"]["typed_strict_eligible_positive_rows"],
                "typed_strict_unique_triples": alignment_audit["summary"]["typed_strict_unique_triples"],
            },
            "split": {
                "split_id": "CODE-SPLIT-1",
                "seed": 42,
                "train": len(split.train_ids),
                "development": len(split.development_ids),
                "test": len(split.test_ids),
            },
            "transformations": [
                "verify-six-annotation-byte-identities",
                "repair-one-tab-expanded-example-id",
                "exclude-explicit-none-pairs-from-positive-gold",
                "preserve-all-relation-rows-as-raw-provenance",
                "derive-typed-strict-gold-from-unique-bio-alignments",
                "retain-ineligible-marker-rows-in-alignment-audit",
                "retain-five-entity-only-sentences",
                "materialize-code-split-1",
            ],
            "artifacts_before_data_manifest": pre_manifest_rows,
        },
    )


def _prepare_corpus(
    annotation_paths: dict[str, Path], contract: DatasetContract
) -> tuple[
    dict[str, EntityRecord],
    dict[str, list[RelationRecord]],
    list[dict[str, Any]],
    Any,
    dict[str, int],
    dict[str, Any],
    list[dict[str, Any]],
]:
    entities_all = _load_entities(annotation_paths["annotated_data/entities/all.csv"])
    entities_train = _load_entity_boundary(
        annotation_paths["annotated_data/entities/train.csv"]
    )
    entities_test = _load_entity_boundary(
        annotation_paths["annotated_data/entities/test.csv"]
    )

    if set(entities_train) & set(entities_test):
        raise DataContractError("official entity train/test sentence IDs overlap")
    if set(entities_train) | set(entities_test) != set(entities_all):
        raise DataContractError("entities/all.csv differs from the train/test ID union")
    entity_counts = {
        "sentences": len(entities_all),
        "entity_train_sentences": len(entities_train),
        "entity_test_sentences": len(entities_test),
        "entity_spans": sum(len(record.entities) for record in entities_all.values()),
    }
    for field, actual in entity_counts.items():
        if actual != contract.counts[field]:
            raise DataContractError(
                f"CODE-ACCORD {field} mismatch: expected {contract.counts[field]}, got {actual}"
            )

    relations, repair_ledger, relation_counts, alignment_audit, raw_relation_rows = _load_relations_all(
            annotation_paths["annotated_data/relations/all.csv"],
            entities_all,
            expected_repaired_uuid=contract.repaired_uuid,
            partition_by_id={
                **{example_id: "train" for example_id in entities_train},
                **{example_id: "test" for example_id in entities_test},
            },
        )
    for field, actual in relation_counts.items():
        if actual != contract.counts[field]:
            raise DataContractError(
                f"CODE-ACCORD {field} mismatch: expected {contract.counts[field]}, got {actual}"
            )

    train_rows, relation_train_ids, train_repairs, train_raw_id_count = _relation_task_ids(
        annotation_paths["annotated_data/relations/train.csv"],
        entities_all,
        "annotated_data/relations/train.csv",
    )
    test_rows, relation_test_ids, test_repairs, test_raw_id_count = _relation_task_ids(
        annotation_paths["annotated_data/relations/test.csv"],
        entities_all,
        "annotated_data/relations/test.csv",
    )
    relation_split_audit = {
        "train_rows": train_rows,
        "test_rows": test_rows,
        # Preserve physical task-file coverage before the approved ID repair;
        # canonicalized IDs below are used for every entity/split intersection.
        "train_unique_sentence_ids": train_raw_id_count,
        "test_unique_sentence_ids": test_raw_id_count,
        "train_test_sentence_id_overlap": len(relation_train_ids & relation_test_ids),
        "entity_test_ids_in_relation_train": len(set(entities_test) & relation_train_ids),
        "entity_test_ids_in_relation_test": len(set(entities_test) & relation_test_ids),
        "entity_train_ids_in_relation_test": len(set(entities_train) & relation_test_ids),
        "task_split_non_uuid_rows_repaired_for_audit": train_repairs + test_repairs,
    }
    partition_rows = {**entities_train, **entities_test}
    entity_difference_counts = Counter()
    differing_entity_rows = 0
    for example_id, partition in partition_rows.items():
        authoritative = entities_all[example_id]
        differences = []
        for field, authoritative_value in (
            ("content", authoritative.content),
            ("processed_content", authoritative.processed_content),
            ("label", " ".join(authoritative.labels)),
            ("metadata", authoritative.metadata),
        ):
            if partition[field] != authoritative_value:
                differences.append(field)
                entity_difference_counts[field] += 1
        if differences:
            differing_entity_rows += 1
    relation_split_audit.update(
        {
            "entity_partition_rows_differing_from_all": differing_entity_rows,
            "entity_partition_content_differences": entity_difference_counts["content"],
            "entity_partition_processed_content_differences": entity_difference_counts[
                "processed_content"
            ],
            "entity_partition_label_differences": entity_difference_counts["label"],
            "entity_partition_metadata_differences": entity_difference_counts["metadata"],
        }
    )
    alignment_modes = Counter(
        mode
        for values in relations.values()
        for relation in values
        for mode in (relation.head_alignment, relation.tail_alignment)
    )
    relation_split_audit.update(
        {
            "marker_exact_span_alignments": alignment_modes["exact_span"],
            "marker_contained_span_alignments": alignment_modes[
                "marker_contained_in_typed_span"
            ],
            "unique_positive_strict_triples": sum(len(values) for values in relations.values()),
            "duplicate_positive_rows_collapsed": contract.counts[
                "positive_relation_rows"
            ]
            - sum(len(values) for values in relations.values()),
        }
    )
    if contract.relation_split_audit is not None:
        for field, expected in contract.relation_split_audit.items():
            actual = relation_split_audit.get(field)
            if actual != expected:
                raise DataContractError(
                    f"relation-task split audit {field} mismatch: expected {expected}, got {actual}"
                )

    split_items: list[SplitItem] = []
    for example_id in entities_train:
        entity = entities_all[example_id]
        relation_types = {item.relation for item in relations.get(example_id, [])}
        labels = {
            *(f"entity:{span['type']}" for span in entity.entities),
            *(f"relation:{relation}" for relation in relation_types),
            f"country:{entity.source_country}",
        }
        split_items.append(SplitItem(example_id, frozenset(labels)))
    split = build_official_code_split(split_items, entities_test)
    return (
        entities_all,
        relations,
        repair_ledger,
        split,
        relation_split_audit,
        alignment_audit,
        raw_relation_rows,
    )


def prepare_run(layout: RunLayout, config: PipelineConfig) -> dict[str, Any]:
    """Extract, validate, reconstruct, and independently rematerialize CODE-ACCORD."""

    layout.require_existing()
    contract = dataset_contract_from_config(config)
    acquisition_path = layout.resolve(
        "manifests/02-input-acquisition-manifest.json", must_exist=True
    )
    acquisition = load_json(acquisition_path)
    archive_value = acquisition.get("archive") if isinstance(acquisition, dict) else None
    if not isinstance(archive_value, dict) or not isinstance(archive_value.get("path"), str):
        raise DataContractError("input acquisition manifest lacks archive.path")
    archive_path = layout.resolve(archive_value["path"], must_exist=True)
    if (
        archive_path.stat().st_size != contract.archive_bytes
        or md5_file(archive_path) != contract.archive_md5
        or archive_value.get("bytes") != contract.archive_bytes
        or archive_value.get("md5") != contract.archive_md5
        or archive_value.get("sha256") != sha256_file(archive_path)
    ):
        raise DataContractError("acquisition manifest/archive identity mismatch")

    manifest_path = layout.resolve("manifests/03-data-preparation-manifest.json")
    data_root = layout.resolve("data-prepared")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        _, actual_tree_hash = _tree_inventory(data_root)
        if manifest.get("dataset_tree_sha256") != actual_tree_hash:
            raise DataContractError("existing prepared dataset tree hash mismatch")
        return manifest

    extracted = extract_zip_safely(
        archive_path,
        layout.resolve("inputs/extracted/CODE-ACCORD-v1.0.0-annotations"),
        include_suffixes=tuple(contract.annotation_files),
    )
    annotation_paths = _locate_annotation_files(extracted, contract)
    (
        entities,
        relations,
        repair_ledger,
        split,
        relation_split_audit,
        alignment_audit,
        raw_relation_rows,
    ) = _prepare_corpus(annotation_paths, contract)

    first = layout.resolve(".prepare-materialization-a.partial")
    second = layout.resolve(".prepare-materialization-b.partial")
    if first.exists() or second.exists():
        raise DataContractError("retained preparation staging tree blocks overwrite")
    _materialize_dataset(
        first,
        contract=contract,
        entities=entities,
        relations=relations,
        repair_ledger=repair_ledger,
        alignment_audit=alignment_audit,
        raw_relation_rows=raw_relation_rows,
        split=split,
        relation_split_audit=relation_split_audit,
    )
    _materialize_dataset(
        second,
        contract=contract,
        entities=entities,
        relations=relations,
        repair_ledger=repair_ledger,
        alignment_audit=alignment_audit,
        raw_relation_rows=raw_relation_rows,
        split=split,
        relation_split_audit=relation_split_audit,
    )
    first_rows, first_hash = _tree_inventory(first)
    second_rows, second_hash = _tree_inventory(second)
    if first_hash != second_hash or first_rows != second_rows:
        raise DataContractError("independent CODE-ACCORD materializations differ")
    if any(data_root.iterdir()):
        raise DataContractError("data-prepared must be empty before atomic promotion")
    data_root.rmdir()
    os.replace(first, data_root)
    shutil.rmtree(second)

    final_rows, final_hash = _tree_inventory(data_root)
    if final_hash != first_hash or final_rows != first_rows:
        raise DataContractError("promoted prepared dataset differs from verified staging")
    manifest = {
        "schema_version": "phase-b-data-preparation-manifest-2.0",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": contract.dataset_id,
        "archive_sha256": sha256_file(archive_path),
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "annotation_bundle_sha256": _annotation_bundle_sha256(contract),
        "tree_hash_revision": TREE_HASH_REVISION,
        "first_materialization_tree_sha256": first_hash,
        "second_materialization_tree_sha256": second_hash,
        "dataset_tree_sha256": final_hash,
        "byte_identical_independent_materializations": True,
        "counts": contract.counts,
        "split": {"train": 586, "development": 103, "test": 173, "seed": 42},
        "repair_rows": 1,
        "unresolved_alignments": alignment_audit["summary"]["unresolved_marker_arguments"],
        "typed_strict_ineligible_rows": alignment_audit["summary"]["affected_positive_relation_rows"],
        "typed_strict_eligible_rows": alignment_audit["summary"]["typed_strict_eligible_positive_rows"],
        "typed_strict_unique_triples": alignment_audit["summary"]["typed_strict_unique_triples"],
        "artifacts": final_rows,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def official_contract_defaults() -> DatasetContract:
    """Expose the immutable official values for focused fixture construction."""

    return DatasetContract(
        dataset_id="CODE-ACCORD-v1.0.0",
        archive_bytes=101265616,
        archive_md5="57e2efa465f41e2f582db62810fb50f5",
        annotation_files=CODE_ACCORD_ANNOTATION_FILES,
        counts=CODE_ACCORD_COUNTS,
        relation_split_audit=CODE_ACCORD_RELATION_SPLIT_AUDIT,
        repaired_uuid=CODE_ACCORD_REPAIRED_UUID,
    )
