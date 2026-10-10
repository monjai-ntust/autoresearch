"""Strict, leakage-safe CODE-ACCORD reconstruction for the Path A workflow."""

from __future__ import annotations

from utils.preparation.materialization import _annotation_bundle_sha256, _materialize_dataset, _tree_inventory

from utils.preparation.annotations import DatasetContract, EntityRecord, RelationRecord, _load_entities, _load_entity_boundary, _load_relations_all, _relation_task_ids

import os
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from utils.common.config import PipelineConfig
from utils.common.constants import PROTOCOL_ID, TREE_HASH_REVISION
from utils.common.artifact_io import DataContractError, atomic_write_json, load_json, md5_file, sha256_file
from utils.common.paths import RunLayout
from utils.preparation.split import SplitItem, build_official_code_split


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
