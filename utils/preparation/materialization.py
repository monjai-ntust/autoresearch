"""preparation materialization responsibilities."""
from __future__ import annotations

from utils.common.execution import bind_json as atomic_write_json, retain_jsonl as atomic_write_jsonl, retain_text as atomic_write_text

from utils.preparation.annotations import DatasetContract, EntityRecord, RelationRecord, _span_for_output
import csv
import hashlib
import io
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from utils.common.constants import PROTOCOL_ID
from utils.common.artifact_io import sha256_file
from utils.preparation.split import official_split_manifest

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
    destination.mkdir(exist_ok=True)
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
    pre_manifest_rows = [row for row in pre_manifest_rows if row["path"] != "data-manifest.json"]
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
