"""Acquisition and deterministic CODE-ACCORD preparation regression tests."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path

from acquisition import ArchiveContract, download_verified_archive
from constants import CODE_ACCORD_COUNTS, CODE_ACCORD_REPAIRED_UUID
from phase_b_io import DataContractError, sha256_file
from paths import discover_source_root
from preparation import (
    DatasetContract,
    EntityRecord,
    _alignment_choice,
    _locate_annotation_files,
    _materialize_dataset,
    _parse_markers,
    _prepare_corpus,
    _tree_inventory,
    extract_zip_safely,
)


SOURCE_ROOT = discover_source_root(Path(__file__))
ENTITY_HEADER = ["example_id", "content", "processed_content", "label", "metadata"]
RELATION_HEADER = [
    "example_id",
    "content",
    "metadata",
    "tagged_sentence",
    "relation_type",
]
RELATIONS = [
    "selection",
    "necessity",
    "part-of",
    "not-part-of",
    "equal",
    "greater",
    "greater-equal",
    "less",
    "less-equal",
]
ENTITY_TYPES = ["object", "property", "quality", "value"]


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-b-prepare-", dir=output_root)


class _Response:
    def __init__(self, payload: bytes, status: int):
        self.payload = payload
        self.status = status
        self.offset = 0

    def read(self, size: int) -> bytes:
        value = self.payload[self.offset : self.offset + size]
        self.offset += len(value)
        return value

    def close(self) -> None:
        pass


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _tagged(words: list[str], head_index: int, tail_index: int) -> str:
    values = list(words)
    values[head_index] = f"<e1>{values[head_index]}</e1>"
    values[tail_index] = f"<e2>{values[tail_index]}</e2>"
    return " ".join(values)


def _build_official_shape_fixture(root: Path) -> DatasetContract:
    identifiers = [CODE_ACCORD_REPAIRED_UUID]
    identifiers.extend(
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"phase-b-code-accord-{index}"))
        for index in range(1, 862)
    )
    if len(set(identifiers)) != 862:
        raise AssertionError("fixture UUID collision")

    entity_rows: list[list[str]] = []
    entity_by_id: dict[str, list[str]] = {}
    words_by_id: dict[str, list[str]] = {}
    for index, example_id in enumerate(identifiers):
        span_count = 5 if index < 849 else 4
        words: list[str] = []
        labels: list[str] = []
        for word_index in range(10):
            if word_index % 2 == 0 and word_index // 2 < span_count:
                entity_index = word_index // 2
                words.append(f"entity{index}x{entity_index}")
                labels.append(f"B-{ENTITY_TYPES[(index + entity_index) % 4]}")
            else:
                words.append(f"token{index}x{word_index}")
                labels.append("O")
        content = " ".join(words)
        country = "UK" if index % 2 == 0 else "Finnish"
        metadata = "{'ID': '%d_%s_DocFixture'}" % (index, country)
        row = [example_id, content, content, " ".join(labels), metadata]
        entity_rows.append(row)
        entity_by_id[example_id] = row
        words_by_id[example_id] = words

    positive_rows: list[list[str]] = []
    for sentence_index, example_id in enumerate(identifiers[:857]):
        relation_count = 4 if sentence_index < 758 else 3
        entity_row = entity_by_id[example_id]
        words = words_by_id[example_id]
        entity_positions = [0, 2, 4, 6, 8]
        for relation_index in range(relation_count):
            head = entity_positions[relation_index % 4]
            tail = entity_positions[(relation_index + 1) % 4]
            relation = RELATIONS[(sentence_index + relation_index) % len(RELATIONS)]
            relation_id = example_id
            if not positive_rows:
                relation_id = f"{example_id}\t{entity_row[1]}\t{entity_row[4]}"
            positive_rows.append(
                [
                    relation_id,
                    entity_row[1],
                    entity_row[4],
                    _tagged(words, head, tail),
                    relation,
                ]
            )
    if len(positive_rows) != 3329:
        raise AssertionError(len(positive_rows))

    none_rows: list[list[str]] = []
    for index in range(1000):
        example_id = identifiers[index % 857]
        entity_row = entity_by_id[example_id]
        none_rows.append(
            [
                example_id,
                entity_row[1],
                entity_row[4],
                _tagged(words_by_id[example_id], 0, 2),
                "none",
            ]
        )
    relation_rows = positive_rows + none_rows

    annotation = root / "CODE-ACCORD-v1.0.0" / "annotated_data"
    _write_csv(annotation / "entities" / "all.csv", ENTITY_HEADER, entity_rows)
    _write_csv(annotation / "entities" / "train.csv", ENTITY_HEADER, entity_rows[:689])
    _write_csv(annotation / "entities" / "test.csv", ENTITY_HEADER, entity_rows[689:])
    _write_csv(annotation / "relations" / "all.csv", RELATION_HEADER, relation_rows)
    task_rows = [list(row) for row in relation_rows]
    task_rows[0][0] = CODE_ACCORD_REPAIRED_UUID
    _write_csv(annotation / "relations" / "train.csv", RELATION_HEADER, task_rows[:3463])
    _write_csv(annotation / "relations" / "test.csv", RELATION_HEADER, task_rows[3463:])

    identities = {}
    for relative in (
        "annotated_data/entities/all.csv",
        "annotated_data/entities/train.csv",
        "annotated_data/entities/test.csv",
        "annotated_data/relations/all.csv",
        "annotated_data/relations/train.csv",
        "annotated_data/relations/test.csv",
    ):
        path = root / "CODE-ACCORD-v1.0.0" / relative
        identities[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return DatasetContract(
        dataset_id="CODE-ACCORD-v1.0.0",
        archive_bytes=1,
        archive_md5="0" * 32,
        annotation_files=identities,
        counts=dict(CODE_ACCORD_COUNTS),
        relation_split_audit=None,
        repaired_uuid=CODE_ACCORD_REPAIRED_UUID,
    )


class AcquisitionTests(unittest.TestCase):
    def test_retained_partial_is_resumed_and_verified_before_promotion(self):
        payload = b"immutable-code-accord-fixture"
        contract = ArchiveContract(
            dataset_id="fixture",
            name="fixture.zip",
            url="https://invalid.example/fixture.zip",
            expected_bytes=len(payload),
            expected_md5=hashlib.md5(payload).hexdigest(),
            license="CC-BY-4.0",
        )
        with _temporary_output_directory() as temporary:
            destination = Path(temporary) / "fixture.zip"
            partial = destination.with_name(destination.name + ".partial")
            partial.write_bytes(payload[:8])

            def opener(request, timeout):
                self.assertEqual(request.headers["Range"], "bytes=8-")
                self.assertEqual(timeout, 120)
                return _Response(payload[8:], status=206)

            identity, status = download_verified_archive(
                contract, destination, opener=opener
            )
            self.assertEqual(status, "downloaded_and_verified")
            self.assertEqual(identity["bytes"], len(payload))
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_archive_checksum_mismatch_is_a_hard_stop(self):
        payload = b"wrong"
        contract = ArchiveContract(
            dataset_id="fixture",
            name="fixture.zip",
            url="https://invalid.example/fixture.zip",
            expected_bytes=len(payload),
            expected_md5="0" * 32,
            license="CC-BY-4.0",
        )
        with _temporary_output_directory() as temporary:
            with self.assertRaises(DataContractError):
                download_verified_archive(
                    contract,
                    Path(temporary) / "fixture.zip",
                    opener=lambda request, timeout: _Response(payload, status=200),
                )


class ExtractionTests(unittest.TestCase):
    def test_zip_path_traversal_is_rejected_and_partial_is_retained(self):
        with _temporary_output_directory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "forbidden")
            destination = root / "extracted"
            with self.assertRaises(DataContractError):
                extract_zip_safely(archive, destination)
            self.assertFalse((root.parent / "escape.txt").exists())
            self.assertTrue(destination.with_name("extracted.partial").is_dir())

    def test_selective_extraction_scans_unselected_members_for_traversal(self):
        with _temporary_output_directory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe-unselected.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("annotated_data/entities/all.csv", "selected")
                bundle.writestr("../unselected-escape.txt", "forbidden")
            destination = root / "selected"
            with self.assertRaises(DataContractError):
                extract_zip_safely(
                    archive,
                    destination,
                    include_suffixes=("annotated_data/entities/all.csv",),
                )
            self.assertFalse((root.parent / "unselected-escape.txt").exists())
            self.assertTrue(destination.with_name("selected.partial").is_dir())

    def test_selective_extraction_skips_irrelevant_long_paths(self):
        with _temporary_output_directory() as temporary:
            root = Path(temporary)
            archive = root / "selective.zip"
            selected = "annotated_data/entities/all.csv"
            irrelevant = "/".join(["very-long-directory-name"] * 20) + "/raw.pdf"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(f"bundle/{selected}", "selected")
                bundle.writestr(f"bundle/{irrelevant}", "irrelevant")
            destination = root / "selected"
            extract_zip_safely(
                archive, destination, include_suffixes=(selected,)
            )
            files = [path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()]
            self.assertEqual(files, [f"bundle/{selected}"])


class PreparationTests(unittest.TestCase):
    def test_marker_contained_in_one_typed_span_resolves_to_full_bio_span(self):
        words = ("A", "suitable", "for", "powering", "the", "lifting", "device")
        entity = EntityRecord(
            example_id=str(uuid.uuid4()),
            content=" ".join(words),
            processed_content=" ".join(words),
            words=words,
            labels=("O", "B-quality", "I-quality", "I-quality", "I-quality", "I-quality", "I-quality"),
            metadata="{'ID': '1_UK_Fixture'}",
            source_document_id="1_UK_Fixture",
            source_country="UK",
            entities=(
                {
                    "start": 1,
                    "end": 6,
                    "type": "Quality",
                    "text": "suitable for powering the lifting device",
                },
            ),
            source_row=2,
        )
        _, markers = _parse_markers(
            "A <e1>suitable for powering</e1> the <e2>lifting device</e2>",
            label="contained fixture",
        )
        span, mode, diagnosis = _alignment_choice(entity, markers["1"])
        self.assertEqual(mode, "marker_contained_in_typed_span")
        self.assertEqual(span, entity.entities[0])
        self.assertEqual(diagnosis["contained_by_one_span"], [entity.entities[0]])

    def test_all_gold_alignment_failures_are_retained_in_raw_view(self):
        with _temporary_output_directory() as temporary:
            root = Path(temporary)
            extracted = root / "extracted"
            contract = _build_official_shape_fixture(extracted)
            relation_path = (
                extracted
                / "CODE-ACCORD-v1.0.0"
                / "annotated_data"
                / "relations"
                / "all.csv"
            )
            with relation_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            for row_index in (1, 2):
                words = rows[row_index][1].split()
                rows[row_index][3] = _tagged(words, 1, 2)
            _write_csv(relation_path, rows[0], rows[1:])
            identity = contract.annotation_files["annotated_data/relations/all.csv"]
            identity["bytes"] = relation_path.stat().st_size
            identity["sha256"] = sha256_file(relation_path)

            paths = _locate_annotation_files(extracted, contract)
            _, relations, _, _, _, audit, raw_rows = _prepare_corpus(paths, contract)
            self.assertEqual(audit["status"], "raw_preserved_typed_strict_materialized")
            self.assertEqual(audit["summary"]["relation_rows_scanned"], 4329)
            self.assertEqual(audit["summary"]["none_relation_rows_scanned"], 1000)
            self.assertEqual(audit["summary"]["accepted_uuid_repair_rows"], 1)
            self.assertEqual(audit["summary"]["marker_arguments_scanned"], 6658)
            self.assertEqual(audit["summary"]["exact_span_alignments"], 6656)
            self.assertEqual(audit["summary"]["unresolved_marker_arguments"], 2)
            self.assertEqual(audit["summary"]["affected_positive_relation_rows"], 2)
            self.assertEqual(
                audit["summary"]["unresolved_categories"],
                {"no_bio_span_overlap": 2},
            )
            self.assertEqual(
                {issue["official_entity_partition"] for issue in audit["issues"]},
                {"train"},
            )
            self.assertEqual(audit["summary"]["typed_strict_eligible_positive_rows"], 3327)
            self.assertEqual(sum(len(values) for values in relations.values()), 3327)
            self.assertEqual(sum(row["relation"] != "none" for row in raw_rows), 3329)

    def test_official_shape_repair_alignment_split_and_second_materialization(self):
        with _temporary_output_directory() as temporary:
            root = Path(temporary)
            extracted = root / "extracted"
            contract = _build_official_shape_fixture(extracted)
            paths = _locate_annotation_files(extracted, contract)
            (
                entities,
                relations,
                repair,
                split,
                task_audit,
                alignment_audit,
                raw_rows,
            ) = _prepare_corpus(paths, contract)
            self.assertEqual(len(entities), 862)
            self.assertEqual(sum(len(value.entities) for value in entities.values()), 4297)
            self.assertEqual(sum(len(value) for value in relations.values()), 3329)
            self.assertEqual(len(relations), 857)
            self.assertEqual(repair[0]["after"], CODE_ACCORD_REPAIRED_UUID)
            self.assertEqual((len(split.train_ids), len(split.development_ids), len(split.test_ids)), (586, 103, 173))

            first = root / "first"
            second = root / "second"
            for destination in (first, second):
                _materialize_dataset(
                    destination,
                    contract=contract,
                    entities=entities,
                    relations=relations,
                    repair_ledger=repair,
                    alignment_audit=alignment_audit,
                    raw_relation_rows=raw_rows,
                    split=split,
                    relation_split_audit=task_audit,
                )
            first_rows, first_hash = _tree_inventory(first)
            second_rows, second_hash = _tree_inventory(second)
            self.assertEqual(first_rows, second_rows)
            self.assertEqual(first_hash, second_hash)

            with (first / "sentences.jsonl").open(encoding="utf-8") as handle:
                sentence_records = [json.loads(line) for line in handle]
            self.assertEqual(len(sentence_records), 862)
            self.assertEqual(sum(not row["relations"] for row in sentence_records), 5)
            with (first / "test-gold.jsonl").open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 173)
            self.assertEqual((first / "repair-ledger.jsonl").read_text(encoding="utf-8").count("\n"), 1)
            self.assertEqual((first / "raw-relation-provenance.jsonl").read_text(encoding="utf-8").count("\n"), 4329)


if __name__ == "__main__":
    unittest.main()
