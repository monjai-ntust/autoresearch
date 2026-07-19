"""Focused tests for publication candidate assembly and pilot freezing."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from config import load_pipeline_config
from constants import PROTOCOL_ID, TRAINING_SEEDS
from paths import RunLayout, discover_source_root
from phase_b_io import DataContractError, atomic_write_json, atomic_write_jsonl, load_json, sha256_file
from publication import assemble_seed_candidates, prepare_verifier_pilot
from records import EntitySpan, StrictTriple, candidate_id_for
from threshold import select_threshold


SOURCE_ROOT = discover_source_root(Path(__file__))
EXAMPLE_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "phase-b-publication-test"))
DOC_ID = "doc-development"
TRIPLE = StrictTriple(
    EXAMPLE_ID,
    EntitySpan(0, 0, "Object"),
    "selection",
    EntitySpan(2, 2, "Quality"),
)


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-b-publication-", dir=output_root)


def _candidate(seed: int) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1:development",
        "training_seed": seed,
        "example_id": EXAMPLE_ID,
        "source_document_id": DOC_ID,
        "candidate_id": candidate_id_for(seed, TRIPLE),
        "head": TRIPLE.head.to_mapping(),
        "relation": TRIPLE.relation,
        "tail": TRIPLE.tail.to_mapping(),
        "triple_confidence": 0.8,
        "input_hashes": {"prediction_ledger": "0" * 64},
    }


def _write_development_inputs(layout: RunLayout) -> None:
    atomic_write_jsonl(
        layout.resolve("data-prepared/development-gold.jsonl"),
        [
            {
                "protocol_id": PROTOCOL_ID,
                "split_id": "CODE-SPLIT-1:development",
                "example_id": EXAMPLE_ID,
                "source_document_id": DOC_ID,
                "gold_triples": [
                    {
                        "head": TRIPLE.head.to_mapping(),
                        "relation": TRIPLE.relation,
                        "tail": TRIPLE.tail.to_mapping(),
                    }
                ],
                "input_hashes": {"gold": "0" * 64},
            }
        ],
    )
    atomic_write_json(
        layout.resolve("data-prepared/split-manifest.json"),
        {"split_id": "CODE-SPLIT-1"},
    )
    for seed in TRAINING_SEEDS:
        atomic_write_jsonl(
            layout.resolve(f"predictions/dev/seed-{seed}-candidates.jsonl"),
            [_candidate(seed)],
        )


class PublicationAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")

    def test_assembly_threshold_and_pilot_bind_one_index(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "publication")
            layout.create()
            _write_development_inputs(layout)

            assembled = assemble_seed_candidates(
                layout, self.config, split="development"
            )
            self.assertEqual(assembled["candidate_count"], 8)
            index_path = layout.resolve("predictions/dev/candidate-index.json")
            index = load_json(index_path)
            self.assertEqual([item["training_seed"] for item in index["files"]], list(TRAINING_SEEDS))

            threshold = select_threshold(
                layout,
                self.config,
                candidates_path=layout.resolve(
                    "predictions/dev/development-candidates.jsonl", must_exist=True
                ),
                gold_path=layout.resolve(
                    "data-prepared/development-gold.jsonl", must_exist=True
                ),
                split_manifest_path=layout.resolve(
                    "data-prepared/split-manifest.json", must_exist=True
                ),
                out_path=layout.resolve("predictions/dev/threshold-selection.json"),
                candidate_index_path=index_path,
            )
            self.assertEqual(
                threshold["development_candidate_index_sha256"], sha256_file(index_path)
            )

            pilot = prepare_verifier_pilot(layout, self.config)
            self.assertEqual(pilot["candidate_count"], 8)
            selection = load_json(layout.resolve("predictions/dev/pilot-selection.json"))
            self.assertEqual(selection["training_seeds"], list(TRAINING_SEEDS))
            self.assertEqual(
                selection["development_candidate_index_sha256"], sha256_file(index_path)
            )

    def test_assembly_rejects_a_seed_file_with_the_wrong_seed(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "wrong-seed")
            layout.create()
            _write_development_inputs(layout)
            atomic_write_jsonl(
                layout.resolve("predictions/dev/seed-49-candidates.jsonl"),
                [_candidate(48)],
            )
            with self.assertRaises(DataContractError):
                assemble_seed_candidates(layout, self.config, split="development")


if __name__ == "__main__":
    unittest.main()
