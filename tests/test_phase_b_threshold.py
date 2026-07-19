"""Synthetic regression tests for the development threshold-selection stage."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from config import load_pipeline_config
from constants import PROTOCOL_ID, TRAINING_SEEDS
from phase_b_io import DataContractError, atomic_write_json, atomic_write_jsonl, sha256_file
from paths import RunLayout, discover_source_root
from records import EntitySpan, StrictTriple, candidate_id_for
from scoring import _load_threshold
from threshold import select_threshold


SOURCE_ROOT = discover_source_root(Path(__file__))
EXAMPLE_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "phase-b-threshold-test"))
DOC_ID = "doc-dev"

T_GOLD = StrictTriple(
    EXAMPLE_ID,
    EntitySpan(0, 0, "Object"),
    "selection",
    EntitySpan(3, 4, "Quality"),
)
U_FALSE = StrictTriple(
    EXAMPLE_ID,
    EntitySpan(0, 0, "Object"),
    "necessity",
    EntitySpan(6, 7, "Value"),
)


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-b-threshold-", dir=output_root)


def _gold_record() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1:development",
        "example_id": EXAMPLE_ID,
        "source_document_id": DOC_ID,
        "gold_triples": [
            {
                "head": T_GOLD.head.to_mapping(),
                "relation": T_GOLD.relation,
                "tail": T_GOLD.tail.to_mapping(),
            }
        ],
        "input_hashes": {"gold": "0" * 64},
    }


def _candidate(seed: int, triple: StrictTriple, confidence: float) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1:development",
        "training_seed": seed,
        "example_id": EXAMPLE_ID,
        "source_document_id": DOC_ID,
        "candidate_id": candidate_id_for(seed, triple),
        "head": triple.head.to_mapping(),
        "relation": triple.relation,
        "tail": triple.tail.to_mapping(),
        "triple_confidence": confidence,
        "input_hashes": {"prediction_ledger": "0" * 64},
    }


def _write_inputs(layout: RunLayout, candidate_rows: list[dict]):
    candidates = layout.resolve("predictions/dev/candidate-index.jsonl")
    gold = layout.resolve("data-prepared/development-gold.jsonl")
    split_manifest = layout.resolve("data-prepared/split-manifest.json")
    # Candidate files must be sorted by (training_seed, example_id, candidate_id).
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (row["training_seed"], row["example_id"], row["candidate_id"]),
    )
    atomic_write_jsonl(candidates, candidate_rows)
    atomic_write_jsonl(gold, [_gold_record()])
    atomic_write_json(split_manifest, {"split_id": "CODE-SPLIT-1"})
    return candidates, gold, split_manifest


class ThresholdSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")

    def test_selects_highest_threshold_at_max_mean_f1(self):
        # Every seed emits exactly the gold triple at confidence 0.8: F1 is 1.0
        # for every threshold <= 0.80 and 0.0 above it, so the higher-threshold
        # tie rule selects 0.80.
        rows = [_candidate(seed, T_GOLD, 0.8) for seed in TRAINING_SEEDS]
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "thresh")
            layout.create()
            candidates, gold, split_manifest = _write_inputs(layout, rows)
            out = layout.resolve("predictions/dev/threshold-selection.json")
            document = select_threshold(
                layout,
                self.config,
                candidates_path=candidates,
                gold_path=gold,
                split_manifest_path=split_manifest,
                out_path=out,
            )
            self.assertEqual(document["selected_threshold"], 0.8)
            self.assertEqual(len(document["per_threshold"]), 20)
            row_080 = document["per_threshold"][16]
            self.assertEqual(row_080["threshold"], 0.8)
            self.assertEqual(row_080["mean_per_seed_development_strict_triple_f1"], 1.0)
            row_085 = document["per_threshold"][17]
            self.assertEqual(row_085["threshold"], 0.85)
            self.assertEqual(row_085["mean_per_seed_development_strict_triple_f1"], 0.0)
            # The emitted file must be consumable by the scorer.
            self.assertEqual(_load_threshold(out, self.config), 0.8)

    def test_false_positive_curve_selects_expected_threshold(self):
        # Each seed emits gold T at 0.5 and a false positive U at 0.9.
        # tau <= 0.5: keep {T, U} -> tp=1, fp=1 -> F1 = 2/3.
        # 0.5 < tau <= 0.9: keep {U} -> F1 = 0. So the best mean 2/3 peaks at 0.50.
        rows = []
        for seed in TRAINING_SEEDS:
            rows.append(_candidate(seed, T_GOLD, 0.5))
            rows.append(_candidate(seed, U_FALSE, 0.9))
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "thresh-fp")
            layout.create()
            candidates, gold, split_manifest = _write_inputs(layout, rows)
            out = layout.resolve("predictions/dev/threshold-selection.json")
            document = select_threshold(
                layout,
                self.config,
                candidates_path=candidates,
                gold_path=gold,
                split_manifest_path=split_manifest,
                out_path=out,
            )
            self.assertEqual(document["selected_threshold"], 0.5)
            row_050 = document["per_threshold"][10]
            self.assertEqual(row_050["threshold"], 0.5)
            self.assertAlmostEqual(
                row_050["mean_per_seed_development_strict_triple_f1"], 2 / 3
            )
            self.assertEqual(_load_threshold(out, self.config), 0.5)

    def test_requires_all_eight_seeds(self):
        rows = [_candidate(seed, T_GOLD, 0.8) for seed in TRAINING_SEEDS[:-1]]
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "thresh-missing")
            layout.create()
            candidates, gold, split_manifest = _write_inputs(layout, rows)
            out = layout.resolve("predictions/dev/threshold-selection.json")
            with self.assertRaises(DataContractError):
                select_threshold(
                    layout,
                    self.config,
                    candidates_path=candidates,
                    gold_path=gold,
                    split_manifest_path=split_manifest,
                    out_path=out,
                )

    def test_refuses_to_overwrite(self):
        rows = [_candidate(seed, T_GOLD, 0.8) for seed in TRAINING_SEEDS]
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "thresh-overwrite")
            layout.create()
            candidates, gold, split_manifest = _write_inputs(layout, rows)
            out = layout.resolve("predictions/dev/threshold-selection.json")
            select_threshold(
                layout,
                self.config,
                candidates_path=candidates,
                gold_path=gold,
                split_manifest_path=split_manifest,
                out_path=out,
            )
            with self.assertRaises(DataContractError):
                select_threshold(
                    layout,
                    self.config,
                    candidates_path=candidates,
                    gold_path=gold,
                    split_manifest_path=split_manifest,
                    out_path=out,
                )

    def test_binds_explicit_candidate_index_when_supplied(self):
        rows = [_candidate(seed, T_GOLD, 0.8) for seed in TRAINING_SEEDS]
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "thresh-index")
            layout.create()
            candidates, gold, split_manifest = _write_inputs(layout, rows)
            candidate_index = layout.resolve("predictions/dev/candidate-index.json")
            atomic_write_json(candidate_index, {"identity": "full-eight-seed-index"})
            out = layout.resolve("predictions/dev/threshold-selection.json")
            document = select_threshold(
                layout,
                self.config,
                candidates_path=candidates,
                gold_path=gold,
                split_manifest_path=split_manifest,
                out_path=out,
                candidate_index_path=candidate_index,
            )
            self.assertEqual(
                document["development_candidate_index_sha256"],
                sha256_file(candidate_index),
            )


if __name__ == "__main__":
    unittest.main()
