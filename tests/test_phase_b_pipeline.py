"""Hand-calculated regression tests for the canonical Phase B core slice."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import uuid
from pathlib import Path

from phase_b_pipeline.config import load_pipeline_config
from phase_b_pipeline.constants import CONDITION_IDS, PROTOCOL_ID
from phase_b_pipeline.io import DataContractError, atomic_write_json, atomic_write_jsonl
from phase_b_pipeline.metrics import binary_metrics, triple_metrics
from phase_b_pipeline.paths import (
    PathContractError,
    RunLayout,
    discover_source_root,
    resolve_tracked_path,
)
from phase_b_pipeline.records import (
    Candidate,
    EntitySpan,
    StrictTriple,
    Verdict,
    candidate_id_for,
)
from phase_b_pipeline.scoring import ScoreInputs, score_run
from phase_b_pipeline.split import (
    SplitItem,
    build_official_code_split,
    iterative_multilabel_split,
    official_split_manifest,
    require_disjoint_partitions,
    split_manifest,
)
from phase_b_pipeline.statistics import (
    exact_wilcoxon_signed_rank,
    holm_adjust,
    paired_hierarchical_triple_f1_bootstrap,
    paired_t_test,
)


SOURCE_ROOT = discover_source_root(Path(__file__))
ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-b-", dir=output_root)


def _span(start: int, end: int, entity_type: str, text: str) -> dict:
    return {"start": start, "end": end, "type": entity_type, "text": text}


def _triple(
    head: dict, relation: str, tail: dict
) -> dict:
    return {"head": head, "relation": relation, "tail": tail}


def _candidate_record(
    seed: int,
    example_id: str,
    document_id: str,
    triple: dict,
    confidence: float,
) -> dict:
    strict = StrictTriple.from_mapping(triple, example_id=example_id, label="fixture")
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1:test",
        "training_seed": seed,
        "example_id": example_id,
        "source_document_id": document_id,
        "candidate_id": candidate_id_for(seed, strict),
        **triple,
        "triple_confidence": confidence,
        "input_hashes": {"checkpoint": ZERO_HASH},
    }


def _verdict_record(candidate: dict, condition: str, action: str, corrected=None) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "condition_id": condition,
        "training_seed": candidate["training_seed"],
        "candidate_id": candidate["candidate_id"],
        "response_status": "valid_response",
        "action": action,
        "corrected": corrected,
        "correction_validation_status": "valid" if corrected is not None else "not_applicable",
        "raw_response_sha256": ONE_HASH,
        "prompt_sha256": ONE_HASH,
        "model_manifest_sha256": ONE_HASH,
        "decoding_sha256": ONE_HASH,
        "attempts": 1,
        "error_category": None,
        "telemetry": {},
    }


class PathContractTests(unittest.TestCase):
    def test_run_layout_confines_every_resolved_path(self):
        with _temporary_output_directory() as temporary:
            root = Path(temporary).resolve()
            layout = RunLayout(root, "run-42")
            layout.create()
            self.assertEqual(
                layout.resolve("metrics/result.json"),
                root / "output" / "run-42" / "metrics" / "result.json",
            )
            with self.assertRaises(PathContractError):
                layout.resolve("../escape.json")
            with self.assertRaises(PathContractError):
                layout.resolve(root / "elsewhere.json")
            with self.assertRaises(PathContractError):
                layout.create()

    def test_run_id_rejects_path_syntax(self):
        with _temporary_output_directory() as temporary:
            root = Path(temporary)
            for run_id in ("../run", "a/b", "a\\b", "_starts-wrong", ""):
                with self.subTest(run_id=run_id), self.assertRaises(PathContractError):
                    RunLayout(root, run_id)

    def test_tracked_resources_reject_parent_relative_syntax(self):
        with self.assertRaises(PathContractError):
            resolve_tracked_path(SOURCE_ROOT, "configs/../pyproject.toml")


class ConfigContractTests(unittest.TestCase):
    def test_approved_config_and_matrix_resolve_from_source_only(self):
        config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")
        self.assertEqual(config.value["training_seeds"], list(range(42, 50)))
        self.assertEqual(
            [row["condition_id"] for row in config.matrix["conditions"]],
            list(CONDITION_IDS),
        )
        self.assertNotIn("TBD_", json.dumps(config.value, sort_keys=True))

    def test_every_tracked_json_contract_parses(self):
        for relative in (
            "configs/phase_b_path_a.json",
            "configs/phase_b_experiment_matrix.json",
            "schemas/phase_b/config.schema.json",
            "schemas/phase_b/experiment-matrix.schema.json",
            "schemas/phase_b/records.schema.json",
            "schemas/phase_b/threshold-selection.schema.json",
            "schemas/phase_b/outcomes.schema.json",
            "schemas/phase_b/metrics.schema.json",
            "schemas/phase_b/checkout-manifest.schema.json",
            "schemas/phase_b/score-manifest.schema.json",
        ):
            with self.subTest(path=relative):
                with (SOURCE_ROOT / relative).open(encoding="utf-8") as handle:
                    self.assertIsInstance(json.load(handle), dict)


class StrictMatcherTests(unittest.TestCase):
    def test_key_is_typed_directed_and_ignores_display_text(self):
        first = StrictTriple(
            "example",
            EntitySpan(0, 0, "Object", "Door"),
            "necessity",
            EntitySpan(2, 3, "Quality", "fire resistant"),
        )
        same_key_new_text = StrictTriple(
            "example",
            EntitySpan(0, 0, "Object", "door"),
            "necessity",
            EntitySpan(2, 3, "Quality", "Fire-resistant"),
        )
        reversed_key = StrictTriple(
            "example", first.tail, "necessity", first.head
        )
        wrong_type = StrictTriple(
            "example",
            EntitySpan(0, 0, "Property", "Door"),
            "necessity",
            first.tail,
        )
        self.assertEqual(first, same_key_new_text)
        self.assertNotEqual(first, reversed_key)
        self.assertNotEqual(first, wrong_type)

    def test_candidate_identity_rejects_tampering(self):
        record = _candidate_record(
            42,
            "example",
            "document",
            _triple(
                _span(0, 0, "Object", "door"),
                "necessity",
                _span(1, 1, "Quality", "rated"),
            ),
            0.8,
        )
        Candidate.from_mapping(record, "candidate")
        record["relation"] = "selection"
        with self.assertRaises(DataContractError):
            Candidate.from_mapping(record, "candidate")

    def test_invalid_correction_is_retained_as_a_failed_non_emission(self):
        record = _candidate_record(
            42,
            "example",
            "document",
            _triple(
                _span(0, 0, "Object", "door"),
                "necessity",
                _span(1, 1, "Quality", "rated"),
            ),
            0.8,
        )
        candidate = Candidate.from_mapping(record, "candidate")
        verdict = _verdict_record(candidate=record, condition="VER-CORRECTIVE", action="CORRECT")
        verdict["correction_validation_status"] = "ambiguous"
        parsed = Verdict.from_mapping(verdict, "verdict", candidate)
        self.assertEqual(parsed.action, "CORRECT")
        self.assertIsNone(parsed.corrected)
        self.assertEqual(parsed.correction_validation_status, "ambiguous")


class SplitContractTests(unittest.TestCase):
    def test_split_is_exact_disjoint_and_input_order_independent(self):
        items = [
            SplitItem(
                f"id-{index:02d}",
                frozenset(
                    {
                        "country:UK" if index % 2 else "country:US",
                        "entity:Object" if index < 10 else "entity:Value",
                        *( ["relation:less"] if index in {2, 7, 13} else [] ),
                    }
                ),
            )
            for index in range(16)
        ]
        first = iterative_multilabel_split(items, development_size=4, seed=42)
        second = iterative_multilabel_split(reversed(items), development_size=4, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first.train_ids), 12)
        self.assertEqual(len(first.development_ids), 4)
        self.assertFalse(set(first.train_ids) & set(first.development_ids))
        self.assertEqual(first.label_counts["relation:less"]["development"], 1)
        self.assertEqual(split_manifest(first, seed=42), split_manifest(second, seed=42))

    def test_split_seed_is_reproducible_and_scientifically_separate(self):
        items = [
            SplitItem(f"id-{index:02d}", frozenset({"country:US", f"bucket:{index % 3}"}))
            for index in range(20)
        ]
        seed_42_a = iterative_multilabel_split(items, development_size=5, seed=42)
        seed_42_b = iterative_multilabel_split(items, development_size=5, seed=42)
        seed_43 = iterative_multilabel_split(items, development_size=5, seed=43)
        self.assertEqual(seed_42_a, seed_42_b)
        self.assertNotEqual(seed_42_a.development_ids, seed_43.development_ids)

    def test_partition_overlap_is_a_hard_leakage_failure(self):
        require_disjoint_partitions(train=["a", "b"], development=["c"], test=["d"])
        with self.assertRaises(DataContractError):
            require_disjoint_partitions(train=["a", "b"], development=["b"], test=["d"])

    def test_official_wrapper_freezes_586_103_173_and_seed_42(self):
        train_items = [
            SplitItem(
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"train-{index}")),
                frozenset({"country:UK" if index % 2 else "country:US", f"entity:{index % 4}"}),
            )
            for index in range(689)
        ]
        test_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"test-{index}")) for index in range(173)
        ]
        result = build_official_code_split(train_items, test_ids)
        manifest = official_split_manifest(result)
        self.assertEqual(
            (manifest["train_count"], manifest["development_count"], manifest["test_count"]),
            (586, 103, 173),
        )
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["overlap_count"], 0)


class MetricContractTests(unittest.TestCase):
    def test_hand_calculated_binary_metrics(self):
        metrics = binary_metrics(tp=2, fp=1, fn=2, tn=5)
        self.assertEqual(metrics["counts"]["total"], 10)
        self.assertTrue(math.isclose(metrics["accuracy"]["value"], 0.7))
        self.assertTrue(math.isclose(metrics["precision"]["value"], 2 / 3))
        self.assertTrue(math.isclose(metrics["recall"]["value"], 1 / 2))
        self.assertTrue(math.isclose(metrics["f1"]["value"], 4 / 7))

    def test_zero_denominators_are_null_and_explicit(self):
        binary = binary_metrics(tp=0, fp=0, fn=0, tn=0)
        triples = triple_metrics(tp=0, fp=0, fn=0)
        for metric in ("accuracy", "precision", "recall", "f1"):
            self.assertIsNone(binary[metric]["value"])
            self.assertEqual(binary[metric]["status"], "undefined_zero_denominator")
        for metric in ("precision", "recall", "f1"):
            self.assertIsNone(triples[metric]["value"])


class StatisticalContractTests(unittest.TestCase):
    def test_exact_wilcoxon_and_holm_match_hand_enumeration(self):
        wilcoxon = exact_wilcoxon_signed_rank([1.0, 2.0, 3.0])
        self.assertEqual(wilcoxon["statistic"], 0.0)
        self.assertEqual(wilcoxon["p_value"], 0.25)
        self.assertEqual(exact_wilcoxon_signed_rank([0.0] * 8)["p_value"], 1.0)
        self.assertEqual(
            holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}),
            {"a": 0.03, "b": 0.06, "c": 0.06},
        )

    def test_paired_t_handles_zero_and_constant_nonzero_variance(self):
        self.assertEqual(paired_t_test([1.0, -1.0])["p_value"], 1.0)
        self.assertTrue(math.isclose(paired_t_test([0.0, 2.0])["p_value"], 0.5))
        constant = paired_t_test([1.0, 1.0, 1.0])
        self.assertIsNone(constant["statistic"])
        self.assertEqual(constant["statistic_status"], "positive_infinity")
        self.assertEqual(constant["p_value"], 0.0)

    def test_hierarchical_bootstrap_is_paired_and_seed_reproducible(self):
        counts = {
            "VER-RAW": {
                42: {"doc-a": {"tp": 1, "fp": 1, "fn": 0}, "doc-b": {"tp": 0, "fp": 0, "fn": 1}},
                43: {"doc-a": {"tp": 1, "fp": 0, "fn": 0}, "doc-b": {"tp": 0, "fp": 1, "fn": 1}},
            },
            "VER-SIMPLE": {
                42: {"doc-a": {"tp": 1, "fp": 0, "fn": 0}, "doc-b": {"tp": 1, "fp": 0, "fn": 0}},
                43: {"doc-a": {"tp": 1, "fp": 0, "fn": 0}, "doc-b": {"tp": 1, "fp": 0, "fn": 0}},
            },
        }
        first = paired_hierarchical_triple_f1_bootstrap(
            counts, reference_condition="VER-RAW", replicates=200, seed=20260717
        )
        second = paired_hierarchical_triple_f1_bootstrap(
            counts, reference_condition="VER-RAW", replicates=200, seed=20260717
        )
        self.assertEqual(first, second)
        self.assertEqual(first["conditions"]["VER-SIMPLE"]["point_estimate"], 1.0)
        self.assertGreater(
            first["deltas_vs_reference"]["VER-SIMPLE"]["point_estimate"], 0
        )


class OfflineScoringTests(unittest.TestCase):
    def test_four_conditions_match_manual_counts_and_collapse_correction_duplicate(self):
        config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "fixture-run")
            layout.create()
            gold_a = _triple(
                _span(0, 0, "Object", "door"),
                "necessity",
                _span(2, 2, "Quality", "rated"),
            )
            invalid_a = _triple(
                _span(0, 0, "Object", "door"),
                "selection",
                _span(2, 2, "Quality", "rated"),
            )
            gold_b = _triple(
                _span(1, 1, "Property", "height"),
                "greater",
                _span(3, 3, "Value", "2m"),
            )
            gold_records = [
                {
                    "protocol_id": PROTOCOL_ID,
                    "split_id": "CODE-SPLIT-1:test",
                    "example_id": "ex-a",
                    "source_document_id": "doc-a",
                    "gold_triples": [gold_a],
                    "input_hashes": {"dataset": ZERO_HASH},
                },
                {
                    "protocol_id": PROTOCOL_ID,
                    "split_id": "CODE-SPLIT-1:test",
                    "example_id": "ex-b",
                    "source_document_id": "doc-b",
                    "gold_triples": [gold_b],
                    "input_hashes": {"dataset": ZERO_HASH},
                },
            ]
            candidates = [
                _candidate_record(42, "ex-a", "doc-a", gold_a, 0.9),
                _candidate_record(42, "ex-a", "doc-a", invalid_a, 0.4),
                _candidate_record(42, "ex-b", "doc-b", gold_b, 0.2),
            ]
            candidates.sort(key=lambda item: (item["training_seed"], item["example_id"], item["candidate_id"]))
            by_relation = {item["relation"]: item for item in candidates}
            simple = [
                _verdict_record(by_relation["necessity"], "VER-SIMPLE", "DISCARD"),
                _verdict_record(by_relation["selection"], "VER-SIMPLE", "KEEP"),
                {
                    **_verdict_record(by_relation["greater"], "VER-SIMPLE", "DISCARD"),
                    "response_status": "malformed",
                    "action": None,
                    "error_category": "malformed_json",
                },
            ]
            simple.sort(key=lambda item: (item["training_seed"], item["candidate_id"]))
            corrective = [
                _verdict_record(by_relation["necessity"], "VER-CORRECTIVE", "KEEP"),
                _verdict_record(
                    by_relation["selection"],
                    "VER-CORRECTIVE",
                    "CORRECT",
                    corrected={**gold_a, "head": {**gold_a["head"], "text": "Door"}},
                ),
            ]
            corrective.sort(key=lambda item: (item["training_seed"], item["candidate_id"]))

            paths = {
                "gold": layout.resolve("data-prepared/test-gold.jsonl"),
                "candidates": layout.resolve("predictions/test/candidates.jsonl"),
                "simple": layout.resolve("verifier/simple/verdicts.jsonl"),
                "corrective": layout.resolve("verifier/corrective/verdicts.jsonl"),
                "threshold": layout.resolve("predictions/dev/threshold-selection.json"),
            }
            atomic_write_jsonl(paths["gold"], gold_records)
            atomic_write_jsonl(paths["candidates"], candidates)
            atomic_write_jsonl(paths["simple"], simple)
            atomic_write_jsonl(paths["corrective"], corrective)
            atomic_write_json(
                paths["threshold"],
                {
                    "protocol_id": PROTOCOL_ID,
                    "selection_split": "development",
                    "objective": "mean_per_seed_development_strict_triple_f1",
                    "tie_rule": "higher_threshold",
                    "grid": config.value["threshold_selection"]["grid"],
                    "selected_threshold": 0.5,
                    "used_test_labels": False,
                    "development_candidate_index_sha256": ZERO_HASH,
                    "development_gold_sha256": ZERO_HASH,
                    "split_manifest_sha256": ZERO_HASH,
                    "per_threshold": [
                        {
                            "threshold": threshold,
                            "per_seed_development_strict_triple_f1": {
                                str(seed): (1.0 if threshold == 0.5 else 0.0)
                                for seed in range(42, 50)
                            },
                            "mean_per_seed_development_strict_triple_f1": (
                                1.0 if threshold == 0.5 else 0.0
                            ),
                        }
                        for threshold in config.value["threshold_selection"]["grid"]
                    ],
                },
            )
            metrics = score_run(
                layout,
                config,
                ScoreInputs(
                    gold=paths["gold"],
                    candidates=paths["candidates"],
                    simple_verdicts=paths["simple"],
                    corrective_verdicts=paths["corrective"],
                    threshold_selection=paths["threshold"],
                ),
            )
            self.assertEqual(
                metrics["conditions"]["VER-RAW"]["candidate_decision"]["counts"],
                {"tp": 2, "fp": 1, "fn": 0, "tn": 0, "total": 3},
            )
            self.assertEqual(
                metrics["conditions"]["VER-RAW"]["end_to_end_strict_triple"]["counts"],
                {"tp": 2, "fp": 1, "fn": 0},
            )
            self.assertEqual(
                metrics["conditions"]["VER-CONFIDENCE"]["candidate_decision"]["counts"],
                {"tp": 1, "fp": 0, "fn": 1, "tn": 1, "total": 3},
            )
            self.assertEqual(
                metrics["conditions"]["VER-SIMPLE"]["candidate_decision"]["counts"],
                {"tp": 0, "fp": 1, "fn": 2, "tn": 0, "total": 3},
            )
            self.assertEqual(
                metrics["conditions"]["VER-SIMPLE"]["diagnostics"]["errors"],
                {"malformed_json": 1},
            )
            corrective_metrics = metrics["conditions"]["VER-CORRECTIVE"]
            self.assertEqual(
                corrective_metrics["candidate_decision"]["counts"],
                {"tp": 1, "fp": 0, "fn": 1, "tn": 1, "total": 3},
            )
            self.assertEqual(
                corrective_metrics["end_to_end_strict_triple"]["counts"],
                {"tp": 1, "fp": 0, "fn": 1},
            )
            self.assertEqual(corrective_metrics["duplicate_emissions_collapsed"], 1)
            self.assertEqual(
                corrective_metrics["diagnostics"]["original_invalid_to_final_valid"], 1
            )
            self.assertEqual(corrective_metrics["diagnostics"]["corrections_gold_valid"], 1)
            self.assertEqual(
                corrective_metrics["diagnostics"]["errors"], {"missing_verdict": 1}
            )
            self.assertFalse(metrics["publication_seed_coverage_complete"])
            self.assertFalse(metrics["publication_data_coverage_complete"])
            self.assertTrue(layout.resolve("manifests/score-manifest.json").is_file())
            created_files = [path for path in Path(temporary).rglob("*") if path.is_file()]
            self.assertTrue(created_files)
            for path in created_files:
                with self.subTest(created_path=path):
                    self.assertTrue(path.resolve().is_relative_to(layout.run_root.resolve()))
            with self.assertRaises(DataContractError):
                score_run(
                    layout,
                    config,
                    ScoreInputs(
                        gold=paths["gold"],
                        candidates=paths["candidates"],
                        simple_verdicts=paths["simple"],
                        corrective_verdicts=paths["corrective"],
                        threshold_selection=paths["threshold"],
                    ),
                )


if __name__ == "__main__":
    unittest.main()
