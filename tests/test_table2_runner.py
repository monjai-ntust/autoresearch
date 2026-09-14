"""Focused same-run and portability tests for downstream Table 2."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pipeline
import table2_runner as runner


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "table2_contract.json").read_text(encoding="utf-8"))
RUN_ID = "research-run-001"


def verifier_environment(model_name: str = "example-model:tag") -> dict:
    return {
        "condition_id": "VER-CORRECTIVE",
        "execution_mode": "live",
        "protocol_id": "B04-PATH-A-1.3",
        "model": {
            "identity_verified": True,
            "name": model_name,
            "tag_digest": "a" * 64,
            "registry_manifest_sha256": "a" * 64,
            "blob_sha256": "b" * 64,
            "model_blob_sha256": "b" * 64,
            "details": {
                "family": "qwen3",
                "parameter_size": "32.8B",
                "quantization_level": "Q4_K_M",
            },
        },
    }


def execution_summary() -> dict:
    return {
        "run_id": RUN_ID,
        "lineage_status": "lineage-valid",
        "parent_artifact_set_sha256": "1" * 64,
        "score_manifest_sha256": "2" * 64,
        "seed": 42,
        "threshold": 0.25,
        "records": {"question_contract": []},
        "projection_sha256": {"records": "3" * 64},
        "graphs": {
            name: {"graph_id": character * 64, "source": "existing"}
            for name, character in zip(("confidence", "corrective", "gold"), "678")
        },
        "model_identity": {
            "name": "example-model:tag",
            "tag_digest": "4" * 64,
            "blob_sha256": "5" * 64,
            "details": {
                "family": "qwen3",
                "parameter_size": "32.8B",
                "quantization_level": "Q4_K_M",
            },
            "verifier_environment_sha256": "6" * 64,
            "verifier_condition_id": "VER-CORRECTIVE",
            "verifier_protocol_id": "B04-PATH-A-1.3",
        },
        "observed_profile": {
            "encoder": {
                "base_model": "microsoft/deberta-large",
                "base_model_revision": "7" * 40,
            }
        },
        "historical_reference_match": {"overall": False, "admission_effect": "none"},
    }


class Table2RunnerTests(unittest.TestCase):
    def test_source_contract_is_valid(self):
        result = runner.validate_source_contract(CONTRACT, require_clean=False)
        self.assertEqual(result["branch"], "publication-refactored-rag")
        self.assertEqual(result["baseline"], CONTRACT["source_baseline"]["commit"])

    def test_table2_cli_exposes_only_run_namespace_model_endpoint_and_dry_run(self):
        parser = pipeline._parser()
        args = parser.parse_args(["table2", "--run-id", RUN_ID, "--dry-run"])
        self.assertEqual(
            set(vars(args)),
            {"source_root", "stage", "run_id", "ollama_url", "dry_run"},
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                ["table2", "--run-id", RUN_ID, "--kg", "foreign.json"]
            )

    def test_arbitrary_valid_run_id_is_accepted_but_path_syntax_is_rejected(self):
        self.assertEqual(
            runner.validate_run_id("another.valid_run-42", require_existing=False).name,
            "another.valid_run-42",
        )
        for value in ("../escape", "nested/run", ".", "-leading"):
            with self.subTest(value=value), self.assertRaises(runner.PhaseEError):
                runner.validate_run_id(value, require_existing=False)

    def test_physical_file_from_another_run_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selected = root / "selected-run"
            foreign = root / "foreign-run"
            selected.mkdir()
            foreign.mkdir()
            foreign_file = foreign / "verdicts.jsonl"
            foreign_file.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.PhaseEError, "unsafe"):
                runner.run_file(selected, str(foreign_file))

    def test_encoder_revision_may_differ_from_reference_but_not_within_a_run(self):
        identity = {
            "base_model": "microsoft/deberta-large",
            "base_model_revision": "f" * 40,
        }
        self.assertEqual(
            runner.require_consistent_encoder_identity([identity, dict(identity)]),
            identity,
        )
        other = {**identity, "base_model_revision": "e" * 40}
        with self.assertRaisesRegex(runner.PhaseEError, "differs across seeds"):
            runner.require_consistent_encoder_identity([identity, other])

    def test_historical_profile_mismatch_is_reported_without_an_admission_gate(self):
        comparison = runner._compare_profile(
            {"base_model_revision": "f" * 40},
            {"base_model_revision": "e" * 40},
        )
        self.assertFalse(comparison["all_fields_match"])
        self.assertFalse(comparison["fields"]["base_model_revision"]["match"])

    def test_qwen_identity_is_derived_from_a_same_run_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "environment.json"
            path.write_text(json.dumps(verifier_environment()), encoding="utf-8")
            identity = runner.load_verifier_model_identity(path, runner.sha256_file(path))
        self.assertEqual(identity["tag_digest"], "a" * 64)
        self.assertEqual(identity["blob_sha256"], "b" * 64)

    def test_live_qwen_must_match_the_selected_run_not_the_reference_profile(self):
        expected = {
            "name": "example-model:tag",
            "tag_digest": "a" * 64,
            "blob_sha256": "b" * 64,
            "details": {
                "family": "qwen3",
                "parameter_size": "32.8B",
                "quantization_level": "Q4_K_M",
            },
            "verifier_environment_sha256": "c" * 64,
            "verifier_condition_id": "VER-CORRECTIVE",
            "verifier_protocol_id": "B04-PATH-A-1.3",
        }
        tags = {"models": [{"name": expected["name"], "digest": expected["tag_digest"]}]}
        show = {
            "details": expected["details"],
            "modelfile": f"FROM /models/blobs/sha256-{expected['blob_sha256']}\n",
        }
        responses = [
            (tags, runner.canonical_json_bytes(tags), "d" * 64),
            (show, runner.canonical_json_bytes(show), "e" * 64),
        ]
        with mock.patch.object(runner, "fetch_ollama_json", side_effect=responses):
            evidence, _tags, _show = runner.fetch_matching_model(
                "http://localhost:11434", expected
            )
        self.assertTrue(evidence["identity_verified"])
        self.assertEqual(evidence["blob_sha256"], "b" * 64)

    def test_projection_preserves_prepared_order_and_private_gold(self):
        identifiers = [f"example-{index:03d}" for index in range(173)]
        prepared = [
            {"example_id": item, "content": f"Head {index} is in Tail {index}."}
            for index, item in enumerate(identifiers)
        ]
        gold = [
            {
                "example_id": item,
                "gold_triples": [{
                    "head": {"text": f"Head {index}"},
                    "relation": "part-of",
                    "tail": {"text": f"Tail {index}"},
                }],
            }
            for index, item in enumerate(identifiers)
        ]
        rows, content, summary = runner.project_evaluator_records(
            prepared, gold, {"test_ids": identifiers}, CONTRACT
        )
        self.assertEqual([row["doc_id"] for row in rows], identifiers)
        self.assertEqual(content, runner.canonical_jsonl_bytes(rows))
        self.assertEqual(summary["questions"], 10)

    def test_dry_run_does_not_contact_model_or_create_child(self):
        args = argparse.Namespace(run_id=RUN_ID, ollama_url="http://localhost:11434", dry_run=True)
        summary = execution_summary()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with mock.patch.object(runner, "validate_source_contract", return_value={}), \
                 mock.patch.object(
                     runner,
                     "preflight_same_run",
                     return_value=(run_dir, summary, {}, {}),
                 ), \
                 mock.patch.object(runner, "fetch_matching_model") as model, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_command(args, CONTRACT), 0)
            model.assert_not_called()
            self.assertFalse((run_dir / "table2").exists())

    def test_rag_output_rejects_failed_model_calls_and_identity_substitution(self):
        modes = CONTRACT["evaluator"]["modes"]
        count = CONTRACT["evaluator"]["max_questions"]
        questions = [{"q": f"question-{index}", "gold": "gold"} for index in range(count)]
        expected_kg = "output/example/table2/projections/confidence.json"
        output = {
            "metadata": {
                "kg": expected_kg,
                "n_questions": count,
                "model": "example-model:tag",
                "time_seconds": 1.0,
            },
            "accuracy": {mode: 0.0 for mode in modes},
            "correct_counts": {mode: 0 for mode in modes},
            "results": {
                mode: [
                    {"q": item["q"], "pred": "answer", "gold": item["gold"], "correct": False}
                    for item in questions
                ]
                for mode in modes
            },
        }
        output["results"]["hybrid"][0]["pred"] = "ERROR: timeout"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(output), encoding="utf-8")
            with self.assertRaisesRegex(runner.PhaseEError, "failed or changed"):
                runner.validate_rag_output(
                    path,
                    CONTRACT,
                    "example-model:tag",
                    expected_questions=questions,
                    expected_kg=expected_kg,
                )

    def test_interrupted_unverified_output_is_quarantined_inside_the_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            child = Path(temporary).resolve() / "table2"
            output = child / "rag-results" / "confidence.json"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"old-unverified")
            status_path = child / "run-status.json"
            status = {"stages": {"rag_confidence": {"status": "running", "attempts": []}}}
            status_path.write_text(json.dumps(status), encoding="utf-8")

            result = runner.run_stage(
                "rag_confidence",
                lambda: {"new": "validated"},
                output,
                lambda path: {"sha256": runner.sha256_file(path)},
                child,
                status,
                status_path,
            )
            recovered = child / "recovery/rag_confidence/unverified-001.json"
            self.assertEqual(recovered.read_bytes(), b"old-unverified")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"new": "validated"})
            self.assertEqual(result["sha256"], runner.sha256_file(output))

    def test_runner_has_no_upstream_execution_or_arbitrary_copy_route(self):
        source = (ROOT / "table2_runner.py").read_text(encoding="utf-8")
        self.assertNotIn("build_kg.py", source)
        self.assertNotIn("copy_input", source)
        self.assertNotIn("--inference", source)
        self.assertNotIn("plan_training", source)
        self.assertNotIn("run_verifier(", source)


if __name__ == "__main__":
    unittest.main()
