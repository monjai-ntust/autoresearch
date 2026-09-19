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
TABLE2_CHILD = CONTRACT["evaluator"]["output_namespace"]
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


def live_model_capture(summary: dict) -> tuple[dict, bytes, bytes]:
    model = summary["model_identity"]
    tags = {"models": [{"name": model["name"], "digest": model["tag_digest"]}]}
    show = {
        "details": model["details"],
        "modelfile": f"FROM /models/blobs/sha256-{model['blob_sha256']}\n",
    }
    tags_bytes = runner.canonical_json_bytes(tags)
    show_bytes = runner.canonical_json_bytes(show)
    evidence = {
        **model,
        "identity_verified": True,
        "tags_response_sha256": runner.sha256_bytes(tags_bytes),
        "show_response_sha256": runner.sha256_bytes(show_bytes),
    }
    return evidence, tags_bytes, show_bytes


def rag_result(summary: dict, kg_identity: str) -> dict:
    modes = runner.evaluator.MODES
    questions = summary["records"]["question_contract"]
    results = {
        mode: [
            {"q": row["q"], "pred": row["gold"], "gold": row["gold"], "correct": True}
            for row in questions
        ]
        for mode in modes
    }
    return {
        "metadata": {
            "kg": kg_identity,
            "n_questions": len(questions),
            "model": summary["model_identity"]["name"],
            "time_seconds": 0.0,
        },
        "accuracy": {mode: 1.0 for mode in modes},
        "correct_counts": {mode: len(questions) for mode in modes},
        "results": results,
    }


def live_preflight(run_dir: Path) -> tuple[dict, dict[str, bytes], dict]:
    summary = execution_summary()
    question_count = CONTRACT["evaluator"]["max_questions"]
    summary["records"] = {
        "records": 1,
        "questions": question_count,
        "question_contract": [
            {"q": f"question-{index}", "gold": f"answer-{index}"}
            for index in range(question_count)
        ],
        "question_sha256": "9" * 64,
    }
    projections = {
        "records": b"{}\n",
        "confidence": b'{"kind":"confidence"}\n',
        "corrective": b'{"kind":"corrective"}\n',
        "gold": b'{"kind":"gold"}\n',
    }
    summary["projection_sha256"] = {
        name: runner.sha256_bytes(value) for name, value in projections.items()
    }
    graphs = {}
    for name in ("confidence", "corrective", "gold"):
        graph = {"condition": name, "graph_id": summary["graphs"][name]["graph_id"]}
        graphs[name] = graph
        summary["graphs"][name].update(
            source="constructed",
            source_sha256=runner.sha256_bytes(runner.canonical_json_bytes(graph)),
        )
    summary["run_dir"] = str(run_dir)
    return summary, projections, graphs


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

    def test_authenticated_older_run_config_uses_explicit_legacy_compatibility(self):
        commit = "3e4e5d9584cf2208eba7830550fc3b01cafa456f"
        path = "configs/phase_b_path_a.json"
        digest = runner.sha256_bytes(runner._git_file_bytes(commit, path))
        checkout = {
            "source": {"commit": commit},
            "tracked_artifact_sha256": {path: digest},
        }
        full = {"source_commit": commit, "config": path, "config_sha256": digest}
        config, legacy = runner._load_authenticated_run_config(
            checkout, full, CONTRACT
        )
        self.assertTrue(legacy)
        self.assertEqual(config["protocol_id"], CONTRACT["pipeline"]["protocol_id"])

    def test_current_checkout_config_requires_new_run_seals(self):
        commit = runner.run_git("rev-parse", "HEAD").stdout.strip()
        path = "configs/pipeline.json"
        digest = runner.sha256_bytes(runner._git_file_bytes(commit, path))
        checkout = {
            "source": {
                "commit": commit,
                "config": {"path": path, "sha256": digest},
            },
        }
        full = {"source_commit": commit, "config": path, "config_sha256": digest}
        config, legacy = runner._load_authenticated_run_config(
            checkout, full, CONTRACT
        )
        self.assertFalse(legacy)
        self.assertEqual(config["protocol_id"], CONTRACT["pipeline"]["protocol_id"])

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
                    "relation": "part-of" if index < 105 else "necessity",
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
        self.assertEqual(summary["questions"], CONTRACT["evaluator"]["max_questions"])

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
            self.assertFalse((run_dir / TABLE2_CHILD).exists())

    def test_rag_output_rejects_failed_model_calls_and_identity_substitution(self):
        modes = runner.evaluator.MODES
        count = CONTRACT["evaluator"]["max_questions"]
        questions = [{"q": f"question-{index}", "gold": "gold"} for index in range(count)]
        expected_kg = f"output/example/{TABLE2_CHILD}/projections/confidence.json"
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

    def test_scientific_json_preserves_table_era_mode_order(self):
        value = {"metadata": {"b": 2, "a": 1}, "accuracy": {"z": 0, "a": 1}}
        expected = json.dumps(value, indent=2).encode("utf-8")
        self.assertEqual(runner.scientific_json_bytes(value), expected)
        self.assertNotEqual(expected, runner.canonical_json_bytes(value, newline=False))

    def test_source_drift_before_first_condition_blocks_model_evaluation(self):
        args = argparse.Namespace(run_id=RUN_ID, ollama_url="http://localhost:11434", dry_run=False)
        output_root = ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            run_dir = Path(temporary).resolve()
            summary, projections, graphs = live_preflight(run_dir)
            source = {"head": "a" * 40, "branch": "publication-refactored-rag", "baseline": "b" * 40}
            drifted = {**source, "head": "c" * 40}
            with mock.patch.object(
                runner, "validate_source_contract", side_effect=[source, drifted]
            ), mock.patch.object(
                runner,
                "preflight_same_run",
                return_value=(run_dir, summary, projections, graphs),
            ), mock.patch.object(
                runner, "fetch_matching_model", return_value=live_model_capture(summary)
            ), mock.patch.object(runner.evaluator, "evaluate") as evaluate:
                with self.assertRaisesRegex(runner.PhaseEError, "Source identity changed"):
                    runner.run_command(args, CONTRACT)
            evaluate.assert_not_called()

    def test_nested_foreign_run_identity_is_rejected(self):
        with self.assertRaisesRegex(runner.PhaseEError, "foreign run_id"):
            runner._reject_foreign_run_ids(
                {"inputs": [{"run_id": "another-run"}]}, RUN_ID, "fixture"
            )

    def test_real_preflight_rejects_foreign_stage_identity_before_child_writes(self):
        output_root = ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output_root, prefix="lineage-negative-"
        ) as temporary:
            run_dir = Path(temporary).resolve()
            run_id = run_dir.name
            commit = runner.run_git("rev-parse", "HEAD").stdout.strip()
            config_path = "configs/pipeline.json"
            config_sha = runner.sha256_bytes(
                runner._git_file_bytes(commit, config_path)
            )
            paths = CONTRACT["pipeline"]["artifacts"]

            def write(relative: str, value: object) -> None:
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(value, bytes):
                    path.write_bytes(value)
                else:
                    path.write_bytes(runner.canonical_json_bytes(value))

            write(
                paths["checkout_manifest"],
                {
                    "schema_version": "phase-b-checkout-manifest-2.0",
                    "status": "pass",
                    "run_id": run_id,
                    "protocol_id": CONTRACT["pipeline"]["protocol_id"],
                    "workflow_id": CONTRACT["pipeline"]["workflow_id"],
                    "matcher_id": CONTRACT["pipeline"]["matcher_id"],
                    "source": {
                        "commit": commit,
                        "worktree_clean": True,
                        "config": {"path": config_path, "sha256": config_sha},
                    },
                },
            )
            write(
                paths["full_run_manifest"],
                {
                    "schema_version": "phase-b-debug-full-run-1.0",
                    "run_id": run_id,
                    "protocol_id": CONTRACT["pipeline"]["protocol_id"],
                    "workflow_id": CONTRACT["pipeline"]["workflow_id"],
                    "training_seeds": CONTRACT["pipeline"]["training_seeds"],
                    "conditional_b07_approval": True,
                    "source_commit": commit,
                    "config": config_path,
                    "config_sha256": config_sha,
                },
            )
            archive_relative = "inputs/downloads/code-accord.zip"
            archive_bytes = b"immutable-input"
            write(archive_relative, archive_bytes)
            write(
                "manifests/02-input-acquisition-manifest.json",
                {
                    "schema_version": "phase-b-input-acquisition-manifest-1.0",
                    "dataset_id": CONTRACT["pipeline"]["dataset_id"],
                    "archive": {
                        "path": archive_relative,
                        "sha256": runner.sha256_bytes(archive_bytes),
                    },
                },
            )
            write(paths["preparation_manifest"], {"run_id": "foreign-run"})
            with self.assertRaisesRegex(runner.PhaseEError, "foreign run_id"):
                runner.preflight_same_run(run_id, CONTRACT)
            self.assertFalse((run_dir / TABLE2_CHILD).exists())

    def test_three_conditions_complete_and_resume_without_model_calls(self):
        args = argparse.Namespace(
            run_id=RUN_ID,
            ollama_url="http://localhost:11434",
            dry_run=False,
        )
        output_root = ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            run_dir = Path(temporary).resolve()
            summary, projections, graphs = live_preflight(run_dir)
            source = {"head": "a" * 40, "branch": "publication-refactored-rag", "baseline": "b" * 40}
            capture = live_model_capture(summary)
            evaluator_calls: list[str] = []

            def evaluate(_records, _kg, *, kg_identity, **_kwargs):
                evaluator_calls.append(kg_identity)
                return rag_result(summary, kg_identity)

            with mock.patch.object(
                runner, "validate_source_contract", return_value=source
            ), mock.patch.object(
                runner,
                "preflight_same_run",
                return_value=(run_dir, summary, projections, graphs),
            ), mock.patch.object(
                runner, "fetch_matching_model", return_value=capture
            ) as fetch, mock.patch.object(
                runner.evaluator, "evaluate", side_effect=evaluate
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_command(args, CONTRACT), 0)
                self.assertEqual(len(evaluator_calls), 3)
                self.assertEqual(fetch.call_count, 4)
                for condition in ("confidence", "corrective", "gold"):
                    result = run_dir / TABLE2_CHILD / "rag-results" / f"{condition}.json"
                    self.assertEqual(
                        result.read_bytes(),
                        runner.scientific_json_bytes(json.loads(result.read_text())),
                    )
                evaluator_calls.clear()
                fetch.reset_mock()
                self.assertEqual(runner.run_command(args, CONTRACT), 0)
                self.assertEqual(evaluator_calls, [])
                fetch.assert_not_called()

    def test_finalization_interruption_resumes_without_repeating_science(self):
        args = argparse.Namespace(
            run_id=RUN_ID,
            ollama_url="http://localhost:11434",
            dry_run=False,
        )
        output_root = ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            run_dir = Path(temporary).resolve()
            summary, projections, graphs = live_preflight(run_dir)
            source = {"head": "a" * 40, "branch": "publication-refactored-rag", "baseline": "b" * 40}
            capture = live_model_capture(summary)
            original_write_status = runner.write_status
            injected = False

            def fail_final_status(path, status, child):
                nonlocal injected
                if status.get("status") == "complete" and not injected:
                    injected = True
                    raise OSError("injected final status crash")
                return original_write_status(path, status, child)

            with mock.patch.object(runner, "validate_source_contract", return_value=source), \
                 mock.patch.object(runner, "preflight_same_run", return_value=(run_dir, summary, projections, graphs)), \
                 mock.patch.object(runner, "fetch_matching_model", return_value=capture) as fetch, \
                 mock.patch.object(runner.evaluator, "evaluate", side_effect=lambda _r, _k, *, kg_identity, **_kw: rag_result(summary, kg_identity)) as evaluate, \
                 mock.patch.object(runner, "write_status", side_effect=fail_final_status), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, "injected"):
                    runner.run_command(args, CONTRACT)
            self.assertTrue((run_dir / TABLE2_CHILD / "artifact-hashes.json").is_file())
            with mock.patch.object(runner, "validate_source_contract", return_value=source), \
                 mock.patch.object(runner, "preflight_same_run", return_value=(run_dir, summary, projections, graphs)), \
                 mock.patch.object(runner, "fetch_matching_model", return_value=capture) as resumed_fetch, \
                 mock.patch.object(runner.evaluator, "evaluate") as resumed_evaluate, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_command(args, CONTRACT), 0)
            resumed_fetch.assert_not_called()
            resumed_evaluate.assert_not_called()

    def test_self_consistent_parent_replacement_fails_final_identity_check(self):
        args = argparse.Namespace(
            run_id=RUN_ID,
            ollama_url="http://localhost:11434",
            dry_run=False,
        )
        output_root = ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            run_dir = Path(temporary).resolve()
            summary, projections, graphs = live_preflight(run_dir)
            replacement = json.loads(json.dumps(summary))
            replacement["parent_artifact_set_sha256"] = "f" * 64
            source = {
                "head": "a" * 40,
                "branch": "publication-refactored-rag",
                "baseline": "b" * 40,
            }
            capture = live_model_capture(summary)
            with mock.patch.object(
                runner, "validate_source_contract", return_value=source
            ), mock.patch.object(
                runner,
                "preflight_same_run",
                side_effect=[
                    (run_dir, summary, projections, graphs),
                    (run_dir, replacement, projections, graphs),
                ],
            ), mock.patch.object(
                runner, "fetch_matching_model", return_value=capture
            ), mock.patch.object(
                runner.evaluator,
                "evaluate",
                side_effect=lambda _r, _k, *, kg_identity, **_kw: rag_result(
                    summary, kg_identity
                ),
            ) as evaluate:
                with self.assertRaisesRegex(
                    runner.PhaseEError, "lineage changed"
                ):
                    runner.run_command(args, CONTRACT)
            self.assertEqual(evaluate.call_count, 3)
            status = runner.load_json(run_dir / TABLE2_CHILD / "run-status.json")
            self.assertNotEqual(status.get("status"), "complete")
            self.assertTrue((run_dir / TABLE2_CHILD / "artifact-hashes.json").is_file())

    def test_missing_post_stage_evidence_retries_that_stage_and_downstream(self):
        args = argparse.Namespace(run_id=RUN_ID, ollama_url="http://localhost:11434", dry_run=False)
        output_root = ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            run_dir = Path(temporary).resolve()
            summary, projections, graphs = live_preflight(run_dir)
            source = {"head": "a" * 40, "branch": "publication-refactored-rag", "baseline": "b" * 40}
            capture = live_model_capture(summary)
            common = [
                mock.patch.object(runner, "validate_source_contract", return_value=source),
                mock.patch.object(runner, "preflight_same_run", return_value=(run_dir, summary, projections, graphs)),
                mock.patch.object(runner, "fetch_matching_model", return_value=capture),
                mock.patch.object(runner.evaluator, "evaluate", side_effect=lambda _r, _k, *, kg_identity, **_kw: rag_result(summary, kg_identity)),
            ]
            with common[0], common[1], common[2], common[3], redirect_stdout(io.StringIO()):
                runner.run_command(args, CONTRACT)
            status_path = run_dir / TABLE2_CHILD / "run-status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["stages"]["rag_corrective"].pop("post_model_identity_check")
            status_path.write_bytes(runner.canonical_json_bytes(status))
            with mock.patch.object(runner, "validate_source_contract", return_value=source), \
                 mock.patch.object(runner, "preflight_same_run", return_value=(run_dir, summary, projections, graphs)), \
                 mock.patch.object(runner, "fetch_matching_model", return_value=capture) as fetch, \
                 mock.patch.object(runner.evaluator, "evaluate", side_effect=lambda _r, _k, *, kg_identity, **_kw: rag_result(summary, kg_identity)) as evaluate, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_command(args, CONTRACT), 0)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(fetch.call_count, 2)

    def test_completed_model_evidence_rejects_missing_mismatched_foreign_and_reordered(self):
        with tempfile.TemporaryDirectory() as temporary:
            child = Path(temporary).resolve() / "table2"
            child.mkdir()
            status_path = child / "run-status.json"
            summary = execution_summary()
            capture = live_model_capture(summary)
            status = {
                "schema_version": "phase-e-same-run-status-3.0",
                "run_id": RUN_ID,
                "status": "complete",
                "stages": {
                    f"rag_{condition}": {"status": "complete"}
                    for condition in ("confidence", "corrective", "gold")
                },
            }
            evidence, tags, show = capture
            phases = ["before-stages", "after-confidence", "after-corrective", "after-gold"]
            records = [
                runner.record_model_check(
                    child, status, status_path, phase, evidence, tags, show
                )
                for phase in phases
            ]
            status["pre_model_identity_check"] = records[0]
            for condition, record in zip(
                ("confidence", "corrective", "gold"), records[1:]
            ):
                status["stages"][f"rag_{condition}"][
                    "post_model_identity_check"
                ] = record
            runner._validate_status_model_checks(
                child, status, summary, require_complete=True
            )

            missing = json.loads(json.dumps(status))
            missing["stages"]["rag_corrective"].pop("post_model_identity_check")
            with self.assertRaises(runner.ModelEvidenceError):
                runner._validate_status_model_checks(
                    child, missing, summary, require_complete=True
                )

            reordered = json.loads(json.dumps(status))
            reordered["stages"]["rag_corrective"]["post_model_identity_check"] = records[1]
            with self.assertRaises(runner.ModelEvidenceError):
                runner._validate_status_model_checks(
                    child, reordered, summary, require_complete=True
                )

            mismatched = json.loads(json.dumps(status))
            mismatched["stages"]["rag_gold"]["post_model_identity_check"][
                "result_sha256"
            ] = "0" * 64
            with self.assertRaises(runner.ModelEvidenceError):
                runner._validate_status_model_checks(
                    child, mismatched, summary, require_complete=True
                )

            foreign = json.loads(json.dumps(status))
            result_path = child / records[3]["result"]
            result = runner.load_json(result_path)
            result["run_id"] = "foreign-run"
            result_path.write_bytes(runner.canonical_json_bytes(result))
            foreign_record = foreign["stages"]["rag_gold"][
                "post_model_identity_check"
            ]
            foreign_record["result_sha256"] = runner.sha256_file(result_path)
            foreign["model_identity_checks"][3] = foreign_record
            with self.assertRaises(runner.ModelEvidenceError):
                runner._validate_status_model_checks(
                    child, foreign, summary, require_complete=True
                )

    def test_runner_has_no_upstream_execution_or_arbitrary_copy_route(self):
        source = (ROOT / "table2_runner.py").read_text(encoding="utf-8")
        self.assertNotIn("build_kg.py", source)
        self.assertNotIn("copy_input", source)
        self.assertNotIn("--inference", source)
        self.assertNotIn("plan_training", source)
        self.assertNotIn("run_verifier(", source)


if __name__ == "__main__":
    unittest.main()
