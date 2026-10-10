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
from utils.rag import runner


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "resources/contracts/table2.json").read_text(encoding="utf-8")
)
TABLE2_CHILD = "table2-code-all"
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
    question_count = CONTRACT["evaluator"]["part_of_questions"]
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


    def test_training_cache_manifest_is_an_exact_parent_lineage_binding(self):
        seed = 42
        digest = "a" * 64
        training_config = {
            "base_model": "microsoft/deberta-large",
            "base_model_revision": "b" * 40,
        }
        checkpoint = {
            **training_config,
            "dataset_compatibility_report": (
                "audit/model-training-dataset-compatibility.json"
            ),
            "model_cache_manifest": "manifests/huggingface-model-cache.json",
            "model_cache_manifest_sha256": digest,
            "restart_state_sha256": "c" * 64,
        }
        inputs = {
            "acquisition_manifest_sha256": "1" * 64,
            "checkout_manifest_sha256": "2" * 64,
            "development_jsonl_sha256": "3" * 64,
            "preparation_manifest_sha256": "4" * 64,
            "split_manifest_sha256": "5" * 64,
            "train_jsonl_sha256": "6" * 64,
        }
        manifest = {
            "recipe": training_config,
            "inputs": {**inputs, "model_cache_manifest_sha256": digest},
            "outputs": {
                "checkpoint": f"checkpoints/seed-{seed}/checkpoint.pt",
                "checkpoint_manifest": (
                    f"checkpoints/seed-{seed}/checkpoint-manifest.json"
                ),
                "dataset_compatibility_report": checkpoint[
                    "dataset_compatibility_report"
                ],
                "model_cache_manifest": checkpoint["model_cache_manifest"],
                "progress_log": f"logs/model-train-seed-{seed}.log",
                "restart_state": f"checkpoints/seed-{seed}/restart-state.pt",
                "training_summary": (
                    f"checkpoints/seed-{seed}/training-summary.json"
                ),
            },
            "resume": {"restart_state_sha256": checkpoint["restart_state_sha256"]},
            "split": {
                "split_id": "CODE-SPLIT-1",
                "seed": 42,
                "train_sentences": 586,
                "development_sentences": 103,
                "test_sentences": 173,
            },
        }
        arguments = {
            "seed": seed,
            "checkpoint_manifest": checkpoint,
            "training_config": training_config,
            "split_seed": 42,
            "split": {"train": 586, "development": 103, "test": 173},
            "expected_inputs": inputs,
        }

        runner._validate_training_checkpoint_identity(
            train_manifest=manifest, **arguments
        )
        for mapping, field in (
            ("inputs", "model_cache_manifest_sha256"),
            ("outputs", "model_cache_manifest"),
        ):
            with self.subTest(mapping=mapping):
                changed = json.loads(json.dumps(manifest))
                changed[mapping].pop(field)
                with self.assertRaisesRegex(
                    runner.PhaseEError,
                    "training and checkpoint identities disagree",
                ):
                    runner._validate_training_checkpoint_identity(
                        train_manifest=changed, **arguments
                    )

        changed = json.loads(json.dumps(manifest))
        changed["outputs"]["unrecognized"] = "same-run-but-undeclared.json"
        with self.assertRaisesRegex(
            runner.PhaseEError, "training and checkpoint identities disagree"
        ):
            runner._validate_training_checkpoint_identity(
                train_manifest=changed, **arguments
            )

    def test_authenticated_older_run_config_uses_explicit_legacy_compatibility(self):
        commit = "3" * 40
        path = "configs/phase_b_path_a.json"
        payload = (ROOT / "resources/configs/pipeline.json").read_bytes()
        digest = runner.sha256_bytes(payload)
        checkout = {
            "source": {"commit": commit},
            "tracked_artifact_sha256": {path: digest},
        }
        full = {"source_commit": commit, "config": path, "config_sha256": digest}
        with mock.patch.object(runner, "_git_file_bytes", return_value=payload):
            config, legacy = runner._load_authenticated_run_config(
                checkout, full, CONTRACT
            )
        self.assertTrue(legacy)
        self.assertEqual(config["protocol_id"], CONTRACT["pipeline"]["protocol_id"])

    def test_current_checkout_config_requires_new_run_seals(self):
        commit = runner.run_git("rev-parse", "HEAD").stdout.strip()
        path = "resources/configs/pipeline.json"
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

    def test_saved_configuration_needs_no_producer_git_history(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            run_dir = Path(temporary)
            relative = "inputs/config/pipeline.json"
            payload = (ROOT / "resources/configs/pipeline.json").read_bytes()
            path = run_dir / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            digest = runner.sha256_bytes(payload)
            checkout = {"source": {"commit": "a"*40, "config": {
                "path": "resources/configs/pipeline.json", "sha256": digest, "run_local_path": relative}}}
            full = {"source_commit": "a"*40, "config": "resources/configs/pipeline.json", "config_sha256": digest}
            with mock.patch.object(runner, "_git_file_bytes", side_effect=AssertionError("old Git object accessed")):
                value, legacy = runner._load_authenticated_run_config(checkout, full, CONTRACT, run_dir=run_dir)
            self.assertFalse(legacy)
            self.assertEqual(value["protocol_id"], CONTRACT["pipeline"]["protocol_id"])

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
        self.assertEqual(summary["questions"], CONTRACT["evaluator"]["part_of_questions"])


    def test_rag_output_rejects_failed_model_calls_and_identity_substitution(self):
        modes = runner.evaluator.MODES
        count = CONTRACT["evaluator"]["part_of_questions"]
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


    def test_scientific_json_preserves_table_era_mode_order(self):
        value = {"metadata": {"b": 2, "a": 1}, "accuracy": {"z": 0, "a": 1}}
        expected = json.dumps(value, indent=2).encode("utf-8")
        self.assertEqual(runner.scientific_json_bytes(value), expected)
        self.assertNotEqual(expected, runner.canonical_json_bytes(value, newline=False))


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
            config_path = "resources/configs/pipeline.json"
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








if __name__ == "__main__":
    unittest.main()
