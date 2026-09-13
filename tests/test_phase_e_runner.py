import argparse
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "phase_e_rag_contract.json").read_text(encoding="utf-8"))


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "phase_e_runner", ROOT / "run_phase_e_rag.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verifier_manifest(model_name="example-model:tag"):
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
                "family": "example-family",
                "parameter_size": "12.3B",
                "quantization_level": "Q4_EXAMPLE",
            },
        },
    }


def mocked_http_response(value):
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(value).encode("utf-8")
    return response


def execution_summary():
    return {
        "run_id": CONTRACT["same_run"]["expected_run_id"],
        "parent_artifact_set_sha256": "1" * 64,
        "score_manifest_sha256": "2" * 64,
        "seed": 42,
        "threshold": 0.25,
        "projection_sha256": {"records": "3" * 64},
        "graphs": {
            name: {"source_graph_id": value["graph_id"]}
            for name, value in CONTRACT["same_run"]["graphs"].items()
        },
        "model_identity": {
            "name": "example-model:tag",
            "tag_digest": "4" * 64,
            "blob_sha256": "5" * 64,
            "verifier_environment_sha256": "6" * 64,
        },
    }


class PhaseERunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_source_contract_is_valid(self):
        result = self.runner.validate_source_contract(CONTRACT, require_clean=False)
        self.assertEqual(result["branch"], "publication-refactored-rag")
        self.assertEqual(result["baseline"], CONTRACT["source_baseline"]["commit"])

    def test_clean_source_gate_rejects_untracked_files(self):
        def fake_git(*args, **_kwargs):
            if args == ("rev-parse", "HEAD"):
                return SimpleNamespace(stdout="f" * 40 + "\n", returncode=0)
            if args == ("branch", "--show-current"):
                return SimpleNamespace(
                    stdout="publication-refactored-rag\n", returncode=0
                )
            if args[:2] == ("ls-files", "--others"):
                return SimpleNamespace(stdout="sitecustomize.py\n", returncode=0)
            return SimpleNamespace(stdout="", returncode=0)

        with mock.patch.object(self.runner, "run_git", side_effect=fake_git):
            with self.assertRaisesRegex(self.runner.PhaseEError, "Untracked"):
                self.runner.validate_source_contract(CONTRACT, require_clean=True)

    def test_only_run_id_selects_scientific_inputs(self):
        parser = self.runner.build_parser()
        args = parser.parse_args(
            ["run", "--run-id", CONTRACT["same_run"]["expected_run_id"], "--dry-run"]
        )
        self.assertEqual(set(vars(args)), {"command", "run_id", "ollama_url", "dry_run"})
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run",
                    "--run-id",
                    CONTRACT["same_run"]["expected_run_id"],
                    "--inference",
                    "foreign.jsonl",
                ]
            )

    def test_run_id_cannot_escape_output(self):
        for value in ("../escape", "nested/run", ".", "-leading"):
            with self.subTest(value=value):
                with self.assertRaises(self.runner.PhaseEError):
                    self.runner.validate_run_id(value, require_existing=False)

    def test_physical_file_from_another_valid_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            selected = output / "selected-run"
            foreign = output / "foreign-run"
            selected.mkdir()
            foreign.mkdir()
            foreign_file = foreign / "verdicts.jsonl"
            foreign_file.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(self.runner.PhaseEError, "unsafe"):
                self.runner.run_file(selected, str(foreign_file))

    def test_contract_rejects_a_different_run_before_files_or_model(self):
        with mock.patch.object(self.runner, "validate_run_id") as path_check:
            with self.assertRaisesRegex(self.runner.PhaseEError, "authenticates run"):
                self.runner.preflight_same_run("another-run", CONTRACT)
        path_check.assert_not_called()

    def test_mixed_graph_seed_is_rejected_before_files_or_model(self):
        mixed = copy.deepcopy(CONTRACT)
        mixed["same_run"]["graphs"]["confidence"]["condition"] = (
            "confidence_filtered:seed-43"
        )
        with mock.patch.object(self.runner, "validate_run_id") as path_check:
            with self.assertRaisesRegex(self.runner.PhaseEError, "mixes a run or seed"):
                self.runner.preflight_same_run(
                    CONTRACT["same_run"]["expected_run_id"], mixed
                )
        path_check.assert_not_called()

    def test_mixed_verifier_stage_bindings_are_rejected_before_files_or_model(self):
        replacements = {
            "predictions/test/candidates.jsonl": "predictions/test/foreign-candidates.jsonl",
            "verifier/corrective/environment-manifest.json": (
                "verifier/foreign/environment-manifest.json"
            ),
            "verifier/corrective/verdicts.jsonl": "verifier/foreign/verdicts.jsonl",
        }
        for original, replacement in replacements.items():
            with self.subTest(original=original):
                mixed = copy.deepcopy(CONTRACT)
                bindings = mixed["same_run"]["corrective_verifier"][
                    "required_bindings"
                ]
                bindings[replacement] = bindings.pop(original)
                with mock.patch.object(self.runner, "validate_run_id") as path_check:
                    with self.assertRaisesRegex(
                        self.runner.PhaseEError, "Corrective-verifier bindings"
                    ):
                        self.runner.preflight_same_run(
                            CONTRACT["same_run"]["expected_run_id"], mixed
                        )
                path_check.assert_not_called()

    def test_model_identity_is_derived_from_same_run_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "environment.json"
            path.write_text(json.dumps(verifier_manifest()), encoding="utf-8")
            digest = self.runner.sha256_file(path)
            identity = self.runner.load_verifier_model_identity(path, digest)
        self.assertEqual(identity["name"], "example-model:tag")
        self.assertEqual(identity["tag_digest"], "a" * 64)
        self.assertEqual(identity["blob_sha256"], "b" * 64)
        self.assertEqual(identity["verifier_environment_sha256"], digest)

    def test_model_match_uses_stable_tag_blob_and_details(self):
        expected = {
            "name": "example-model:tag",
            "tag_digest": "a" * 64,
            "blob_sha256": "b" * 64,
            "details": {
                "family": "example-family",
                "parameter_size": "12.3B",
                "quantization_level": "Q4_EXAMPLE",
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
        with mock.patch.object(
            self.runner.urllib.request,
            "urlopen",
            side_effect=[mocked_http_response(tags), mocked_http_response(show)],
        ):
            evidence, tags_bytes, show_bytes = self.runner.fetch_matching_model(
                "http://localhost:11434", expected
            )
        self.assertTrue(evidence["identity_verified"])
        self.assertEqual(evidence["blob_sha256"], expected["blob_sha256"])
        self.assertEqual(tags_bytes, self.runner.canonical_json_bytes(tags))
        self.assertEqual(show_bytes, self.runner.canonical_json_bytes(show))

    def test_model_match_rejects_foreign_tag(self):
        expected = {
            "name": "example-model:tag",
            "tag_digest": "a" * 64,
            "blob_sha256": "b" * 64,
            "details": {},
        }
        tags = {"models": [{"name": expected["name"], "digest": "d" * 64}]}
        with mock.patch.object(
            self.runner.urllib.request,
            "urlopen",
            return_value=mocked_http_response(tags),
        ):
            with self.assertRaisesRegex(self.runner.PhaseEError, "differs"):
                self.runner.fetch_matching_model("http://localhost:11434", expected)

    def test_projection_preserves_prepared_order_and_private_gold(self):
        prepared = []
        gold = []
        identifiers = []
        for index in range(173):
            example_id = f"example-{index:03d}"
            identifiers.append(example_id)
            prepared.append(
                {"example_id": example_id, "content": f"Head {index} is in Tail {index}."}
            )
            gold.append(
                {
                    "example_id": example_id,
                    "gold_triples": [
                        {
                            "head": {"text": f"Head {index}"},
                            "relation": "part-of",
                            "tail": {"text": f"Tail {index}"},
                        }
                    ],
                }
            )
        rows, content, summary = self.runner.project_evaluator_records(
            prepared, gold, {"test_ids": identifiers}, CONTRACT
        )
        self.assertEqual([row["doc_id"] for row in rows], identifiers)
        self.assertEqual(rows[0]["gold_triples"][0]["head_text"], "Head 0")
        self.assertEqual(content, self.runner.canonical_jsonl_bytes(rows))
        self.assertEqual(summary["questions"], 10)

    def test_graph_projection_joins_ids_without_filtering_or_reordering(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            relative_root = "graph-rag/graphs/example"
            entities = [
                {"entity_id": "e1", "canonical_label": "Head", "entity_type": "Object"},
                {"entity_id": "e2", "canonical_label": "Tail", "entity_type": "Object"},
            ]
            relations = [{"relation_id": "r1", "label": "part-of"}]
            triples = [
                {
                    "triple_id": "t1",
                    "head_id": "e1",
                    "relation_id": "r1",
                    "tail_id": "e2",
                }
            ]
            collections = {"entities": entities, "relations": relations, "triples": triples}
            ledger = {}
            for name, rows in collections.items():
                path = run_dir / relative_root / f"{name}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.runner.canonical_jsonl_bytes(rows))
                ledger[f"graphs/example/{name}.jsonl"] = {
                    "path": f"graphs/example/{name}.jsonl",
                    "sha256": self.runner.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            manifest = {
                "schema_version": "rag-graph-snapshot-1.0",
                "condition": "example:seed-42",
                "construction_recipe": "example-v1",
                "graph_id": "g" * 64,
                **collections,
                "entity_sha256": self.runner.sha256_bytes(
                    self.runner.canonical_json_bytes(entities, newline=False)
                ),
                "relation_sha256": self.runner.sha256_bytes(
                    self.runner.canonical_json_bytes(relations, newline=False)
                ),
                "triple_sha256": self.runner.sha256_bytes(
                    self.runner.canonical_json_bytes(triples, newline=False)
                ),
            }
            manifest_path = run_dir / relative_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            ledger["graphs/example/manifest.json"] = {
                "path": "graphs/example/manifest.json",
                "sha256": self.runner.sha256_file(manifest_path),
                "bytes": manifest_path.stat().st_size,
            }
            specification = {
                "run_id": "same-run",
                "manifest": f"{relative_root}/manifest.json",
                "manifest_sha256": self.runner.sha256_file(manifest_path),
                "condition": "example:seed-42",
                "construction_recipe": "example-v1",
                "graph_id": "g" * 64,
                "entity_count": 2,
                "relation_count": 1,
                "triple_count": 1,
            }
            graph, _content, summary = self.runner.validate_graph_snapshot(
                run_dir, "graph-rag", specification, ledger
            )
        self.assertEqual([node["id"] for node in graph["nodes"]], ["Head", "Tail"])
        self.assertEqual(
            graph["edges"],
            [{"head": "Head", "relation": "part-of", "tail": "Tail", "triple_id": "t1"}],
        )
        self.assertEqual(summary["edges"], 1)

    def test_transport_shards_reconstruct_the_ledger_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            traces = run_dir / "graph-rag" / "traces"
            parts_dir = traces / "retrieval.parts"
            parts_dir.mkdir(parents=True)
            contents = [b'{"record":1}\n', b'{"record":2}\n']
            parts = []
            for index, content in enumerate(contents, 1):
                part_path = parts_dir / f"retrieval-{index:04d}.jsonl"
                part_path.write_bytes(content)
                parts.append(
                    {
                        "bytes": len(content),
                        "first_record": index,
                        "last_record": index,
                        "path": f"retrieval.parts/{part_path.name}",
                        "records": 1,
                        "sha256": self.runner.sha256_bytes(content),
                    }
                )
            original = b"".join(contents)
            entry = {
                "path": "traces/retrieval.jsonl",
                "bytes": len(original),
                "sha256": self.runner.sha256_bytes(original),
            }
            manifest = {
                "encoding": "utf-8",
                "join_mode": "byte_concatenation_in_listed_order",
                "logical_path": entry["path"],
                "original": {**entry, "records": 2},
                "parts": parts,
                "record_format": "jsonl",
                "representation": "transport_sharding_only",
                "schema_version": "graph-rag-jsonl-shards-1.0",
                "split_boundary": "complete_jsonl_record",
            }
            manifest["original"].pop("path")
            (traces / "retrieval.parts.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.runner.validate_sharded_ledger_artifact(
                run_dir, "graph-rag", entry["path"], entry
            )
            parts[1]["last_record"] = 3
            (traces / "retrieval.parts.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(self.runner.PhaseEError, "contiguous"):
                self.runner.validate_sharded_ledger_artifact(
                    run_dir, "graph-rag", entry["path"], entry
                )

    def test_dry_run_has_no_model_call_or_output_write(self):
        summary = {
            "run_id": CONTRACT["same_run"]["expected_run_id"],
            "model_identity": {"name": "example"},
        }
        args = argparse.Namespace(
            run_id=CONTRACT["same_run"]["expected_run_id"],
            ollama_url="http://localhost:11434",
            dry_run=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with mock.patch.object(self.runner, "validate_source_contract", return_value={}):
                with mock.patch.object(
                    self.runner, "preflight_same_run", return_value=(run_dir, summary, {})
                ):
                    with mock.patch.object(self.runner, "fetch_matching_model") as model:
                        with redirect_stdout(io.StringIO()):
                            self.assertEqual(self.runner.run_command(args, CONTRACT), 0)
            model.assert_not_called()
            self.assertFalse((run_dir / "phase-e-rag").exists())

    def test_model_failure_occurs_before_child_write(self):
        summary = execution_summary()
        args = argparse.Namespace(
            run_id=CONTRACT["same_run"]["expected_run_id"],
            ollama_url="http://localhost:11434",
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with mock.patch.object(
                self.runner, "validate_source_contract", return_value={"head": "7" * 40}
            ):
                with mock.patch.object(
                    self.runner, "preflight_same_run", return_value=(run_dir, summary, {})
                ):
                    with mock.patch.object(
                        self.runner,
                        "fetch_matching_model",
                        side_effect=self.runner.PhaseEError("model mismatch"),
                    ):
                        with self.assertRaisesRegex(self.runner.PhaseEError, "model mismatch"):
                            self.runner.run_command(args, CONTRACT)
            self.assertFalse((run_dir / "phase-e-rag").exists())

    def test_rag_output_rejects_failed_model_calls(self):
        modes = CONTRACT["evaluator"]["modes"]
        count = CONTRACT["evaluator"]["max_questions"]
        output = {
            "metadata": {
                "kg": "output/same-run/phase-e-rag/projections/confidence.json",
                "n_questions": count,
                "model": "example",
                "time_seconds": 1.0,
            },
            "accuracy": {mode: 0.0 for mode in modes},
            "correct_counts": {mode: 0 for mode in modes},
            "results": {
                mode: [
                    {"q": "q", "pred": "answer", "gold": "gold", "correct": False}
                    for _ in range(count)
                ]
                for mode in modes
            },
        }
        output["results"]["hybrid"][0]["pred"] = "ERROR: timeout"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(output), encoding="utf-8")
            with self.assertRaisesRegex(self.runner.PhaseEError, "failed model call"):
                self.runner.validate_rag_output(path, CONTRACT, "example")

    def test_rag_output_rejects_condition_or_question_substitution(self):
        modes = CONTRACT["evaluator"]["modes"]
        count = CONTRACT["evaluator"]["max_questions"]
        questions = [{"q": f"question-{index}", "gold": "gold"} for index in range(count)]
        expected_kg = "output/same-run/phase-e-rag/projections/confidence.json"
        output = {
            "metadata": {
                "kg": "output/same-run/phase-e-rag/projections/corrective.json",
                "n_questions": count,
                "model": "example",
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
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(output), encoding="utf-8")
            with self.assertRaisesRegex(self.runner.PhaseEError, "same-run graph"):
                self.runner.validate_rag_output(
                    path,
                    CONTRACT,
                    "example",
                    expected_questions=questions,
                    expected_kg=expected_kg,
                )
            output["metadata"]["kg"] = expected_kg
            output["results"]["hybrid"][0]["q"] = "foreign question"
            path.write_text(json.dumps(output), encoding="utf-8")
            with self.assertRaisesRegex(self.runner.PhaseEError, "question/gold"):
                self.runner.validate_rag_output(
                    path,
                    CONTRACT,
                    "example",
                    expected_questions=questions,
                    expected_kg=expected_kg,
                )

    def test_incomplete_finalization_does_not_call_model_again(self):
        summary = execution_summary()
        source = {"head": "7" * 40}
        args = argparse.Namespace(
            run_id=CONTRACT["same_run"]["expected_run_id"],
            ollama_url="http://localhost:11434",
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            child = run_dir / "phase-e-rag"
            child.mkdir()
            status = {
                "schema_version": "phase-e-same-run-status-1.0",
                "run_id": args.run_id,
                "identity": self.runner.build_child_identity(args.run_id, source, summary),
                "status": "running",
                "created_at": "2026-09-13T00:00:00+00:00",
                "stages": {},
            }
            (child / "run-status.json").write_text(json.dumps(status), encoding="utf-8")
            (child / "artifact-hashes.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(
                self.runner, "validate_source_contract", return_value=source
            ), mock.patch.object(
                self.runner,
                "preflight_same_run",
                return_value=(run_dir, summary, {}),
            ), mock.patch.object(
                self.runner, "validate_phase_e_child"
            ) as validate_child, mock.patch.object(
                self.runner, "fetch_matching_model"
            ) as model, redirect_stdout(io.StringIO()):
                self.assertEqual(self.runner.run_command(args, CONTRACT), 0)
            validate_child.assert_called_once()
            model.assert_not_called()
            recovered = json.loads((child / "run-status.json").read_text(encoding="utf-8"))
            self.assertEqual(recovered["status"], "complete")
            self.assertTrue(recovered["recovered_finalization"])

    def test_empty_crash_created_child_can_resume_identity_initialization(self):
        summary = execution_summary()
        source = {"head": "7" * 40}
        args = argparse.Namespace(
            run_id=CONTRACT["same_run"]["expected_run_id"],
            ollama_url="http://localhost:11434",
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            child = run_dir / "phase-e-rag"
            child.mkdir()
            with mock.patch.object(
                self.runner, "validate_source_contract", return_value=source
            ), mock.patch.object(
                self.runner,
                "preflight_same_run",
                return_value=(run_dir, summary, {}),
            ), mock.patch.object(
                self.runner,
                "fetch_matching_model",
                side_effect=self.runner.PhaseEError("model unavailable"),
            ) as model:
                with self.assertRaisesRegex(self.runner.PhaseEError, "model unavailable"):
                    self.runner.run_command(args, CONTRACT)
            model.assert_called_once()
            self.assertEqual(list(child.iterdir()), [])

    def test_child_rejects_projection_manifest_not_bound_to_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            child = Path(temporary).resolve() / "phase-e-rag"
            projection_dir = child / "projections"
            projection_dir.mkdir(parents=True)
            projection_paths = {
                "records": projection_dir / "test-with-private-gold.jsonl",
                "confidence": projection_dir / "confidence-seed-42.json",
                "corrective": projection_dir / "corrective-seed-42.json",
                "gold": projection_dir / "gold-oracle.json",
            }
            for name, path in projection_paths.items():
                path.write_text(name, encoding="utf-8")
            (projection_dir / "manifest.json").write_text(
                json.dumps({"run_id": "foreign-run"}), encoding="utf-8"
            )
            preflight = {
                "run_id": "same-run",
                "projection_sha256": {
                    name: self.runner.sha256_file(path)
                    for name, path in projection_paths.items()
                },
                "records": {"question_contract": []},
                "parent_artifact_set_sha256": "1" * 64,
                "score_manifest_sha256": "2" * 64,
                "seed": 42,
                "threshold": 0.25,
                "graphs": {},
            }
            with mock.patch.object(self.runner, "validate_child_hashes"):
                with self.assertRaisesRegex(
                    self.runner.PhaseEError, "projection manifest"
                ):
                    self.runner.validate_phase_e_child(
                        child, {}, CONTRACT, preflight, projection_paths
                    )

    def test_unverified_interrupted_output_is_quarantined_and_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            child = Path(temporary).resolve() / "phase-e-rag"
            output = child / "rag-results" / "confidence.json"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"old-unverified")
            status_path = child / "run-status.json"
            status = {
                "stages": {
                    "rag_confidence": {
                        "status": "output-validated",
                        "attempts": [],
                    }
                }
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")

            def execute(*_args, **_kwargs):
                output.write_bytes(b"new-validated")
                return SimpleNamespace(returncode=0)

            with mock.patch.object(self.runner.subprocess, "run", side_effect=execute):
                result = self.runner.run_stage(
                    "rag_confidence",
                    ["evaluator"],
                    output,
                    lambda path: {"sha256": self.runner.sha256_file(path)},
                    child,
                    status,
                    status_path,
                )
            recovered = child / "recovery" / "rag_confidence" / "unverified-001.json"
            self.assertEqual(recovered.read_bytes(), b"old-unverified")
            self.assertEqual(output.read_bytes(), b"new-validated")
            self.assertEqual(status["stages"]["rag_confidence"]["status"], "output-validated")
            self.assertEqual(result["sha256"], self.runner.sha256_file(output))

    def test_child_link_is_rejected_before_model_call(self):
        summary = execution_summary()
        source = {"head": "7" * 40}
        args = argparse.Namespace(
            run_id=CONTRACT["same_run"]["expected_run_id"],
            ollama_url="http://localhost:11434",
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "outside"
            target.mkdir()
            child = root / "phase-e-rag"
            try:
                child.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are unavailable")
            with mock.patch.object(
                self.runner, "validate_source_contract", return_value=source
            ), mock.patch.object(
                self.runner,
                "preflight_same_run",
                return_value=(root, summary, {}),
            ), mock.patch.object(
                self.runner, "fetch_matching_model"
            ) as model:
                with self.assertRaisesRegex(self.runner.PhaseEError, "physical"):
                    self.runner.run_command(args, CONTRACT)
            model.assert_not_called()

    def test_runner_has_no_upstream_execution_or_copy_route(self):
        source = (ROOT / "run_phase_e_rag.py").read_text(encoding="utf-8")
        self.assertNotIn("build_kg.py", source)
        self.assertNotIn("phase_b.py", source)
        self.assertNotIn("copy_input", source)
        self.assertNotIn("--inference", source)
        self.assertIn('child_dir = run_dir / "phase-e-rag"', source)


if __name__ == "__main__":
    unittest.main()
