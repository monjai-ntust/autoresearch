"""Static contracts for the single Python table-pipeline entry point."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pipeline
from utils.common.artifact_io import DataContractError, sha256_file
from utils.common.paths import RunLayout, discover_source_root


SOURCE_ROOT = discover_source_root(Path(__file__))
ENTRYPOINT = SOURCE_ROOT / "pipeline.py"
PIPELINE_SOURCE = ENTRYPOINT.read_text(encoding="utf-8")


class PipelineEntrypointTests(unittest.TestCase):
    def test_python_entrypoint_replaces_the_shell_dispatch_pair(self):
        self.assertTrue(ENTRYPOINT.is_file())
        self.assertFalse((SOURCE_ROOT / "phase_b.py").exists())
        self.assertFalse((SOURCE_ROOT / "phase_b.sh").exists())
        self.assertIn('prog="python pipeline.py"', PIPELINE_SOURCE)
        self.assertIn('"full"', PIPELINE_SOURCE)
        self.assertIn('"table2"', PIPELINE_SOURCE)

    def test_help_is_available_without_data_model_or_network_access(self):
        completed = subprocess.run(
            [sys.executable, "-B", "pipeline.py", "--help"],
            cwd=SOURCE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("full", completed.stdout)
        self.assertIn("table2", completed.stdout)
        for removed in (
            "doctor",
            "fetch",
            "prepare-pilot",
            "assemble-candidates",
            "pilot-verifier",
            "select-threshold",
            "inspect-table2-model",
        ):
            self.assertNotIn(removed, completed.stdout)

    def test_full_pipeline_keeps_the_approved_stage_order(self):
        start = PIPELINE_SOURCE.index("def _run_full_pipeline(")
        end = PIPELINE_SOURCE.index("\ndef _parser(", start)
        body = PIPELINE_SOURCE[start:end]
        ordered = (
            "preparation_stage.run",
            "encoder_stage.train_and_generate_development",
            "verifier_stage.prepare_development_gate",
            "verifier_stage.run_pilot",
            "encoder_stage.generate_test_candidates",
            "verifier_stage.run_test",
            "evaluation_stage.run",
            "rag_stage.run",
        )
        positions = [body.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_trainer_is_reached_only_through_the_pipeline(self):
        model_source = (SOURCE_ROOT / "utils/encoder/model.py").read_text(
            encoding="utf-8"
        )
        trainer_source = (SOURCE_ROOT / "stages/encoder.py").read_text(encoding="utf-8")
        self.assertIn('"pipeline.py",', model_source)
        self.assertIn('"_train-encoder",', model_source)
        self.assertNotIn('if __name__ == "__main__"', trainer_source)
        self.assertIn('if __name__ == "__main__"', PIPELINE_SOURCE)

    def test_stage_controllers_are_separate_internal_modules(self):
        for name in ("preparation", "encoder", "verifier", "evaluation", "rag"):
            source = (SOURCE_ROOT / "stages" / f"{name}.py").read_text(
                encoding="utf-8"
            )
            self.assertNotIn('if __name__ == "__main__"', source)

    def test_private_trainer_rejects_user_supplied_artifact_paths(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "pipeline.py",
                "_train-encoder",
                "--run-id",
                "safe-run",
                "--training-seed",
                "42",
                "--prepared-dir",
                "foreign/run/data",
            ],
            cwd=SOURCE_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unrecognized arguments", completed.stderr)

    def test_generation_manifest_without_seal_is_sealed_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            layout = RunLayout(source, "recovery-run")
            layout.run_root.mkdir(parents=True)
            manifest_path = layout.resolve(
                "manifests/model-generate-candidates-live-seed-42-test.json"
            )
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}", encoding="utf-8")
            manifest = {"outputs": {}}
            config = SimpleNamespace(value={})
            with mock.patch.object(
                pipeline,
                "_validate_generation_stage",
                return_value=manifest,
            ) as validate, mock.patch.object(
                pipeline, "_write_same_run_seal"
            ) as seal, mock.patch.object(
                pipeline, "generate_candidates"
            ) as generate:
                self.assertEqual(
                    pipeline._resume_generation_stage(
                        layout, config, seed=42, split="test"
                    ),
                    manifest,
                )
            self.assertEqual(validate.call_count, 2)
            seal.assert_called_once_with(layout, manifest_path, manifest)
            generate.assert_not_called()

    def test_interrupted_pilot_is_quarantined_and_never_uses_a_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            layout = RunLayout(source, "recovery-run")
            layout.run_root.mkdir(parents=True)
            prefix = "pilot/simple-repeat-1"
            request = layout.resolve(f"verifier/{prefix}/simple/requests.jsonl")
            request.parent.mkdir(parents=True)
            request.write_text(json.dumps({"partial": True}) + "\n", encoding="utf-8")
            inputs = {}
            for name in (
                "sentences.jsonl",
                "candidates.jsonl",
                "development.jsonl",
                "warmup.jsonl",
                "model.bin",
                "pilot-selection.json",
            ):
                path = layout.resolve(f"fixtures/{name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
                inputs[name] = path
            manifest = {"outputs": {}}
            config = SimpleNamespace(value={})
            with mock.patch.object(
                pipeline, "run_verifier", return_value=manifest
            ) as verifier, mock.patch.object(
                pipeline, "_write_same_run_seal"
            ), mock.patch.object(
                pipeline, "_validate_verifier_stage", return_value=manifest
            ):
                pipeline._resume_verifier_stage(
                    layout,
                    config,
                    mode="simple",
                    sentences=inputs["sentences.jsonl"],
                    candidates=inputs["candidates.jsonl"],
                    development=inputs["development.jsonl"],
                    warmup_candidates=inputs["warmup.jsonl"],
                    model_blob=inputs["model.bin"],
                    ollama_url="http://localhost:11434",
                    artifact_prefix=prefix,
                    pilot_selection=inputs["pilot-selection.json"],
                    allow_response_cache=False,
                )
            call = verifier.call_args.kwargs
            self.assertIsNone(call["cache_ledger_path"])
            self.assertEqual(call["artifact_prefix"], prefix)
            self.assertEqual(call["pilot_selection_path"], inputs["pilot-selection.json"])
            self.assertFalse(request.exists())
            quarantine = layout.resolve(
                "inputs/recovery/quarantine/verifier-pilot-simple-repeat-1-simple"
            )
            self.assertEqual(len(list(quarantine.glob("*-requests.jsonl"))), 1)

    def test_private_trainer_inputs_are_bound_to_preparation_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            layout = RunLayout(source, "trainer-input-run")
            layout.run_root.mkdir(parents=True)
            config = SimpleNamespace(
                value={
                    "protocol_id": "B04-PATH-A-1.3",
                    "dataset": {"dataset_id": "CODE-ACCORD-v1.0.0"},
                    "split": {
                        "split_id": "CODE-SPLIT-1",
                        "seed": 42,
                        "train_sentences": 1,
                        "development_sentences": 1,
                        "test_sentences": 1,
                    },
                }
            )
            archive = layout.resolve("inputs/downloads/archive.zip")
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"archive")
            acquisition = layout.resolve(
                "manifests/02-input-acquisition-manifest.json"
            )
            acquisition.parent.mkdir(parents=True)
            acquisition.write_text(
                json.dumps(
                    {
                        "dataset_id": "CODE-ACCORD-v1.0.0",
                        "archive": {
                            "path": "inputs/downloads/archive.zip",
                            "sha256": sha256_file(archive),
                        },
                    }
                ),
                encoding="utf-8",
            )
            data = {}
            for relative in ("train.jsonl", "development.jsonl"):
                path = layout.resolve(f"data-prepared/{relative}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                data[relative] = path
            split_path = layout.resolve("data-prepared/split-manifest.json")
            split_path.write_text(
                json.dumps(
                    {
                        "split_id": "CODE-SPLIT-1",
                        "seed": 42,
                        "train_count": 1,
                        "development_count": 1,
                        "test_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            data["split-manifest.json"] = split_path
            preparation = layout.resolve(
                "manifests/03-data-preparation-manifest.json"
            )
            preparation.write_text(
                json.dumps(
                    {
                        "protocol_id": "B04-PATH-A-1.3",
                        "dataset_id": "CODE-ACCORD-v1.0.0",
                        "byte_identical_independent_materializations": True,
                        "acquisition_manifest_sha256": sha256_file(acquisition),
                        "archive_sha256": sha256_file(archive),
                        "artifacts": [
                            {
                                "path": relative,
                                "bytes": path.stat().st_size,
                                "sha256": sha256_file(path),
                            }
                            for relative, path in data.items()
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pipeline._validate_private_training_inputs(layout, config)
            data["train.jsonl"].write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                DataContractError, "differs from preparation"
            ):
                pipeline._validate_private_training_inputs(layout, config)

    def test_no_tracked_source_filename_uses_a_phase_name(self):
        ignored_roots = {".git", ".venv", ".uv-cache", "output", "__pycache__"}
        offenders = []
        for path in SOURCE_ROOT.rglob("*"):
            relative = path.relative_to(SOURCE_ROOT)
            if any(part in ignored_roots for part in relative.parts):
                continue
            if path.is_file() and "phase_" in path.name.lower():
                offenders.append(relative.as_posix())
        self.assertEqual(offenders, [])

    def test_non_table_entrypoints_are_removed(self):
        removed = (
            "build_kg.py",
            "eval_graph_rag.py",
            "smoke.py",
            "data/download_scierc.py",
            "provenance/inference_kg.py",
            "provenance/verify_triples_llm.py",
        )
        for relative in removed:
            with self.subTest(path=relative):
                self.assertFalse((SOURCE_ROOT / relative).exists())


if __name__ == "__main__":
    unittest.main()
