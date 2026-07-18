"""Synthetic regression tests for the B-05U candidate-generation adapter."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from config import load_pipeline_config
from constants import PROTOCOL_ID
from model import generate_candidates, plan_training
from phase_b_io import (
    DataContractError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)
from paths import RunLayout, discover_source_root
from records import EntitySpan, StrictTriple, candidate_id_for
from verifier import _load_candidates, _load_sentences


SOURCE_ROOT = discover_source_root(Path(__file__))
SCHEMA_ROOT = SOURCE_ROOT / "schemas" / "phase_b"
EXAMPLE_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "phase-b-model-test-a"))
WORDS = ["Door", "shall", "be", "fire", "rated", "and", "steel", "framed", "."]


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-b-model-", dir=output_root)


def _sentence(split: str = "test") -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": "CODE-ACCORD-v1.0.0",
        "example_id": EXAMPLE_ID,
        "source_document_id": "doc-a",
        "source_country": "UK",
        "split": split,
        "content": " ".join(WORDS),
        "processed_content": " ".join(WORDS),
        "words": list(WORDS),
        "entities": [],
        "relations": [],
    }


def _checkpoint_manifest(layout: RunLayout, config) -> dict:
    acquisition = layout.resolve("manifests/02-input-acquisition-manifest.json")
    preparation = layout.resolve("manifests/03-data-preparation-manifest.json")
    split = layout.resolve("data-prepared/split-manifest.json")
    train = layout.resolve("data-prepared/train.jsonl")
    development = layout.resolve("data-prepared/development.jsonl")
    compatibility = layout.resolve("audit/model-training-dataset-compatibility.json")
    return {
        "schema_version": "phase-b-model-checkpoint-manifest-2.0",
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1",
        "training_seed": 42,
        "base_model": "microsoft/deberta-large",
        "base_model_revision": "9a8befc6d3fbfa800e65f5279aa34d27eaf6d1b0",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_step": 1900,
        "split_manifest_sha256": sha256_file(split),
        "max_span_width": 8,
        "context_between_spans": True,
        "archive_sha256": "c" * 64,
        "acquisition_manifest_sha256": sha256_file(acquisition),
        "annotation_bundle_sha256": "d" * 64,
        "prepared_dataset_tree_sha256": "e" * 64,
        "train_jsonl_sha256": sha256_file(train),
        "development_jsonl_sha256": sha256_file(development),
        "config_sha256": sha256_file(config.path),
        "trainer_sha256": sha256_file(layout.source_root / "train_span.py"),
        "data_adapter_sha256": sha256_file(layout.source_root / "data/code_accord.py"),
        "model_helper_sha256": sha256_file(
            layout.source_root / "models/bert_kg_encoder.py"
        ),
        "source_commit": "3" * 40,
        "selected_metric": "development_strict_triple_f1",
        "selected_metric_value": 0.4,
        "restart_state_sha256": sha256_file(
            layout.resolve("checkpoints/seed-42/restart-state.pt")
        ),
        "dataset_compatibility_report": "audit/model-training-dataset-compatibility.json",
        "dataset_compatibility_sha256": sha256_file(compatibility),
        "historical_comparability": "partial_match_full_legacy_equivalence_unavailable",
    }


def _prepare_run_identities(layout: RunLayout, *, compatibility=True):
    atomic_write_json(
        layout.resolve("manifests/00-checkout-manifest.json"),
        {"status": "pass", "source": {"commit": "3" * 40}},
    )
    atomic_write_bytes(layout.source_root / "train_span.py", b"trainer\n")
    atomic_write_bytes(layout.source_root / "data/code_accord.py", b"adapter\n")
    atomic_write_bytes(
        layout.source_root / "models/bert_kg_encoder.py", b"model\n"
    )
    atomic_write_bytes(
        layout.resolve("checkpoints/seed-42/restart-state.pt"), b"restart\n"
    )
    acquisition = layout.resolve("manifests/02-input-acquisition-manifest.json")
    atomic_write_json(acquisition, {"archive": {"sha256": "c" * 64}})
    atomic_write_json(
        layout.resolve("manifests/03-data-preparation-manifest.json"),
        {
            "archive_sha256": "c" * 64,
            "acquisition_manifest_sha256": sha256_file(acquisition),
            "annotation_bundle_sha256": "d" * 64,
            "dataset_tree_sha256": "e" * 64,
            "byte_identical_independent_materializations": True,
        },
    )
    atomic_write_json(layout.resolve("data-prepared/split-manifest.json"), {"split_id": "CODE-SPLIT-1"})
    atomic_write_jsonl(layout.resolve("data-prepared/train.jsonl"), [_sentence("train")])
    atomic_write_jsonl(
        layout.resolve("data-prepared/development.jsonl"), [_sentence("development")]
    )
    if compatibility:
        atomic_write_json(
            layout.resolve("audit/model-training-dataset-compatibility.json"),
            {"status": "partial_match_full_legacy_equivalence_unavailable"},
        )


def _ledger_record(relations: list[dict] | None = None) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "example_id": EXAMPLE_ID,
        "training_seed": 42,
        "predicted_spans": [
            {"start": 0, "end": 0, "type": "Object", "confidence": 0.9},
            {"start": 0, "end": 1, "type": "Object", "confidence": 0.4},
            {"start": 3, "end": 4, "type": "Quality", "confidence": 0.8},
            {"start": 6, "end": 7, "type": "Value", "confidence": 0.7},
        ],
        "predicted_relations": relations
        if relations is not None
        else [
            {
                "head": {"start": 0, "end": 0},
                "tail": {"start": 3, "end": 4},
                "relation": "selection",
                "re_confidence": 0.5,
            },
            {
                "head": {"start": 0, "end": 0},
                "tail": {"start": 6, "end": 7},
                "relation": "necessity",
                "re_confidence": 1.0,
            },
            {
                "head": {"start": 0, "end": 0},
                "tail": {"start": 3, "end": 4},
                "relation": "selection",
                "re_confidence": 0.9,
            },
        ],
    }


def _expected_candidate_id(head: tuple, relation: str, tail: tuple) -> str:
    triple = StrictTriple(
        EXAMPLE_ID,
        EntitySpan(head[0], head[1], head[2]),
        relation,
        EntitySpan(tail[0], tail[1], tail[2]),
    )
    return candidate_id_for(42, triple)


def _schema_required(name: str) -> set[str]:
    return set(json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))["required"])


class ModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")

    def _prepare(self, layout: RunLayout, split: str = "test"):
        _prepare_run_identities(layout)
        sentences = layout.resolve("data-prepared/test.jsonl")
        checkpoint = layout.resolve("checkpoints/checkpoint-manifest.json")
        candidates = layout.resolve("predictions/test/candidates.jsonl")
        atomic_write_jsonl(sentences, [_sentence(split)])
        atomic_write_json(checkpoint, _checkpoint_manifest(layout, self.config))
        return sentences, checkpoint, candidates

    def test_dry_run_plans_without_producing_a_candidate(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "dry")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            manifest = generate_candidates(
                layout,
                self.config,
                execution_mode="dry-run",
                sentences_path=sentences,
                checkpoint_manifest_path=checkpoint,
                candidates_out_path=candidates,
            )
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(manifest["candidate_count"], 0)
            self.assertIsNone(manifest["candidates_output"])
            self.assertFalse(candidates.exists())
            self.assertEqual(set(manifest), _schema_required("model-generation-manifest.schema.json"))
            plan_path = layout.resolve("predictions/test/seed-42-generation-plan.jsonl")
            plan = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(plan), 1)
            self.assertEqual(set(plan[0]), _schema_required("model-generation-plan.schema.json"))
            self.assertEqual(plan[0]["token_count"], len(WORDS))
            for path in layout.run_root.rglob("*"):
                if path.is_file():
                    self.assertTrue(path.resolve().is_relative_to(layout.run_root.resolve()))

    def test_replay_reproduces_greedy_confidence_and_dedup(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "replay")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            ledger = layout.resolve("predictions/test/prediction-ledger.jsonl")
            atomic_write_jsonl(ledger, [_ledger_record()])
            manifest = generate_candidates(
                layout,
                self.config,
                execution_mode="replay",
                sentences_path=sentences,
                checkpoint_manifest_path=checkpoint,
                candidates_out_path=candidates,
                prediction_ledger_path=ledger,
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["selected_entity_count"], 3)
            self.assertEqual(manifest["candidate_count"], 2)
            self.assertEqual(manifest["duplicate_candidate_collapsed"], 1)
            self.assertEqual(set(manifest), _schema_required("model-generation-manifest.schema.json"))

            emitted = [
                json.loads(line)
                for line in candidates.read_text(encoding="utf-8").splitlines()
            ]
            by_id = {record["candidate_id"]: record for record in emitted}
            selection_id = _expected_candidate_id((0, 0, "Object"), "selection", (3, 4, "Quality"))
            necessity_id = _expected_candidate_id((0, 0, "Object"), "necessity", (6, 7, "Value"))
            self.assertEqual(set(by_id), {selection_id, necessity_id})
            # min(0.9, 0.8) * 0.9 keeps the higher-confidence duplicate.
            self.assertEqual(by_id[selection_id]["triple_confidence"], 0.72)
            # min(0.9, 0.7) * 1.0
            self.assertEqual(by_id[necessity_id]["triple_confidence"], 0.7)
            self.assertEqual(by_id[selection_id]["head"]["text"], "Door")
            self.assertEqual(by_id[selection_id]["tail"]["text"], "fire rated")

            # The emitted file must round-trip through the downstream loader.
            loaded_sentences = _load_sentences(sentences)
            rows = _load_candidates(candidates, loaded_sentences)
            self.assertEqual(len(rows), 2)

    def test_replay_rejects_relation_over_unselected_span(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "orphan")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            ledger = layout.resolve("predictions/test/prediction-ledger.jsonl")
            atomic_write_jsonl(
                ledger,
                [
                    _ledger_record(
                        relations=[
                            {
                                "head": {"start": 0, "end": 1},
                                "tail": {"start": 3, "end": 4},
                                "relation": "selection",
                                "re_confidence": 0.5,
                            }
                        ]
                    )
                ],
            )
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="replay",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                    prediction_ledger_path=ledger,
                )

    def test_span_exceeding_max_span_width_is_rejected(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "wide")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            ledger = layout.resolve("predictions/test/prediction-ledger.jsonl")
            record = _ledger_record(relations=[])
            # A 9-token span (0-8) exceeds the recipe max_span_width of 8 while
            # still lying inside the sentence's token boundaries.
            record["predicted_spans"].append(
                {"start": 0, "end": 8, "type": "Object", "confidence": 0.5}
            )
            record["predicted_spans"].sort(
                key=lambda span: (span["start"], span["end"], span["type"])
            )
            atomic_write_jsonl(ledger, [record])
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="replay",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                    prediction_ledger_path=ledger,
                )

    def test_execution_argument_guards(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "guards")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            ledger = layout.resolve("predictions/test/prediction-ledger.jsonl")
            atomic_write_jsonl(ledger, [_ledger_record()])
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="replay",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                    prediction_ledger_path=None,
                )
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="dry-run",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                    prediction_ledger_path=ledger,
                )

    def test_live_argument_guards(self):
        # These validations run before any torch import, so they are offline.
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "live-guards")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            # live requires a checkpoint blob
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="live",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                )
            # a checkpoint blob is accepted only by live execution
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="dry-run",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                    checkpoint_blob_path=checkpoint,
                )

    def test_replay_refuses_to_overwrite_existing_candidates(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "no-overwrite")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            ledger = layout.resolve("predictions/test/prediction-ledger.jsonl")
            atomic_write_jsonl(ledger, [_ledger_record()])
            generate_candidates(
                layout,
                self.config,
                execution_mode="replay",
                sentences_path=sentences,
                checkpoint_manifest_path=checkpoint,
                candidates_out_path=candidates,
                prediction_ledger_path=ledger,
            )
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="replay",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                    prediction_ledger_path=ledger,
                )

    def test_checkpoint_recipe_mismatch_is_rejected(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "recipe")
            layout.create()
            sentences, _checkpoint, candidates = self._prepare(layout)
            wrong = _checkpoint_manifest(layout, self.config)
            wrong["base_model"] = "bert-base-uncased"
            checkpoint_wrong = layout.resolve("checkpoints/wrong-manifest.json")
            atomic_write_json(checkpoint_wrong, wrong)
            with self.assertRaises(DataContractError):
                generate_candidates(
                    layout,
                    self.config,
                    execution_mode="dry-run",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint_wrong,
                    candidates_out_path=candidates,
                )


class ModelTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")

    def test_dry_run_emits_recipe_bound_training_plan(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-dry")
            layout.create()
            manifest = plan_training(
                layout, self.config, execution_mode="dry-run", training_seed=42
            )
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(manifest["training_seed"], 42)
            self.assertTrue(manifest["final_test_selection_forbidden"])
            self.assertEqual(manifest["live_execution_status"], "gated_external_accelerator")
            self.assertEqual(manifest["expected_checkpoint_dir"], "checkpoints/seed-42")
            # The plan is bound to the exact frozen recipe and split.
            self.assertEqual(manifest["recipe"], self.config.value["training"])
            self.assertEqual(manifest["split"], self.config.value["split"])
            self.assertEqual(set(manifest), _schema_required("model-train-manifest.schema.json"))
            for path in Path(temporary).rglob("*"):
                if path.is_file():
                    self.assertTrue(path.resolve().is_relative_to(layout.run_root.resolve()))

    def test_invalid_seed_is_rejected(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-badseed")
            layout.create()
            with self.assertRaises(DataContractError):
                plan_training(layout, self.config, execution_mode="dry-run", training_seed=7)

    def test_live_execution_requires_bootstrap_artifacts(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-live")
            layout.create()
            with self.assertRaises(DataContractError):
                plan_training(layout, self.config, execution_mode="live", training_seed=42)

    def test_live_execution_resumes_and_emits_schema_bound_checkpoint(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-live-complete")
            layout.create()
            _prepare_run_identities(layout, compatibility=False)
            atomic_write_json(
                layout.resolve("manifests/00-checkout-manifest.json"),
                {
                    "status": "pass",
                    "source": {"commit": "5" * 40},
                },
            )
            extracted = layout.resolve(
                "inputs/extracted/CODE-ACCORD-v1.0.0-annotations/"
                "annotated_data/entities/train.csv"
            )
            historical = SOURCE_ROOT / "data/code_accord/entities/train.csv"
            historical_copy = layout.source_root / "data/code_accord/entities/train.csv"
            atomic_write_bytes(
                historical_copy, historical.read_bytes().replace(b"\r\n", b"\n")
            )
            atomic_write_bytes(layout.source_root / "train_span.py", b"trainer\n")
            atomic_write_bytes(layout.source_root / "data/code_accord.py", b"adapter\n")
            atomic_write_bytes(
                layout.source_root / "models/bert_kg_encoder.py", b"model\n"
            )
            atomic_write_bytes(
                extracted, historical.read_bytes().replace(b"\r\n", b"\n")
            )
            observed_commands = []

            def interrupted(command, *, cwd, check):
                observed_commands.append(command)
                restart = Path(command[command.index("--save-last-to") + 1])
                atomic_write_bytes(restart, b"partial restart")
                raise subprocess.CalledProcessError(9, command)

            with self.assertRaises(DataContractError):
                plan_training(
                    layout,
                    self.config,
                    execution_mode="live",
                    training_seed=42,
                    command_runner=interrupted,
                )

            def completed(command, *, cwd, check):
                observed_commands.append(command)
                self.assertIn("--resume-from", command)
                checkpoint = Path(command[command.index("--save-best-to") + 1])
                restart = Path(command[command.index("--save-last-to") + 1])
                summary = Path(command[command.index("--run-summary-out") + 1])
                progress = Path(command[command.index("--progress-log") + 1])
                atomic_write_bytes(checkpoint, b"checkpoint")
                atomic_write_bytes(restart, b"restart")
                atomic_write_bytes(progress, b"completed\n")
                atomic_write_json(
                    summary,
                    {
                        "status": "completed",
                        "canonical_mode": True,
                        "seed": 42,
                        "test_evaluated": False,
                        "selected_step": 100,
                        "selected_metrics": {"ner_f1": 0.5, "triple_f1": 0.4},
                        "resumed_from": str(restart),
                        "environment": {
                            "python": "3.10.20",
                            "torch": "2.9.1+cu130",
                            "cuda_runtime": "13.0",
                            "cuda_available": True,
                            "cuda_device": "fixture",
                        },
                    },
                )

            manifest = plan_training(
                layout,
                self.config,
                execution_mode="live",
                training_seed=42,
                command_runner=completed,
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["historical_comparability"],
                "partial_match_full_legacy_equivalence_unavailable",
            )
            self.assertTrue(manifest["resume"]["resumed"])
            self.assertIn("--canonical-mode", observed_commands[-1])
            self.assertIn("--skip-test-eval", observed_commands[-1])
            checkpoint_manifest = layout.resolve(
                "checkpoints/seed-42/checkpoint-manifest.json"
            )
            checkpoint_value = json.loads(
                checkpoint_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(checkpoint_value),
                _schema_required("model-checkpoint-manifest.schema.json"),
            )
            self.assertEqual(checkpoint_value["checkpoint_step"], 100)
            self.assertFalse(
                json.loads(
                    layout.resolve(
                        "audit/model-training-dataset-compatibility.json"
                    ).read_text(encoding="utf-8")
                )["historical_statistical_continuity_claim_permitted"]
            )

    def test_train_refuses_to_overwrite(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-overwrite")
            layout.create()
            plan_training(layout, self.config, execution_mode="dry-run", training_seed=42)
            with self.assertRaises(DataContractError):
                plan_training(layout, self.config, execution_mode="dry-run", training_seed=42)


if __name__ == "__main__":
    unittest.main()
