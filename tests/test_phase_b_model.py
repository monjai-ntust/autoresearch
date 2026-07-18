"""Synthetic regression tests for the B-05U candidate-generation adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from config import load_pipeline_config
from constants import PROTOCOL_ID
from model import generate_candidates, plan_training
from phase_b_io import DataContractError, atomic_write_json, atomic_write_jsonl
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


def _checkpoint_manifest() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1",
        "training_seed": 42,
        "base_model": "microsoft/deberta-large",
        "base_model_revision": "9a8befc6d3fbfa800e65f5279aa34d27eaf6d1b0",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_step": 1900,
        "split_manifest_sha256": "b" * 64,
        "max_span_width": 8,
        "context_between_spans": True,
    }


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
        sentences = layout.resolve("data-prepared/test.jsonl")
        checkpoint = layout.resolve("checkpoints/checkpoint-manifest.json")
        candidates = layout.resolve("predictions/test/candidates.jsonl")
        atomic_write_jsonl(sentences, [_sentence(split)])
        atomic_write_json(checkpoint, _checkpoint_manifest())
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
            plan_path = layout.resolve("predictions/test/generation-plan.jsonl")
            plan = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(plan), 1)
            self.assertEqual(set(plan[0]), _schema_required("model-generation-plan.schema.json"))
            self.assertEqual(plan[0]["token_count"], len(WORDS))
            for path in Path(temporary).rglob("*"):
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
            wrong = _checkpoint_manifest()
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

    def test_live_execution_is_not_implemented(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-live")
            layout.create()
            with self.assertRaises(DataContractError):
                plan_training(layout, self.config, execution_mode="live", training_seed=42)

    def test_train_refuses_to_overwrite(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-overwrite")
            layout.create()
            plan_training(layout, self.config, execution_mode="dry-run", training_seed=42)
            with self.assertRaises(DataContractError):
                plan_training(layout, self.config, execution_mode="dry-run", training_seed=42)


if __name__ == "__main__":
    unittest.main()
