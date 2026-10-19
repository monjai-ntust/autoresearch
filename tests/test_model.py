"""Synthetic regression tests for the B-05U candidate-generation adapter."""

from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from utils.test.fixtures import load_pipeline_config
from utils.common.constants import PROTOCOL_ID
from utils.encoder.cache import (
    MANIFEST_RELATIVE as HF_CACHE_MANIFEST_RELATIVE,
    write_cache_manifest,
)
from utils.encoder.model import (
    _load_checkpoint_manifest,
    _validate_checkpoint_run_identity,
    canonical_trainer_arguments,
    generate_candidates,
    plan_training,
    validate_prediction_artifacts,
)
from stages import encoder as encoder_stage
from stages.encoder import _validate_training_stage
from utils.common.artifact_io import (
    DataContractError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)
from utils.common.paths import RunLayout, discover_source_root
from utils.common.records import EntitySpan, StrictTriple, candidate_id_for
from utils.verifier.requests import _load_candidates, _load_sentences


SOURCE_ROOT = discover_source_root(Path(__file__))
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
        "schema_version": "encoder-checkpoint-1",
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1",
        "training_seed": 42,
        "base_model": "microsoft/deberta-large",
        "base_model_revision": "28c23d9eb93ea6cf11f845501ab7aeb2a497658b",
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
        "config": config.source_path,
        "trainer_sha256": sha256_file(layout.source_root / "stages/encoder.py"),
        "data_adapter_sha256": sha256_file(
            layout.source_root / "utils/encoder/data.py"
        ),
        "model_helper_sha256": sha256_file(
            layout.source_root / "utils/encoder/network.py"
        ),
        "model_cache_manifest": HF_CACHE_MANIFEST_RELATIVE,
        "model_cache_manifest_sha256": sha256_file(
            layout.resolve(HF_CACHE_MANIFEST_RELATIVE, must_exist=True)
        ),
        "model_cache_tree_sha256": json.loads(
            layout.resolve(HF_CACHE_MANIFEST_RELATIVE, must_exist=True).read_text(
                encoding="utf-8"
            )
        )["tree_sha256"],
        "source_commit": "3" * 40,
        "selected_metric": "development_strict_triple_f1",
        "selected_metric_value": 0.4,
        "restart_state_sha256": sha256_file(
            layout.resolve("checkpoints/seed-42/restart-state.pt")
        ),
    }


def _prepare_run_identities(layout: RunLayout, *, compatibility=True):
    atomic_write_json(layout.resolve('manifests/encoder-runtime.json'), {})
    atomic_write_json(layout.resolve('manifests/encoder-model-runtime.json'), {})
    atomic_write_json(
        layout.resolve("manifests/00-checkout-manifest.json"),
        {"status": "pass", "source": {"commit": "3" * 40}},
    )
    atomic_write_bytes(layout.source_root / "stages/encoder.py", b"trainer\n")
    atomic_write_bytes(
        layout.source_root / "utils/encoder/data.py", b"adapter\n"
    )
    atomic_write_bytes(
        layout.source_root / "utils/encoder/network.py", b"model\n"
    )
    atomic_write_bytes(
        layout.resolve("checkpoints/seed-42/restart-state.pt"), b"restart\n"
    )
    atomic_write_bytes(
        layout.resolve("inputs/huggingface/models--fixture/blobs/model.bin"),
        b"pinned model/tokenizer fixture\n",
    )
    write_cache_manifest(
        layout,
        model="microsoft/deberta-large",
        revision="28c23d9eb93ea6cf11f845501ab7aeb2a497658b",
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


class ModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config(SOURCE_ROOT, "resources/configs/pipeline.json")

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
            plan_path = layout.resolve("predictions/test/seed-42-generation-plan.jsonl")
            plan = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["token_count"], len(WORDS))
            self.assertNotIn("max_span_width", plan[0])

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
            validate_prediction_artifacts(
                sentences_path=sentences,
                prediction_ledger_path=ledger,
                candidates_path=candidates,
                training_seed=42,
                max_span_width=8,
                input_hashes=emitted[0]["input_hashes"],
            )
            emitted[0]["triple_confidence"] = 0.01
            atomic_write_jsonl(candidates, emitted)
            with self.assertRaisesRegex(
                DataContractError, "differs from deterministic prediction"
            ):
                validate_prediction_artifacts(
                    sentences_path=sentences,
                    prediction_ledger_path=ledger,
                    candidates_path=candidates,
                    training_seed=42,
                    max_span_width=8,
                    input_hashes=emitted[0]["input_hashes"],
                )

    def test_live_prediction_rejects_imported_ledger_without_a_producer(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "live-cache")
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            checkpoint_blob = layout.resolve("checkpoints/seed-42/checkpoint.pt")
            atomic_write_bytes(checkpoint_blob, b"checkpoint fixture")
            checkpoint_document = json.loads(checkpoint.read_text(encoding="utf-8"))
            checkpoint_document["checkpoint_sha256"] = sha256_file(checkpoint_blob)
            atomic_write_json(checkpoint, checkpoint_document)
            cache = layout.resolve("predictions/test/seed-42-prediction-ledger.jsonl")
            atomic_write_jsonl(cache, [_ledger_record()])
            with mock.patch(
                "utils.encoder.model._live_inference_records"
            ) as inference:
                with self.assertRaisesRegex(DataContractError, 'authenticated completion'):
                    generate_candidates(
                        layout, self.config, execution_mode="live",
                        sentences_path=sentences, checkpoint_manifest_path=checkpoint,
                        candidates_out_path=candidates, checkpoint_blob_path=checkpoint_blob,
                    )
            inference.assert_not_called()

    def test_live_prediction_resume_authenticates_original_ledger_before_reuse(self):
        from utils.encoder import model as implementation
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), 'prediction-resume')
            layout.create()
            sentences, checkpoint, candidates = self._prepare(layout)
            blob = layout.resolve('checkpoints/seed-42/checkpoint.pt')
            atomic_write_bytes(blob, b'fixture')
            document = json.loads(checkpoint.read_text())
            document['checkpoint_sha256'] = sha256_file(blob)
            atomic_write_json(checkpoint, document)
            manifest_path = layout.resolve('manifests/model-generate-candidates-live-seed-42-test.json')
            arguments = dict(execution_mode='live', sentences_path=sentences,
                checkpoint_manifest_path=checkpoint, candidates_out_path=candidates,
                checkpoint_blob_path=blob)
            def interrupt(path, value):
                if path == manifest_path:
                    raise RuntimeError('final receipt interrupted')
                atomic_write_json(path, value)
            with (mock.patch.object(implementation, '_live_inference_records', return_value=[_ledger_record()]) as inference,
                  mock.patch.object(implementation, 'atomic_write_json', side_effect=interrupt)):
                with self.assertRaisesRegex(RuntimeError, 'receipt interrupted'):
                    generate_candidates(layout, self.config, **arguments)
                inference.assert_called_once()
            ledger = layout.resolve('predictions/test/seed-42-prediction-ledger.jsonl')
            original = ledger.read_bytes()
            with mock.patch.object(implementation, '_live_inference_records') as inference:
                generate_candidates(layout, self.config, **arguments)
                inference.assert_not_called()
            self.assertEqual(ledger.read_bytes(), original)
            manifest_path.unlink()
            ledger.write_bytes(original.replace(b'0.9', b'0.8'))
            with mock.patch.object(implementation, '_live_inference_records') as inference:
                with self.assertRaisesRegex(DataContractError, 'authenticated completion'):
                    generate_candidates(layout, self.config, **arguments)
                inference.assert_not_called()
            ledger.unlink()
            with mock.patch.object(implementation, '_live_inference_records') as inference:
                with self.assertRaisesRegex(DataContractError, 'completed prediction ledger is missing'):
                    generate_candidates(layout, self.config, **arguments)
                inference.assert_not_called()

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

    def test_replay_continues_without_changing_existing_candidates(self):
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
            before = candidates.read_bytes()
            generate_candidates(
                    layout,
                    self.config,
                    execution_mode="replay",
                    sentences_path=sentences,
                    checkpoint_manifest_path=checkpoint,
                    candidates_out_path=candidates,
                    prediction_ledger_path=ledger,
                )
            self.assertEqual(candidates.read_bytes(), before)

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
        cls.config = load_pipeline_config(SOURCE_ROOT, "resources/configs/pipeline.json")

    def test_dry_run_emits_recipe_bound_training_plan(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-dry")
            layout.create()
            manifest = plan_training(
                layout, self.config, execution_mode="dry-run", training_seed=42
            )
            self.assertEqual(
                _validate_training_stage(
                    layout, self.config, 42, execution_mode="dry-run"
                ),
                manifest,
            )
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(manifest["training_seed"], 42)
            self.assertTrue(manifest["final_test_selection_forbidden"])
            self.assertEqual(manifest["expected_checkpoint_dir"], "checkpoints/seed-42")
            self.assertEqual(manifest['config'], self.config.source_path)
            self.assertNotIn('recipe', manifest)
            self.assertNotIn('split', manifest)
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
            atomic_write_bytes(layout.source_root / "stages/encoder.py", b"trainer\n")
            atomic_write_bytes(
                layout.source_root / "utils/encoder/data.py", b"adapter\n"
            )
            atomic_write_bytes(
                layout.source_root / "utils/encoder/network.py", b"model\n"
            )
            atomic_write_bytes(
                extracted, b"fixture annotation input\n"
            )
            layout.resolve(HF_CACHE_MANIFEST_RELATIVE, must_exist=True).unlink()
            layout.resolve(
                "inputs/huggingface/models--fixture/blobs/model.bin", must_exist=True
            ).unlink()
            observed_commands = []
            observed_trainer_commands = []

            def interrupted(command, *, cwd, check):
                observed_commands.append(command)
                trainer_command = canonical_trainer_arguments(layout, self.config, 42)
                observed_trainer_commands.append(trainer_command)
                restart = Path(
                    trainer_command.save_last_to
                )
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
                trainer_command = canonical_trainer_arguments(layout, self.config, 42)
                observed_trainer_commands.append(trainer_command)
                self.assertIsNotNone(trainer_command.resume_from)
                checkpoint = Path(trainer_command.save_best_to)
                restart = Path(trainer_command.save_last_to)
                summary = Path(trainer_command.run_summary_out)
                progress = Path(trainer_command.progress_log)
                cache_dir = Path(trainer_command.model_cache_dir)
                atomic_write_bytes(cache_dir / "models--fixture/blobs/model.bin", b"model\n")
                write_cache_manifest(layout, model=self.config.value['training']['base_model'],
                                     revision=self.config.value['training']['base_model_revision'])
                from utils.encoder.runtime import restart_identity
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
                        "completed_steps": self.config.value['training']['max_steps'],
                        "run_identity": restart_identity(layout.resolve('data-prepared')),
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
            self.assertTrue(manifest["resume"]["resumed"])
            self.assertEqual(observed_commands[-1][3], "_train-encoder")
            self.assertNotIn("--prepared-dir", observed_commands[-1])
            self.assertIsNotNone(observed_trainer_commands[-1].model_cache_dir)
            checkpoint_manifest = layout.resolve(
                "checkpoints/seed-42/checkpoint-manifest.json"
            )
            checkpoint_value = json.loads(
                checkpoint_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint_value["schema_version"],
                "encoder-checkpoint-1",
            )
            self.assertNotIn("trainer_sha256", checkpoint_value)
            self.assertNotIn("data_adapter_sha256", checkpoint_value)
            self.assertNotIn("model_helper_sha256", checkpoint_value)
            self.assertFalse({"base_model", "base_model_revision", "max_span_width",
                              "context_between_spans", "selected_metric", "config_sha256"}
                             & checkpoint_value.keys())
            self.assertEqual(checkpoint_value["checkpoint_step"], 100)
            self.assertEqual(
                _validate_training_stage(
                    layout, self.config, 42, execution_mode="live"
                ),
                manifest,
            )

            # Consume the actual fresh writer's compact checkpoint manifest.
            # Its model identity comes from configuration; cache binding still
            # authenticates those configured model bytes before generation.
            config = deepcopy(self.config)
            config.value["training_seeds"] = [42]
            candidates = layout.resolve("predictions/dev/development-candidates.jsonl")
            atomic_write_bytes(candidates, b"")
            atomic_write_json(layout.resolve("predictions/dev/candidate-index.json"), {})

            def validate_generation_inputs(current_layout, current_config, **kwargs):
                checkpoint = _load_checkpoint_manifest(checkpoint_manifest, current_config)
                _validate_checkpoint_run_identity(current_layout, current_config, checkpoint)

            with (
                mock.patch.object(encoder_stage, "_resume_generation_stage", side_effect=validate_generation_inputs) as generate,
                mock.patch.object(encoder_stage, "_validate_candidate_assembly"),
            ):
                identity, _hardware, actual_candidates = encoder_stage.train_and_generate_development(layout, config)
                self.assertEqual(identity, {
                    "base_model": config.value["training"]["base_model"],
                    "base_model_revision": config.value["training"]["base_model_revision"],
                })
                self.assertEqual(actual_candidates, candidates)
                generate.assert_called_once_with(layout, config, seed=42, split="development")

                model_blob = layout.resolve("inputs/huggingface/models--fixture/blobs/model.bin")
                original_blob = model_blob.read_bytes()
                atomic_write_bytes(model_blob, b"changed model bytes")
                with self.assertRaisesRegex(DataContractError, "inventory drifted"):
                    encoder_stage.train_and_generate_development(layout, config)
                atomic_write_bytes(model_blob, original_blob)
                config.value["training"]["base_model"] = "different-model"
                with self.assertRaisesRegex(DataContractError, "model/revision differs"):
                    encoder_stage.train_and_generate_development(layout, config)

            atomic_write_bytes(
                layout.resolve("checkpoints/seed-42/checkpoint.pt"), b"tampered"
            )
            with self.assertRaisesRegex(DataContractError, "producer output changed"):
                _validate_training_stage(
                    layout, self.config, 42, execution_mode="live"
                )

    def test_completed_worker_can_finalize_after_exact_cache_restoration_without_retraining(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-completed-cache-drift")
            layout.create()
            _prepare_run_identities(layout, compatibility=False)
            cache_manifest_path = layout.resolve(HF_CACHE_MANIFEST_RELATIVE, must_exist=True)
            cache_manifest_path.unlink()
            addition = layout.resolve("inputs/huggingface/unexpected-generated-file")
            durable = {}

            def completed(command, *, cwd, check):
                args = canonical_trainer_arguments(layout, self.config, 42)
                write_cache_manifest(layout,
                    model=self.config.value["training"]["base_model"],
                    revision=self.config.value["training"]["base_model_revision"])
                from utils.encoder.runtime import restart_identity
                atomic_write_bytes(Path(args.save_best_to), b"selected checkpoint")
                atomic_write_bytes(Path(args.save_last_to), b"completed restart")
                atomic_write_bytes(Path(args.progress_log), b"BEST DEV fixture\n")
                atomic_write_json(Path(args.run_summary_out), {
                    "status": "completed", "seed": 42, "test_evaluated": False,
                    "completed_steps": self.config.value["training"]["max_steps"],
                    "selected_step": 100, "selected_metrics": {"ner_f1": 0.5, "triple_f1": 0.4},
                    "run_identity": restart_identity(layout.resolve("data-prepared")),
                    "elapsed_seconds": 123.25, "resumed_from": None,
                })
                for path in (Path(args.save_best_to), Path(args.save_last_to),
                             Path(args.progress_log), Path(args.run_summary_out), cache_manifest_path):
                    durable[path] = path.read_bytes()
                atomic_write_bytes(addition, b"synthetic unexpected cache addition")

            worker = mock.Mock(side_effect=completed)
            with self.assertRaisesRegex(DataContractError, "differs from its frozen manifest"):
                plan_training(layout, self.config, execution_mode="live", training_seed=42,
                              command_runner=worker)
            worker.assert_called_once()
            self.assertFalse(layout.resolve("manifests/model-train-live-seed-42.json").exists())
            self.assertFalse(layout.resolve("checkpoints/seed-42/checkpoint-manifest.json").exists())
            for path, original in durable.items():
                self.assertEqual(path.read_bytes(), original)

            never_train = mock.Mock(side_effect=AssertionError("completed worker must not run again"))
            with self.assertRaisesRegex(DataContractError, "inventory drifted"):
                plan_training(layout, self.config, execution_mode="live", training_seed=42,
                              command_runner=never_train)
            never_train.assert_not_called()

            # Restore only our synthetic addition. The frozen inventory and
            # all completed scientific outputs are retained byte for byte.
            addition.unlink()
            manifest = plan_training(layout, self.config, execution_mode="live", training_seed=42,
                                     command_runner=never_train)
            never_train.assert_not_called()
            self.assertEqual(manifest["status"], "completed")
            self.assertFalse(manifest["resume"]["resumed"])
            for path, original in durable.items():
                self.assertEqual(path.read_bytes(), original)
            self.assertEqual(_validate_training_stage(layout, self.config, 42, execution_mode="live"), manifest)

    def test_train_refuses_to_overwrite(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "train-overwrite")
            layout.create()
            plan_training(layout, self.config, execution_mode="dry-run", training_seed=42)
            with self.assertRaises(DataContractError):
                plan_training(layout, self.config, execution_mode="dry-run", training_seed=42)


if __name__ == "__main__":
    unittest.main()
