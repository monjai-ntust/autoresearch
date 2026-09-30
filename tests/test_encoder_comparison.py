"""Focused contracts for the Phase G historical-derived encoder path."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from stages import encoder_comparison as comparison_stage
from stages import preparation as preparation_stage
from utils.common.artifact_io import (
    DataContractError,
    atomic_write_bytes,
    atomic_write_json,
    load_json,
    sha256_file,
)
from utils.common.paths import RunLayout, discover_source_root
from utils.encoder import data as canonical_data
from utils.encoder.cache import write_cache_manifest
from utils.encoder_comparison import data as comparison_data
from utils.encoder_comparison.config import (
    BASE_PARAMETER_COUNT_SOURCE,
    DEFAULT_CONFIG,
    HISTORICAL_BLOBS,
    HISTORICAL_COMMIT,
    _validate_profile,
    bind_selection,
    load_comparison_config,
    validate_selection,
)
from utils.encoder_comparison.network import HistoricalKGExtractor
from utils.encoder_comparison.trainer import pipeline_comparison_boost


SOURCE_ROOT = discover_source_root(Path(__file__))
ENTRYPOINT = SOURCE_ROOT / "encoder_comparison.py"
PIPELINE = SOURCE_ROOT / "pipeline.py"
PHASE_F_BASELINE = "5db3bce"
PHASE_F_ENCODER_FLOW_PATHS = (
    "resources/configs/experiment_matrix.json",
    "resources/configs/pipeline.json",
    "resources/configs/section5_evidence.json",
    "resources/contracts/table2.json",
    "resources/prompts/code-verifier-1-corrective.txt",
    "resources/prompts/code-verifier-1-simple.txt",
    "resources/prompts/code-verifier-1-system.txt",
    "resources/schemas/verifier-corrective-response.schema.json",
    "resources/schemas/verifier-simple-response.schema.json",
    "stages/encoder.py",
    "stages/evaluation.py",
    "stages/preparation.py",
    "stages/verifier.py",
    "utils/common/artifact_io.py",
    "utils/common/config.py",
    "utils/common/constants.py",
    "utils/common/environment.py",
    "utils/common/paths.py",
    "utils/common/records.py",
    "utils/encoder/cache.py",
    "utils/encoder/data.py",
    "utils/encoder/model.py",
    "utils/encoder/network.py",
    "utils/evaluation/metrics.py",
    "utils/evaluation/publication.py",
    "utils/evaluation/reconciliation.py",
    "utils/evaluation/scoring.py",
    "utils/evaluation/statistics.py",
    "utils/preparation/acquisition.py",
    "utils/preparation/preparation.py",
    "utils/preparation/split.py",
    "utils/rag/graph.py",
    "utils/verifier/pilot.py",
    "utils/verifier/threshold.py",
    "utils/verifier/verifier.py",
)


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-g-", dir=output_root)


class _Encoding(dict):
    def __init__(self):
        super().__init__(input_ids=[101, 11, 12, 13, 102], attention_mask=[1] * 5)

    def word_ids(self):
        return [None, 0, 1, 1, None]


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, *_args, **_kwargs):
        return _Encoding()


class _FrozenBackbone(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(32, hidden_size)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, *, inputs_embeds, attention_mask):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class EncoderComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_comparison_config(SOURCE_ROOT, DEFAULT_CONFIG)

    def test_historical_source_crosswalk_resolves_to_frozen_git_blobs(self):
        self.assertEqual(self.config.value["historical_source"]["commit"], HISTORICAL_COMMIT)
        for name, (path, expected_blob) in HISTORICAL_BLOBS.items():
            actual_blob = subprocess.run(
                ["git", "rev-parse", f"{HISTORICAL_COMMIT}:{path}"],
                cwd=SOURCE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(actual_blob, expected_blob, name)

    def test_original_phase_f_encoder_and_shared_flow_is_byte_identical(self):
        for path in PHASE_F_ENCODER_FLOW_PATHS:
            baseline = subprocess.run(
                ["git", "rev-parse", f"{PHASE_F_BASELINE}:{path}"],
                cwd=SOURCE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            current = subprocess.run(
                ["git", "hash-object", path],
                cwd=SOURCE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(current, baseline, path)

    def test_profile_registry_is_finite_pinned_and_cli_validates_offline(self):
        self.assertEqual(
            set(self.config.profiles),
            {"bert-base-uncased", "deberta-base-v1", "deberta-large-v1"},
        )
        for profile in self.config.profiles.values():
            self.assertRegex(profile.revision, r"^[0-9a-f]{40}$")
            self.assertRegex(profile.weight_sha256, r"^[0-9a-f]{64}$")
            self.assertEqual(
                profile.base_parameter_count_source, BASE_PARAMETER_COUNT_SOURCE
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ENTRYPOINT),
                    "validate-profile",
                    "--profile",
                    profile.profile_id,
                ],
                cwd=SOURCE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(profile.revision, completed.stdout)

    def test_only_populated_table1_arms_and_two_historical_recipes_are_registered(self):
        self.assertEqual(
            set(self.config.recipes),
            {"historical-common-base", "historical-a20-a21-a12"},
        )
        self.assertEqual(
            set(self.config.arms),
            {
                "bert-base-common",
                "deberta-base-common",
                "deberta-large-common",
                "deberta-large-a20-a21-a12",
            },
        )
        base = self.config.recipes["historical-common-base"].value
        self.assertEqual(base["bio_loss_weight"], 0.1)
        self.assertEqual(base["label_smoothing"], 0.0)
        self.assertFalse(base["context_between_spans"])
        self.assertEqual(base["comparison_boost"]["schedule"], "disabled")
        self.assertEqual(base["comparison_boost"]["initial"], 1.0)
        method = self.config.recipes["historical-a20-a21-a12"].value
        self.assertEqual(method["comparison_boost"]["schedule"], "pipeline-a20")
        self.assertEqual(method["re_no_rel_weight"], 1.0)
        self.assertEqual(
            self.config.value["statistical_protocol"]["historical_statistics_role"],
            "provenance_only",
        )

        for arm_id in self.config.arms:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ENTRYPOINT),
                    "validate-arm",
                    "--arm",
                    arm_id,
                ],
                cwd=SOURCE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(arm_id, completed.stdout)

    def test_unknown_profile_and_mutable_revision_fail_closed(self):
        with self.assertRaisesRegex(DataContractError, "unknown approved Table 1 arm"):
            self.config.select_arm("not-reviewed")
        with self.assertRaisesRegex(DataContractError, "not one approved"):
            self.config.select("bert-base-uncased", "historical-a20-a21-a12")
        profile = self.config.profiles["bert-base-uncased"].manifest()
        profile["revision"] = "main"
        with self.assertRaisesRegex(DataContractError, "40 lowercase hex"):
            _validate_profile(profile, 0)

    def test_entrypoint_is_thin_and_does_not_enter_pipeline_dispatch(self):
        entry_source = ENTRYPOINT.read_text(encoding="utf-8")
        definitions = [
            node.name
            for node in ast.parse(entry_source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(definitions, ["_parser", "main"])
        self.assertLessEqual(len(entry_source.splitlines()), 140)
        pipeline_source = PIPELINE.read_text(encoding="utf-8")
        self.assertNotIn("encoder_comparison", pipeline_source)

    def test_stage_reuses_current_preparation_and_data_cache_contracts(self):
        self.assertIs(comparison_stage.run_preparation, preparation_stage.run)
        self.assertIs(
            comparison_stage.generate_candidates,
            __import__("utils.encoder.model", fromlist=["generate_candidates"]).generate_candidates,
        )
        examples = [
            {
                "example_id": "example-a",
                "words": ["fire", "door"],
                "ner": [(0, 1, "Object")],
                "relations": [],
            }
        ]
        canonical_item = canonical_data.CodeAccordDataset(
            examples, _Tokenizer(), 16
        )[0]
        self.assertEqual(
            set(canonical_item),
            {
                "input_ids",
                "attention_mask",
                "word_ids",
                "gold_entities",
                "gold_relations",
                "num_words",
            },
        )
        item = comparison_data.HistoricalCodeAccordDataset(
            examples, _Tokenizer(), 16
        )[0]
        self.assertEqual(item["input_ids"], [101, 11, 12, 13, 102])
        self.assertEqual(item["attention_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(item["word_ids"], [None, 0, 1, 1, None])
        self.assertEqual(item["gold_entities"], [(0, 1, "Object")])
        self.assertEqual(item["gold_relations"], [])
        self.assertEqual(item["num_words"], 2)
        self.assertEqual(item["ner_labels"][0], -100)
        self.assertEqual(
            item["ner_labels"][1], comparison_data.BIO_TAG2ID["B-Object"]
        )
        self.assertEqual(
            item["ner_labels"][2], comparison_data.BIO_TAG2ID["I-Object"]
        )
        self.assertEqual(item["ner_labels"][3], -100)
        batch = comparison_data.collate_fn([item])
        self.assertTrue(
            torch.equal(batch["input_ids"], torch.tensor([[101, 11, 12, 13, 102]]))
        )
        self.assertTrue(
            torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1, 1, 1]]))
        )
        self.assertEqual(batch["word_ids"], [[None, 0, 1, 1, None]])
        self.assertEqual(batch["gold_entities"], [[(0, 1, "Object")]])
        self.assertEqual(batch["gold_relations"], [[]])
        self.assertEqual(batch["num_words"], [2])
        self.assertEqual(batch["example_ids"], ["example-a"])
        self.assertEqual(batch["words"], [["fire", "door"]])

    def test_historical_model_uses_profile_hidden_size_and_a12_three_h_head(self):
        calls = []

        def load_model(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return _FrozenBackbone(hidden_size=8)

        with mock.patch(
            "utils.encoder_comparison.network.AutoModel.from_pretrained",
            side_effect=load_model,
        ):
            model = HistoricalKGExtractor(
                "fixture/model",
                model_revision="a" * 40,
                model_cache_dir="fixture-cache",
                model_local_files_only=True,
                expected_hidden_size=8,
                num_bio_tags=5,
                num_relations=4,
                num_entity_types=3,
                max_span_width=8,
                context_between_spans=True,
            )
            self.assertEqual(model.re_head[0].in_features, 24)
            self.assertEqual(calls[0][1]["revision"], "a" * 40)
            self.assertIs(calls[0][1]["use_safetensors"], False)
            with self.assertRaisesRegex(ValueError, "hidden size"):
                HistoricalKGExtractor(
                    "fixture/model",
                    model_revision="a" * 40,
                    model_cache_dir="fixture-cache",
                    model_local_files_only=True,
                    expected_hidden_size=12,
                    num_bio_tags=5,
                    num_relations=4,
                    num_entity_types=3,
                    max_span_width=8,
                    context_between_spans=False,
                )

    def test_pipeline_a20_middle_value_lasts_exactly_one_batch(self):
        at_gate = pipeline_comparison_boost(
            step=1000,
            max_steps=3500,
            initial=5.0,
            end=2.0,
            adaptive_step=1000,
            threshold_low=0.35,
            threshold_high=0.40,
            middle=3.5,
            adaptive_triggered=False,
            adaptive_switched=False,
            adaptive_triple_f1=0.36,
        )
        self.assertEqual(at_gate, (3.5, True, True))
        after_gate = pipeline_comparison_boost(
            step=1001,
            max_steps=3500,
            initial=5.0,
            end=2.0,
            adaptive_step=1000,
            threshold_low=0.35,
            threshold_high=0.40,
            middle=3.5,
            adaptive_triggered=at_gate[1],
            adaptive_switched=at_gate[2],
        )
        self.assertEqual(after_gate, (2.0, True, True))

    def test_run_binding_rejects_other_profile_and_cross_run_manifest(self):
        with _temporary_output_directory() as temporary:
            source = Path(temporary)
            first = RunLayout(source, "first")
            second = RunLayout(source, "second")
            first.create()
            second.create()
            large = self.config.select_arm("deberta-large-a20-a21-a12")
            base = self.config.select_arm("deberta-base-common")
            bind_selection(first, large)
            with self.assertRaisesRegex(DataContractError, "different.*selection"):
                bind_selection(first, base)
            shutil.copyfile(
                first.resolve("manifests/encoder-comparison-selection.json"),
                second.resolve("manifests/encoder-comparison-selection.json"),
            )
            with self.assertRaisesRegex(DataContractError, "differs"):
                validate_selection(second, large)

    def test_run_local_cache_uses_current_manifest_and_rejects_digest_mismatch(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "cache")
            layout.create()
            selected = self.config.select_arm("bert-base-common")
            revision = selected.profile.revision
            snapshot = layout.resolve(
                f"inputs/huggingface/models--fixture/snapshots/{revision}"
            )
            weight_bytes = b"fixture encoder weight\n"
            atomic_write_bytes(snapshot / "pytorch_model.bin", weight_bytes)
            for name in selected.profile.tokenizer_files:
                atomic_write_bytes(snapshot / name, (name + "\n").encode())
            profile = replace(
                selected.profile,
                weight_bytes=len(weight_bytes),
                weight_sha256=hashlib.sha256(weight_bytes).hexdigest(),
            )
            fixture_selection = replace(selected, profile=profile)
            write_cache_manifest(layout, model=profile.model, revision=profile.revision)
            identity = comparison_stage._verify_profile_cache(layout, fixture_selection)
            self.assertEqual(identity["weight_sha256"], profile.weight_sha256)
            missing_tokenizer = snapshot / profile.tokenizer_files[0]
            missing_tokenizer.unlink()
            with self.assertRaisesRegex(DataContractError, "lacks tokenizer files"):
                comparison_stage._verify_profile_cache(layout, fixture_selection)
            atomic_write_bytes(
                missing_tokenizer,
                (profile.tokenizer_files[0] + "\n").encode(),
            )
            bad_selection = replace(
                fixture_selection,
                profile=replace(profile, weight_sha256="0" * 64),
            )
            with self.assertRaisesRegex(DataContractError, "weight bytes differ"):
                comparison_stage._verify_profile_cache(layout, bad_selection)
            torch.save(
                {"first": torch.zeros(3, 4), "second": torch.zeros(2)},
                snapshot / "pytorch_model.bin",
            )
            counted_selection = replace(
                fixture_selection,
                profile=replace(
                    profile,
                    weight_bytes=(snapshot / "pytorch_model.bin").stat().st_size,
                    weight_sha256=sha256_file(snapshot / "pytorch_model.bin"),
                ),
            )
            self.assertEqual(
                comparison_stage._base_parameter_count(layout, counted_selection), 14
            )

    def test_checkpoint_summary_substitution_fails_before_manifest_promotion(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "checkpoint")
            layout.create()
            selected = self.config.select_arm("deberta-large-a20-a21-a12")
            checkpoint_dir = layout.resolve("checkpoints/seed-42")
            atomic_write_bytes(checkpoint_dir / "checkpoint.pt", b"checkpoint\n")
            atomic_write_bytes(checkpoint_dir / "restart-state.pt", b"restart\n")
            atomic_write_bytes(
                checkpoint_dir / "training-summary.json",
                (
                    json.dumps(
                        {
                            "status": "completed",
                            "test_evaluated": False,
                            "seed": 42,
                            "model_name": "foreign/model",
                            "model_revision": selected.profile.revision,
                            "selected_step": 100,
                            "selected_metrics": {"triple_f1": 0.1},
                        }
                    )
                    + "\n"
                ).encode(),
            )
            with self.assertRaisesRegex(DataContractError, "violates its selection"):
                comparison_stage._checkpoint_manifest(
                    layout,
                    selected,
                    42,
                    {
                        "cache_manifest_sha256": "0" * 64,
                        "cache_tree_sha256": "1" * 64,
                    },
                )

    def test_smoke_is_bounded_validated_and_required_for_admission(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "smoke")
            layout.create()
            selection = self.config.select_arm("bert-base-common")
            bind_selection(layout, selection)
            atomic_write_json(layout.resolve("data-prepared/split-manifest.json"), {})
            arguments = comparison_stage._smoke_training_arguments(
                layout, selection, 42
            )
            self.assertEqual(
                arguments[arguments.index("--max-steps") + 1], "1"
            )
            self.assertIn(
                "diagnostics/encoder-comparison-smoke",
                "/".join(arguments).replace("\\", "/"),
            )

            paths = comparison_stage._smoke_paths(layout)
            identity = comparison_stage._smoke_identity(layout, selection, 42)
            atomic_write_json(paths["identity"], identity)
            atomic_write_bytes(paths["checkpoint"], b"checkpoint\n")
            atomic_write_bytes(paths["restart"], b"restart\n")
            atomic_write_bytes(paths["progress"], b"smoke\n")
            checkpoint_sha256 = sha256_file(paths["checkpoint"])
            atomic_write_bytes(
                paths["checkpoint_before_resume"],
                (checkpoint_sha256 + "\n").encode("ascii"),
            )
            summary = {
                "status": "completed",
                "seed": 42,
                "model_name": selection.profile.model,
                "model_revision": selection.profile.revision,
                "completed_steps": 1,
                "selected_metrics": {"triple_f1": 0.0},
                "test_evaluated": False,
                "resumed_from": None,
                "environment": {"cuda_available": True, "cuda_device": "fixture"},
            }
            atomic_write_json(paths["training_summary"], summary)
            atomic_write_json(
                paths["resume_summary"],
                {**summary, "resumed_from": str(paths["restart"])},
            )
            atomic_write_bytes(
                paths["sentence"],
                b'{"example_id":"example-1","words":["x"]}\n',
            )
            atomic_write_bytes(
                paths["ledger"],
                (
                    json.dumps(
                        {
                            "protocol_id": "B04-PATH-A-1.3",
                            "example_id": "example-1",
                            "training_seed": 42,
                            "predicted_spans": [],
                            "predicted_relations": [],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode(),
            )
            atomic_write_json(
                paths["roundtrip"],
                {
                    "schema_version": "phase-g-encoder-smoke-roundtrip-1.0",
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_unchanged_after_resume": True,
                    "candidate_rows": 1,
                    "example_id": "example-1",
                },
            )
            evidence = comparison_stage._validate_smoke_outputs(
                layout, selection, 42
            )
            self.assertEqual(evidence["cuda_device"], "fixture")

            smoke_path = layout.resolve(comparison_stage.SMOKE_MANIFEST_RELATIVE)
            atomic_write_json(
                smoke_path,
                {
                    "status": "smoke-complete-awaiting-admission",
                    "run_id": layout.run_id,
                    "table1_arm": selection.arm.manifest(),
                },
            )
            admission = comparison_stage.record_admission(
                layout,
                selection,
                decision="admit",
                reason="fixture review passed",
            )
            self.assertEqual(admission["decision"], "admit")
            self.assertEqual(
                comparison_stage._require_admission(layout, selection), admission
            )
            atomic_write_json(smoke_path, {**load_json(smoke_path), "changed": True})
            with self.assertRaisesRegex(DataContractError, "lacks an admitted smoke"):
                comparison_stage._require_admission(layout, selection)


if __name__ == "__main__":
    unittest.main()
