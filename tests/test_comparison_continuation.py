"""Comparison admission boundaries and durable continuation without model work."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stages import comparison_matrix, encoder_comparison as stage
from utils.common.artifact_io import DataContractError, atomic_write_bytes, atomic_write_json, load_json
from utils.common.execution import exclusive_run
from utils.common.paths import RunLayout
from utils.encoder_comparison.config import DEFAULT_CONFIG, load_comparison_config
from utils.encoder_comparison import trainer


ROOT = Path(__file__).resolve().parents[1]


class ComparisonContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_comparison_config(ROOT)
        cls.selection = cls.config.select_arm("bert-base-common")

    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.layout = RunLayout(self.root, "comparison-arm")
        self.layout.create()

    def _prediction_inputs(self):
        for relative in (
            "manifests/00-checkout-manifest.json",
            "manifests/encoder-comparison-selection.json",
            stage.CACHE_MANIFEST_RELATIVE,
            "checkpoints/seed-42/checkpoint-manifest.json",
            "checkpoints/seed-42/checkpoint.pt",
            "data-prepared/test.jsonl",
        ):
            atomic_write_bytes(self.layout.resolve(relative), b"fixture\n")
        return self.layout.resolve("predictions/test/seed-42-prediction-ledger.jsonl")

    def test_direct_smoke_worker_requires_accounted_matrix_dispatch(self):
        matrix = RunLayout(self.root, "matrix")
        with (
            mock.patch.object(stage, "discover_source_root", return_value=self.root),
            mock.patch.object(stage, "load_comparison_config", return_value=self.config),
            mock.patch.object(comparison_matrix, "selected_matrix", return_value=(matrix, {})),
            mock.patch.object(comparison_matrix, "require_matrix_progress"),
            mock.patch.object(stage, "_execute_private") as execute,
        ):
            with self.assertRaisesRegex(DataContractError, "accounted matrix dispatch"):
                stage.run_private(["--matrix-id", matrix.run_id, "--run-id", self.layout.run_id,
                    "--arm", self.selection.arm.arm_id, "--training-seed", "42"],
                    action="smoke", default_config=DEFAULT_CONFIG)
            execute.assert_not_called()

    def test_matrix_accounts_smoke_separately_and_holds_every_arm_lock(self):
        matrix = RunLayout(self.root, "matrix")
        identity = {"members": {arm: f"matrix-{arm}" for arm in self.config.arms},
                    "training_seeds": self.config.value["training_seeds"]}

        def model_stage(layout, selection, seed, *args, command_runner=None):
            with self.assertRaisesRegex(DataContractError, "another command owns"):
                with exclusive_run(layout):
                    pass
            self.assertIsNotNone(command_runner)
            command_runner(["fixture"], cwd=layout.source_root, check=True)

        with (
            mock.patch.object(comparison_matrix, "matrix_identity", return_value=identity),
            mock.patch.object(comparison_matrix, "checked_process") as dispatch,
            mock.patch.object(comparison_matrix.shutil, "disk_usage", return_value=SimpleNamespace(free=2e11)),
            mock.patch.object(stage, "prepare"),
            mock.patch.object(stage, "run_smoke", side_effect=model_stage),
            mock.patch.object(stage, "run_training", side_effect=model_stage),
            mock.patch.object(stage, "generate", side_effect=model_stage),
            mock.patch("utils.evaluation.reporting.report_comparison"),
            mock.patch("utils.evaluation.reporting.report_matrix"),
        ):
            self.assertEqual(comparison_matrix.run_matrix(matrix, self.config), 0)
        labels = [call.kwargs["label"] for call in dispatch.call_args_list]
        self.assertEqual(labels, ["comparison-smoke"] * 4 + ["comparison-full"] * 64)
        for call in dispatch.call_args_list[4:]:
            self.assertEqual(call.kwargs["limit_seconds"], 48 * 3600)

    def test_existing_raw_ledger_needs_its_original_producer_receipt(self):
        ledger = self._prediction_inputs()
        atomic_write_bytes(ledger, b"foreign same-seed ledger\n")
        command = mock.Mock()
        with (
            mock.patch.object(comparison_matrix, "require_matrix_progress"),
            mock.patch.object(stage, "_validate_prepared_inputs"),
            mock.patch.object(stage, "_verify_profile_cache"),
            mock.patch.object(stage, "generate_candidates") as convert,
        ):
            with self.assertRaisesRegex(DataContractError, "producer evidence"):
                stage.generate(self.layout, self.selection, 42, "test", command_runner=command)
            command.assert_not_called()
            convert.assert_not_called()

            identity = stage._prediction_identity(self.layout, self.selection, 42, "test")
            receipt = ledger.with_suffix(".jsonl.manifest.json")
            atomic_write_json(receipt, stage._prediction_receipt(ledger, identity))
            stage.generate(self.layout, self.selection, 42, "test", command_runner=command)
            command.assert_not_called()
            convert.assert_called_once()
            atomic_write_bytes(ledger, b"changed ledger\n")
            with self.assertRaisesRegex(DataContractError, "producer evidence"):
                stage.generate(self.layout, self.selection, 42, "test", command_runner=command)
            self.assertEqual(convert.call_count, 1)

    def test_private_prediction_seals_once_and_cannot_overwrite(self):
        ledger = self._prediction_inputs()
        args = SimpleNamespace(training_seed=42, split="test")
        with (
            mock.patch.object(stage, "_validate_prepared_inputs"),
            mock.patch.object(trainer, "main", side_effect=lambda _: atomic_write_bytes(ledger, b"original\n")) as predict,
        ):
            self.assertEqual(stage._execute_private(self.layout, self.selection, args, "predict"), 0)
            self.assertEqual(load_json(ledger.with_suffix(".jsonl.manifest.json")),
                stage._prediction_receipt(ledger, stage._prediction_identity(self.layout, self.selection, 42, "test")))
            with self.assertRaisesRegex(DataContractError, "overwrite retained"):
                stage._execute_private(self.layout, self.selection, args, "predict")
            predict.assert_called_once()

    def test_prediction_rejects_numerical_drift_before_model_access(self):
        with (
            mock.patch.object(trainer, "record_runtime", side_effect=DataContractError("runtime differs")),
            mock.patch.object(trainer, "_model") as model,
            mock.patch.object(trainer, "_tokenizer") as tokenizer,
        ):
            with self.assertRaisesRegex(DataContractError, "runtime differs"):
                trainer.predict(SimpleNamespace(prepared_dir="fixture"))
            model.assert_not_called()
            tokenizer.assert_not_called()

    def test_restart_rejects_another_run_before_loading_state(self):
        args = SimpleNamespace(seed=42, model_name="fixture", model_revision="a" * 40,
            max_steps=3500, prepared_dir="fixture", resume_from="fixture.pt")
        model = mock.Mock()
        with mock.patch.object(trainer, "restart_identity", return_value={"run_id": "foreign"}):
            state = trainer._restart_payload(args, model, mock.Mock(), mock.Mock(),
                SimpleNamespace(sampler=mock.Mock()), next_step=5, best_metrics={"triple_f1": 0.1},
                best_step=4, boost_adaptive_triggered=False, boost_adaptive_switched=False)
        model.reset_mock()
        with (
            mock.patch.object(trainer, "restart_identity", return_value={"run_id": "current"}),
            mock.patch.object(trainer.torch, "load", return_value=state),
        ):
            with self.assertRaisesRegex(ValueError, "run_identity mismatch"):
                trainer._load_restart(args, model, mock.Mock(), mock.Mock(), mock.Mock())
            model.load_state_dict.assert_not_called()

    def test_incomplete_training_cannot_be_promoted_as_full_development(self):
        for name in ("checkpoint.pt", "restart-state.pt"):
            atomic_write_bytes(self.layout.resolve(f"checkpoints/seed-42/{name}"), b"fixture")
        summary = {"status": "completed", "test_evaluated": False, "seed": 42,
            "run_identity": {"run_id": self.layout.run_id},
            "completed_steps": 3500,
            "selected_step": 100, "selected_metrics": {"triple_f1": 0.2}}
        for change in ({"completed_steps": 1}, {"test_evaluated": True},
                       {"run_identity": {"run_id": "foreign"}},
                       {"selected_step": 3501}, {"selected_metrics": {"triple_f1": float("nan")}},
                       {"selected_metrics": {"triple_f1": 1.1}}):
            with self.subTest(change=change):
                atomic_write_bytes(self.layout.resolve("checkpoints/seed-42/training-summary.json"),
                    json.dumps({**summary, **change}).encode("utf-8"))
                with mock.patch.object(stage, "restart_identity", return_value={"run_id": self.layout.run_id}), self.assertRaises(DataContractError):
                    stage._checkpoint_manifest(self.layout, self.selection, 42, {})

    def test_final_test_barrier_authenticates_required_development_hashes(self):
        atomic_write_json(self.layout.resolve(stage.SMOKE_MANIFEST_RELATIVE), {})
        for relative in ("manifests/encoder-comparison-train-plan-seed-42.json",
            "checkpoints/seed-42/checkpoint-manifest.json", "checkpoints/seed-42/training-summary.json",
            "checkpoints/seed-42/restart-state.pt", "logs/encoder-comparison-seed-42.log"):
            atomic_write_json(self.layout.resolve(relative), {})
        checkpoint = self.layout.resolve("checkpoints/seed-42/checkpoint-manifest.json")
        plan = {"training_seed": 42, "run_id": self.layout.run_id}
        result_path = self.layout.resolve("manifests/encoder-comparison-train-live-seed-42.json")
        good = stage._training_result(self.layout, plan, checkpoint, {})
        matrix = {"members": {self.selection.arm.arm_id: self.layout.run_id}}
        with (
            mock.patch.dict(self.config.value, {'training_seeds': [42]}),
            mock.patch.object(comparison_matrix, "selected_matrix", return_value=(self.layout, matrix)),
            mock.patch.object(stage, "run_smoke"),
            mock.patch.object(stage, "_verify_profile_cache", return_value={}),
            mock.patch.object(stage, "_checkpoint_manifest", return_value={}),
            mock.patch.object(stage, "plan_training", return_value=plan),
        ):
            atomic_write_json(result_path, good)
            comparison_matrix.require_matrix_progress(self.layout, self.selection, boundary="test")
            for change in ({"output_hashes": {}}, {"training_seed": 49}):
                with self.subTest(change=change):
                    atomic_write_json(result_path, {**good, **change})
                    with self.assertRaisesRegex(DataContractError, "completion evidence changed"):
                        comparison_matrix.require_matrix_progress(self.layout, self.selection, boundary="test")


if __name__ == "__main__":
    unittest.main()
