"""Fresh blocked doctors continue under the original ID without importing work."""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stages import preparation
from utils.common.artifact_io import (
    DataContractError, atomic_write_bytes, atomic_write_json, load_json, sha256_file,
)
from utils.common.config import PipelineConfig
from utils.common.paths import RunLayout
from utils.preparation import checkout

ROOT = Path(__file__).resolve().parents[1]


class CheckoutRecoveryTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "resources/configs/pipeline.json"
        path.parent.mkdir(parents=True)
        value = {"protocol_id": "protocol", "workflow_id": "workflow", "matcher_id": "matcher"}
        path.write_bytes((json.dumps(value, indent=2) + "\r\n").encode("utf-8"))
        self.config = PipelineConfig(path, value)
        self.layout = RunLayout(self.root, "research-1019")
        self.layout.create()
        self.config_bytes = path.read_bytes()
        committed = mock.patch.object(checkout, '_committed_config', return_value=self.config_bytes)
        committed.start()
        self.addCleanup(committed.stop)
        self.blocked = {
            "schema_version": "phase-b-checkout-manifest-2.0", "lifecycle_version": 1,
            **value, "run_id": self.layout.run_id, "status": "blocked",
            "source": {"commit": "a" * 40, "worktree_clean": False, "config": {
                "path": "resources/configs/pipeline.json"}},
            "environment": {"machine": "aarch64"},
            "checks": [{"check_id": "git-head", "status": "pass"},
                       {"check_id": "clean-tracked-source", "status": "fail"}],
        }
        self.original = (json.dumps(self.blocked, indent=2) + "\r\n").encode("utf-8")
        atomic_write_bytes(self.layout.resolve(checkout.CHECKOUT), self.original)
        self.fresh = copy.deepcopy(self.blocked)
        self.fresh["status"] = "pass"
        self.fresh["source"].update(commit="b" * 40, worktree_clean=True)
        self.fresh["checks"][-1]["status"] = "pass"

    def _initialize(self):
        responses = [subprocess.CompletedProcess([], 0, stdout="b" * 40 + "\n"),
                     subprocess.CompletedProcess([], 0, stdout="")]
        with (mock.patch.object(checkout, "inspect_checkout", return_value=(copy.deepcopy(self.fresh), True)),
              mock.patch.object(preparation.subprocess, "run", side_effect=responses)):
            preparation.initialize(self.layout, self.config)

    def _receipt(self):
        return next(self.layout.resolve(checkout.RECOVERY).glob("receipt-*.json"))

    def test_initialize_retains_exact_failure_and_source_selector_then_continues_same_run(self):
        self._initialize()
        root = self.layout.resolve(checkout.RECOVERY)
        self.assertEqual((root / "previous-checkout.json").read_bytes(), self.original)
        self.assertFalse((root / 'config.json').exists())
        self.assertFalse(self.layout.resolve('inputs/config/pipeline.json').exists())
        admitted = self.layout.resolve(checkout.CHECKOUT).read_bytes()
        self.assertEqual((root / "fresh-checkout.json").read_bytes(), admitted)
        document = load_json(self.layout.resolve(checkout.CHECKOUT))
        self.assertEqual(document["run_id"], "research-1019")
        self.assertEqual(document["source"]["commit"], "b" * 40)
        self.assertTrue(document["source"]["worktree_clean"])
        receipt_path = self._receipt()
        receipt = load_json(receipt_path)
        self.assertEqual(set(receipt["artifacts"]), {"previous_checkout", "fresh_checkout"})
        self.assertEqual(receipt_path.stem, "receipt-" + sha256_file(receipt_path))
        for identity in receipt["artifacts"].values():
            self.assertEqual(sha256_file(self.layout.resolve(identity["path"])), identity["sha256"])
        self._initialize()
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), admitted)

    def test_passing_checkout_is_not_rewritten_or_doctored(self):
        atomic_write_json(self.layout.resolve(checkout.CHECKOUT), self.fresh)
        before = self.layout.resolve(checkout.CHECKOUT).read_bytes()
        with mock.patch.object(checkout, "inspect_checkout") as inspect:
            checkout.continue_checkout(self.layout, self.config)
            inspect.assert_not_called()
        self._initialize()
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), before)
        self.assertFalse(self.layout.resolve(checkout.RECOVERY).exists())

    def test_still_dirty_run_stays_blocked_without_replacing_any_evidence(self):
        with mock.patch.object(checkout, "inspect_checkout", return_value=(self.blocked, False)):
            with self.assertRaisesRegex(DataContractError, "Checkout doctor failed"):
                checkout.continue_checkout(self.layout, self.config)
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), self.original)
        self.assertFalse(self.layout.resolve(checkout.RECOVERY).exists())

    def test_old_workflows_and_other_doctor_failures_are_not_recovered(self):
        for mutation in ("no_lifecycle", "boolean_lifecycle", "other_failure", "foreign_run"):
            with self.subTest(mutation=mutation):
                document = copy.deepcopy(self.blocked)
                if mutation == "no_lifecycle":
                    document.pop("lifecycle_version")
                elif mutation == "boolean_lifecycle":
                    document["lifecycle_version"] = True
                elif mutation == "other_failure":
                    document["checks"][0]["status"] = "fail"
                else:
                    document["run_id"] = "foreign"
                atomic_write_json(self.layout.resolve(checkout.CHECKOUT), document)
                with mock.patch.object(checkout, "inspect_checkout") as inspect:
                    with self.assertRaises(DataContractError):
                        checkout.continue_checkout(self.layout, self.config)
                    inspect.assert_not_called()
                self.assertFalse(self.layout.resolve(checkout.RECOVERY).exists())

    def test_retired_check_only_failures_continue_after_current_doctor_passes(self):
        for check_id in ("root-output-ignore-rule", "root-output-ignore-negative", "source-bytecode-disabled"):
            with self.subTest(check_id=check_id):
                document = copy.deepcopy(self.blocked)
                document["source"]["worktree_clean"] = True
                document["checks"][-1]["status"] = "pass"
                document["checks"].append({"check_id": check_id, "status": "fail"})
                atomic_write_json(self.layout.resolve(checkout.CHECKOUT), document)
                self._initialize()
                self.assertEqual(load_json(self.layout.resolve(checkout.CHECKOUT))["status"], "pass")
                # Each subcase has its own independent checkout-only namespace.
                self.layout = RunLayout(self.root, self.layout.run_id + "x")
                self.layout.create()
                self.blocked["run_id"] = self.layout.run_id
                self.fresh["run_id"] = self.layout.run_id

    def test_any_acquired_or_downstream_file_prevents_checkout_replacement(self):
        for relative in ("inputs/dataset.zip", "data-prepared/train.jsonl",
                "checkpoints/restart-state.pt", "predictions/dev/ledger.jsonl",
                "manifests/comparison-matrix-link.json", "manifests/encoder-comparison-selection.json",
                "diagnostics/encoder-comparison-smoke/completed.json", "unknown.txt"):
            with self.subTest(relative=relative):
                path = self.layout.resolve(relative)
                atomic_write_bytes(path, b"existing evidence\n")
                with self.assertRaisesRegex(DataContractError, "downstream or unknown"):
                    checkout.continue_checkout(self.layout, self.config)
                self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), self.original)
                self.assertEqual(path.read_bytes(), b"existing evidence\n")
                path.unlink()

    def test_changed_configuration_is_rejected_before_recovery_writes(self):
        self.config.path.write_bytes(b"changed\n")
        with self.assertRaisesRegex(DataContractError, "configuration differs"):
            checkout.continue_checkout(self.layout, self.config)
        self.assertFalse(self.layout.resolve(checkout.RECOVERY).exists())
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), self.original)

    def test_interruption_before_receipt_can_finish_identical_snapshots(self):
        original_retain = checkout.retain_bytes
        def interrupt(path, value):
            if path.name == "fresh-checkout.json":
                raise RuntimeError("snapshot interrupted")
            original_retain(path, value)
        with (mock.patch.object(checkout, "inspect_checkout", return_value=(copy.deepcopy(self.fresh), True)),
              mock.patch.object(checkout, "retain_bytes", side_effect=interrupt)):
            with self.assertRaisesRegex(RuntimeError, "snapshot interrupted"):
                checkout.continue_checkout(self.layout, self.config)
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), self.original)
        self._initialize()

    def test_interruption_after_receipt_replays_without_losing_old_manifest(self):
        with (mock.patch.object(checkout, "inspect_checkout", return_value=(copy.deepcopy(self.fresh), True)),
              mock.patch.object(checkout, "atomic_write_bytes", side_effect=RuntimeError("promotion interrupted"))):
            with self.assertRaisesRegex(RuntimeError, "promotion interrupted"):
                checkout.continue_checkout(self.layout, self.config)
        receipt = self._receipt().read_bytes()
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), self.original)
        self._initialize()
        self.assertEqual(self._receipt().read_bytes(), receipt)

    def test_completed_recovery_keeps_proof_required_after_downstream_work(self):
        self._initialize()
        atomic_write_bytes(self.layout.resolve("data-prepared/train.jsonl"), b"later scientific work\n")
        self._initialize()
        self._receipt().unlink()
        with self.assertRaisesRegex(DataContractError, "receipt is missing"):
            self._initialize()

    def test_changed_or_missing_retained_bytes_fail_closed(self):
        self._initialize()
        path = self.layout.resolve(f"{checkout.RECOVERY}/previous-checkout.json")
        original = path.read_bytes()
        for mutation in ("change", "delete"):
            with self.subTest(mutation=mutation):
                if mutation == "change":
                    path.write_bytes(original + b"\n")
                else:
                    path.unlink()
                admitted = self.layout.resolve(checkout.CHECKOUT).read_bytes()
                with self.assertRaises((DataContractError, FileNotFoundError)):
                    self._initialize()
                self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), admitted)
                path.write_bytes(original)

    def test_changed_or_foreign_receipt_never_admits_retained_evidence(self):
        self._initialize()
        receipt_path = self._receipt()
        original = receipt_path.read_bytes()
        receipt_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(DataContractError, "receipt hash differs"):
            self._initialize()
        receipt_path.write_bytes(original)
        foreign = load_json(receipt_path)
        foreign["run_id"] = "foreign"
        receipt_path.unlink()
        temporary_path = self.layout.resolve(f"{checkout.RECOVERY}/foreign-receipt.json")
        atomic_write_json(temporary_path, foreign)
        temporary_path.rename(temporary_path.with_name("receipt-" + sha256_file(temporary_path) + ".json"))
        with self.assertRaisesRegex(DataContractError, "different or changed evidence"):
            self._initialize()

    def test_pending_recovery_rejects_new_downstream_work(self):
        with (mock.patch.object(checkout, "inspect_checkout", return_value=(copy.deepcopy(self.fresh), True)),
              mock.patch.object(checkout, "atomic_write_bytes", side_effect=RuntimeError("interrupted"))):
            with self.assertRaises(RuntimeError):
                checkout.continue_checkout(self.layout, self.config)
        atomic_write_bytes(self.layout.resolve("inputs/dataset.zip"), b"unadmitted input\n")
        with self.assertRaisesRegex(DataContractError, "downstream or unknown"):
            self._initialize()
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), self.original)

    def test_pending_recovery_rejects_changed_source_instead_of_promoting_stale_identity(self):
        with (mock.patch.object(checkout, "inspect_checkout", return_value=(copy.deepcopy(self.fresh), True)),
              mock.patch.object(checkout, "atomic_write_bytes", side_effect=RuntimeError("interrupted"))):
            with self.assertRaises(RuntimeError):
                checkout.continue_checkout(self.layout, self.config)
        changed = copy.deepcopy(self.fresh)
        changed["source"]["commit"] = "c" * 40
        with mock.patch.object(checkout, "inspect_checkout", return_value=(changed, True)):
            with self.assertRaisesRegex(DataContractError, "recorded clean source"):
                checkout.continue_checkout(self.layout, self.config)
        self.assertEqual(self.layout.resolve(checkout.CHECKOUT).read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
