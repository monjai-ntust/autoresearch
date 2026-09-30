"""Same-run recovery tests for comparison checkouts with no downstream input."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import encoder_comparison
from stages import encoder_comparison as comparison
from stages import preparation
from utils.common.artifact_io import (
    DataContractError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
)
from utils.common.paths import RunLayout, discover_source_root
from utils.encoder_comparison.config import DEFAULT_CONFIG, load_comparison_config


SOURCE_ROOT = discover_source_root(Path(__file__))
CHECKOUT = "manifests/00-checkout-manifest.json"
HISTORY = "audit/checkout-recovery"


@contextmanager
def git_status(clean=True):
    """Simulate source cleanliness while all other doctor/Git checks run."""

    original = subprocess.run

    def run(arguments, *args, **kwargs):
        if arguments[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(
                arguments, 0, stdout="" if clean else "?? nohup.out\n", stderr=""
            )
        return original(arguments, *args, **kwargs)

    with mock.patch.object(subprocess, "run", side_effect=run), \
         mock.patch.object(sys, "dont_write_bytecode", True):
        yield


class ComparisonCheckoutRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_comparison_config(SOURCE_ROOT, DEFAULT_CONFIG)
        cls.previous_commit = subprocess.run(
            ["git", "rev-parse", "HEAD^"], cwd=SOURCE_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    @contextmanager
    def blocked_run(self):
        output = SOURCE_ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-checkout-recovery-", dir=output) as temporary:
            layout = RunLayout(SOURCE_ROOT, Path(temporary).name)
            pipeline = self.config.pipeline
            document = {
                "schema_version": "phase-b-checkout-manifest-2.0",
                "protocol_id": pipeline.value["protocol_id"],
                "workflow_id": pipeline.value["workflow_id"],
                "matcher_id": pipeline.value["matcher_id"],
                "run_id": layout.run_id,
                "status": "blocked",
                "source": {
                    "commit": self.previous_commit,
                    "worktree_clean": False,
                    "config": {
                        "path": pipeline.path.relative_to(SOURCE_ROOT).as_posix(),
                        "sha256": sha256_file(pipeline.path),
                    },
                },
                "checks": [
                    {"check_id": "git-head", "status": "pass"},
                    {"check_id": "clean-tracked-source", "status": "fail"},
                ],
            }
            # Noncanonical whitespace proves the old file is preserved as bytes,
            # not reconstructed from its parsed JSON.
            atomic_write_bytes(
                layout.resolve(CHECKOUT),
                (json.dumps(document, indent=2) + "\r\n").encode("utf-8"),
            )
            yield layout, document

    def selection(self, arm="bert-base-common"):
        return self.config.select_arm(arm)

    def test_normal_prepare_still_refuses_the_original_blocked_checkout(self):
        with self.blocked_run() as (layout, _document), \
             mock.patch.object(comparison, "bind_selection") as bind:
            with self.assertRaisesRegex(DataContractError, "status differs"):
                comparison.prepare(layout, self.selection())
            bind.assert_not_called()
            self.assertEqual(
                [p.relative_to(layout.run_root).as_posix()
                 for p in layout.run_root.rglob("*") if p.is_file()], [CHECKOUT]
            )

    def test_recovery_preserves_failed_bytes_and_reruns_real_doctor_for_all_arms(self):
        for arm in self.config.arms:
            with self.subTest(arm=arm), self.blocked_run() as (layout, _), git_status(), \
                 mock.patch.object(comparison, "run_preparation") as prepare, \
                 mock.patch.object(comparison, "run_smoke") as smoke, \
                 mock.patch.object(comparison, "bind_selection") as bind, \
                 mock.patch.object(comparison, "run_doctor", wraps=comparison.run_doctor) as doctor:
                original = layout.resolve(CHECKOUT).read_bytes()
                receipt = comparison.recover_checkout(layout, self.selection(arm))
                self.assertEqual(receipt["status"], "checkout-admitted")
                self.assertEqual(receipt["run_id"], layout.run_id)
                self.assertEqual(receipt["arm_id"], arm)
                archived = layout.resolve(receipt["previous_checkout"]["path"])
                self.assertEqual(archived.read_bytes(), original)
                self.assertEqual(sha256_file(archived), receipt["previous_checkout"]["sha256"])
                fresh = load_json(layout.resolve(CHECKOUT))
                self.assertEqual(fresh["status"], "pass")
                self.assertTrue(fresh["source"]["worktree_clean"])
                self.assertNotEqual(fresh["source"]["commit"], self.previous_commit)
                self.assertEqual(
                    layout.resolve(CHECKOUT).read_bytes(),
                    layout.resolve(receipt["attempt_checkout"]["path"]).read_bytes(),
                )
                self.assertEqual(
                    load_json(layout.resolve(receipt["recovery_manifest"]["path"])),
                    {key: value for key, value in receipt.items() if key != "recovery_manifest"},
                )
                preparation._validate_existing_checkout(layout, self.config.pipeline)
                doctor.assert_called_once()
                prepare.assert_not_called()
                smoke.assert_not_called()
                bind.assert_not_called()
                self.assertFalse(layout.resolve("inputs").exists())
                self.assertFalse(layout.resolve("checkpoints").exists())
                self.assertFalse(list(layout.run_root.rglob("*.lock")))
                self.assertFalse(list(layout.run_root.rglob(".pending-*")))

    def test_repeated_recovery_is_a_no_write_operation_after_admission(self):
        with self.blocked_run() as (layout, _), git_status():
            comparison.recover_checkout(layout, self.selection())
            before = {p: p.read_bytes() for p in layout.run_root.rglob("*") if p.is_file()}
            with mock.patch.object(comparison, "run_doctor") as doctor:
                result = comparison.recover_checkout(layout, self.selection())
            self.assertEqual(result["status"], "checkout-already-admitted")
            doctor.assert_not_called()
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertEqual(set(before), {p for p in layout.run_root.rglob("*") if p.is_file()})

    def test_dirty_retry_retains_both_failures_and_can_be_retried_cleanly(self):
        with self.blocked_run() as (layout, _):
            original = layout.resolve(CHECKOUT).read_bytes()
            with git_status(clean=False), self.assertRaisesRegex(
                DataContractError, "recovery doctor failed"
            ):
                comparison.recover_checkout(layout, self.selection())
            self.assertEqual(layout.resolve(CHECKOUT).read_bytes(), original)
            receipts = list(layout.resolve(HISTORY).glob("recovery-*.json"))
            self.assertEqual(len(receipts), 1)
            failed = load_json(receipts[0])
            self.assertEqual(failed["status"], "blocked")
            attempted = layout.resolve(failed["attempt_checkout"]["path"])
            self.assertEqual(load_json(attempted)["status"], "blocked")
            retained = {p: p.read_bytes() for p in layout.resolve(HISTORY).glob("*.json")}
            with git_status():
                result = comparison.recover_checkout(layout, self.selection())
            self.assertEqual(result["status"], "checkout-admitted")
            for path, content in retained.items():
                self.assertEqual(path.read_bytes(), content)

    def test_any_downstream_or_unknown_file_blocks_recovery_without_writes(self):
        for relative in (
            "inputs/archive.zip", "data-prepared/train.jsonl",
            "checkpoints/seed-42/checkpoint.pt", "logs/partial.log",
            "manifests/encoder-comparison-selection.json",
            "manifests/encoder-comparison-smoke.json",
            "manifests/encoder-comparison-admission.json",
            f"{HISTORY}/unknown.json",
        ):
            with self.subTest(path=relative), self.blocked_run() as (layout, _), \
                 mock.patch.object(comparison, "run_doctor") as doctor:
                original = layout.resolve(CHECKOUT).read_bytes()
                atomic_write_bytes(layout.resolve(relative), b"retained fixture\n")
                before = {p: p.read_bytes() for p in layout.run_root.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(DataContractError, "unknown artifact"):
                    comparison.recover_checkout(layout, self.selection())
                doctor.assert_not_called()
                self.assertEqual(layout.resolve(CHECKOUT).read_bytes(), original)
                self.assertEqual(before, {p: p.read_bytes() for p in layout.run_root.rglob("*") if p.is_file()})

    def test_foreign_identity_config_or_failure_reason_is_not_recoverable(self):
        for change in (
            {"run_id": "foreign-run"}, {"status": "unknown"},
            {"schema_version": "unsupported"},
            {"checks": [{"check_id": "root-output-ignore-rule", "status": "fail"}]},
            {"checks": [{"check_id": "clean-tracked-source", "status": "fail"}] * 2},
            {"source": {"worktree_clean": True}},
            {"source": {"worktree_clean": False, "commit": "main"}},
            {"source": {"worktree_clean": False, "commit": "f" * 40, "config": {}}},
        ):
            with self.subTest(change=change), self.blocked_run() as (layout, document), \
                 mock.patch.object(comparison, "run_doctor") as doctor:
                atomic_write_json(layout.resolve(CHECKOUT), {**document, **change})
                original = layout.resolve(CHECKOUT).read_bytes()
                with self.assertRaises(DataContractError):
                    comparison.recover_checkout(layout, self.selection())
                doctor.assert_not_called()
                self.assertEqual(layout.resolve(CHECKOUT).read_bytes(), original)
                self.assertFalse(layout.resolve(HISTORY).exists())

    def test_nonancestor_checkout_and_forged_git_config_fail_closed(self):
        for mode in ("ancestor", "config"):
            with self.subTest(mode=mode), self.blocked_run() as (layout, document):
                original_run = subprocess.run

                def git(arguments, *args, **kwargs):
                    if mode == "ancestor" and arguments[:2] == ["git", "merge-base"]:
                        return subprocess.CompletedProcess(arguments, 1, b"", b"")
                    if mode == "config" and arguments[:2] == ["git", "show"]:
                        return subprocess.CompletedProcess(arguments, 0, b"different", b"")
                    return original_run(arguments, *args, **kwargs)

                with mock.patch.object(subprocess, "run", side_effect=git), \
                     self.assertRaisesRegex(DataContractError, "ancestor|authenticated"):
                    comparison.recover_checkout(layout, self.selection())
                self.assertEqual(load_json(layout.resolve(CHECKOUT)), document)
                self.assertFalse(layout.resolve(HISTORY).exists())

    def test_recovery_history_cannot_be_tampered_or_rebound_to_another_arm(self):
        with self.blocked_run() as (layout, _), git_status():
            receipt = comparison.recover_checkout(layout, self.selection())
            with self.assertRaisesRegex(DataContractError, "arm_id differs"):
                comparison.recover_checkout(layout, self.selection("deberta-base-common"))
            path = layout.resolve(receipt["previous_checkout"]["path"])
            atomic_write_bytes(path, path.read_bytes() + b" ")
            current = layout.resolve(CHECKOUT).read_bytes()
            with self.assertRaisesRegex(DataContractError, "hash differs"):
                comparison.recover_checkout(layout, self.selection())
            self.assertEqual(layout.resolve(CHECKOUT).read_bytes(), current)

    def test_invalid_recovery_receipt_bindings_fail_closed(self):
        with self.blocked_run() as (layout, _), git_status():
            receipt = comparison.recover_checkout(layout, self.selection())
            path = layout.resolve(receipt["recovery_manifest"]["path"])
            document = load_json(path)
            document["previous_checkout"]["path"] = "../foreign/checkout.json"
            comparison._write_checkout_evidence(
                layout, "recovery", canonical_json_bytes(document)
            )
            with self.assertRaisesRegex(DataContractError, "binding differs"):
                comparison.recover_checkout(layout, self.selection())

    def test_symlinked_artifact_is_rejected_without_following_it(self):
        with self.blocked_run() as (layout, _):
            link = layout.run_root / "config-link.json"
            try:
                link.symlink_to(self.config.pipeline.path)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaisesRegex(DataContractError, "symlinked"):
                comparison.recover_checkout(layout, self.selection())
            self.assertFalse(layout.resolve(HISTORY).exists())

    def test_existing_recovery_lock_is_preserved_and_blocks_another_retry(self):
        with self.blocked_run() as (layout, _), \
             mock.patch.object(comparison, "run_doctor") as doctor:
            lock = layout.resolve(f"{HISTORY}/recovery.lock")
            atomic_write_bytes(lock, b"retained prior attempt")
            original = layout.resolve(CHECKOUT).read_bytes()
            with self.assertRaisesRegex(DataContractError, "active or interrupted"):
                comparison.recover_checkout(layout, self.selection())
            doctor.assert_not_called()
            self.assertEqual(lock.read_bytes(), b"retained prior attempt")
            self.assertEqual(layout.resolve(CHECKOUT).read_bytes(), original)

    def test_source_change_after_doctor_does_not_promote_the_fresh_manifest(self):
        with self.blocked_run() as (layout, _), git_status():
            original = layout.resolve(CHECKOUT).read_bytes()
            with mock.patch.object(
                comparison, "_validate_existing_checkout",
                side_effect=DataContractError("source changed after doctor"),
            ), self.assertRaisesRegex(DataContractError, "source changed"):
                comparison.recover_checkout(layout, self.selection())
            self.assertEqual(layout.resolve(CHECKOUT).read_bytes(), original)
            receipt = load_json(next(layout.resolve(HISTORY).glob("recovery-*.json")))
            self.assertEqual(receipt["status"], "blocked")
            self.assertEqual(
                load_json(layout.resolve(receipt["attempt_checkout"]["path"]))["status"],
                "pass",
            )

    def test_failed_promotion_keeps_original_and_can_be_retried(self):
        with self.blocked_run() as (layout, _), git_status():
            original = layout.resolve(CHECKOUT).read_bytes()
            original_write = comparison.atomic_write_bytes

            def write(path, payload):
                if path == layout.resolve(CHECKOUT):
                    raise OSError("simulated interrupted promotion")
                original_write(path, payload)

            with mock.patch.object(comparison, "atomic_write_bytes", side_effect=write), \
                 self.assertRaisesRegex(OSError, "interrupted promotion"):
                comparison.recover_checkout(layout, self.selection())
            self.assertEqual(layout.resolve(CHECKOUT).read_bytes(), original)
            with mock.patch.object(comparison, "atomic_write_bytes", wraps=original_write):
                result = comparison.recover_checkout(layout, self.selection())
            self.assertEqual(result["status"], "checkout-admitted")

    def test_recovery_cannot_silently_readmit_a_successful_old_source(self):
        with self.blocked_run() as (layout, document), git_status():
            document["status"] = "pass"
            document["source"]["worktree_clean"] = True
            atomic_write_json(layout.resolve(CHECKOUT), document)
            with self.assertRaisesRegex(DataContractError, "differs from the current"):
                comparison.recover_checkout(layout, self.selection())
            self.assertFalse(layout.resolve(HISTORY).exists())

    def test_isolated_cli_exposes_recovery_and_dispatches_without_prepare(self):
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "encoder_comparison.py", "recover-checkout", "--help"],
            cwd=SOURCE_ROOT, check=True, capture_output=True, text=True,
        )
        self.assertIn("--run-id", completed.stdout)
        with mock.patch.object(encoder_comparison, "recover_checkout", return_value={"status": "fixture"}) as recover, \
             mock.patch.object(encoder_comparison, "prepare") as prepare:
            with mock.patch("builtins.print"):
                result = encoder_comparison.main([
                    "recover-checkout", "--arm", "bert-base-common", "--run-id", "same-run",
                ])
        self.assertEqual(result, 0)
        self.assertEqual(recover.call_args.args[0].run_id, "same-run")
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
