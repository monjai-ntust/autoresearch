"""Checkout diagnostics preserve the clean-source gate and show its evidence."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.common import environment
from utils.common.config import PipelineConfig
from utils.common.paths import RunLayout


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class CheckoutDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        output_root = SOURCE_ROOT / "output"
        output_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="test-checkout-diagnostics-", dir=output_root
        )
        self.addCleanup(self.temporary.cleanup)
        self.source_root = Path(self.temporary.name)
        (self.source_root / ".gitignore").write_text("/output/\n", encoding="utf-8")
        config_path = self.source_root / "resources" / "configs" / "pipeline.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}\n", encoding="utf-8")
        self.config = PipelineConfig(
            config_path,
            {"environment": {"python": "3.10.20", "uv": "0.11.26"}},
        )

    def doctor(
        self, *, status_stdout="", status_stderr="", status_returncode=0, inspect_only=False
    ):
        def command(source_root, arguments):
            self.assertEqual(source_root, self.source_root)
            if arguments == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(arguments, 0, "a" * 40 + "\n", "")
            if arguments == ["git", "status", "--porcelain", "--untracked-files=normal"]:
                return subprocess.CompletedProcess(
                    arguments, status_returncode, status_stdout, status_stderr
                )
            if arguments[:3] == ["git", "check-ignore", "-q"]:
                return subprocess.CompletedProcess(
                    arguments, 1 if arguments[-1].startswith("nested/") else 0, "", ""
                )
            self.fail(f"unexpected checkout command: {arguments!r}")

        with mock.patch.object(environment, "_command", side_effect=command), mock.patch.object(
            environment, "_uv_version", return_value=("0.11.26", "uv 0.11.26")
        ):
            inspect = environment.inspect_checkout if inspect_only else environment.run_doctor
            manifest, passed = inspect(
                RunLayout(self.source_root, "diagnostic-run"), self.config
            )
        check = next(
            check for check in manifest["checks"] if check["check_id"] == "clean-tracked-source"
        )
        return manifest, passed, check

    def test_dirty_checkout_reports_original_status_paths_and_still_blocks(self):
        status = ' M README.md\n?? nohup.out\nR  "old name.py" -> "new name.py"\n'
        manifest, passed, check = self.doctor(status_stdout=status)
        self.assertFalse(passed)
        self.assertEqual(manifest["status"], "blocked")
        self.assertFalse(manifest["source"]["worktree_clean"])
        self.assertEqual(check["status"], "fail")
        self.assertIn(status.rstrip("\n"), check["detail"])

    def test_git_status_failure_reports_stderr_and_still_blocks(self):
        manifest, passed, check = self.doctor(
            status_returncode=128, status_stderr="fatal: detected dubious ownership\n"
        )
        self.assertFalse(passed)
        self.assertFalse(manifest["source"]["worktree_clean"])
        self.assertEqual(check["status"], "fail")
        self.assertIn("git status failed (exit 128)", check["detail"])
        self.assertIn("fatal: detected dubious ownership", check["detail"])

    def test_clean_checkout_still_passes(self):
        manifest, passed, check = self.doctor()
        self.assertTrue(passed)
        self.assertEqual(manifest["status"], "pass")
        self.assertTrue(manifest["source"]["worktree_clean"])
        self.assertEqual(
            check, {"check_id": "clean-tracked-source", "status": "pass", "detail": "clean"}
        )

    def test_inspection_does_not_create_or_change_run_files(self):
        before = {
            path.relative_to(self.source_root): path.read_bytes()
            for path in self.source_root.rglob("*") if path.is_file()
        }
        manifest, passed, _ = self.doctor(inspect_only=True)
        self.assertTrue(passed)
        self.assertEqual(manifest["status"], "pass")
        self.assertFalse((self.source_root / "output").exists())
        after = {
            path.relative_to(self.source_root): path.read_bytes()
            for path in self.source_root.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)

    def test_large_dirty_checkout_detail_is_bounded_and_marks_truncation(self):
        status = "?? first.py\n" + "?? " + "x" * 10000 + "\n?? omitted-last.py\n"
        _, passed, check = self.doctor(status_stdout=status)
        self.assertFalse(passed)
        self.assertIn("?? first.py", check["detail"])
        self.assertNotIn("omitted-last.py", check["detail"])
        self.assertIn("diagnostic truncated", check["detail"])
        self.assertIn("git status --short", check["detail"])
        self.assertLess(len(check["detail"]), 4500)

    def test_nohup_ignore_rule_covers_only_the_checkout_root(self):
        root = environment._command(
            SOURCE_ROOT, ["git", "check-ignore", "-q", "--", "nohup.out"]
        )
        nested = environment._command(
            SOURCE_ROOT, ["git", "check-ignore", "-q", "--", "nested/nohup.out"]
        )
        self.assertEqual(root.returncode, 0, root.stderr)
        self.assertEqual(nested.returncode, 1, nested.stderr)


if __name__ == "__main__":
    unittest.main()
