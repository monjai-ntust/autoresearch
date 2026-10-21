"""Comparison disk estimates warn without preventing durable continuation."""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stages import comparison_matrix, encoder_comparison
from utils.common.paths import RunLayout
from utils.common import storage
from utils.encoder_comparison.config import load_comparison_config
from utils.rag import runner


ROOT = Path(__file__).resolve().parents[1]


class ComparisonStorageTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.layout = RunLayout(self.root, "matrix")
        self.layout.create()
        self.config = load_comparison_config(ROOT)
        self.members = {arm: f"matrix-{arm}" for arm in self.config.arms}

    def _run(self, *, free, existing):
        def dispatch(layout, selection, seed, *args, command_runner=None):
            command_runner(["fixture"])

        messages = io.StringIO()
        with (
            mock.patch.object(comparison_matrix, "matrix_identity", return_value={"members": self.members}),
            mock.patch.object(storage, "existing_storage_bytes", side_effect=existing) as estimate,
            mock.patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=free)),
            mock.patch.object(comparison_matrix, "checked_process") as process,
            mock.patch.object(encoder_comparison, "prepare"),
            mock.patch.object(encoder_comparison, "run_smoke", side_effect=dispatch),
            mock.patch.object(encoder_comparison, "run_training", side_effect=dispatch),
            mock.patch.object(encoder_comparison, "generate", side_effect=dispatch),
            mock.patch("utils.evaluation.reporting.report_comparison"),
            mock.patch("utils.evaluation.reporting.report_matrix"),
            redirect_stderr(messages),
        ):
            self.assertEqual(comparison_matrix.run_matrix(self.layout, self.config), 0)
        self.assertEqual(estimate.call_count, 64)
        self.assertEqual([call.kwargs["label"] for call in process.call_args_list],
            ["comparison-smoke"] * 4 + ["comparison-full"] * 64)
        self.assertTrue(all(call.kwargs["limit_seconds"] == 48 * 3600
            for call in process.call_args_list[4:]))
        return messages.getvalue()

    def test_default_budget_is_180_decimal_gigabytes(self):
        self.assertEqual(self.config.value["execution_limits"]["minimum_free_storage_bytes_before_full_run"],
            180_000_000_000)

    def test_low_space_warns_with_existing_output_subtracted_and_dispatches(self):
        self.config.value["execution_limits"]["minimum_free_storage_bytes_before_full_run"] = 200_000_000_000
        messages = self._run(free=159_000_000_000, existing=[40_000_000_000] * 64)
        self.assertEqual(messages.count("warning:"), 64)
        self.assertIn("200,000,000,000 bytes (200 GB) minus 40,000,000,000 existing", messages)
        self.assertIn("160,000,000,000 additional bytes estimated; 159,000,000,000 bytes free", messages)

    def test_exact_remaining_space_does_not_warn(self):
        self.assertEqual(self._run(free=140_000_000_000, existing=[40_000_000_000] * 64), "")

    def test_budget_covered_by_existing_output_clamps_remaining_to_zero(self):
        self.assertEqual(self._run(free=0, existing=[200_000_000_000] * 64), "")

    def test_existing_output_is_remeasured_before_each_dispatch(self):
        messages = self._run(free=100_000_000_000,
            existing=[40_000_000_000] + [80_000_000_000] * 63)
        self.assertEqual(messages.count("warning:"), 1)
        self.assertIn("140,000,000,000 additional", messages)

    def test_unavailable_storage_estimate_warns_and_dispatches(self):
        messages = self._run(free=0, existing=OSError("fixture stat failure"))
        self.assertEqual(messages.count("cannot estimate storage"), 64)
        self.assertIn("Continuing execution.", messages)

    def _stored_files(self):
        paths = []
        for run_id in (self.layout.run_id, *self.members.values()):
            path = RunLayout(self.root, run_id).resolve("checkpoints/fixture.pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint fixture\n")
            paths.append(path)
        unrelated = RunLayout(self.root, "research-other").resolve("checkpoint.pt")
        unrelated.parent.mkdir(parents=True)
        unrelated.write_bytes(b"unrelated\n" * 1000)
        expected = sum(info.st_blocks * 512 if hasattr(info, "st_blocks") else info.st_size
            for info in (path.stat() for path in paths))
        return paths, unrelated, expected

    def test_counts_only_matrix_and_members_once_by_allocated_size(self):
        paths, _, expected = self._stored_files()
        os.link(paths[0], paths[-1].with_name("hardlink.pt"))
        self.assertEqual(storage.existing_storage_bytes(self.layout.run_root,
            [self.layout.run_root, *(RunLayout(self.root, run_id).run_root for run_id in self.members.values())]), expected)

    def test_symlinked_files_and_directories_receive_no_credit(self):
        paths, unrelated, expected = self._stored_files()
        try:
            paths[0].with_name("foreign.pt").symlink_to(unrelated)
            paths[0].with_name("foreign-directory").symlink_to(unrelated.parent, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        self.assertEqual(storage.existing_storage_bytes(self.layout.run_root,
            [self.layout.run_root, *(RunLayout(self.root, run_id).run_root for run_id in self.members.values())]), expected)

    def test_files_on_another_device_receive_no_credit(self):
        paths, _, _ = self._stored_files()
        info = paths[0].stat()
        foreign = SimpleNamespace(st_mode=info.st_mode, st_dev=info.st_dev + 1,
            st_ino=info.st_ino, st_size=info.st_size)
        with mock.patch.object(Path, "lstat", return_value=foreign):
            self.assertEqual(storage.existing_storage_bytes(self.layout.run_root, [self.layout.run_root]), 0)

    def test_pipeline_has_its_own_budget_and_excludes_rag_output(self):
        import pipeline

        class StopAfterStorageCheck(Exception):
            pass

        with (
            mock.patch.object(storage, "warn_storage") as warning,
            mock.patch("stages.preparation.run", side_effect=StopAfterStorageCheck),
            self.assertRaises(StopAfterStorageCheck),
        ):
            pipeline._run_full_pipeline(self.layout, self.config.pipeline,
                ollama_url="http://localhost:11434", model_blob_source=None)
        warning.assert_called_once_with(self.layout.run_root, budget=100_000_000_000,
            roots=[self.layout.run_root], label="pipeline",
            excluded=(self.layout.resolve("table2-code-all"),))

    def test_excluded_rag_files_do_not_reduce_pipeline_requirement(self):
        path = self.layout.resolve("table2-code-all/responses.jsonl")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"response fixture\n")
        self.assertEqual(storage.existing_storage_bytes(self.layout.run_root,
            [self.layout.run_root], excluded=(path.parent,)), 0)

    def test_rag_has_its_own_budget_and_counts_only_rag_output(self):
        messages = io.StringIO()
        with (
            mock.patch.object(storage, "existing_storage_bytes", return_value=200_000_000) as existing,
            mock.patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=799_000_000)),
            mock.patch.object(runner.shutil, "which", return_value="curl"),
            redirect_stderr(messages),
        ):
            environment = runner.validate_environment(100, self.layout.run_root)
        existing.assert_called_once_with(self.layout.run_root,
            [self.layout.resolve("table2-code-all")], excluded=())
        self.assertEqual(environment["required_free_bytes"], 800_000_000)
        self.assertIn("rag: warning: storage budget 1,000,000,000 bytes (1 GB)", messages.getvalue())


if __name__ == "__main__":
    unittest.main()
