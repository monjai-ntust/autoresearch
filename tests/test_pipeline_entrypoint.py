"""Static contracts for the single Python table-pipeline entry point."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from paths import discover_source_root


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
            "run_doctor",
            "reconcile_section5_evidence",
            "fetch_run",
            "prepare_run",
            "plan_training",
            'split="development"',
            "select_threshold",
            "prepare_verifier_pilot",
            "run_verifier_pilot",
            'split="test"',
            "score_run",
            "table2_runner.run_command",
        )
        positions = [body.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_trainer_is_reached_only_through_the_pipeline(self):
        model_source = (SOURCE_ROOT / "model.py").read_text(encoding="utf-8")
        trainer_source = (SOURCE_ROOT / "train_span.py").read_text(encoding="utf-8")
        self.assertIn('"pipeline.py",', model_source)
        self.assertIn('"_train-encoder",', model_source)
        self.assertNotIn('if __name__ == "__main__"', trainer_source)
        self.assertIn('if __name__ == "__main__"', PIPELINE_SOURCE)

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
