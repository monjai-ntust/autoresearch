import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GraphRagStandaloneTests(unittest.TestCase):
    def test_doctor_runs_from_copy_without_parent_or_research_submodules(self):
        output = ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            clone = Path(temporary) / "standalone"
            clone.mkdir()
            shutil.copy2(ROOT / "graph_rag.py", clone / "graph_rag.py")
            shutil.copytree(ROOT / "graph_rag_eval", clone / "graph_rag_eval")
            shutil.copytree(ROOT / "configs", clone / "configs")
            shutil.copytree(ROOT / "schemas", clone / "schemas")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(clone)
            result = subprocess.run(
                [
                    sys.executable,
                    "graph_rag.py",
                    "doctor",
                    "--config",
                    "configs/phase_b_graph_rag_synthetic.json",
                    "--run-id",
                    "standalone-proof",
                ],
                cwd=clone,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"status": "ready"', result.stdout)
            self.assertTrue(
                (clone / "output/standalone-proof/graph-rag/manifests/run.json").is_file()
            )
            self.assertFalse((clone / "papers").exists())
            self.assertFalse((clone / "research-logs").exists())


if __name__ == "__main__":
    unittest.main()
