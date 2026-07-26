import hashlib
import json
import shutil
import unittest
from pathlib import Path

from graph_rag_eval.runner import create_context, evaluate
from graph_rag_eval.trace import confined_path, resolve_within


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GraphRagArtifactTests(unittest.TestCase):
    run_id = "unittest-graph-rag-artifacts"

    def setUp(self):
        self.root = ROOT / "output" / self.run_id
        if self.root.exists():
            shutil.rmtree(self.root)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_complete_output_contract_and_deterministic_derivatives(self):
        context = create_context(
            "configs/phase_b_graph_rag_synthetic.json", run_id=self.run_id
        )
        result = evaluate(context)
        self.assertEqual(result["status"], "synthetic_validation_complete")
        run_root = context.run_root
        expected = (
            "manifests/environment.json",
            "manifests/dataset.json",
            "manifests/config.json",
            "manifests/graph.json",
            "manifests/index.json",
            "manifests/question-set.json",
            "manifests/run.json",
            "canonical/documents.jsonl",
            "canonical/chunks.jsonl",
            "canonical/entities.jsonl",
            "canonical/relations.jsonl",
            "canonical/triples.jsonl",
            "canonical/questions.jsonl",
            "traces/retrieval.jsonl",
            "traces/generation.jsonl",
            "traces/failures.jsonl",
            "metrics/per-question.jsonl",
            "metrics/per-document.jsonl",
            "metrics/per-seed.jsonl",
            "metrics/aggregate.jsonl",
            "metrics/comparisons.jsonl",
            "tables/synthetic-results.csv",
            "figures/synthetic-supported-answer.svg",
        )
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((run_root / relative).is_file())
        index_manifests = list((run_root / "indexes" / "bm25_text").glob("*/manifest.json"))
        self.assertEqual(len(index_manifests), 1)
        tracked = (
            "canonical/documents.jsonl",
            "metrics/per-question.jsonl",
            "metrics/aggregate.jsonl",
            "tables/synthetic-results.csv",
            "figures/synthetic-supported-answer.svg",
        )
        before = {relative: digest(run_root / relative) for relative in tracked}
        evaluate(context)
        after = {relative: digest(run_root / relative) for relative in tracked}
        self.assertEqual(before, after)
        for relative in ("traces/retrieval.jsonl", "traces/generation.jsonl"):
            for line in (run_root / relative).read_text(encoding="utf-8").splitlines():
                self.assertIn("schema_version", json.loads(line))

    def test_output_containment_rejects_path_syntax(self):
        with self.assertRaises(ValueError):
            confined_path(ROOT, "../escape")
        root = confined_path(ROOT, "safe")
        with self.assertRaises(ValueError):
            resolve_within(root, "../../escape")


if __name__ == "__main__":
    unittest.main()
