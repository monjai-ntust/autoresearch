import json
import shutil
import unittest
from pathlib import Path

from graph_rag_eval.runner import create_context, doctor, evaluate, prepare


ROOT = Path(__file__).resolve().parents[1]


class GraphRagRunnerTests(unittest.TestCase):
    run_ids = (
        "unittest-graph-rag-doctor",
        "unittest-graph-rag-prepare",
        "unittest-graph-rag-blocked",
    )

    def tearDown(self):
        for run_id in self.run_ids:
            path = ROOT / "output" / run_id
            if path.exists():
                shutil.rmtree(path)

    def test_synthetic_doctor_and_prepare_are_offline_and_complete(self):
        doctor_context = create_context(
            "configs/phase_b_graph_rag_synthetic.json", run_id=self.run_ids[0]
        )
        self.assertEqual(doctor(doctor_context)["status"], "ready")
        prepare_context = create_context(
            "configs/phase_b_graph_rag_synthetic.json", run_id=self.run_ids[1]
        )
        result = prepare(prepare_context)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["counts"]["questions"], 4)
        environment = json.loads(
            (prepare_context.run_root / "manifests/environment.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(environment["network_used"])
        self.assertFalse(environment["gpu_execution"])

    def test_code_accord_evaluation_fails_closed_without_domain_expert(self):
        context = create_context(
            "configs/phase_b_graph_rag_code_accord.json", run_id=self.run_ids[2]
        )
        result = evaluate(context, smoke=True)
        self.assertEqual(result["status"], "blocked")
        codes = {item["code"] for item in result["gates"]}
        self.assertIn("regime-q-human-independence-unavailable", codes)
        self.assertFalse(result["scientific_claims_enabled"])
        records = [
            json.loads(line)
            for line in (context.run_root / "metrics/aggregate.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertTrue(records)
        self.assertTrue(all(item["status"] == "not_applicable" for item in records))


if __name__ == "__main__":
    unittest.main()
