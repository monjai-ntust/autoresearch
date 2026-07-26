import unittest

from graph_rag_eval.adapters.code_accord import CodeAccordAdapter
from graph_rag_eval.adapters.synthetic import SyntheticAdapter
from graph_rag_eval.capabilities import CapabilityUnavailable


class GraphRagAdapterTests(unittest.TestCase):
    def test_synthetic_adapter_exercises_answerable_unanswerable_and_both_regimes(self):
        adapter = SyntheticAdapter()
        self.assertTrue(adapter.validate().ready)
        bundle = adapter.load()
        self.assertEqual(len(bundle.documents), 4)
        self.assertEqual(
            {item.regime for item in bundle.questions},
            {"diagnostic_fact_probe", "question_only"},
        )
        self.assertTrue(any(item.abstention_expected for item in bundle.answers))
        self.assertEqual(adapter.generated_triple_ids(), ("t-01", "t-02", "t-04"))

    def test_code_accord_declares_missing_gold_and_regime_q_instead_of_zero(self):
        report = CodeAccordAdapter().validate()
        self.assertTrue(report.ready)
        codes = {item.code for item in report.issues}
        self.assertIn("incomplete-corpus", codes)
        self.assertIn("no-domain-expert", codes)
        self.assertNotIn("questions", report.capabilities)
        unavailable = CapabilityUnavailable.for_required(
            "regime_q",
            ("questions", "evidence_sets"),
            report.capabilities,
        )
        self.assertIsNotNone(unavailable)
        self.assertEqual(unavailable.status, "not_applicable")


if __name__ == "__main__":
    unittest.main()
