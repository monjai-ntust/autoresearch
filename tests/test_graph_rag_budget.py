import unittest

from graph_rag_eval.budget import Budget, apply_budget, assert_matched_budgets
from graph_rag_eval.retrieval.base import EvidenceItem


class GraphRagBudgetTests(unittest.TestCase):
    def test_duplicates_and_token_truncation_are_recorded(self):
        rows = (
            EvidenceItem("a", "chunk", "one two", 1.0, 1),
            EvidenceItem("a", "chunk", "one two", 0.9, 2),
            EvidenceItem("b", "chunk", "three four five", 0.8, 3),
        )
        selected, decision = apply_budget(rows, Budget(2, 4))
        self.assertEqual([item.evidence_id for item in selected], ["a"])
        self.assertEqual(decision.dropped_duplicate_ids, ("a",))
        self.assertEqual(decision.truncated_ids, ("b",))
        self.assertEqual(decision.realized_tokens, 2)

    def test_unmatched_comparison_budget_is_a_hard_failure(self):
        _, left = apply_budget((), Budget(1, 10))
        _, right = apply_budget((), Budget(2, 10))
        with self.assertRaisesRegex(ValueError, "unmatched"):
            assert_matched_budgets((left, right))


if __name__ == "__main__":
    unittest.main()
