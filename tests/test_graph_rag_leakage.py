import dataclasses
import json
import unittest
from pathlib import Path

from graph_rag_eval.adapters.synthetic import SyntheticAdapter
from graph_rag_eval.budget import Budget
from graph_rag_eval.contracts import AnswerAlias, canonical_json
from graph_rag_eval.retrieval.bm25 import BM25Retriever
from graph_rag_eval.runner import LeakageError, _assert_no_private_leakage


ROOT = Path(__file__).resolve().parents[1]


class GraphRagLeakageTests(unittest.TestCase):
    def test_private_canaries_cannot_enter_query_or_index_identity(self):
        bundle = SyntheticAdapter().load()
        canary = "PRIVATE_ANSWER_CANARY_839217"
        private = dataclasses.replace(
            bundle.answers[0], normalized_answers=(canary,)
        )
        query = bundle.questions[0].public_view()
        retriever = BM25Retriever(bundle.chunks)
        before = retriever.index_fingerprint
        result = retriever.retrieve(query, Budget(4, 80))
        self.assertNotIn(canary, canonical_json(query))
        self.assertNotIn(canary, canonical_json(result.query))
        self.assertNotIn(canary, before)
        self.assertIn(canary, canonical_json(private))
        self.assertEqual(before, retriever.index_fingerprint)

    def test_config_freezes_complete_public_private_boundary(self):
        config = json.loads(
            (ROOT / "configs/phase_b_graph_rag_synthetic.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            config["protocol"]["question_query_fields"],
            ["text", "public_metadata"],
        )
        self.assertTrue(
            {
                "answers",
                "answer_aliases",
                "gold_relation",
                "gold_triple",
                "gold_path",
                "evidence_ids",
                "source_ids",
                "condition",
            }
            <= set(config["protocol"]["forbidden_query_fields"])
        )

    def test_runner_sentinel_stops_when_private_material_reaches_the_query(self):
        bundle = SyntheticAdapter().load()
        question = bundle.questions[0]
        forbidden = {"answers", "evidence_ids", "source_ids"}
        _assert_no_private_leakage(bundle, question.public_view(), question, forbidden)
        answer = bundle.answers[0].normalized_answers[0]
        leaked_question = dataclasses.replace(
            question, public_metadata={"hint": f"the answer is {answer}"}
        )
        with self.assertRaisesRegex(LeakageError, "private answer or evidence identity"):
            _assert_no_private_leakage(
                bundle, leaked_question.public_view(), leaked_question, forbidden
            )
        forbidden_field = dataclasses.replace(
            question, public_metadata={"evidence_ids": ["chunk-01"]}
        )
        with self.assertRaisesRegex(LeakageError, "forbidden fields"):
            _assert_no_private_leakage(
                bundle, forbidden_field.public_view(), forbidden_field, forbidden
            )
        rewritten = dataclasses.replace(question, text="a different question")
        with self.assertRaisesRegex(LeakageError, "diverges from the public question"):
            _assert_no_private_leakage(
                bundle, question.public_view(), rewritten, forbidden
            )

    def test_no_answer_alias_type_is_accepted_by_retriever(self):
        bundle = SyntheticAdapter().load()
        retriever = BM25Retriever(bundle.chunks)
        with self.assertRaises(AttributeError):
            retriever.retrieve(
                AnswerAlias("x", "q", ("private",), "entity"), Budget(1, 10)
            )


if __name__ == "__main__":
    unittest.main()
