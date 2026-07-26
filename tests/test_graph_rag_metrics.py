import unittest

from graph_rag_eval.adapters.synthetic import SyntheticAdapter
from graph_rag_eval.budget import Budget
from graph_rag_eval.evaluation.generation import (
    ABSTENTION,
    FrozenTransformerGenerator,
    answer_metrics,
    token_f1,
)
from graph_rag_eval.evaluation.intrinsic import evaluate_intrinsic
from graph_rag_eval.evaluation.retrieval import retrieval_metrics
from graph_rag_eval.evaluation.statistics import (
    PairedObservation,
    clustered_paired_bootstrap,
    holm_adjust,
    mcnemar_counts,
)
from graph_rag_eval.retrieval.bm25 import BM25Retriever

from test_graph_rag_graph_retrieval import graphs


class GraphRagMetricTests(unittest.TestCase):
    def test_answer_and_abstention_metrics_match_hand_values(self):
        bundle = SyntheticAdapter().load()
        self.assertEqual(token_f1("steel frame", "steel frames"), 0.5)
        answerable = answer_metrics("steel frames", bundle.answers[0])
        self.assertEqual(answerable["exact_match"], 1.0)
        unanswerable = answer_metrics(ABSTENTION, bundle.answers[-1])
        self.assertEqual(unanswerable["exact_match"], 1.0)

    def test_retrieval_alternative_evidence_set_is_scored_without_answer_substring(self):
        bundle = SyntheticAdapter().load()
        result = BM25Retriever(bundle.chunks).retrieve(
            bundle.questions[0].public_view(), Budget(1, 80)
        )
        metrics = retrieval_metrics(
            result,
            bundle.evidence_sets,
            bundle.relevance_judgments,
            capabilities=bundle.descriptor.capabilities,
            k=1,
        )
        self.assertEqual(metrics["evidence_recall_at_k"], 1.0)
        self.assertEqual(metrics["success_at_k"], 1.0)
        self.assertIsNotNone(metrics["ndcg_at_k"])

    def test_missing_capability_is_not_applicable_not_zero(self):
        bundle, generated, gold = graphs()
        outcome = evaluate_intrinsic(
            generated, gold, capabilities=("documents",)
        )
        self.assertEqual(outcome.status, "not_applicable")
        self.assertIn("gold_triples", outcome.missing)

    def test_clustered_bootstrap_is_deterministic_and_cluster_aware(self):
        rows = (
            PairedObservation("q1", "d1", 0.0, 1.0),
            PairedObservation("q2", "d1", 0.0, 1.0),
            PairedObservation("q3", "d2", 1.0, 0.0),
        )
        first = clustered_paired_bootstrap(rows, resamples=500, seed=7)
        second = clustered_paired_bootstrap(rows, resamples=500, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 2)
        descriptive = clustered_paired_bootstrap(rows[:2], resamples=10)
        self.assertEqual(descriptive["status"], "descriptive_only")

    def test_comparison_helpers_keep_descriptive_and_family_semantics(self):
        self.assertEqual(
            mcnemar_counts([True, False], [False, True])["status"],
            "descriptive_only",
        )
        adjusted = holm_adjust((0.01, 0.04, 0.20))
        self.assertEqual(adjusted, (0.03, 0.08, 0.2))

    def test_frozen_generator_requires_revision_and_public_prompt_fields(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            FrozenTransformerGenerator(
                model_id="example/model",
                revision="latest",
                cache_dir="output/cache",
                prompt_template="{question}\n{evidence}",
            )
        with self.assertRaisesRegex(ValueError, "question and evidence"):
            FrozenTransformerGenerator(
                model_id="example/model",
                revision="0123456789abcdef",
                cache_dir="output/cache",
                prompt_template="{question}",
            )
        generator = FrozenTransformerGenerator(
            model_id="example/model",
            revision="0123456789abcdef",
            cache_dir="output/cache",
            prompt_template="{question}\n{evidence}",
        )
        self.assertEqual(generator.revision, "0123456789abcdef")


if __name__ == "__main__":
    unittest.main()
