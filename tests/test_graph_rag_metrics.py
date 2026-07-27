import unittest

from graph_rag_eval.adapters.synthetic import SyntheticAdapter
from graph_rag_eval.budget import Budget
from graph_rag_eval.evaluation.coupled import coupled_metrics
from graph_rag_eval.evaluation.generation import (
    ABSTENTION,
    FrozenTransformerGenerator,
    answer_metrics,
    support_metrics,
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
from graph_rag_eval.retrieval.graph import GraphRetriever
from graph_rag_eval.runner import _aggregate

from test_graph_rag_graph_retrieval import graphs


class GraphRagMetricTests(unittest.TestCase):
    def test_answer_and_abstention_metrics_match_hand_values(self):
        bundle = SyntheticAdapter().load()
        self.assertEqual(token_f1("steel frame", "steel frames"), 0.5)
        answerable = answer_metrics("steel frames", bundle.answers[0])
        self.assertEqual(answerable["exact_match"], 1.0)
        self.assertEqual(answerable["answer_correct"], 1.0)
        unanswerable = answer_metrics(ABSTENTION, bundle.answers[-1])
        # A correct abstention must not enter the pooled answer-accuracy mean, or
        # a condition that abstains everywhere reports inflated accuracy.
        self.assertNotIn("exact_match", unanswerable)
        self.assertNotIn("token_f1", unanswerable)
        self.assertEqual(unanswerable["abstention_correct"], 1.0)
        self.assertEqual(unanswerable["answer_correct"], 1.0)
        self.assertEqual(answer_metrics("800 mm", bundle.answers[-1])["answer_correct"], 0.0)

    def test_unmet_prerequisites_emit_no_numeric_value(self):
        bundle = SyntheticAdapter().load()
        support = support_metrics(("chunk-01",), bundle.evidence_sets, "q-unjudged")
        self.assertEqual(support["status"], "not_applicable")
        self.assertEqual(support["denominator"], 0)
        self.assertNotIn("support_precision", support)
        self.assertNotIn("support_recall", support)
        coupled = coupled_metrics({"status": "available"}, {"status": "available"}, support)
        self.assertEqual(coupled["status"], "not_applicable")
        self.assertNotIn("supported_answer", coupled)
        self.assertEqual(coupled["missing"], ["support"])

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
        self.assertEqual(metrics["sufficient_evidence_at_k"], 1.0)
        self.assertIsNotNone(metrics["ndcg_at_k"])

    def test_success_at_k_keeps_the_kilt_definition_apart_from_sufficiency(self):
        bundle = SyntheticAdapter().load()
        # q-02 needs both graph triples to be sufficient; retrieving one of them
        # is a KILT success but is not a sufficient evidence set.
        _, generated, _ = graphs()
        result = GraphRetriever(generated, max_hops=1).retrieve(
            bundle.questions[1].public_view(), Budget(1, 200)
        )
        metrics = retrieval_metrics(
            result,
            bundle.evidence_sets,
            bundle.relevance_judgments,
            capabilities=bundle.descriptor.capabilities,
            k=1,
        )
        self.assertEqual(metrics["success_at_k"], 1.0)
        self.assertEqual(metrics["sufficient_evidence_at_k"], 0.0)

    def test_missing_capability_is_not_applicable_not_zero(self):
        bundle, generated, gold = graphs()
        outcome = evaluate_intrinsic(
            generated, gold, capabilities=("documents",)
        )
        self.assertEqual(outcome.status, "not_applicable")
        self.assertIn("gold_triples", outcome.missing)

    def test_aggregate_excludes_unavailable_families_from_the_denominator(self):
        rows = [
            {
                "condition": "bm25_text",
                "retrieval": {"status": "available", "evidence_recall_at_k": 1.0, "k": 4},
                "answer": {"status": "available", "answer_correct": 1.0, "abstained": False},
                "support": {"status": "available", "support_recall": 1.0, "denominator": 1},
                "coupled": {"status": "available", "supported_answer": 1.0},
            },
            {
                "condition": "bm25_text",
                "retrieval": {"status": "not_applicable", "denominator": 0},
                "answer": {"status": "available", "answer_correct": 0.0, "abstained": True},
                "support": {"status": "not_applicable", "denominator": 0},
                "coupled": {"status": "not_applicable", "missing": ["support"]},
            },
        ]
        aggregate = {row["metric"]: row for row in _aggregate(rows)}
        self.assertEqual(aggregate["retrieval.evidence_recall_at_k"]["denominator"], 1)
        self.assertEqual(aggregate["retrieval.evidence_recall_at_k"]["value"], 1.0)
        self.assertEqual(aggregate["support.support_recall"]["denominator"], 1)
        self.assertEqual(aggregate["coupled.supported_answer"]["value"], 1.0)
        self.assertEqual(aggregate["answer.answer_correct"]["denominator"], 2)
        self.assertNotIn("answer.abstained", aggregate)
        self.assertNotIn("retrieval.k", aggregate)

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
