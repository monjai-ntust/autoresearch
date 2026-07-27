import unittest

from graph_rag_eval.adapters.synthetic import SyntheticAdapter
from graph_rag_eval.budget import Budget
from graph_rag_eval.contracts import QueryView, content_sha256
from graph_rag_eval.graphs.corruptions import (
    add_edges,
    drop_edges,
    merge_entities,
    relabel_relations,
    remove_provenance,
    rewire_endpoints,
    split_entity,
)
from graph_rag_eval.graphs.matching import match_graphs
from graph_rag_eval.graphs.snapshots import build_snapshot, structure_summary
from graph_rag_eval.retrieval.bm25 import BM25Retriever
from graph_rag_eval.retrieval.graph import GraphRetriever
from graph_rag_eval.retrieval.hybrid import ReciprocalRankFusionRetriever


def graphs():
    adapter = SyntheticAdapter()
    bundle = adapter.load()
    predicted = adapter.generated_graph(bundle)
    common = dict(
        dataset_id=bundle.descriptor.dataset_id,
        descriptor_sha256=content_sha256(bundle.descriptor),
        corpus_sha256=content_sha256(bundle.chunks),
        adapter_sha256=content_sha256({"adapter": adapter.adapter_id}),
        extractor_sha256=content_sha256({"extractor": "fixture"}),
    )
    gold = build_snapshot(
        condition="gold",
        entities=bundle.entities,
        relations=bundle.relations,
        triples=bundle.triples,
        construction_recipe="gold",
        **common,
    )
    generated = build_snapshot(
        condition="generated",
        entities=predicted.entities,
        relations=predicted.relations,
        triples=predicted.triples,
        construction_recipe=predicted.construction_recipe,
        **common,
    )
    return bundle, generated, gold


class GraphRagGraphRetrievalTests(unittest.TestCase):
    def test_question_only_linking_expansion_and_provenance(self):
        bundle, generated, _ = graphs()
        result = GraphRetriever(generated, max_hops=2).retrieve(
            bundle.questions[0].public_view(), Budget(4, 80)
        )
        self.assertIn("t-01", [item.evidence_id for item in result.items])
        self.assertIn("chunk-01", result.items[0].provenance_ids)
        self.assertTrue(result.seed_ids)
        self.assertNotIn("steel frames", result.query.public_metadata.values())

    def test_empty_expansion_is_typed_and_never_gold_repaired(self):
        bundle, generated, _ = graphs()
        result = GraphRetriever(generated, max_hops=2).retrieve(
            QueryView(
                bundle.descriptor.dataset_id,
                "q-empty",
                "Which astronomy observation is relevant?",
                "question_only",
            ),
            Budget(4, 80),
        )
        self.assertEqual(result.failure, "empty_entity_link")
        self.assertEqual(result.items, ())

    def test_matching_counts_both_missing_and_hallucinated_facts(self):
        _, generated, gold = graphs()
        result = match_graphs(generated, gold)
        self.assertEqual(result.triples.true_positive, 3)
        self.assertEqual(result.triples.false_negative, 1)
        self.assertEqual(result.triples.false_positive, 1)
        self.assertAlmostEqual(result.triples.recall, 0.75)
        # A predicted graph derived only as a gold subset would pin precision at
        # 1.0 and make extraction false positives unobservable.
        self.assertAlmostEqual(result.triples.precision, 0.75)
        self.assertEqual(result.entities.false_positive, 1)
        self.assertEqual(result.entities.false_negative, 1)
        self.assertEqual(result.relations.false_positive, 1)
        self.assertEqual(result.provenance.false_positive, 1)

    def test_structure_summary_reports_measured_graph_shape(self):
        _, generated, gold = graphs()
        gold_structure = structure_summary(gold)
        generated_structure = structure_summary(generated)
        self.assertEqual(gold_structure["triples"], 4)
        self.assertEqual(generated_structure["triples"], 4)
        self.assertEqual(gold_structure["isolated_entities"], 1)
        self.assertEqual(gold_structure["connected_components"], 4)
        self.assertAlmostEqual(gold_structure["average_degree"], 8 / 8)
        self.assertEqual(generated_structure["triples_with_provenance"], 4)

    def test_hybrid_reports_a_child_failure_even_when_the_other_child_answers(self):
        bundle, generated, _ = graphs()
        query = QueryView(
            bundle.descriptor.dataset_id,
            "q-empty",
            "Which astronomy observation is relevant?",
            "question_only",
        )
        graph = GraphRetriever(generated, max_hops=2)
        text = BM25Retriever(bundle.chunks)
        hybrid = ReciprocalRankFusionRetriever(graph, text)
        self.assertEqual(graph.retrieve(query, Budget(4, 80)).failure, "empty_entity_link")
        fused = hybrid.retrieve(query, Budget(4, 80))
        self.assertIn("question-only-graph-v1:empty_entity_link", fused.failure)

    def test_graph_ranking_breaks_seed_ties_by_question_overlap(self):
        bundle, generated, _ = graphs()
        # Both duct triples hang off the same seed at the same hop, so the seed
        # score alone cannot order them; only the question terms can.
        result = GraphRetriever(generated, max_hops=1).retrieve(
            QueryView(
                bundle.descriptor.dataset_id,
                "q-ducts",
                "Which aluminium covers apply to ventilation ducts?",
                "question_only",
            ),
            Budget(4, 200),
        )
        ranked = [item.evidence_id for item in result.items]
        self.assertEqual(ranked[0], "t-04")
        self.assertIn("t-90", ranked)
        self.assertLess(ranked.index("t-04"), ranked.index("t-90"))

    def test_all_corruptions_are_deterministic_and_parent_bound(self):
        _, graph, _ = graphs()
        functions = (
            lambda value: drop_edges(value, 0.5, 42),
            lambda value: add_edges(value, 0.5, 42),
            lambda value: relabel_relations(value, 0.5, 42),
            lambda value: rewire_endpoints(value, 0.5, 42),
            lambda value: remove_provenance(value, 0.5, 42),
            lambda value: merge_entities(value, 42),
            lambda value: split_entity(value, 42),
        )
        for function in functions:
            with self.subTest(function=function):
                first = function(graph)
                second = function(graph)
                self.assertEqual(first.graph_id, second.graph_id)
                self.assertEqual(first.parent_graph_id, graph.graph_id)
                self.assertIsNotNone(first.corruption_recipe)


if __name__ == "__main__":
    unittest.main()
