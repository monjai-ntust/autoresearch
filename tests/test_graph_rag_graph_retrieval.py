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
from graph_rag_eval.graphs.snapshots import build_snapshot
from graph_rag_eval.retrieval.graph import GraphRetriever


def graphs():
    adapter = SyntheticAdapter()
    bundle = adapter.load()
    common = dict(
        dataset_id=bundle.descriptor.dataset_id,
        entities=bundle.entities,
        relations=bundle.relations,
        descriptor_sha256=content_sha256(bundle.descriptor),
        corpus_sha256=content_sha256(bundle.chunks),
        adapter_sha256=content_sha256({"adapter": adapter.adapter_id}),
        extractor_sha256=content_sha256({"extractor": "fixture"}),
    )
    gold = build_snapshot(
        condition="gold", triples=bundle.triples,
        construction_recipe="gold", **common
    )
    generated = build_snapshot(
        condition="generated",
        triples=tuple(t for t in bundle.triples if t.triple_id in adapter.generated_triple_ids()),
        construction_recipe="generated", **common
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

    def test_matching_counts_missing_generated_fact(self):
        _, generated, gold = graphs()
        result = match_graphs(generated, gold)
        self.assertEqual(result.triples.true_positive, 3)
        self.assertEqual(result.triples.false_negative, 1)
        self.assertAlmostEqual(result.triples.recall, 0.75)

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
