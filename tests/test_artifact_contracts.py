"""Offline regression tests for publication-critical artifact contracts."""

import importlib
import math
import unittest

from provenance.build_kg import (
    build_entity_clusters,
    filter_triple,
    is_substring_match,
    normalize_entity,
    pluralize_match,
    string_similarity,
)
from data.scierc import BIO_TAG2ID
from provenance.diagnose_evidence_paths import build_adjacency, norm, path_candidates, sim
from eval.triple_f1 import _bio_to_spans, _is_valid_bio_transition, _prf
from provenance.rule_engine import filter_triples, is_valid_triple
from provenance.verify_triples_llm import parse_verdict_correct, parse_verdict_simple


class MetricContractTests(unittest.TestCase):
    def test_prf_uses_zero_safe_micro_formula(self):
        precision, recall, f1 = _prf(tp=2, fp=1, fn=2)
        self.assertTrue(math.isclose(precision, 2 / 3))
        self.assertTrue(math.isclose(recall, 1 / 2))
        self.assertTrue(math.isclose(f1, 4 / 7))
        self.assertEqual(_prf(0, 0, 0), (0.0, 0.0, 0.0))

    def test_bio_decoder_repairs_invalid_starts_and_type_changes(self):
        ids = [
            BIO_TAG2ID["I-Task"],
            BIO_TAG2ID["I-Task"],
            BIO_TAG2ID["O"],
            BIO_TAG2ID["B-Method"],
            BIO_TAG2ID["I-Metric"],
        ]
        self.assertEqual(
            _bio_to_spans(ids),
            [(0, 1, "Task"), (3, 3, "Method"), (4, 4, "Metric")],
        )

    def test_bio_transition_constraint_requires_matching_type(self):
        self.assertTrue(
            _is_valid_bio_transition(
                BIO_TAG2ID["B-Task"], BIO_TAG2ID["I-Task"]
            )
        )
        self.assertFalse(
            _is_valid_bio_transition(BIO_TAG2ID["O"], BIO_TAG2ID["I-Task"])
        )
        self.assertFalse(
            _is_valid_bio_transition(
                BIO_TAG2ID["B-Task"], BIO_TAG2ID["I-Method"]
            )
        )


class RuleContractTests(unittest.TestCase):
    def test_rules_normalize_case_and_reject_unknown_types(self):
        self.assertTrue(is_valid_triple("object", "NECESSITY", "quality"))
        self.assertFalse(is_valid_triple("Unknown", "necessity", "quality"))
        self.assertFalse(is_valid_triple("Task", "used-for", "Method"))

    def test_filter_preserves_order_and_reports_rejections(self):
        valid = {
            "head": "door",
            "head_type": "Object",
            "relation": "necessity",
            "tail": "fire resistant",
            "tail_type": "Quality",
            "confidence": 0.9,
        }
        invalid = {**valid, "head_type": "Unknown"}
        retained, rejected = filter_triples([valid, invalid])
        self.assertEqual(retained, [valid])
        self.assertEqual(rejected, 1)


class GraphContractTests(unittest.TestCase):
    def test_entity_normalization_and_guarded_similarity(self):
        self.assertEqual(
            normalize_entity("The machine translation -lrb- MT -rrb-"),
            "machine translation",
        )
        self.assertEqual(string_similarity("neural model", "neural network"), 1 / 3)
        self.assertTrue(is_substring_match("graph neural model", "neural model"))
        self.assertFalse(is_substring_match("model", "neural model"))
        self.assertTrue(pluralize_match("graph model", "graph models"))

    def test_cluster_canonical_is_the_most_informative_mention(self):
        mapping, clusters = build_entity_clusters(
            ["the neural network", "neural network", "dataset"],
            sim_threshold=0.8,
        )
        self.assertEqual(mapping["the neural network"], "the neural network")
        self.assertEqual(mapping["neural network"], "the neural network")
        self.assertEqual(len(clusters), 2)

    def test_filter_modes_encode_the_documented_threshold_contract(self):
        triple = {"triple_conf": 0.5, "llm_verdict": "correct"}
        self.assertTrue(filter_triple(triple, "all", 0.9))
        self.assertTrue(filter_triple(triple, "confidence", 0.5))
        self.assertFalse(filter_triple(triple, "confidence", 0.6))
        self.assertTrue(filter_triple(triple, "llm", 0.9))
        self.assertTrue(filter_triple(triple, "verified", 0.5))

    def test_evidence_paths_are_bidirectional_and_bounded(self):
        edges = [
            {"head": "A", "relation": "part-of", "tail": "B"},
            {"head": "B", "relation": "part-of", "tail": "C"},
        ]
        adjacency = build_adjacency(edges)
        paths = path_candidates(adjacency, [norm("A")], max_hops=2)
        self.assertIn([edges[0]], paths)
        self.assertIn([edges[0], edges[1]], paths)
        self.assertEqual(sim("Graph-based model", "graph based model"), 1.0)


class VerifierContractTests(unittest.TestCase):
    def test_simple_verdict_parser_is_prefix_based(self):
        self.assertEqual(parse_verdict_simple("Yes."), {"action": "keep"})
        self.assertEqual(parse_verdict_simple("No - unsupported"), {"action": "discard"})
        self.assertEqual(
            parse_verdict_simple("Uncertain"), {"action": "uncertain"}
        )
        self.assertEqual(parse_verdict_simple("Maybe"), {"action": "unknown"})

    def test_correct_verdict_parser_normalizes_relation(self):
        self.assertEqual(
            parse_verdict_correct("CORRECT: ('encoder', used for, \"extraction\")"),
            {
                "action": "correct",
                "corrected_head": "encoder",
                "corrected_relation": "USED-FOR",
                "corrected_tail": "extraction",
            },
        )
        self.assertEqual(
            parse_verdict_correct("CORRECT: (only, two)"),
            {"action": "correct_failed", "raw": "only, two"},
        )
        self.assertEqual(parse_verdict_correct("KEEP\nexplanation"), {"action": "keep"})
        self.assertEqual(parse_verdict_correct("DISCARD"), {"action": "discard"})


class ImportSafetyTests(unittest.TestCase):
    def test_retained_utilities_do_not_execute_on_import(self):
        for module_name in (
            "provenance.bench_gpu",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
