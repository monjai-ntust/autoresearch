"""Offline regression tests for publication-critical artifact contracts."""

import math
import unittest

from data.scierc import BIO_TAG2ID
from eval.triple_f1 import _bio_to_spans, _is_valid_bio_transition, _prf
from graph_construction import build_table_graphs, canonical_json, project_graph
from records import StrictTriple, candidate_id_for


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


def _graph_inputs():
    example_id = "example-001"
    source_id = "document-001"
    prepared_sha = "1" * 64
    strict_mapping = {
        "head": {"start": 2, "end": 2, "type": "Quality", "text": "insulation"},
        "relation": "part-of",
        "tail": {"start": 0, "end": 0, "type": "Object", "text": "Wall"},
    }
    strict = StrictTriple.from_mapping(
        strict_mapping, example_id=example_id, label="fixture triple"
    )
    candidate_id = candidate_id_for(42, strict)
    prepared = [{
        "dataset_id": "CODE-ACCORD-v1.0.0",
        "example_id": example_id,
        "source_document_id": source_id,
        "content": "Wall contains insulation .",
        "words": ["Wall", "contains", "insulation", "."],
    }]
    gold = [{
        "protocol_id": "B04-PATH-A-1.3",
        "split_id": "CODE-SPLIT-1",
        "example_id": example_id,
        "source_document_id": source_id,
        "gold_triples": [strict_mapping],
        "input_hashes": {"prepared_sentences": prepared_sha},
    }]
    candidates = [{
        "protocol_id": "B04-PATH-A-1.3",
        "split_id": "CODE-SPLIT-1",
        "training_seed": 42,
        "example_id": example_id,
        "source_document_id": source_id,
        "candidate_id": candidate_id,
        **strict_mapping,
        "triple_confidence": 0.75,
        "input_hashes": {"prepared_sentences": prepared_sha},
    }]
    verdicts = [{
        "protocol_id": "B04-PATH-A-1.3",
        "condition_id": "VER-CORRECTIVE",
        "training_seed": 42,
        "candidate_id": candidate_id,
        "response_status": "valid_response",
        "action": "KEEP",
        "reason_code": "SUPPORTED",
        "corrected": None,
        "correction_validation_status": "not_applicable",
        "raw_response_sha256": "2" * 64,
        "prompt_sha256": "3" * 64,
        "model_manifest_sha256": "4" * 64,
        "decoding_sha256": "5" * 64,
        "attempts": 1,
        "error_category": None,
        "telemetry": {},
    }]
    return prepared, gold, candidates, verdicts, prepared_sha


def _build_graphs(run_id: str):
    prepared, gold, candidates, verdicts, prepared_sha = _graph_inputs()
    return build_table_graphs(
        run_id=run_id,
        prepared_rows=prepared,
        gold_rows=gold,
        candidate_rows=candidates,
        corrective_rows=verdicts,
        selected_threshold=0.25,
        seed=42,
        prepared_sha256=prepared_sha,
        split_sha256="6" * 64,
        gold_sha256="7" * 64,
        candidate_sha256="8" * 64,
        corrective_sha256="9" * 64,
    )


class CanonicalGraphContractTests(unittest.TestCase):
    def test_fixed_fixture_is_deterministic_and_preserves_all_three_conditions(self):
        first = _build_graphs("research-run-alpha")
        second = _build_graphs("research-run-alpha")
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(list(first), ["confidence", "corrective", "gold"])
        for snapshot in first.values():
            self.assertEqual(len(snapshot["entities"]), 2)
            self.assertEqual(len(snapshot["relations"]), 1)
            self.assertEqual(len(snapshot["triples"]), 1)
            self.assertEqual(
                project_graph(snapshot)["edges"][0]["relation"], "part-of"
            )

    def test_run_namespace_changes_identity_but_not_evaluator_projection(self):
        first = _build_graphs("research-run-alpha")
        other = _build_graphs("research-run-beta")
        self.assertNotEqual(
            first["confidence"]["graph_id"], other["confidence"]["graph_id"]
        )
        for condition in first:
            self.assertEqual(
                project_graph(first[condition]), project_graph(other[condition])
            )


if __name__ == "__main__":
    unittest.main()
