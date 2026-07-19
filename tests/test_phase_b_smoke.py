"""Focused checks for non-publication Phase B smoke artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from constants import PROTOCOL_ID, TRAINING_SEEDS
from phase_b_io import atomic_write_jsonl, iter_jsonl
from records import EntitySpan, StrictTriple, candidate_id_for
from smoke import clone_verdicts, prepare


def _candidate(seed: int = 42) -> dict:
    triple = StrictTriple(
        "example-a",
        EntitySpan(0, 0, "Object", "Door"),
        "selection",
        EntitySpan(3, 4, "Quality", "fire rated"),
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1:test",
        "training_seed": seed,
        "example_id": "example-a",
        "source_document_id": "document-a",
        "candidate_id": candidate_id_for(seed, triple),
        "head": triple.head.to_mapping(include_text=True),
        "relation": triple.relation,
        "tail": triple.tail.to_mapping(include_text=True),
        "triple_confidence": 0.75,
        "input_hashes": {"checkpoint": "0" * 64},
    }


def _rows(path: Path) -> list[dict]:
    return [value for _, value in iter_jsonl(path)]


class SmokeArtifactTests(unittest.TestCase):
    def test_prepare_and_clone_verdicts_create_all_pseudo_seed_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dev = root / "development.jsonl"
            generated_test = root / "test.jsonl"
            atomic_write_jsonl(source_dev, [_candidate()])
            atomic_write_jsonl(generated_test, [_candidate()])
            prepare(source_dev, generated_test, root)
            candidates = _rows(root / "predictions/smoke/test-candidates.jsonl")
            self.assertEqual([item["training_seed"] for item in candidates], list(TRAINING_SEEDS))
            self.assertEqual(len({item["candidate_id"] for item in candidates}), len(TRAINING_SEEDS))
            self.assertEqual(
                _rows(root / "predictions/smoke/test-live-candidate.jsonl"), [candidates[0]]
            )
            source_verdict = root / "verdicts.jsonl"
            atomic_write_jsonl(
                source_verdict,
                [
                    {
                        "protocol_id": PROTOCOL_ID,
                        "condition_id": "VER-SIMPLE",
                        "training_seed": 42,
                        "candidate_id": candidates[0]["candidate_id"],
                        "response_status": "valid_response",
                        "action": "KEEP",
                        "reason_code": "SUPPORTED",
                        "corrected": None,
                        "correction_validation_status": "not_applicable",
                        "raw_response_sha256": "1" * 64,
                        "prompt_sha256": "2" * 64,
                        "model_manifest_sha256": "3" * 64,
                        "decoding_sha256": "4" * 64,
                        "attempts": 1,
                        "error_category": None,
                        "telemetry": {"attempt_latencies_seconds": [0.1]},
                    }
                ],
            )
            destination = root / "pseudo-seed-verdicts.jsonl"
            clone_verdicts(source_verdict, root / "predictions/smoke/test-candidates.jsonl", destination)
            verdicts = _rows(destination)
            self.assertEqual([item["training_seed"] for item in verdicts], list(TRAINING_SEEDS))
            self.assertEqual([item["candidate_id"] for item in verdicts], [item["candidate_id"] for item in candidates])


if __name__ == "__main__":
    unittest.main()
