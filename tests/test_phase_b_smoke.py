"""Focused checks for non-publication Phase B smoke artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from constants import PROTOCOL_ID, TRAINING_SEEDS
from phase_b import _write_same_run_seal
from phase_b_io import atomic_write_json, atomic_write_jsonl, iter_jsonl, sha256_file
from paths import RunLayout
from records import EntitySpan, StrictTriple, candidate_id_for
from smoke import clone_verdicts, main, prepare, seal_threshold


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
    def test_threshold_seal_binds_same_run_inputs_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout = RunLayout(Path(temporary), "same-run-threshold")
            layout.create()
            candidates = layout.resolve(
                "predictions/smoke/development-candidates.jsonl"
            )
            gold = layout.resolve("data-prepared/development-gold.jsonl")
            split = layout.resolve("data-prepared/split-manifest.json")
            threshold = layout.resolve(
                "predictions/smoke/threshold-selection.json"
            )
            atomic_write_jsonl(candidates, [_candidate()])
            atomic_write_jsonl(gold, [{"example_id": "example-a"}])
            atomic_write_json(split, {"test_ids": []})
            atomic_write_json(
                threshold,
                {
                    "selection_split": "development",
                    "used_test_labels": False,
                    "development_candidate_index_sha256": sha256_file(candidates),
                    "development_gold_sha256": sha256_file(gold),
                    "split_manifest_sha256": sha256_file(split),
                },
            )
            seal_threshold(layout)
            manifest = layout.resolve(
                "predictions/smoke/threshold-selection.manifest.json"
            )
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["run_id"],
                layout.run_id,
            )
            atomic_write_jsonl(candidates, [_candidate(), _candidate()])
            with self.assertRaisesRegex(ValueError, "same-run inputs"):
                seal_threshold(layout)

    def test_prepare_and_clone_verdicts_create_all_pseudo_seed_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = RunLayout(root, "same-run")
            layout.create()
            source_dev = layout.resolve("predictions/dev/seed-42-candidates.jsonl")
            generated_test = layout.resolve(
                "predictions/smoke/test-seed-42-candidates.jsonl"
            )
            generated_test.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_jsonl(source_dev, [_candidate()])
            atomic_write_jsonl(generated_test, [_candidate()])
            for split, source in (
                ("development", source_dev),
                ("test", generated_test),
            ):
                producer_path = layout.resolve(
                    f"manifests/model-generate-candidates-live-seed-42-{split}.json"
                )
                producer = {
                    "stage": "model-generate-candidates",
                    "execution_mode": "live",
                    "status": "completed",
                    "training_seed": 42,
                    "candidates_output": layout.relative_identity(source),
                    "inputs": {"prediction_ledger": None},
                }
                atomic_write_json(producer_path, producer)
                _write_same_run_seal(layout, producer_path, producer)
            prepare(layout, 42)
            candidates = _rows(layout.resolve("predictions/smoke/test-candidates.jsonl"))
            self.assertEqual([item["training_seed"] for item in candidates], list(TRAINING_SEEDS))
            self.assertEqual(len({item["candidate_id"] for item in candidates}), len(TRAINING_SEEDS))
            self.assertEqual(
                _rows(layout.resolve("predictions/smoke/test-live-candidate.jsonl")),
                [candidates[0]],
            )
            source_verdict = layout.resolve("verifier/smoke/simple/verdicts.jsonl")
            source_verdict.parent.mkdir(parents=True, exist_ok=True)
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
            verifier_manifest_path = layout.resolve(
                "manifests/verifier-smoke-simple-live.json"
            )
            verifier_manifest = {
                "condition_id": "VER-SIMPLE",
                "execution_mode": "live",
                "status": "completed",
                "outputs": {
                    layout.relative_identity(source_verdict): sha256_file(source_verdict)
                },
            }
            atomic_write_json(verifier_manifest_path, verifier_manifest)
            _write_same_run_seal(
                layout, verifier_manifest_path, verifier_manifest
            )
            clone_verdicts(layout, "simple")
            destination = layout.resolve(
                "verifier/smoke/simple/pseudo-seed-verdicts.jsonl"
            )
            verdicts = _rows(destination)
            self.assertEqual([item["training_seed"] for item in verdicts], list(TRAINING_SEEDS))
            self.assertEqual([item["candidate_id"] for item in verdicts], [item["candidate_id"] for item in candidates])

    def test_cli_rejects_the_removed_arbitrary_path_interface(self):
        with patch(
            "sys.argv",
            ["smoke.py", "prepare", "--source-dev", "other-run/candidates.jsonl"],
        ), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
