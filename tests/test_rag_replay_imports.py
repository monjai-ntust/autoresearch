"""Exercise RAG's lazy replay imports after the Phase F file relocation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.common.artifact_io import (
    DataContractError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)
from utils.rag import runner


RUN_ID = "rag-replay-import-test"
PIPELINE = {"protocol_id": "B04-PATH-A-1.3", "matcher_id": "CODE-STRICT-1"}
REPLAY_ERRORS = (DataContractError, OSError, ValueError)


class RagReplayImportTests(unittest.TestCase):
    def test_candidate_stage_calls_relocated_validator_and_preserves_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / RUN_ID
            prepared = "data-prepared/test.jsonl"
            prediction = "predictions/test/seed-42-prediction-ledger.jsonl"
            candidates = "predictions/test/seed-42-candidates.jsonl"
            for relative in (prepared, prediction):
                atomic_write_jsonl(run_dir / relative, [{}])
            hashes = {
                "checkpoint_manifest": "a" * 64,
                "prepared_sentences": runner.sha256_file(run_dir / prepared),
                "prediction_ledger": runner.sha256_file(run_dir / prediction),
            }
            rows = [{"training_seed": 42, "input_hashes": hashes}]
            atomic_write_jsonl(run_dir / candidates, rows)
            atomic_write_json(
                run_dir / "manifests/model-generate-candidates-live-seed-42-test.json",
                {
                    **PIPELINE,
                    "run_id": RUN_ID,
                    "stage": "model-generate-candidates",
                    "status": "completed",
                    "execution_mode": "live",
                    "training_seed": 42,
                    "candidates_output": candidates,
                    "inputs": {
                        "checkpoint_manifest": {
                            "path": "checkpoints/seed-42/checkpoint-manifest.json",
                            "sha256": hashes["checkpoint_manifest"],
                        },
                        "prepared_sentences": {
                            "path": prepared,
                            "sha256": hashes["prepared_sentences"],
                        },
                        "prediction_ledger": {
                            "path": prediction,
                            "sha256": hashes["prediction_ledger"],
                        },
                    },
                    "checkpoint": {
                        "sha256": "b" * 64,
                        "split_manifest_sha256": "c" * 64,
                    },
                    "candidate_count": 1,
                    "sentence_count": 1,
                    "ledger_sentence_count": 1,
                },
            )

            def validate():
                return runner._validate_candidate_generation(
                    run_dir, RUN_ID, PIPELINE, {}, seed=42, split="test",
                    checkpoint_manifest_relative="checkpoints/seed-42/checkpoint-manifest.json",
                    checkpoint_manifest_sha256=hashes["checkpoint_manifest"],
                    checkpoint_sha256="b" * 64, max_span_width=8,
                    prepared_relative=prepared,
                    prepared_sha256=hashes["prepared_sentences"],
                    expected_sentence_count=1, split_manifest_sha256="c" * 64,
                    require_seal=False,
                )

            with mock.patch("utils.encoder.model.validate_prediction_artifacts") as replay:
                self.assertEqual(validate(), (candidates, rows))
                replay.assert_called_once_with(
                    sentences_path=run_dir / prepared,
                    prediction_ledger_path=run_dir / prediction,
                    candidates_path=run_dir / candidates,
                    training_seed=42, max_span_width=8, input_hashes=hashes,
                )
                for error_class in REPLAY_ERRORS:
                    error = error_class("invalid replay")
                    replay.side_effect = error
                    with self.subTest(error=type(error).__name__), self.assertRaisesRegex(
                        runner.PhaseEError, "prediction/candidate replay is invalid"
                    ) as caught:
                        validate()
                    self.assertIs(caught.exception.__cause__, error)

            # Also enter the real offline validator: malformed prepared data must
            # fail closed as a domain error, not an import/unbound-local error.
            with self.assertRaisesRegex(
                runner.PhaseEError, "prediction/candidate replay is invalid"
            ) as caught:
                validate()
            self.assertIsInstance(caught.exception.__cause__, DataContractError)

    def test_verifier_stages_call_relocated_validator_and_preserve_errors(self):
        for mode in ("simple", "corrective"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / RUN_ID
                prefix = f"verifier/{mode}"
                condition = "VER-" + mode.upper()
                model_digest = "a" * 64
                blob_bytes = b"fixture-model-blob\n"
                blob_digest = runner.sha256_bytes(blob_bytes)
                model_path = f"inputs/ollama/blobs/sha256-{blob_digest}"
                artifacts = {
                    "prepared_test": "data-prepared/test.jsonl",
                    "candidates": "predictions/test/candidates.jsonl",
                    f"{mode}_verifier_manifest": f"manifests/verifier-{mode}-live.json",
                    f"{mode}_verdicts": f"{prefix}/verdicts.jsonl",
                    f"{mode}_environment": f"{prefix}/environment-manifest.json",
                }
                config = {"verifier": {f"{mode}_bundle_sha256": "c" * 64}}
                decoding = runner._verifier_decoding(config)
                decoding_digest = runner._verifier_decoding_sha256(config)
                candidate = {"training_seed": 42, "candidate_id": "fixture-candidate"}
                candidate_digest = runner.sha256_bytes(
                    runner.canonical_json_bytes(candidate)
                )
                payload = {"model": "fixture-model:tag", **decoding}
                cache_identity = {
                    "model_manifest_sha256": model_digest,
                    "prompt_sha256": "c" * 64,
                    "decoding_sha256": decoding_digest,
                    "mode": mode,
                    "candidate_sha256": candidate_digest,
                }
                request = {
                    **candidate, "condition_id": condition,
                    "protocol_id": PIPELINE["protocol_id"],
                    **cache_identity, "payload": payload,
                    "request_sha256": runner.sha256_bytes(
                        runner.canonical_json_bytes(payload)
                    ),
                    "cache_key": runner.sha256_bytes(
                        runner.canonical_json_bytes(cache_identity)
                    ),
                }
                verdict = {**request, "model_manifest_sha256": model_digest}
                for relative, rows in (
                    (artifacts["prepared_test"], [{}]),
                    (artifacts["candidates"], [candidate]),
                    (f"{prefix}/requests.jsonl", [request]),
                    (f"{prefix}/responses.jsonl", [{}]),
                    (artifacts[f"{mode}_verdicts"], [verdict]),
                ):
                    atomic_write_jsonl(run_dir / relative, rows)
                tags, show = {"models": []}, {"details": {}}
                atomic_write_json(run_dir / f"{prefix}/model/tags.json", tags)
                atomic_write_json(run_dir / f"{prefix}/model/show.json", show)
                atomic_write_json(run_dir / f"{prefix}/model/ollama-modelfile.txt", blob_digest)
                atomic_write_bytes(run_dir / model_path, blob_bytes)
                environment = {
                    "model": {
                        "tags_response_sha256": runner.sha256_file(run_dir / f"{prefix}/model/tags.json"),
                        "show_response_sha256": runner.sha256_file(run_dir / f"{prefix}/model/show.json"),
                    }
                }
                atomic_write_json(run_dir / artifacts[f"{mode}_environment"], environment)
                identity = {
                    "name": payload["model"], "tag_digest": model_digest,
                    "blob_sha256": blob_digest, "verifier_condition_id": condition,
                }
                inputs = {
                    relative: runner.sha256_file(run_dir / relative)
                    for relative in (artifacts["prepared_test"], artifacts["candidates"])
                }
                inputs[model_path] = blob_digest
                outputs = {
                    relative: runner.sha256_file(run_dir / relative)
                    for relative in (
                        artifacts[f"{mode}_verdicts"], artifacts[f"{mode}_environment"],
                        f"{prefix}/requests.jsonl", f"{prefix}/responses.jsonl",
                    )
                }
                manifest = {
                    "condition_id": condition, "execution_mode": "live",
                    "status": "completed", "protocol_id": PIPELINE["protocol_id"],
                    "inputs": inputs, "outputs": outputs, "verdict_count": 1,
                    **{key: cache_identity[key] for key in (
                        "model_manifest_sha256", "prompt_sha256", "decoding_sha256"
                    )},
                }
                atomic_write_json(run_dir / artifacts[f"{mode}_verifier_manifest"], manifest)
                ledger = {
                    relative: {"bytes": (run_dir / relative).stat().st_size, "sha256": digest}
                    for relative, digest in inputs.items()
                }

                def validate():
                    return runner._validate_verifier_stage(
                        run_dir, RUN_ID, ledger, artifacts, PIPELINE, config, mode,
                        require_seal=False,
                    )

                # Raw model evidence is unrelated to these replay import gates;
                # candidate/request/verdict binding checks above stay active.
                with mock.patch.object(runner, "load_verifier_model_identity", return_value=identity), \
                     mock.patch.object(runner, "_validate_stored_model_response"):
                    with mock.patch("utils.verifier.verifier.validate_verifier_artifacts") as replay:
                        self.assertEqual(validate()[0], manifest)
                        replay.assert_called_once_with(
                            mode=mode, sentences_path=run_dir / artifacts["prepared_test"],
                            candidates_path=run_dir / artifacts["candidates"],
                            requests_path=run_dir / f"{prefix}/requests.jsonl",
                            responses_path=run_dir / f"{prefix}/responses.jsonl",
                            verdicts_path=run_dir / artifacts[f"{mode}_verdicts"],
                        )
                        for error_class in REPLAY_ERRORS:
                            error = error_class("invalid replay")
                            replay.side_effect = error
                            with self.subTest(error=type(error).__name__), self.assertRaisesRegex(
                                runner.PhaseEError, "response/verdict replay is invalid"
                            ) as caught:
                                validate()
                            self.assertIs(caught.exception.__cause__, error)
                    with self.assertRaisesRegex(
                        runner.PhaseEError, "response/verdict replay is invalid"
                    ) as caught:
                        validate()
                    self.assertIsInstance(caught.exception.__cause__, DataContractError)


if __name__ == "__main__":
    unittest.main()
