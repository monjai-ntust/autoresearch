"""Synthetic regression tests for the frozen B-06 verifier contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import load_pipeline_config
from constants import PROTOCOL_ID
from phase_b_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)
from paths import RunLayout, discover_source_root
from records import StrictTriple, candidate_id_for
from verifier import HttpResult, _verify_live_model, run_verifier


SOURCE_ROOT = discover_source_root(Path(__file__))
ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-b-verifier-", dir=output_root)


def _sentence(
    example_id: str = "ex-a",
    source_document_id: str = "doc-a",
    split: str = "test",
) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset_id": "CODE-ACCORD-v1.0.0",
        "example_id": example_id,
        "source_document_id": source_document_id,
        "source_country": "UK",
        "split": split,
        "content": "Door shall be fire rated.",
        "processed_content": "Door shall be fire rated .",
        "words": ["Door", "shall", "be", "fire", "rated", "."],
        "entities": [],
        "relations": [],
    }


def _candidate(
    example_id: str = "ex-a",
    source_document_id: str = "doc-a",
    split: str = "test",
) -> dict:
    triple = {
        "head": {"start": 0, "end": 0, "type": "Object", "text": "Door"},
        "relation": "selection",
        "tail": {
            "start": 3,
            "end": 4,
            "type": "Quality",
            "text": "fire rated",
        },
    }
    strict = StrictTriple.from_mapping(triple, example_id=example_id, label="fixture")
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": f"CODE-SPLIT-1:{split}",
        "training_seed": 42,
        "example_id": example_id,
        "source_document_id": source_document_id,
        "candidate_id": candidate_id_for(42, strict),
        **triple,
        "triple_confidence": 0.75,
        "input_hashes": {"checkpoint": ZERO_HASH},
    }


def _write_inputs(layout: RunLayout) -> tuple[Path, Path]:
    sentences = layout.resolve("data-prepared/test.jsonl")
    candidates = layout.resolve("predictions/test/candidates.jsonl")
    atomic_write_jsonl(sentences, [_sentence()])
    atomic_write_jsonl(candidates, [_candidate()])
    return sentences, candidates


def _response_record(request: dict, content: str, *, attempts: int = 1) -> dict:
    values = []
    for attempt in range(1, attempts + 1):
        response_content = content if attempt == attempts else "not-json"
        raw = {
            "message": {"role": "assistant", "content": response_content},
            "done_reason": "stop",
            "total_duration": 2_000_000_000,
            "prompt_eval_count": 100,
            "eval_count": 10,
        }
        values.append(
            {
                "attempt": attempt,
                "started_at_utc": f"2026-07-17T00:00:0{attempt}Z",
                "completed_at_utc": f"2026-07-17T00:00:0{attempt + 1}Z",
                "elapsed_seconds": float(attempt),
                "transport_status": "response",
                "http_status": 200,
                "raw_response": raw,
                "raw_body_sha256": hashlib.sha256(
                    json.dumps(raw, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "condition_id": request["condition_id"],
        "candidate_id": request["candidate_id"],
        "cache_key": request["cache_key"],
        "cache_hit": False,
        "attempts": values,
    }


class VerifierReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")

    def _planned_request(self, mode: str) -> dict:
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), f"plan-{mode}")
            layout.create()
            sentences, candidates = _write_inputs(layout)
            manifest = run_verifier(
                layout,
                self.config,
                mode=mode,
                execution_mode="dry-run",
                sentences_path=sentences,
                candidates_path=candidates,
            )
            self.assertEqual(manifest["status"], "planned")
            with layout.resolve(f"verifier/{mode}/requests.jsonl").open(
                encoding="utf-8"
            ) as handle:
                return json.loads(handle.readline())

    def test_dry_run_materializes_exact_schema_request_without_a_verdict(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "dry-simple")
            layout.create()
            sentences, candidates = _write_inputs(layout)
            manifest = run_verifier(
                layout,
                self.config,
                mode="simple",
                execution_mode="dry-run",
                sentences_path=sentences,
                candidates_path=candidates,
            )
            self.assertEqual(manifest["candidate_count"], 1)

            self.assertEqual(manifest["verdict_count"], 0)
            request = json.loads(
                layout.resolve("verifier/simple/requests.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(request["condition_id"], "VER-SIMPLE")
            self.assertEqual(request["payload"]["options"]["seed"], 42)
            self.assertFalse(request["payload"]["stream"])
            self.assertFalse(request["payload"]["think"])
            self.assertEqual(
                request["payload"]["format"]["required"], ["action", "reason_code"]
            )
            self.assertIn("sentence_tokens", request["payload"]["messages"][1]["content"])
            self.assertFalse(layout.resolve("verifier/simple/verdicts.jsonl").exists())
            schema_root = SOURCE_ROOT / "schemas" / "phase_b"
            request_schema = json.loads(
                (schema_root / "verifier-replay.schema.json").read_text(encoding="utf-8")
            )["$defs"]["request"]
            self.assertEqual(set(request), set(request_schema["required"]))
            environment = json.loads(
                layout.resolve("verifier/simple/environment-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            environment_schema = json.loads(
                (schema_root / "verifier-environment.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(environment), set(environment_schema["required"]))
            manifest_schema = json.loads(
                (schema_root / "verifier-manifest.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(manifest), set(manifest_schema["required"]))
            run_log_schema = json.loads(
                (schema_root / "verifier-run-log.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            for line in layout.resolve("verifier/simple/run-log.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                self.assertEqual(set(json.loads(line)), set(run_log_schema["required"]))
            for path in Path(temporary).rglob("*"):
                if path.is_file():
                    self.assertTrue(path.resolve().is_relative_to(layout.run_root.resolve()))

    def test_noncanonical_artifact_prefix_isolated_from_canonical_paths(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "prefixed-simple")
            layout.create()
            sentences, candidates = _write_inputs(layout)
            manifest = run_verifier(
                layout,
                self.config,
                mode="simple",
                execution_mode="dry-run",
                sentences_path=sentences,
                candidates_path=candidates,
                artifact_prefix="smoke",
            )
            self.assertEqual(manifest["status"], "planned")
            self.assertTrue(layout.resolve("verifier/smoke/simple/requests.jsonl").is_file())
            self.assertTrue(
                layout.resolve("manifests/verifier-smoke-simple-dry-run.json").is_file()
            )
            self.assertFalse(layout.resolve("verifier/simple/requests.jsonl").exists())

    def test_replay_retries_malformed_then_emits_schema_valid_simple_verdict(self):
        request = self._planned_request("simple")
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "replay-simple")
            layout.create()
            sentences, candidates = _write_inputs(layout)
            response_path = layout.resolve("inputs/simple-responses.jsonl")
            response = _response_record(
                request,
                '{"action":"KEEP","reason_code":"SUPPORTED"}',
                attempts=2,
            )
            atomic_write_jsonl(response_path, [response])
            manifest = run_verifier(
                layout,
                self.config,
                mode="simple",
                execution_mode="replay",
                sentences_path=sentences,
                candidates_path=candidates,
                response_ledger_path=response_path,
            )
            self.assertEqual(manifest["status"], "completed")
            verdict = json.loads(
                layout.resolve("verifier/simple/verdicts.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(verdict["action"], "KEEP")
            self.assertEqual(verdict["reason_code"], "SUPPORTED")
            self.assertEqual(verdict["attempts"], 2)
            self.assertTrue(verdict["telemetry"]["latency_observation_included"])

    def test_pilot_live_execution_rejects_any_cache_ledger(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "pilot-cache-rejected")
            layout.create()
            placeholder = layout.resolve("inputs/placeholder")
            with self.assertRaisesRegex(
                DataContractError, "cannot use a response cache ledger"
            ):
                run_verifier(
                    layout,
                    self.config,
                    mode="simple",
                    execution_mode="live",
                    sentences_path=placeholder,
                    candidates_path=placeholder,
                    warmup_sentences_path=placeholder,
                    warmup_candidates_path=placeholder,
                    cache_ledger_path=placeholder,
                    pilot_selection_path=placeholder,
                    model_blob_path=placeholder,
                )

    def test_replay_outputs_verdicts_in_seed_candidate_order(self):
        with _temporary_output_directory() as temporary:
            plan_layout = RunLayout(Path(temporary), "plan-order")
            plan_layout.create()
            plan_sentences = plan_layout.resolve("data-prepared/test.jsonl")
            plan_candidates = plan_layout.resolve("predictions/test/candidates.jsonl")
            sentence_rows = [
                _sentence("ex-a", "doc-a"),
                _sentence("ex-c", "doc-c"),
            ]
            candidate_rows = [
                _candidate("ex-a", "doc-a"),
                _candidate("ex-c", "doc-c"),
            ]
            atomic_write_jsonl(plan_sentences, sentence_rows)
            atomic_write_jsonl(plan_candidates, candidate_rows)
            run_verifier(
                plan_layout,
                self.config,
                mode="simple",
                execution_mode="dry-run",
                sentences_path=plan_sentences,
                candidates_path=plan_candidates,
            )
            requests = [
                json.loads(line)
                for line in plan_layout.resolve("verifier/simple/requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertGreater(requests[0]["candidate_id"], requests[1]["candidate_id"])

            replay_layout = RunLayout(Path(temporary), "replay-order")
            replay_layout.create()
            sentences = replay_layout.resolve("data-prepared/test.jsonl")
            candidates = replay_layout.resolve("predictions/test/candidates.jsonl")
            atomic_write_jsonl(sentences, sentence_rows)
            atomic_write_jsonl(candidates, candidate_rows)
            responses = [
                _response_record(
                    request,
                    '{"action":"KEEP","reason_code":"SUPPORTED"}',
                )
                for request in sorted(requests, key=lambda item: item["candidate_id"])
            ]
            response_path = replay_layout.resolve("inputs/simple-responses.jsonl")
            atomic_write_jsonl(response_path, responses)
            run_verifier(
                replay_layout,
                self.config,
                mode="simple",
                execution_mode="replay",
                sentences_path=sentences,
                candidates_path=candidates,
                response_ledger_path=response_path,
            )
            verdict_ids = [
                json.loads(line)["candidate_id"]
                for line in replay_layout.resolve("verifier/simple/verdicts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(verdict_ids, sorted(verdict_ids))

    def test_corrective_replay_source_maps_one_valid_relation_rewrite(self):
        request = self._planned_request("corrective")
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "replay-corrective")
            layout.create()
            sentences, candidates = _write_inputs(layout)
            response_path = layout.resolve("inputs/corrective-responses.jsonl")
            content = json.dumps(
                {
                    "action": "CORRECT",
                    "reason_code": "WRONG_RELATION",
                    "corrected": {
                        "head_text": "Door",
                        "head_type": "Object",
                        "relation": "necessity",
                        "tail_text": "fire rated",
                        "tail_type": "Quality",
                    },
                },
                separators=(",", ":"),
            )
            atomic_write_jsonl(response_path, [_response_record(request, content)])
            run_verifier(
                layout,
                self.config,
                mode="corrective",
                execution_mode="replay",
                sentences_path=sentences,
                candidates_path=candidates,
                response_ledger_path=response_path,
            )
            verdict = json.loads(
                layout.resolve("verifier/corrective/verdicts.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(verdict["action"], "CORRECT")
            self.assertEqual(verdict["correction_validation_status"], "valid")
            self.assertEqual(verdict["corrected"]["relation"], "necessity")
            self.assertEqual(verdict["corrected"]["head"]["start"], 0)
            self.assertEqual(verdict["corrected"]["tail"]["end"], 4)

    def test_live_execution_retains_and_excludes_one_development_warmup(self):
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "live-simple")
            layout.create()
            sentences = layout.resolve("data-prepared/development-pilot.jsonl")
            candidates = layout.resolve("predictions/dev/pilot-candidates.jsonl")
            atomic_write_jsonl(sentences, [_sentence(split="development")])
            atomic_write_jsonl(candidates, [_candidate(split="development")])
            pilot_selection = layout.resolve("predictions/dev/pilot-selection.json")
            atomic_write_json(pilot_selection, {"fixture": True})
            warmup_sentence = {**_sentence(), "example_id": "ex-warm", "split": "development"}
            warmup_triple = {
                "head": {"start": 0, "end": 0, "type": "Object", "text": "Door"},
                "relation": "selection",
                "tail": {
                    "start": 3,
                    "end": 4,
                    "type": "Quality",
                    "text": "fire rated",
                },
            }
            strict = StrictTriple.from_mapping(
                warmup_triple, example_id="ex-warm", label="warmup"
            )
            warmup_candidate = {
                "protocol_id": PROTOCOL_ID,
                "split_id": "CODE-SPLIT-1:development",
                "training_seed": 42,
                "example_id": "ex-warm",
                "source_document_id": "doc-a",
                "candidate_id": candidate_id_for(42, strict),
                **warmup_triple,
                "triple_confidence": 0.5,
                "input_hashes": {"checkpoint": ZERO_HASH},
            }
            warmup_sentences = layout.resolve("data-prepared/development.jsonl")
            warmup_candidates = layout.resolve(
                "predictions/dev/verifier-warmup-candidate.jsonl"
            )
            atomic_write_jsonl(warmup_sentences, [warmup_sentence])
            atomic_write_jsonl(warmup_candidates, [warmup_candidate])
            model_blob = layout.resolve("inputs/ollama/model-blob")
            model_blob.parent.mkdir(parents=True)
            model_blob.write_bytes(b"fixture")
            calls = []

            def transport(method, url, payload, timeout):
                calls.append((method, url, payload))
                raw = {
                    "message": {
                        "role": "assistant",
                        "content": '{"action":"KEEP","reason_code":"SUPPORTED"}',
                    },
                    "done_reason": "stop",
                    "total_duration": 1,
                }
                return HttpResult(
                    200,
                    raw,
                    hashlib.sha256(json.dumps(raw).encode("utf-8")).hexdigest(),
                )

            environment = {
                "protocol_id": PROTOCOL_ID,
                "condition_id": "VER-SIMPLE",
                "execution_mode": "live",
                "created_at_utc": "2026-07-17T00:00:00Z",
                "python": {},
                "platform": {},
                "hardware": {},
                "software": {},
                "ollama": {},
                "model": {},
            }
            model_evidence = {
                "identity_verified": True,
                "tag_digest": self.config.value["verifier"][
                    "registry_manifest_sha256"
                ],
                "blob_sha256": self.config.value["verifier"]["model_blob_sha256"],
                "details": {},
                "tags_response_sha256": ONE_HASH,
                "show_response_sha256": ONE_HASH,
            }

            def verify_model(*args, **kwargs):
                atomic_write_jsonl(
                    layout.resolve("verifier/simple/model/tags.json"),
                    [{"fixture": "tags"}],
                )
                atomic_write_jsonl(
                    layout.resolve("verifier/simple/model/show.json"),
                    [{"fixture": "show"}],
                )
                layout.resolve("verifier/simple/model/ollama-modelfile.txt").write_text(
                    "FROM fixture\n", encoding="utf-8"
                )
                return model_evidence

            with patch(
                "verifier._environment_manifest",
                return_value=environment,
            ), patch(
                "verifier._verify_live_model",
                side_effect=verify_model,
            ):
                manifest = run_verifier(
                    layout,
                    self.config,
                    mode="simple",
                    execution_mode="live",
                    sentences_path=sentences,
                    candidates_path=candidates,
                    warmup_sentences_path=warmup_sentences,
                    warmup_candidates_path=warmup_candidates,
                    model_blob_path=model_blob,
                    pilot_selection_path=pilot_selection,
                    transport=transport,
                    sleep=lambda _: None,
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(manifest["warmup_candidate_count"], 1)
            self.assertFalse(manifest["warmup_latency_observation_included"])
            self.assertEqual(
                manifest["inputs"]["predictions/dev/pilot-selection.json"],
                sha256_file(pilot_selection),
            )
            self.assertTrue(layout.resolve("verifier/simple/warmup-response.jsonl").is_file())
            verdict = json.loads(
                layout.resolve("verifier/simple/verdicts.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(verdict["action"], "KEEP")


class LiveModelIdentityTests(unittest.TestCase):
    def test_preflight_requires_tag_blob_details_and_modelfile_identity(self):
        config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")
        verifier = config.value["verifier"]
        with _temporary_output_directory() as temporary:
            layout = RunLayout(Path(temporary), "model-proof")
            layout.create()
            blob = layout.resolve("inputs/ollama/model-blob")
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"fixture")

            def transport(method, url, payload, timeout):
                if url.endswith("/api/tags"):
                    body = {
                        "models": [
                            {
                                "name": verifier["model"],
                                "digest": "sha256:" + verifier["registry_manifest_sha256"],
                            }
                        ]
                    }
                else:
                    body = {
                        "details": {
                            "family": "qwen3",
                            "parameter_size": "32.8B",
                            "quantization_level": "Q4_K_M",
                        },
                        "modelfile": "FROM sha256:" + verifier["model_blob_sha256"],
                    }
                return HttpResult(200, body, hashlib.sha256(b"body").hexdigest())

            real_sha256_file = __import__(
                "verifier", fromlist=["sha256_file"]
            ).sha256_file

            def sha256_file(path):
                if path == blob:
                    return verifier["model_blob_sha256"]
                return real_sha256_file(path)

            command = {
                "status": "available",
                "returncode": 0,
                "stdout": "FROM sha256:" + verifier["model_blob_sha256"],
            }
            with patch("verifier.sha256_file", side_effect=sha256_file), patch(
                "verifier._command_output", return_value=command
            ):
                evidence = _verify_live_model(
                    layout,
                    config,
                    "simple",
                    "http://localhost:11434",
                    blob,
                    transport,
                )
            self.assertTrue(evidence["identity_verified"])
            self.assertTrue(layout.resolve("verifier/simple/model/tags.json").is_file())
            self.assertTrue(layout.resolve("verifier/simple/model/show.json").is_file())


if __name__ == "__main__":
    unittest.main()
