"""Synthetic tests for the development-only B-07 verifier pilot audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from config import load_pipeline_config
from constants import PROTOCOL_ID
from phase_b_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    sha256_file,
)
from paths import RunLayout, discover_source_root
from pilot import PilotInputs, run_verifier_pilot
from records import StrictTriple, candidate_id_for
from verifier import (
    _load_candidates,
    _load_sentences,
    _verdict_from_response,
    run_verifier,
)


SOURCE_ROOT = discover_source_root(Path(__file__))
ZERO_HASH = "0" * 64


def _temporary_output_directory():
    output_root = SOURCE_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-phase-b-pilot-", dir=output_root)


def _triple(relation: str) -> dict:
    return {
        "head": {"start": 0, "end": 0, "type": "Object", "text": "Door"},
        "relation": relation,
        "tail": {
            "start": 3,
            "end": 4,
            "type": "Quality",
            "text": "fire rated",
        },
    }


def _candidate(
    relation: str, *, split: str = "development", training_seed: int = 42
) -> dict:
    triple = _triple(relation)
    strict = StrictTriple.from_mapping(triple, example_id="ex-a", label="fixture")
    return {
        "protocol_id": PROTOCOL_ID,
        "split_id": f"CODE-SPLIT-1:{split}",
        "training_seed": training_seed,
        "example_id": "ex-a",
        "source_document_id": "doc-a",
        "candidate_id": candidate_id_for(training_seed, strict),
        **triple,
        "triple_confidence": 0.75,
        "input_hashes": {"checkpoint": ZERO_HASH},
    }


def _response(request: dict, content: str, *, repeat: int) -> dict:
    raw = {
        "message": {"role": "assistant", "content": content},
        "done_reason": "stop",
        "total_duration": 2_000_000_000 + repeat,
        "prompt_eval_count": 100,
        "eval_count": 10,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "condition_id": request["condition_id"],
        "candidate_id": request["candidate_id"],
        "cache_key": request["cache_key"],
        "cache_hit": False,
        "attempts": [
            {
                "attempt": 1,
                "started_at_utc": f"2026-07-17T00:00:0{repeat}Z",
                "completed_at_utc": f"2026-07-17T00:00:1{repeat}Z",
                "elapsed_seconds": float(repeat),
                "transport_status": "response",
                "http_status": 200,
                "raw_response": raw,
                "raw_body_sha256": hashlib.sha256(
                    json.dumps(raw, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        ],
    }


class VerifierPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")

    def _fixture(self, root: str, *, mismatch: bool = False) -> tuple[RunLayout, PilotInputs]:
        layout = RunLayout(Path(root), "pilot")
        layout.create()
        sentences = layout.resolve("data-prepared/development.jsonl")
        gold = layout.resolve("data-prepared/development-gold.jsonl")
        candidates_path = layout.resolve("predictions/dev/candidates.jsonl")
        warmup_candidates = layout.resolve(
            "predictions/dev/verifier-warmup-candidate.jsonl"
        )
        split_manifest = layout.resolve("data-prepared/split-manifest.json")
        candidate_index = layout.resolve("predictions/dev/candidate-index.json")
        pilot_selection = layout.resolve("predictions/dev/pilot-selection.json")
        capture_index = layout.resolve("inputs/pilot/capture-index.json")
        threshold = layout.resolve("predictions/dev/threshold-selection.json")
        sentence = {
            "protocol_id": PROTOCOL_ID,
            "dataset_id": "CODE-ACCORD-v1.0.0",
            "example_id": "ex-a",
            "source_document_id": "doc-a",
            "source_country": "UK",
            "split": "development",
            "content": "Door shall be fire rated.",
            "processed_content": "Door shall be fire rated .",
            "words": ["Door", "shall", "be", "fire", "rated", "."],
            "entities": [],
            "relations": [],
        }
        candidate_records = sorted(
            [_candidate("necessity"), _candidate("selection")],
            key=lambda item: (
                item["training_seed"],
                item["example_id"],
                item["candidate_id"],
            ),
        )
        atomic_write_jsonl(sentences, [sentence])
        atomic_write_jsonl(
            gold,
            [
                {
                    "protocol_id": PROTOCOL_ID,
                    "split_id": "CODE-SPLIT-1:development",
                    "example_id": "ex-a",
                    "source_document_id": "doc-a",
                    "gold_triples": [_triple("necessity")],
                    "input_hashes": {"dataset": ZERO_HASH},
                }
            ],
        )
        atomic_write_jsonl(candidates_path, candidate_records)
        atomic_write_jsonl(warmup_candidates, [candidate_records[0]])
        development_ids = ["ex-a"] + [f"dev-{index:03d}" for index in range(102)]
        train_ids = [f"train-{index:03d}" for index in range(586)]
        test_ids = [f"test-{index:03d}" for index in range(173)]
        atomic_write_json(
            split_manifest,
            {
                "split_id": "CODE-SPLIT-1",
                "algorithm_revision": "iterative-multilabel-two-fold-1.0",
                "seed": 42,
                "train_count": 586,
                "development_count": 103,
                "test_count": 173,
                "train_ids": sorted(train_ids),
                "development_ids": sorted(development_ids),
                "test_ids": sorted(test_ids),
                "label_counts": {},
                "overlap_count": 0,
            },
        )
        indexed_files = []
        for seed in self.config.value["training_seeds"]:
            file_candidates = (
                candidate_records
                if seed == 42
                else sorted(
                    [
                        _candidate("necessity", training_seed=seed),
                        _candidate("selection", training_seed=seed),
                    ],
                    key=lambda item: (
                        item["training_seed"],
                        item["example_id"],
                        item["candidate_id"],
                    ),
                )
            )
            candidate_file = layout.resolve(f"predictions/dev/seed-{seed}.jsonl")
            atomic_write_jsonl(candidate_file, file_candidates)
            refs = [
                {
                    "candidate_id": item["candidate_id"],
                    "example_id": item["example_id"],
                }
                for item in file_candidates
            ]
            indexed_files.append(
                {
                    "training_seed": seed,
                    "path": f"predictions/dev/seed-{seed}.jsonl",
                    "sha256": sha256_file(candidate_file),
                    "candidate_count": len(refs),
                    "candidates": refs,
                }
            )
        atomic_write_json(
            candidate_index,
            {
                "schema_version": "phase-b-candidate-index-1.0",
                "protocol_id": PROTOCOL_ID,
                "workflow_id": "PATH-A-WORKFLOW-1.3",
                "split_id": "CODE-SPLIT-1:development",
                "split_manifest_sha256": sha256_file(split_manifest),
                "files": indexed_files,
                "candidate_count": sum(item["candidate_count"] for item in indexed_files),
            },
        )
        atomic_write_json(
            pilot_selection,
            {
                "schema_version": "phase-b-verifier-pilot-selection-1.0",
                "protocol_id": PROTOCOL_ID,
                "workflow_id": "PATH-A-WORKFLOW-1.3",
                "selection_split": "development",
                "split_manifest_sha256": sha256_file(split_manifest),
                "development_candidate_index_sha256": sha256_file(candidate_index),
                "pilot_candidates_sha256": sha256_file(candidates_path),
                "selected_before_live_calls": True,
                "used_test_labels": False,
                "selection_rule": "synthetic two-candidate contract fixture",
                "candidate_count": len(candidate_records),
                "training_seeds": [42],
                "candidate_ids": sorted(item["candidate_id"] for item in candidate_records),
                "example_ids": ["ex-a"],
            },
        )
        grid = self.config.value["threshold_selection"]["grid"]
        atomic_write_json(
            threshold,
            {
                "protocol_id": PROTOCOL_ID,
                "selection_split": "development",
                "objective": "mean_per_seed_development_strict_triple_f1",
                "tie_rule": "higher_threshold",
                "grid": grid,
                "selected_threshold": 0.5,
                "used_test_labels": False,
                "development_candidate_index_sha256": sha256_file(candidate_index),
                "development_gold_sha256": sha256_file(gold),
                "split_manifest_sha256": sha256_file(split_manifest),
                "per_threshold": [
                    {
                        "threshold": value,
                        "per_seed_development_strict_triple_f1": {
                            str(seed): 1.0 if value == 0.5 else 0.0
                            for seed in self.config.value["training_seeds"]
                        },
                        "mean_per_seed_development_strict_triple_f1": (
                            1.0 if value == 0.5 else 0.0
                        ),
                    }
                    for value in grid
                ],
            },
        )

        request_paths = {}
        for mode in ("simple", "corrective"):
            run_verifier(
                layout,
                self.config,
                mode=mode,
                execution_mode="dry-run",
                sentences_path=sentences,
                candidates_path=candidates_path,
            )
            request_paths[mode] = [
                json.loads(line)
                for line in layout.resolve(f"verifier/{mode}/requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        sentence_objects = _load_sentences(sentences)
        candidate_objects = {
            candidate.candidate_id: (candidate, sentence)
            for candidate, _, sentence in _load_candidates(
                candidates_path, sentence_objects
            )
        }
        warmup_candidate_id = candidate_records[0]["candidate_id"]

        capture_entries = []
        event_time = 0
        for mode in ("simple", "corrective"):
            for repeat in (1, 2):
                responses = []
                for request in sorted(
                    request_paths[mode],
                    key=lambda item: (item["training_seed"], item["candidate_id"]),
                ):
                    candidate = next(
                        item
                        for item in candidate_records
                        if item["candidate_id"] == request["candidate_id"]
                    )
                    if candidate["relation"] == "necessity":
                        content = '{"action":"KEEP","reason_code":"SUPPORTED"}'
                        if mode == "corrective":
                            content = (
                                '{"action":"KEEP","reason_code":"SUPPORTED",'
                                '"corrected":null}'
                            )
                    elif mode == "simple":
                        action = "KEEP" if mismatch and repeat == 2 else "DISCARD"
                        reason = "SUPPORTED" if action == "KEEP" else "WRONG_RELATION"
                        content = json.dumps(
                            {"action": action, "reason_code": reason}, separators=(",", ":")
                        )
                    else:
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
                    responses.append(_response(request, content, repeat=repeat))
                response_by_id = {
                    response["candidate_id"]: response for response in responses
                }
                verdicts = []
                for request in request_paths[mode]:
                    candidate, sentence_object = candidate_objects[
                        request["candidate_id"]
                    ]
                    verdicts.append(
                        _verdict_from_response(
                            mode,
                            response_by_id[request["candidate_id"]],
                            request,
                            candidate,
                            sentence_object,
                        )
                    )
                warmup_request = next(
                    request
                    for request in request_paths[mode]
                    if request["candidate_id"] == warmup_candidate_id
                )
                warmup_content = next(
                    attempt["raw_response"]["message"]["content"]
                    for response in responses
                    if response["candidate_id"] == warmup_candidate_id
                    for attempt in response["attempts"]
                )
                warmup_response = _response(
                    warmup_request, warmup_content, repeat=repeat + 2
                )
                capture_id = f"{mode}-repeat-{repeat}"
                source_run_id = f"synthetic-{capture_id}"
                capture_relative = f"inputs/pilot/captures/{capture_id}"
                capture_root = layout.resolve(capture_relative)
                verifier_root = capture_root / "verifier" / mode
                manifest_root = capture_root / "manifests"
                requests_path = verifier_root / "requests.jsonl"
                responses_path = verifier_root / "responses.jsonl"
                environment_path = verifier_root / "environment-manifest.json"
                run_log_path = verifier_root / "run-log.jsonl"
                atomic_write_jsonl(requests_path, request_paths[mode])
                atomic_write_jsonl(responses_path, responses)
                atomic_write_jsonl(verifier_root / "verdicts.jsonl", verdicts)
                atomic_write_jsonl(verifier_root / "warmup-request.jsonl", [warmup_request])
                atomic_write_jsonl(verifier_root / "warmup-response.jsonl", [warmup_response])
                atomic_write_json(verifier_root / "model" / "tags.json", {"fixture": True})
                atomic_write_json(verifier_root / "model" / "show.json", {"fixture": True})
                atomic_write_text(
                    verifier_root / "model" / "ollama-modelfile.txt",
                    self.config.value["verifier"]["model_blob_sha256"] + "\n",
                )
                atomic_write_json(
                    environment_path,
                    {
                        "protocol_id": PROTOCOL_ID,
                        "condition_id": f"VER-{mode.upper()}",
                        "execution_mode": "live",
                        "created_at_utc": "2026-07-17T00:00:00Z",
                        "python": {},
                        "platform": {},
                        "hardware": {},
                        "software": {},
                        "ollama": {},
                        "model": {
                            "identity_verified": True,
                            "registry_manifest_sha256": self.config.value["verifier"][
                                "registry_manifest_sha256"
                            ],
                            "model_blob_sha256": self.config.value["verifier"][
                                "model_blob_sha256"
                            ],
                            "tag_digest": self.config.value["verifier"][
                                "registry_manifest_sha256"
                            ],
                            "blob_sha256": self.config.value["verifier"][
                                "model_blob_sha256"
                            ],
                            "details": {
                                "family": "qwen3",
                                "parameter_size": "32.8B",
                                "quantization_level": "Q4_K_M",
                            },
                        },
                    },
                )
                events = []

                def add_event(
                    event,
                    status,
                    candidate_id=None,
                    attempt=None,
                    details=None,
                ):
                    nonlocal event_time
                    event_time += 1
                    events.append(
                        {
                            "protocol_id": PROTOCOL_ID,
                            "condition_id": f"VER-{mode.upper()}",
                            "execution_mode": "live",
                            "event": event,
                            "sequence": len(events) + 1,
                            "timestamp_utc": (
                                f"2026-07-17T00:{event_time // 60:02d}:"
                                f"{event_time % 60:02d}Z"
                            ),
                            "candidate_id": candidate_id,
                            "attempt": attempt,
                            "status": status,
                            "details": details or {},
                        }
                    )

                add_event("run_started", "started")
                add_event("model_identity_verified", "verified")
                add_event(
                    "request_planned",
                    "warmup_planned",
                    warmup_request["candidate_id"],
                    details={
                        "request_sha256": warmup_request["request_sha256"],
                        "latency_observation_included": False,
                    },
                )
                for attempt in warmup_response["attempts"]:
                    add_event(
                        "request_attempted",
                        attempt["transport_status"],
                        warmup_request["candidate_id"],
                        attempt["attempt"],
                        {
                            "http_status": attempt["http_status"],
                            "elapsed_seconds": attempt["elapsed_seconds"],
                        },
                    )
                for request, response in zip(request_paths[mode], responses):
                    add_event(
                        "request_planned",
                        "planned",
                        request["candidate_id"],
                        details={"request_sha256": request["request_sha256"]},
                    )
                    for attempt in response["attempts"]:
                        add_event(
                            "request_attempted",
                            attempt["transport_status"],
                            request["candidate_id"],
                            attempt["attempt"],
                            {
                                "http_status": attempt["http_status"],
                                "elapsed_seconds": attempt["elapsed_seconds"],
                            },
                        )
                    add_event(
                        "candidate_completed", "valid_response", request["candidate_id"]
                    )
                add_event("run_completed", "completed")
                atomic_write_jsonl(run_log_path, events)
                output_paths = [
                    requests_path,
                    responses_path,
                    environment_path,
                    run_log_path,
                    verifier_root / "verdicts.jsonl",
                    verifier_root / "warmup-request.jsonl",
                    verifier_root / "warmup-response.jsonl",
                    verifier_root / "model" / "tags.json",
                    verifier_root / "model" / "show.json",
                    verifier_root / "model" / "ollama-modelfile.txt",
                ]
                atomic_write_json(
                    manifest_root / "00-checkout-manifest.json",
                    {
                        "schema_version": "phase-b-checkout-manifest-1.0",
                        "protocol_id": PROTOCOL_ID,
                        "workflow_id": "PATH-A-WORKFLOW-1.3",
                        "matcher_id": "CODE-STRICT-1",
                        "run_id": source_run_id,
                        "status": "pass",
                        "publication_execution_admitted": False,
                        "publication_execution_gate": "B-07-go-no-go-not-yet-approved",
                        "source": {
                            "commit": "a" * 40,
                            "worktree_clean": True,
                            "lockfile_sha256": ZERO_HASH,
                        },
                        "environment": {},
                        "tracked_artifact_sha256": {"fixture": ZERO_HASH},
                        "checks": [],
                    },
                )
                atomic_write_json(
                    manifest_root / f"verifier-{mode}-live.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "condition_id": f"VER-{mode.upper()}",
                        "execution_mode": "live",
                        "status": "completed",
                        "candidate_count": len(candidate_records),
                        "warmup_candidate_count": 1,
                        "warmup_latency_observation_included": False,
                        "verdict_count": len(candidate_records),
                        "prompt_sha256": request_paths[mode][0]["prompt_sha256"],
                        "model_manifest_sha256": request_paths[mode][0][
                            "model_manifest_sha256"
                        ],
                        "decoding_sha256": request_paths[mode][0]["decoding_sha256"],
                        "inputs": {
                            "data-prepared/development.jsonl": sha256_file(sentences),
                            "predictions/dev/pilot-candidates.jsonl": sha256_file(
                                candidates_path
                            ),
                            "predictions/dev/verifier-warmup-candidate.jsonl": sha256_file(
                                warmup_candidates
                            ),
                            "predictions/dev/pilot-selection.json": sha256_file(
                                pilot_selection
                            ),
                            "inputs/model-blob": self.config.value["verifier"][
                                "model_blob_sha256"
                            ],
                        },
                        "outputs": {
                            path.relative_to(capture_root).as_posix(): sha256_file(path)
                            for path in output_paths
                        },
                    },
                )
                capture_entries.append(
                    {
                        "capture_id": capture_id,
                        "mode": mode,
                        "repeat": repeat,
                        "source_run_id": source_run_id,
                        "capture_root": capture_relative,
                    }
                )

        atomic_write_json(
            capture_index,
            {
                "schema_version": "phase-b-verifier-pilot-captures-1.0",
                "protocol_id": PROTOCOL_ID,
                "workflow_id": "PATH-A-WORKFLOW-1.3",
                "captures": capture_entries,
            },
        )

        return layout, PilotInputs(
            sentences=sentences,
            gold=gold,
            candidates=candidates_path,
            warmup_candidates=warmup_candidates,
            split_manifest=split_manifest,
            candidate_index=candidate_index,
            pilot_selection=pilot_selection,
            threshold_selection=threshold,
            capture_index=capture_index,
        )

    def test_synthetic_pilot_proves_contract_but_cannot_admit_execution(self):
        with _temporary_output_directory() as temporary:
            layout, inputs = self._fixture(temporary)
            audit = run_verifier_pilot(
                layout, self.config, inputs, evidence_class="synthetic-fixture"
            )
            self.assertEqual(audit["pilot_status"], "pass")
            self.assertEqual(audit["go_no_go_status"], "blocked_synthetic_fixture")
            self.assertFalse(audit["publication_execution_admitted"])
            self.assertEqual(
                audit["candidate_class_balance"],
                {"gold_valid": 1, "gold_invalid": 1},
            )
            diagnostics = {
                (item["condition_id"], item["repeat"]): item
                for item in audit["decision_diagnostics"]
            }
            self.assertEqual(
                diagnostics[("VER-SIMPLE", 1)]["disagreement_count"], 0
            )
            self.assertEqual(
                diagnostics[("VER-CORRECTIVE", 1)]["disagreement_count"], 0
            )
            correction = audit["corrective_outcome_diagnostics"][0]
            self.assertEqual(correction["correction_attempted"], 1)
            self.assertEqual(correction["correction_strict_gold_valid"], 1)
            self.assertEqual(correction["original_invalid_to_final_valid"], 1)
            self.assertEqual(len(audit["capture_provenance"]), 4)
            self.assertTrue(audit["determinism"]["simple"]["normalized_identical"])
            self.assertTrue(audit["determinism"]["corrective"]["normalized_identical"])
            schema = json.loads(
                (SOURCE_ROOT / "schemas/phase_b/verifier-pilot-audit.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(audit), set(schema["required"]))
            self.assertIn("allOf", schema)
            for relative, digest in audit["output_sha256"].items():
                self.assertEqual(sha256_file(layout.resolve(relative)), digest)

    def test_nondeterministic_repeat_is_retained_as_no_go(self):
        with _temporary_output_directory() as temporary:
            layout, inputs = self._fixture(temporary, mismatch=True)
            audit = run_verifier_pilot(
                layout, self.config, inputs, evidence_class="synthetic-fixture"
            )
            self.assertEqual(audit["pilot_status"], "fail")
            self.assertEqual(audit["go_no_go_status"], "blocked_synthetic_fixture")
            self.assertEqual(audit["determinism"]["simple"]["mismatch_count"], 1)
            self.assertTrue(audit["material_protocol_review_required"])

    def test_indexed_candidate_file_hash_tampering_is_rejected(self):
        with _temporary_output_directory() as temporary:
            layout, inputs = self._fixture(temporary)
            candidate_file = layout.resolve("predictions/dev/seed-42.jsonl")
            atomic_write_text(
                candidate_file,
                candidate_file.read_text(encoding="utf-8") + "\n",
            )
            with self.assertRaisesRegex(DataContractError, "candidate-file hash"):
                run_verifier_pilot(
                    layout, self.config, inputs, evidence_class="synthetic-fixture"
                )

    def test_captured_verdicts_must_equal_response_replay(self):
        with _temporary_output_directory() as temporary:
            layout, inputs = self._fixture(temporary)
            root = layout.resolve("inputs/pilot/captures/simple-repeat-1")
            verdicts = root / "verifier/simple/verdicts.jsonl"
            atomic_write_jsonl(verdicts, [{"tampered": True}])
            manifest_path = root / "manifests/verifier-simple-live.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["verifier/simple/verdicts.jsonl"] = sha256_file(
                verdicts
            )
            atomic_write_json(manifest_path, manifest)
            with self.assertRaisesRegex(DataContractError, "differ from response-ledger"):
                run_verifier_pilot(
                    layout, self.config, inputs, evidence_class="synthetic-fixture"
                )

    def test_malformed_runtime_manifest_retains_contract_failure(self):
        with _temporary_output_directory() as temporary:
            layout, inputs = self._fixture(temporary)
            root = layout.resolve("inputs/pilot/captures/simple-repeat-1")
            environment_path = root / "verifier/simple/environment-manifest.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment.pop("software")
            atomic_write_json(environment_path, environment)
            manifest_path = root / "manifests/verifier-simple-live.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["verifier/simple/environment-manifest.json"] = (
                sha256_file(environment_path)
            )
            atomic_write_json(manifest_path, manifest)
            with self.assertRaisesRegex(DataContractError, "runtime fields must be objects"):
                run_verifier_pilot(
                    layout, self.config, inputs, evidence_class="synthetic-fixture"
                )
            failure = json.loads(
                layout.resolve("audit/verifier-pilot/pilot-failure.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["error_category"], "DataContractError")

    def test_development_evidence_rejects_incomplete_official_partition(self):
        with _temporary_output_directory() as temporary:
            layout, inputs = self._fixture(temporary)
            with self.assertRaisesRegex(DataContractError, "103-sentence development"):
                run_verifier_pilot(
                    layout, self.config, inputs, evidence_class="development-pilot"
                )
            failure = json.loads(
                layout.resolve("audit/verifier-pilot/pilot-failure.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["go_no_go_status"], "no_go_contract_failure")
            self.assertFalse(failure["publication_execution_admitted"])

    def test_development_evidence_rejects_incomplete_seed_coverage(self):
        with _temporary_output_directory() as temporary:
            layout, inputs = self._fixture(temporary)
            split = json.loads(inputs.split_manifest.read_text(encoding="utf-8"))
            sentence_template = json.loads(
                inputs.sentences.read_text(encoding="utf-8")
            )
            sentence_records = []
            gold_records = []
            for example_id in split["development_ids"]:
                source_document_id = "doc-a" if example_id == "ex-a" else f"doc-{example_id}"
                sentence_records.append(
                    {
                        **sentence_template,
                        "example_id": example_id,
                        "source_document_id": source_document_id,
                    }
                )
                gold_records.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "split_id": "CODE-SPLIT-1:development",
                        "example_id": example_id,
                        "source_document_id": source_document_id,
                        "gold_triples": [_triple("necessity")]
                        if example_id == "ex-a"
                        else [],
                        "input_hashes": {"dataset": ZERO_HASH},
                    }
                )
            atomic_write_jsonl(inputs.sentences, sentence_records)
            atomic_write_jsonl(inputs.gold, gold_records)
            threshold = json.loads(
                inputs.threshold_selection.read_text(encoding="utf-8")
            )
            threshold["development_gold_sha256"] = sha256_file(inputs.gold)
            atomic_write_json(inputs.threshold_selection, threshold)
            with self.assertRaisesRegex(DataContractError, "seeds 42-49"):
                run_verifier_pilot(
                    layout, self.config, inputs, evidence_class="development-pilot"
                )


if __name__ == "__main__":
    unittest.main()
