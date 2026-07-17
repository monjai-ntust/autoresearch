"""Development-only B-07 verifier pilot audit.

The pilot consumes two independently captured frozen response ledgers for each
verifier mode.  It never calls a model and never admits publication execution;
its job is to prove development-only pairing, schema/grounding behavior, and
byte-stable normalized decisions before the user-facing go/no-go checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .config import PipelineConfig
from .constants import PROTOCOL_ID, WORKFLOW_ID
from .io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    iter_jsonl,
    load_json,
    sha256_file,
)
from .paths import PathContractError, RunLayout
from .records import Candidate, StrictTriple, Verdict
from .scoring import _load_gold, _load_threshold
from .split import SPLIT_ALGORITHM_REVISION
from .verifier import (
    MODE_CONDITIONS,
    _load_candidates,
    _load_response_ledger,
    _load_sentences,
    _prompt_bundle,
    _request_record,
    _verdict_from_response,
)


PILOT_EVIDENCE_CLASSES = {"development-pilot", "synthetic-fixture"}
_NORMALIZED_FIELDS = (
    "protocol_id",
    "condition_id",
    "training_seed",
    "candidate_id",
    "response_status",
    "action",
    "reason_code",
    "corrected",
    "correction_validation_status",
)


@dataclass(frozen=True)
class PilotInputs:
    sentences: Path
    gold: Path
    candidates: Path
    warmup_candidates: Path
    split_manifest: Path
    candidate_index: Path
    pilot_selection: Path
    threshold_selection: Path
    capture_index: Path


@dataclass(frozen=True)
class _Capture:
    capture_id: str
    mode: str
    repeat: int
    source_run_id: str
    root: Path
    checkout_manifest: Path
    verifier_manifest: Path
    environment_manifest: Path
    requests: Path
    responses: Path
    run_log: Path


@dataclass(frozen=True)
class _Repeat:
    responses: dict[str, dict[str, Any]]
    verdicts: list[dict[str, Any]]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be an object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise DataContractError(
            f"{label} fields differ from the contract; missing={missing}, extra={extra}"
        )
    return value


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DataContractError(f"{label} must be a UTC date-time ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise DataContractError(f"{label} is not an ISO date-time") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DataContractError(f"{label} is not UTC")
    return parsed


def _check(check_id: str, passed: bool, detail: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _load_split_manifest(path: Path) -> tuple[dict[str, Any], set[str]]:
    value = _exact_keys(
        load_json(path),
        {
            "split_id",
            "algorithm_revision",
            "seed",
            "train_count",
            "development_count",
            "test_count",
            "train_ids",
            "development_ids",
            "test_ids",
            "label_counts",
            "overlap_count",
        },
        path.name,
    )
    expected_scalars = {
        "split_id": "CODE-SPLIT-1",
        "algorithm_revision": SPLIT_ALGORITHM_REVISION,
        "seed": 42,
        "train_count": 586,
        "development_count": 103,
        "test_count": 173,
        "overlap_count": 0,
    }
    for field, expected in expected_scalars.items():
        if value[field] != expected:
            raise DataContractError(
                f"{path.name}.{field} must be {expected!r}, got {value[field]!r}"
            )
    partitions: dict[str, set[str]] = {}
    for field, expected_count in (
        ("train_ids", 586),
        ("development_ids", 103),
        ("test_ids", 173),
    ):
        identifiers = value[field]
        if (
            not isinstance(identifiers, list)
            or len(identifiers) != expected_count
            or any(not isinstance(item, str) or not item for item in identifiers)
            or identifiers != sorted(identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise DataContractError(
                f"{path.name}.{field} must contain {expected_count} unique sorted IDs"
            )
        partitions[field] = set(identifiers)
    if (
        partitions["train_ids"] & partitions["development_ids"]
        or partitions["train_ids"] & partitions["test_ids"]
        or partitions["development_ids"] & partitions["test_ids"]
    ):
        raise DataContractError(f"{path.name} contains cross-partition overlap")
    if not isinstance(value["label_counts"], dict):
        raise DataContractError(f"{path.name}.label_counts must be an object")
    return value, partitions["development_ids"]


def _load_candidate_index(
    path: Path,
    config: PipelineConfig,
    *,
    layout: RunLayout,
    sentences: dict[str, Any],
    split_manifest_sha256: str,
    development_ids: set[str],
) -> dict[str, tuple[int, str]]:
    value = _exact_keys(
        load_json(path),
        {
            "schema_version",
            "protocol_id",
            "workflow_id",
            "split_id",
            "split_manifest_sha256",
            "files",
            "candidate_count",
        },
        path.name,
    )
    expected = {
        "schema_version": "phase-b-candidate-index-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "split_id": "CODE-SPLIT-1:development",
        "split_manifest_sha256": split_manifest_sha256,
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise DataContractError(
                f"{path.name}.{field} must be {expected_value!r}"
            )
    files = value["files"]
    seeds = config.value["training_seeds"]
    if not isinstance(files, list) or len(files) != len(seeds):
        raise DataContractError(
            f"{path.name}.files must contain one development file per training seed"
        )
    indexed: dict[str, tuple[int, str]] = {}
    total = 0
    observed_seeds: list[int] = []
    for index, (entry, expected_seed) in enumerate(zip(files, seeds)):
        label = f"{path.name}.files[{index}]"
        entry = _exact_keys(
            entry,
            {"training_seed", "path", "sha256", "candidate_count", "candidates"},
            label,
        )
        if entry["training_seed"] != expected_seed:
            raise DataContractError(f"{label}.training_seed must be {expected_seed}")
        observed_seeds.append(expected_seed)
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise DataContractError(f"{label}.path must be a safe run-relative path")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise DataContractError(f"{label}.sha256 must be lowercase SHA-256")
        candidate_file = layout.resolve(relative, must_exist=True)
        if not candidate_file.is_file() or sha256_file(candidate_file) != digest:
            raise DataContractError(
                f"{label}.path does not resolve to the indexed candidate-file hash"
            )
        file_rows = _load_candidates(candidate_file, sentences)
        if any(candidate.training_seed != expected_seed for candidate, _, _ in file_rows):
            raise DataContractError(
                f"{label}.path contains a candidate from another training seed"
            )
        candidates = entry["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise DataContractError(f"{label}.candidates must be a nonempty array")
        observed_references: list[dict[str, str]] = []
        for candidate_index, candidate_ref in enumerate(candidates):
            ref_label = f"{label}.candidates[{candidate_index}]"
            candidate_ref = _exact_keys(
                candidate_ref, {"candidate_id", "example_id"}, ref_label
            )
            candidate_id = candidate_ref["candidate_id"]
            example_id = candidate_ref["example_id"]
            if (
                not isinstance(candidate_id, str)
                or not re.fullmatch(r"cand-[0-9a-f]{64}", candidate_id)
                or not isinstance(example_id, str)
                or example_id not in development_ids
            ):
                raise DataContractError(
                    f"{ref_label} must identify one indexed development candidate"
                )
            if candidate_id in indexed:
                raise DataContractError(f"{ref_label} duplicates candidate_id")
            indexed[candidate_id] = (expected_seed, example_id)
            observed_references.append(candidate_ref)
        actual_references = [
            {
                "candidate_id": candidate.candidate_id,
                "example_id": candidate.triple.example_id,
            }
            for candidate, _, _ in file_rows
        ]
        if observed_references != actual_references:
            raise DataContractError(
                f"{label}.candidates differ from the parsed indexed candidate file"
            )
        if entry["candidate_count"] != len(candidates):
            raise DataContractError(f"{label}.candidate_count is inconsistent")
        total += len(candidates)
    if observed_seeds != seeds or value["candidate_count"] != total:
        raise DataContractError(f"{path.name} candidate totals are inconsistent")
    return indexed


def _load_pilot_selection(
    path: Path,
    *,
    split_manifest_sha256: str,
    candidate_index_sha256: str,
    candidates_sha256: str,
) -> dict[str, Any]:
    value = _exact_keys(
        load_json(path),
        {
            "schema_version",
            "protocol_id",
            "workflow_id",
            "selection_split",
            "split_manifest_sha256",
            "development_candidate_index_sha256",
            "pilot_candidates_sha256",
            "selected_before_live_calls",
            "used_test_labels",
            "selection_rule",
            "candidate_count",
            "training_seeds",
            "candidate_ids",
            "example_ids",
        },
        path.name,
    )
    expected = {
        "schema_version": "phase-b-verifier-pilot-selection-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "selection_split": "development",
        "split_manifest_sha256": split_manifest_sha256,
        "development_candidate_index_sha256": candidate_index_sha256,
        "pilot_candidates_sha256": candidates_sha256,
        "selected_before_live_calls": True,
        "used_test_labels": False,
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise DataContractError(
                f"{path.name}.{field} must be {expected_value!r}"
            )
    if not isinstance(value["selection_rule"], str) or not value["selection_rule"].strip():
        raise DataContractError(f"{path.name}.selection_rule must be nonempty")
    for field in ("candidate_ids", "example_ids"):
        identifiers = value[field]
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or any(not isinstance(item, str) or not item for item in identifiers)
            or identifiers != sorted(identifiers)
            or len(identifiers) != len(set(identifiers))
        ):
            raise DataContractError(f"{path.name}.{field} must be unique and sorted")
    if value["candidate_count"] != len(value["candidate_ids"]):
        raise DataContractError(f"{path.name}.candidate_count is inconsistent")
    seeds = value["training_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or seeds != sorted(set(seeds))
    ):
        raise DataContractError(f"{path.name}.training_seeds must be unique and sorted")
    return value


def _capture_member(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DataContractError(f"capture artifact path is unsafe: {relative!r}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise DataContractError(f"capture artifact escapes its root: {relative!r}")
    return resolved


def _load_capture_index(layout: RunLayout, path: Path) -> list[_Capture]:
    value = _exact_keys(
        load_json(path),
        {"schema_version", "protocol_id", "workflow_id", "captures"},
        path.name,
    )
    expected_header = {
        "schema_version": "phase-b-verifier-pilot-captures-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
    }
    for field, expected in expected_header.items():
        if value[field] != expected:
            raise DataContractError(f"{path.name}.{field} must be {expected!r}")
    raw_captures = value["captures"]
    expected_order = [
        ("simple-repeat-1", "simple", 1),
        ("simple-repeat-2", "simple", 2),
        ("corrective-repeat-1", "corrective", 1),
        ("corrective-repeat-2", "corrective", 2),
    ]
    if not isinstance(raw_captures, list) or len(raw_captures) != 4:
        raise DataContractError(f"{path.name}.captures must contain exactly four entries")
    captures: list[_Capture] = []
    source_run_ids: set[str] = set()
    for index, (raw, expected_identity) in enumerate(zip(raw_captures, expected_order)):
        label = f"{path.name}.captures[{index}]"
        raw = _exact_keys(
            raw,
            {"capture_id", "mode", "repeat", "source_run_id", "capture_root"},
            label,
        )
        observed_identity = (raw["capture_id"], raw["mode"], raw["repeat"])
        if observed_identity != expected_identity:
            raise DataContractError(
                f"{label} must identify {expected_identity}, got {observed_identity}"
            )
        source_run_id = raw["source_run_id"]
        if not isinstance(source_run_id, str) or not source_run_id:
            raise DataContractError(f"{label}.source_run_id must be nonempty")
        if source_run_id in source_run_ids:
            raise DataContractError(f"{label}.source_run_id must be unique")
        source_run_ids.add(source_run_id)
        capture_root = raw["capture_root"]
        if not isinstance(capture_root, str) or not capture_root:
            raise DataContractError(f"{label}.capture_root must be nonempty")
        root = layout.resolve(capture_root, must_exist=True)
        if not root.is_dir():
            raise DataContractError(f"{label}.capture_root must be a directory")
        mode = raw["mode"]
        captures.append(
            _Capture(
                capture_id=raw["capture_id"],
                mode=mode,
                repeat=raw["repeat"],
                source_run_id=source_run_id,
                root=root,
                checkout_manifest=_capture_member(
                    root, "manifests/00-checkout-manifest.json"
                ),
                verifier_manifest=_capture_member(
                    root, f"manifests/verifier-{mode}-live.json"
                ),
                environment_manifest=_capture_member(
                    root, f"verifier/{mode}/environment-manifest.json"
                ),
                requests=_capture_member(root, f"verifier/{mode}/requests.jsonl"),
                responses=_capture_member(root, f"verifier/{mode}/responses.jsonl"),
                run_log=_capture_member(root, f"verifier/{mode}/run-log.jsonl"),
            )
        )
    return captures


def _normalized(verdict: dict[str, Any]) -> dict[str, Any]:
    return {field: verdict[field] for field in _NORMALIZED_FIELDS}


def _distribution(verdicts: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    response_status = Counter(item["response_status"] for item in verdicts)
    action = Counter(item["action"] or "null" for item in verdicts)
    reason = Counter(item["reason_code"] or "null" for item in verdicts)
    correction = Counter(item["correction_validation_status"] for item in verdicts)
    return {
        "response_status": dict(sorted(response_status.items())),
        "action": dict(sorted(action.items())),
        "reason_code": dict(sorted(reason.items())),
        "correction_validation_status": dict(sorted(correction.items())),
    }


def _decision_diagnostic(
    *,
    condition_id: str,
    repeat: int,
    predicted_valid_by_id: dict[str, bool],
    candidates_by_id: dict[str, tuple[Candidate, Any]],
    gold: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter()
    for candidate_id, (candidate, _) in candidates_by_id.items():
        gold_valid = candidate.triple in gold[candidate.triple.example_id].triples
        predicted_valid = predicted_valid_by_id[candidate_id]
        if gold_valid and predicted_valid:
            counts["tp"] += 1
        elif not gold_valid and predicted_valid:
            counts["fp"] += 1
        elif gold_valid:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    candidate_count = tp + fp + fn + tn
    disagreement_count = fp + fn
    return {
        "condition_id": condition_id,
        "repeat": repeat,
        "candidate_count": candidate_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "agreement_count": tp + tn,
        "disagreement_count": disagreement_count,
        "disagreement_rate": disagreement_count / candidate_count,
    }


def _corrective_outcome_diagnostic(
    *,
    repeat: int,
    verdicts: list[dict[str, Any]],
    candidates_by_id: dict[str, tuple[Candidate, Any]],
    gold: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter()
    original_candidate_keys = {
        (candidate.training_seed, candidate.triple)
        for candidate, _ in candidates_by_id.values()
    }
    for verdict in verdicts:
        candidate, _ = candidates_by_id[verdict["candidate_id"]]
        gold_triples = gold[candidate.triple.example_id].triples
        original_valid = candidate.triple in gold_triples
        emitted = None
        if verdict["response_status"] == "valid_response":
            if verdict["action"] == "KEEP":
                emitted = candidate.triple
            elif (
                verdict["action"] == "CORRECT"
                and verdict["correction_validation_status"] == "valid"
                and verdict["corrected"] is not None
            ):
                emitted = StrictTriple.from_mapping(
                    verdict["corrected"],
                    example_id=candidate.triple.example_id,
                    label=f"pilot corrected {candidate.candidate_id}",
                )
        final_valid = emitted in gold_triples if emitted is not None else False
        transition = (
            ("valid" if original_valid else "invalid")
            + "_to_"
            + ("valid" if final_valid else "invalid")
        )
        counts[transition] += 1
        if verdict["action"] == "CORRECT":
            counts["correction_attempted"] += 1
            if emitted is not None:
                counts["correction_source_valid"] += 1
                if emitted in gold_triples:
                    counts["correction_strict_gold_valid"] += 1
                if (candidate.training_seed, emitted) in original_candidate_keys:
                    counts["correction_duplicate_emission"] += 1
    return {
        "repeat": repeat,
        "candidate_count": len(verdicts),
        "correction_attempted": counts["correction_attempted"],
        "correction_source_valid": counts["correction_source_valid"],
        "correction_strict_gold_valid": counts["correction_strict_gold_valid"],
        "correction_duplicate_emission": counts["correction_duplicate_emission"],
        "original_valid_to_final_valid": counts["valid_to_valid"],
        "original_valid_to_final_invalid": counts["valid_to_invalid"],
        "original_invalid_to_final_valid": counts["invalid_to_valid"],
        "original_invalid_to_final_invalid": counts["invalid_to_invalid"],
    }


def _telemetry(
    responses: dict[str, dict[str, Any]], verdicts: list[dict[str, Any]]
) -> dict[str, Any]:
    attempts = [
        attempt
        for response in responses.values()
        for attempt in response["attempts"]
    ]
    prompt_tokens = 0
    generated_tokens = 0
    total_duration_ns = 0
    for attempt in attempts:
        raw = attempt.get("raw_response")
        if not isinstance(raw, dict):
            continue
        for field, target in (
            ("prompt_eval_count", "prompt"),
            ("eval_count", "generated"),
            ("total_duration", "duration"),
        ):
            value = raw.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                if target == "prompt":
                    prompt_tokens += value
                elif target == "generated":
                    generated_tokens += value
                else:
                    total_duration_ns += value
    return {
        "candidate_calls": len(responses),
        "attempt_count": len(attempts),
        "retry_candidate_count": sum(
            len(response["attempts"]) > 1 for response in responses.values()
        ),
        "cache_hit_count": sum(response["cache_hit"] for response in responses.values()),
        "failed_verdict_count": sum(
            item["response_status"] != "valid_response" for item in verdicts
        ),
        "transport_status": dict(
            sorted(Counter(item["transport_status"] for item in attempts).items())
        ),
        "attempt_elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in attempts),
        "prompt_eval_count": prompt_tokens,
        "eval_count": generated_tokens,
        "ollama_total_duration_ns": total_duration_ns,
    }


def _validate_retry_timing(
    responses: dict[str, dict[str, Any]], config: PipelineConfig, label: str
) -> None:
    backoffs = config.value["verifier"]["backoff_seconds"]
    for candidate_id, response in responses.items():
        attempts = response["attempts"]
        for index, attempt in enumerate(attempts):
            started = _utc_timestamp(
                attempt["started_at_utc"],
                f"{label}.{candidate_id}.attempts[{index}].started_at_utc",
            )
            completed = _utc_timestamp(
                attempt["completed_at_utc"],
                f"{label}.{candidate_id}.attempts[{index}].completed_at_utc",
            )
            if completed < started:
                raise DataContractError(
                    f"{label}.{candidate_id}.attempts[{index}] completes before it starts"
                )
            if index:
                previous_completed = _utc_timestamp(
                    attempts[index - 1]["completed_at_utc"],
                    f"{label}.{candidate_id}.attempts[{index - 1}].completed_at_utc",
                )
                minimum_gap = float(backoffs[index - 1])
                observed_gap = (started - previous_completed).total_seconds()
                if observed_gap + 0.05 < minimum_gap:
                    raise DataContractError(
                        f"{label}.{candidate_id}.attempts[{index}] violates the "
                        f"{minimum_gap:g}-second retry backoff"
                    )


def _load_run_log(path: Path, condition: str) -> list[dict[str, Any]]:
    required = {
        "protocol_id",
        "condition_id",
        "execution_mode",
        "event",
        "sequence",
        "timestamp_utc",
        "candidate_id",
        "attempt",
        "status",
        "details",
    }
    events: list[dict[str, Any]] = []
    for line_number, raw in iter_jsonl(path):
        label = f"{path.name}:{line_number}"
        value = _exact_keys(raw, required, label)
        if (
            value["protocol_id"] != PROTOCOL_ID
            or value["condition_id"] != condition
            or value["execution_mode"] != "live"
        ):
            raise DataContractError(f"{label} is not a live canonical event")
        if value["sequence"] != line_number:
            raise DataContractError(f"{label}.sequence is not consecutive from one")
        _utc_timestamp(value["timestamp_utc"], f"{label}.timestamp_utc")
        if not isinstance(value["details"], dict):
            raise DataContractError(f"{label}.details must be an object")
        events.append(value)
    if (
        not events
        or events[0]["event"] != "run_started"
        or events[-1]["event"] != "run_completed"
        or any(item["event"] == "run_failed" for item in events)
        or not any(item["event"] == "model_identity_verified" for item in events)
    ):
        raise DataContractError(f"{path.name} is not one completed live verifier run")
    timestamps = [
        _utc_timestamp(item["timestamp_utc"], f"{path.name}.timestamp_utc")
        for item in events
    ]
    if timestamps != sorted(timestamps):
        raise DataContractError(f"{path.name} timestamps are not chronological")
    return events


def _validate_capture(
    *,
    capture: _Capture,
    config: PipelineConfig,
    requests_by_id: dict[str, dict[str, Any]],
    candidates_by_id: dict[str, tuple[Candidate, Any]],
    warmup_request: dict[str, Any],
    warmup_candidate: Candidate,
    warmup_sentence: Any,
    inputs: PilotInputs,
) -> tuple[_Repeat, dict[str, Any]]:
    for path in (
        capture.checkout_manifest,
        capture.verifier_manifest,
        capture.environment_manifest,
        capture.requests,
        capture.responses,
        capture.run_log,
    ):
        if not path.is_file():
            raise DataContractError(
                f"capture {capture.capture_id} is missing {path.relative_to(capture.root)}"
            )

    checkout = load_json(capture.checkout_manifest)
    if (
        not isinstance(checkout, dict)
        or checkout.get("schema_version") != "phase-b-checkout-manifest-1.0"
        or checkout.get("protocol_id") != PROTOCOL_ID
        or checkout.get("workflow_id") != WORKFLOW_ID
        or checkout.get("run_id") != capture.source_run_id
        or checkout.get("status") != "pass"
        or checkout.get("source", {}).get("worktree_clean") is not True
    ):
        raise DataContractError(
            f"capture {capture.capture_id} lacks a passing clean-checkout manifest"
        )
    source_commit = checkout.get("source", {}).get("commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
        raise DataContractError(
            f"capture {capture.capture_id} has no full source commit identity"
        )

    condition = MODE_CONDITIONS[capture.mode]
    stage = load_json(capture.verifier_manifest)
    if not isinstance(stage, dict):
        raise DataContractError(f"capture {capture.capture_id} manifest must be an object")
    expected_stage = {
        "protocol_id": PROTOCOL_ID,
        "condition_id": condition,
        "execution_mode": "live",
        "status": "completed",
        "candidate_count": len(requests_by_id),
        "warmup_candidate_count": 1,
        "warmup_latency_observation_included": False,
        "verdict_count": len(requests_by_id),
    }
    for field, expected in expected_stage.items():
        if stage.get(field) != expected:
            raise DataContractError(
                f"capture {capture.capture_id} manifest {field} must be {expected!r}"
            )
    prompt_sha256, model_sha256, decoding_sha256 = (
        next(iter(requests_by_id.values()))[field]
        for field in ("prompt_sha256", "model_manifest_sha256", "decoding_sha256")
    )
    for field, expected in {
        "prompt_sha256": prompt_sha256,
        "model_manifest_sha256": model_sha256,
        "decoding_sha256": decoding_sha256,
    }.items():
        if stage.get(field) != expected:
            raise DataContractError(
                f"capture {capture.capture_id} manifest {field} is noncanonical"
            )
    stage_inputs = stage.get("inputs")
    if not isinstance(stage_inputs, dict):
        raise DataContractError(f"capture {capture.capture_id} inputs must be an object")
    expected_named_inputs = {
        config.paths["warmup_sentences"]: sha256_file(inputs.sentences),
        config.paths["development_pilot_candidates"]: sha256_file(inputs.candidates),
        config.paths["warmup_candidates"]: sha256_file(inputs.warmup_candidates),
        config.paths["verifier_pilot_selection"]: sha256_file(inputs.pilot_selection),
    }
    if any(stage_inputs.get(key) != digest for key, digest in expected_named_inputs.items()):
        raise DataContractError(
            f"capture {capture.capture_id} does not bind the fixed sentence, candidate, "
            "warm-up, and selection input roles"
        )
    remaining_inputs = {
        key: digest
        for key, digest in stage_inputs.items()
        if key not in expected_named_inputs
    }
    if len(remaining_inputs) != 1 or set(remaining_inputs.values()) != {
        config.value["verifier"]["model_blob_sha256"]
    }:
        raise DataContractError(
            f"capture {capture.capture_id} must bind exactly one model blob and no cache ledger"
        )
    outputs = stage.get("outputs")
    if not isinstance(outputs, dict):
        raise DataContractError(f"capture {capture.capture_id} outputs must be an object")
    required_outputs = {
        f"verifier/{capture.mode}/requests.jsonl",
        f"verifier/{capture.mode}/responses.jsonl",
        f"verifier/{capture.mode}/environment-manifest.json",
        f"verifier/{capture.mode}/run-log.jsonl",
        f"verifier/{capture.mode}/verdicts.jsonl",
        f"verifier/{capture.mode}/warmup-request.jsonl",
        f"verifier/{capture.mode}/warmup-response.jsonl",
        f"verifier/{capture.mode}/model/tags.json",
        f"verifier/{capture.mode}/model/show.json",
        f"verifier/{capture.mode}/model/ollama-modelfile.txt",
    }
    if not required_outputs <= set(outputs):
        raise DataContractError(
            f"capture {capture.capture_id} omits required live-run evidence"
        )
    for relative, digest in outputs.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DataContractError(
                f"capture {capture.capture_id} has malformed output identities"
            )
        artifact = _capture_member(capture.root, relative)
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise DataContractError(
                f"capture {capture.capture_id} output differs from its manifest: {relative}"
            )

    environment = load_json(capture.environment_manifest)
    runtime_fields = ("python", "platform", "hardware", "software", "ollama", "model")
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(field), dict) for field in runtime_fields
    ):
        raise DataContractError(
            f"capture {capture.capture_id} environment runtime fields must be objects"
        )
    model = environment["model"]
    if (
        environment.get("protocol_id") != PROTOCOL_ID
        or environment.get("condition_id") != condition
        or environment.get("execution_mode") != "live"
        or not isinstance(model, dict)
        or model.get("identity_verified") is not True
        or model.get("registry_manifest_sha256")
        != config.value["verifier"]["registry_manifest_sha256"]
        or model.get("model_blob_sha256")
        != config.value["verifier"]["model_blob_sha256"]
        or model.get("tag_digest")
        != config.value["verifier"]["registry_manifest_sha256"]
        or model.get("blob_sha256") != config.value["verifier"]["model_blob_sha256"]
        or model.get("details")
        != {
            "family": "qwen3",
            "parameter_size": "32.8B",
            "quantization_level": "Q4_K_M",
        }
    ):
        raise DataContractError(
            f"capture {capture.capture_id} does not prove the pinned live Qwen identity"
        )

    request_rows = [value for _, value in iter_jsonl(capture.requests)]
    expected_requests = list(requests_by_id.values())
    if request_rows != expected_requests:
        raise DataContractError(
            f"capture {capture.capture_id} request ledger differs from canonical requests"
        )
    repeat = _interpret_repeat(
        mode=capture.mode,
        path=capture.responses,
        requests_by_id=requests_by_id,
        candidates_by_id=candidates_by_id,
    )
    _validate_retry_timing(repeat.responses, config, capture.capture_id)
    captured_verdicts = [
        value
        for _, value in iter_jsonl(
            _capture_member(capture.root, f"verifier/{capture.mode}/verdicts.jsonl")
        )
    ]
    if captured_verdicts != repeat.verdicts:
        raise DataContractError(
            f"capture {capture.capture_id} verdicts differ from response-ledger replay"
        )

    warmup_requests_path = _capture_member(
        capture.root, f"verifier/{capture.mode}/warmup-request.jsonl"
    )
    warmup_request_rows = [value for _, value in iter_jsonl(warmup_requests_path)]
    if warmup_request_rows != [warmup_request]:
        raise DataContractError(
            f"capture {capture.capture_id} warm-up request differs from the fixed input"
        )
    warmup_response_path = _capture_member(
        capture.root, f"verifier/{capture.mode}/warmup-response.jsonl"
    )
    warmup_responses = _load_response_ledger(
        warmup_response_path,
        condition,
        {warmup_candidate.candidate_id: warmup_request},
    )
    if set(warmup_responses) != {warmup_candidate.candidate_id}:
        raise DataContractError(
            f"capture {capture.capture_id} does not contain exactly one warm-up response"
        )
    warmup_verdict = _verdict_from_response(
        capture.mode,
        warmup_responses[warmup_candidate.candidate_id],
        warmup_request,
        warmup_candidate,
        warmup_sentence,
    )
    Verdict.from_mapping(
        warmup_verdict,
        f"capture {capture.capture_id} warm-up verdict",
        warmup_candidate,
    )
    if (
        warmup_verdict["response_status"] != "valid_response"
        or warmup_responses[warmup_candidate.candidate_id]["cache_hit"]
    ):
        raise DataContractError(
            f"capture {capture.capture_id} warm-up is invalid or cache-derived"
        )
    _validate_retry_timing(warmup_responses, config, f"{capture.capture_id}.warmup")
    events = _load_run_log(capture.run_log, condition)
    planned_events = Counter(
        (
            event["candidate_id"],
            event["status"],
            event["details"].get("request_sha256"),
            event["details"].get("latency_observation_included"),
        )
        for event in events
        if event["event"] == "request_planned"
    )
    expected_planned_events = Counter(
        (
            request["candidate_id"],
            "planned",
            request["request_sha256"],
            None,
        )
        for request in requests_by_id.values()
    )
    expected_planned_events[
        (
            warmup_request["candidate_id"],
            "warmup_planned",
            warmup_request["request_sha256"],
            False,
        )
    ] += 1
    if planned_events != expected_planned_events:
        raise DataContractError(
            f"capture {capture.capture_id} request-planned events do not prove the "
            "fixed warm-up and candidate call universe"
        )

    attempted_events = Counter(
        (
            event["candidate_id"],
            event["attempt"],
            event["status"],
            event["details"].get("http_status"),
            event["details"].get("elapsed_seconds"),
        )
        for event in events
        if event["event"] == "request_attempted"
    )
    expected_attempted_events: Counter[tuple[Any, ...]] = Counter()
    for responses in (repeat.responses, warmup_responses):
        for candidate_id, response in responses.items():
            for attempt in response["attempts"]:
                expected_attempted_events[
                    (
                        candidate_id,
                        attempt["attempt"],
                        attempt["transport_status"],
                        attempt["http_status"],
                        attempt["elapsed_seconds"],
                    )
                ] += 1
    if attempted_events != expected_attempted_events:
        raise DataContractError(
            f"capture {capture.capture_id} request-attempt events differ from the "
            "warm-up and candidate response ledgers"
        )
    for candidate_id in repeat.responses:
        if not any(
            event["event"] == "candidate_completed"
            and event["candidate_id"] == candidate_id
            for event in events
        ):
            raise DataContractError(
                f"capture {capture.capture_id} lacks completion provenance for {candidate_id}"
            )
    runtime_identity = {field: environment[field] for field in runtime_fields}
    runtime_identity_sha256 = hashlib.sha256(
        canonical_json_bytes(runtime_identity)
    ).hexdigest()
    provenance = {
        "capture_id": capture.capture_id,
        "mode": capture.mode,
        "repeat": capture.repeat,
        "source_run_id": capture.source_run_id,
        "source_commit": source_commit,
        "checkout_manifest_sha256": sha256_file(capture.checkout_manifest),
        "verifier_manifest_sha256": sha256_file(capture.verifier_manifest),
        "environment_manifest_sha256": sha256_file(capture.environment_manifest),
        "request_ledger_sha256": sha256_file(capture.requests),
        "response_ledger_sha256": sha256_file(capture.responses),
        "run_log_sha256": sha256_file(capture.run_log),
        "runtime_identity_sha256": runtime_identity_sha256,
    }
    return repeat, provenance


def _interpret_repeat(
    *,
    mode: str,
    path: Path,
    requests_by_id: dict[str, dict[str, Any]],
    candidates_by_id: dict[str, tuple[Candidate, Any]],
) -> _Repeat:
    condition = MODE_CONDITIONS[mode]
    responses = _load_response_ledger(path, condition, requests_by_id)
    if set(responses) != set(requests_by_id):
        missing = sorted(set(requests_by_id) - set(responses))
        extra = sorted(set(responses) - set(requests_by_id))
        raise DataContractError(
            f"{path.name}: pilot response coverage mismatch; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    verdicts: list[dict[str, Any]] = []
    ordered_ids = sorted(
        requests_by_id,
        key=lambda candidate_id: (
            requests_by_id[candidate_id]["training_seed"],
            candidate_id,
        ),
    )
    for candidate_id in ordered_ids:
        candidate, sentence = candidates_by_id[candidate_id]
        verdict = _verdict_from_response(
            mode,
            responses[candidate_id],
            requests_by_id[candidate_id],
            candidate,
            sentence,
        )
        Verdict.from_mapping(verdict, f"pilot verdict {candidate_id}", candidate)
        verdicts.append(verdict)
    return _Repeat(responses=responses, verdicts=verdicts)


def _request_input_is_source_only(request: dict[str, Any]) -> bool:
    messages = request["payload"].get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return False
    user_content = messages[1].get("content") if isinstance(messages[1], dict) else None
    if not isinstance(user_content, str) or "INPUT_JSON:\n" not in user_content:
        return False
    _, raw = user_content.rsplit("INPUT_JSON:\n", 1)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return False
    if set(value) != {"sentence", "sentence_processed", "sentence_tokens", "candidate"}:
        return False
    candidate = value.get("candidate")
    return isinstance(candidate, dict) and set(candidate) == {"head", "relation", "tail"}


def _run_verifier_pilot(
    layout: RunLayout,
    config: PipelineConfig,
    inputs: PilotInputs,
    *,
    evidence_class: str,
) -> dict[str, Any]:
    """Audit two independent development response ledgers per verifier mode."""

    layout.require_existing()
    if evidence_class not in PILOT_EVIDENCE_CLASSES:
        raise DataContractError(
            "pilot evidence class must be development-pilot or synthetic-fixture"
        )
    output_dir = layout.resolve("audit/verifier-pilot")
    paths = {
        "simple_first": output_dir / "simple-repeat-1-normalized.jsonl",
        "simple_second": output_dir / "simple-repeat-2-normalized.jsonl",
        "corrective_first": output_dir / "corrective-repeat-1-normalized.jsonl",
        "corrective_second": output_dir / "corrective-repeat-2-normalized.jsonl",
        "audit": output_dir / "pilot-audit.json",
    }
    existing = [layout.relative_identity(path) for path in paths.values() if path.exists()]
    if existing:
        raise DataContractError(
            "pilot refuses to overwrite existing artifacts: " + ", ".join(existing)
        )

    sentences = _load_sentences(inputs.sentences)
    if any(sentence.split != "development" for sentence in sentences.values()):
        raise DataContractError("pilot accepts development prepared sentences only")
    _, development_ids = _load_split_manifest(inputs.split_manifest)
    if not set(sentences) <= development_ids:
        raise DataContractError(
            "pilot prepared sentences are not members of the authoritative development split"
        )
    if evidence_class == "development-pilot" and set(sentences) != development_ids:
        raise DataContractError(
            "development-pilot evidence must supply the complete authoritative "
            "103-sentence development partition"
        )
    gold = _load_gold(inputs.gold, "CODE-SPLIT-1:development")
    if set(gold) != set(sentences):
        raise DataContractError(
            "pilot sentences and development gold must contain the same example IDs"
        )
    split_manifest_sha256 = sha256_file(inputs.split_manifest)
    candidate_index_sha256 = sha256_file(inputs.candidate_index)
    indexed_candidates = _load_candidate_index(
        inputs.candidate_index,
        config,
        layout=layout,
        sentences=sentences,
        split_manifest_sha256=split_manifest_sha256,
        development_ids=development_ids,
    )
    selected_threshold = _load_threshold(inputs.threshold_selection, config)
    threshold_record = load_json(inputs.threshold_selection)
    if threshold_record["development_gold_sha256"] != sha256_file(inputs.gold):
        raise DataContractError(
            "pilot threshold selection does not bind the supplied development gold"
        )
    if threshold_record["development_candidate_index_sha256"] != candidate_index_sha256:
        raise DataContractError(
            "pilot threshold selection does not bind the full development candidate index"
        )
    if threshold_record["split_manifest_sha256"] != split_manifest_sha256:
        raise DataContractError(
            "pilot threshold selection does not bind the authoritative split manifest"
        )

    candidate_rows = _load_candidates(inputs.candidates, sentences)
    if not candidate_rows:
        raise DataContractError("pilot requires at least one development candidate")
    candidates_by_id: dict[str, tuple[Candidate, Any]] = {}
    for candidate, _, sentence in candidate_rows:
        if candidate.split_id != "CODE-SPLIT-1:development":
            raise DataContractError("pilot accepts development candidates only")
        if candidate.triple.example_id not in gold:
            raise DataContractError("pilot candidate has no paired development gold")
        candidates_by_id[candidate.candidate_id] = (candidate, sentence)
    selection = _load_pilot_selection(
        inputs.pilot_selection,
        split_manifest_sha256=split_manifest_sha256,
        candidate_index_sha256=candidate_index_sha256,
        candidates_sha256=sha256_file(inputs.candidates),
    )
    selected_candidate_ids = sorted(candidates_by_id)
    selected_example_ids = sorted(
        {candidate.triple.example_id for candidate, _ in candidates_by_id.values()}
    )
    selected_seeds = sorted(
        {candidate.training_seed for candidate, _ in candidates_by_id.values()}
    )
    if selection["candidate_ids"] != selected_candidate_ids:
        raise DataContractError(
            "pilot selection candidate IDs differ from the supplied candidate subset"
        )
    if selection["example_ids"] != selected_example_ids:
        raise DataContractError(
            "pilot selection example IDs differ from the supplied candidate subset"
        )
    if selection["training_seeds"] != selected_seeds:
        raise DataContractError(
            "pilot selection seeds differ from the supplied candidate subset"
        )
    for candidate_id, (candidate, _) in candidates_by_id.items():
        if indexed_candidates.get(candidate_id) != (
            candidate.training_seed,
            candidate.triple.example_id,
        ):
            raise DataContractError(
                f"pilot candidate {candidate_id} is absent from the full development index"
            )
    if evidence_class == "development-pilot" and selected_seeds != config.value[
        "training_seeds"
    ]:
        raise DataContractError(
            "development-pilot evidence must cover source candidates from seeds 42-49"
        )

    warmup_rows = _load_candidates(inputs.warmup_candidates, sentences)
    if len(warmup_rows) != 1 or warmup_rows[0][2].split != "development":
        raise DataContractError(
            "pilot requires exactly one fixed source-grounded development warm-up candidate"
        )
    warmup_candidate, warmup_raw, warmup_sentence = warmup_rows[0]

    requests_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    warmup_requests_by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("simple", "corrective"):
        bundle = _prompt_bundle(config, mode)
        requests_by_mode[mode] = {
            candidate.candidate_id: _request_record(candidate, raw, sentence, bundle, config)
            for candidate, raw, sentence in candidate_rows
        }
        warmup_requests_by_mode[mode] = _request_record(
            warmup_candidate,
            warmup_raw,
            warmup_sentence,
            bundle,
            config,
        )
    source_only = all(
        _request_input_is_source_only(request)
        for requests in requests_by_mode.values()
        for request in requests.values()
    )

    captures = _load_capture_index(layout, inputs.capture_index)
    repeat_lists: dict[str, list[_Repeat]] = {"simple": [], "corrective": []}
    capture_provenance: list[dict[str, Any]] = []
    for capture in captures:
        repeat, provenance = _validate_capture(
            capture=capture,
            config=config,
            requests_by_id=requests_by_mode[capture.mode],
            candidates_by_id=candidates_by_id,
            warmup_request=warmup_requests_by_mode[capture.mode],
            warmup_candidate=warmup_candidate,
            warmup_sentence=warmup_sentence,
            inputs=inputs,
        )
        repeat_lists[capture.mode].append(repeat)
        capture_provenance.append(provenance)
    repeats = {
        mode: (mode_repeats[0], mode_repeats[1])
        for mode, mode_repeats in repeat_lists.items()
    }
    source_commits = {item["source_commit"] for item in capture_provenance}
    if len(source_commits) != 1:
        raise DataContractError("pilot live captures must use one identical source commit")
    runtime_identities = {
        item["runtime_identity_sha256"] for item in capture_provenance
    }
    if len(runtime_identities) != 1:
        raise DataContractError(
            "pilot live captures must use one identical software/Ollama/hardware runtime"
        )
    audit_checkout_path: Path | None = None
    if evidence_class == "development-pilot":
        audit_checkout_path = layout.resolve(
            "manifests/00-checkout-manifest.json", must_exist=True
        )
        audit_checkout = load_json(audit_checkout_path)
        if (
            not isinstance(audit_checkout, dict)
            or audit_checkout.get("status") != "pass"
            or audit_checkout.get("source", {}).get("worktree_clean") is not True
            or audit_checkout.get("source", {}).get("commit") not in source_commits
        ):
            raise DataContractError(
                "development-pilot capture commits must match the passing clean audit checkout"
            )

    normalized: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    mismatch_ids: dict[str, list[str]] = {}
    for mode, (first, second) in repeats.items():
        first_normalized = [_normalized(item) for item in first.verdicts]
        second_normalized = [_normalized(item) for item in second.verdicts]
        normalized[mode] = first_normalized, second_normalized
        mismatch_ids[mode] = [
            left["candidate_id"]
            for left, right in zip(first_normalized, second_normalized)
            if left != right
        ]

    all_repeats = [repeat for pair in repeats.values() for repeat in pair]
    valid_responses = all(
        verdict["response_status"] == "valid_response"
        for repeat in all_repeats
        for verdict in repeat.verdicts
    )
    no_cache_hits = all(
        not response["cache_hit"]
        for repeat in all_repeats
        for response in repeat.responses.values()
    )
    correction_grounded = all(
        verdict["action"] != "CORRECT"
        or verdict["correction_validation_status"] == "valid"
        for repeat in repeats["corrective"]
        for verdict in repeat.verdicts
    )
    independent_ledgers = {
        mode: capture_provenance[offset]["response_ledger_sha256"]
        != capture_provenance[offset + 1]["response_ledger_sha256"]
        and capture_provenance[offset]["verifier_manifest_sha256"]
        != capture_provenance[offset + 1]["verifier_manifest_sha256"]
        and capture_provenance[offset]["source_run_id"]
        != capture_provenance[offset + 1]["source_run_id"]
        for mode, offset in (("simple", 0), ("corrective", 2))
    }

    checks = [
        _check(
            "development-only-inputs",
            True,
            "all records are members of the hashed authoritative development split",
        ),
        _check(
            "full-development-index",
            True,
            "threshold and pilot subset bind the eight-seed development candidate index",
        ),
        _check(
            "predeclared-pilot-subset",
            True,
            "selection manifest binds candidate IDs before all four live calls",
        ),
        _check(
            "live-capture-provenance",
            True,
            "four clean completed live runs bind requests, model, outputs, and run logs",
        ),
        _check(
            "retry-error-policy",
            True,
            "attempt limits, chronology, 2/4-second backoff, and run-log events validate",
        ),
        _check(
            "gold-pairing",
            True,
            "every pilot candidate has paired typed development gold",
        ),
        _check(
            "development-threshold",
            True,
            f"validated selected threshold {selected_threshold}",
        ),
        _check(
            "prompt-evidence-separation",
            source_only,
            "request INPUT_JSON contains only sentence/tokens and one candidate",
        ),
        _check(
            "simple-independent-calls",
            independent_ledgers["simple"],
            "repeat ledgers have distinct file identities",
        ),
        _check(
            "corrective-independent-calls",
            independent_ledgers["corrective"],
            "repeat ledgers have distinct file identities",
        ),
        _check(
            "response-schema",
            valid_responses,
            "every final repeated response is schema-valid",
        ),
        _check(
            "no-cache-reuse",
            no_cache_hits,
            "both determinism calls are independently observed",
        ),
        _check(
            "source-grounded-corrections",
            correction_grounded,
            "every CORRECT action maps to one exact source-token triple",
        ),
        _check(
            "simple-two-call-determinism",
            not mismatch_ids["simple"],
            f"normalized mismatches={len(mismatch_ids['simple'])}",
        ),
        _check(
            "corrective-two-call-determinism",
            not mismatch_ids["corrective"],
            f"normalized mismatches={len(mismatch_ids['corrective'])}",
        ),
        _check(
            "paired-condition-universe",
            True,
            f"all conditions and four ledgers cover {len(candidate_rows)} candidates",
        ),
    ]
    pilot_passed = all(item["status"] == "pass" for item in checks)

    class_counts = Counter()
    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate, _, _ in candidate_rows:
        valid = candidate.triple in gold[candidate.triple.example_id].triples
        label = "gold_valid" if valid else "gold_invalid"
        class_counts[label] += 1
        relation_counts[candidate.triple.relation][label] += 1

    decision_diagnostics = [
        _decision_diagnostic(
            condition_id="VER-RAW",
            repeat=0,
            predicted_valid_by_id={candidate_id: True for candidate_id in candidates_by_id},
            candidates_by_id=candidates_by_id,
            gold=gold,
        ),
        _decision_diagnostic(
            condition_id="VER-CONFIDENCE",
            repeat=0,
            predicted_valid_by_id={
                candidate_id: candidate.triple_confidence >= selected_threshold
                for candidate_id, (candidate, _) in candidates_by_id.items()
            },
            candidates_by_id=candidates_by_id,
            gold=gold,
        ),
    ]
    for mode in ("simple", "corrective"):
        condition_id = MODE_CONDITIONS[mode]
        for repeat_number, repeat in enumerate(repeats[mode], start=1):
            decision_diagnostics.append(
                _decision_diagnostic(
                    condition_id=condition_id,
                    repeat=repeat_number,
                    predicted_valid_by_id={
                        verdict["candidate_id"]: (
                            verdict["response_status"] == "valid_response"
                            and verdict["action"] == "KEEP"
                        )
                        for verdict in repeat.verdicts
                    },
                    candidates_by_id=candidates_by_id,
                    gold=gold,
                )
            )

    corrective_outcome_diagnostics = [
        _corrective_outcome_diagnostic(
            repeat=repeat_number,
            verdicts=repeat.verdicts,
            candidates_by_id=candidates_by_id,
            gold=gold,
        )
        for repeat_number, repeat in enumerate(repeats["corrective"], start=1)
    ]

    output_records = {
        "simple_first": normalized["simple"][0],
        "simple_second": normalized["simple"][1],
        "corrective_first": normalized["corrective"][0],
        "corrective_second": normalized["corrective"][1],
    }
    for key, records in output_records.items():
        atomic_write_jsonl(paths[key], records)

    input_paths = [
        inputs.sentences,
        inputs.gold,
        inputs.candidates,
        inputs.warmup_candidates,
        inputs.split_manifest,
        inputs.candidate_index,
        inputs.pilot_selection,
        inputs.threshold_selection,
        inputs.capture_index,
        *([audit_checkout_path] if audit_checkout_path is not None else []),
    ]
    audit = {
        "schema_version": "phase-b-verifier-pilot-audit-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "evidence_class": evidence_class,
        "pilot_scope": "development-only",
        "pilot_status": "pass" if pilot_passed else "fail",
        "go_no_go_status": (
            "eligible_for_user_review"
            if pilot_passed and evidence_class == "development-pilot"
            else "blocked_synthetic_fixture"
            if evidence_class == "synthetic-fixture"
            else "no_go_pilot_failed"
        ),
        "publication_execution_admitted": False,
        "requires_user_go_no_go_approval": True,
        "material_protocol_review_required": not pilot_passed,
        "candidate_count": len(candidate_rows),
        "example_count": len(
            {candidate.triple.example_id for candidate, _, _ in candidate_rows}
        ),
        "source_document_count": len(
            {candidate.source_document_id for candidate, _, _ in candidate_rows}
        ),
        "training_seeds": sorted(
            {candidate.training_seed for candidate, _, _ in candidate_rows}
        ),
        "selected_development_threshold": selected_threshold,
        "input_sha256": {
            layout.relative_identity(path): sha256_file(path) for path in input_paths
        },
        "prompt_identities": {
            mode: {
                "prompt_sha256": next(iter(requests.values()))["prompt_sha256"],
                "model_manifest_sha256": next(iter(requests.values()))[
                    "model_manifest_sha256"
                ],
                "decoding_sha256": next(iter(requests.values()))["decoding_sha256"],
            }
            for mode, requests in requests_by_mode.items()
        },
        "candidate_class_balance": {
            key: class_counts.get(key, 0) for key in ("gold_valid", "gold_invalid")
        },
        "candidate_class_balance_by_relation": {
            relation: {
                key: counts.get(key, 0) for key in ("gold_valid", "gold_invalid")
            }
            for relation, counts in sorted(relation_counts.items())
        },
        "capture_provenance": capture_provenance,
        "decision_diagnostics": decision_diagnostics,
        "corrective_outcome_diagnostics": corrective_outcome_diagnostics,
        "checks": checks,
        "determinism": {
            mode: {
                "required_repeats": 2,
                "normalized_identical": not ids,
                "mismatch_count": len(ids),
                "mismatch_candidate_ids": ids,
            }
            for mode, ids in mismatch_ids.items()
        },
        "distributions": {
            mode: {
                "repeat_1": _distribution(pair[0].verdicts),
                "repeat_2": _distribution(pair[1].verdicts),
            }
            for mode, pair in repeats.items()
        },
        "telemetry": {
            mode: {
                "repeat_1": _telemetry(pair[0].responses, pair[0].verdicts),
                "repeat_2": _telemetry(pair[1].responses, pair[1].verdicts),
            }
            for mode, pair in repeats.items()
        },
        "output_sha256": {
            layout.relative_identity(paths[key]): sha256_file(paths[key])
            for key in output_records
        },
    }
    atomic_write_json(paths["audit"], audit)
    return audit


def run_verifier_pilot(
    layout: RunLayout,
    config: PipelineConfig,
    inputs: PilotInputs,
    *,
    evidence_class: str,
) -> dict[str, Any]:
    """Run the pilot audit and retain a no-go record for contract failures."""

    try:
        return _run_verifier_pilot(
            layout, config, inputs, evidence_class=evidence_class
        )
    except (DataContractError, PathContractError) as exc:
        output_dir = layout.resolve("audit/verifier-pilot")
        failure_path = output_dir / "pilot-failure.json"
        if not output_dir.exists() or not any(output_dir.iterdir()):
            input_sha256: dict[str, str] = {}
            for path in inputs.__dict__.values():
                if isinstance(path, Path) and path.is_file():
                    try:
                        identity = layout.relative_identity(path)
                    except PathContractError:
                        continue
                    input_sha256[identity] = sha256_file(path)
            failure = {
                "schema_version": "phase-b-verifier-pilot-failure-1.0",
                "protocol_id": PROTOCOL_ID,
                "workflow_id": WORKFLOW_ID,
                "evidence_class": evidence_class,
                "pilot_status": "fail",
                "go_no_go_status": "no_go_contract_failure",
                "publication_execution_admitted": False,
                "error_category": type(exc).__name__,
                "error_message": str(exc),
                "input_sha256": input_sha256,
            }
            atomic_write_json(failure_path, failure)
        raise
