"""Frozen CODE verifier request planning, live execution, and offline replay."""

from __future__ import annotations

from utils.verifier.interpretation import _parse_content

from utils.verifier.requests import PreparedSentence, _exact_keys, _load_candidates, _load_sentences, _prompt_bundle, _request_record, _sha256_value, verifier_template_id

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from utils.common.config import PipelineConfig
from utils.common.constants import PROTOCOL_ID
from utils.common.artifact_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    iter_jsonl,
    load_json,
    sha256_file,
)
from utils.common.paths import RunLayout
from utils.common.records import Candidate, Verdict
from utils.common.execution import bind_json
from utils.verifier.runtime import bind_runtime, runtime_identity


from utils.verifier.requests import MODE_CONDITIONS
EXECUTION_MODES = {"dry-run", "live", "replay"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    json_body: dict[str, Any] | None
    body_sha256: str


HttpTransport = Callable[[str, str, dict[str, Any] | None, float], HttpResult]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _EventLog:
    def __init__(self, path: Path, condition: str, execution_mode: str) -> None:
        self.path = path
        self.condition = condition
        self.execution_mode = execution_mode
        self.events = [row for _, row in iter_jsonl(path)] if path.exists() else []
        if any(row.get("sequence") != index or row.get("condition_id") != condition
               for index, row in enumerate(self.events, 1)):
            raise DataContractError("partial verifier event log is corrupt or foreign")
        self.sequence = len(self.events)

    def append(
        self,
        event: str,
        status: str,
        *,
        candidate_id: str | None = None,
        attempt: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if event in {"run_started", "request_planned", "request_attempted", "candidate_completed", "run_completed", "environment_recorded"}:
            if any(row.get("event") == event and row.get("candidate_id") == candidate_id
                   and row.get("attempt") == attempt and row.get("details") == (details or {})
                   for row in self.events):
                return
        self.sequence += 1
        record = {
            "protocol_id": PROTOCOL_ID,
            "condition_id": self.condition,
            "execution_mode": self.execution_mode,
            "event": event,
            "sequence": self.sequence,
            "timestamp_utc": _utc_now(),
            "candidate_id": candidate_id,
            "attempt": attempt,
            "status": status,
            "details": details or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(record))
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(record)




def _environment_manifest(
    config: PipelineConfig, condition: str, execution_mode: str
) -> dict[str, Any]:
    """Stage selection plus a reference to actual shared server identity."""
    return {"condition_id": condition, "execution_mode": execution_mode,
            "model": {"identity_verified": False}}


def _safe_ollama_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise DataContractError(
            "Ollama URL must be an HTTP(S) origin without credentials, path, query, or fragment"
        )
    return value.rstrip("/")


def _default_http_transport(
    method: str, url: str, payload: dict[str, Any] | None, timeout: float
) -> HttpResult:
    try:
        import requests
    except ImportError as exc:
        raise DataContractError(
            "live verifier execution requires the locked requests package"
        ) from exc
    try:
        response = requests.request(method, url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise TimeoutError("Ollama request timed out") from exc
    except requests.RequestException as exc:
        raise RuntimeError("Ollama transport failed") from exc
    body = response.content
    parsed: dict[str, Any] | None = None
    try:
        candidate = response.json()
        if isinstance(candidate, dict):
            parsed = candidate
    except ValueError:
        pass
    return HttpResult(response.status_code, parsed, hashlib.sha256(body).hexdigest())


def _verify_live_model(
    layout: RunLayout,
    config: PipelineConfig,
    mode: str,
    ollama_url: str,
    model_blob: Path,
    transport: HttpTransport,
    artifact_base: str | None = None,
) -> dict[str, Any]:
    verifier = config.value["verifier"]
    artifact_base = artifact_base or f"verifier/{mode}"
    if sha256_file(model_blob) != verifier["model_blob_sha256"]:
        raise DataContractError("the run-local Ollama model blob hash differs from the protocol")
    tags_result = transport("GET", f"{ollama_url}/api/tags", None, 30)
    if tags_result.json_body is not None:
        atomic_write_json(
            layout.resolve(f"{artifact_base}/model/tags.json"),
            tags_result.json_body,
        )
    if tags_result.status_code != 200 or tags_result.json_body is None:
        raise DataContractError("Ollama /api/tags did not return a JSON success response")
    models = tags_result.json_body.get("models")
    if not isinstance(models, list):
        raise DataContractError("Ollama /api/tags response lacks models")
    matches = [
        item
        for item in models
        if isinstance(item, dict)
        and verifier["model"] in {item.get("name"), item.get("model")}
    ]
    if len(matches) != 1:
        raise DataContractError("Ollama tag inventory does not uniquely contain the frozen model")
    observed_manifest = str(matches[0].get("digest", "")).removeprefix("sha256:")
    if observed_manifest != verifier["registry_manifest_sha256"]:
        raise DataContractError("Ollama registry manifest digest differs from the protocol")
    show_result = transport(
        "POST", f"{ollama_url}/api/show", {"model": verifier["model"]}, 30
    )
    if show_result.json_body is not None:
        atomic_write_json(
            layout.resolve(f"{artifact_base}/model/show.json"),
            show_result.json_body,
        )
    if show_result.status_code != 200 or show_result.json_body is None:
        raise DataContractError("Ollama /api/show did not return a JSON success response")
    details = show_result.json_body.get("details")
    if not isinstance(details, dict):
        raise DataContractError("Ollama /api/show response lacks model details")
    observed = {
        "family": str(details.get("family", "")).lower(),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }
    if observed != {
        "family": "qwen3",
        "parameter_size": "32.8B",
        "quantization_level": "Q4_K_M",
    }:
        raise DataContractError(f"Ollama model details differ from the protocol: {observed}")
    modelfile = show_result.json_body.get("modelfile")
    from_lines = [line for line in modelfile.splitlines()
                  if line.lstrip().upper().startswith("FROM ")] if isinstance(modelfile, str) else []
    from_blobs = {match.lower() for line in from_lines
                  for match in re.findall(r"sha256[-:]([0-9a-f]{64})", line, re.IGNORECASE)}
    if len(from_lines) != 1 or from_blobs != {verifier["model_blob_sha256"]}:
        raise DataContractError("Ollama modelfile does not reference the frozen model blob")
    if (not layout.resolve("runtime/ollama.json").exists()
            and any(layout.resolve("verifier").rglob("environment-manifest.json"))):
        raise DataContractError(
            "historical verifier evidence lacks the fresh server runtime identity; "
            "continue it with its original pinned workflow"
        )
    version_result = transport("GET", f"{ollama_url}/api/version", None, 30)
    if version_result.status_code != 200 or version_result.json_body is None:
        raise DataContractError("Ollama /api/version did not return the actual server version")
    runtime = runtime_identity(ollama_url, version_result.json_body, show_result.json_body, {
        "name": verifier["model"], "tag_digest": observed_manifest,
        "blob_sha256": verifier["model_blob_sha256"], "details": observed,
    })
    runtime_reference = bind_runtime(layout, runtime)
    return {
        "identity_verified": True,
        "tag_digest": observed_manifest,
        "blob_sha256": verifier["model_blob_sha256"],
        "details": observed,
        "tags_response_sha256": _sha256_value(tags_result.json_body),
        "show_response_sha256": _sha256_value(show_result.json_body),
        "ollama_runtime": runtime_reference,
    }


def _attempt_result(
    mode: str,
    attempt: dict[str, Any],
    candidate: Candidate,
    sentence: PreparedSentence,
) -> dict[str, Any]:
    transport_status = attempt["transport_status"]
    raw_response = attempt["raw_response"]
    raw_hash = attempt["raw_body_sha256"]
    if transport_status == "timeout":
        return {"response_status": "timeout_exhausted", "error_category": "timeout"}
    if transport_status != "response" or attempt["http_status"] != 200:
        return {"response_status": "transport_exhausted", "error_category": "transport"}
    if not isinstance(raw_response, dict):
        return {
            "response_status": "malformed",
            "error_category": "non_json_http_response",
            "raw_response_sha256": raw_hash,
        }
    content = raw_response.get("message", {}).get("content")
    if not isinstance(content, str) or not content:
        return {
            "response_status": "schema_invalid",
            "error_category": "missing_message_content",
            "raw_response_sha256": raw_hash,
        }
    status, action, reason_code, corrected, correction_status = _parse_content(
        mode, content, candidate, sentence
    )
    result: dict[str, Any] = {
        "response_status": status,
        "action": action,
        "reason_code": reason_code,
        "corrected": corrected,
        "correction_validation_status": correction_status,
        "raw_response_sha256": raw_hash,
        "error_category": None if status == "valid_response" else status,
    }
    result["telemetry"] = {
        key: raw_response.get(key)
        for key in (
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
        if key in raw_response
    }
    result["telemetry"]["attempt_elapsed_seconds"] = attempt["elapsed_seconds"]
    return result


def _validate_attempt(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be an object")
    required = {
        "attempt",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_seconds",
        "transport_status",
        "http_status",
        "raw_response",
        "raw_body_sha256",
    }
    _exact_keys(value, required, label)
    if not isinstance(value["attempt"], int) or value["attempt"] < 1:
        raise DataContractError(f"{label}.attempt must be a positive integer")
    for field in ("started_at_utc", "completed_at_utc"):
        timestamp = value[field]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise DataContractError(f"{label}.{field} must be a UTC date-time ending in Z")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
        except ValueError as exc:
            raise DataContractError(f"{label}.{field} is not an ISO date-time") from exc
        if parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp):
            raise DataContractError(f"{label}.{field} is not UTC")
    if value["transport_status"] not in {"response", "timeout", "transport_error"}:
        raise DataContractError(f"{label}.transport_status is unsupported")
    if value["http_status"] is not None and (
        not isinstance(value["http_status"], int) or isinstance(value["http_status"], bool)
    ):
        raise DataContractError(f"{label}.http_status must be an integer or null")
    if value["raw_response"] is not None and not isinstance(value["raw_response"], dict):
        raise DataContractError(f"{label}.raw_response must be an object or null")
    digest = value["raw_body_sha256"]
    if digest is not None and (not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)):
        raise DataContractError(f"{label}.raw_body_sha256 must be SHA-256 or null")
    if value["transport_status"] == "response":
        if value["http_status"] is None or digest is None:
            raise DataContractError(
                f"{label}: HTTP responses require status and raw-body SHA-256"
            )
    elif any(
        item is not None
        for item in (value["http_status"], value["raw_response"], digest)
    ):
        raise DataContractError(
            f"{label}: timeout/transport failures cannot claim an HTTP response"
        )
    elapsed = value["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise DataContractError(f"{label}.elapsed_seconds must be nonnegative")
    return value


def _load_response_ledger(
    path: Path, condition: str, requests_by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    observed_order: list[tuple[int, str]] = []
    for line_number, value in iter_jsonl(path):
        label = f"{path.name}:{line_number}"
        _exact_keys(
            value,
            {"protocol_id", "condition_id", "candidate_id", "cache_key", "cache_hit", "attempts"},
            label,
        )
        if value["protocol_id"] != PROTOCOL_ID or value["condition_id"] != condition:
            raise DataContractError(f"{label} protocol/condition identity mismatch")
        candidate_id = value["candidate_id"]
        request = requests_by_id.get(candidate_id)
        if request is None:
            raise DataContractError(f"{label} refers to an unknown candidate")
        if value["cache_key"] != request["cache_key"]:
            raise DataContractError(f"{label}.cache_key differs from the frozen request")
        if not isinstance(value["cache_hit"], bool):
            raise DataContractError(f"{label}.cache_hit must be boolean")
        attempts = value["attempts"]
        if not isinstance(attempts, list) or not attempts:
            raise DataContractError(f"{label}.attempts must be a nonempty array")
        parsed = [
            _validate_attempt(item, f"{label}.attempts[{index}]")
            for index, item in enumerate(attempts)
        ]
        if [item["attempt"] for item in parsed] != list(range(1, len(parsed) + 1)):
            raise DataContractError(f"{label}.attempts must be consecutively numbered from one")
        if len(parsed) > 3:
            raise DataContractError(f"{label}.attempts exceeds the frozen three-attempt limit")
        if candidate_id in records:
            raise DataContractError(f"{label} duplicates a candidate response")
        records[candidate_id] = value
        observed_order.append((request["training_seed"], candidate_id))
    if observed_order != sorted(observed_order):
        raise DataContractError(
            f"{path.name}: responses must be sorted by training_seed, candidate_id"
        )
    return records


def _verdict_from_response(
    mode: str,
    response: dict[str, Any],
    request: dict[str, Any],
    candidate: Candidate,
    sentence: PreparedSentence,
) -> dict[str, Any]:
    interpreted: dict[str, Any] | None = None
    for index, attempt in enumerate(response["attempts"]):
        interpreted = _attempt_result(mode, attempt, candidate, sentence)
        if interpreted["response_status"] == "valid_response":
            if index != len(response["attempts"]) - 1:
                raise DataContractError(
                    f"response ledger continues after a valid attempt for {candidate.candidate_id}"
                )
            break
    assert interpreted is not None
    valid = interpreted["response_status"] == "valid_response"
    telemetry = dict(interpreted.get("telemetry", {}))
    telemetry["cache_hit"] = response["cache_hit"]
    telemetry["latency_observation_included"] = not response["cache_hit"]
    return {
        "protocol_id": PROTOCOL_ID,
        "condition_id": MODE_CONDITIONS[mode],
        "training_seed": candidate.training_seed,
        "candidate_id": candidate.candidate_id,
        "response_status": interpreted["response_status"],
        "action": interpreted.get("action") if valid else None,
        "reason_code": interpreted.get("reason_code") if valid else None,
        "corrected": interpreted.get("corrected") if valid else None,
        "correction_validation_status": (
            interpreted.get("correction_validation_status", "not_applicable")
            if valid
            else "not_applicable"
        ),
        "raw_response_sha256": interpreted.get("raw_response_sha256"),
        "model_manifest_sha256": request["model_manifest_sha256"],
        **({"template_id": request["template_id"], "request_sha256": request["request_sha256"]}
           if "template_id" in request else
           {field: request[field] for field in ("prompt_sha256", "decoding_sha256")}),
        "attempts": len(response["attempts"]),
        "error_category": interpreted.get("error_category") if not valid else None,
        "telemetry": telemetry,
    }


def _attempt_event_details(attempt, journal):
    return {"http_status": attempt["http_status"],
            "elapsed_seconds": attempt["elapsed_seconds"], "journal": str(journal),
            "attempt_sha256": _sha256_value(attempt)}


def _saved_attempts(request, mode, event_log):
    """Validate durable responses against their original event receipts."""
    journal = request.get("_journal_path")
    binding = {"request_sha256": request["request_sha256"], "cache_key": request["cache_key"]}
    attempts = []
    if journal is not None and journal.exists():
        saved = load_json(journal)
        if saved.get("binding") != binding or set(saved) != {"binding", "attempts"}:
            raise DataContractError("verifier attempt journal belongs to another request")
        if not isinstance(saved["attempts"], list):
            raise DataContractError("verifier attempt journal is malformed")
        attempts = [_validate_attempt(a, "saved verifier attempt") for a in saved["attempts"]]
        if len(attempts) > 3 or [a["attempt"] for a in attempts] != list(range(1, len(attempts)+1)):
            raise DataContractError("verifier attempt journal exceeds its frozen budget")
    for event in event_log.events:
        if (event.get("event") != "request_attempted"
                or event.get("candidate_id") != request["candidate_id"]
                or event.get("details", {}).get("journal") != str(journal)):
            continue
        number = event.get("attempt")
        if not isinstance(number, int) or not 1 <= number <= len(attempts):
            raise DataContractError("saved verifier attempt evidence is missing")
        if event["details"] != _attempt_event_details(attempts[number-1], journal):
            raise DataContractError("saved verifier attempt changed after its event receipt")
    terminal = False
    for attempt in attempts:
        if terminal:
            raise DataContractError("verifier journal continues after a valid response")
        terminal = _attempt_result(mode, attempt, request["_candidate"], request["_sentence"])["response_status"] == "valid_response"
    return attempts, terminal


def _write_verified_jsonl(path, rows):
    expected = b"".join(canonical_json_bytes(row) for row in rows)
    if path.exists():
        if path.read_bytes() != expected:
            raise DataContractError(f"saved verifier artifact differs: {path.name}")
    else:
        atomic_write_jsonl(path, rows)


def _environment_receipt(paths):
    names = ("environment", "model_tags", "model_show")
    if paths["environment"].exists() and "ollama_runtime" not in load_json(paths["environment"]).get("model", {}):
        names += ("model_modelfile",)
    evidence_paths = {name: paths[name] for name in names}
    if any(not path.is_file() for path in evidence_paths.values()):
        raise DataContractError("saved verifier model environment evidence is missing")
    return {name: sha256_file(path) for name, path in evidence_paths.items()}


def _live_response(
    request: dict[str, Any],
    config: PipelineConfig,
    mode: str,
    ollama_url: str,
    transport: HttpTransport,
    event_log: _EventLog,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    journal = request.get("_journal_path")
    binding = {"request_sha256": request["request_sha256"], "cache_key": request["cache_key"]}
    attempts, terminal = _saved_attempts(request, mode, event_log)
    candidate_id = request["candidate_id"]
    for saved in attempts:
        event_log.append("request_attempted", saved["transport_status"], candidate_id=candidate_id,
            attempt=saved["attempt"], details=_attempt_event_details(saved, journal))
    if attempts and not terminal and len(attempts) < config.value["verifier"]["attempts"]:
        sleep(config.value["verifier"]["backoff_seconds"][len(attempts)-1])
    for attempt_number in ([] if terminal else range(len(attempts)+1, config.value["verifier"]["attempts"]+1)):
        started_at = _utc_now()
        started = time.perf_counter()
        status = "response"
        http_status: int | None = None
        raw_response: dict[str, Any] | None = None
        raw_body_sha256: str | None = None
        try:
            result = transport(
                "POST",
                f"{ollama_url}/api/chat",
                request["payload"],
                config.value["verifier"]["timeout_seconds"],
            )
            http_status = result.status_code
            raw_response = result.json_body
            raw_body_sha256 = result.body_sha256
        except TimeoutError:
            status = "timeout"
        except (OSError, RuntimeError):
            status = "transport_error"
        elapsed = time.perf_counter() - started
        attempt = {
            "attempt": attempt_number,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "transport_status": status,
            "http_status": http_status,
            "raw_response": raw_response,
            "raw_body_sha256": raw_body_sha256,
        }
        attempts.append(attempt)
        if journal is not None:
            atomic_write_json(journal, {"binding": binding, "attempts": attempts})
        event_log.append(
            "request_attempted",
            status,
            candidate_id=candidate_id,
            attempt=attempt_number,
            details=_attempt_event_details(attempt, journal),
        )
        candidate_stub = request["_candidate"]
        sentence_stub = request["_sentence"]
        if _attempt_result(mode, attempt, candidate_stub, sentence_stub)[
            "response_status"
        ] == "valid_response":
            break
        if attempt_number < config.value["verifier"]["attempts"]:
            sleep(config.value["verifier"]["backoff_seconds"][attempt_number - 1])
    return {
        "protocol_id": PROTOCOL_ID,
        "condition_id": MODE_CONDITIONS[mode],
        "candidate_id": candidate_id,
        "cache_key": request["cache_key"],
        "cache_hit": False,
        "attempts": attempts,
    }


def validate_response_cache(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    mode: str,
    sentences_path: Path,
    candidates_path: Path,
    cache_ledger_path: Path,
) -> None:
    """Validate an interrupted response ledger without writing or calling a model."""

    layout.require_existing()
    if mode not in MODE_CONDITIONS:
        raise DataContractError(f"unsupported verifier mode: {mode!r}")
    layout.relative_identity(cache_ledger_path)
    sentences = _load_sentences(sentences_path)
    candidate_rows = _load_candidates(candidates_path, sentences)
    bundle = _prompt_bundle(config, mode)
    requests_by_id = {
        candidate.candidate_id: _request_record(
            candidate, raw, sentence, bundle, config
        )
        for candidate, raw, sentence in candidate_rows
    }
    _load_response_ledger(
        cache_ledger_path, MODE_CONDITIONS[mode], requests_by_id
    )


def validate_verifier_artifacts(
    *,
    mode: str,
    sentences_path: Path,
    candidates_path: Path,
    requests_path: Path,
    responses_path: Path,
    verdicts_path: Path,
) -> None:
    """Reconstruct completed verdicts from authenticated requests/responses."""

    if mode not in MODE_CONDITIONS:
        raise DataContractError(f"unsupported verifier mode: {mode!r}")
    sentences = _load_sentences(sentences_path)
    candidate_rows = _load_candidates(candidates_path, sentences)
    requests = [value for _, value in iter_jsonl(requests_path)]
    requests_by_id = {
        request.get("candidate_id"): request
        for request in requests
        if isinstance(request, dict)
    }
    candidate_ids = {candidate.candidate_id for candidate, _, _ in candidate_rows}
    if (
        len(requests_by_id) != len(requests)
        or set(requests_by_id) != candidate_ids
    ):
        raise DataContractError(
            "completed verifier requests do not cover the candidate ledger exactly"
        )
    responses = _load_response_ledger(
        responses_path, MODE_CONDITIONS[mode], requests_by_id
    )
    if set(responses) != candidate_ids:
        raise DataContractError(
            "completed verifier responses do not cover the candidate ledger exactly"
        )
    expected = []
    for candidate, _raw, sentence in sorted(
        candidate_rows,
        key=lambda item: (item[0].training_seed, item[0].candidate_id),
    ):
        verdict = _verdict_from_response(
            mode,
            responses[candidate.candidate_id],
            requests_by_id[candidate.candidate_id],
            candidate,
            sentence,
        )
        Verdict.from_mapping(
            verdict,
            f"reconstructed verdict {candidate.candidate_id}",
            candidate,
        )
        expected.append(verdict)
    observed = [value for _, value in iter_jsonl(verdicts_path)]
    if observed != expected:
        raise DataContractError(
            "verdict output differs from deterministic response interpretation"
        )


def run_verifier(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    mode: str,
    execution_mode: str,
    sentences_path: Path,
    candidates_path: Path,
    warmup_sentences_path: Path | None = None,
    warmup_candidates_path: Path | None = None,
    response_ledger_path: Path | None = None,
    cache_ledger_path: Path | None = None,
    pilot_selection_path: Path | None = None,
    artifact_prefix: str = "",
    ollama_url: str = "http://localhost:11434",
    model_blob_path: Path | None = None,
    transport: HttpTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Plan exact requests, call the frozen model optionally, or replay frozen responses."""

    layout.require_existing()
    if mode not in MODE_CONDITIONS:
        raise DataContractError(f"unsupported verifier mode: {mode!r}")
    if execution_mode not in EXECUTION_MODES:
        raise DataContractError(f"unsupported verifier execution mode: {execution_mode!r}")
    if execution_mode == "replay" and response_ledger_path is None:
        raise DataContractError("replay requires a run-relative response ledger")
    if execution_mode != "replay" and response_ledger_path is not None:
        raise DataContractError("a response ledger is accepted only by replay execution")
    if execution_mode != "live" and cache_ledger_path is not None:
        raise DataContractError("a response cache is accepted only by live execution")
    if pilot_selection_path is not None and cache_ledger_path is not None:
        raise DataContractError(
            "development-pilot live calls cannot use a response cache ledger"
        )
    if execution_mode != "live" and any(
        value is not None
        for value in (
            warmup_sentences_path,
            warmup_candidates_path,
            model_blob_path,
            pilot_selection_path,
        )
    ):
        raise DataContractError(
            "warm-up, model-blob, and pilot-selection inputs are accepted only by "
            "live execution"
        )
    if execution_mode == "live" and model_blob_path is None:
        raise DataContractError("live execution requires the verified run-local model blob")
    if execution_mode == "live" and (
        warmup_sentences_path is None or warmup_candidates_path is None
    ):
        raise DataContractError(
            "live execution requires one frozen development warm-up candidate and its sentences"
        )
    condition = MODE_CONDITIONS[mode]
    prefix = artifact_prefix.strip("/")
    base = f"verifier/{prefix}/{mode}" if prefix else f"verifier/{mode}"
    manifest_name = f"verifier-{prefix.replace('/', '-') + '-' if prefix else ''}{mode}-{execution_mode}.json"
    paths = {
        "requests": layout.resolve(f"{base}/requests.jsonl"),
        "responses": layout.resolve(f"{base}/responses.jsonl"),
        "warmup_request": layout.resolve(f"{base}/warmup-request.jsonl"),
        "warmup_response": layout.resolve(f"{base}/warmup-response.jsonl"),
        "model_tags": layout.resolve(f"{base}/model/tags.json"),
        "model_show": layout.resolve(f"{base}/model/show.json"),
        "model_modelfile": layout.resolve(f"{base}/model/ollama-modelfile.txt"),
        "verdicts": layout.resolve(f"{base}/verdicts.jsonl"),
        "environment": layout.resolve(f"{base}/environment-manifest.json"),
        "log": layout.resolve(f"{base}/run-log.jsonl"),
        "manifest": layout.resolve(f"manifests/{manifest_name}"),
    }
    planned = [paths["requests"], paths["environment"], paths["log"], paths["manifest"]]
    if execution_mode != "dry-run":
        planned.append(paths["verdicts"])
    if execution_mode == "live":
        planned.extend(
            [
                paths["responses"],
                paths["warmup_request"],
                paths["warmup_response"],
                paths["model_tags"],
                paths["model_show"],
            ]
        )
    existing = [layout.relative_identity(path) for path in planned if path.exists()]
    if existing and execution_mode != "live":
        raise DataContractError("verifier refuses to overwrite existing artifacts: " + ", ".join(existing))
    progress_path = layout.resolve(f"{base}/progress-identity.json")
    identity = {"run_id": layout.run_id, "mode": mode, "execution_mode": execution_mode,
        "config": config.source_path, "artifact_prefix": prefix,
        "sentences_sha256": sha256_file(sentences_path), "candidates_sha256": sha256_file(candidates_path),
        "model_sha256": sha256_file(model_blob_path) if model_blob_path else None,
        "model_registry": config.value["verifier"]["registry_manifest_sha256"],
        "checkout_sha256": sha256_file(layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)) if execution_mode == "live" else None,
        "pilot_selection_sha256": sha256_file(pilot_selection_path) if pilot_selection_path else None,
        "warmup_sentences_sha256": sha256_file(warmup_sentences_path) if warmup_sentences_path else None,
        "warmup_candidates_sha256": sha256_file(warmup_candidates_path) if warmup_candidates_path else None}
    if existing and not progress_path.exists():
        raise DataContractError("partial verifier evidence lacks its original progress identity")
    bind_json(progress_path, identity)
    sentences = _load_sentences(sentences_path)
    candidate_rows = _load_candidates(candidates_path, sentences)
    if pilot_selection_path is not None and any(
        sentence.split != "development" for _, _, sentence in candidate_rows
    ):
        raise DataContractError(
            "a pilot selection manifest is accepted only with development candidates"
        )
    bundle = _prompt_bundle(config, mode)
    serialized_requests: list[dict[str, Any]] = []
    runtime_requests: list[dict[str, Any]] = []
    candidates_by_id: dict[str, tuple[Candidate, PreparedSentence]] = {}
    for candidate, raw, sentence in candidate_rows:
        record = _request_record(candidate, raw, sentence, bundle, config)
        serialized_requests.append(record)
        runtime = dict(record)
        runtime["_candidate"] = candidate
        runtime["_sentence"] = sentence
        runtime_requests.append(runtime)
        candidates_by_id[candidate.candidate_id] = (candidate, sentence)
    verdict_runtime_requests = sorted(
        runtime_requests,
        key=lambda item: (item["training_seed"], item["candidate_id"]),
    )
    requests_by_id = {record["candidate_id"]: record for record in serialized_requests}
    cached: dict[str, dict[str, Any]] = {}
    if execution_mode == "live" and cache_ledger_path is not None:
        # Validate recovery before event-log creation, output writes, warm-up,
        # model discovery, or any live request.
        cached = _load_response_ledger(
            cache_ledger_path, condition, requests_by_id
        )
    warmup_runtime: dict[str, Any] | None = None
    if execution_mode == "live":
        assert warmup_sentences_path is not None and warmup_candidates_path is not None
        warmup_sentences = _load_sentences(warmup_sentences_path)
        warmup_rows = _load_candidates(warmup_candidates_path, warmup_sentences)
        if len(warmup_rows) != 1 or warmup_rows[0][2].split != "development":
            raise DataContractError(
                "live warm-up input must contain exactly one development candidate"
            )
        warmup_candidate, warmup_raw, warmup_sentence = warmup_rows[0]
        warmup_record = _request_record(
            warmup_candidate, warmup_raw, warmup_sentence, bundle, config
        )
        warmup_runtime = dict(warmup_record)
        warmup_runtime["_candidate"] = warmup_candidate
        warmup_runtime["_sentence"] = warmup_sentence

    if execution_mode == "live":
        for runtime in runtime_requests:
            runtime["_journal_path"] = layout.resolve(f"{base}/journal/test/{runtime['candidate_id']}.json")
        if warmup_runtime is not None:
            warmup_runtime["_journal_path"] = layout.resolve(f"{base}/journal/warmup/{warmup_runtime['candidate_id']}.json")
    event_log = _EventLog(paths["log"], condition, execution_mode)
    retained_responses = {}
    if execution_mode == "live":
        assert warmup_runtime is not None
        journal_requests = [warmup_runtime, *verdict_runtime_requests]
        saved_by_path = {}
        terminal_by_path = {}
        for runtime in journal_requests:
            attempts, terminal = _saved_attempts(runtime, mode, event_log)
            saved_by_path[runtime["_journal_path"]] = attempts
            terminal_by_path[runtime["_journal_path"]] = terminal
        environment_events = [event for event in event_log.events
                              if event.get("event") == "environment_recorded"]
        if any(saved_by_path.values()) and not environment_events:
            raise DataContractError("saved verifier calls lack their original model environment receipt")
        for event in environment_events:
            if event.get("details") != _environment_receipt(paths):
                raise DataContractError("saved verifier model environment changed after its receipt")
        for ledger_path, requests in (
                (paths["warmup_response"], [warmup_runtime]),
                (paths["responses"], verdict_runtime_requests)):
            if not ledger_path.exists():
                continue
            ledger = _load_response_ledger(ledger_path, condition,
                {request["candidate_id"]: request for request in requests})
            if list(ledger) != [request["candidate_id"] for request in requests[:len(ledger)]]:
                raise DataContractError("saved verifier response ledger is not its completed prefix")
            for runtime in requests[:len(ledger)]:
                response = ledger[runtime["candidate_id"]]
                journal_path = runtime["_journal_path"]
                if ledger_path == paths["responses"] and runtime["candidate_id"] in cached:
                    if response != cached[runtime["candidate_id"]]:
                        raise DataContractError("saved verifier response ledger differs from its selected cache")
                elif (response["cache_hit"] or response["attempts"] != saved_by_path[journal_path]
                      or (not terminal_by_path[journal_path]
                          and len(response["attempts"]) != config.value["verifier"]["attempts"])):
                    raise DataContractError("saved verifier response ledger differs from its journal")
            if ledger_path == paths["responses"]:
                retained_responses = ledger
        if paths["verdicts"].exists():
            if set(retained_responses) != set(requests_by_id):
                raise DataContractError("saved verifier verdicts lack their complete response ledger")
            expected_verdicts = [_verdict_from_response(mode,
                retained_responses[runtime["candidate_id"]], runtime,
                runtime["_candidate"], runtime["_sentence"])
                for runtime in verdict_runtime_requests]
            _write_verified_jsonl(paths["verdicts"], expected_verdicts)
    event_log.append(
        "run_started",
        "started",
        details={"mode": mode, "candidate_count": len(serialized_requests)},
    )
    try:
        expected_requests = b"".join(canonical_json_bytes(r) for r in serialized_requests)
        if paths["requests"].exists() and paths["requests"].read_bytes() != expected_requests:
            raise DataContractError("saved verifier request plan changed")
        if not paths["requests"].exists():
            atomic_write_jsonl(paths["requests"], serialized_requests)
        if warmup_runtime is not None:
            _write_verified_jsonl(
                paths["warmup_request"],
                [{key: value for key, value in warmup_runtime.items() if not key.startswith("_")}],
            )
            event_log.append(
                "request_planned",
                "warmup_planned",
                candidate_id=warmup_runtime["candidate_id"],
                details={
                    "request_sha256": warmup_runtime["request_sha256"],
                    "latency_observation_included": False,
                },
            )
        for record in serialized_requests:
            event_log.append(
                "request_planned",
                "planned",
                candidate_id=record["candidate_id"],
                details={"request_sha256": record["request_sha256"]},
            )
        environment = _environment_manifest(config, condition, execution_mode)
        active_transport = transport or _default_http_transport
        endpoint: str | None = None
        if execution_mode == "live":
            endpoint = _safe_ollama_url(ollama_url)
            original_environment = load_json(paths["environment"]) if paths["environment"].exists() else None
            if original_environment is not None:
                runtime_reference = original_environment.get("model", {}).get("ollama_runtime")
                if runtime_reference is not None:
                    from utils.verifier.runtime import load_runtime
                    load_runtime(layout.run_root, runtime_reference)
            model_evidence = _verify_live_model(
                layout,
                config,
                mode,
                endpoint,
                model_blob_path,
                active_transport,
                artifact_base=(base if original_environment is None else f"{base}/resume-checks/{event_log.sequence:06d}"),
            )
            if "ollama_runtime" in model_evidence:
                environment["model"] = {key: model_evidence[key] for key in (
                    "identity_verified", "ollama_runtime", "tags_response_sha256", "show_response_sha256")}
            else:
                environment["model"].update(model_evidence)
            event_log.append(
                "model_identity_verified",
                "verified",
                details=({"ollama_runtime": model_evidence["ollama_runtime"]}
                         if "ollama_runtime" in model_evidence else {
                    "registry_manifest_sha256": model_evidence["tag_digest"],
                    "model_blob_sha256": model_evidence["blob_sha256"],
                }),
            )
            if original_environment is None:
                atomic_write_json(paths["environment"], environment)
            else:
                environment = original_environment
            event_log.append("environment_recorded", "verified", details=_environment_receipt(paths))
            assert warmup_runtime is not None
            warmup_response = _live_response(
                warmup_runtime,
                config,
                mode,
                endpoint,
                active_transport,
                event_log,
                sleep,
            )
            _write_verified_jsonl(paths["warmup_response"], [warmup_response])
            warmup_verdict = _verdict_from_response(
                mode,
                warmup_response,
                {key: value for key, value in warmup_runtime.items() if not key.startswith("_")},
                warmup_runtime["_candidate"],
                warmup_runtime["_sentence"],
            )
            Verdict.from_mapping(
                warmup_verdict,
                "generated warm-up verdict",
                warmup_runtime["_candidate"],
            )
            if warmup_verdict["response_status"] != "valid_response":
                raise DataContractError(
                    "the retained development warm-up exhausted without a valid response"
                )
        else:
            atomic_write_json(paths["environment"], environment)

        verdicts: list[dict[str, Any]] = []
        responses: dict[str, dict[str, Any]] = {}
        response_input_hash: str | None = None
        if execution_mode == "replay":
            assert response_ledger_path is not None
            responses = _load_response_ledger(
                response_ledger_path, condition, requests_by_id
            )
            response_input_hash = sha256_file(response_ledger_path)
            if set(responses) != set(requests_by_id):
                missing = sorted(set(requests_by_id) - set(responses))
                extra = sorted(set(responses) - set(requests_by_id))
                raise DataContractError(
                    "replay response coverage mismatch; "
                    f"missing={len(missing)}, extra={len(extra)}"
                )
        elif execution_mode == "live":
            with paths["responses"].open("ab") as response_handle:
                for runtime in verdict_runtime_requests:
                    candidate_id = runtime["candidate_id"]
                    if candidate_id in cached:
                        response = dict(cached[candidate_id])
                    else:
                        assert endpoint is not None
                        response = _live_response(
                            runtime,
                            config,
                            mode,
                            endpoint,
                            active_transport,
                            event_log,
                            sleep,
                        )
                    responses[candidate_id] = response
                    if candidate_id in retained_responses:
                        if retained_responses[candidate_id] != response:
                            raise DataContractError("saved verifier response changed during continuation")
                    else:
                        response_handle.write(canonical_json_bytes(response))
                        response_handle.flush()
                        os.fsync(response_handle.fileno())
            response_input_hash = sha256_file(paths["responses"])

        if execution_mode != "dry-run":
            for runtime in verdict_runtime_requests:
                candidate_id = runtime["candidate_id"]
                candidate, sentence = candidates_by_id[candidate_id]
                verdict = _verdict_from_response(
                    mode,
                    responses[candidate_id],
                    requests_by_id[candidate_id],
                    candidate,
                    sentence,
                )
                Verdict.from_mapping(verdict, f"generated verdict {candidate_id}", candidate)
                verdicts.append(verdict)
                event_log.append(
                    "candidate_completed",
                    verdict["response_status"],
                    candidate_id=candidate_id,
                    details={
                        "action": verdict["action"],
                        "correction_validation_status": verdict[
                            "correction_validation_status"
                        ],
                        "cache_hit": verdict["telemetry"]["cache_hit"],
                    },
                )
            _write_verified_jsonl(paths["verdicts"], verdicts)

        status = "planned" if execution_mode == "dry-run" else "completed"
        event_log.append(
            "run_completed",
            status,
            details={
                "request_count": len(serialized_requests),
                "verdict_count": len(verdicts),
            },
        )
        output_paths = [paths["requests"], paths["environment"], paths["log"]]
        if execution_mode != "dry-run":
            output_paths.append(paths["verdicts"])
        if execution_mode == "live":
            output_paths.extend(
                [
                    paths["responses"],
                    paths["warmup_request"],
                    paths["warmup_response"],
                    paths["model_tags"],
                    paths["model_show"],
                ]
            )
        if execution_mode == "live":
            output_paths.extend(p for p in layout.resolve(f"{base}/journal").rglob("*.json") if p.is_file())
            resume_dir = layout.resolve(f"{base}/resume-checks")
            if resume_dir.exists():
                output_paths.extend(p for p in resume_dir.rglob("*") if p.is_file())
            runtime_reference = environment.get("model", {}).get("ollama_runtime")
            if runtime_reference is not None:
                output_paths.append(layout.resolve(runtime_reference["path"], must_exist=True))
            elif paths["model_modelfile"].is_file():
                output_paths.append(paths["model_modelfile"])
        output_paths.append(progress_path)
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "condition_id": condition,
            "execution_mode": execution_mode,
            "status": status,
            "candidate_count": len(serialized_requests),
            "warmup_candidate_count": 1 if execution_mode == "live" else 0,
            "warmup_latency_observation_included": False,
            "verdict_count": len(verdicts),
            "template_id": verifier_template_id(config, mode),
            "model_manifest_sha256": config.value["verifier"][
                "registry_manifest_sha256"
            ],
            "inputs": {
                layout.relative_identity(sentences_path): sha256_file(sentences_path),
                layout.relative_identity(candidates_path): sha256_file(candidates_path),
                **(
                    {
                        layout.relative_identity(warmup_sentences_path): sha256_file(
                            warmup_sentences_path
                        ),
                        layout.relative_identity(warmup_candidates_path): sha256_file(
                            warmup_candidates_path
                        ),
                        layout.relative_identity(model_blob_path): config.value["verifier"][
                            "model_blob_sha256"
                        ],
                    }
                    if execution_mode == "live"
                    and warmup_sentences_path is not None
                    and warmup_candidates_path is not None
                    and model_blob_path is not None
                    else {}
                ),
                **(
                    {layout.relative_identity(response_ledger_path): response_input_hash}
                    if response_ledger_path is not None and response_input_hash is not None
                    else {}
                ),
                **(
                    {
                        layout.relative_identity(cache_ledger_path): sha256_file(
                            cache_ledger_path
                        ),
                        **(
                            {
                                layout.relative_identity(
                                    cache_ledger_path.with_suffix(
                                        cache_ledger_path.suffix + ".manifest.json"
                                    )
                                ): sha256_file(
                                    cache_ledger_path.with_suffix(
                                        cache_ledger_path.suffix + ".manifest.json"
                                    )
                                )
                            }
                            if cache_ledger_path.with_suffix(
                                cache_ledger_path.suffix + ".manifest.json"
                            ).is_file()
                            else {}
                        ),
                    }
                    if cache_ledger_path is not None
                    else {}
                ),
                **(
                    {
                        layout.relative_identity(pilot_selection_path): sha256_file(
                            pilot_selection_path
                        )
                    }
                    if pilot_selection_path is not None
                    else {}
                ),
            },
            "outputs": {
                layout.relative_identity(path): sha256_file(path) for path in output_paths
            },
        }
        atomic_write_json(paths["manifest"], manifest)
        return manifest
    except BaseException as exc:
        event_log.append(
            "run_failed",
            type(exc).__name__,
            details={"error_category": type(exc).__name__},
        )
        raise
