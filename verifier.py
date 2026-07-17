"""Frozen CODE verifier request planning, live execution, and offline replay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from config import PipelineConfig
from constants import ENTITY_TYPES, PROTOCOL_ID, RELATION_TYPES
from phase_b_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_bytes,
    iter_jsonl,
    load_json,
    sha256_file,
)
from paths import RunLayout, resolve_tracked_path
from records import Candidate, EntitySpan, StrictTriple, Verdict


REASON_CODES = {
    "SUPPORTED",
    "WRONG_RELATION",
    "WRONG_HEAD",
    "WRONG_TAIL",
    "UNSUPPORTED",
    "AMBIGUOUS",
}
MODE_CONDITIONS = {"simple": "VER-SIMPLE", "corrective": "VER-CORRECTIVE"}
EXECUTION_MODES = {"dry-run", "live", "replay"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PreparedSentence:
    example_id: str
    source_document_id: str
    split: str
    content: str
    processed_content: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class PromptBundle:
    mode: str
    system: str
    instruction: str
    response_schema: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    json_body: dict[str, Any] | None
    body_sha256: str


HttpTransport = Callable[[str, str, dict[str, Any] | None, float], HttpResult]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_keys(
    value: dict[str, Any], required: set[str], label: str, optional: set[str] | None = None
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing:
        raise DataContractError(f"{label} is missing required fields: {', '.join(missing)}")
    if extra:
        raise DataContractError(f"{label} contains unsupported fields: {', '.join(extra)}")


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DataContractError(f"could not read UTF-8 prompt {path.name}: {exc}") from exc


def _prompt_bundle(config: PipelineConfig, mode: str) -> PromptBundle:
    if mode not in MODE_CONDITIONS:
        raise DataContractError(f"unsupported verifier mode: {mode!r}")
    verifier = config.value["verifier"]
    source_root = config.path.parent.parent
    system_path = resolve_tracked_path(source_root, verifier["system_prompt"])
    instruction_path = resolve_tracked_path(source_root, verifier[f"{mode}_prompt"])
    schema_path = resolve_tracked_path(source_root, verifier[f"{mode}_response_schema"])
    expected_files = {
        system_path: verifier["system_prompt_sha256"],
        instruction_path: verifier[f"{mode}_prompt_sha256"],
        schema_path: verifier[f"{mode}_response_schema_sha256"],
    }
    for path, expected in expected_files.items():
        actual = sha256_file(path)
        if actual != expected:
            raise DataContractError(
                f"tracked verifier resource hash mismatch for {path.name}: {actual}"
            )
    system = _read_utf8(system_path)
    instruction = _read_utf8(instruction_path)
    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        raise DataContractError(f"{schema_path.name} must contain a JSON object")
    identity = {
        "protocol_id": PROTOCOL_ID,
        "prompt_revision": verifier["prompt_revision"],
        "mode": mode,
        "system": system,
        "instruction": instruction,
        "response_schema": schema,
    }
    digest = _sha256_value(identity)
    if digest != verifier[f"{mode}_bundle_sha256"]:
        raise DataContractError(f"config verifier {mode} prompt-bundle hash is stale")
    return PromptBundle(mode, system, instruction, schema, digest)


def _load_sentences(path: Path) -> dict[str, PreparedSentence]:
    required = {
        "protocol_id",
        "dataset_id",
        "example_id",
        "source_document_id",
        "source_country",
        "split",
        "content",
        "processed_content",
        "words",
        "entities",
        "relations",
    }
    records: dict[str, PreparedSentence] = {}
    observed_order: list[str] = []
    for line_number, value in iter_jsonl(path):
        label = f"{path.name}:{line_number}"
        _exact_keys(value, required, label)
        if value["protocol_id"] != PROTOCOL_ID:
            raise DataContractError(f"{label}.protocol_id differs from {PROTOCOL_ID}")
        if value["dataset_id"] != "CODE-ACCORD-v1.0.0":
            raise DataContractError(f"{label}.dataset_id is not CODE-ACCORD-v1.0.0")
        for field in ("example_id", "source_document_id", "content", "processed_content"):
            if not isinstance(value[field], str) or not value[field]:
                raise DataContractError(f"{label}.{field} must be a nonempty string")
        if value["split"] not in {"development", "test"}:
            raise DataContractError(f"{label}.split must be development or test")
        words = value["words"]
        if not isinstance(words, list) or not words or any(
            not isinstance(word, str) or not word for word in words
        ):
            raise DataContractError(f"{label}.words must be a nonempty string array")
        if tuple(value["processed_content"].split()) != tuple(words):
            raise DataContractError(
                f"{label}.processed_content tokens differ from the words field"
            )
        if not isinstance(value["entities"], list) or not isinstance(value["relations"], list):
            raise DataContractError(f"{label}.entities and relations must be arrays")
        example_id = value["example_id"]
        if example_id in records:
            raise DataContractError(f"{label} duplicates example_id {example_id}")
        records[example_id] = PreparedSentence(
            example_id=example_id,
            source_document_id=value["source_document_id"],
            split=value["split"],
            content=value["content"],
            processed_content=value["processed_content"],
            words=tuple(words),
        )
        observed_order.append(example_id)
    if not records:
        raise DataContractError(f"{path.name} contains no prepared sentences")
    if observed_order != sorted(observed_order):
        raise DataContractError(f"{path.name} must be sorted by example_id")
    return records


def _source_text(sentence: PreparedSentence, span: EntitySpan, label: str) -> str:
    if span.end >= len(sentence.words):
        raise DataContractError(f"{label} span exceeds sentence token boundaries")
    text = " ".join(sentence.words[span.start : span.end + 1])
    if span.text is not None and span.text != text:
        raise DataContractError(f"{label} text differs from the source token span")
    return text


def _load_candidates(
    path: Path, sentences: dict[str, PreparedSentence]
) -> list[tuple[Candidate, dict[str, Any], PreparedSentence]]:
    records: list[tuple[Candidate, dict[str, Any], PreparedSentence]] = []
    seen: set[str] = set()
    observed_order: list[tuple[int, str, str]] = []
    for line_number, value in iter_jsonl(path):
        label = f"{path.name}:{line_number}"
        candidate = Candidate.from_mapping(value, label)
        sentence = sentences.get(candidate.triple.example_id)
        if sentence is None:
            raise DataContractError(f"{label} refers to an unknown prepared sentence")
        if candidate.split_id != f"CODE-SPLIT-1:{sentence.split}":
            raise DataContractError(f"{label}.split_id differs from its prepared sentence")
        if candidate.source_document_id != sentence.source_document_id:
            raise DataContractError(f"{label}.source_document_id differs from its sentence")
        _source_text(sentence, candidate.triple.head, f"{label}.head")
        _source_text(sentence, candidate.triple.tail, f"{label}.tail")
        if candidate.candidate_id in seen:
            raise DataContractError(f"{label} duplicates candidate_id")
        seen.add(candidate.candidate_id)
        records.append((candidate, value, sentence))
        observed_order.append(candidate.sort_key())
    if not records:
        raise DataContractError(f"{path.name} contains no candidates")
    if observed_order != sorted(observed_order):
        raise DataContractError(
            f"{path.name} candidates must be sorted by training_seed, example_id, candidate_id"
        )
    return records


def _decoding(config: PipelineConfig) -> dict[str, Any]:
    value = config.value["verifier"]
    return {
        "stream": value["stream"],
        "think": value["think"],
        "options": {
            "temperature": value["temperature"],
            "seed": value["seed"],
            "top_k": value["top_k"],
            "top_p": value["top_p"],
            "min_p": value["min_p"],
            "repeat_penalty": value["repeat_penalty"],
            "num_ctx": value["num_ctx"],
            "num_predict": value["num_predict"],
        },
    }


def verifier_identity(config: PipelineConfig, mode: str) -> tuple[str, str, str]:
    """Return the frozen prompt, model-manifest, and decoding identities."""

    bundle = _prompt_bundle(config, mode)
    return (
        bundle.sha256,
        config.value["verifier"]["registry_manifest_sha256"],
        _sha256_value(_decoding(config)),
    )


def _request_record(
    candidate: Candidate,
    candidate_raw: dict[str, Any],
    sentence: PreparedSentence,
    bundle: PromptBundle,
    config: PipelineConfig,
) -> dict[str, Any]:
    condition = MODE_CONDITIONS[bundle.mode]
    head_text = _source_text(sentence, candidate.triple.head, "candidate.head")
    tail_text = _source_text(sentence, candidate.triple.tail, "candidate.tail")
    input_record = {
        "sentence": sentence.content,
        "sentence_processed": sentence.processed_content,
        "sentence_tokens": list(sentence.words),
        "candidate": {
            "head": {
                "text": head_text,
                "type": candidate.triple.head.entity_type,
                "span": [candidate.triple.head.start, candidate.triple.head.end],
            },
            "relation": candidate.triple.relation,
            "tail": {
                "text": tail_text,
                "type": candidate.triple.tail.entity_type,
                "span": [candidate.triple.tail.start, candidate.triple.tail.end],
            },
        },
    }
    user_content = (
        bundle.instruction.rstrip("\n")
        + "\n\nINPUT_JSON:\n"
        + canonical_json_bytes(input_record).decode("utf-8").rstrip("\n")
    )
    decoding = _decoding(config)
    payload = {
        "model": config.value["verifier"]["model"],
        "messages": [
            {"role": "system", "content": bundle.system},
            {"role": "user", "content": user_content},
        ],
        "format": bundle.response_schema,
        **decoding,
    }
    candidate_sha256 = _sha256_value(candidate_raw)
    decoding_sha256 = _sha256_value(decoding)
    cache_identity = {
        "model_manifest_sha256": config.value["verifier"]["registry_manifest_sha256"],
        "prompt_sha256": bundle.sha256,
        "decoding_sha256": decoding_sha256,
        "mode": bundle.mode,
        "candidate_sha256": candidate_sha256,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "condition_id": condition,
        "training_seed": candidate.training_seed,
        "example_id": candidate.triple.example_id,
        "source_document_id": candidate.source_document_id,
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate_sha256,
        "cache_key": _sha256_value(cache_identity),
        "prompt_sha256": bundle.sha256,
        "model_manifest_sha256": config.value["verifier"]["registry_manifest_sha256"],
        "decoding_sha256": decoding_sha256,
        "request_sha256": _sha256_value(payload),
        "payload": payload,
    }


class _EventLog:
    def __init__(self, path: Path, condition: str, execution_mode: str) -> None:
        self.path = path
        self.condition = condition
        self.execution_mode = execution_mode
        self.sequence = 0

    def append(
        self,
        event: str,
        status: str,
        *,
        candidate_id: str | None = None,
        attempt: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
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


def _command_output(arguments: list[str], timeout: float = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "returncode": None, "stdout": None}
    return {
        "status": "available" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip() or None,
    }


def _physical_memory_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
        except (AttributeError, OSError):
            return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * page_count)


def _environment_manifest(
    config: PipelineConfig, condition: str, execution_mode: str
) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("requests", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware: dict[str, Any] = {
        "logical_cpu_count": os.cpu_count(),
        "processor": platform.processor() or None,
        "physical_memory_bytes": _physical_memory_bytes(),
        "cuda_available": False,
        "cuda_runtime": None,
        "cuda_devices": [],
        "cuda_probe_status": "not_executed_non_live",
    }
    if execution_mode == "live":
        probe_script = (
            "import json,torch; available=bool(torch.cuda.is_available()); "
            "print(json.dumps({'cuda_available':available,'cuda_runtime':torch.version.cuda,"
            "'cuda_devices':[{'index':i,'name':torch.cuda.get_device_name(i),"
            "'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} "
            "for i in range(torch.cuda.device_count())] if available else []},"
            "sort_keys=True))"
        )
        probe = _command_output([sys.executable, "-B", "-c", probe_script], timeout=30)
        hardware["cuda_probe_status"] = probe["status"]
        if probe["status"] == "available" and probe["stdout"] is not None:
            try:
                observed_hardware = json.loads(probe["stdout"])
            except json.JSONDecodeError:
                hardware["cuda_probe_status"] = "malformed"
            else:
                if isinstance(observed_hardware, dict):
                    hardware.update(observed_hardware)
    return {
        "protocol_id": PROTOCOL_ID,
        "condition_id": condition,
        "execution_mode": execution_mode,
        "created_at_utc": _utc_now(),
        "python": {
            "expected": config.value["environment"]["python"],
            "observed": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "hardware": hardware,
        "software": {
            "expected_uv": config.value["environment"]["uv"],
            "uv_version_command": _command_output(["uv", "--version"]),
            "packages": packages,
        },
        "ollama": {"version_command": _command_output(["ollama", "--version"])},
        "model": {
            "name": config.value["verifier"]["model"],
            "registry_manifest_sha256": config.value["verifier"][
                "registry_manifest_sha256"
            ],
            "model_blob_sha256": config.value["verifier"]["model_blob_sha256"],
            "architecture": "Qwen3",
            "parameters": "32.8B",
            "quantization": "Q4_K_M",
            "identity_verified": False,
        },
    }


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
) -> dict[str, Any]:
    verifier = config.value["verifier"]
    if sha256_file(model_blob) != verifier["model_blob_sha256"]:
        raise DataContractError("the run-local Ollama model blob hash differs from the protocol")
    tags_result = transport("GET", f"{ollama_url}/api/tags", None, 30)
    if tags_result.json_body is not None:
        atomic_write_json(
            layout.resolve(f"verifier/{mode}/model/tags.json"),
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
            layout.resolve(f"verifier/{mode}/model/show.json"),
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
    if not isinstance(modelfile, str) or verifier["model_blob_sha256"] not in modelfile:
        raise DataContractError("Ollama modelfile does not reference the frozen model blob")
    cli_modelfile = _command_output(
        ["ollama", "show", "--modelfile", verifier["model"]], timeout=60
    )
    if cli_modelfile["stdout"] is not None:
        atomic_write_text(
            layout.resolve(f"verifier/{mode}/model/ollama-modelfile.txt"),
            cli_modelfile["stdout"] + "\n",
        )
    if (
        cli_modelfile["status"] != "available"
        or verifier["model_blob_sha256"] not in (cli_modelfile["stdout"] or "")
    ):
        raise DataContractError("ollama show --modelfile did not prove the frozen blob identity")
    return {
        "identity_verified": True,
        "tag_digest": observed_manifest,
        "blob_sha256": verifier["model_blob_sha256"],
        "details": observed,
        "tags_response_sha256": _sha256_value(tags_result.json_body),
        "show_response_sha256": _sha256_value(show_result.json_body),
    }


def _token_span(
    words: tuple[str, ...], text: str, entity_type: str
) -> tuple[EntitySpan | None, str]:
    tokens = tuple(text.split())
    if not tokens or " ".join(tokens) != text:
        return None, "out_of_source"
    matches = [
        start
        for start in range(len(words) - len(tokens) + 1)
        if words[start : start + len(tokens)] == tokens
    ]
    if not matches:
        return None, "out_of_source"
    if len(matches) > 1:
        return None, "ambiguous"
    start = matches[0]
    return EntitySpan(start, start + len(tokens) - 1, entity_type, text), "valid"


def _parse_content(
    mode: str,
    content: str,
    candidate: Candidate,
    sentence: PreparedSentence,
) -> tuple[str, str | None, str | None, dict[str, Any] | None, str]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return "malformed", None, None, None, "not_applicable"
    if not isinstance(value, dict):
        return "schema_invalid", None, None, None, "not_applicable"
    expected = {"action", "reason_code"} if mode == "simple" else {
        "action",
        "reason_code",
        "corrected",
    }
    if set(value) != expected:
        return "schema_invalid", None, None, None, "not_applicable"
    actions = {"KEEP", "DISCARD", "UNCERTAIN"}
    if mode == "corrective":
        actions.add("CORRECT")
    action = value.get("action")
    reason_code = value.get("reason_code")
    if action not in actions or reason_code not in REASON_CODES:
        return "schema_invalid", None, None, None, "not_applicable"
    if mode == "simple":
        return "valid_response", action, reason_code, None, "not_applicable"
    corrected = value.get("corrected")
    if action != "CORRECT":
        if corrected is not None:
            return "schema_invalid", None, None, None, "not_applicable"
        return "valid_response", action, reason_code, None, "not_applicable"
    if not isinstance(corrected, dict):
        return "schema_invalid", None, None, None, "schema_invalid"
    fields = {"head_text", "head_type", "relation", "tail_text", "tail_type"}
    if set(corrected) != fields:
        return "schema_invalid", None, None, None, "schema_invalid"
    if (
        not isinstance(corrected["head_text"], str)
        or not corrected["head_text"]
        or not isinstance(corrected["tail_text"], str)
        or not corrected["tail_text"]
        or corrected["head_type"] not in ENTITY_TYPES
        or corrected["tail_type"] not in ENTITY_TYPES
        or corrected["relation"] not in RELATION_TYPES
    ):
        return "schema_invalid", None, None, None, "schema_invalid"
    head, head_status = _token_span(
        sentence.words, corrected["head_text"], corrected["head_type"]
    )
    tail, tail_status = _token_span(
        sentence.words, corrected["tail_text"], corrected["tail_type"]
    )
    if "ambiguous" in {head_status, tail_status}:
        return "valid_response", action, reason_code, None, "ambiguous"
    if "out_of_source" in {head_status, tail_status}:
        return "valid_response", action, reason_code, None, "out_of_source"
    assert head is not None and tail is not None
    triple = StrictTriple(
        example_id=candidate.triple.example_id,
        head=head,
        relation=corrected["relation"],
        tail=tail,
    )
    if triple.key() == candidate.triple.key():
        return "valid_response", action, reason_code, None, "schema_invalid"
    return (
        "valid_response",
        action,
        reason_code,
        {
            "head": head.to_mapping(),
            "relation": triple.relation,
            "tail": tail.to_mapping(),
        },
        "valid",
    )


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
        "prompt_sha256": request["prompt_sha256"],
        "model_manifest_sha256": request["model_manifest_sha256"],
        "decoding_sha256": request["decoding_sha256"],
        "attempts": len(response["attempts"]),
        "error_category": interpreted.get("error_category") if not valid else None,
        "telemetry": telemetry,
    }


def _live_response(
    request: dict[str, Any],
    config: PipelineConfig,
    mode: str,
    ollama_url: str,
    transport: HttpTransport,
    event_log: _EventLog,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    candidate_id = request["candidate_id"]
    for attempt_number in range(1, config.value["verifier"]["attempts"] + 1):
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
        event_log.append(
            "request_attempted",
            status,
            candidate_id=candidate_id,
            attempt=attempt_number,
            details={"http_status": http_status, "elapsed_seconds": elapsed},
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
    base = f"verifier/{mode}"
    paths = {
        "requests": layout.resolve(f"{base}/requests.jsonl"),
        "responses": layout.resolve(f"{base}/responses.jsonl"),
        "warmup_request": layout.resolve(f"{base}/warmup-request.jsonl"),
        "warmup_response": layout.resolve(f"{base}/warmup-response.jsonl"),
        "model_tags": layout.resolve(f"{base}/model/tags.json"),
        "model_show": layout.resolve(f"{base}/model/show.json"),
        "model_modelfile": layout.resolve(f"{base}/model/ollama-modelfile.txt"),
        "verdicts": layout.resolve(config.paths[f"{mode}_verdicts"]),
        "environment": layout.resolve(f"{base}/environment-manifest.json"),
        "log": layout.resolve(f"{base}/run-log.jsonl"),
        "manifest": layout.resolve(f"manifests/verifier-{mode}-{execution_mode}.json"),
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
                paths["model_modelfile"],
            ]
        )
    existing = [layout.relative_identity(path) for path in planned if path.exists()]
    if existing:
        raise DataContractError(
            "verifier refuses to overwrite existing artifacts: " + ", ".join(existing)
        )

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

    event_log = _EventLog(paths["log"], condition, execution_mode)
    event_log.append(
        "run_started",
        "started",
        details={"mode": mode, "candidate_count": len(serialized_requests)},
    )
    try:
        atomic_write_jsonl(paths["requests"], serialized_requests)
        if warmup_runtime is not None:
            atomic_write_jsonl(
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
            model_evidence = _verify_live_model(
                layout,
                config,
                mode,
                endpoint,
                model_blob_path,
                active_transport,
            )
            environment["model"].update(model_evidence)
            environment["ollama"]["origin"] = endpoint
            event_log.append(
                "model_identity_verified",
                "verified",
                details={
                    "registry_manifest_sha256": model_evidence["tag_digest"],
                    "model_blob_sha256": model_evidence["blob_sha256"],
                },
            )
            atomic_write_json(paths["environment"], environment)
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
            atomic_write_jsonl(paths["warmup_response"], [warmup_response])
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
            cached: dict[str, dict[str, Any]] = {}
            if cache_ledger_path is not None:
                cached = _load_response_ledger(
                    cache_ledger_path, condition, requests_by_id
                )
            with paths["responses"].open("ab") as response_handle:
                for runtime in verdict_runtime_requests:
                    candidate_id = runtime["candidate_id"]
                    if candidate_id in cached:
                        response = {**cached[candidate_id], "cache_hit": True}
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
            atomic_write_jsonl(paths["verdicts"], verdicts)

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
                    paths["model_modelfile"],
                ]
            )
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "condition_id": condition,
            "execution_mode": execution_mode,
            "status": status,
            "candidate_count": len(serialized_requests),
            "warmup_candidate_count": 1 if execution_mode == "live" else 0,
            "warmup_latency_observation_included": False,
            "verdict_count": len(verdicts),
            "prompt_sha256": bundle.sha256,
            "model_manifest_sha256": config.value["verifier"][
                "registry_manifest_sha256"
            ],
            "decoding_sha256": serialized_requests[0]["decoding_sha256"],
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
                    {layout.relative_identity(cache_ledger_path): sha256_file(cache_ledger_path)}
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
