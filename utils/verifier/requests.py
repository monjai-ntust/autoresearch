"""verifier requests responsibilities."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from utils.common.config import PipelineConfig
from utils.common.constants import PROTOCOL_ID
from utils.common.artifact_io import DataContractError, canonical_json_bytes, iter_jsonl, load_json
from utils.common.paths import discover_source_root, resolve_tracked_path
from utils.common.records import Candidate, EntitySpan

MODE_CONDITIONS = {"simple": "VER-SIMPLE", "corrective": "VER-CORRECTIVE"}

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
    source_root = discover_source_root(config.path)
    system_path = resolve_tracked_path(source_root, verifier["system_prompt"])
    instruction_path = resolve_tracked_path(source_root, verifier[f"{mode}_prompt"])
    schema_path = resolve_tracked_path(source_root, verifier[f"{mode}_response_schema"])
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
