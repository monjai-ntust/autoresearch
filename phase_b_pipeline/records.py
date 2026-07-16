"""Typed machine-readable records for ``CODE-STRICT-1`` scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .constants import ENTITY_TYPES, PROTOCOL_ID, RELATION_TYPES, TRAINING_SEEDS
from .io import DataContractError


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise DataContractError(f"{label} is missing required field {key!r}")
    return mapping[key]


def _exact_keys(
    mapping: dict[str, Any], required: set[str], label: str, optional: set[str] | None = None
) -> None:
    optional = optional or set()
    missing = sorted(required - set(mapping))
    extra = sorted(set(mapping) - required - optional)
    if missing:
        raise DataContractError(f"{label} is missing required fields: {', '.join(missing)}")
    if extra:
        raise DataContractError(f"{label} contains unsupported fields: {', '.join(extra)}")


def _string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise DataContractError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise DataContractError(f"{label} must be >= {minimum}")
    return value


def _sha256(value: Any, label: str, *, allow_null: bool = False) -> str | None:
    if value is None and allow_null:
        return None
    text = _string(value, label).lower()
    if not _SHA256_RE.fullmatch(text):
        raise DataContractError(f"{label} must be a 64-character SHA-256 hex digest")
    return text


def _input_hashes(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise DataContractError(f"{label} must be a nonempty object")
    result: dict[str, str] = {}
    for key, digest in value.items():
        result[_string(key, f"{label} key")] = _sha256(digest, f"{label}.{key}") or ""
    return result


@dataclass(frozen=True, order=True)
class EntitySpan:
    start: int
    end: int
    entity_type: str
    text: str | None = field(default=None, compare=False)

    @classmethod
    def from_mapping(cls, value: Any, label: str) -> "EntitySpan":
        if not isinstance(value, dict):
            raise DataContractError(f"{label} must be an object")
        _exact_keys(value, {"start", "end", "type"}, label, {"text"})
        start = _integer(_require(value, "start", label), f"{label}.start", minimum=0)
        end = _integer(_require(value, "end", label), f"{label}.end", minimum=0)
        if end < start:
            raise DataContractError(f"{label}.end must be >= {label}.start")
        entity_type = _string(_require(value, "type", label), f"{label}.type")
        if entity_type not in ENTITY_TYPES:
            raise DataContractError(
                f"{label}.type must be one of {list(ENTITY_TYPES)}, got {entity_type!r}"
            )
        text = value.get("text")
        if text is not None:
            text = _string(text, f"{label}.text")
        return cls(start=start, end=end, entity_type=entity_type, text=text)

    def key(self) -> tuple[int, int, str]:
        return self.start, self.end, self.entity_type

    def to_mapping(self, *, include_text: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
        }
        if include_text and self.text is not None:
            value["text"] = self.text
        return value


@dataclass(frozen=True, order=True)
class StrictTriple:
    example_id: str
    head: EntitySpan
    relation: str
    tail: EntitySpan

    @classmethod
    def from_mapping(
        cls, value: Any, *, example_id: str, label: str
    ) -> "StrictTriple":
        if not isinstance(value, dict):
            raise DataContractError(f"{label} must be an object")
        _exact_keys(value, {"head", "relation", "tail"}, label)
        relation = _string(_require(value, "relation", label), f"{label}.relation")
        if relation not in RELATION_TYPES:
            raise DataContractError(
                f"{label}.relation must be one of {list(RELATION_TYPES)}, got {relation!r}"
            )
        return cls(
            example_id=example_id,
            head=EntitySpan.from_mapping(_require(value, "head", label), f"{label}.head"),
            relation=relation,
            tail=EntitySpan.from_mapping(_require(value, "tail", label), f"{label}.tail"),
        )

    def key(self) -> tuple[str, int, int, str, str, int, int, str]:
        return (
            self.example_id,
            self.head.start,
            self.head.end,
            self.head.entity_type,
            self.relation,
            self.tail.start,
            self.tail.end,
            self.tail.entity_type,
        )

    def to_mapping(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "head": self.head.to_mapping(include_text=include_text),
            "relation": self.relation,
            "tail": self.tail.to_mapping(include_text=include_text),
        }


def candidate_id_for(training_seed: int, triple: StrictTriple) -> str:
    identity = {
        "protocol_id": PROTOCOL_ID,
        "training_seed": training_seed,
        "strict_key": list(triple.key()),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "cand-" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Candidate:
    split_id: str
    training_seed: int
    source_document_id: str
    candidate_id: str
    triple: StrictTriple
    triple_confidence: float
    input_hashes: dict[str, str]

    @classmethod
    def from_mapping(cls, value: dict[str, Any], label: str) -> "Candidate":
        _exact_keys(
            value,
            {
                "protocol_id",
                "split_id",
                "training_seed",
                "example_id",
                "source_document_id",
                "candidate_id",
                "head",
                "relation",
                "tail",
                "triple_confidence",
                "input_hashes",
            },
            label,
        )
        if _require(value, "protocol_id", label) != PROTOCOL_ID:
            raise DataContractError(f"{label}.protocol_id does not match {PROTOCOL_ID}")
        seed = _integer(_require(value, "training_seed", label), f"{label}.training_seed")
        if seed not in TRAINING_SEEDS:
            raise DataContractError(f"{label}.training_seed is outside seeds 42-49")
        example_id = _string(_require(value, "example_id", label), f"{label}.example_id")
        triple = StrictTriple.from_mapping(
            {field: value[field] for field in ("head", "relation", "tail")},
            example_id=example_id,
            label=label,
        )
        confidence = _require(value, "triple_confidence", label)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise DataContractError(f"{label}.triple_confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise DataContractError(f"{label}.triple_confidence must be finite and in [0, 1]")
        candidate_id = _string(_require(value, "candidate_id", label), f"{label}.candidate_id")
        expected_id = candidate_id_for(seed, triple)
        if candidate_id != expected_id:
            raise DataContractError(
                f"{label}.candidate_id is not the canonical identity; expected {expected_id}"
            )
        return cls(
            split_id=_string(_require(value, "split_id", label), f"{label}.split_id"),
            training_seed=seed,
            source_document_id=_string(
                _require(value, "source_document_id", label),
                f"{label}.source_document_id",
            ),
            candidate_id=candidate_id,
            triple=triple,
            triple_confidence=confidence,
            input_hashes=_input_hashes(_require(value, "input_hashes", label), f"{label}.input_hashes"),
        )

    def sort_key(self) -> tuple[int, str, str]:
        return self.training_seed, self.triple.example_id, self.candidate_id


@dataclass(frozen=True)
class GoldRecord:
    split_id: str
    example_id: str
    source_document_id: str
    triples: frozenset[StrictTriple]
    input_hashes: dict[str, str]

    @classmethod
    def from_mapping(cls, value: dict[str, Any], label: str) -> "GoldRecord":
        _exact_keys(
            value,
            {
                "protocol_id",
                "split_id",
                "example_id",
                "source_document_id",
                "gold_triples",
                "input_hashes",
            },
            label,
        )
        if _require(value, "protocol_id", label) != PROTOCOL_ID:
            raise DataContractError(f"{label}.protocol_id does not match {PROTOCOL_ID}")
        example_id = _string(_require(value, "example_id", label), f"{label}.example_id")
        raw_triples = _require(value, "gold_triples", label)
        if not isinstance(raw_triples, list):
            raise DataContractError(f"{label}.gold_triples must be an array")
        triples = [
            StrictTriple.from_mapping(item, example_id=example_id, label=f"{label}.gold_triples[{index}]")
            for index, item in enumerate(raw_triples)
        ]
        if len(set(triples)) != len(triples):
            raise DataContractError(f"{label}.gold_triples contains a duplicate strict key")
        return cls(
            split_id=_string(_require(value, "split_id", label), f"{label}.split_id"),
            example_id=example_id,
            source_document_id=_string(
                _require(value, "source_document_id", label),
                f"{label}.source_document_id",
            ),
            triples=frozenset(triples),
            input_hashes=_input_hashes(_require(value, "input_hashes", label), f"{label}.input_hashes"),
        )


_RESPONSE_STATUSES = {
    "valid_response",
    "malformed",
    "timeout_exhausted",
    "transport_exhausted",
    "schema_invalid",
}
_ACTIONS = {"KEEP", "CORRECT", "DISCARD", "UNCERTAIN"}
_CORRECTION_STATUSES = {
    "not_applicable",
    "valid",
    "out_of_source",
    "ambiguous",
    "schema_invalid",
    "direction_invalid",
}


@dataclass(frozen=True)
class Verdict:
    condition_id: str
    training_seed: int
    candidate_id: str
    response_status: str
    action: str | None
    corrected: StrictTriple | None
    correction_validation_status: str
    raw_response_sha256: str | None
    prompt_sha256: str
    model_manifest_sha256: str
    decoding_sha256: str
    attempts: int
    error_category: str | None
    telemetry: dict[str, Any]

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], label: str, candidate: Candidate
    ) -> "Verdict":
        _exact_keys(
            value,
            {
                "protocol_id",
                "condition_id",
                "training_seed",
                "candidate_id",
                "response_status",
                "action",
                "corrected",
                "correction_validation_status",
                "raw_response_sha256",
                "prompt_sha256",
                "model_manifest_sha256",
                "decoding_sha256",
                "attempts",
                "error_category",
                "telemetry",
            },
            label,
        )
        if _require(value, "protocol_id", label) != PROTOCOL_ID:
            raise DataContractError(f"{label}.protocol_id does not match {PROTOCOL_ID}")
        condition = _string(_require(value, "condition_id", label), f"{label}.condition_id")
        if condition not in {"VER-SIMPLE", "VER-CORRECTIVE"}:
            raise DataContractError(f"{label}.condition_id is not a verifier condition")
        seed = _integer(_require(value, "training_seed", label), f"{label}.training_seed")
        candidate_id = _string(_require(value, "candidate_id", label), f"{label}.candidate_id")
        if seed != candidate.training_seed or candidate_id != candidate.candidate_id:
            raise DataContractError(f"{label} identity does not match its candidate")
        response_status = _string(
            _require(value, "response_status", label), f"{label}.response_status"
        )
        if response_status not in _RESPONSE_STATUSES:
            raise DataContractError(f"{label}.response_status is unsupported")
        action = value.get("action")
        if action is not None:
            action = _string(action, f"{label}.action")
            if action not in _ACTIONS:
                raise DataContractError(f"{label}.action is unsupported")
        if response_status == "valid_response" and action is None:
            raise DataContractError(f"{label}.action is required for a valid response")
        if response_status != "valid_response" and action is not None:
            raise DataContractError(f"{label}.action must be null for a failed response")
        if condition == "VER-SIMPLE" and action == "CORRECT":
            raise DataContractError(f"{label}: simple mode cannot return CORRECT")

        correction_status = _string(
            _require(value, "correction_validation_status", label),
            f"{label}.correction_validation_status",
        )
        if correction_status not in _CORRECTION_STATUSES:
            raise DataContractError(f"{label}.correction_validation_status is unsupported")
        corrected_raw = value.get("corrected")
        corrected = None
        if corrected_raw is not None:
            corrected = StrictTriple.from_mapping(
                corrected_raw,
                example_id=candidate.triple.example_id,
                label=f"{label}.corrected",
            )
        if action == "CORRECT" and correction_status == "valid" and corrected is None:
            raise DataContractError(f"{label}: a valid correction requires corrected")
        if corrected is not None and not (action == "CORRECT" and correction_status == "valid"):
            raise DataContractError(f"{label}: corrected is allowed only for a valid CORRECT action")
        if action != "CORRECT" and correction_status != "not_applicable":
            raise DataContractError(
                f"{label}: non-CORRECT actions require correction_validation_status=not_applicable"
            )

        error_category = value.get("error_category")
        if error_category is not None:
            error_category = _string(error_category, f"{label}.error_category")
        if response_status != "valid_response" and error_category is None:
            raise DataContractError(f"{label}.error_category is required for a failed response")
        if response_status == "valid_response" and error_category is not None:
            raise DataContractError(f"{label}.error_category must be null for a valid response")
        telemetry = value.get("telemetry")
        if not isinstance(telemetry, dict):
            raise DataContractError(f"{label}.telemetry must be an object")
        raw_response_sha256 = _sha256(
            value.get("raw_response_sha256"),
            f"{label}.raw_response_sha256",
            allow_null=True,
        )
        if response_status == "valid_response" and raw_response_sha256 is None:
            raise DataContractError(
                f"{label}.raw_response_sha256 is required for a valid response"
            )
        return cls(
            condition_id=condition,
            training_seed=seed,
            candidate_id=candidate_id,
            response_status=response_status,
            action=action,
            corrected=corrected,
            correction_validation_status=correction_status,
            raw_response_sha256=raw_response_sha256,
            prompt_sha256=_sha256(_require(value, "prompt_sha256", label), f"{label}.prompt_sha256") or "",
            model_manifest_sha256=_sha256(
                _require(value, "model_manifest_sha256", label),
                f"{label}.model_manifest_sha256",
            ) or "",
            decoding_sha256=_sha256(
                _require(value, "decoding_sha256", label), f"{label}.decoding_sha256"
            ) or "",
            attempts=_integer(_require(value, "attempts", label), f"{label}.attempts", minimum=1),
            error_category=error_category,
            telemetry=telemetry,
        )
