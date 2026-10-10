"""verifier interpretation responsibilities."""
from __future__ import annotations

from utils.verifier.requests import PreparedSentence
import json
from typing import Any
from utils.common.constants import ENTITY_TYPES, RELATION_TYPES
from utils.common.records import Candidate, EntitySpan, StrictTriple

REASON_CODES = {
    "SUPPORTED",
    "WRONG_RELATION",
    "WRONG_HEAD",
    "WRONG_TAIL",
    "UNSUPPORTED",
    "AMBIGUOUS",
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
