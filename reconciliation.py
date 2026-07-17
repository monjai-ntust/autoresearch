"""Fail-closed reconciliation of frozen, secondary Section 5 ledgers."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from config import PipelineConfig
from constants import PROTOCOL_ID, WORKFLOW_ID
from phase_b_io import DataContractError, atomic_write_json, load_json, sha256_file
from paths import RunLayout, resolve_tracked_path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _newline_style(text: str) -> str:
    crlf_count = text.count("\r\n")
    without_crlf = text.replace("\r\n", "")
    lone_cr_count = without_crlf.count("\r")
    lone_lf_count = without_crlf.count("\n")
    if lone_cr_count or (crlf_count and lone_lf_count):
        return "mixed"
    if crlf_count:
        return "crlf"
    return "lf"


def _normalized_lines(raw: bytes, label: str) -> tuple[str, bytes, list[str], str, bool]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DataContractError(f"{label}: ledger is not valid UTF-8: {exc}") from exc
    style = _newline_style(text)
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized_text.encode("utf-8")
    ends_with_newline = normalized_text.endswith("\n")
    lines = normalized_text.split("\n")
    if ends_with_newline:
        lines.pop()
    return normalized_text, normalized, lines, style, ends_with_newline


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise DataContractError(f"{label}: expected {expected!r}, got {actual!r}")


def audit_ledger(source_root: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate one ledger's portable LF identity and known physical structure."""

    path_value = contract.get("path")
    if not isinstance(path_value, str):
        raise DataContractError("ledger contract path must be a string")
    path = resolve_tracked_path(source_root, path_value)
    raw = path.read_bytes()
    _, normalized, lines, style, ends_with_newline = _normalized_lines(
        raw, path_value
    )
    allowed_styles = contract.get("allowed_newline_styles")
    if not isinstance(allowed_styles, list) or style not in allowed_styles:
        raise DataContractError(
            f"{path_value}: newline style {style!r} is not one of {allowed_styles!r}"
        )
    _require_equal(
        f"{path_value} normalized LF bytes",
        len(normalized),
        contract.get("normalized_lf_bytes"),
    )
    _require_equal(
        f"{path_value} normalized LF SHA-256",
        _sha256_bytes(normalized),
        contract.get("normalized_lf_sha256"),
    )
    _require_equal(f"{path_value} logical line count", len(lines), contract.get("line_count"))
    _require_equal(
        f"{path_value} final newline",
        ends_with_newline,
        contract.get("ends_with_newline"),
    )

    blank_lines = [index for index, line in enumerate(lines, start=1) if not line]
    field_counts = Counter(len(line.split("\t")) for line in lines)
    observed_histogram = {
        str(key): field_counts[key] for key in sorted(field_counts)
    }
    structure = contract.get("expected_structure")
    if not isinstance(structure, dict):
        raise DataContractError(f"{path_value}: expected_structure must be an object")
    _require_equal(
        f"{path_value} blank physical lines",
        blank_lines,
        structure.get("blank_lines"),
    )
    _require_equal(
        f"{path_value} field-count histogram",
        observed_histogram,
        structure.get("field_count_histogram"),
    )
    concatenated = structure.get("concatenated_record_lines")
    if not isinstance(concatenated, list):
        raise DataContractError(
            f"{path_value}: concatenated_record_lines must be an array"
        )
    nominal_fields = contract.get("nominal_field_count")
    for line_number in concatenated:
        if (
            not isinstance(line_number, int)
            or line_number < 1
            or line_number > len(lines)
            or len(lines[line_number - 1].split("\t")) != nominal_fields * 2 - 1
        ):
            raise DataContractError(
                f"{path_value}: physical line {line_number!r} is not the frozen "
                "two-record concatenation"
            )

    audit = {
        "path": path_value,
        "raw_checkout_bytes": len(raw),
        "raw_checkout_sha256": _sha256_bytes(raw),
        "newline_style": style,
        "normalized_lf_bytes": len(normalized),
        "normalized_lf_sha256": _sha256_bytes(normalized),
        "line_count": len(lines),
        "ends_with_newline": ends_with_newline,
        "has_header": contract.get("has_header"),
        "nominal_field_count": nominal_fields,
        "blank_lines": blank_lines,
        "field_count_histogram": observed_histogram,
        "concatenated_record_lines": concatenated,
        "known_structure_matches_contract": True,
        "structural_status": contract.get("structural_status"),
    }
    return audit, lines


def _validate_claims(
    claims: Any, ledger_lines: dict[str, list[str]], ledger_text: dict[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(claims, list) or not claims:
        raise DataContractError("Section 5 evidence register must contain claim families")
    output: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise DataContractError(f"claim_families[{index}] must be an object")
        claim_id = claim.get("claim_id")
        if claim.get("canonical_eligible") is not False:
            raise DataContractError(f"{claim_id}: historical evidence cannot be canonical")
        references = claim.get("ledger_references")
        if not isinstance(references, list):
            raise DataContractError(f"{claim_id}: ledger_references must be an array")
        for reference in references:
            if not isinstance(reference, dict):
                raise DataContractError(f"{claim_id}: ledger reference must be an object")
            ledger_path = reference.get("path")
            line_number = reference.get("line")
            anchors = reference.get("contains_all")
            lines = ledger_lines.get(ledger_path)
            if (
                lines is None
                or not isinstance(line_number, int)
                or not 1 <= line_number <= len(lines)
                or not isinstance(anchors, list)
                or not all(isinstance(anchor, str) and anchor for anchor in anchors)
            ):
                raise DataContractError(f"{claim_id}: invalid ledger reference")
            line = lines[line_number - 1]
            missing = [anchor for anchor in anchors if anchor not in line]
            if missing:
                raise DataContractError(
                    f"{claim_id}: {ledger_path}:{line_number} lacks anchors {missing!r}"
                )
        absence_queries = claim.get("absence_queries", [])
        if not isinstance(absence_queries, list):
            raise DataContractError(f"{claim_id}: absence_queries must be an array")
        corpus = "\n".join(ledger_text.values())
        present = [query for query in absence_queries if query in corpus]
        if present:
            raise DataContractError(
                f"{claim_id}: frozen-ledger absence assertion failed for {present!r}"
            )
        output.append(
            {
                "claim_id": claim_id,
                "draft_scope": claim.get("draft_scope"),
                "support_status": claim.get("support_status"),
                "canonical_eligible": False,
                "ledger_references": references,
                "absence_queries_verified": absence_queries,
                "limitations": claim.get("limitations"),
                "canonical_next_action": claim.get("canonical_next_action"),
            }
        )
    return output


def reconcile_section5_evidence(layout: RunLayout, config: PipelineConfig) -> dict[str, Any]:
    """Write one immutable audit without promoting historical results."""

    layout.require_existing()
    checkout_path = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    checkout = load_json(checkout_path)
    if not isinstance(checkout, dict) or checkout.get("status") != "pass":
        raise DataContractError("reconcile requires a passing doctor checkout manifest")
    output_path = layout.resolve("audit/section5-evidence-reconciliation.json")
    if output_path.exists():
        raise DataContractError(
            "reconcile refuses to overwrite audit/section5-evidence-reconciliation.json"
        )

    register = config.section5_evidence
    contracts = register.get("ledgers")
    if not isinstance(contracts, list) or not contracts:
        raise DataContractError("Section 5 evidence register ledgers must be a nonempty array")
    ledger_audits: list[dict[str, Any]] = []
    ledger_lines: dict[str, list[str]] = {}
    ledger_text: dict[str, str] = {}
    for contract in contracts:
        if not isinstance(contract, dict):
            raise DataContractError("each Section 5 ledger contract must be an object")
        audit, lines = audit_ledger(layout.source_root, contract)
        path = audit["path"]
        if path in ledger_lines:
            raise DataContractError(f"duplicate Section 5 ledger contract: {path}")
        ledger_audits.append(audit)
        ledger_lines[path] = lines
        ledger_text[path] = "\n".join(lines)

    claims = _validate_claims(register.get("claim_families"), ledger_lines, ledger_text)
    status_counts = Counter(claim["support_status"] for claim in claims)
    audit = {
        "schema_version": "phase-b-section5-evidence-reconciliation-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "run_id": layout.run_id,
        "stage": "secondary_evidence_reconciliation",
        "status": "reconciled_secondary_evidence",
        "authority": "secondary_only",
        "canonical_publication_eligible": False,
        "publication_execution_admitted": False,
        "publication_execution_gate": "B-07-go-no-go-not-yet-approved",
        "register": {
            "path": config.section5_evidence_path.relative_to(layout.source_root).as_posix(),
            "sha256": sha256_file(config.section5_evidence_path),
            "revision": register.get("register_revision"),
        },
        "ledgers": ledger_audits,
        "claim_families": claims,
        "summary": {
            "claim_family_count": len(claims),
            "support_status_counts": {
                key: status_counts[key] for key in sorted(status_counts)
            },
            "canonical_ready_claim_family_count": 0,
            "historical_ledgers_are_secondary_only": True,
            "new_measurements_must_use_canonical_framework": True,
        },
        "hard_stops": [
            "historical-ledgers-cannot-be-promoted-to-canonical-evidence",
            "missing-or-incompatible-claim-families-require-new-or-separately-hashed-evidence",
        ],
    }
    atomic_write_json(output_path, audit)
    return audit
