"""Fail-closed reconciliation of archived, secondary Section 5 ledgers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from config import PipelineConfig
from constants import PROTOCOL_ID, WORKFLOW_ID
from phase_b_io import DataContractError, atomic_write_json, load_json, sha256_file
from paths import RunLayout


def _digest(value: Any, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataContractError(f"{label} must be a lowercase {length}-digit hex digest")
    return value


def _archived_ledgers(contracts: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate frozen archive facts without requiring an external archive."""

    if not isinstance(contracts, list) or not contracts:
        raise DataContractError("Section 5 evidence register ledgers must be a nonempty array")
    audits: list[dict[str, Any]] = []
    line_counts: dict[str, int] = {}
    for index, contract in enumerate(contracts):
        if not isinstance(contract, dict):
            raise DataContractError(f"ledgers[{index}] must be an object")
        path = contract.get("path")
        line_count = contract.get("line_count")
        structure = contract.get("expected_structure")
        if not isinstance(path, str) or not path:
            raise DataContractError(f"ledgers[{index}].path must be a nonempty string")
        if path in line_counts:
            raise DataContractError(f"duplicate Section 5 ledger contract: {path}")
        if contract.get("archival_status") != "hash_verified_external_archive":
            raise DataContractError(f"{path}: archival status is not verified")
        if not isinstance(contract.get("source_checkout_bytes"), int) or contract[
            "source_checkout_bytes"
        ] <= 0:
            raise DataContractError(f"{path}: invalid source checkout byte count")
        _digest(contract.get("source_checkout_sha256"), 64, f"{path} source SHA-256")
        _digest(contract.get("source_git_blob"), 40, f"{path} Git blob")
        _digest(contract.get("normalized_lf_sha256"), 64, f"{path} normalized SHA-256")
        if not isinstance(line_count, int) or line_count < 1:
            raise DataContractError(f"{path}: invalid logical line count")
        if not isinstance(structure, dict):
            raise DataContractError(f"{path}: expected_structure must be an object")
        blank_lines = structure.get("blank_lines")
        concatenated = structure.get("concatenated_record_lines")
        histogram = structure.get("field_count_histogram")
        if (
            not isinstance(blank_lines, list)
            or not all(isinstance(item, int) and 1 <= item <= line_count for item in blank_lines)
            or not isinstance(concatenated, list)
            or not all(isinstance(item, int) and 1 <= item <= line_count for item in concatenated)
            or not isinstance(histogram, dict)
            or not histogram
            or any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or value < 1
                for key, value in histogram.items()
            )
        ):
            raise DataContractError(f"{path}: invalid frozen physical-structure facts")
        audits.append(
            {
                **contract,
                "checkout_dependency": False,
                "evidence_location": "external_archive_not_required_at_runtime",
                "fact_verification": "hash_verified_before_source_removal",
            }
        )
        line_counts[path] = line_count
    return audits, line_counts


def _validate_claims(claims: Any, line_counts: dict[str, int]) -> list[dict[str, Any]]:
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
            if (
                ledger_path not in line_counts
                or not isinstance(line_number, int)
                or not 1 <= line_number <= line_counts[ledger_path]
                or not isinstance(anchors, list)
                or not all(isinstance(anchor, str) and anchor for anchor in anchors)
            ):
                raise DataContractError(f"{claim_id}: invalid archived-ledger reference")
        absence_queries = claim.get("absence_queries", [])
        if not isinstance(absence_queries, list) or not all(
            isinstance(query, str) and query for query in absence_queries
        ):
            raise DataContractError(f"{claim_id}: absence_queries must be an array")
        output.append(
            {
                "claim_id": claim_id,
                "draft_scope": claim.get("draft_scope"),
                "support_status": claim.get("support_status"),
                "canonical_eligible": False,
                "ledger_references": references,
                "reference_verification": "frozen_pre_archival_facts",
                "absence_queries_verified": absence_queries,
                "absence_verification": "frozen_pre_archival_facts",
                "limitations": claim.get("limitations"),
                "canonical_next_action": claim.get("canonical_next_action"),
            }
        )
    return output


def reconcile_section5_evidence(layout: RunLayout, config: PipelineConfig) -> dict[str, Any]:
    """Write a runtime-independent audit of the archived secondary evidence."""

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
    ledger_audits, line_counts = _archived_ledgers(register.get("ledgers"))
    claims = _validate_claims(register.get("claim_families"), line_counts)
    status_counts = Counter(claim["support_status"] for claim in claims)
    audit = {
        "schema_version": "phase-b-section5-evidence-reconciliation-2.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "run_id": layout.run_id,
        "stage": "secondary_evidence_reconciliation",
        "status": "reconciled_secondary_evidence",
        "authority": "secondary_only",
        "canonical_publication_eligible": False,
        "publication_execution_admitted": False,
        "publication_execution_gate": "mandatory-passing-b07-pilot-audit",
        "register": {
            "path": config.section5_evidence_path.relative_to(layout.source_root).as_posix(),
            "sha256": sha256_file(config.section5_evidence_path),
            "revision": register.get("register_revision"),
            "runtime_external_archive_dependency": False,
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
            "historical_ledger_source_files_removed": True,
            "archived_facts_available_without_external_archive": True,
            "new_measurements_must_use_canonical_framework": True,
        },
        "hard_stops": [
            "archived-historical-ledgers-cannot-be-promoted-to-canonical-evidence",
            "archived-ledger-facts-cannot-be-used-as-runtime-scientific-inputs",
            "missing-or-incompatible-claim-families-require-new-or-separately-hashed-evidence",
        ],
    }
    atomic_write_json(output_path, audit)
    return audit
