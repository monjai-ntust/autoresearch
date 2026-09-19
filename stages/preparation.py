"""Preparation-stage controller for the canonical workflow."""

from __future__ import annotations

import subprocess

from utils.common.artifact_io import (
    DataContractError,
    _load_manifest,
    _require_fields,
    sha256_file,
)
from utils.common.config import PipelineConfig
from utils.common.environment import run_doctor
from utils.common.paths import RunLayout
from utils.evaluation.reconciliation import reconcile_section5_evidence
from utils.preparation.acquisition import fetch_run
from utils.preparation.preparation import prepare_run


def run(
    layout: RunLayout,
    config: PipelineConfig,
) -> None:
    """Validate the checkout, reconcile evidence, acquire inputs, and prepare data."""

    doctor_path = layout.resolve("manifests/00-checkout-manifest.json")
    if doctor_path.is_file():
        _validate_existing_checkout(layout, config)
    else:
        manifest, passed = run_doctor(layout, config)
        if not passed:
            raise DataContractError(f"Checkout doctor failed: {manifest}")

    reconciliation_path = layout.resolve("audit/section5-evidence-reconciliation.json")
    if reconciliation_path.is_file():
        _validate_reconciliation(layout, config)
    else:
        reconcile_section5_evidence(layout, config)
        _validate_reconciliation(layout, config)

    # Both producers authenticate existing artifacts on resume.
    fetch_run(layout, config)
    prepare_run(layout, config)


def _validate_existing_checkout(layout: RunLayout, config) -> dict:
    """Authenticate an existing run namespace before any downstream work."""
    path = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    manifest = _load_manifest(path, "checkout manifest")
    if manifest.get("schema_version") not in {
        "phase-b-checkout-manifest-1.0",
        "phase-b-checkout-manifest-2.0",
    }:
        raise DataContractError("checkout manifest schema_version is unsupported")
    _require_fields(
        manifest,
        {
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "matcher_id": config.value["matcher_id"],
            "run_id": layout.run_id,
            "status": "pass",
        },
        "checkout manifest",
    )
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("worktree_clean") is not True:
        raise DataContractError("existing run was not admitted from a clean checkout")
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=layout.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=layout.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source.get("commit") != current or status:
        raise DataContractError(
            "existing run checkout differs from the current clean source commit"
        )
    config_relative = config.path.resolve().relative_to(layout.source_root).as_posix()
    source_config = source.get("config")
    tracked = manifest.get("tracked_artifact_sha256")
    recorded_config_hash = (
        source_config.get("sha256")
        if isinstance(source_config, dict)
        and source_config.get("path") == config_relative
        else tracked.get(config_relative) if isinstance(tracked, dict) else None
    )
    if recorded_config_hash != sha256_file(config.path):
        raise DataContractError(
            "existing run checkout does not authenticate its selected config"
        )
    return manifest


def _validate_reconciliation(layout: RunLayout, config) -> dict:
    path = layout.resolve("audit/section5-evidence-reconciliation.json", must_exist=True)
    audit = _load_manifest(path, "Section-5 evidence reconciliation")
    _require_fields(
        audit,
        {
            "run_id": layout.run_id,
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "status": "reconciled_secondary_evidence",
            "authority": "secondary_only",
            "canonical_publication_eligible": False,
        },
        "Section-5 evidence reconciliation",
    )
    register = audit.get("register")
    if (
        not isinstance(register, dict)
        or register.get("path")
        != config.section5_evidence_path.relative_to(layout.source_root).as_posix()
        or register.get("sha256") != sha256_file(config.section5_evidence_path)
    ):
        raise DataContractError("Section-5 reconciliation differs from its source register")
    return audit
