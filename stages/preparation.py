"""Preparation-stage controller for the canonical workflow."""

from __future__ import annotations

from collections.abc import Callable

from utils.common.artifact_io import DataContractError
from utils.common.config import PipelineConfig
from utils.common.environment import run_doctor
from utils.common.paths import RunLayout
from utils.evaluation.reconciliation import reconcile_section5_evidence
from utils.preparation.acquisition import fetch_run
from utils.preparation.preparation import prepare_run


def run(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    validate_checkout: Callable[[RunLayout, PipelineConfig], object],
    validate_reconciliation: Callable[[RunLayout, PipelineConfig], object],
) -> None:
    """Validate the checkout, reconcile evidence, acquire inputs, and prepare data."""

    doctor_path = layout.resolve("manifests/00-checkout-manifest.json")
    if doctor_path.is_file():
        validate_checkout(layout, config)
    else:
        manifest, passed = run_doctor(layout, config)
        if not passed:
            raise DataContractError(f"Checkout doctor failed: {manifest}")

    reconciliation_path = layout.resolve("audit/section5-evidence-reconciliation.json")
    if reconciliation_path.is_file():
        validate_reconciliation(layout, config)
    else:
        reconcile_section5_evidence(layout, config)
        validate_reconciliation(layout, config)

    # Both producers authenticate existing artifacts on resume.
    fetch_run(layout, config)
    prepare_run(layout, config)
