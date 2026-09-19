"""Verifier-stage controller for development gating and final inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.common.artifact_io import DataContractError, atomic_write_json
from utils.common.config import PipelineConfig
from utils.common.paths import RunLayout
from utils.evaluation.publication import prepare_verifier_pilot
from utils.verifier.pilot import PilotInputs, run_verifier_pilot
from utils.verifier.threshold import select_threshold


@dataclass(frozen=True)
class VerifierContext:
    development: Path
    warmup_candidates: Path
    model_blob: Path


def prepare_development_gate(
    layout: RunLayout,
    config: PipelineConfig,
    development_candidates: Path,
    *,
    validate_threshold: Callable[[RunLayout, PipelineConfig], object],
    validate_pilot_inputs: Callable[[RunLayout, PipelineConfig], object],
) -> tuple[Path, Path]:
    """Select the development threshold and materialize pilot inputs."""

    threshold_path = layout.resolve("predictions/dev/threshold-selection.json")
    if not threshold_path.is_file():
        select_threshold(
            layout,
            config,
            candidates_path=development_candidates,
            gold_path=layout.resolve(
                "data-prepared/development-gold.jsonl", must_exist=True
            ),
            split_manifest_path=layout.resolve(
                "data-prepared/split-manifest.json", must_exist=True
            ),
            candidate_index_path=layout.resolve(
                "predictions/dev/candidate-index.json", must_exist=True
            ),
            out_path=threshold_path,
        )
    validate_threshold(layout, config)

    pilot_selection = layout.resolve("predictions/dev/pilot-selection.json")
    if not pilot_selection.is_file():
        prepare_verifier_pilot(layout, config)
    validate_pilot_inputs(layout, config)
    return threshold_path, pilot_selection


def run_pilot(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    threshold_path: Path,
    pilot_selection: Path,
    ollama_url: str,
    model_blob_source: str | None,
    encoder_identity: dict[str, Any],
    training_hardware: dict[str, object],
    discover_live_model: Callable[..., tuple[Path, dict[str, object]]],
    materialize_model_blob: Callable[..., tuple[str, Path]],
    write_full_run_manifest: Callable[..., None],
    resume_verifier_stage: Callable[..., object],
    load_manifest: Callable[[Path, str], dict],
    validate_pilot_capture_seals: Callable[[RunLayout, Path], object],
    validate_pilot_audit: Callable[[RunLayout, PipelineConfig], object],
) -> VerifierContext:
    """Authenticate the live model and run the development-only verifier gate."""

    source, model_identity = discover_live_model(
        config,
        layout.source_root,
        ollama_url=ollama_url,
        model_blob_source=model_blob_source,
    )
    model_relative, model_blob = materialize_model_blob(
        layout, source, str(model_identity["blob_sha256"])
    )
    write_full_run_manifest(
        layout,
        config,
        model_relative,
        model_identity,
        encoder_identity,
        training_hardware,
    )

    pilot_candidates = layout.resolve(
        "predictions/dev/pilot-candidates.jsonl", must_exist=True
    )
    warmup_candidates = layout.resolve(
        "predictions/dev/verifier-warmup-candidate.jsonl", must_exist=True
    )
    development = layout.resolve("data-prepared/development.jsonl", must_exist=True)
    capture_rows = []
    for mode in ("simple", "corrective"):
        for repeat in (1, 2):
            capture_id = f"{mode}-repeat-{repeat}"
            prefix = f"pilot/{capture_id}"
            resume_verifier_stage(
                layout,
                config,
                mode=mode,
                sentences=development,
                candidates=pilot_candidates,
                development=development,
                warmup_candidates=warmup_candidates,
                model_blob=model_blob,
                ollama_url=ollama_url,
                artifact_prefix=prefix,
                pilot_selection=pilot_selection,
                allow_response_cache=False,
            )
            capture_rows.append(
                {
                    "capture_id": capture_id,
                    "mode": mode,
                    "repeat": repeat,
                    "artifact_prefix": prefix,
                }
            )

    capture_index = layout.resolve("predictions/dev/pilot-captures.json")
    capture_document = {
        "schema_version": "phase-b-verifier-pilot-captures-2.0",
        "protocol_id": config.value["protocol_id"],
        "workflow_id": config.value["workflow_id"],
        "run_id": layout.run_id,
        "captures": capture_rows,
    }
    if capture_index.exists():
        if load_manifest(capture_index, "pilot capture index") != capture_document:
            raise DataContractError("Existing pilot capture index differs from this run")
    else:
        atomic_write_json(capture_index, capture_document)
    validate_pilot_capture_seals(layout, capture_index)

    audit_path = layout.resolve("audit/verifier-pilot/pilot-audit.json")
    if not audit_path.is_file():
        audit = run_verifier_pilot(
            layout,
            config,
            PilotInputs(
                sentences=development,
                gold=layout.resolve(
                    "data-prepared/development-gold.jsonl", must_exist=True
                ),
                candidates=pilot_candidates,
                warmup_candidates=warmup_candidates,
                split_manifest=layout.resolve(
                    "data-prepared/split-manifest.json", must_exist=True
                ),
                candidate_index=layout.resolve(
                    "predictions/dev/candidate-index.json", must_exist=True
                ),
                pilot_selection=pilot_selection,
                threshold_selection=threshold_path,
                capture_index=capture_index,
            ),
            evidence_class="development-pilot",
        )
        if (
            audit.get("pilot_status") != "pass"
            or audit.get("material_protocol_review_required") is not False
        ):
            raise DataContractError(
                "Development verifier pilot did not admit final-test execution"
            )
    validate_pilot_audit(layout, config)
    return VerifierContext(development, warmup_candidates, model_blob)


def run_test(
    layout: RunLayout,
    config: PipelineConfig,
    context: VerifierContext,
    test_candidates: Path,
    *,
    ollama_url: str,
    resume_verifier_stage: Callable[..., object],
) -> None:
    """Run the simple and corrective verifier over final-test candidates."""

    for mode in ("simple", "corrective"):
        resume_verifier_stage(
            layout,
            config,
            mode=mode,
            sentences=layout.resolve("data-prepared/test.jsonl", must_exist=True),
            candidates=test_candidates,
            development=context.development,
            warmup_candidates=context.warmup_candidates,
            model_blob=context.model_blob,
            ollama_url=ollama_url,
        )
