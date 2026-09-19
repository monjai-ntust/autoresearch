"""Final-test scoring controller for the canonical workflow."""

from __future__ import annotations

from pathlib import Path

from utils.common.artifact_io import (
    DataContractError,
    _load_manifest,
    _require_fields,
    sha256_file,
)
from utils.common.config import PipelineConfig
from utils.common.paths import RunLayout
from utils.evaluation.scoring import ScoreInputs, score_run


def run(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    test_candidates: Path,
    threshold_path: Path,
) -> None:
    """Score the four canonical final-test conditions and validate outputs."""

    score_manifest = layout.resolve("manifests/score-manifest.json")
    if not score_manifest.is_file():
        score_run(
            layout,
            config,
            ScoreInputs(
                gold=layout.resolve("data-prepared/test-gold.jsonl", must_exist=True),
                candidates=test_candidates,
                simple_verdicts=layout.resolve(
                    "verifier/simple/verdicts.jsonl", must_exist=True
                ),
                corrective_verdicts=layout.resolve(
                    "verifier/corrective/verdicts.jsonl", must_exist=True
                ),
                threshold_selection=threshold_path,
                nonpublication_smoke=False,
                output_prefix="",
            ),
        )
    _validate_score(layout, config)


def _validate_score(layout: RunLayout, config) -> dict:
    path = layout.resolve("manifests/score-manifest.json", must_exist=True)
    manifest = _load_manifest(path, "score manifest")
    _require_fields(
        manifest,
        {
            "schema_version": "phase-b-score-manifest-1.0",
            "run_id": layout.run_id,
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "matcher_id": config.value["matcher_id"],
            "nonpublication_smoke": False,
        },
        "score manifest",
    )
    for mapping_name in ("inputs", "outputs"):
        mapping = manifest.get(mapping_name)
        if not isinstance(mapping, dict):
            raise DataContractError(f"score manifest {mapping_name} are malformed")
        for relative, expected in mapping.items():
            if sha256_file(layout.resolve(relative, must_exist=True)) != expected:
                raise DataContractError(f"score artifact changed: {relative}")
    return manifest
