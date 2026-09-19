"""Final-test scoring controller for the canonical workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from utils.common.config import PipelineConfig
from utils.common.paths import RunLayout
from utils.evaluation.scoring import ScoreInputs, score_run


def run(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    test_candidates: Path,
    threshold_path: Path,
    validate_score: Callable[[RunLayout, PipelineConfig], object],
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
    validate_score(layout, config)
