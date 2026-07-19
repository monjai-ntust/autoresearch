"""Development-only confidence-threshold selection for VER-CONFIDENCE.

This stage is the canonical producer of the `threshold-selection.json` that the
`score` stage consumes (schema `threshold-selection.schema.json`). It was
previously required as an externally supplied file; it is now reproducible from
the eight-seed development candidate universe and development gold.

The objective and tie rule are frozen by the protocol
(`config.threshold_selection`): for every grid threshold it computes the
per-seed development strict Triple F1 (the same VER-CONFIDENCE end-to-end
matching the scorer applies -- keep candidates with confidence >= threshold,
collapse duplicate strict keys per seed/sentence, intersect the typed directed
set with gold), averages across the eight seeds, and selects the argmax with
ties broken toward the higher threshold. It never inspects the test labels.

The computation is fully deterministic and offline; only producing the eight-seed
development candidates upstream (live encoder inference) is externally gated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from config import PipelineConfig
from constants import PROTOCOL_ID, TRAINING_SEEDS
from paths import RunLayout
from phase_b_io import DataContractError, atomic_write_json, sha256_file
from scoring import _load_candidates, _load_gold

_DEV_SPLIT = "CODE-SPLIT-1:development"


def _strict_triple_f1(
    threshold: float,
    examples_for_seed: dict[str, list],
    gold,
) -> float:
    """Micro end-to-end strict Triple F1 for one seed at one threshold.

    Mirrors the scorer's VER-CONFIDENCE end-to-end computation: keep candidates
    with confidence >= threshold, collapse duplicate strict keys per sentence
    into a set, and intersect the typed directed set with gold.
    """

    tp = fp = fn = 0
    for example_id, gold_record in gold.items():
        gold_set = set(gold_record.triples)
        kept = {
            candidate.triple
            for candidate in examples_for_seed.get(example_id, [])
            if candidate.triple_confidence >= threshold
        }
        tp += len(kept & gold_set)
        fp += len(kept - gold_set)
        fn += len(gold_set - kept)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2 * tp) / denominator


def select_threshold(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    candidates_path: Path,
    gold_path: Path,
    split_manifest_path: Path,
    out_path: Path,
    candidate_index_path: Path | None = None,
) -> dict[str, Any]:
    """Compute and write the frozen development confidence-threshold selection."""

    layout.require_existing()
    if out_path.exists():
        raise DataContractError(
            "select-threshold refuses to overwrite existing artifact: "
            + layout.relative_identity(out_path)
        )

    gold = _load_gold(gold_path, _DEV_SPLIT)
    candidates, _ = _load_candidates(candidates_path, _DEV_SPLIT, gold)

    observed_seeds = sorted({candidate.training_seed for candidate in candidates})
    if observed_seeds != list(TRAINING_SEEDS):
        raise DataContractError(
            "development candidates must cover exactly seeds 42-49; observed "
            f"{observed_seeds}"
        )

    by_seed_example: dict[int, dict[str, list]] = {seed: {} for seed in TRAINING_SEEDS}
    for candidate in candidates:
        by_seed_example[candidate.training_seed].setdefault(
            candidate.triple.example_id, []
        ).append(candidate)

    grid = config.value["threshold_selection"]["grid"]
    per_threshold: list[dict[str, Any]] = []
    recomputed: list[tuple[float, float]] = []
    for threshold in grid:
        per_seed = {
            str(seed): _strict_triple_f1(threshold, by_seed_example[seed], gold)
            for seed in TRAINING_SEEDS
        }
        mean = math.fsum(per_seed.values()) / len(TRAINING_SEEDS)
        per_threshold.append(
            {
                "threshold": threshold,
                "per_seed_development_strict_triple_f1": per_seed,
                "mean_per_seed_development_strict_triple_f1": mean,
            }
        )
        recomputed.append((mean, threshold))

    selected = max(recomputed, key=lambda item: (item[0], item[1]))[1]

    document = {
        "protocol_id": PROTOCOL_ID,
        "selection_split": "development",
        "objective": "mean_per_seed_development_strict_triple_f1",
        "tie_rule": "higher_threshold",
        "grid": list(grid),
        "selected_threshold": selected,
        "used_test_labels": False,
        "development_candidate_index_sha256": sha256_file(
            candidate_index_path or candidates_path
        ),
        "development_gold_sha256": sha256_file(gold_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "per_threshold": per_threshold,
    }
    atomic_write_json(out_path, document)
    return document
