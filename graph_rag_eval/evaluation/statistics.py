"""Deterministic document-clustered paired bootstrap and comparisons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class PairedObservation:
    item_id: str
    cluster_id: str
    baseline: float
    treatment: float

    @property
    def delta(self) -> float:
        return self.treatment - self.baseline


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def clustered_paired_bootstrap(
    observations: Iterable[PairedObservation],
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    rows = tuple(observations)
    if not rows:
        return {
            "status": "not_applicable",
            "reason": "no paired observations",
            "questions": 0,
            "clusters": 0,
        }
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row.cluster_id].append(row.delta)
    cluster_ids = sorted(by_cluster)
    if len(cluster_ids) < 2:
        return {
            "status": "descriptive_only",
            "reason": "fewer than two independent clusters",
            "estimate": mean(row.delta for row in rows),
            "questions": len(rows),
            "clusters": len(cluster_ids),
        }
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        deltas = [delta for cluster_id in sampled for delta in by_cluster[cluster_id]]
        estimates.append(mean(deltas))
    alpha = (1.0 - confidence) / 2.0
    return {
        "status": "available",
        "estimate": mean(row.delta for row in rows),
        "confidence": confidence,
        "lower": percentile(estimates, alpha),
        "upper": percentile(estimates, 1.0 - alpha),
        "resamples": resamples,
        "seed": seed,
        "questions": len(rows),
        "clusters": len(cluster_ids),
    }


def mcnemar_counts(
    baseline: Iterable[bool],
    treatment: Iterable[bool],
) -> dict[str, int | str]:
    left = tuple(baseline)
    right = tuple(treatment)
    if len(left) != len(right):
        raise ValueError("paired outcomes have different lengths")
    b = sum(a and not c for a, c in zip(left, right))
    c = sum(not a and c for a, c in zip(left, right))
    return {
        "status": "descriptive_only" if b + c < 10 else "test_eligible",
        "baseline_only": b,
        "treatment_only": c,
        "discordant": b + c,
    }


def holm_adjust(p_values: Iterable[float]) -> tuple[float, ...]:
    values = tuple(p_values)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (count - rank)))
        adjusted[index] = running
    return tuple(adjusted)
