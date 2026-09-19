"""Hand-recomputable confusion-count metrics with explicit undefined states."""

from __future__ import annotations

from typing import Any


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"value": None, "status": "undefined_zero_denominator"}
    return {"value": numerator / denominator, "status": "defined"}


def binary_metrics(*, tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    """Compute finite-universe candidate metrics without inventing zero values."""

    for label, value in {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    total = tp + fp + fn + tn
    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total},
        "accuracy": _ratio(tp + tn, total),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def triple_metrics(*, tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Compute strict end-to-end micro metrics (there is no triple TN)."""

    for label, value in {"tp": tp, "fp": fp, "fn": fn}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
