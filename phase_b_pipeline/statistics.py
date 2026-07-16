"""Predeclared paired statistical procedures for the Phase B protocol."""

from __future__ import annotations

import itertools
import math
import random
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

from .io import DataContractError
from .metrics import triple_metrics


def _finite_values(values: Sequence[float], label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise DataContractError(f"{label} must contain at least one finite value")
    return result


def exact_wilcoxon_signed_rank(deltas: Sequence[float]) -> dict[str, Any]:
    """Implement frozen ``WILCOX-EXACT-1`` without an asymptotic shortcut."""

    values = _finite_values(deltas, "Wilcoxon deltas")
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return {
            "method": "WILCOX-EXACT-1",
            "n_total": len(values),
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    ordered = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(nonzero)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and abs(ordered[end][1]) == abs(ordered[cursor][1]):
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average_rank
        cursor = end

    rank_total = sum(ranks)
    w_plus = sum(rank for rank, delta in zip(ranks, nonzero) if delta > 0)
    w_minus = rank_total - w_plus
    observed = min(w_plus, w_minus)
    extreme = 0
    total_assignments = 2 ** len(nonzero)
    for signs in itertools.product((False, True), repeat=len(nonzero)):
        enumerated_plus = sum(rank for rank, positive in zip(ranks, signs) if positive)
        enumerated_statistic = min(enumerated_plus, rank_total - enumerated_plus)
        if enumerated_statistic <= observed + 1e-12:
            extreme += 1
    return {
        "method": "WILCOX-EXACT-1",
        "n_total": len(values),
        "n_nonzero": len(nonzero),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": observed,
        "p_value": extreme / total_assignments,
    }


def _continued_beta_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + doubled) * (a + doubled)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + doubled) * (qap + doubled)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("regularized incomplete beta fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_beta_fraction(a, b, x) / a
    return 1.0 - front * _continued_beta_fraction(b, a, 1.0 - x) / b


def paired_t_test(deltas: Sequence[float]) -> dict[str, Any]:
    """Two-sided paired t-test on precomputed paired differences."""

    values = _finite_values(deltas, "paired t-test deltas")
    if len(values) < 2:
        raise DataContractError("paired t-test requires at least two paired deltas")
    mean = fmean(values)
    sample_sd = stdev(values)
    if sample_sd == 0.0:
        statistic = 0.0 if mean == 0.0 else None
        statistic_status = (
            "finite"
            if mean == 0.0
            else "positive_infinity" if mean > 0.0 else "negative_infinity"
        )
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        statistic = mean / (sample_sd / math.sqrt(len(values)))
        statistic_status = "finite"
        degrees = len(values) - 1
        x = degrees / (degrees + statistic * statistic)
        p_value = _regularized_incomplete_beta(degrees / 2.0, 0.5, x)
    return {
        "method": "paired_t_test_two_sided",
        "n": len(values),
        "mean_delta": mean,
        "sample_standard_deviation": sample_sd,
        "degrees_of_freedom": len(values) - 1,
        "statistic": statistic,
        "statistic_status": statistic_status,
        "p_value": p_value,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values for one declared family."""

    if not p_values:
        raise DataContractError("Holm adjustment requires at least one p-value")
    for label, value in p_values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise DataContractError(f"Holm p-value for {label!r} must be finite and in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(value)))
        adjusted[label] = running
    return {label: adjusted[label] for label in p_values}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise DataContractError("cannot compute a percentile of no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_hierarchical_triple_f1_bootstrap(
    counts: Mapping[str, Mapping[int, Mapping[str, Mapping[str, int]]]],
    *,
    reference_condition: str,
    replicates: int,
    seed: int,
    undefined_limit: float = 0.05,
) -> dict[str, Any]:
    """Resample seeds and document clusters with shared indices across conditions."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise DataContractError("bootstrap replicates must be a positive integer")
    if reference_condition not in counts:
        raise DataContractError("bootstrap reference condition is missing")
    conditions = list(counts)
    seeds = sorted(counts[reference_condition])
    if not seeds:
        raise DataContractError("bootstrap requires at least one training seed")
    documents = sorted(counts[reference_condition][seeds[0]])
    if not documents:
        raise DataContractError("bootstrap requires at least one source-document cluster")
    for condition in conditions:
        if sorted(counts[condition]) != seeds:
            raise DataContractError("bootstrap conditions have different seed identities")
        for training_seed in seeds:
            if sorted(counts[condition][training_seed]) != documents:
                raise DataContractError(
                    "bootstrap conditions/seeds have different document-cluster identities"
                )

    def condition_value(condition: str, sampled_seeds: list[int], sampled_docs: list[str]):
        per_seed: list[float] = []
        for training_seed in sampled_seeds:
            total = {"tp": 0, "fp": 0, "fn": 0}
            for document in sampled_docs:
                record = counts[condition][training_seed][document]
                for field in total:
                    amount = record.get(field)
                    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                        raise DataContractError("bootstrap counts must be nonnegative integers")
                    total[field] += amount
            value = triple_metrics(**total)["f1"]["value"]
            if value is None:
                return None
            per_seed.append(value)
        return fmean(per_seed)

    point: dict[str, float | None] = {
        condition: condition_value(condition, seeds, documents) for condition in conditions
    }
    generator = random.Random(seed)
    sampled_values: dict[str, list[float]] = {condition: [] for condition in conditions}
    undefined: dict[str, int] = {condition: 0 for condition in conditions}
    delta_values: dict[str, list[float]] = {
        condition: [] for condition in conditions if condition != reference_condition
    }
    delta_undefined: dict[str, int] = {condition: 0 for condition in delta_values}
    for _ in range(replicates):
        sampled_seeds = [seeds[generator.randrange(len(seeds))] for _ in seeds]
        sampled_docs = [documents[generator.randrange(len(documents))] for _ in documents]
        replicate_values = {
            condition: condition_value(condition, sampled_seeds, sampled_docs)
            for condition in conditions
        }
        for condition, value in replicate_values.items():
            if value is None:
                undefined[condition] += 1
            else:
                sampled_values[condition].append(value)
        reference_value = replicate_values[reference_condition]
        for condition in delta_values:
            value = replicate_values[condition]
            if value is None or reference_value is None:
                delta_undefined[condition] += 1
            else:
                delta_values[condition].append(value - reference_value)

    def interval(values: list[float], n_undefined: int) -> dict[str, Any]:
        if n_undefined / replicates > undefined_limit:
            return {
                "lower": None,
                "upper": None,
                "status": "withheld_excess_undefined_replicates",
                "undefined_replicates": n_undefined,
            }
        return {
            "lower": _percentile(values, 0.025),
            "upper": _percentile(values, 0.975),
            "status": "defined",
            "undefined_replicates": n_undefined,
        }

    return {
        "method": "paired_hierarchical_percentile",
        "metric": "end_to_end_strict_triple_f1",
        "replicates": replicates,
        "seed": seed,
        "seed_count": len(seeds),
        "document_cluster_count": len(documents),
        "conditions": {
            condition: {
                "point_estimate": point[condition],
                "interval_95": interval(sampled_values[condition], undefined[condition]),
            }
            for condition in conditions
        },
        "deltas_vs_reference": {
            condition: {
                "point_estimate": (
                    None
                    if point[condition] is None or point[reference_condition] is None
                    else point[condition] - point[reference_condition]
                ),
                "interval_95": interval(delta_values[condition], delta_undefined[condition]),
            }
            for condition in delta_values
        },
    }
