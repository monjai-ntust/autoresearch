"""evaluation tables responsibilities."""
from __future__ import annotations

import math
from typing import Any
from utils.common.constants import CONDITION_IDS
from utils.common.artifact_io import DataContractError

def publication_products(metrics: dict[str, Any]) -> tuple[str, str]:
    """Render deterministic, prose-neutral publication table products."""

    def value(mapping: Any, *keys: str) -> Any:
        current = mapping
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def number(item: Any) -> str:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return "NA"
        return f"{float(item):.6f}"

    statistics = metrics.get("statistics")
    bootstrap = value(statistics, "bootstrap") or {}
    bootstrap_conditions = value(bootstrap, "conditions") or {}
    bootstrap_deltas = value(bootstrap, "deltas_vs_reference") or {}
    wilcoxon = value(statistics, "paired_tests", "wilcoxon_primary") or {}
    observed_seeds = metrics.get("observed_training_seeds")
    seeds = observed_seeds if isinstance(observed_seeds, list) else []
    rows = [
        "condition\tcandidate_accuracy\ttriple_tp\ttriple_fp\ttriple_fn\t"
        "precision\trecall\tf1\tmean_seed_f1\tsd_seed_f1\t"
        "hierarchical_f1_ci95_lower\thierarchical_f1_ci95_upper\t"
        "delta_f1_vs_raw\tdelta_ci95_lower\tdelta_ci95_upper\t"
        "wilcoxon_holm_p"
    ]
    markdown_rows = [
        "| Condition | Accuracy | TP | FP | FN | Precision | Recall | F1 | "
        "Mean seed F1 ± SD | Hierarchical 95% CI | ΔF1 vs raw | Holm-adjusted Wilcoxon p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    conditions = metrics.get("conditions")
    if not isinstance(conditions, dict):
        raise DataContractError("metrics conditions must be an object")
    for condition_id in CONDITION_IDS:
        condition = conditions.get(condition_id)
        if not isinstance(condition, dict):
            raise DataContractError(f"metrics conditions lacks {condition_id}")
        triple = value(condition, "end_to_end_strict_triple") or {}
        counts = value(triple, "counts") or {}
        per_seed = condition.get("per_seed")
        seed_f1: list[float] = []
        if isinstance(per_seed, dict):
            for seed in seeds:
                metric = value(
                    per_seed,
                    str(seed),
                    "end_to_end_strict_triple",
                    "f1",
                    "value",
                )
                if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                    seed_f1.append(float(metric))
        mean_seed_f1 = sum(seed_f1) / len(seed_f1) if seed_f1 else None
        sd_seed_f1 = None
        if len(seed_f1) > 1 and mean_seed_f1 is not None:
            sd_seed_f1 = math.sqrt(
                sum((metric - mean_seed_f1) ** 2 for metric in seed_f1)
                / (len(seed_f1) - 1)
            )
        interval = value(bootstrap_conditions, condition_id, "interval_95") or {}
        delta = bootstrap_deltas.get(condition_id, {})
        delta_interval = value(delta, "interval_95") or {}
        adjusted_p = value(wilcoxon, condition_id, "holm_adjusted_p_value")
        fields = (
            condition_id,
            number(value(condition, "candidate_decision", "accuracy", "value")),
            str(counts.get("tp", "NA")),
            str(counts.get("fp", "NA")),
            str(counts.get("fn", "NA")),
            number(value(triple, "precision", "value")),
            number(value(triple, "recall", "value")),
            number(value(triple, "f1", "value")),
            number(mean_seed_f1),
            number(sd_seed_f1),
            number(interval.get("lower")),
            number(interval.get("upper")),
            number(delta.get("point_estimate")),
            number(delta_interval.get("lower")),
            number(delta_interval.get("upper")),
            number(adjusted_p),
        )
        rows.append("\t".join(fields))
        mean_sd = (
            f"{number(mean_seed_f1)} ± {number(sd_seed_f1)}"
            if mean_seed_f1 is not None
            else "NA"
        )
        ci = (
            f"[{number(interval.get('lower'))}, {number(interval.get('upper'))}]"
            if interval
            else "NA"
        )
        markdown_rows.append(
            "| "
            + " | ".join(
                (
                    condition_id,
                    fields[1],
                    fields[2],
                    fields[3],
                    fields[4],
                    fields[5],
                    fields[6],
                    fields[7],
                    mean_sd,
                    ci,
                    fields[12],
                    fields[15],
                )
            )
            + " |"
        )
    tsv = "\n".join(rows) + "\n"
    summary = "\n".join(
        (
            "# Phase B verifier metrics",
            "",
            f"- Protocol: `{metrics.get('protocol_id')}`",
            f"- Workflow: `{metrics.get('workflow_id')}`",
            f"- Matcher: `{metrics.get('matcher_id')}`",
            f"- Evaluation split: `{metrics.get('evaluation_split_id')}`",
            f"- Seeds: `{','.join(str(seed) for seed in seeds)}`",
            f"- Selected confidence threshold: `{number(metrics.get('selected_confidence_threshold'))}`",
            f"- Publication seed coverage complete: `{str(metrics.get('publication_seed_coverage_complete')).lower()}`",
            f"- Publication data coverage complete: `{str(metrics.get('publication_data_coverage_complete')).lower()}`",
            f"- Uncertainty status: `{metrics.get('uncertainty_status')}`",
            "",
            *markdown_rows,
            "",
            "Accuracy is candidate-decision accuracy. TP/FP/FN and Precision/Recall/F1 are end-to-end CODE-STRICT-1 triple metrics. Confidence, simple, and corrective deltas use VER-RAW as the paired reference. `NA` means the required coverage or statistic is unavailable.",
            "",
        )
    )
    return tsv, summary
