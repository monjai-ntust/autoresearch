"""Offline scoring of the four frozen Path A verifier conditions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from config import PipelineConfig
from constants import CONDITION_IDS, MATCHER_ID, PROTOCOL_ID, WORKFLOW_ID
from phase_b_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    iter_jsonl,
    load_json,
    sha256_file,
)
from metrics import binary_metrics, triple_metrics
from paths import RunLayout
from records import Candidate, GoldRecord, StrictTriple, Verdict
from phase_b_statistics import (
    exact_wilcoxon_signed_rank,
    holm_adjust,
    paired_hierarchical_triple_f1_bootstrap,
    paired_t_test,
)
from verifier import verifier_identity


@dataclass(frozen=True)
class ScoreInputs:
    gold: Path
    candidates: Path
    simple_verdicts: Path
    corrective_verdicts: Path
    threshold_selection: Path
    nonpublication_smoke: bool = False
    output_prefix: str = ""


def _load_gold(path: Path, expected_split: str) -> dict[str, GoldRecord]:
    records: dict[str, GoldRecord] = {}
    observed_order: list[str] = []
    for line_number, value in iter_jsonl(path):
        record = GoldRecord.from_mapping(value, f"{path.name}:{line_number}")
        if record.split_id != expected_split:
            raise DataContractError(
                f"{path.name}:{line_number}: split mismatch; expected {expected_split!r}"
            )
        if record.example_id in records:
            raise DataContractError(f"{path.name}:{line_number}: duplicate example_id")
        records[record.example_id] = record
        observed_order.append(record.example_id)
    if observed_order != sorted(observed_order):
        raise DataContractError(f"{path.name}: gold records must be sorted by example_id")
    if not records:
        raise DataContractError(f"{path.name}: gold input is empty")
    return records


def _load_candidates(
    path: Path, expected_split: str, gold: dict[str, GoldRecord]
) -> tuple[list[Candidate], dict[str, Candidate]]:
    candidates: list[Candidate] = []
    by_id: dict[str, Candidate] = {}
    for line_number, value in iter_jsonl(path):
        candidate = Candidate.from_mapping(value, f"{path.name}:{line_number}")
        if candidate.split_id != expected_split:
            raise DataContractError(
                f"{path.name}:{line_number}: split mismatch; expected {expected_split!r}"
            )
        gold_record = gold.get(candidate.triple.example_id)
        if gold_record is None:
            raise DataContractError(
                f"{path.name}:{line_number}: candidate example has no gold record"
            )
        if candidate.source_document_id != gold_record.source_document_id:
            raise DataContractError(
                f"{path.name}:{line_number}: source_document_id differs from gold"
            )
        if candidate.candidate_id in by_id:
            raise DataContractError(f"{path.name}:{line_number}: duplicate candidate_id")
        candidates.append(candidate)
        by_id[candidate.candidate_id] = candidate
    if not candidates:
        raise DataContractError(f"{path.name}: candidate input is empty")
    if [item.sort_key() for item in candidates] != sorted(item.sort_key() for item in candidates):
        raise DataContractError(
            f"{path.name}: candidates must be sorted by training_seed, example_id, candidate_id"
        )
    return candidates, by_id


def _load_verdicts(
    path: Path,
    condition_id: str,
    candidates: dict[str, Candidate],
    expected_identity: tuple[str, str, str],
) -> dict[str, Verdict]:
    records: dict[str, Verdict] = {}
    observed_order: list[tuple[int, str]] = []
    for line_number, value in iter_jsonl(path):
        candidate_id = value.get("candidate_id")
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise DataContractError(
                f"{path.name}:{line_number}: verdict refers to an unknown candidate"
            )
        verdict = Verdict.from_mapping(value, f"{path.name}:{line_number}", candidate)
        if verdict.condition_id != condition_id:
            raise DataContractError(
                f"{path.name}:{line_number}: expected condition {condition_id}"
            )
        if verdict.candidate_id in records:
            raise DataContractError(f"{path.name}:{line_number}: duplicate verdict")
        records[verdict.candidate_id] = verdict
        observed_order.append((verdict.training_seed, verdict.candidate_id))
    if observed_order != sorted(observed_order):
        raise DataContractError(
            f"{path.name}: verdicts must be sorted by training_seed, candidate_id"
        )
    identities = {
        (item.prompt_sha256, item.model_manifest_sha256, item.decoding_sha256)
        for item in records.values()
    }
    if len(identities) > 1:
        raise DataContractError(
            f"{path.name}: prompt/model/decoding identities differ within one condition"
        )
    if identities and identities != {expected_identity}:
        raise DataContractError(
            f"{path.name}: prompt/model/decoding identity differs from the frozen config"
        )
    return records


def _load_threshold(path: Path, config: PipelineConfig) -> float:
    value = load_json(path)
    if not isinstance(value, dict):
        raise DataContractError(f"{path.name}: threshold selection must be an object")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "selection_split": "development",
        "objective": "mean_per_seed_development_strict_triple_f1",
        "tie_rule": "higher_threshold",
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise DataContractError(
                f"{path.name}: {field} must be {expected_value!r}"
            )
    configured_grid = config.value["threshold_selection"]["grid"]
    if value.get("grid") != configured_grid:
        raise DataContractError(f"{path.name}: threshold grid differs from frozen config")
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    for field in (
        "development_candidate_index_sha256",
        "development_gold_sha256",
        "split_manifest_sha256",
    ):
        digest = value.get(field)
        if not isinstance(digest, str) or not sha_pattern.fullmatch(digest):
            raise DataContractError(f"{path.name}: {field} must be a lowercase SHA-256")
    trace = value.get("per_threshold")
    if not isinstance(trace, list) or len(trace) != len(configured_grid):
        raise DataContractError(
            f"{path.name}: per_threshold must contain one row for every frozen grid value"
        )
    expected_seed_keys = [str(seed) for seed in config.value["training_seeds"]]
    recomputed: list[tuple[float, float]] = []
    for index, (row, expected_threshold) in enumerate(zip(trace, configured_grid)):
        if not isinstance(row, dict) or row.get("threshold") != expected_threshold:
            raise DataContractError(
                f"{path.name}: per_threshold[{index}] is not ordered on the frozen grid"
            )
        per_seed = row.get("per_seed_development_strict_triple_f1")
        if not isinstance(per_seed, dict) or list(per_seed) != expected_seed_keys:
            raise DataContractError(
                f"{path.name}: per_threshold[{index}] must contain ordered seeds 42-49"
            )
        scores = list(per_seed.values())
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            for score in scores
        ):
            raise DataContractError(
                f"{path.name}: per_threshold[{index}] has an invalid development F1"
            )
        mean = math.fsum(float(score) for score in scores) / len(scores)
        reported_mean = row.get("mean_per_seed_development_strict_triple_f1")
        if (
            isinstance(reported_mean, bool)
            or not isinstance(reported_mean, (int, float))
            or not math.isclose(float(reported_mean), mean, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise DataContractError(
                f"{path.name}: per_threshold[{index}] mean does not recompute"
            )
        recomputed.append((mean, expected_threshold))
    expected_selected = max(recomputed, key=lambda item: (item[0], item[1]))[1]
    threshold = value.get("selected_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise DataContractError(f"{path.name}: selected_threshold must be numeric")
    threshold = float(threshold)
    if threshold not in configured_grid:
        raise DataContractError(f"{path.name}: selected_threshold is outside the frozen grid")
    if threshold != expected_selected:
        raise DataContractError(
            f"{path.name}: selected_threshold violates the frozen objective/tie rule; "
            f"expected {expected_selected}"
        )
    if value.get("used_test_labels") is not False:
        raise DataContractError(f"{path.name}: threshold selection must attest no test-label use")
    return threshold


def _condition_result(
    condition_id: str,
    candidate: Candidate,
    threshold: float,
    simple: dict[str, Verdict],
    corrective: dict[str, Verdict],
) -> tuple[bool, StrictTriple | None, str | None, str, str | None]:
    if condition_id == "VER-RAW":
        return True, candidate.triple, None, "not_applicable", None
    if condition_id == "VER-CONFIDENCE":
        keep = candidate.triple_confidence >= threshold
        return keep, candidate.triple if keep else None, None, "not_applicable", None

    verdicts = simple if condition_id == "VER-SIMPLE" else corrective
    verdict = verdicts.get(candidate.candidate_id)
    if verdict is None:
        return False, None, None, "not_applicable", "missing_verdict"
    if verdict.response_status != "valid_response":
        return (
            False,
            None,
            None,
            verdict.correction_validation_status,
            verdict.error_category or verdict.response_status,
        )
    if verdict.action == "KEEP":
        return True, candidate.triple, verdict.action, "not_applicable", None
    if verdict.action == "CORRECT":
        emitted = verdict.corrected if verdict.correction_validation_status == "valid" else None
        return False, emitted, verdict.action, verdict.correction_validation_status, None
    return False, None, verdict.action, "not_applicable", None


def score_run(layout: RunLayout, config: PipelineConfig, inputs: ScoreInputs) -> dict[str, Any]:
    """Validate all inputs, score all four conditions, then write atomic outputs."""

    layout.require_existing()
    prefix = inputs.output_prefix.strip("/")
    output_root = f"{prefix}/" if prefix else ""
    planned_outputs = (
        f"{output_root}outcomes/candidate-outcomes.jsonl",
        f"{output_root}outcomes/sentence-outcomes.jsonl",
        f"{output_root}metrics/metrics.json",
        f"{output_root}manifests/score-manifest.json",
    )
    existing = [relative for relative in planned_outputs if layout.resolve(relative).exists()]
    if existing:
        raise DataContractError(
            "score refuses to overwrite existing artifacts: " + ", ".join(existing)
        )
    expected_split = config.value["evaluation_split_id"]
    gold = _load_gold(inputs.gold, expected_split)
    candidates, candidates_by_id = _load_candidates(inputs.candidates, expected_split, gold)
    simple = _load_verdicts(
        inputs.simple_verdicts,
        "VER-SIMPLE",
        candidates_by_id,
        verifier_identity(config, "simple"),
    )
    corrective = _load_verdicts(
        inputs.corrective_verdicts,
        "VER-CORRECTIVE",
        candidates_by_id,
        verifier_identity(config, "corrective"),
    )
    threshold = _load_threshold(inputs.threshold_selection, config)

    candidate_outcomes: list[dict[str, Any]] = []
    emitted: dict[tuple[str, int, str], dict[StrictTriple, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    candidate_counts: dict[str, dict[str, int]] = {
        condition: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for condition in CONDITION_IDS
    }
    per_seed_candidate_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    diagnostics: dict[str, dict[str, Any]] = {
        condition: {
            "actions": defaultdict(int),
            "errors": defaultdict(int),
            "corrections_attempted": 0,
            "corrections_source_valid": 0,
            "corrections_gold_valid": 0,
            "corrections_failed": 0,
            "original_invalid_to_final_valid": 0,
            "original_valid_to_final_invalid": 0,
            "exclusions": 0,
        }
        for condition in CONDITION_IDS
    }

    for candidate in candidates:
        gold_valid = candidate.triple in gold[candidate.triple.example_id].triples
        for condition_id in CONDITION_IDS:
            condition_verdict = (
                simple.get(candidate.candidate_id)
                if condition_id == "VER-SIMPLE"
                else corrective.get(candidate.candidate_id)
                if condition_id == "VER-CORRECTIVE"
                else None
            )
            (
                predicted_valid,
                final_triple,
                action,
                correction_status,
                error_category,
            ) = _condition_result(condition_id, candidate, threshold, simple, corrective)
            quadrant = (
                "tp" if predicted_valid and gold_valid else
                "fp" if predicted_valid else
                "fn" if gold_valid else
                "tn"
            )
            candidate_counts[condition_id][quadrant] += 1
            per_seed_candidate_counts[(condition_id, candidate.training_seed)][quadrant] += 1
            if action is not None:
                diagnostics[condition_id]["actions"][action] += 1
            if error_category is not None:
                diagnostics[condition_id]["errors"][error_category] += 1
            if condition_id == "VER-CORRECTIVE" and action == "CORRECT":
                diagnostics[condition_id]["corrections_attempted"] += 1
                if final_triple is not None:
                    diagnostics[condition_id]["corrections_source_valid"] += 1
                else:
                    diagnostics[condition_id]["corrections_failed"] += 1
            final_valid = final_triple is not None and final_triple in gold[
                candidate.triple.example_id
            ].triples
            if condition_id == "VER-CORRECTIVE" and not gold_valid and final_valid:
                diagnostics[condition_id]["original_invalid_to_final_valid"] += 1
            if condition_id == "VER-CORRECTIVE" and action == "CORRECT" and final_valid:
                diagnostics[condition_id]["corrections_gold_valid"] += 1
            if condition_id == "VER-CORRECTIVE" and gold_valid and not final_valid:
                diagnostics[condition_id]["original_valid_to_final_invalid"] += 1
            if final_triple is not None:
                emitted[(condition_id, candidate.training_seed, candidate.triple.example_id)][
                    final_triple
                ].append(candidate.candidate_id)
            candidate_outcomes.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "matcher_id": MATCHER_ID,
                    "condition_id": condition_id,
                    "training_seed": candidate.training_seed,
                    "example_id": candidate.triple.example_id,
                    "source_document_id": candidate.source_document_id,
                    "candidate_id": candidate.candidate_id,
                    "original_strict_key": list(candidate.triple.key()),
                    "gold_original_valid": gold_valid,
                    "candidate_predicted_valid": predicted_valid,
                    "action": action,
                    "reason_code": condition_verdict.reason_code if condition_verdict else None,
                    "correction_validation_status": correction_status,
                    "final_emitted_key": list(final_triple.key()) if final_triple else None,
                    "error_category": error_category,
                }
            )

    observed_seeds = sorted({candidate.training_seed for candidate in candidates})
    seed_examples = [
        (seed, example_id) for seed in observed_seeds for example_id in sorted(gold)
    ]
    sentence_outcomes: list[dict[str, Any]] = []
    triple_counts: dict[str, dict[str, int]] = {
        condition: {"tp": 0, "fp": 0, "fn": 0} for condition in CONDITION_IDS
    }
    per_seed_triple_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    per_document_triple_counts: dict[tuple[str, int, str], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    duplicate_counts: dict[str, int] = {condition: 0 for condition in CONDITION_IDS}

    for seed, example_id in seed_examples:
        gold_set = set(gold[example_id].triples)
        for condition_id in CONDITION_IDS:
            emitted_sources = emitted.get((condition_id, seed, example_id), {})
            predicted_set = set(emitted_sources)
            duplicates = sum(max(0, len(ids) - 1) for ids in emitted_sources.values())
            duplicate_counts[condition_id] += duplicates
            counts = {
                "tp": len(predicted_set & gold_set),
                "fp": len(predicted_set - gold_set),
                "fn": len(gold_set - predicted_set),
            }
            for field, amount in counts.items():
                triple_counts[condition_id][field] += amount
                per_seed_triple_counts[(condition_id, seed)][field] += amount
                per_document_triple_counts[
                    (condition_id, seed, gold[example_id].source_document_id)
                ][field] += amount
            sentence_outcomes.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "matcher_id": MATCHER_ID,
                    "condition_id": condition_id,
                    "training_seed": seed,
                    "example_id": example_id,
                    "source_document_id": gold[example_id].source_document_id,
                    "gold_strict_keys": [list(item.key()) for item in sorted(gold_set)],
                    "emitted_strict_keys": [list(item.key()) for item in sorted(predicted_set)],
                    "duplicate_emissions_collapsed": duplicates,
                    "counts": counts,
                }
            )

    conditions: dict[str, Any] = {}
    for condition_id in CONDITION_IDS:
        condition_diagnostics = diagnostics[condition_id]
        conditions[condition_id] = {
            "candidate_decision": binary_metrics(**candidate_counts[condition_id]),
            "end_to_end_strict_triple": triple_metrics(**triple_counts[condition_id]),
            "duplicate_emissions_collapsed": duplicate_counts[condition_id],
            "diagnostics": {
                **{
                    field: value
                    for field, value in condition_diagnostics.items()
                    if field not in {"actions", "errors"}
                },
                "actions": dict(sorted(condition_diagnostics["actions"].items())),
                "errors": dict(sorted(condition_diagnostics["errors"].items())),
            },
            "per_seed": {
                str(seed): {
                    "candidate_decision": binary_metrics(
                        **per_seed_candidate_counts[(condition_id, seed)]
                    ),
                    "end_to_end_strict_triple": triple_metrics(
                        **per_seed_triple_counts[(condition_id, seed)]
                    ),
                }
                for seed in observed_seeds
            },
        }

    seed_coverage_complete = (
        observed_seeds == config.value["training_seeds"]
        and not inputs.nonpublication_smoke
    )
    data_coverage_complete = len(gold) == config.value["split"]["test_sentences"]
    if seed_coverage_complete and data_coverage_complete:
        clustered_counts = {
            condition: {
                seed: {
                    document: per_document_triple_counts[(condition, seed, document)]
                    for document in sorted(
                        {record.source_document_id for record in gold.values()}
                    )
                }
                for seed in observed_seeds
            }
            for condition in CONDITION_IDS
        }
        bootstrap = paired_hierarchical_triple_f1_bootstrap(
            clustered_counts,
            reference_condition="VER-RAW",
            replicates=config.value["statistics"]["bootstrap_replicates"],
            seed=config.value["statistics"]["bootstrap_seed"],
            undefined_limit=config.value["statistics"]["undefined_replicate_limit"],
        )
        contrast_ids = [condition for condition in CONDITION_IDS if condition != "VER-RAW"]
        deltas: dict[str, list[float]] = {}
        tests_available = True
        for condition in contrast_ids:
            condition_deltas: list[float] = []
            for seed in observed_seeds:
                raw = conditions["VER-RAW"]["per_seed"][str(seed)][
                    "end_to_end_strict_triple"
                ]["f1"]["value"]
                compared = conditions[condition]["per_seed"][str(seed)][
                    "end_to_end_strict_triple"
                ]["f1"]["value"]
                if raw is None or compared is None:
                    tests_available = False
                    break
                condition_deltas.append(compared - raw)
            deltas[condition] = condition_deltas
        if tests_available:
            wilcoxon = {
                condition: exact_wilcoxon_signed_rank(deltas[condition])
                for condition in contrast_ids
            }
            t_tests = {
                condition: paired_t_test(deltas[condition]) for condition in contrast_ids
            }
            wilcoxon_adjusted = holm_adjust(
                {condition: result["p_value"] for condition, result in wilcoxon.items()}
            )
            t_adjusted = holm_adjust(
                {condition: result["p_value"] for condition, result in t_tests.items()}
            )
            for condition in contrast_ids:
                wilcoxon[condition]["holm_adjusted_p_value"] = wilcoxon_adjusted[condition]
                t_tests[condition]["holm_adjusted_p_value"] = t_adjusted[condition]
            tests = {
                "status": "defined",
                "per_seed_deltas": deltas,
                "wilcoxon_primary": wilcoxon,
                "paired_t_sensitivity": t_tests,
            }
        else:
            tests = {
                "status": "withheld_undefined_per_seed_f1",
                "per_seed_deltas": deltas,
            }
        statistical_results = {"bootstrap": bootstrap, "paired_tests": tests}
        uncertainty_status = "computed"
    else:
        statistical_results = None
        uncertainty_status = "blocked_incomplete_publication_coverage"

    metrics = {
        "schema_version": "phase-b-metrics-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "matcher_id": MATCHER_ID,
        "evaluation_split_id": expected_split,
        "selected_confidence_threshold": threshold,
        "observed_training_seeds": observed_seeds,
        "required_training_seeds": config.value["training_seeds"],
        "publication_seed_coverage_complete": seed_coverage_complete,
        "publication_data_coverage_complete": data_coverage_complete,
        "nonpublication_smoke": inputs.nonpublication_smoke,
        "n_gold_examples": len(gold),
        "n_raw_candidates": len(candidates),
        "conditions": conditions,
        "uncertainty_status": uncertainty_status,
        "statistics": statistical_results,
    }

    candidate_path = layout.resolve(planned_outputs[0])
    sentence_path = layout.resolve(planned_outputs[1])
    metrics_path = layout.resolve(planned_outputs[2])
    atomic_write_jsonl(candidate_path, candidate_outcomes)
    atomic_write_jsonl(sentence_path, sentence_outcomes)
    atomic_write_json(metrics_path, metrics)
    output_hashes = {
        layout.relative_identity(candidate_path): sha256_file(candidate_path),
        layout.relative_identity(sentence_path): sha256_file(sentence_path),
        layout.relative_identity(metrics_path): sha256_file(metrics_path),
    }
    manifest = {
        "schema_version": "phase-b-score-manifest-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "matcher_id": MATCHER_ID,
        "run_id": layout.run_id,
        "inputs": {
            layout.relative_identity(path): sha256_file(path)
            for path in (
                inputs.gold,
                inputs.candidates,
                inputs.simple_verdicts,
                inputs.corrective_verdicts,
                inputs.threshold_selection,
            )
        },
        "outputs": output_hashes,
        "nonpublication_smoke": inputs.nonpublication_smoke,
    }
    manifest_path = layout.resolve(planned_outputs[3])
    atomic_write_json(manifest_path, manifest)
    metrics["score_manifest"] = {
        "path": layout.relative_identity(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    return metrics
