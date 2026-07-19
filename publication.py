"""Deterministic assembly helpers for the canonical Phase B publication run.

The expensive model stage deliberately writes one candidate ledger per seed.
This module validates and combines those ledgers, emits the full development
candidate index required by the B-07 auditor, and freezes a small label-blind
pilot subset before any live verifier call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import PipelineConfig
from constants import PROTOCOL_ID, TRAINING_SEEDS, WORKFLOW_ID
from paths import RunLayout
from phase_b_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    iter_jsonl,
    sha256_file,
)
from records import Candidate
from scoring import _load_candidates, _load_gold


def _split_contract(split: str) -> tuple[str, str, str]:
    if split == "development":
        return (
            "CODE-SPLIT-1:development",
            "data-prepared/development-gold.jsonl",
            "predictions/dev",
        )
    if split == "test":
        return (
            "CODE-SPLIT-1:test",
            "data-prepared/test-gold.jsonl",
            "predictions/test",
        )
    raise DataContractError("candidate assembly split must be development or test")


def assemble_seed_candidates(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    split: str,
) -> dict[str, Any]:
    """Validate and combine the eight independent seed candidate ledgers."""

    layout.require_existing()
    split_id, gold_relative, directory = _split_contract(split)
    configured_seeds = tuple(config.value["training_seeds"])
    if configured_seeds != TRAINING_SEEDS:
        raise DataContractError("configured training seeds differ from frozen seeds 42-49")

    gold = _load_gold(layout.resolve(gold_relative, must_exist=True), split_id)
    combined_relative = (
        "predictions/dev/development-candidates.jsonl"
        if split == "development"
        else "predictions/test/candidates.jsonl"
    )
    combined_path = layout.resolve(combined_relative)
    index_path = (
        layout.resolve("predictions/dev/candidate-index.json")
        if split == "development"
        else None
    )
    planned = [combined_path] + ([index_path] if index_path is not None else [])
    existing = [layout.relative_identity(path) for path in planned if path.exists()]
    if existing:
        raise DataContractError(
            "candidate assembly refuses to overwrite existing artifacts: "
            + ", ".join(existing)
        )

    combined_rows: list[dict[str, Any]] = []
    index_files: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        relative = f"{directory}/seed-{seed}-candidates.jsonl"
        path = layout.resolve(relative, must_exist=True)
        candidates, _ = _load_candidates(path, split_id, gold)
        if {candidate.training_seed for candidate in candidates} != {seed}:
            raise DataContractError(f"{relative} must contain only training seed {seed}")
        raw_rows = [value for _, value in iter_jsonl(path)]
        if len(raw_rows) != len(candidates):
            raise DataContractError(f"{relative} parsed candidate count is inconsistent")
        combined_rows.extend(raw_rows)
        if index_path is not None:
            index_files.append(
                {
                    "training_seed": seed,
                    "path": relative,
                    "sha256": sha256_file(path),
                    "candidate_count": len(candidates),
                    "candidates": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "example_id": candidate.triple.example_id,
                        }
                        for candidate in candidates
                    ],
                }
            )

    parsed = [Candidate.from_mapping(row, "assembled candidate") for row in combined_rows]
    if [candidate.sort_key() for candidate in parsed] != sorted(
        candidate.sort_key() for candidate in parsed
    ):
        raise DataContractError("assembled candidates are not in canonical stable order")
    if len({candidate.candidate_id for candidate in parsed}) != len(parsed):
        raise DataContractError("assembled candidates contain duplicate candidate IDs")

    atomic_write_jsonl(combined_path, combined_rows)
    result: dict[str, Any] = {
        "split": split,
        "candidate_count": len(parsed),
        "candidates": layout.relative_identity(combined_path),
        "candidates_sha256": sha256_file(combined_path),
    }
    if index_path is not None:
        index = {
            "schema_version": "phase-b-candidate-index-1.0",
            "protocol_id": PROTOCOL_ID,
            "workflow_id": WORKFLOW_ID,
            "split_id": split_id,
            "split_manifest_sha256": sha256_file(
                layout.resolve("data-prepared/split-manifest.json", must_exist=True)
            ),
            "files": index_files,
            "candidate_count": len(parsed),
        }
        atomic_write_json(index_path, index)
        result.update(
            {
                "candidate_index": layout.relative_identity(index_path),
                "candidate_index_sha256": sha256_file(index_path),
            }
        )
    return result


def prepare_verifier_pilot(
    layout: RunLayout,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Freeze one label-blind development candidate per seed plus one warm-up."""

    layout.require_existing()
    candidates_path = layout.resolve(
        "predictions/dev/development-candidates.jsonl", must_exist=True
    )
    index_path = layout.resolve("predictions/dev/candidate-index.json", must_exist=True)
    split_path = layout.resolve("data-prepared/split-manifest.json", must_exist=True)
    gold = _load_gold(
        layout.resolve("data-prepared/development-gold.jsonl", must_exist=True),
        "CODE-SPLIT-1:development",
    )
    candidates, _ = _load_candidates(
        candidates_path, "CODE-SPLIT-1:development", gold
    )
    raw_by_id = {
        value["candidate_id"]: value for _, value in iter_jsonl(candidates_path)
    }

    selected: list[Candidate] = []
    for seed in TRAINING_SEEDS:
        seed_candidates = [item for item in candidates if item.training_seed == seed]
        if not seed_candidates:
            raise DataContractError(f"development pilot has no candidate for seed {seed}")
        selected.append(min(seed_candidates, key=lambda item: item.candidate_id))
    selected.sort(key=Candidate.sort_key)
    pilot_rows = [raw_by_id[item.candidate_id] for item in selected]
    warmup = min(candidates, key=Candidate.sort_key)

    pilot_path = layout.resolve("predictions/dev/pilot-candidates.jsonl")
    warmup_path = layout.resolve("predictions/dev/verifier-warmup-candidate.jsonl")
    selection_path = layout.resolve("predictions/dev/pilot-selection.json")
    planned = [pilot_path, warmup_path, selection_path]
    existing = [layout.relative_identity(path) for path in planned if path.exists()]
    if existing:
        raise DataContractError(
            "pilot preparation refuses to overwrite existing artifacts: "
            + ", ".join(existing)
        )

    atomic_write_jsonl(pilot_path, pilot_rows)
    atomic_write_jsonl(warmup_path, [raw_by_id[warmup.candidate_id]])
    candidate_ids = sorted(item.candidate_id for item in selected)
    selection = {
        "schema_version": "phase-b-verifier-pilot-selection-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "selection_split": "development",
        "split_manifest_sha256": sha256_file(split_path),
        "development_candidate_index_sha256": sha256_file(index_path),
        "pilot_candidates_sha256": sha256_file(pilot_path),
        "selected_before_live_calls": True,
        "used_test_labels": False,
        "selection_rule": (
            "lexicographically smallest canonical candidate_id independently "
            "within each frozen training seed 42-49"
        ),
        "candidate_count": len(selected),
        "training_seeds": list(TRAINING_SEEDS),
        "candidate_ids": candidate_ids,
        "example_ids": sorted({item.triple.example_id for item in selected}),
    }
    atomic_write_json(selection_path, selection)
    return {
        "candidate_count": len(selected),
        "pilot_candidates": layout.relative_identity(pilot_path),
        "pilot_selection": layout.relative_identity(selection_path),
        "warmup_candidate": layout.relative_identity(warmup_path),
    }
