"""Primary CODE-ACCORD publication and same-run Table-2 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Python isolated mode intentionally omits the script directory from sys.path
# on some platforms. The entry point restores only its own resolved checkout
# root so local modules remain importable without accepting ambient paths.
_ENTRY_ROOT = Path(__file__).resolve().parent
if str(_ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENTRY_ROOT))

from acquisition import fetch_run
from config import load_pipeline_config
from doctor import run_doctor
from model import (
    canonical_trainer_arguments,
    generate_candidates,
    plan_training,
    validate_prediction_cache,
)
from threshold import select_threshold
from artifact_io import DataContractError, atomic_write_json, iter_jsonl, sha256_file
from paths import PathContractError, RunLayout, discover_source_root
from pilot import PilotInputs, run_verifier_pilot
from preparation import prepare_run
from publication import assemble_seed_candidates, prepare_verifier_pilot
from reconciliation import reconcile_section5_evidence
from records import Candidate
from scoring import ScoreInputs, score_run
from verifier import run_verifier, validate_response_cache
import table2_runner


DEFAULT_CONFIG = "configs/pipeline.json"


def _load_manifest(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_outputs(layout: RunLayout, manifest: dict) -> dict[str, str]:
    declared = manifest.get("outputs")
    if isinstance(declared, dict):
        outputs = declared
    else:
        outputs = {}
        for field in ("generation_plan_output", "candidates_output"):
            relative = manifest.get(field)
            if isinstance(relative, str):
                path = layout.resolve(relative, must_exist=True)
                outputs[relative] = sha256_file(path)
        if (
            manifest.get("stage") == "model-generate-candidates"
            and manifest.get("execution_mode") == "live"
        ):
            ledger = manifest.get("inputs", {}).get("prediction_ledger")
            if isinstance(ledger, dict) and isinstance(ledger.get("path"), str):
                outputs[ledger["path"]] = ledger.get("sha256")
    if not outputs:
        raise DataContractError("stage producer manifest does not declare an output")
    for relative, expected in outputs.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise DataContractError("stage producer manifest has an invalid output binding")
        path = layout.resolve(relative, must_exist=True)
        if not path.is_file() or sha256_file(path) != expected:
            raise DataContractError(f"stage output differs from its producer: {relative}")
    return outputs


def _same_run_seal_path(producer_manifest: Path) -> Path:
    return producer_manifest.with_name(f"same-run-{producer_manifest.name}")


def _write_same_run_seal(
    layout: RunLayout, producer_manifest: Path, manifest: dict
) -> Path:
    outputs = _manifest_outputs(layout, manifest)
    seal_path = _same_run_seal_path(producer_manifest)
    document = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": layout.run_id,
        "producer_manifest": {
            "path": layout.relative_identity(producer_manifest),
            "sha256": sha256_file(producer_manifest),
        },
        "outputs": outputs,
    }
    atomic_write_json(seal_path, document)
    return seal_path


def _validate_same_run_seal(
    layout: RunLayout, producer_manifest: Path, manifest: dict
) -> None:
    outputs = _manifest_outputs(layout, manifest)
    seal = _load_manifest(
        _same_run_seal_path(producer_manifest), "same-run stage seal"
    )
    expected = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": layout.run_id,
        "producer_manifest": {
            "path": layout.relative_identity(producer_manifest),
            "sha256": sha256_file(producer_manifest),
        },
        "outputs": outputs,
    }
    if seal != expected:
        raise DataContractError("stage seal does not authenticate the selected run")


def _validate_pilot_capture_seals(layout: RunLayout, capture_index: Path) -> None:
    index = _load_manifest(capture_index, "pilot capture index")
    if index.get("run_id") != layout.run_id:
        raise DataContractError("pilot capture index carries another run identity")
    captures = index.get("captures")
    if not isinstance(captures, list) or len(captures) != 4:
        raise DataContractError("pilot capture index must contain four same-run captures")
    for capture in captures:
        if not isinstance(capture, dict):
            raise DataContractError("pilot capture index contains a malformed capture")
        mode = capture.get("mode")
        prefix = capture.get("artifact_prefix")
        if mode not in {"simple", "corrective"} or not isinstance(prefix, str):
            raise DataContractError("pilot capture index contains an invalid namespace")
        producer = layout.resolve(
            f"manifests/verifier-{prefix.replace('/', '-')}-{mode}-live.json",
            must_exist=True,
        )
        manifest = _load_manifest(producer, "pilot verifier producer manifest")
        _validate_same_run_seal(layout, producer, manifest)


def _require_fields(value: dict, expected: dict[str, object], label: str) -> None:
    for field, item in expected.items():
        if value.get(field) != item:
            raise DataContractError(f"{label}.{field} differs from this pipeline run")


def _validate_existing_checkout(layout: RunLayout, config) -> dict:
    """Authenticate an existing run namespace before any downstream work."""
    path = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    manifest = _load_manifest(path, "checkout manifest")
    if manifest.get("schema_version") not in {
        "phase-b-checkout-manifest-1.0",
        "phase-b-checkout-manifest-2.0",
    }:
        raise DataContractError("checkout manifest schema_version is unsupported")
    _require_fields(
        manifest,
        {
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "matcher_id": config.value["matcher_id"],
            "run_id": layout.run_id,
            "status": "pass",
        },
        "checkout manifest",
    )
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("worktree_clean") is not True:
        raise DataContractError("existing run was not admitted from a clean checkout")
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=layout.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=layout.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source.get("commit") != current or status:
        raise DataContractError(
            "existing run checkout differs from the current clean source commit"
        )
    config_relative = config.path.resolve().relative_to(layout.source_root).as_posix()
    source_config = source.get("config")
    tracked = manifest.get("tracked_artifact_sha256")
    recorded_config_hash = (
        source_config.get("sha256")
        if isinstance(source_config, dict)
        and source_config.get("path") == config_relative
        else tracked.get(config_relative) if isinstance(tracked, dict) else None
    )
    if recorded_config_hash != sha256_file(config.path):
        raise DataContractError(
            "existing run checkout does not authenticate its selected config"
        )
    return manifest


def _validate_private_training_inputs(layout: RunLayout, config) -> None:
    """Authenticate the run-local files consumed by the internal trainer."""

    acquisition_path = layout.resolve(
        "manifests/02-input-acquisition-manifest.json", must_exist=True
    )
    preparation_path = layout.resolve(
        "manifests/03-data-preparation-manifest.json", must_exist=True
    )
    acquisition = _load_manifest(acquisition_path, "acquisition manifest")
    preparation = _load_manifest(preparation_path, "preparation manifest")
    archive = acquisition.get("archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("path"), str):
        raise DataContractError("training acquisition manifest lacks archive identity")
    archive_path = layout.resolve(archive["path"], must_exist=True)
    if (
        acquisition.get("dataset_id") != config.value["dataset"]["dataset_id"]
        or archive.get("sha256") != sha256_file(archive_path)
        or preparation.get("protocol_id") != config.value["protocol_id"]
        or preparation.get("dataset_id") != config.value["dataset"]["dataset_id"]
        or preparation.get("byte_identical_independent_materializations") is not True
        or preparation.get("acquisition_manifest_sha256")
        != sha256_file(acquisition_path)
        or preparation.get("archive_sha256") != archive.get("sha256")
    ):
        raise DataContractError(
            "private trainer inputs are not bound to one acquisition/preparation"
        )
    artifacts = preparation.get("artifacts")
    if not isinstance(artifacts, list):
        raise DataContractError("preparation manifest lacks its artifact ledger")
    by_path = {
        item.get("path"): item
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for relative in ("train.jsonl", "development.jsonl", "split-manifest.json"):
        record = by_path.get(relative)
        path = layout.resolve(f"data-prepared/{relative}", must_exist=True)
        if (
            not isinstance(record, dict)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise DataContractError(
                f"private trainer input differs from preparation: {relative}"
            )
    split = _load_manifest(
        layout.resolve("data-prepared/split-manifest.json", must_exist=True),
        "split manifest",
    )
    expected_split = config.value["split"]
    if (
        split.get("split_id") != expected_split["split_id"]
        or split.get("seed") != expected_split["seed"]
        or split.get("train_count") != expected_split["train_sentences"]
        or split.get("development_count")
        != expected_split["development_sentences"]
        or split.get("test_count") != expected_split["test_sentences"]
    ):
        raise DataContractError("private trainer split differs from the selected config")


def _validate_reconciliation(layout: RunLayout, config) -> dict:
    path = layout.resolve("audit/section5-evidence-reconciliation.json", must_exist=True)
    audit = _load_manifest(path, "Section-5 evidence reconciliation")
    _require_fields(
        audit,
        {
            "run_id": layout.run_id,
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "status": "reconciled_secondary_evidence",
            "authority": "secondary_only",
            "canonical_publication_eligible": False,
        },
        "Section-5 evidence reconciliation",
    )
    register = audit.get("register")
    if (
        not isinstance(register, dict)
        or register.get("path")
        != config.section5_evidence_path.relative_to(layout.source_root).as_posix()
        or register.get("sha256") != sha256_file(config.section5_evidence_path)
    ):
        raise DataContractError("Section-5 reconciliation differs from its source register")
    return audit


def _validate_training_stage(
    layout: RunLayout, config, seed: int, *, execution_mode: str
) -> dict:
    manifest_path = layout.resolve(
        f"manifests/model-train-{execution_mode}-seed-{seed}.json", must_exist=True
    )
    manifest = _load_manifest(manifest_path, "training manifest")
    _require_fields(
        manifest,
        {
            "stage": "model-train",
            "execution_mode": execution_mode,
            "status": "planned" if execution_mode == "dry-run" else "completed",
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
            "expected_checkpoint_dir": f"checkpoints/seed-{seed}",
        },
        f"seed-{seed} training manifest",
    )
    if manifest.get("recipe") != config.value["training"]:
        raise DataContractError(f"seed-{seed} training recipe differs from this run")
    if execution_mode == "dry-run":
        return manifest

    checkpoint_relative = f"checkpoints/seed-{seed}/checkpoint.pt"
    checkpoint_manifest_relative = f"checkpoints/seed-{seed}/checkpoint-manifest.json"
    expected_outputs = {
        "checkpoint": checkpoint_relative,
        "checkpoint_manifest": checkpoint_manifest_relative,
        "restart_state": f"checkpoints/seed-{seed}/restart-state.pt",
        "training_summary": f"checkpoints/seed-{seed}/training-summary.json",
        "progress_log": f"logs/model-train-seed-{seed}.log",
        "dataset_compatibility_report": "audit/model-training-dataset-compatibility.json",
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or any(
        outputs.get(field) != relative for field, relative in expected_outputs.items()
    ):
        raise DataContractError(f"seed-{seed} training outputs use unexpected paths")
    for relative in outputs.values():
        if isinstance(relative, str):
            layout.resolve(relative, must_exist=True)

    checkpoint_path = layout.resolve(checkpoint_relative, must_exist=True)
    identity_path = layout.resolve(checkpoint_manifest_relative, must_exist=True)
    identity = _load_manifest(identity_path, "checkpoint manifest")
    if identity.get("schema_version") not in {
        "phase-b-model-checkpoint-manifest-2.0",
        "phase-b-model-checkpoint-manifest-3.0",
        "phase-b-model-checkpoint-manifest-4.0",
    }:
        raise DataContractError(f"seed-{seed} checkpoint schema is unsupported")
    _require_fields(
        identity,
        {
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
            "base_model": config.value["training"]["base_model"],
            "base_model_revision": config.value["training"]["base_model_revision"],
            "max_span_width": config.value["training"]["max_span_width"],
            "context_between_spans": config.value["training"]["context_between_spans"],
        },
        f"seed-{seed} checkpoint manifest",
    )
    revision = identity.get("base_model_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise DataContractError(f"seed-{seed} encoder revision is not immutable")
    digest = identity.get("checkpoint_sha256")
    if not isinstance(digest, str) or sha256_file(checkpoint_path) != digest:
        raise DataContractError(f"seed-{seed} checkpoint bytes differ from their manifest")

    known_inputs = {
        "checkout_manifest_sha256": "manifests/00-checkout-manifest.json",
        "acquisition_manifest_sha256": "manifests/02-input-acquisition-manifest.json",
        "preparation_manifest_sha256": "manifests/03-data-preparation-manifest.json",
        "split_manifest_sha256": "data-prepared/split-manifest.json",
        "train_jsonl_sha256": "data-prepared/train.jsonl",
        "development_jsonl_sha256": "data-prepared/development.jsonl",
    }
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise DataContractError(f"seed-{seed} training manifest lacks input hashes")
    for field, relative in known_inputs.items():
        path = layout.resolve(relative, must_exist=True)
        if inputs.get(field) != sha256_file(path):
            raise DataContractError(f"seed-{seed} training input changed: {relative}")
    for field, relative in {
        "split_manifest_sha256": "data-prepared/split-manifest.json",
        "acquisition_manifest_sha256": "manifests/02-input-acquisition-manifest.json",
        "train_jsonl_sha256": "data-prepared/train.jsonl",
        "development_jsonl_sha256": "data-prepared/development.jsonl",
    }.items():
        if field in identity and identity[field] != sha256_file(
            layout.resolve(relative, must_exist=True)
        ):
            raise DataContractError(f"seed-{seed} checkpoint identity changed: {relative}")
    restart = layout.resolve(expected_outputs["restart_state"], must_exist=True)
    if manifest.get("resume", {}).get("restart_state_sha256") != sha256_file(restart):
        raise DataContractError(f"seed-{seed} restart state differs from training manifest")
    compatibility = layout.resolve(
        expected_outputs["dataset_compatibility_report"], must_exist=True
    )
    if (
        identity.get("dataset_compatibility_sha256") is not None
        and identity.get("dataset_compatibility_sha256") != sha256_file(compatibility)
    ):
        raise DataContractError(f"seed-{seed} compatibility report changed")
    summary = _load_manifest(
        layout.resolve(expected_outputs["training_summary"], must_exist=True),
        "training summary",
    )
    if (
        summary.get("status") != "completed"
        or summary.get("canonical_mode") is not True
        or summary.get("seed") != seed
        or summary.get("test_evaluated") is not False
    ):
        raise DataContractError(f"seed-{seed} training summary is not publication-safe")
    return manifest


def _validate_generation_stage(
    layout: RunLayout,
    config,
    seed: int,
    split: str,
    *,
    require_seal: bool = True,
) -> dict:
    manifest_path = layout.resolve(
        f"manifests/model-generate-candidates-live-seed-{seed}-{split}.json",
        must_exist=True,
    )
    manifest = _load_manifest(manifest_path, "candidate-generation manifest")
    _require_fields(
        manifest,
        {
            "stage": "model-generate-candidates",
            "status": "completed",
            "execution_mode": "live",
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
        },
        f"seed-{seed} {split} generation manifest",
    )
    directory = "dev" if split == "development" else "test"
    candidates_relative = f"predictions/{directory}/seed-{seed}-candidates.jsonl"
    checkpoint_relative = f"checkpoints/seed-{seed}/checkpoint-manifest.json"
    prepared_relative = f"data-prepared/{split}.jsonl"
    checkpoint_path = layout.resolve(checkpoint_relative, must_exist=True)
    prepared_path = layout.resolve(prepared_relative, must_exist=True)
    inputs = manifest.get("inputs", {})
    checkpoint_input = inputs.get("checkpoint_manifest", {})
    prepared_input = inputs.get("prepared_sentences", {})
    if (
        manifest.get("candidates_output") != candidates_relative
        or checkpoint_input.get("path") != checkpoint_relative
        or checkpoint_input.get("sha256") != sha256_file(checkpoint_path)
        or prepared_input.get("path") != prepared_relative
        or prepared_input.get("sha256") != sha256_file(prepared_path)
    ):
        raise DataContractError(
            f"seed-{seed} {split} candidates are not bound to same-run inputs"
        )
    checkpoint = _load_manifest(checkpoint_path, "checkpoint manifest")
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint.get(
        "checkpoint_sha256"
    ):
        raise DataContractError(f"seed-{seed} {split} checkpoint identity differs")
    if require_seal:
        _validate_same_run_seal(layout, manifest_path, manifest)
    return manifest


def _validate_candidate_assembly(layout: RunLayout, config, split: str) -> None:
    directory = "dev" if split == "development" else "test"
    combined_relative = (
        "predictions/dev/development-candidates.jsonl"
        if split == "development"
        else "predictions/test/candidates.jsonl"
    )
    combined = layout.resolve(combined_relative, must_exist=True)
    rows: list[dict] = []
    expected_bytes = b""
    files = []
    for seed in config.value["training_seeds"]:
        relative = f"predictions/{directory}/seed-{seed}-candidates.jsonl"
        path = layout.resolve(relative, must_exist=True)
        seed_rows = [value for _, value in iter_jsonl(path)]
        if any(row.get("training_seed") != seed for row in seed_rows):
            raise DataContractError(f"{relative} contains another training seed")
        rows.extend(seed_rows)
        expected_bytes += path.read_bytes()
        files.append(
            {
                "training_seed": seed,
                "path": relative,
                "sha256": sha256_file(path),
                "candidate_count": len(seed_rows),
                "candidates": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "example_id": row.get("example_id"),
                    }
                    for row in seed_rows
                ],
            }
        )
    if combined.read_bytes() != expected_bytes:
        raise DataContractError(
            f"{combined_relative} is not the exact ordered same-run seed assembly"
        )
    if split == "development":
        index = _load_manifest(
            layout.resolve("predictions/dev/candidate-index.json", must_exist=True),
            "development candidate index",
        )
        expected = {
            "schema_version": "phase-b-candidate-index-1.0",
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "split_id": "CODE-SPLIT-1:development",
            "split_manifest_sha256": sha256_file(
                layout.resolve("data-prepared/split-manifest.json", must_exist=True)
            ),
            "files": files,
            "candidate_count": len(rows),
        }
        if index != expected:
            raise DataContractError("development candidate index differs from seed ledgers")


def _validate_threshold(layout: RunLayout, config) -> dict:
    path = layout.resolve("predictions/dev/threshold-selection.json", must_exist=True)
    document = _load_manifest(path, "threshold selection")
    expected = config.value["threshold_selection"]
    _require_fields(
        document,
        {
            "protocol_id": config.value["protocol_id"],
            "selection_split": expected["selection_split"],
            "objective": expected["objective"],
            "tie_rule": expected["tie_rule"],
            "grid": expected["grid"],
            "used_test_labels": False,
            "development_candidate_index_sha256": sha256_file(
                layout.resolve("predictions/dev/candidate-index.json", must_exist=True)
            ),
            "development_gold_sha256": sha256_file(
                layout.resolve("data-prepared/development-gold.jsonl", must_exist=True)
            ),
            "split_manifest_sha256": sha256_file(
                layout.resolve("data-prepared/split-manifest.json", must_exist=True)
            ),
        },
        "threshold selection",
    )
    rows = document.get("per_threshold")
    if not isinstance(rows, list) or [row.get("threshold") for row in rows] != expected["grid"]:
        raise DataContractError("threshold statistics do not cover the frozen grid")
    try:
        selected = max(
            (row["mean_per_seed_development_strict_triple_f1"], row["threshold"])
            for row in rows
        )[1]
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("threshold statistics are malformed") from exc
    if document.get("selected_threshold") != selected:
        raise DataContractError("selected threshold differs from the recorded argmax")
    return document


def _validate_pilot_inputs(layout: RunLayout, config) -> None:
    combined = layout.resolve(
        "predictions/dev/development-candidates.jsonl", must_exist=True
    )
    all_rows = [value for _, value in iter_jsonl(combined)]
    parsed = [Candidate.from_mapping(row, "development candidate") for row in all_rows]
    raw_by_id = {row["candidate_id"]: row for row in all_rows}
    selected: list[Candidate] = []
    for seed in config.value["training_seeds"]:
        candidates = [row for row in parsed if row.training_seed == seed]
        if not candidates:
            raise DataContractError(f"development candidates lack seed {seed}")
        selected.append(min(candidates, key=lambda row: row.candidate_id))
    selected.sort(key=Candidate.sort_key)
    expected_rows = [raw_by_id[row.candidate_id] for row in selected]
    pilot_path = layout.resolve("predictions/dev/pilot-candidates.jsonl", must_exist=True)
    warmup_path = layout.resolve(
        "predictions/dev/verifier-warmup-candidate.jsonl", must_exist=True
    )
    pilot_rows = [value for _, value in iter_jsonl(pilot_path)]
    warmup_rows = [value for _, value in iter_jsonl(warmup_path)]
    warmup = min(parsed, key=Candidate.sort_key)
    if pilot_rows != expected_rows or warmup_rows != [raw_by_id[warmup.candidate_id]]:
        raise DataContractError("pilot candidate files differ from deterministic selection")
    selection = _load_manifest(
        layout.resolve("predictions/dev/pilot-selection.json", must_exist=True),
        "pilot selection",
    )
    _require_fields(
        selection,
        {
            "schema_version": "phase-b-verifier-pilot-selection-1.0",
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "selection_split": "development",
            "split_manifest_sha256": sha256_file(
                layout.resolve("data-prepared/split-manifest.json", must_exist=True)
            ),
            "development_candidate_index_sha256": sha256_file(
                layout.resolve("predictions/dev/candidate-index.json", must_exist=True)
            ),
            "pilot_candidates_sha256": sha256_file(pilot_path),
            "selected_before_live_calls": True,
            "used_test_labels": False,
            "candidate_count": len(expected_rows),
            "training_seeds": config.value["training_seeds"],
            "candidate_ids": sorted(row.candidate_id for row in selected),
            "example_ids": sorted({row.triple.example_id for row in selected}),
        },
        "pilot selection",
    )


def _validate_verifier_stage(
    layout: RunLayout,
    config,
    manifest_path: Path,
    *,
    mode: str,
    artifact_prefix: str,
    require_seal: bool = True,
) -> dict:
    manifest = _load_manifest(manifest_path, "verifier manifest")
    _require_fields(
        manifest,
        {
            "protocol_id": config.value["protocol_id"],
            "condition_id": f"VER-{mode.upper()}",
            "execution_mode": "live",
            "status": "completed",
            "model_manifest_sha256": config.value["verifier"][
                "registry_manifest_sha256"
            ],
        },
        f"{mode} verifier manifest",
    )
    for mapping_name in ("inputs", "outputs"):
        mapping = manifest.get(mapping_name)
        if not isinstance(mapping, dict):
            raise DataContractError(f"{mode} verifier {mapping_name} are malformed")
        for relative, expected in mapping.items():
            path = layout.resolve(relative, must_exist=True)
            if not isinstance(expected, str) or sha256_file(path) != expected:
                raise DataContractError(f"{mode} verifier artifact changed: {relative}")
    model_inputs = {
        relative: digest
        for relative, digest in manifest["inputs"].items()
        if relative.startswith("inputs/ollama/blobs/sha256-")
    }
    expected_blob = config.value["verifier"]["model_blob_sha256"]
    if model_inputs != {f"inputs/ollama/blobs/sha256-{expected_blob}": expected_blob}:
        raise DataContractError(f"{mode} verifier uses another Qwen identity")
    prefix = artifact_prefix.strip("/")
    base = f"verifier/{prefix}/{mode}" if prefix else f"verifier/{mode}"
    environment_relative = f"{base}/environment-manifest.json"
    environment_hash = manifest["outputs"].get(environment_relative)
    if not isinstance(environment_hash, str):
        raise DataContractError(f"{mode} verifier lacks its environment identity")
    identity = table2_runner.load_verifier_model_identity(
        layout.resolve(environment_relative, must_exist=True), environment_hash
    )
    if (
        identity["name"] != config.value["verifier"]["model"]
        or identity["tag_digest"]
        != config.value["verifier"]["registry_manifest_sha256"]
        or identity["blob_sha256"] != expected_blob
    ):
        raise DataContractError(f"{mode} verifier environment uses another Qwen")
    if require_seal:
        _validate_same_run_seal(layout, manifest_path, manifest)
    return manifest


def _next_recovery_path(layout: RunLayout, category: str, filename: str) -> Path:
    directory = layout.resolve(f"inputs/recovery/{category}")
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob(f"*-{filename}"))
    return directory / f"{len(existing) + 1:03d}-{filename}"


def _quarantine_run_artifact(
    layout: RunLayout, path: Path, category: str
) -> Path | None:
    if not path.exists():
        return None
    layout.relative_identity(path)
    if not path.is_file() or path.is_symlink():
        raise DataContractError(f"recovery target is not a physical file: {path}")
    destination = _next_recovery_path(layout, f"quarantine/{category}", path.name)
    os.replace(path, destination)
    return destination


def _write_recovery_provenance(
    layout: RunLayout,
    cache: Path,
    *,
    kind: str,
    identity: dict[str, object],
    bindings: dict[str, Path],
) -> Path:
    provenance = cache.with_suffix(cache.suffix + ".manifest.json")
    checkout = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    document = {
        "schema_version": "phase-b-same-run-recovery-2.0",
        "run_id": layout.run_id,
        "kind": kind,
        "identity": identity,
        "cache": {
            "path": layout.relative_identity(cache),
            "sha256": sha256_file(cache),
        },
        "checkout_manifest": {
            "path": layout.relative_identity(checkout),
            "sha256": sha256_file(checkout),
        },
        "bindings": {
            name: {
                "path": layout.relative_identity(path),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(bindings.items())
        },
    }
    atomic_write_json(provenance, document)
    return provenance


def _validate_recovery_provenance(
    layout: RunLayout,
    cache: Path,
    *,
    kind: str,
    identity: dict[str, object],
    bindings: dict[str, Path],
) -> None:
    provenance_path = cache.with_suffix(cache.suffix + ".manifest.json")
    provenance = _load_manifest(provenance_path, "recovery provenance")
    checkout = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    expected = {
        "schema_version": "phase-b-same-run-recovery-2.0",
        "run_id": layout.run_id,
        "kind": kind,
        "identity": identity,
        "cache": {
            "path": layout.relative_identity(cache),
            "sha256": sha256_file(cache),
        },
        "checkout_manifest": {
            "path": layout.relative_identity(checkout),
            "sha256": sha256_file(checkout),
        },
        "bindings": {
            name: {
                "path": layout.relative_identity(path),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(bindings.items())
        },
    }
    if provenance != expected:
        raise DataContractError("recovery cache is not bound to the selected run")


def _resume_generation_stage(
    layout: RunLayout, config, *, seed: int, split: str
) -> dict:
    manifest_path = layout.resolve(
        f"manifests/model-generate-candidates-live-seed-{seed}-{split}.json"
    )
    seal_path = _same_run_seal_path(manifest_path)
    if manifest_path.is_file():
        manifest = _validate_generation_stage(
            layout, config, seed, split, require_seal=False
        )
        if not seal_path.is_file():
            _write_same_run_seal(layout, manifest_path, manifest)
        return _validate_generation_stage(layout, config, seed, split)

    directory = "dev" if split == "development" else "test"
    prepared = layout.resolve(f"data-prepared/{split}.jsonl", must_exist=True)
    checkpoint_manifest = layout.resolve(
        f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True
    )
    checkpoint_blob = layout.resolve(
        f"checkpoints/seed-{seed}/checkpoint.pt", must_exist=True
    )
    candidates = layout.resolve(f"predictions/{directory}/seed-{seed}-candidates.jsonl")
    live_ledger = layout.resolve(
        f"predictions/{directory}/seed-{seed}-prediction-ledger.jsonl"
    )
    bindings = {"prepared": prepared, "checkpoint_manifest": checkpoint_manifest}
    identity: dict[str, object] = {"training_seed": seed, "split": split}
    cache: Path | None = None
    if live_ledger.is_file():
        try:
            validate_prediction_cache(
                layout,
                config,
                sentences_path=prepared,
                checkpoint_manifest_path=checkpoint_manifest,
                cache_ledger_path=live_ledger,
            )
        except DataContractError:
            _quarantine_run_artifact(
                layout, live_ledger, f"generation-seed-{seed}-{split}"
            )
        else:
            cache = _next_recovery_path(
                layout,
                f"generation-seed-{seed}-{split}",
                "prediction-ledger.jsonl",
            )
            os.replace(live_ledger, cache)
            _write_recovery_provenance(
                layout,
                cache,
                kind="prediction-ledger",
                identity=identity,
                bindings=bindings,
            )
    if cache is None:
        directory_path = layout.resolve(
            f"inputs/recovery/generation-seed-{seed}-{split}"
        )
        if directory_path.is_dir():
            for recovery_candidate in sorted(
                directory_path.glob("*-prediction-ledger.jsonl"), reverse=True
            ):
                try:
                    _validate_recovery_provenance(
                        layout,
                        recovery_candidate,
                        kind="prediction-ledger",
                        identity=identity,
                        bindings=bindings,
                    )
                    validate_prediction_cache(
                        layout,
                        config,
                        sentences_path=prepared,
                        checkpoint_manifest_path=checkpoint_manifest,
                        cache_ledger_path=recovery_candidate,
                    )
                except DataContractError:
                    continue
                cache = recovery_candidate
                break
    for partial in (candidates, seal_path):
        _quarantine_run_artifact(
            layout, partial, f"generation-seed-{seed}-{split}"
        )
    manifest = generate_candidates(
        layout,
        config,
        execution_mode="live",
        sentences_path=prepared,
        checkpoint_manifest_path=checkpoint_manifest,
        candidates_out_path=candidates,
        prediction_ledger_path=None,
        cache_ledger_path=cache,
        checkpoint_blob_path=checkpoint_blob,
        base_model=None,
        device=None,
    )
    _write_same_run_seal(layout, manifest_path, manifest)
    return _validate_generation_stage(layout, config, seed, split)


def _resume_verifier_stage(
    layout: RunLayout,
    config,
    *,
    mode: str,
    sentences: Path,
    candidates: Path,
    development: Path,
    warmup_candidates: Path,
    model_blob: Path,
    ollama_url: str,
    artifact_prefix: str = "",
    pilot_selection: Path | None = None,
    allow_response_cache: bool = True,
) -> dict:
    prefix = artifact_prefix.strip("/")
    manifest_name = (
        f"verifier-{prefix.replace('/', '-')}-{mode}-live.json"
        if prefix
        else f"verifier-{mode}-live.json"
    )
    manifest_path = layout.resolve(f"manifests/{manifest_name}")
    seal_path = _same_run_seal_path(manifest_path)
    if manifest_path.is_file():
        manifest = _validate_verifier_stage(
            layout,
            config,
            manifest_path,
            mode=mode,
            artifact_prefix=prefix,
            require_seal=False,
        )
        if not seal_path.is_file():
            _write_same_run_seal(layout, manifest_path, manifest)
        return _validate_verifier_stage(
            layout, config, manifest_path, mode=mode, artifact_prefix=prefix
        )

    bindings = {"sentences": sentences, "candidates": candidates}
    identity: dict[str, object] = {
        "mode": mode,
        "condition_id": f"VER-{mode.upper()}",
    }
    if prefix:
        identity["artifact_prefix"] = prefix
    base = f"verifier/{prefix}/{mode}" if prefix else f"verifier/{mode}"
    response_path = layout.resolve(f"{base}/responses.jsonl")
    recovery_label = f"verifier-{prefix.replace('/', '-') + '-' if prefix else ''}{mode}"
    cache: Path | None = None
    if (
        allow_response_cache
        and response_path.is_file()
        and response_path.stat().st_size
    ):
        try:
            validate_response_cache(
                layout,
                config,
                mode=mode,
                sentences_path=sentences,
                candidates_path=candidates,
                cache_ledger_path=response_path,
            )
        except DataContractError:
            _quarantine_run_artifact(layout, response_path, recovery_label)
        else:
            cache = _next_recovery_path(
                layout, recovery_label, "responses.jsonl"
            )
            os.replace(response_path, cache)
            _write_recovery_provenance(
                layout,
                cache,
                kind="verifier-responses",
                identity=identity,
                bindings=bindings,
            )
    if cache is None:
        directory = layout.resolve(f"inputs/recovery/{recovery_label}")
        if allow_response_cache and directory.is_dir():
            for recovery_candidate in sorted(
                directory.glob("*-responses.jsonl"), reverse=True
            ):
                try:
                    _validate_recovery_provenance(
                        layout,
                        recovery_candidate,
                        kind="verifier-responses",
                        identity=identity,
                        bindings=bindings,
                    )
                    validate_response_cache(
                        layout,
                        config,
                        mode=mode,
                        sentences_path=sentences,
                        candidates_path=candidates,
                        cache_ledger_path=recovery_candidate,
                    )
                except DataContractError:
                    continue
                cache = recovery_candidate
                break
    partials = [
        layout.resolve(f"{base}/{name}")
        for name in (
            "requests.jsonl",
            "warmup-request.jsonl",
            "warmup-response.jsonl",
            "verdicts.jsonl",
            "environment-manifest.json",
            "run-log.jsonl",
            "model/tags.json",
            "model/show.json",
            "model/ollama-modelfile.txt",
        )
    ]
    partials.extend([response_path, seal_path])
    for partial in partials:
        _quarantine_run_artifact(layout, partial, recovery_label)
    manifest = run_verifier(
        layout,
        config,
        mode=mode,
        execution_mode="live",
        sentences_path=sentences,
        candidates_path=candidates,
        warmup_sentences_path=development,
        warmup_candidates_path=warmup_candidates,
        response_ledger_path=None,
        cache_ledger_path=cache,
        ollama_url=ollama_url,
        model_blob_path=model_blob,
        pilot_selection_path=pilot_selection,
        artifact_prefix=prefix,
    )
    _write_same_run_seal(layout, manifest_path, manifest)
    return _validate_verifier_stage(
        layout, config, manifest_path, mode=mode, artifact_prefix=prefix
    )


def _validate_pilot_audit(layout: RunLayout, config) -> dict:
    path = layout.resolve("audit/verifier-pilot/pilot-audit.json", must_exist=True)
    audit = _load_manifest(path, "verifier pilot audit")
    _require_fields(
        audit,
        {
            "run_id": layout.run_id,
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "evidence_class": "development-pilot",
            "pilot_status": "pass",
            "material_protocol_review_required": False,
        },
        "verifier pilot audit",
    )
    inputs = audit.get("input_sha256")
    if not isinstance(inputs, dict):
        raise DataContractError("verifier pilot audit lacks input hashes")
    for relative, expected in inputs.items():
        if sha256_file(layout.resolve(relative, must_exist=True)) != expected:
            raise DataContractError(f"verifier pilot input changed: {relative}")
    return audit


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


def _discover_live_model(
    config,
    source_root: Path,
    *,
    ollama_url: str,
    model_blob_source: str | None,
) -> tuple[Path, dict[str, object]]:
    """Resolve the current Qwen bytes; tracked digests are comparison metadata."""
    verifier_config = config.value["verifier"]
    origin = table2_runner.validate_ollama_url(ollama_url)
    tags, _tags_bytes, _tags_hash = table2_runner.fetch_ollama_json(
        "GET", origin + "/api/tags"
    )
    models = tags.get("models")
    if not isinstance(models, list):
        raise DataContractError("Ollama /api/tags response lacks a model inventory")
    matches = [
        item
        for item in models if isinstance(item, dict)
        and verifier_config["model"] in {item.get("name"), item.get("model")}
    ]
    if len(matches) != 1:
        raise DataContractError("Ollama does not uniquely expose the configured Qwen tag")
    tag_digest = table2_runner.validate_sha256(
        str(matches[0].get("digest", "")).removeprefix("sha256:"),
        "Ollama registry manifest",
    )
    show, _show_bytes, _show_hash = table2_runner.fetch_ollama_json(
        "POST", origin + "/api/show", {"model": verifier_config["model"]}
    )
    details = show.get("details")
    observed_details = {
        "family": str(details.get("family", "")).lower() if isinstance(details, dict) else "",
        "parameter_size": details.get("parameter_size") if isinstance(details, dict) else None,
        "quantization_level": details.get("quantization_level") if isinstance(details, dict) else None,
    }
    required_details = {
        "family": "qwen3",
        "parameter_size": "32.8B",
        "quantization_level": "Q4_K_M",
    }
    if observed_details != required_details:
        raise DataContractError(
            f"Ollama model family/size/quantization differs from the method: {observed_details}"
        )
    modelfile = show.get("modelfile")
    if not isinstance(modelfile, str):
        raise DataContractError("Ollama /api/show response lacks a Modelfile")
    from_values = [
        line.split(maxsplit=1)[1].strip()
        for line in modelfile.splitlines()
        if line.lstrip().upper().startswith("FROM ") and len(line.split(maxsplit=1)) == 2
    ]
    if len(from_values) != 1:
        raise DataContractError("Ollama Modelfile must name exactly one model blob")
    source = Path(model_blob_source or from_values[0]).expanduser().resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise DataContractError("Qwen model blob source must be one physical regular file")
    digest = sha256_file(source)
    declared = table2_runner.MODEL_BLOB_RE.findall(from_values[0])
    if declared and {item.lower() for item in declared} != {digest}:
        raise DataContractError("Ollama Modelfile and model blob bytes disagree")
    output_root = (source_root / "output").resolve()
    try:
        source.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise DataContractError(
            "A new run may not import a Qwen blob from another output run; use the external Ollama store"
        )
    reference = {
        "tag_digest": verifier_config["reference_registry_manifest_sha256"],
        "blob_sha256": verifier_config["reference_model_blob_sha256"],
    }
    # Downstream verifier and pilot code consume the authenticated identity of
    # this run. The tracked values above remain comparison metadata only.
    verifier_config["registry_manifest_sha256"] = tag_digest
    verifier_config["model_blob_sha256"] = digest
    return source, {
        "name": verifier_config["model"],
        "tag_digest": tag_digest,
        "blob_sha256": digest,
        "details": observed_details,
        "historical_reference": reference,
        "historical_reference_match": {
            "tag_digest": tag_digest == reference["tag_digest"],
            "blob_sha256": digest == reference["blob_sha256"],
        },
    }


def _materialize_model_blob(
    layout: RunLayout, source: Path, digest: str
) -> tuple[str, Path]:
    relative = f"inputs/ollama/blobs/sha256-{digest}"
    destination = layout.resolve(relative)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink() or sha256_file(destination) != digest:
            raise DataContractError("Existing run-local Qwen blob is not the selected model")
        return relative, destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if sha256_file(destination) != digest:
        raise DataContractError("Run-local Qwen copy failed its post-copy hash check")
    return relative, destination


def _write_full_run_manifest(
    layout: RunLayout,
    config,
    model_relative: str,
    model_identity: dict[str, object],
    encoder_identity: dict[str, str],
    hardware: dict[str, object],
) -> None:
    path = layout.resolve("manifests/debug-full-run.json")
    source = layout.source_root
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    contract = table2_runner.load_json(table2_runner.CONTRACT_PATH)
    reference = contract["reference_profile"]
    observed_profile = {
        "hardware": hardware,
        "encoder": encoder_identity,
        "qwen": {
            key: model_identity[key]
            for key in ("name", "tag_digest", "blob_sha256", "details")
        },
    }
    comparisons = {
        key: table2_runner._compare_profile(observed_profile[key], reference[key])
        for key in ("hardware", "encoder", "qwen")
    }
    document = {
        "schema_version": "phase-b-debug-full-run-1.0",
        "run_id": layout.run_id,
        "config": config.path.relative_to(source).as_posix(),
        "config_sha256": sha256_file(config.path),
        "source_commit": commit,
        "source_branch": branch,
        "source_upstream": "recorded-by-python-pipeline",
        "ollama_model": model_identity["name"],
        "ollama_tag_digest": model_identity["tag_digest"],
        "run_local_model_blob": model_relative,
        "model_blob_sha256": model_identity["blob_sha256"],
        "observed_profile": observed_profile,
        "historical_reference_profile": reference,
        "historical_reference_match": {
            **comparisons,
            "overall": all(
                value["all_fields_match"] for value in comparisons.values()
            ),
            "admission_effect": "none",
        },
        "conditional_b07_approval": True,
        "training_seeds": config.value["training_seeds"],
        "protocol_id": config.value["protocol_id"],
        "workflow_id": config.value["workflow_id"],
        "python_version": platform.python_version(),
        "uv_version": "environment-managed",
        "ollama_blob_discovery": "Ollama /api/show Modelfile FROM",
        "failure_policy": "fail_closed_and_resume_valid_completed_stages",
    }
    if path.exists():
        previous = _load_manifest(path, "full-run manifest")
        for key, expected in document.items():
            if previous.get(key) != expected:
                raise DataContractError(f"Existing full-run manifest differs at {key}")
        return
    atomic_write_json(path, document)


def _run_full_pipeline(
    layout: RunLayout,
    config,
    *,
    ollama_url: str,
    model_blob_source: str | None,
) -> int:
    """Run the sole supported canonical CODE-ACCORD and Table-2 workflow."""
    doctor_path = layout.resolve("manifests/00-checkout-manifest.json")
    if doctor_path.is_file():
        _validate_existing_checkout(layout, config)
    else:
        manifest, passed = run_doctor(layout, config)
        if not passed:
            raise DataContractError(f"Checkout doctor failed: {manifest}")
    reconciliation_path = layout.resolve("audit/section5-evidence-reconciliation.json")
    if reconciliation_path.is_file():
        _validate_reconciliation(layout, config)
    else:
        reconcile_section5_evidence(layout, config)
        _validate_reconciliation(layout, config)
    # Both producers authenticate existing artifacts, so calling them on every
    # resume closes the gap between a status flag and the bytes it describes.
    fetch_run(layout, config)
    prepare_run(layout, config)

    encoder_identities = []
    training_hardware: dict[str, object] = {}
    for seed in config.value["training_seeds"]:
        dry_manifest = layout.resolve(f"manifests/model-train-dry-run-seed-{seed}.json")
        if not dry_manifest.is_file():
            plan_training(
                layout, config, execution_mode="dry-run", training_seed=seed
            )
        _validate_training_stage(layout, config, seed, execution_mode="dry-run")
        live_manifest = layout.resolve(f"manifests/model-train-live-seed-{seed}.json")
        if not live_manifest.is_file():
            plan_training(layout, config, execution_mode="live", training_seed=seed)
        training = _validate_training_stage(
            layout, config, seed, execution_mode="live"
        )
        checkpoint_identity = _load_manifest(
            layout.resolve(
                f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True
            ),
            "checkpoint manifest",
        )
        encoder_identities.append(
            {
                "base_model": checkpoint_identity["base_model"],
                "base_model_revision": checkpoint_identity["base_model_revision"],
            }
        )
        if not training_hardware and isinstance(training.get("environment"), dict):
            training_hardware = training["environment"]
        _resume_generation_stage(
            layout, config, seed=seed, split="development"
        )

    encoder_identity = table2_runner.require_consistent_encoder_identity(
        encoder_identities
    )

    development_candidates = layout.resolve("predictions/dev/development-candidates.jsonl")
    if not development_candidates.is_file():
        assemble_seed_candidates(layout, config, split="development")
    _validate_candidate_assembly(layout, config, "development")
    threshold_path = layout.resolve("predictions/dev/threshold-selection.json")
    if not threshold_path.is_file():
        select_threshold(
            layout,
            config,
            candidates_path=development_candidates,
            gold_path=layout.resolve("data-prepared/development-gold.jsonl", must_exist=True),
            split_manifest_path=layout.resolve("data-prepared/split-manifest.json", must_exist=True),
            candidate_index_path=layout.resolve("predictions/dev/candidate-index.json", must_exist=True),
            out_path=threshold_path,
        )
    _validate_threshold(layout, config)
    pilot_selection = layout.resolve("predictions/dev/pilot-selection.json")
    if not pilot_selection.is_file():
        prepare_verifier_pilot(layout, config)
    _validate_pilot_inputs(layout, config)

    # The verifier model is deliberately discovered only at the first stage
    # that uses it. Its observed digests replace runtime slots for this run;
    # historical values remain non-blocking comparisons.
    source, model_identity = _discover_live_model(
        config,
        layout.source_root,
        ollama_url=ollama_url,
        model_blob_source=model_blob_source,
    )
    model_relative, model_blob = _materialize_model_blob(
        layout, source, str(model_identity["blob_sha256"])
    )
    _write_full_run_manifest(
        layout,
        config,
        model_relative,
        model_identity,
        encoder_identity,
        training_hardware,
    )

    pilot_candidates = layout.resolve("predictions/dev/pilot-candidates.jsonl", must_exist=True)
    warmup_candidates = layout.resolve(
        "predictions/dev/verifier-warmup-candidate.jsonl", must_exist=True
    )
    development = layout.resolve("data-prepared/development.jsonl", must_exist=True)
    capture_rows = []
    for mode in ("simple", "corrective"):
        for repeat in (1, 2):
            capture_id = f"{mode}-repeat-{repeat}"
            prefix = f"pilot/{capture_id}"
            _resume_verifier_stage(
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
                {"capture_id": capture_id, "mode": mode, "repeat": repeat, "artifact_prefix": prefix}
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
        if _load_manifest(capture_index, "pilot capture index") != capture_document:
            raise DataContractError("Existing pilot capture index differs from this run")
    else:
        atomic_write_json(capture_index, capture_document)
    _validate_pilot_capture_seals(layout, capture_index)
    audit_path = layout.resolve("audit/verifier-pilot/pilot-audit.json")
    if not audit_path.is_file():
        audit = run_verifier_pilot(
            layout,
            config,
            PilotInputs(
                sentences=development,
                gold=layout.resolve("data-prepared/development-gold.jsonl", must_exist=True),
                candidates=pilot_candidates,
                warmup_candidates=warmup_candidates,
                split_manifest=layout.resolve("data-prepared/split-manifest.json", must_exist=True),
                candidate_index=layout.resolve("predictions/dev/candidate-index.json", must_exist=True),
                pilot_selection=pilot_selection,
                threshold_selection=threshold_path,
                capture_index=capture_index,
            ),
            evidence_class="development-pilot",
        )
        if audit.get("pilot_status") != "pass" or audit.get("material_protocol_review_required") is not False:
            raise DataContractError("Development verifier pilot did not admit final-test execution")
    _validate_pilot_audit(layout, config)

    for seed in config.value["training_seeds"]:
        _resume_generation_stage(layout, config, seed=seed, split="test")
    test_candidates = layout.resolve("predictions/test/candidates.jsonl")
    if not test_candidates.is_file():
        assemble_seed_candidates(layout, config, split="test")
    _validate_candidate_assembly(layout, config, "test")

    for mode in ("simple", "corrective"):
        _resume_verifier_stage(
            layout,
            config,
            mode=mode,
            sentences=layout.resolve("data-prepared/test.jsonl", must_exist=True),
            candidates=test_candidates,
            development=development,
            warmup_candidates=warmup_candidates,
            model_blob=model_blob,
            ollama_url=ollama_url,
        )
    score_manifest = layout.resolve("manifests/score-manifest.json")
    if not score_manifest.is_file():
        score_run(
            layout,
            config,
            ScoreInputs(
                gold=layout.resolve("data-prepared/test-gold.jsonl", must_exist=True),
                candidates=test_candidates,
                simple_verdicts=layout.resolve("verifier/simple/verdicts.jsonl", must_exist=True),
                corrective_verdicts=layout.resolve("verifier/corrective/verdicts.jsonl", must_exist=True),
                threshold_selection=threshold_path,
                nonpublication_smoke=False,
                output_prefix="",
            ),
        )
    _validate_score(layout, config)
    contract = table2_runner.load_json(table2_runner.CONTRACT_PATH)
    table2_runner.validate_parent_lineage(layout.run_root, layout.run_id, contract)
    table_args = argparse.Namespace(
        run_id=layout.run_id, ollama_url=ollama_url, dry_run=False
    )
    return table2_runner.run_command(table_args, contract)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python pipeline.py",
        description=(
            "Canonical CODE-ACCORD pipeline with a complete-run route and "
            "a same-run downstream Table-2 route."
        ),
    )
    parser.add_argument(
        "--source-root",
        help="Direct src checkout root; normally discovered from the current directory.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    full = subparsers.add_parser(
        "full", help="Run or resume the canonical CODE-ACCORD and Table-2 pipeline"
    )
    full.add_argument("--config", default=DEFAULT_CONFIG)
    full.add_argument("--run-id", required=True)
    full.add_argument("--ollama-url", default="http://localhost:11434")
    full.add_argument("--model-blob-source")

    table2 = subparsers.add_parser(
        "table2", help="Run only downstream Table-2 RAG from a completed same run"
    )
    table2.add_argument("--run-id", required=True)
    table2.add_argument("--ollama-url", default="http://localhost:11434")
    table2.add_argument("--dry-run", action="store_true")

    validate_table2 = subparsers.add_parser(
        "validate-table2", help="Validate the frozen Table-2 source contract"
    )

    return parser


def _run_private_trainer(arguments: list[str]) -> int:
    """Resolve the trainer only from a run ID, approved seed, and tracked config."""

    parser = argparse.ArgumentParser(prog="pipeline.py _train-encoder")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(arguments)
    source_root = discover_source_root()
    config = load_pipeline_config(source_root, args.config)
    layout = RunLayout(source_root=source_root, run_id=args.run_id)
    layout.require_existing()
    if args.training_seed not in config.value["training_seeds"]:
        raise DataContractError(
            f"training seed {args.training_seed} is not approved by the run config"
        )
    _validate_existing_checkout(layout, config)
    _validate_training_stage(
        layout, config, args.training_seed, execution_mode="dry-run"
    )
    _validate_private_training_inputs(layout, config)
    # Import only after the narrow namespace contract has passed.  The trainer
    # never sees user-supplied artifact paths.
    from train_span import main as train_encoder

    train_encoder(canonical_trainer_arguments(layout, config, args.training_seed))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_train-encoder":
        try:
            return _run_private_trainer(arguments[1:])
        except (DataContractError, PathContractError, OSError, ValueError) as exc:
            print(f"pipeline: error: {exc}", file=sys.stderr)
            return 2
    args = _parser().parse_args(arguments)
    try:
        if args.stage in {"table2", "validate-table2"}:
            contract = table2_runner.load_json(table2_runner.CONTRACT_PATH)
            if args.stage == "validate-table2":
                print(json.dumps(table2_runner.validate_source_contract(contract, False), indent=2))
                return 0
            return table2_runner.run_command(args, contract)
        source_root = discover_source_root(Path(args.source_root) if args.source_root else None)
        config = load_pipeline_config(source_root, args.config)
        layout = RunLayout(source_root=source_root, run_id=args.run_id)
        if args.stage == "full":
            return _run_full_pipeline(
                layout,
                config,
                ollama_url=args.ollama_url,
                model_blob_source=args.model_blob_source,
            )
    except (
        DataContractError,
        PathContractError,
        table2_runner.PhaseEError,
        OSError,
        ValueError,
    ) as exc:
        print(f"pipeline: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
