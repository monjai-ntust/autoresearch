"""Canonical CODE-ACCORD span-NER and relation training.

This private module is invoked only by ``pipeline.py _train-encoder`` with
arguments derived from the validated run configuration.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


from utils.common.artifact_io import DataContractError, _load_manifest, _require_fields, validate_stage_receipt, write_stage_receipt, iter_jsonl, sha256_file
from utils.common.config import PipelineConfig, load_pipeline_config
from utils.common.paths import RunLayout, discover_source_root
from utils.encoder.model import canonical_trainer_arguments, generate_candidates, plan_training
from utils.evaluation.publication import assemble_seed_candidates
from utils.rag.runner import require_consistent_encoder_identity


def train_and_generate_development(
    layout: RunLayout,
    config: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, object], Path]:
    """Train all seeds and assemble authenticated development candidates."""

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
        _resume_generation_stage(layout, config, seed=seed, split="development")

    encoder_identity = require_consistent_encoder_identity(encoder_identities)
    development_candidates = layout.resolve(
        "predictions/dev/development-candidates.jsonl"
    )
    if not development_candidates.is_file() or not layout.resolve("predictions/dev/candidate-index.json").is_file():
        assemble_seed_candidates(layout, config, split="development")
    _validate_candidate_assembly(layout, config, "development")
    return encoder_identity, training_hardware, development_candidates


def generate_test_candidates(
    layout: RunLayout,
    config: PipelineConfig,
) -> Path:
    """Generate and assemble authenticated final-test candidates."""

    for seed in config.value["training_seeds"]:
        _resume_generation_stage(layout, config, seed=seed, split="test")
    test_candidates = layout.resolve("predictions/test/candidates.jsonl")
    if not test_candidates.is_file():
        assemble_seed_candidates(layout, config, split="test")
    _validate_candidate_assembly(layout, config, "test")
    return test_candidates


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
            "run_id": layout.run_id,
            "execution_mode": execution_mode,
            "status": "planned" if execution_mode == "dry-run" else "completed",
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
            "expected_checkpoint_dir": f"checkpoints/seed-{seed}",
        },
        f"seed-{seed} training manifest",
    )
    if manifest.get("config") != config.source_path:
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
        "model_cache_manifest": "manifests/huggingface-model-cache.json",
    }
    outputs = manifest.get("outputs")
    if outputs != expected_outputs:
        raise DataContractError(f"seed-{seed} training outputs use unexpected paths")
    for relative in outputs.values():
        if isinstance(relative, str):
            layout.resolve(relative, must_exist=True)

    hashes = manifest.get('output_hashes')
    if not isinstance(hashes, dict) or set(hashes) != set(expected_outputs.values()):
        raise DataContractError('training producer output hash coverage is incomplete')
    for relative, expected in hashes.items():
        if sha256_file(layout.resolve(relative, must_exist=True)) != expected:
            raise DataContractError("training producer output changed")
    checkpoint_path = layout.resolve(checkpoint_relative, must_exist=True)
    identity_path = layout.resolve(checkpoint_manifest_relative, must_exist=True)
    identity = _load_manifest(identity_path, "checkpoint manifest")
    if identity.get("schema_version") not in {
        "encoder-checkpoint-1",
    }:
        raise DataContractError(f"seed-{seed} checkpoint schema is unsupported")
    _require_fields(
        identity,
        {
            "protocol_id": config.value["protocol_id"],
            "training_seed": seed,
            "config": config.source_path,
        },
        f"seed-{seed} checkpoint manifest",
    )
    revision = config.value["training"]["base_model_revision"]
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
        "model_cache_manifest_sha256": "manifests/huggingface-model-cache.json",
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
    summary = _load_manifest(
        layout.resolve(expected_outputs["training_summary"], must_exist=True),
        "training summary",
    )
    from utils.encoder.runtime import restart_identity
    if (
        summary.get("status") != "completed"
        or summary.get("seed") != seed
        or summary.get("test_evaluated") is not False
        or summary.get('completed_steps') != config.value['training']['max_steps']
        or summary.get('run_identity') != restart_identity(layout.resolve('data-prepared'))
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
    for relative, expected in manifest.get("output_hashes", {}).items():
        if sha256_file(layout.resolve(relative, must_exist=True)) != expected:
            raise DataContractError("training producer output changed")
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
        validate_stage_receipt(layout, manifest_path, manifest)
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


def _resume_generation_stage(layout: RunLayout, config, *, seed: int, split: str) -> dict:
    manifest_path = layout.resolve(f"manifests/model-generate-candidates-live-seed-{seed}-{split}.json")
    if manifest_path.exists():
        manifest = _validate_generation_stage(layout, config, seed, split, require_seal=False)
        if manifest.get("receipt_version") != 1:
            write_stage_receipt(layout, manifest_path, manifest)
        return _validate_generation_stage(layout, config, seed, split)
    directory = "dev" if split == "development" else "test"
    manifest = generate_candidates(layout, config, execution_mode="live",
        sentences_path=layout.resolve(f"data-prepared/{split}.jsonl", must_exist=True),
        checkpoint_manifest_path=layout.resolve(f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True),
        candidates_out_path=layout.resolve(f"predictions/{directory}/seed-{seed}-candidates.jsonl"),
        checkpoint_blob_path=layout.resolve(f"checkpoints/seed-{seed}/checkpoint.pt", must_exist=True))
    write_stage_receipt(layout, manifest_path, manifest)
    return _validate_generation_stage(layout, config, seed, split)


def run_private_trainer(
    arguments: list[str], *, default_config: str
) -> int:
    """Resolve the trainer only from a run ID, approved seed, and tracked config."""

    parser = argparse.ArgumentParser(prog="pipeline.py _train-encoder")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--config", default=default_config)
    args = parser.parse_args(arguments)
    source_root = discover_source_root()
    config = load_pipeline_config(source_root, args.config)
    layout = RunLayout(source_root=source_root, run_id=args.run_id)
    layout.require_existing()
    if args.training_seed not in config.value["training_seeds"]:
        raise DataContractError(
            f"training seed {args.training_seed} is not approved by the run config"
        )

    from stages.preparation import _validate_existing_checkout, require_canonical_namespace
    from utils.common.execution import require_active_dispatch

    require_canonical_namespace(layout)
    require_active_dispatch(layout, ['_train-encoder', *arguments],
                            label=f'canonical-train-seed-{args.training_seed}')
    _validate_existing_checkout(layout, config)
    _validate_training_stage(
        layout, config, args.training_seed, execution_mode="dry-run"
    )
    _validate_private_training_inputs(layout, config)
    # Import resolution occurs only after the narrow namespace contract passes.
    # The trainer never sees user-supplied artifact paths.
    from utils.encoder.trainer import main
    from utils.common.execution import exclusive_run
    with exclusive_run(layout, f"train-{args.training_seed}"):
        main(canonical_trainer_arguments(layout, config, args.training_seed))
    return 0
