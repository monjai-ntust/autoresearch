"""Controller for the separate historical-derived encoder comparison path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from stages.preparation import run as run_preparation
from utils.common.artifact_io import (
    DataContractError,
    atomic_write_json,
    load_json,
    sha256_file,
)
from utils.common.constants import MATCHER_ID, PROTOCOL_ID, TRAINING_SEEDS
from utils.common.paths import RunLayout, discover_source_root
from utils.encoder.cache import (
    CACHE_RELATIVE,
    MANIFEST_RELATIVE as CACHE_MANIFEST_RELATIVE,
    verify_cache_manifest,
    write_cache_manifest,
)
from utils.encoder.model import (
    _dataset_compatibility_report,
    generate_candidates,
)
from utils.encoder_comparison.config import (
    ComparisonSelection,
    bind_selection,
    load_comparison_config,
    validate_selection,
)


def describe(selection: ComparisonSelection) -> dict[str, Any]:
    """Return the fully validated selection without creating a run."""

    return selection.manifest()


def prepare(layout: RunLayout, selection: ComparisonSelection) -> dict[str, Any]:
    """Reuse canonical preparation, then bind this run to one profile/recipe."""

    run_preparation(layout, selection.config.pipeline)
    return bind_selection(layout, selection)


def _validate_prepared_inputs(layout: RunLayout, selection: ComparisonSelection) -> None:
    from stages.encoder import _validate_private_training_inputs

    validate_selection(layout, selection)
    _validate_private_training_inputs(layout, selection.config.pipeline)


def _cache_snapshot_root(layout: RunLayout, selection: ComparisonSelection) -> Path:
    suffix = Path("snapshots") / selection.profile.revision
    matches = [
        path
        for path in layout.resolve(CACHE_RELATIVE, must_exist=True).rglob(
            selection.profile.revision
        )
        if path.is_dir() and path.as_posix().endswith(suffix.as_posix())
    ]
    if len(matches) != 1:
        raise DataContractError(
            "run-local Hugging Face cache lacks exactly one selected revision snapshot"
        )
    return matches[0]


def _verify_profile_cache(
    layout: RunLayout, selection: ComparisonSelection
) -> dict[str, Any]:
    snapshot = _cache_snapshot_root(layout, selection)
    weight = snapshot / selection.profile.weight_path
    if not weight.is_file():
        raise DataContractError("selected encoder snapshot lacks its pinned weight file")
    resolved = weight.resolve()
    if (
        resolved.stat().st_size != selection.profile.weight_bytes
        or sha256_file(resolved) != selection.profile.weight_sha256
    ):
        raise DataContractError("selected encoder weight bytes differ from profile pin")
    missing = [
        name for name in selection.profile.tokenizer_files if not (snapshot / name).is_file()
    ]
    if missing:
        raise DataContractError(
            "selected encoder snapshot lacks tokenizer files: " + ", ".join(missing)
        )
    manifest = verify_cache_manifest(
        layout,
        model=selection.profile.model,
        revision=selection.profile.revision,
    )
    return {
        "snapshot": layout.relative_identity(snapshot),
        "weight_sha256": selection.profile.weight_sha256,
        "weight_bytes": selection.profile.weight_bytes,
        "cache_manifest_sha256": sha256_file(
            layout.resolve(CACHE_MANIFEST_RELATIVE, must_exist=True)
        ),
        "cache_tree_sha256": manifest["tree_sha256"],
    }


def trainer_arguments(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    *,
    mode: str,
    split: str | None = None,
) -> list[str]:
    """Derive all private trainer paths from one run/profile/seed namespace."""

    if seed not in TRAINING_SEEDS:
        raise DataContractError(f"training seed {seed} is outside seeds 42-49")
    recipe = selection.recipe.value
    checkpoint_dir = layout.resolve(f"checkpoints/seed-{seed}")
    args = [
        "--mode",
        mode,
        "--prepared-dir",
        str(layout.resolve("data-prepared", must_exist=True)),
        "--model-name",
        selection.profile.model,
        "--model-revision",
        selection.profile.revision,
        "--expected-hidden-size",
        str(selection.profile.hidden_size),
        "--model-cache-dir",
        str(layout.resolve(CACHE_RELATIVE)),
        "--batch-size",
        str(recipe["batch_size"]),
        "--max-length",
        str(recipe["max_length"]),
        "--max-span-width",
        str(recipe["max_span_width"]),
        "--seed",
        str(seed),
    ]
    if recipe["context_between_spans"]:
        args.append("--context-between-spans")
    if layout.resolve(CACHE_MANIFEST_RELATIVE).is_file():
        args.append("--model-local-files-only")
    if mode == "train":
        boost = recipe["comparison_boost"]
        args.extend(
            [
                "--lr",
                str(recipe["learning_rate"]),
                "--max-steps",
                str(recipe["max_steps"]),
                "--warmup-steps",
                str(recipe["warmup_steps"]),
                "--re-weight",
                str(recipe["re_loss_weight"]),
                "--re-no-rel-weight",
                str(recipe["re_no_rel_weight"]),
                "--neg-sample-ratio",
                str(recipe["ner_negative_ratio"]),
                "--focal-gamma",
                str(recipe["ner_focal_gamma"]),
                "--bio-loss-weight",
                str(recipe["bio_loss_weight"]),
                "--label-smoothing",
                str(recipe["label_smoothing"]),
                "--eval-every",
                str(recipe["evaluation_every_steps"]),
                "--re-comparison-boost",
                str(boost["initial"]),
                "--re-boost-adaptive-steps",
                str(boost["adaptive_step"]),
                "--re-boost-adaptive-threshold",
                str(boost["threshold_low"]),
                "--re-boost-adaptive-threshold2",
                str(boost["threshold_high"]),
                "--re-boost-mid",
                str(boost["middle"]),
                "--re-boost-end",
                str(boost["end"]),
                "--save-best-to",
                str(checkpoint_dir / "checkpoint.pt"),
                "--save-last-to",
                str(checkpoint_dir / "restart-state.pt"),
                "--progress-log",
                str(layout.resolve(f"logs/encoder-comparison-seed-{seed}.log")),
                "--run-summary-out",
                str(checkpoint_dir / "training-summary.json"),
            ]
        )
        restart = checkpoint_dir / "restart-state.pt"
        summary = checkpoint_dir / "training-summary.json"
        if restart.is_file() and not summary.is_file():
            args.extend(["--resume-from", str(restart)])
    elif mode == "predict":
        if split not in {"development", "test"}:
            raise DataContractError("prediction split must be development or test")
        args.extend(
            [
                "--checkpoint",
                str(checkpoint_dir / "checkpoint.pt"),
                "--sentences",
                str(layout.resolve(f"data-prepared/{split}.jsonl", must_exist=True)),
                "--prediction-ledger-out",
                str(
                    layout.resolve(
                        f"predictions/{'dev' if split == 'development' else 'test'}/"
                        f"seed-{seed}-prediction-ledger.jsonl"
                    )
                ),
            ]
        )
    else:
        raise DataContractError(f"unsupported comparison trainer mode: {mode!r}")
    return args


def _private_command(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    action: str,
    *,
    split: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "encoder_comparison.py",
        f"_{action}-encoder",
        "--run-id",
        layout.run_id,
        "--profile",
        selection.profile.profile_id,
        "--recipe",
        selection.recipe.recipe_id,
        "--training-seed",
        str(seed),
        "--config",
        str(selection.config.path.resolve()),
    ]
    if split:
        command.extend(["--split", split])
    return command


def plan_training(
    layout: RunLayout, selection: ComparisonSelection, seed: int
) -> dict[str, Any]:
    _validate_prepared_inputs(layout, selection)
    if seed not in TRAINING_SEEDS:
        raise DataContractError(f"training seed {seed} is outside seeds 42-49")
    path = layout.resolve(
        f"manifests/encoder-comparison-train-plan-seed-{seed}.json"
    )
    manifest = {
        "schema_version": "phase-g-encoder-train-plan-1.0",
        "protocol_id": PROTOCOL_ID,
        "matcher_id": MATCHER_ID,
        "run_id": layout.run_id,
        "stage": "encoder-comparison-train",
        "status": "planned",
        "training_seed": seed,
        "selection_manifest_sha256": sha256_file(
            layout.resolve("manifests/encoder-comparison-selection.json", must_exist=True)
        ),
        "split_manifest_sha256": sha256_file(
            layout.resolve("data-prepared/split-manifest.json", must_exist=True)
        ),
        "train_jsonl_sha256": sha256_file(
            layout.resolve("data-prepared/train.jsonl", must_exist=True)
        ),
        "development_jsonl_sha256": sha256_file(
            layout.resolve("data-prepared/development.jsonl", must_exist=True)
        ),
        "encoder_profile": selection.profile.manifest(),
        "recipe": selection.recipe.value,
        "historical_source": selection.config.value["historical_source"],
        "final_test_selection_forbidden": True,
    }
    if path.is_file():
        if load_json(path) != manifest:
            raise DataContractError("existing comparison training plan differs")
    else:
        atomic_write_json(path, manifest)
    return manifest


def _checkpoint_manifest(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    cache_identity: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = layout.resolve(f"checkpoints/seed-{seed}/checkpoint.pt", must_exist=True)
    restart = layout.resolve(f"checkpoints/seed-{seed}/restart-state.pt", must_exist=True)
    summary_path = layout.resolve(
        f"checkpoints/seed-{seed}/training-summary.json", must_exist=True
    )
    summary = load_json(summary_path)
    if (
        summary.get("status") != "completed"
        or summary.get("test_evaluated") is not False
        or summary.get("seed") != seed
        or summary.get("model_name") != selection.profile.model
        or summary.get("model_revision") != selection.profile.revision
    ):
        raise DataContractError("comparison trainer summary violates its selection")
    selected_step = summary.get("selected_step")
    selected_f1 = summary.get("selected_metrics", {}).get("triple_f1")
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step < 1
        or isinstance(selected_f1, bool)
        or not isinstance(selected_f1, (int, float))
    ):
        raise DataContractError("comparison trainer summary lacks selected dev F1")
    acquisition_path = layout.resolve(
        "manifests/02-input-acquisition-manifest.json", must_exist=True
    )
    preparation_path = layout.resolve(
        "manifests/03-data-preparation-manifest.json", must_exist=True
    )
    checkout = load_json(
        layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    )
    acquisition = load_json(acquisition_path)
    preparation = load_json(preparation_path)
    compatibility_path, compatibility = _dataset_compatibility_report(
        layout, selection.pipeline_view(), preparation
    )
    return {
        "schema_version": "phase-b-model-checkpoint-manifest-4.0",
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1",
        "training_seed": seed,
        "base_model": selection.profile.model,
        "base_model_revision": selection.profile.revision,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": selected_step,
        "split_manifest_sha256": sha256_file(
            layout.resolve("data-prepared/split-manifest.json", must_exist=True)
        ),
        "max_span_width": selection.recipe.value["max_span_width"],
        "context_between_spans": selection.recipe.value["context_between_spans"],
        "archive_sha256": acquisition["archive"]["sha256"],
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "annotation_bundle_sha256": preparation["annotation_bundle_sha256"],
        "prepared_dataset_tree_sha256": preparation["dataset_tree_sha256"],
        "train_jsonl_sha256": sha256_file(
            layout.resolve("data-prepared/train.jsonl", must_exist=True)
        ),
        "development_jsonl_sha256": sha256_file(
            layout.resolve("data-prepared/development.jsonl", must_exist=True)
        ),
        "config_sha256": selection.config.sha256,
        "trainer_sha256": sha256_file(
            layout.source_root / "utils/encoder_comparison/trainer.py"
        ),
        "data_adapter_sha256": sha256_file(
            layout.source_root / "utils/encoder/data.py"
        ),
        "model_helper_sha256": sha256_file(
            layout.source_root / "utils/encoder_comparison/network.py"
        ),
        "model_cache_manifest": CACHE_MANIFEST_RELATIVE,
        "model_cache_manifest_sha256": cache_identity["cache_manifest_sha256"],
        "model_cache_tree_sha256": cache_identity["cache_tree_sha256"],
        "source_commit": checkout["source"]["commit"],
        "selected_metric": "development_strict_triple_f1",
        "selected_metric_value": float(selected_f1),
        "restart_state_sha256": sha256_file(restart),
        "dataset_compatibility_report": layout.relative_identity(compatibility_path),
        "dataset_compatibility_sha256": sha256_file(compatibility_path),
        "historical_comparability": compatibility["status"],
    }


def run_training(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    *,
    command_runner=subprocess.run,
) -> dict[str, Any]:
    plan = plan_training(layout, selection, seed)
    checkpoint_dir = layout.resolve(f"checkpoints/seed-{seed}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    layout.resolve(CACHE_RELATIVE).mkdir(parents=True, exist_ok=True)
    summary_path = checkpoint_dir / "training-summary.json"
    if not summary_path.is_file():
        command = _private_command(layout, selection, seed, "train")
        try:
            command_runner(command, cwd=layout.source_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise DataContractError(
                f"comparison trainer failed with exit status {exc.returncode}; "
                "rerun the same run/profile/seed to resume"
            ) from exc
    if layout.resolve(CACHE_MANIFEST_RELATIVE).is_file():
        verify_cache_manifest(
            layout,
            model=selection.profile.model,
            revision=selection.profile.revision,
        )
    else:
        write_cache_manifest(
            layout,
            model=selection.profile.model,
            revision=selection.profile.revision,
        )
    cache_identity = _verify_profile_cache(layout, selection)
    checkpoint_manifest = _checkpoint_manifest(
        layout, selection, seed, cache_identity
    )
    checkpoint_manifest_path = checkpoint_dir / "checkpoint-manifest.json"
    if checkpoint_manifest_path.is_file():
        if load_json(checkpoint_manifest_path) != checkpoint_manifest:
            raise DataContractError("existing comparison checkpoint manifest differs")
    else:
        atomic_write_json(checkpoint_manifest_path, checkpoint_manifest)
    stage_manifest = {
        **plan,
        "schema_version": "phase-g-encoder-train-result-1.0",
        "status": "completed",
        "checkpoint_manifest": layout.relative_identity(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "cache_identity": cache_identity,
    }
    result_path = layout.resolve(
        f"manifests/encoder-comparison-train-live-seed-{seed}.json"
    )
    if result_path.is_file():
        if load_json(result_path) != stage_manifest:
            raise DataContractError("existing comparison training result differs")
    else:
        atomic_write_json(result_path, stage_manifest)
    return stage_manifest


def generate(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    split: str,
    *,
    command_runner=subprocess.run,
) -> dict[str, Any]:
    _validate_prepared_inputs(layout, selection)
    if split not in {"development", "test"}:
        raise DataContractError("comparison generation split must be development or test")
    _verify_profile_cache(layout, selection)
    checkpoint_manifest = layout.resolve(
        f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True
    )
    short_split = "dev" if split == "development" else "test"
    ledger = layout.resolve(
        f"predictions/{short_split}/seed-{seed}-prediction-ledger.jsonl"
    )
    if not ledger.is_file():
        command = _private_command(
            layout, selection, seed, "predict", split=split
        )
        command_runner(command, cwd=layout.source_root, check=True)
    candidates = layout.resolve(
        f"predictions/{short_split}/seed-{seed}-candidates.jsonl"
    )
    return generate_candidates(
        layout,
        selection.pipeline_view(),
        execution_mode="replay",
        sentences_path=layout.resolve(
            f"data-prepared/{split}.jsonl", must_exist=True
        ),
        checkpoint_manifest_path=checkpoint_manifest,
        candidates_out_path=candidates,
        prediction_ledger_path=ledger,
    )


def run_private(arguments: list[str], *, action: str, default_config: str) -> int:
    parser = argparse.ArgumentParser(prog=f"encoder_comparison.py _{action}-encoder")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--config", default=default_config)
    if action == "predict":
        parser.add_argument("--split", choices=("development", "test"), required=True)
    args = parser.parse_args(arguments)
    source_root = discover_source_root()
    config = load_comparison_config(source_root, args.config)
    selection = config.select(args.profile, args.recipe)
    layout = RunLayout(source_root=source_root, run_id=args.run_id)
    _validate_prepared_inputs(layout, selection)
    if action == "train":
        plan_path = layout.resolve(
            f"manifests/encoder-comparison-train-plan-seed-{args.training_seed}.json",
            must_exist=True,
        )
        if load_json(plan_path) != plan_training(layout, selection, args.training_seed):
            raise DataContractError("private trainer plan changed during dispatch")
        mode = "train"
        split = None
    elif action == "predict":
        mode = "predict"
        split = args.split
    else:
        raise DataContractError(f"unsupported private comparison action: {action}")
    from utils.encoder_comparison.trainer import main as trainer_main

    trainer_main(
        trainer_arguments(
            layout,
            selection,
            args.training_seed,
            mode=mode,
            split=split,
        )
    )
    return 0
