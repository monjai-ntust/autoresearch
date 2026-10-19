"""Controller for the separate historical-derived encoder comparison path."""

from __future__ import annotations

from utils.encoder.runtime import restart_identity, training_environment

from dataclasses import replace

from utils.encoder_comparison.options import ComparisonOptions

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from stages.preparation import _validate_existing_checkout, run as run_preparation
from utils.common.artifact_io import (
    DataContractError,
    _load_manifest,
    _require_fields,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
)
from utils.common.constants import MATCHER_ID, PROTOCOL_ID, TRAINING_SEEDS
from utils.common.environment import run_doctor
from utils.common.paths import RunLayout, discover_source_root
from utils.encoder.cache import (
    CACHE_RELATIVE,
    MANIFEST_RELATIVE as CACHE_MANIFEST_RELATIVE,
    verify_cache_manifest,
    write_cache_manifest,
)
from utils.encoder.model import (
    generate_candidates,
)
from utils.encoder_comparison.config import (
    ComparisonSelection,
    bind_selection,
    load_comparison_config,
    validate_selection,
)




def prepare(layout: RunLayout, selection: ComparisonSelection) -> dict[str, Any]:
    """Reuse canonical preparation, then bind this run to one profile/recipe."""

    run_preparation(layout, selection.config.pipeline)
    return bind_selection(layout, selection)


_CHECKOUT_RELATIVE = "manifests/00-checkout-manifest.json"



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


def _base_parameter_count(
    layout: RunLayout, selection: ComparisonSelection
) -> int:
    """Count pinned backbone parameters from the verified PyTorch state dict."""

    import torch

    weight = _cache_snapshot_root(layout, selection) / selection.profile.weight_path
    if (
        weight.stat().st_size != selection.profile.weight_bytes
        or sha256_file(weight) != selection.profile.weight_sha256
    ):
        raise DataContractError("selected encoder weight bytes differ from profile pin")
    state = torch.load(weight, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise DataContractError("selected encoder weight is not a state dictionary")
    tensors = []
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif (
            name == "config"
            and selection.profile.profile_id == "deberta-large-v1"
            and isinstance(value, dict)
        ):
            # The pinned 2021 large checkpoint embeds configuration metadata.
            # It has no parameters; the complete file is still digest-verified.
            continue
        else:
            raise DataContractError(
                f"selected encoder state dictionary contains non-tensors: {name!r}"
            )
    if not tensors:
        raise DataContractError("selected encoder state dictionary contains no tensors")
    return sum(value.numel() for value in tensors)


def trainer_arguments(layout: RunLayout, selection: ComparisonSelection, seed: int, *, mode: str, split: str | None=None) -> ComparisonOptions:
    """Derive all private trainer paths from one run/profile/seed namespace."""
    if seed not in TRAINING_SEEDS:
        raise DataContractError(f'training seed {seed} is outside seeds 42-49')
    recipe = selection.recipe.value
    checkpoint_dir = layout.resolve(f'checkpoints/seed-{seed}')
    args = ComparisonOptions(
        mode=mode,
        prepared_dir=str(layout.resolve('data-prepared', must_exist=True)),
        model_name=selection.profile.model,
        model_revision=selection.profile.revision,
        expected_hidden_size=selection.profile.hidden_size,
        model_cache_dir=str(layout.resolve(CACHE_RELATIVE)),
        batch_size=recipe['batch_size'],
        max_length=recipe['max_length'],
        max_span_width=recipe['max_span_width'],
        seed=seed,
    )
    if recipe['context_between_spans']:
        args.context_between_spans = True
    if layout.resolve(CACHE_MANIFEST_RELATIVE).is_file():
        args.model_local_files_only = True
    if mode == 'train':
        boost = recipe['comparison_boost']
        args = replace(args,
            lr=float(recipe['learning_rate']),
            max_steps=recipe['max_steps'],
            warmup_steps=recipe['warmup_steps'],
            re_weight=float(recipe['re_loss_weight']),
            re_no_rel_weight=float(recipe['re_no_rel_weight']),
            neg_sample_ratio=float(recipe['ner_negative_ratio']),
            focal_gamma=float(recipe['ner_focal_gamma']),
            bio_loss_weight=float(recipe['bio_loss_weight']),
            label_smoothing=float(recipe['label_smoothing']),
            eval_every=recipe['evaluation_every_steps'],
            comparison_boost_schedule=str(boost['schedule']),
            re_comparison_boost=float(boost['initial']),
            re_boost_adaptive_steps=boost['adaptive_step'],
            re_boost_adaptive_threshold=float(boost['threshold_low']),
            re_boost_adaptive_threshold2=float(boost['threshold_high']),
            re_boost_mid=float(boost['middle']),
            re_boost_end=float(boost['end']),
            save_best_to=str(checkpoint_dir / 'checkpoint.pt'),
            save_last_to=str(checkpoint_dir / 'restart-state.pt'),
            progress_log=str(layout.resolve(f'logs/encoder-comparison-seed-{seed}.log')),
            run_summary_out=str(checkpoint_dir / 'training-summary.json'),
        )
        restart = checkpoint_dir / 'restart-state.pt'
        summary = checkpoint_dir / 'training-summary.json'
        if restart.is_file() and (not summary.is_file()):
            args = replace(args,
                resume_from=str(restart),
            )
    elif mode == 'predict':
        if split not in {'development', 'test'}:
            raise DataContractError('prediction split must be development or test')
        args = replace(args,
            checkpoint=str(checkpoint_dir / 'checkpoint.pt'),
            sentences=str(layout.resolve(f'data-prepared/{split}.jsonl', must_exist=True)),
            prediction_ledger_out=str(layout.resolve(f"predictions/{('dev' if split == 'development' else 'test')}/seed-{seed}-prediction-ledger.jsonl")),
        )
    else:
        raise DataContractError(f'unsupported comparison trainer mode: {mode!r}')
    return args


def _private_command(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    action: str,
    *,
    split: str | None = None,
) -> list[str]:
    from stages.comparison_matrix import selected_matrix
    matrix, _identity = selected_matrix(layout, selection)
    command = [
        sys.executable,
        "-B",
        "encoder_comparison.py",
        f"_{action}-encoder",
        "--matrix-id",
        matrix.run_id,
        "--run-id",
        layout.run_id,
        "--arm",
        selection.arm.arm_id,
        "--training-seed",
        str(seed),
        "--config",
        str(selection.config.path.resolve()),
    ]
    if split:
        command.extend(["--split", split])
    return command


SMOKE_RELATIVE = "diagnostics/encoder-comparison-smoke"
SMOKE_MANIFEST_RELATIVE = "manifests/encoder-comparison-smoke.json"




def _smoke_paths(layout: RunLayout) -> dict[str, Path]:
    root = layout.resolve(SMOKE_RELATIVE)
    return {
        "root": root,
        "identity": root / "identity.json",
        "checkpoint": root / "checkpoint.pt",
        "restart": root / "restart-state.pt",
        "progress": root / "progress.log",
        "training_summary": root / "training-summary.json",
        "resume_summary": root / "resume-summary.json",
        "checkpoint_before_resume": root / "checkpoint-before-resume.sha256",
        "sentence": root / "development-one.jsonl",
        "ledger": root / "prediction-ledger.jsonl",
        "roundtrip": root / "roundtrip.json",
    }


def _smoke_identity(
    layout: RunLayout, selection: ComparisonSelection, seed: int
) -> dict[str, Any]:
    return {
        "schema_version": "phase-g-encoder-smoke-identity-1.0",
        "run_id": layout.run_id,
        "training_seed": seed,
        "arm_id": selection.arm.arm_id,
        "selection_manifest_sha256": sha256_file(
            layout.resolve("manifests/encoder-comparison-selection.json", must_exist=True)
        ),
        "split_manifest_sha256": sha256_file(
            layout.resolve("data-prepared/split-manifest.json", must_exist=True)
        ),
    }


def _smoke_training_arguments(
    layout: RunLayout, selection: ComparisonSelection, seed: int
) -> list[str]:
    paths = _smoke_paths(layout)
    arguments = trainer_arguments(layout, selection, seed, mode="train")
    replacements = {
        "--max-steps": "1",
        "--eval-every": "1",
        "--save-best-to": str(paths["checkpoint"]),
        "--save-last-to": str(paths["restart"]),
        "--progress-log": str(paths["progress"]),
        "--run-summary-out": str(paths["training_summary"]),
    }
    types = {"--max-steps": int, "--eval-every": int}
    for name, value in replacements.items():
        setattr(arguments, name[2:].replace("-", "_"), types[name](value) if name in types else value)
    return arguments


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"invalid smoke JSONL: {path.name}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise DataContractError(f"smoke JSONL contains a non-object: {path.name}")
    return values


def _validate_smoke_outputs(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
) -> dict[str, Any]:
    paths = _smoke_paths(layout)
    identity = _smoke_identity(layout, selection, seed)
    if load_json(paths["identity"]) != identity:
        raise DataContractError("encoder comparison smoke identity differs")
    first = load_json(paths["training_summary"])
    resumed = load_json(paths["resume_summary"])
    for label, summary in (("training", first), ("resume", resumed)):
        if (
            summary.get("status") != "completed"
            or summary.get("seed") != seed
            or summary.get("completed_steps") != 1
            or summary.get("test_evaluated") is not False
            or summary.get("run_identity") != restart_identity(layout.resolve("data-prepared"))
            or (training_environment(layout.run_root, summary) or {}).get("cuda_available") is not True
        ):
            raise DataContractError(f"encoder comparison {label} smoke summary differs")
    if first.get("resumed_from") not in {None, str(paths["restart"])}:
        raise DataContractError("initial encoder smoke unexpectedly resumed")
    if resumed.get("resumed_from") != str(paths["restart"]):
        raise DataContractError("encoder smoke did not round-trip its restart state")
    if first.get("selected_metrics") != resumed.get("selected_metrics"):
        raise DataContractError("encoder smoke resume changed selected metrics")
    ledger = _read_jsonl(paths["ledger"])
    sentence = _read_jsonl(paths["sentence"])
    if (
        len(sentence) != 1
        or len(ledger) != 1
        or ledger[0].get("protocol_id") != PROTOCOL_ID
        or ledger[0].get("training_seed") != seed
        or ledger[0].get("example_id") != sentence[0].get("example_id")
        or not isinstance(ledger[0].get("predicted_spans"), list)
        or not isinstance(ledger[0].get("predicted_relations"), list)
    ):
        raise DataContractError("encoder smoke candidate ledger is invalid")
    roundtrip = load_json(paths["roundtrip"])
    before_resume = paths["checkpoint_before_resume"].read_text(encoding="ascii").strip()
    if (
        roundtrip.get("schema_version") != "phase-g-encoder-smoke-roundtrip-1.0"
        or roundtrip.get("checkpoint_unchanged_after_resume") is not True
        or roundtrip.get("checkpoint_sha256") != before_resume
        or sha256_file(paths["checkpoint"]) != before_resume
        or roundtrip.get("candidate_rows") != 1
        or roundtrip.get("example_id") != sentence[0].get("example_id")
    ):
        raise DataContractError("encoder smoke round-trip evidence is invalid")
    return {
        "cuda_device": training_environment(layout.run_root, first)["cuda_device"],
        "selected_metrics": first["selected_metrics"],
        "artifacts": {
            layout.relative_identity(path): sha256_file(path)
            for name, path in paths.items()
            if name != "root"
        },
    }


def run_smoke(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    *,
    command_runner=subprocess.run,
) -> dict[str, Any]:
    """Run and authenticate one bounded accelerator smoke."""

    _validate_prepared_inputs(layout, selection)
    if seed not in TRAINING_SEEDS:
        raise DataContractError(f"training seed {seed} is outside seeds 42-49")
    paths = _smoke_paths(layout)
    identity = _smoke_identity(layout, selection, seed)
    if paths["identity"].is_file():
        if load_json(paths["identity"]) != identity:
            raise DataContractError("existing encoder smoke belongs to another identity")
    else:
        atomic_write_json(paths["identity"], identity)
    manifest_path = layout.resolve(SMOKE_MANIFEST_RELATIVE)
    if not manifest_path.is_file():
        if paths["roundtrip"].is_file():
            # A smoke can finish its CUDA work before manifest finalization fails.
            # Authenticate that same-run evidence without executing it again.
            _validate_smoke_outputs(layout, selection, seed)
        else:
            command = _private_command(layout, selection, seed, "smoke")
            try:
                command_runner(command, cwd=layout.source_root, check=True)
            except subprocess.CalledProcessError as exc:
                raise DataContractError(
                    f"encoder comparison smoke failed with exit status {exc.returncode}; "
                    "retain this run and rerun the same command after diagnosis"
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
        cache_identity = {
            **_verify_profile_cache(layout, selection),
            "base_parameter_count": _base_parameter_count(layout, selection),
            "base_parameter_count_source": selection.profile.base_parameter_count_source,
        }
        evidence = _validate_smoke_outputs(layout, selection, seed)
        manifest = {
            "schema_version": "phase-g-encoder-smoke-1.0",
            "status": "completed",
            "run_id": layout.run_id,
            "training_seed": seed,
            "arm_id": selection.arm.arm_id,
            "identity": identity,
            "cache_identity": cache_identity,
            "evidence": evidence,
            "final_test_accessed": False,
        }
        atomic_write_json(manifest_path, manifest)
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != "phase-g-encoder-smoke-1.0"
        or manifest.get("status") != "completed"
        or manifest.get("run_id") != layout.run_id
        or manifest.get("training_seed") != seed
        or manifest.get("arm_id") != selection.arm.arm_id
        or manifest.get("identity") != identity
        or manifest.get("final_test_accessed") is not False
    ):
        raise DataContractError("encoder comparison smoke manifest differs")
    current_cache_identity = {
        **_verify_profile_cache(layout, selection),
        "base_parameter_count": _base_parameter_count(layout, selection),
        "base_parameter_count_source": selection.profile.base_parameter_count_source,
    }
    if manifest.get("cache_identity") != current_cache_identity:
        raise DataContractError("encoder comparison smoke cache identity changed")
    evidence = _validate_smoke_outputs(layout, selection, seed)
    if manifest.get("evidence") != evidence:
        raise DataContractError("encoder comparison smoke artifacts changed")
    return manifest


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
        "arm_id": selection.arm.arm_id,
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
        or summary.get("completed_steps") != selection.recipe.value["max_steps"]
        or summary.get("run_identity") != restart_identity(layout.resolve("data-prepared"))
    ):
        raise DataContractError("comparison trainer summary violates its selection")
    selected_step = summary.get("selected_step")
    selected_f1 = summary.get("selected_metrics", {}).get("triple_f1")
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step < 1
        or selected_step > selection.recipe.value["max_steps"]
        or isinstance(selected_f1, bool)
        or not isinstance(selected_f1, (int, float))
        or not math.isfinite(selected_f1)
        or not 0 <= selected_f1 <= 1
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
    return {
        "schema_version": "encoder-checkpoint-1",
        "protocol_id": PROTOCOL_ID,
        "split_id": "CODE-SPLIT-1",
        "training_seed": seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": selected_step,
        "split_manifest_sha256": sha256_file(
            layout.resolve("data-prepared/split-manifest.json", must_exist=True)
        ),
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
        "config": selection.config.source_path,
        "model_cache_manifest": CACHE_MANIFEST_RELATIVE,
        "model_cache_manifest_sha256": cache_identity["cache_manifest_sha256"],
        "model_cache_tree_sha256": cache_identity["cache_tree_sha256"],
        "source_commit": checkout["source"]["commit"],
        "selected_metric_value": float(selected_f1),
        "restart_state_sha256": sha256_file(restart),
    }


def _training_result(layout, plan, checkpoint_manifest_path, cache_identity) -> dict:
    seed = plan["training_seed"]
    checkpoint_dir = checkpoint_manifest_path.parent
    return {
        **plan,
        "schema_version": "phase-g-encoder-train-result-1.0",
        "status": "completed",
        "checkpoint_manifest": layout.relative_identity(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "cache_identity": cache_identity,
        "output_hashes": {layout.relative_identity(p): sha256_file(p) for p in (
            checkpoint_manifest_path, checkpoint_dir / "training-summary.json",
            checkpoint_dir / "restart-state.pt",
            layout.resolve(f"logs/encoder-comparison-seed-{seed}.log"))},
    }


def run_training(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    *,
    command_runner=subprocess.run,
) -> dict[str, Any]:
    from stages.comparison_matrix import require_matrix_progress
    require_matrix_progress(layout, selection, boundary="training")
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
    stage_manifest = _training_result(layout, plan, checkpoint_manifest_path, cache_identity)
    result_path = layout.resolve(
        f"manifests/encoder-comparison-train-live-seed-{seed}.json"
    )
    if result_path.is_file():
        if load_json(result_path) != stage_manifest:
            raise DataContractError("existing comparison training result differs")
    else:
        atomic_write_json(result_path, stage_manifest)
    return stage_manifest


def _prediction_identity(layout, selection, seed, split) -> dict:
    """Bind raw model output before the deterministic candidate conversion."""
    inputs = (
        "manifests/00-checkout-manifest.json",
        "manifests/encoder-comparison-selection.json",
        CACHE_MANIFEST_RELATIVE,
        f"checkpoints/seed-{seed}/checkpoint-manifest.json",
        f"checkpoints/seed-{seed}/checkpoint.pt",
        f"data-prepared/{split}.jsonl",
    )
    return {"run_id": layout.run_id, "arm_id": selection.arm.arm_id,
        "training_seed": seed, "split": split,
        "inputs": {name: sha256_file(layout.resolve(name, must_exist=True)) for name in inputs}}


def _prediction_receipt(ledger: Path, identity: dict) -> dict:
    return {"identity": identity, "ledger_sha256": sha256_file(ledger)}


def generate(
    layout: RunLayout,
    selection: ComparisonSelection,
    seed: int,
    split: str,
    *,
    command_runner=subprocess.run,
) -> dict[str, Any]:
    from stages.comparison_matrix import require_matrix_progress
    require_matrix_progress(layout, selection, boundary="test" if split == "test" else "training")
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
    identity = _prediction_identity(layout, selection, seed, split)
    receipt = ledger.with_suffix(".jsonl.manifest.json")
    if receipt.exists() and not ledger.is_file():
        raise DataContractError("completed comparison prediction ledger is missing")
    if not ledger.is_file():
        command = _private_command(
            layout, selection, seed, "predict", split=split
        )
        command_runner(command, cwd=layout.source_root, check=True)
    if not receipt.is_file() or load_json(receipt) != _prediction_receipt(ledger, identity):
        raise DataContractError("comparison prediction lacks matching producer evidence")
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
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--config", default=default_config)
    if action == "predict":
        parser.add_argument("--split", choices=("development", "test"), required=True)
    args = parser.parse_args(arguments)
    source_root = discover_source_root()
    config = load_comparison_config(source_root, args.config)
    selection = config.select_arm(args.arm)
    layout = RunLayout(source_root=source_root, run_id=args.run_id)
    from stages.comparison_matrix import selected_matrix, require_matrix_progress
    from utils.common.execution import exclusive_run, require_active_dispatch
    matrix, _identity = selected_matrix(layout, selection)
    if args.matrix_id != matrix.run_id:
        raise DataContractError("private worker names another matrix")
    require_matrix_progress(layout, selection,
        boundary="smoke" if action == "smoke" else "test" if action == "predict" and args.split == "test" else "training")
    require_active_dispatch(matrix, [f"_{action}-encoder", *arguments],
        label="comparison-smoke" if action == "smoke" else "comparison-full")
    with exclusive_run(layout, "comparison-worker"):
        return _execute_private(layout, selection, args, action)


def _execute_private(layout, selection, args, action):
    _validate_prepared_inputs(layout, selection)
    if action == "smoke":
        from utils.encoder.runtime import configure_numerics
        configure_numerics()
        import torch

        if not torch.cuda.is_available():
            raise DataContractError("encoder comparison smoke requires CUDA")
        paths = _smoke_paths(layout)
        if load_json(paths["identity"]) != _smoke_identity(
            layout, selection, args.training_seed
        ):
            raise DataContractError("private smoke identity changed during dispatch")
        from utils.encoder_comparison.trainer import main as trainer_main

        training_arguments = _smoke_training_arguments(
            layout, selection, args.training_seed
        )
        if not paths["training_summary"].is_file():
            if paths["restart"].is_file():
                training_arguments.resume_from = str(paths["restart"])
            trainer_main(training_arguments)
        if not paths["checkpoint"].is_file() or not paths["restart"].is_file():
            raise DataContractError("private smoke did not write checkpoint state")
        checkpoint_sha256 = sha256_file(paths["checkpoint"])
        if paths["checkpoint_before_resume"].is_file():
            if (
                paths["checkpoint_before_resume"].read_text(encoding="ascii").strip()
                != checkpoint_sha256
            ):
                raise DataContractError("smoke checkpoint changed before resume")
        else:
            atomic_write_bytes(
                paths["checkpoint_before_resume"],
                (checkpoint_sha256 + "\n").encode("ascii"),
            )
        if not paths["resume_summary"].is_file():
            resume_arguments = replace(training_arguments)
            resume_arguments = replace(resume_arguments, run_summary_out=str(paths['resume_summary']))
            resume_arguments.resume_from = str(paths["restart"])
            if not resume_arguments.model_local_files_only:
                resume_arguments.model_local_files_only = True
            trainer_main(resume_arguments)
        if sha256_file(paths["checkpoint"]) != checkpoint_sha256:
            raise DataContractError("smoke resume changed the selected checkpoint")
        if not paths["sentence"].is_file():
            development = layout.resolve(
                "data-prepared/development.jsonl", must_exist=True
            )
            first_line = next(
                (line for line in development.read_bytes().splitlines() if line), None
            )
            if first_line is None:
                raise DataContractError("development split is empty during smoke")
            atomic_write_bytes(paths["sentence"], first_line + b"\n")
        if not paths["ledger"].is_file():
            prediction_arguments = trainer_arguments(
                layout,
                selection,
                args.training_seed,
                mode="predict",
                split="development",
            )
            prediction_arguments = replace(prediction_arguments, checkpoint=str(paths['checkpoint']))
            prediction_arguments = replace(prediction_arguments, sentences=str(paths['sentence']))
            prediction_arguments = replace(prediction_arguments, prediction_ledger_out=str(paths['ledger']))
            if not prediction_arguments.model_local_files_only:
                prediction_arguments.model_local_files_only = True
            trainer_main(prediction_arguments)
        ledger = _read_jsonl(paths["ledger"])
        sentence = _read_jsonl(paths["sentence"])
        atomic_write_json(
            paths["roundtrip"],
            {
                "schema_version": "phase-g-encoder-smoke-roundtrip-1.0",
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_unchanged_after_resume": (
                    sha256_file(paths["checkpoint"]) == checkpoint_sha256
                ),
                "candidate_rows": len(ledger),
                "example_id": sentence[0].get("example_id") if sentence else None,
            },
        )
        return 0
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
        identity = _prediction_identity(layout, selection, args.training_seed, split)
        from utils.common.execution import bind_json
        ledger = Path(trainer_arguments(layout, selection, args.training_seed,
            mode=mode, split=split).prediction_ledger_out)
        if ledger.exists() or ledger.with_suffix(".jsonl.manifest.json").exists():
            raise DataContractError("private prediction cannot overwrite retained output")
        bind_json(ledger.with_suffix(".jsonl.work.json"), identity)
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
    if action == "predict":
        bind_json(ledger.with_suffix(".jsonl.manifest.json"), _prediction_receipt(ledger, identity))
    return 0
