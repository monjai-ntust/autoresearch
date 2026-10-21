"""The four-arm smoke/development/test barriers for fresh comparisons."""
from __future__ import annotations

import subprocess

from utils.common.artifact_io import DataContractError, load_json, sha256_file
from utils.common.execution import bind_json, checked_process, exclusive_run
from utils.common.paths import RunLayout
from utils.common.storage import warn_storage
from utils.encoder_comparison.config import ComparisonConfig, validate_selection

MATRIX_RECEIPT = "manifests/comparison-matrix.json"
ARM_LINK = "manifests/comparison-matrix-link.json"


def matrix_identity(layout: RunLayout, config: ComparisonConfig) -> dict:
    from utils.rag.runner import run_git
    # Validate all derived names before making the first namespace/write.
    members = {arm: RunLayout(layout.source_root, f"{layout.run_id}-{arm}").run_id
               for arm in config.arms}
    source = run_git("rev-parse", "HEAD").stdout.strip()
    if run_git("status", "--porcelain", "--untracked-files=normal").stdout.strip():
        raise DataContractError("comparison matrix requires clean current source")
    for relative in (config.source_path, config.pipeline.source_path):
        if run_git('ls-files', '--error-unmatch', '--', relative, check=False).returncode != 0:
            raise DataContractError('comparison configurations must be tracked by the source commit')
    return {"matrix_id": layout.run_id, "source_commit": source,
        "config": config.source_path, "pipeline_config": config.pipeline.source_path,
        "members": members}


def selected_matrix(layout: RunLayout, selection) -> tuple[RunLayout, dict]:
    link = load_json(layout.resolve(ARM_LINK, must_exist=True))
    matrix = RunLayout(layout.source_root, link.get("matrix_id", ""))
    path = matrix.resolve(MATRIX_RECEIPT, must_exist=True)
    identity = load_json(path)
    expected = matrix_identity(matrix, selection.config)
    if identity != expected or link != {"matrix_id": matrix.run_id,
            "matrix_sha256": sha256_file(path), "run_id": layout.run_id,
            "arm_id": selection.arm.arm_id}:
        raise DataContractError("comparison matrix membership/source/config binding differs")
    if identity["members"].get(selection.arm.arm_id) != layout.run_id:
        raise DataContractError("arm run is not a member of its matrix")
    validate_selection(layout, selection)
    return matrix, identity


def require_matrix_progress(layout: RunLayout, selection, *, boundary: str) -> None:
    """Private workers enforce the same gates before importing model code."""
    matrix, identity = selected_matrix(layout, selection)
    if boundary == "smoke":
        return
    from stages.encoder_comparison import (SMOKE_MANIFEST_RELATIVE, run_smoke,
        _checkpoint_manifest, _verify_profile_cache, plan_training, _training_result)
    for arm, run_id in identity["members"].items():
        arm_layout = RunLayout(layout.source_root, run_id)
        arm_selection = selection.config.select_arm(arm)
        if not arm_layout.resolve(SMOKE_MANIFEST_RELATIVE).is_file():
            raise DataContractError("all four authenticated smokes must complete before training")
        run_smoke(arm_layout, arm_selection, 42)
        if boundary == "test":
            for seed in selection.config.value["training_seeds"]:
                path = arm_layout.resolve(f"manifests/encoder-comparison-train-live-seed-{seed}.json")
                if not path.is_file():
                    raise DataContractError("all 32 development runs must complete before test access")
                result = load_json(path)
                checkpoint = arm_layout.resolve(f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True)
                cache = _verify_profile_cache(arm_layout, arm_selection)
                expected = _checkpoint_manifest(arm_layout, arm_selection, seed, cache)
                arm_layout.resolve(f"manifests/encoder-comparison-train-plan-seed-{seed}.json", must_exist=True)
                plan = plan_training(arm_layout, arm_selection, seed)
                if (load_json(checkpoint) != expected
                        or result != _training_result(arm_layout, plan, checkpoint, cache)):
                    raise DataContractError("development completion evidence changed")


def run_matrix(layout: RunLayout, config: ComparisonConfig) -> int:
    identity = matrix_identity(layout, config)
    layout.run_root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(layout):
        path = layout.resolve(MATRIX_RECEIPT)
        if not path.exists():
            unexpected = [p for p in layout.run_root.rglob("*") if p.is_file() and p.suffix != ".lock"]
            if unexpected:
                raise DataContractError("matrix namespace contains unbound evidence")
        bind_json(path, identity)
        from stages.encoder_comparison import prepare, run_smoke, run_training, generate
        def smoke(command, **kwargs):
            checked_process(layout, command, label="comparison-smoke")

        def execute(command, **kwargs):
            budget = config.value["execution_limits"]["minimum_free_storage_bytes_before_full_run"]
            roots = [layout.run_root, *(RunLayout(layout.source_root, run_id).run_root
                for run_id in identity["members"].values())]
            warn_storage(layout.run_root, budget=budget, roots=roots, label="encoder_comparison")
            checked_process(layout, command, label="comparison-full",
                limit_seconds=config.value["execution_limits"]["maximum_total_gpu_hours"]*3600)
        for arm, run_id in identity["members"].items():
            arm_layout = RunLayout(layout.source_root, run_id)
            selection = config.select_arm(arm)
            arm_layout.run_root.mkdir(parents=True, exist_ok=True)
            with exclusive_run(arm_layout):
                prepare(arm_layout, selection)
                bind_json(arm_layout.resolve(ARM_LINK), {"matrix_id": layout.run_id,
                    "matrix_sha256": sha256_file(path), "run_id": run_id, "arm_id": arm})
                run_smoke(arm_layout, selection, 42, command_runner=smoke)
        for arm, run_id in identity["members"].items():
            arm_layout = RunLayout(layout.source_root, run_id)
            selection = config.select_arm(arm)
            with exclusive_run(arm_layout):
                for seed in config.value["training_seeds"]:
                    run_training(arm_layout, selection, seed, command_runner=execute)
        for arm, run_id in identity["members"].items():
            arm_layout = RunLayout(layout.source_root, run_id)
            selection = config.select_arm(arm)
            with exclusive_run(arm_layout):
                for seed in config.value["training_seeds"]:
                    generate(arm_layout, selection, seed, "test", command_runner=execute)
                from utils.evaluation.reporting import report_comparison
                report_comparison(arm_layout, selection)
        from utils.evaluation.reporting import report_matrix
        report_matrix(layout, config, identity)
    return 0
