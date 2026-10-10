"""The four-arm smoke/development/test barriers for fresh comparisons."""
from __future__ import annotations

import shutil
import subprocess

from utils.common.artifact_io import DataContractError, load_json, sha256_file
from utils.common.execution import bind_json, checked_process, exclusive_run
from utils.common.paths import RunLayout
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
    return {"schema_version": 1, "matrix_id": layout.run_id, "source_commit": source,
        "protocol_id": config.value["protocol_id"], "config_sha256": config.sha256,
        "pipeline_config_sha256": sha256_file(config.pipeline.path), "members": members,
        "arms": {arm: config.select_arm(arm).manifest(run) for arm, run in members.items()},
        "training_seeds": config.value["training_seeds"],
        "execution_limits": config.value["execution_limits"]}


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
    from stages.encoder_comparison import SMOKE_MANIFEST_RELATIVE, run_smoke, _checkpoint_manifest, _verify_profile_cache
    for arm, run_id in identity["members"].items():
        arm_layout = RunLayout(layout.source_root, run_id)
        arm_selection = selection.config.select_arm(arm)
        if not arm_layout.resolve(SMOKE_MANIFEST_RELATIVE).is_file():
            raise DataContractError("all four authenticated smokes must complete before training")
        run_smoke(arm_layout, arm_selection, 42)
        if boundary == "test":
            for seed in identity["training_seeds"]:
                path = arm_layout.resolve(f"manifests/encoder-comparison-train-live-seed-{seed}.json")
                if not path.is_file():
                    raise DataContractError("all 32 development runs must complete before test access")
                result = load_json(path)
                checkpoint = arm_layout.resolve(f"checkpoints/seed-{seed}/checkpoint-manifest.json", must_exist=True)
                expected = _checkpoint_manifest(arm_layout, arm_selection, seed, _verify_profile_cache(arm_layout, arm_selection))
                if (load_json(checkpoint) != expected or result.get("status") != "completed"
                        or result.get("run_id") != run_id
                        or result.get("checkpoint_manifest_sha256") != sha256_file(checkpoint)):
                    raise DataContractError("development completion evidence changed")
                for relative, digest in result.get("output_hashes", {}).items():
                    if sha256_file(arm_layout.resolve(relative, must_exist=True)) != digest:
                        raise DataContractError("development producer output changed")


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
        def execute(command, **kwargs):
            required = config.value["execution_limits"]["minimum_free_storage_bytes_before_full_run"]
            if shutil.disk_usage(layout.run_root).free < required:
                raise DataContractError(f"comparison full execution requires {required} free bytes")
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
                run_smoke(arm_layout, selection, 42)
        for arm, run_id in identity["members"].items():
            arm_layout = RunLayout(layout.source_root, run_id)
            selection = config.select_arm(arm)
            for seed in identity["training_seeds"]:
                run_training(arm_layout, selection, seed, command_runner=execute)
        for arm, run_id in identity["members"].items():
            arm_layout = RunLayout(layout.source_root, run_id)
            selection = config.select_arm(arm)
            for seed in identity["training_seeds"]:
                generate(arm_layout, selection, seed, "test", command_runner=execute)
            from utils.evaluation.reporting import report_comparison
            report_comparison(arm_layout, selection)
        from utils.evaluation.reporting import report_matrix
        report_matrix(layout, config, identity)
    return 0
