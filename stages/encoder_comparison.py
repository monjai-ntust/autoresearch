"""Controller for the separate historical-derived encoder comparison path."""

from __future__ import annotations

import argparse
import hashlib
import json
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


_CHECKOUT_RELATIVE = "manifests/00-checkout-manifest.json"
_CHECKOUT_RECOVERY = "audit/checkout-recovery"


def _checkout_only_files(
    layout: RunLayout,
    selection: ComparisonSelection,
    *,
    transient: tuple[str, ...] = (),
) -> None:
    """Admit only checkout evidence, never acquired or downstream artifacts."""

    receipts = []
    snapshots = set()
    for path in layout.run_root.rglob("*"):
        if path.is_symlink():
            raise DataContractError("checkout recovery refuses symlinked run paths")
        if not path.is_file():
            continue
        relative = path.relative_to(layout.run_root).as_posix()
        if relative == _CHECKOUT_RELATIVE or relative in transient:
            continue
        if relative == f"{_CHECKOUT_RECOVERY}/recovery.lock":
            raise DataContractError("another checkout recovery is active or interrupted")
        match = re.fullmatch(
            rf"{_CHECKOUT_RECOVERY}/(blocked-checkout|checkout-attempt|recovery)-"
            r"([0-9a-f]{64})\.json",
            relative,
        )
        if match is None:
            raise DataContractError(
                f"checkout recovery refuses downstream or unknown artifact: {relative}"
            )
        if sha256_file(path) != match[2]:
            raise DataContractError(f"checkout recovery evidence hash differs: {relative}")
        document = _load_manifest(path, "checkout recovery evidence")
        if match[1] == "recovery":
            _require_fields(
                document,
                {
                    "schema_version": "phase-g-checkout-recovery-1.0",
                    "run_id": layout.run_id,
                    "arm_id": selection.arm.arm_id,
                    "comparison_config_sha256": selection.config.sha256,
                },
                "checkout recovery evidence",
            )
            if document.get("status") not in {"checkout-admitted", "blocked"}:
                raise DataContractError("checkout recovery evidence status is unsupported")
            receipts.append(document)
        else:
            _require_fields(
                document,
                {
                    "schema_version": "phase-b-checkout-manifest-2.0",
                    "protocol_id": selection.config.pipeline.value["protocol_id"],
                    "workflow_id": selection.config.pipeline.value["workflow_id"],
                    "matcher_id": selection.config.pipeline.value["matcher_id"],
                    "run_id": layout.run_id,
                },
                "archived checkout evidence",
            )
            if document.get("status") not in {"pass", "blocked"}:
                raise DataContractError("archived checkout status is unsupported")
            if match[1] == "blocked-checkout" and document["status"] != "blocked":
                raise DataContractError("failed checkout archive was not blocked")
            snapshots.add(relative)
    for receipt in receipts:
        for key, prefix in (
            ("previous_checkout", "blocked-checkout"),
            ("attempt_checkout", "checkout-attempt"),
        ):
            binding = receipt.get(key)
            if (
                not isinstance(binding, dict)
                or binding.get("path") not in snapshots
                or binding["path"]
                != f"{_CHECKOUT_RECOVERY}/{prefix}-{binding.get('sha256')}.json"
            ):
                raise DataContractError("checkout recovery evidence binding differs")


def _write_checkout_evidence(layout: RunLayout, kind: str, payload: bytes) -> dict[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    relative = f"{_CHECKOUT_RECOVERY}/{kind}-{digest}.json"
    path = layout.resolve(relative)
    if path.exists():
        if path.read_bytes() != payload:
            raise DataContractError("immutable checkout recovery evidence differs")
    else:
        atomic_write_bytes(path, payload)
    if sha256_file(path) != digest:
        raise DataContractError("checkout recovery evidence copy failed hash verification")
    return {"path": relative, "sha256": digest}


def recover_checkout(layout: RunLayout, selection: ComparisonSelection) -> dict[str, Any]:
    """Re-admit a dirty-checkout-only failure without importing any old input."""

    layout.require_existing()
    _checkout_only_files(layout, selection)
    path = layout.resolve(_CHECKOUT_RELATIVE, must_exist=True)
    previous_bytes = path.read_bytes()
    previous = _load_manifest(path, "blocked checkout manifest")
    config = selection.config.pipeline
    _require_fields(
        previous,
        {
            "schema_version": "phase-b-checkout-manifest-2.0",
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "matcher_id": config.value["matcher_id"],
            "run_id": layout.run_id,
        },
        "blocked checkout manifest",
    )
    if previous.get("status") == "pass":
        _validate_existing_checkout(layout, config)
        return {"run_id": layout.run_id, "status": "checkout-already-admitted"}
    source = previous.get("source")
    checks = previous.get("checks")
    if (
        previous.get("status") != "blocked"
        or not isinstance(source, dict)
        or source.get("worktree_clean") is not False
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(check, dict)
            or not isinstance(check.get("check_id"), str)
            or check.get("status") not in {"pass", "fail"}
            for check in checks
        )
        or len({check["check_id"] for check in checks}) != len(checks)
        or [check.get("check_id") for check in checks if check["status"] == "fail"]
        != ["clean-tracked-source"]
    ):
        raise DataContractError("recovery requires a dirty-checkout-only blocked run")
    commit = source.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise DataContractError("blocked checkout lacks an immutable source commit")
    relative_config = config.path.relative_to(layout.source_root).as_posix()
    if source.get("config") != {
        "path": relative_config, "sha256": sha256_file(config.path)
    }:
        raise DataContractError("blocked checkout configuration differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=layout.source_root, capture_output=True, check=False,
    )
    if ancestor.returncode != 0:
        raise DataContractError("blocked checkout source is not an ancestor of HEAD")
    original_config = subprocess.run(
        ["git", "show", f"{commit}:{relative_config}"],
        cwd=layout.source_root, capture_output=True, check=False,
    )
    if (
        original_config.returncode != 0
        or hashlib.sha256(original_config.stdout).hexdigest()
        != source["config"]["sha256"]
    ):
        raise DataContractError("blocked checkout configuration is not authenticated by Git")

    attempt_id = uuid.uuid4().hex
    pending_relative = f"{_CHECKOUT_RECOVERY}/.pending-{attempt_id}.json"
    lock_relative = f"{_CHECKOUT_RECOVERY}/recovery.lock"
    lock = layout.resolve(lock_relative)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("xb") as stream:
            stream.write(attempt_id.encode("ascii"))
    except FileExistsError as exc:
        raise DataContractError("another checkout recovery is active or interrupted") from exc

    class RetryLayout(RunLayout):
        def create(self) -> Path:
            return self.require_existing()

        def resolve(self, relative, *, must_exist=False) -> Path:
            if Path(relative).as_posix() == _CHECKOUT_RELATIVE:
                relative = pending_relative
            return super().resolve(relative, must_exist=must_exist)

    retry = RetryLayout(layout.source_root, layout.run_id)
    pending = layout.resolve(pending_relative)
    try:
        _checkout_only_files(layout, selection, transient=(lock_relative,))
        archived = _write_checkout_evidence(layout, "blocked-checkout", previous_bytes)
        manifest, passed = run_doctor(retry, config)
        fresh_bytes = pending.read_bytes()
        attempted = _write_checkout_evidence(layout, "checkout-attempt", fresh_bytes)
        receipt = {
            "schema_version": "phase-g-checkout-recovery-1.0",
            "run_id": layout.run_id,
            "attempt_id": attempt_id,
            "arm_id": selection.arm.arm_id,
            "comparison_config_sha256": selection.config.sha256,
            "previous_checkout": archived,
            "attempt_checkout": attempted,
            "status": "blocked",
        }
        try:
            if not passed:
                raise DataContractError(f"Checkout recovery doctor failed: {manifest}")
            _validate_existing_checkout(retry, config)
            _checkout_only_files(
                layout, selection, transient=(lock_relative, pending_relative)
            )
            if path.read_bytes() != previous_bytes:
                raise DataContractError("blocked checkout changed during recovery")
            if pending.read_bytes() != fresh_bytes:
                raise DataContractError("checkout attempt changed during recovery")
            atomic_write_bytes(path, fresh_bytes)
            receipt["status"] = "checkout-admitted"
        except (DataContractError, OSError, ValueError) as exc:
            receipt["error"] = str(exc)
            _write_checkout_evidence(layout, "recovery", canonical_json_bytes(receipt))
            raise
        receipt_binding = _write_checkout_evidence(
            layout, "recovery", canonical_json_bytes(receipt)
        )
        return {**receipt, "recovery_manifest": receipt_binding}
    finally:
        # These are this invocation's transient files, never scientific output.
        pending.unlink(missing_ok=True)
        lock.unlink()


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
    state = torch.load(weight, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise DataContractError("selected encoder weight is not a state dictionary")
    tensors = list(state.values())
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise DataContractError("selected encoder state dictionary contains non-tensors")
    return sum(value.numel() for value in tensors)


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
                "--comparison-boost-schedule",
                str(boost["schedule"]),
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
ADMISSION_RELATIVE = "manifests/encoder-comparison-admission.json"


def _replace_argument(arguments: list[str], name: str, value: str) -> None:
    try:
        index = arguments.index(name)
    except ValueError as exc:
        raise DataContractError(f"private comparison arguments lack {name}") from exc
    arguments[index + 1] = value


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
        "table1_arm": selection.arm.manifest(),
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
    for name, value in replacements.items():
        _replace_argument(arguments, name, value)
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
            or summary.get("model_name") != selection.profile.model
            or summary.get("model_revision") != selection.profile.revision
            or summary.get("completed_steps") != 1
            or summary.get("test_evaluated") is not False
            or summary.get("environment", {}).get("cuda_available") is not True
        ):
            raise DataContractError(f"encoder comparison {label} smoke summary differs")
    if first.get("resumed_from") is not None:
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
        "cuda_device": first["environment"]["cuda_device"],
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
    """Run one bounded accelerator smoke and retain an admission boundary."""

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
            "status": "smoke-complete-awaiting-admission",
            "run_id": layout.run_id,
            "training_seed": seed,
            "table1_arm": selection.arm.manifest(),
            "identity": identity,
            "cache_identity": cache_identity,
            "evidence": evidence,
            "full_run_admitted": False,
            "final_test_accessed": False,
        }
        atomic_write_json(manifest_path, manifest)
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != "phase-g-encoder-smoke-1.0"
        or manifest.get("run_id") != layout.run_id
        or manifest.get("training_seed") != seed
        or manifest.get("table1_arm") != selection.arm.manifest()
        or manifest.get("identity") != identity
        or manifest.get("full_run_admitted") is not False
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


def record_admission(
    layout: RunLayout,
    selection: ComparisonSelection,
    *,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    """Record the explicit post-smoke decision needed before full training."""

    if decision not in {"admit", "reject"} or not reason.strip():
        raise DataContractError("comparison admission needs admit/reject and a reason")
    smoke_path = layout.resolve(SMOKE_MANIFEST_RELATIVE, must_exist=True)
    smoke = load_json(smoke_path)
    selection_manifest = validate_selection(layout, selection)
    if (
        smoke.get("status") != "smoke-complete-awaiting-admission"
        or smoke.get("run_id") != layout.run_id
        or smoke.get("table1_arm") != selection.arm.manifest()
    ):
        raise DataContractError("comparison smoke is not eligible for admission")
    value = {
        "schema_version": "phase-g-encoder-admission-1.0",
        "run_id": layout.run_id,
        "table1_arm": selection.arm.manifest(),
        "selection_manifest_sha256": sha256_file(
            layout.resolve("manifests/encoder-comparison-selection.json", must_exist=True)
        ),
        "smoke_manifest_sha256": sha256_file(smoke_path),
        "decision": decision,
        "reason": reason.strip(),
        "final_test_accessed": False,
        "selection": selection_manifest,
    }
    path = layout.resolve(ADMISSION_RELATIVE)
    if path.is_file():
        if load_json(path) != value:
            raise DataContractError("existing comparison admission decision differs")
    else:
        atomic_write_json(path, value)
    return value


def _require_admission(
    layout: RunLayout, selection: ComparisonSelection
) -> dict[str, Any]:
    path = layout.resolve(ADMISSION_RELATIVE, must_exist=True)
    value = load_json(path)
    smoke_path = layout.resolve(SMOKE_MANIFEST_RELATIVE, must_exist=True)
    if (
        value.get("schema_version") != "phase-g-encoder-admission-1.0"
        or value.get("run_id") != layout.run_id
        or value.get("table1_arm") != selection.arm.manifest()
        or value.get("decision") != "admit"
        or value.get("smoke_manifest_sha256") != sha256_file(smoke_path)
        or value.get("final_test_accessed") is not False
    ):
        raise DataContractError("encoder comparison arm lacks an admitted smoke decision")
    return value


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
        "table1_arm": selection.arm.manifest(),
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
            layout.source_root / "utils/encoder_comparison/data.py"
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
    _require_admission(layout, selection)
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
    _validate_prepared_inputs(layout, selection)
    if action == "smoke":
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
            resume_arguments = list(training_arguments)
            _replace_argument(
                resume_arguments, "--run-summary-out", str(paths["resume_summary"])
            )
            resume_arguments.extend(["--resume-from", str(paths["restart"])])
            if "--model-local-files-only" not in resume_arguments:
                resume_arguments.append("--model-local-files-only")
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
            _replace_argument(
                prediction_arguments, "--checkpoint", str(paths["checkpoint"])
            )
            _replace_argument(
                prediction_arguments, "--sentences", str(paths["sentence"])
            )
            _replace_argument(
                prediction_arguments,
                "--prediction-ledger-out",
                str(paths["ledger"]),
            )
            if "--model-local-files-only" not in prediction_arguments:
                prediction_arguments.append("--model-local-files-only")
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
