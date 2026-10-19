"""Fail-closed same-run runner for the frozen downstream Table-2 workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from utils.common.artifact_io import DataContractError
from utils.common.constants import PROTOCOL_ID
from utils.encoder.cache import MANIFEST_RELATIVE as HF_CACHE_MANIFEST_RELATIVE
from utils.rag import evaluator
from utils.rag.graph import build_table_graphs, canonical_json, project_graph


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (ROOT / "output").resolve()
CONTRACT_PATH = ROOT / "resources/contracts/table2.json"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MODEL_BLOB_RE = re.compile(r"sha256[-:]([0-9a-f]{64})", re.IGNORECASE)


class PhaseEError(RuntimeError):
    """Raised when a frozen-method or same-run lineage gate fails."""


class ModelEvidenceError(PhaseEError):
    """Raised when a completed answer stage lacks valid contemporaneous evidence."""

    def __init__(self, message: str, condition: str | None = None):
        super().__init__(message)
        self.condition = condition


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseEError(f"Cannot read valid JSON from {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PhaseEError(
                        f"JSONL record {line_number} is not an object: {path}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseEError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    if not rows:
        raise PhaseEError(f"Required JSONL is empty: {path}")
    return rows


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (canonical_json(value) + suffix).encode("utf-8")


def canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def scientific_json_bytes(value: Any) -> bytes:
    """Serialize evaluator output exactly as the table-era CLI did.

    The historical evaluator used ``json.dump(..., indent=2)`` without key
    sorting and without a trailing newline.  Scientific result ordering is an
    evaluated contract; canonical sorted JSON remains appropriate for
    manifests, status, and hash ledgers only.
    """

    return json.dumps(value, indent=2, allow_nan=False).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if not SHA256_RE.fullmatch(normalized):
        raise PhaseEError(f"{label} must be exactly 64 hexadecimal characters")
    return normalized


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PhaseEError(f"git {' '.join(args)} failed: {detail}")
    return result


def validate_source_contract(
    contract: dict[str, Any], require_clean: bool
) -> dict[str, str]:
    if contract.get("schema_version") != 7 or contract.get("evaluator") != {"part_of_questions": 105}:
        raise PhaseEError("Unsupported frozen RAG contract")
    head = run_git("rev-parse", "HEAD").stdout.strip()
    if require_clean and run_git("status", "--porcelain", "--untracked-files=normal").stdout.strip():
        raise PhaseEError("RAG requires a clean source checkout")
    return {"head": head}


def validate_run_id(run_id: str, *, require_existing: bool = True) -> Path:
    if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise PhaseEError(
            "--run-id must be 1-64 characters, start with an alphanumeric, and "
            "contain only alphanumerics, '.', '_', or '-'"
        )
    declared_output = ROOT / "output"
    if OUTPUT_ROOT != declared_output:
        raise PhaseEError("output/ must be a physical directory, not a symlink or junction")
    run_dir = OUTPUT_ROOT / run_id
    if require_existing and not run_dir.is_dir():
        raise PhaseEError(f"Selected completed run does not exist: {run_dir}")
    if run_dir.exists() and run_dir.resolve() != run_dir:
        raise PhaseEError("Selected run root is a symlink, junction, or escaped path")
    return run_dir


def run_file(run_dir: Path, relative: str, *, required: bool = True) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise PhaseEError(f"Run-relative path is unsafe: {relative!r}")
    candidate = run_dir / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise PhaseEError(f"Run-relative path escapes the selected run: {relative}") from exc
    if required and (not candidate.is_file() or candidate.resolve() != candidate):
        raise PhaseEError(f"Required physical same-run file is missing: {relative}")
    return candidate


def _require_fields(value: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for field, required in expected.items():
        if value.get(field) != required:
            raise PhaseEError(f"{label}.{field} differs from the run contract")


def _manifest_hash(mapping: Any, relative: str, label: str) -> str:
    if not isinstance(mapping, dict):
        raise PhaseEError(f"{label} is not a hash mapping")
    value = mapping.get(relative)
    if not isinstance(value, str):
        raise PhaseEError(f"{label} does not bind {relative}")
    return validate_sha256(value, f"{label}.{relative}")


def _artifact_hash(document: dict[str, Any], relative: str, label: str) -> str:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise PhaseEError(f"{label} lacks an artifact ledger")
    matches = [row for row in artifacts if isinstance(row, dict) and row.get("path") == relative]
    if len(matches) != 1:
        raise PhaseEError(f"{label} does not uniquely bind {relative}")
    return validate_sha256(str(matches[0].get("sha256", "")), f"{label}.{relative}")


def _verified_file(
    run_dir: Path,
    relative: str,
    ledger: dict[str, dict[str, Any]],
    *,
    expected: str | None = None,
) -> Path:
    path = run_file(run_dir, relative)
    existing = ledger.get(relative)
    if existing is None:
        observed = sha256_file(path)
        ledger[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    else:
        observed = existing["sha256"]
        if path.stat().st_size != existing["bytes"] or sha256_file(path) != observed:
            raise PhaseEError(f"Same-run artifact changed during preflight: {relative}")
    if expected is not None and observed != validate_sha256(expected, f"{relative} expected hash"):
        raise PhaseEError(f"Same-run artifact hash differs from its producer: {relative}")
    return path


def _validate_training_checkpoint_identity(
    *,
    seed: int,
    train_manifest: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    training_config: dict[str, Any],
    split_seed: int,
    split: dict[str, Any],
    expected_inputs: dict[str, str],
) -> None:
    """Require the training manifest to bind every checkpoint dependency."""

    checkpoint_manifest_relative = f"checkpoints/seed-{seed}/checkpoint-manifest.json"
    checkpoint_blob_relative = f"checkpoints/seed-{seed}/checkpoint.pt"
    model_cache_relative = checkpoint_manifest.get("model_cache_manifest")
    model_cache_sha256 = checkpoint_manifest.get("model_cache_manifest_sha256")
    compatibility_relative = checkpoint_manifest.get("dataset_compatibility_report")
    if (
        model_cache_relative != HF_CACHE_MANIFEST_RELATIVE
        or not isinstance(model_cache_sha256, str)
        or SHA256_RE.fullmatch(model_cache_sha256) is None
    ):
        raise PhaseEError(f"seed-{seed} checkpoint lacks a valid training dependency")

    expected_outputs = {
        "checkpoint": checkpoint_blob_relative,
        "checkpoint_manifest": checkpoint_manifest_relative,
        "model_cache_manifest": model_cache_relative,
        "progress_log": f"logs/model-train-seed-{seed}.log",
        "restart_state": f"checkpoints/seed-{seed}/restart-state.pt",
        "training_summary": f"checkpoints/seed-{seed}/training-summary.json",
    }
    if compatibility_relative is not None:
        expected_outputs["dataset_compatibility_report"] = compatibility_relative
    expected_inputs = {
        **expected_inputs,
        "model_cache_manifest_sha256": model_cache_sha256,
    }
    fresh_config = isinstance(train_manifest.get("config"), str)
    recipe = training_config if fresh_config else train_manifest.get("recipe")
    outputs = train_manifest.get("outputs")
    training_inputs = train_manifest.get("inputs")
    resume = train_manifest.get("resume")
    train_split = train_manifest.get("split")
    if (
        not isinstance(recipe, dict)
        or recipe != training_config
        or (fresh_config and train_manifest.get("config") != checkpoint_manifest.get("config"))
        or (not fresh_config and (
            recipe.get("base_model") != checkpoint_manifest.get("base_model")
            or recipe.get("base_model_revision") != checkpoint_manifest.get("base_model_revision")))
        or outputs != expected_outputs
        or training_inputs != expected_inputs
        or not isinstance(resume, dict)
        or resume.get("restart_state_sha256")
        != checkpoint_manifest.get("restart_state_sha256")
        or (not fresh_config and (not isinstance(train_split, dict)
            or train_split.get("split_id") != "CODE-SPLIT-1"
            or train_split.get("seed") != split_seed
            or train_split.get("train_sentences") != split.get("train")
            or train_split.get("development_sentences") != split.get("development")
            or train_split.get("test_sentences") != split.get("test")))
    ):
        raise PhaseEError(f"seed-{seed} training and checkpoint identities disagree")


def load_verifier_model_identity(path: Path, expected_hash: str, *, run_dir: Path | None = None) -> dict[str, Any]:
    if sha256_file(path) != validate_sha256(expected_hash, "verifier environment hash"):
        raise PhaseEError("Verifier environment differs from its stage manifest")
    manifest = load_json(path)
    if not isinstance(manifest, dict) or manifest.get("execution_mode") != "live":
        raise PhaseEError("Verifier environment must describe a completed live verifier run")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("identity_verified") is not True:
        raise PhaseEError("Verifier environment lacks a verified model identity")
    runtime = None
    if "ollama_runtime" in model:
        from utils.verifier.runtime import load_runtime
        from utils.common.artifact_io import DataContractError
        if run_dir is None:
            raise PhaseEError("Verifier runtime evidence requires the selected run root")
        try:
            runtime = load_runtime(run_dir, model["ollama_runtime"])
            actual_model = runtime["model"]
            model = {**model, **actual_model,
                "registry_manifest_sha256": actual_model["tag_digest"],
                "model_blob_sha256": actual_model["blob_sha256"]}
        except (DataContractError, KeyError, TypeError) as exc:
            raise PhaseEError("Verifier shared Ollama runtime is invalid") from exc
    name = model.get("name")
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise PhaseEError("Verifier environment has an invalid model tag")
    tag = validate_sha256(str(model.get("tag_digest", "")).removeprefix("sha256:"), "tag")
    registry = validate_sha256(
        str(model.get("registry_manifest_sha256", "")).removeprefix("sha256:"),
        "registry",
    )
    blob = validate_sha256(str(model.get("blob_sha256", "")).removeprefix("sha256:"), "blob")
    recorded_blob = validate_sha256(
        str(model.get("model_blob_sha256", "")).removeprefix("sha256:"),
        "recorded blob",
    )
    if tag != registry or blob != recorded_blob:
        raise PhaseEError("Verifier environment contains conflicting model identities")
    details = model.get("details")
    normalized_details = {
        "family": str(details.get("family", "")).lower() if isinstance(details, dict) else "",
        "parameter_size": details.get("parameter_size") if isinstance(details, dict) else None,
        "quantization_level": details.get("quantization_level") if isinstance(details, dict) else None,
    }
    if any(not isinstance(item, str) or not item for item in normalized_details.values()):
        raise PhaseEError("Verifier environment has incomplete model details")
    identity = {
        "name": name,
        "tag_digest": tag,
        "blob_sha256": blob,
        "details": normalized_details,
        "verifier_environment_sha256": expected_hash,
        "verifier_condition_id": manifest.get("condition_id"),
        "verifier_protocol_id": PROTOCOL_ID if runtime is not None else manifest.get("protocol_id"),
    }
    if runtime is not None:
        identity["ollama_runtime"] = model["ollama_runtime"]
        identity["ollama_runtime_identity"] = runtime
    return identity


def require_consistent_encoder_identity(
    identities: list[dict[str, Any]],
) -> dict[str, str]:
    """Require one immutable base encoder identity throughout a run."""
    if not identities:
        raise PhaseEError("Selected run contains no encoder identity")
    for identity in identities:
        model = identity.get("base_model")
        revision = identity.get("base_model_revision")
        if (
            not isinstance(model, str)
            or not model
            or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        ):
            raise PhaseEError("Encoder identity is incomplete or mutable")
    if len({canonical_json(value) for value in identities}) != 1:
        raise PhaseEError("Encoder base-model digest differs across seeds in the same run")
    return identities[0]


def _reject_foreign_run_ids(value: Any, run_id: str, label: str) -> None:
    """Reject any explicit nested run identity that differs from the selected run."""

    if isinstance(value, dict):
        if "run_id" in value and value["run_id"] != run_id:
            raise PhaseEError(f"{label} carries foreign run_id {value['run_id']!r}")
        for child in value.values():
            _reject_foreign_run_ids(child, run_id, label)
    elif isinstance(value, list):
        for child in value:
            _reject_foreign_run_ids(child, run_id, label)


def _git_file_bytes(commit: str, relative: str) -> bytes:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise PhaseEError(f"Checkout source path is unsafe: {relative!r}")
    result = subprocess.run(
        ["git", "show", f"{commit}:{raw.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        raise PhaseEError(
            f"Checkout source path is not recoverable from Git: {relative}"
        )
    return result.stdout


def _load_authenticated_run_config(
    checkout: dict[str, Any], full: dict[str, Any], contract: dict[str, Any],
    *, run_dir: Path | None = None, ledger: dict | None = None,
) -> tuple[dict[str, Any], bool]:
    source = checkout.get("source")
    commit = source.get("commit") if isinstance(source, dict) else None
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise PhaseEError("Checkout manifest lacks an immutable source commit")
    # Older authenticated parents are read-only. Fresh runs carry their config
    # in checkout.source; older completed parents used a tracked-file inventory.
    legacy_compatibility = "tracked_artifact_sha256" in checkout
    if full.get("source_commit") != commit:
        raise PhaseEError("Full-run and checkout manifests name different source commits")
    config_path = full.get("config")
    config_hash = full.get("config_sha256")
    source_config = source.get("config")
    config_bytes = None
    if (isinstance(source_config, dict) and set(source_config) == {"path"}
            and isinstance(config_path, str) and source_config["path"] == config_path
            and config_hash is None):
        config_bytes = _git_file_bytes(commit, config_path)
    tracked = checkout.get("tracked_artifact_sha256")
    checkout_config_hash = (
        source_config.get("sha256")
        if isinstance(source_config, dict) and source_config.get("path") == config_path
        else tracked.get(config_path) if isinstance(tracked, dict) else None
    )
    if config_bytes is None and (
        not isinstance(config_path, str)
        or not isinstance(config_hash, str)
        or checkout_config_hash != config_hash
    ):
        raise PhaseEError("Full-run config is not bound by the checkout manifest")
    if config_bytes is not None:
        pass
    elif isinstance(source_config, dict) and source_config.get("run_local_path") and run_dir is not None:
        relative = source_config["run_local_path"]
        config_bytes = _verified_file(run_dir, relative, ledger if ledger is not None else {}, expected=config_hash).read_bytes()
    else:
        config_bytes = _git_file_bytes(commit, config_path)
    if config_hash is not None and sha256_bytes(config_bytes) != config_hash:
        raise PhaseEError("Full-run config differs from the recorded Git checkout")
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseEError("Authenticated run config is not valid UTF-8 JSON") from exc
    if not isinstance(config, dict):
        raise PhaseEError("Authenticated run config is not an object")
    return config, legacy_compatibility


def _validate_stage_seal(
    run_dir: Path,
    run_id: str,
    producer_relative: str,
    producer: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
    *,
    required: bool,
) -> None:
    producer_path = run_file(run_dir, producer_relative)
    if producer.get("receipt_version") == 1:
        checkout = _verified_file(run_dir, "manifests/00-checkout-manifest.json", ledger)
        if (producer.get("run_id") != run_id
                or producer.get("checkout_sha256") != sha256_file(checkout)):
            raise PhaseEError("Stage receipt differs from the selected checkout")
        return
    seal_relative = str(
        Path(producer_relative).with_name(
            f"same-run-{Path(producer_relative).name}"
        )
    ).replace("\\", "/")
    seal_path = run_file(run_dir, seal_relative, required=False)
    if not seal_path.is_file():
        if required:
            raise PhaseEError(f"New-run stage lacks same-run seal: {producer_relative}")
        return
    outputs: dict[str, str] = {}
    declared = producer.get("outputs")
    if isinstance(declared, dict):
        outputs = dict(declared)
    else:
        for field in ("generation_plan_output", "candidates_output"):
            relative = producer.get(field)
            if isinstance(relative, str):
                output_path = _verified_file(run_dir, relative, ledger)
                outputs[relative] = sha256_file(output_path)
        producer_inputs = producer.get("inputs")
        prediction = (
            producer_inputs.get("prediction_ledger")
            if isinstance(producer_inputs, dict)
            else None
        )
        if isinstance(prediction, dict) and isinstance(prediction.get("path"), str):
            outputs[prediction["path"]] = prediction.get("sha256")
    for relative, expected in outputs.items():
        _verified_file(run_dir, relative, ledger, expected=expected)
    seal = load_json(_verified_file(run_dir, seal_relative, ledger))
    expected_seal = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": run_id,
        "producer_manifest": {
            "path": producer_relative,
            "sha256": sha256_file(producer_path),
        },
        "outputs": outputs,
    }
    if seal != expected_seal:
        raise PhaseEError(f"Stage seal differs from producer: {producer_relative}")


def _verifier_decoding(config: dict[str, Any]) -> dict[str, Any]:
    verifier = config.get("verifier")
    if not isinstance(verifier, dict):
        raise PhaseEError("Authenticated config lacks verifier settings")
    return {
        "stream": verifier.get("stream"),
        "think": verifier.get("think"),
        "options": {
            "temperature": verifier.get("temperature"),
            "seed": verifier.get("seed"),
            "top_k": verifier.get("top_k"),
            "top_p": verifier.get("top_p"),
            "min_p": verifier.get("min_p"),
            "repeat_penalty": verifier.get("repeat_penalty"),
            "num_ctx": verifier.get("num_ctx"),
            "num_predict": verifier.get("num_predict"),
        },
    }


def _verifier_decoding_sha256(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_verifier_decoding(config)))


def _validate_candidate_generation(
    run_dir: Path,
    run_id: str,
    pipeline: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
    *,
    seed: int,
    split: str,
    checkpoint_manifest_relative: str,
    checkpoint_manifest_sha256: str,
    checkpoint_sha256: str,
    max_span_width: int,
    prepared_relative: str,
    prepared_sha256: str,
    expected_sentence_count: int,
    split_manifest_sha256: str,
    require_seal: bool,
) -> tuple[str, list[dict[str, Any]]]:
    directory = "dev" if split == "development" else "test"
    manifest_relative = (
        f"manifests/model-generate-candidates-live-seed-{seed}-{split}.json"
    )
    candidates_relative = f"predictions/{directory}/seed-{seed}-candidates.jsonl"
    ledger_relative = f"predictions/{directory}/seed-{seed}-prediction-ledger.jsonl"
    generation = load_json(_verified_file(run_dir, manifest_relative, ledger))
    _reject_foreign_run_ids(
        generation, run_id, f"seed-{seed} {split} candidate generation"
    )
    _require_fields(
        generation,
        {
            "stage": "model-generate-candidates",
            "status": "completed",
            "execution_mode": "live",
            "protocol_id": pipeline["protocol_id"],
            "matcher_id": pipeline["matcher_id"],
            "training_seed": seed,
            "candidates_output": candidates_relative,
        },
        f"seed-{seed} {split} candidate generation",
    )
    inputs = generation.get("inputs")
    if not isinstance(inputs, dict):
        raise PhaseEError(f"seed-{seed} {split} generation lacks inputs")
    checkpoint_input = inputs.get("checkpoint_manifest")
    prepared_input = inputs.get("prepared_sentences")
    prediction_input = inputs.get("prediction_ledger")
    if (
        not isinstance(checkpoint_input, dict)
        or checkpoint_input.get("path") != checkpoint_manifest_relative
        or checkpoint_input.get("sha256") != checkpoint_manifest_sha256
        or not isinstance(prepared_input, dict)
        or prepared_input.get("path") != prepared_relative
        or prepared_input.get("sha256") != prepared_sha256
        or not isinstance(prediction_input, dict)
        or prediction_input.get("path") != ledger_relative
        or generation.get("checkpoint", {}).get("sha256") != checkpoint_sha256
        or generation.get("checkpoint", {}).get("split_manifest_sha256")
        != split_manifest_sha256
    ):
        raise PhaseEError(
            f"seed-{seed} {split} generation is not bound to checkpoint/data"
        )
    prediction_path = _verified_file(
        run_dir,
        ledger_relative,
        ledger,
        expected=prediction_input.get("sha256"),
    )
    recovery_input = inputs.get("recovery_prediction_ledger")
    if recovery_input is not None:
        if (
            not isinstance(recovery_input, dict)
            or not isinstance(recovery_input.get("path"), str)
            or recovery_input.get("sha256") != prediction_input.get("sha256")
        ):
            raise PhaseEError(
                f"seed-{seed} {split} recovery ledger binding is malformed"
            )
        recovery_path = _verified_file(
            run_dir,
            recovery_input["path"],
            ledger,
            expected=recovery_input["sha256"],
        )
        provenance = inputs.get("recovery_prediction_ledger_provenance")
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("path"), str)
        ):
            raise PhaseEError(
                f"seed-{seed} {split} recovery provenance is missing"
            )
        provenance_path = _verified_file(
            run_dir,
            provenance["path"],
            ledger,
            expected=provenance.get("sha256"),
        )
        recovery_provenance = load_json(provenance_path)
        checkout_relative = "manifests/00-checkout-manifest.json"
        expected_provenance = {
            "schema_version": "phase-b-same-run-recovery-2.0",
            "run_id": run_id,
            "kind": "prediction-ledger",
            "identity": {"training_seed": seed, "split": split},
            "cache": {
                "path": recovery_input["path"],
                "sha256": sha256_file(recovery_path),
            },
            "checkout_manifest": {
                "path": checkout_relative,
                "sha256": ledger[checkout_relative]["sha256"],
            },
            "bindings": {
                "checkpoint_manifest": {
                    "path": checkpoint_manifest_relative,
                    "sha256": checkpoint_manifest_sha256,
                },
                "prepared": {
                    "path": prepared_relative,
                    "sha256": prepared_sha256,
                },
            },
        }
        if recovery_provenance != expected_provenance:
            raise PhaseEError(
                f"seed-{seed} {split} recovery provenance is not same-run"
            )
    candidates_path = _verified_file(run_dir, candidates_relative, ledger)
    rows = load_jsonl(candidates_path)
    input_hashes = {
        "checkpoint_manifest": checkpoint_manifest_sha256,
        "prepared_sentences": prepared_sha256,
        "prediction_ledger": sha256_file(prediction_path),
    }
    try:
        from utils.encoder.model import validate_prediction_artifacts

        validate_prediction_artifacts(
            sentences_path=run_file(run_dir, prepared_relative),
            prediction_ledger_path=prediction_path,
            candidates_path=candidates_path,
            training_seed=seed,
            max_span_width=max_span_width,
            input_hashes=input_hashes,
        )
    except (DataContractError, OSError, ValueError) as exc:
        raise PhaseEError(
            f"seed-{seed} {split} prediction/candidate replay is invalid: {exc}"
        ) from exc
    if (
        generation.get("candidate_count") != len(rows)
        or generation.get("sentence_count") != expected_sentence_count
        or generation.get("ledger_sentence_count") != expected_sentence_count
        or any(
            row.get("training_seed") != seed
            or row.get("input_hashes") != input_hashes
            for row in rows
        )
    ):
        raise PhaseEError(f"seed-{seed} {split} candidate ledger is inconsistent")
    _validate_stage_seal(
        run_dir,
        run_id,
        manifest_relative,
        generation,
        ledger,
        required=require_seal,
    )
    return candidates_relative, rows


def _validate_verifier_stage(
    run_dir: Path,
    run_id: str,
    ledger: dict[str, dict[str, Any]],
    artifacts: dict[str, str],
    pipeline: dict[str, Any],
    run_config: dict[str, Any],
    mode: str,
    *,
    require_seal: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    condition = "VER-" + mode.upper()
    manifest_relative = artifacts[f"{mode}_verifier_manifest"]
    manifest_path = _verified_file(run_dir, manifest_relative, ledger)
    manifest = load_json(manifest_path)
    _reject_foreign_run_ids(manifest, run_id, f"{mode} verifier manifest")
    _require_fields(
        manifest,
        {
            "condition_id": condition,
            "execution_mode": "live",
            "status": "completed",
            "protocol_id": pipeline["protocol_id"],
        },
        f"{mode} verifier manifest",
    )
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise PhaseEError(f"{mode} verifier manifest mappings are malformed")
    for relative, expected in {**inputs, **outputs}.items():
        _verified_file(run_dir, relative, ledger, expected=expected)
    for required in (artifacts["prepared_test"], artifacts["candidates"]):
        if _manifest_hash(inputs, required, f"{mode}.inputs") != ledger[required]["sha256"]:
            raise PhaseEError(f"{mode} verifier input differs from same-run data")
    verdict_relative = artifacts[f"{mode}_verdicts"]
    environment_relative = artifacts[f"{mode}_environment"]
    verdict_hash = _manifest_hash(outputs, verdict_relative, f"{mode}.outputs")
    environment_hash = _manifest_hash(outputs, environment_relative, f"{mode}.outputs")
    verdicts = load_jsonl(_verified_file(run_dir, verdict_relative, ledger, expected=verdict_hash))
    if manifest.get("verdict_count") != len(verdicts):
        raise PhaseEError(f"{mode} verifier manifest count differs from its ledger")
    environment_path = _verified_file(
        run_dir, environment_relative, ledger, expected=environment_hash
    )
    identity = load_verifier_model_identity(environment_path, environment_hash, run_dir=run_dir)
    if "ollama_runtime" in identity:
        reference = identity["ollama_runtime"]
        _verified_file(run_dir, reference["path"], ledger, expected=reference["sha256"])
    if identity["verifier_condition_id"] != condition:
        raise PhaseEError(f"{mode} environment carries another condition")
    model_manifest = validate_sha256(str(manifest.get("model_manifest_sha256", "")), "model manifest")
    if model_manifest != identity["tag_digest"]:
        raise PhaseEError(f"{mode} manifest and environment disagree about Qwen")
    if any(row.get("model_manifest_sha256") != model_manifest for row in verdicts):
        raise PhaseEError(f"{mode} verdicts contain a conflicting Qwen digest")
    verifier_config = run_config.get("verifier")
    if not isinstance(verifier_config, dict):
        raise PhaseEError("Authenticated run config lacks verifier settings")
    fresh_template = f"{verifier_config.get('prompt_revision')}:{mode}"
    fresh_identity = "template_id" in manifest
    if fresh_identity:
        if manifest.get("template_id") != fresh_template:
            raise PhaseEError(f"{mode} producer template differs from its authenticated config")
        prompt_sha = decoding_sha = None
    else:
        prompt_sha = validate_sha256(
            str(manifest.get("prompt_sha256", "")), f"{mode} producer prompt bundle")
        decoding_sha = _verifier_decoding_sha256(run_config)
        if manifest.get("decoding_sha256") != decoding_sha:
            raise PhaseEError(f"{mode} verifier decoding differs from run config")
    requests_relative = f"verifier/{mode}/requests.jsonl"
    requests_hash = _manifest_hash(outputs, requests_relative, f"{mode}.outputs")
    requests = load_jsonl(
        _verified_file(run_dir, requests_relative, ledger, expected=requests_hash)
    )
    candidate_rows = load_jsonl(run_file(run_dir, artifacts["candidates"]))
    candidates_by_id = {row.get("candidate_id"): row for row in candidate_rows}
    candidate_keys = [
        (row.get("training_seed"), row.get("candidate_id"))
        for row in candidate_rows
    ]
    request_keys = [(row.get("training_seed"), row.get("candidate_id")) for row in requests]
    requests_by_id = {row.get("candidate_id"): row for row in requests}
    verdict_keys = [(row.get("training_seed"), row.get("candidate_id")) for row in verdicts]
    expected_decoding = _verifier_decoding(run_config)
    def valid_request(row: dict[str, Any]) -> bool:
        candidate = candidates_by_id.get(row.get("candidate_id"))
        payload = row.get("payload")
        if not isinstance(candidate, dict) or not isinstance(payload, dict):
            return False
        candidate_sha = sha256_bytes(canonical_json_bytes(candidate))
        request_sha = sha256_bytes(canonical_json_bytes(payload))
        cache_identity = {"model_manifest_sha256": model_manifest, "mode": mode,
                          "candidate_sha256": candidate_sha}
        cache_identity.update({"request_sha256": request_sha} if fresh_identity else
            {"prompt_sha256": prompt_sha, "decoding_sha256": decoding_sha})
        return (
            row.get("candidate_sha256") == candidate_sha
            and row.get("request_sha256")
            == sha256_bytes(canonical_json_bytes(payload))
            and row.get("cache_key")
            == sha256_bytes(canonical_json_bytes(cache_identity))
            and payload.get("model") == identity["name"]
            and {key: payload.get(key) for key in expected_decoding}
            == expected_decoding
        )
    if (
        len(requests) != len(candidate_rows)
        or len(candidates_by_id) != len(candidate_rows)
        or request_keys != candidate_keys
        or verdict_keys != sorted(candidate_keys)
        or any(
            row.get("condition_id") != condition
            or row.get("protocol_id") != pipeline["protocol_id"]
            or (row.get("template_id") != fresh_template if fresh_identity else
                row.get("prompt_sha256") != prompt_sha or row.get("decoding_sha256") != decoding_sha)
            or row.get("model_manifest_sha256") != model_manifest
            or not valid_request(row)
            for row in requests
        )
        or any(
            row.get("condition_id") != condition
            or row.get("protocol_id") != pipeline["protocol_id"]
            or ((row.get("template_id") != fresh_template
                 or row.get("request_sha256") != requests_by_id.get(row.get("candidate_id"), {}).get("request_sha256")) if fresh_identity else
                row.get("prompt_sha256") != prompt_sha or row.get("decoding_sha256") != decoding_sha)
            for row in verdicts
        )
    ):
        raise PhaseEError(f"{mode} verifier request/verdict lineage is inconsistent")
    model_inputs = {
        path: digest
        for path, digest in inputs.items()
        if isinstance(path, str) and path.startswith("inputs/ollama/blobs/sha256-")
    }
    if len(model_inputs) != 1:
        raise PhaseEError(f"{mode} verifier must bind exactly one run-local model blob")
    model_path, model_digest = next(iter(model_inputs.items()))
    if model_path.removeprefix("inputs/ollama/blobs/sha256-") != model_digest:
        raise PhaseEError(f"{mode} verifier model path and hash disagree")
    tags_relative = f"verifier/{mode}/model/tags.json"
    show_relative = f"verifier/{mode}/model/show.json"
    modelfile_relative = f"verifier/{mode}/model/ollama-modelfile.txt"
    environment_document = load_json(environment_path)
    environment_model = environment_document.get("model")
    if not isinstance(environment_model, dict):
        raise PhaseEError(f"{mode} verifier environment lacks model evidence")
    tags_path = _verified_file(run_dir, tags_relative, ledger)
    show_path = _verified_file(run_dir, show_relative, ledger)
    if (
        environment_model.get("tags_response_sha256") != sha256_file(tags_path)
        or environment_model.get("show_response_sha256") != sha256_file(show_path)
    ):
        raise PhaseEError(f"{mode} verifier raw model evidence hashes disagree")
    _validate_stored_model_response(
        load_json(tags_path), load_json(show_path), identity
    )
    if "ollama_runtime" not in identity:
        modelfile_path = _verified_file(run_dir, modelfile_relative, ledger)
        try:
            cli_modelfile = modelfile_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise PhaseEError(f"{mode} verifier CLI Modelfile is unreadable") from exc
        if identity["blob_sha256"] not in cli_modelfile:
            raise PhaseEError(f"{mode} verifier CLI Modelfile names another blob")
    recovery_prefix = f"inputs/recovery/verifier-{mode}/"
    recovery_cache_paths = [
        path
        for path in inputs
        if path.startswith(recovery_prefix) and path.endswith("-responses.jsonl")
    ]
    recovery_provenance_paths = [
        path
        for path in inputs
        if path.startswith(recovery_prefix)
        and path.endswith("-responses.jsonl.manifest.json")
    ]
    if recovery_cache_paths or recovery_provenance_paths:
        if len(recovery_cache_paths) != 1 or len(recovery_provenance_paths) != 1:
            raise PhaseEError(f"{mode} verifier recovery evidence is incomplete")
        recovery_relative = recovery_cache_paths[0]
        provenance_relative = recovery_provenance_paths[0]
        if provenance_relative != recovery_relative + ".manifest.json":
            raise PhaseEError(f"{mode} verifier recovery evidence paths disagree")
        recovery_provenance = load_json(run_file(run_dir, provenance_relative))
        expected_provenance = {
            "schema_version": "phase-b-same-run-recovery-2.0",
            "run_id": run_id,
            "kind": "verifier-responses",
            "identity": {"mode": mode, "condition_id": condition},
            "cache": {
                "path": recovery_relative,
                "sha256": inputs[recovery_relative],
            },
            "checkout_manifest": {
                "path": "manifests/00-checkout-manifest.json",
                "sha256": ledger["manifests/00-checkout-manifest.json"]["sha256"],
            },
            "bindings": {
                "candidates": {
                    "path": artifacts["candidates"],
                    "sha256": ledger[artifacts["candidates"]]["sha256"],
                },
                "sentences": {
                    "path": artifacts["prepared_test"],
                    "sha256": ledger[artifacts["prepared_test"]]["sha256"],
                },
            },
        }
        if recovery_provenance != expected_provenance:
            raise PhaseEError(f"{mode} verifier recovery provenance is not same-run")
    responses_relative = f"verifier/{mode}/responses.jsonl"
    responses_hash = _manifest_hash(outputs, responses_relative, f"{mode}.outputs")
    responses_path = _verified_file(
        run_dir, responses_relative, ledger, expected=responses_hash
    )
    try:
        from utils.verifier.verifier import validate_verifier_artifacts

        validate_verifier_artifacts(
            mode=mode,
            sentences_path=run_file(run_dir, artifacts["prepared_test"]),
            candidates_path=run_file(run_dir, artifacts["candidates"]),
            requests_path=run_file(run_dir, requests_relative),
            responses_path=responses_path,
            verdicts_path=run_file(run_dir, verdict_relative),
        )
    except (DataContractError, OSError, ValueError) as exc:
        raise PhaseEError(
            f"{mode} verifier response/verdict replay is invalid: {exc}"
        ) from exc
    _validate_stage_seal(
        run_dir,
        run_id,
        manifest_relative,
        manifest,
        ledger,
        required=require_seal,
    )
    return manifest, {**identity, "model_path": model_path}, verdicts


def validate_parent_lineage(
    run_dir: Path, run_id: str, contract: dict[str, Any]
) -> dict[str, Any]:
    pipeline = contract["pipeline"]
    paths = dict(pipeline["artifacts"])
    ledger: dict[str, dict[str, Any]] = {}

    checkout = load_json(_verified_file(run_dir, paths["checkout_manifest"], ledger))
    _reject_foreign_run_ids(checkout, run_id, "checkout manifest")
    if checkout.get("schema_version") not in {
        "phase-b-checkout-manifest-1.0",
        "phase-b-checkout-manifest-2.0",
    }:
        raise PhaseEError("Checkout manifest schema is unsupported")
    _require_fields(
        checkout,
        {
            "status": "pass",
            "run_id": run_id,
            "protocol_id": pipeline["protocol_id"],
            "workflow_id": pipeline["workflow_id"],
            "matcher_id": pipeline["matcher_id"],
        },
        "checkout manifest",
    )
    if checkout.get("source", {}).get("worktree_clean") is not True:
        raise PhaseEError("Selected run was not created from a clean source checkout")

    if (not run_file(run_dir, paths["full_run_manifest"], required=False).exists()
            and checkout.get("lifecycle_version") is None):
        paths["full_run_manifest"] = "manifests/debug-full-run.json"
    full = load_json(_verified_file(run_dir, paths["full_run_manifest"], ledger))
    _reject_foreign_run_ids(full, run_id, "full-run manifest")
    _require_fields(
        full,
        ({"run_id": run_id} if full.get("schema_version") == "pipeline-run-2" else {
            "run_id": run_id,
            "protocol_id": pipeline["protocol_id"],
            "workflow_id": pipeline["workflow_id"],
            "training_seeds": pipeline["training_seeds"],
        }),
        "full-run manifest",
    )
    if full.get("schema_version") not in {"pipeline-run-1", "pipeline-run-2", "phase-b-debug-full-run-1.0"}:
        raise PhaseEError("Full-run manifest schema is unsupported")
    run_config, legacy_compatibility = _load_authenticated_run_config(
        checkout, full, contract, run_dir=run_dir, ledger=ledger
    )
    for field, expected in {
        "protocol_id": pipeline["protocol_id"],
        "workflow_id": pipeline["workflow_id"],
        "matcher_id": pipeline["matcher_id"],
        "training_seeds": pipeline["training_seeds"],
    }.items():
        if run_config.get(field) != expected:
            raise PhaseEError(f"Authenticated run config differs at {field}")
    dataset_config = run_config.get("dataset")
    training_config = run_config.get("training")
    if (
        not isinstance(dataset_config, dict)
        or dataset_config.get("dataset_id") != pipeline["dataset_id"]
    ):
        raise PhaseEError("Authenticated run config names another dataset")
    if not isinstance(training_config, dict):
        raise PhaseEError("Authenticated run config lacks training settings")

    acquisition_relative = "manifests/02-input-acquisition-manifest.json"
    acquisition_path = _verified_file(run_dir, acquisition_relative, ledger)
    acquisition = load_json(acquisition_path)
    _reject_foreign_run_ids(acquisition, run_id, "acquisition manifest")
    _require_fields(
        acquisition,
        {
            "schema_version": "phase-b-input-acquisition-manifest-1.0",
            "dataset_id": pipeline["dataset_id"],
        },
        "acquisition manifest",
    )
    archive = acquisition.get("archive")
    if not isinstance(archive, dict):
        raise PhaseEError("Acquisition manifest lacks its immutable archive")
    archive_path = archive.get("path")
    archive_sha = archive.get("sha256")
    if not isinstance(archive_path, str) or not isinstance(archive_sha, str):
        raise PhaseEError("Acquisition archive identity is incomplete")
    _verified_file(run_dir, archive_path, ledger, expected=archive_sha)

    preparation = load_json(_verified_file(run_dir, paths["preparation_manifest"], ledger))
    _reject_foreign_run_ids(preparation, run_id, "preparation manifest")
    _require_fields(
        preparation,
        {
            "schema_version": "phase-b-data-preparation-manifest-2.0",
            "protocol_id": pipeline["protocol_id"],
            "dataset_id": pipeline["dataset_id"],
            "byte_identical_independent_materializations": True,
        },
        "preparation manifest",
    )
    if (
        preparation.get("acquisition_manifest_sha256")
        != ledger[acquisition_relative]["sha256"]
        or preparation.get("archive_sha256") != validate_sha256(archive_sha, "archive")
    ):
        raise PhaseEError("Preparation is not bound to the acquired same-run archive")
    split = preparation.get("split")
    if not isinstance(split, dict) or split.get("seed") != pipeline["split_seed"] or split.get("test") != pipeline["test_records"]:
        raise PhaseEError("Preparation split differs from the frozen table method")
    prep_names = {
        "prepared_train": "train.jsonl",
        "prepared_development": "development.jsonl",
        "prepared_test": "test.jsonl",
        "private_gold": "test-gold.jsonl",
        "split_manifest": "split-manifest.json",
        "development_gold": "development-gold.jsonl",
    }
    for key, basename in prep_names.items():
        _verified_file(
            run_dir,
            paths[key],
            ledger,
            expected=_artifact_hash(preparation, basename, "preparation manifest"),
        )
    split_manifest = load_json(run_file(run_dir, paths["split_manifest"]))
    _reject_foreign_run_ids(split_manifest, run_id, "split manifest")
    if (
        split_manifest.get("split_id") != "CODE-SPLIT-1"
        or split_manifest.get("seed") != pipeline["split_seed"]
        or split_manifest.get("train_count") != split.get("train")
        or split_manifest.get("development_count") != split.get("development")
        or split_manifest.get("test_count") != split.get("test")
        or len(split_manifest.get("train_ids", [])) != split.get("train")
        or len(split_manifest.get("development_ids", [])) != split.get("development")
        or len(split_manifest.get("test_ids", [])) != split.get("test")
    ):
        raise PhaseEError("Prepared split manifest differs from the frozen split")

    score_path = _verified_file(run_dir, paths["score_manifest"], ledger)
    score = load_json(score_path)
    _reject_foreign_run_ids(score, run_id, "score manifest")
    _require_fields(
        score,
        {
            "schema_version": "phase-b-score-manifest-1.0",
            "run_id": run_id,
            "protocol_id": pipeline["protocol_id"],
            "workflow_id": pipeline["workflow_id"],
            "matcher_id": pipeline["matcher_id"],
            "nonpublication_smoke": False,
        },
        "score manifest",
    )
    score_inputs = score.get("inputs")
    score_outputs = score.get("outputs")
    if not isinstance(score_inputs, dict) or not isinstance(score_outputs, dict):
        raise PhaseEError("Score manifest input/output mappings are malformed")
    for relative, expected in {**score_inputs, **score_outputs}.items():
        _verified_file(run_dir, relative, ledger, expected=expected)
    for key in ("private_gold", "threshold_selection", "candidates", "simple_verdicts", "corrective_verdicts"):
        _manifest_hash(score_inputs, paths[key], "score.inputs")
    for key in ("sentence_outcomes", "metrics"):
        _manifest_hash(score_outputs, paths[key], "score.outputs")

    threshold = load_json(_verified_file(
        run_dir,
        paths["threshold_selection"],
        ledger,
        expected=_manifest_hash(score_inputs, paths["threshold_selection"], "score.inputs"),
    ))
    _reject_foreign_run_ids(threshold, run_id, "threshold selection")
    selected_threshold = threshold.get("selected_threshold")
    threshold_contract = run_config.get("threshold_selection")
    if (
        not isinstance(threshold_contract, dict)
        or
        isinstance(selected_threshold, bool)
        or not isinstance(selected_threshold, (int, float))
        or not 0 <= float(selected_threshold) <= 1
        or threshold.get("selection_split") != threshold_contract.get("selection_split")
        or threshold.get("objective") != threshold_contract.get("objective")
        or threshold.get("tie_rule") != threshold_contract.get("tie_rule")
        or threshold.get("grid") != threshold_contract.get("grid")
        or threshold.get("used_test_labels") is not False
        or threshold.get("split_manifest_sha256") != ledger[paths["split_manifest"]]["sha256"]
        or threshold.get("development_gold_sha256") != ledger[paths["development_gold"]]["sha256"]
    ):
        raise PhaseEError("Threshold selection is not bound to development-only same-run data")
    development_index = _verified_file(run_dir, paths["development_candidate_index"], ledger)
    if threshold.get("development_candidate_index_sha256") != ledger[paths["development_candidate_index"]]["sha256"]:
        raise PhaseEError("Threshold selection differs from the same-run development candidates")
    threshold_rows = threshold.get("per_threshold")
    if (
        not isinstance(threshold_rows, list)
        or [row.get("threshold") for row in threshold_rows if isinstance(row, dict)]
        != threshold_contract.get("grid")
    ):
        raise PhaseEError("Threshold statistics do not cover the frozen grid")
    try:
        recorded_argmax = max(
            (row["mean_per_seed_development_strict_triple_f1"], row["threshold"])
            for row in threshold_rows
        )[1]
    except (KeyError, TypeError, ValueError) as exc:
        raise PhaseEError("Threshold statistics are malformed") from exc
    if selected_threshold != recorded_argmax:
        raise PhaseEError("Selected threshold differs from the recorded development argmax")

    prepared_rows = load_jsonl(run_file(run_dir, paths["prepared_test"]))
    gold_rows = load_jsonl(run_file(run_dir, paths["private_gold"]))
    if len(prepared_rows) != pipeline["test_records"] or len(gold_rows) != pipeline["test_records"]:
        raise PhaseEError("Prepared test and private gold have the wrong row count")

    encoder_identities = []
    checkpoint_manifest_hashes = {}
    hardware = None
    generation_rows: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "test": [],
    }
    generation_files: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "test": [],
    }
    split_manifest_sha = ledger[paths["split_manifest"]]["sha256"]
    for seed in pipeline["training_seeds"]:
        checkpoint_manifest_relative = paths["checkpoint_manifest"].format(seed=seed)
        checkpoint_blob_relative = paths["checkpoint_blob"].format(seed=seed)
        checkpoint_manifest_path = _verified_file(run_dir, checkpoint_manifest_relative, ledger)
        checkpoint_manifest = load_json(checkpoint_manifest_path)
        _reject_foreign_run_ids(
            checkpoint_manifest, run_id, f"seed-{seed} checkpoint manifest"
        )
        _require_fields(
            checkpoint_manifest,
            {
                "protocol_id": pipeline["protocol_id"],
                "training_seed": seed,
            },
            f"seed-{seed} checkpoint manifest",
        )
        if checkpoint_manifest.get("schema_version") not in {
            "phase-b-model-checkpoint-manifest-2.0",
            "phase-b-model-checkpoint-manifest-3.0",
            "phase-b-model-checkpoint-manifest-4.0",
            "encoder-checkpoint-1",
        }:
            raise PhaseEError(f"seed-{seed} checkpoint manifest schema is unsupported")
        checkpoint_digest = validate_sha256(
            str(checkpoint_manifest.get("checkpoint_sha256", "")),
            f"seed-{seed} checkpoint",
        )
        _verified_file(run_dir, checkpoint_blob_relative, ledger, expected=checkpoint_digest)
        if (
            checkpoint_manifest.get("split_id") != "CODE-SPLIT-1"
            or checkpoint_manifest.get("split_manifest_sha256") != split_manifest_sha
            or ("config_sha256" in full and any(checkpoint_manifest.get(field) != training_config.get(field)
                for field in ("base_model", "base_model_revision", "max_span_width", "context_between_spans")))
            or checkpoint_manifest.get("archive_sha256") != archive_sha
            or checkpoint_manifest.get("acquisition_manifest_sha256")
            != ledger[acquisition_relative]["sha256"]
            or checkpoint_manifest.get("train_jsonl_sha256")
            != ledger[paths["prepared_train"]]["sha256"]
            or checkpoint_manifest.get("development_jsonl_sha256")
            != ledger[paths["prepared_development"]]["sha256"]
            or (checkpoint_manifest.get("config_sha256") != full["config_sha256"]
                if "config_sha256" in full else checkpoint_manifest.get("config") != full.get("config"))
            or checkpoint_manifest.get("source_commit")
            != checkout["source"]["commit"]
        ):
            raise PhaseEError(
                f"seed-{seed} checkpoint is not transitively bound to checkout/data/config"
            )
        restart_relative = f"checkpoints/seed-{seed}/restart-state.pt"
        summary_relative = f"checkpoints/seed-{seed}/training-summary.json"
        progress_relative = f"logs/model-train-seed-{seed}.log"
        compatibility_relative = checkpoint_manifest.get(
            "dataset_compatibility_report"
        )
        if checkpoint_manifest.get("schema_version") != "encoder-checkpoint-1" and not isinstance(compatibility_relative, str):
            raise PhaseEError(f"seed-{seed} checkpoint lacks compatibility evidence")
        _verified_file(
            run_dir,
            restart_relative,
            ledger,
            expected=checkpoint_manifest.get("restart_state_sha256"),
        )
        if compatibility_relative is not None:
            compatibility_path = _verified_file(
                run_dir,
                compatibility_relative,
                ledger,
                expected=checkpoint_manifest.get("dataset_compatibility_sha256"),
            )
        model_cache_relative = checkpoint_manifest.get("model_cache_manifest")
        model_cache_sha256 = checkpoint_manifest.get("model_cache_manifest_sha256")
        if (
            model_cache_relative != HF_CACHE_MANIFEST_RELATIVE
            or not isinstance(model_cache_sha256, str)
        ):
            raise PhaseEError(f"seed-{seed} checkpoint lacks model-cache evidence")
        _verified_file(
            run_dir,
            model_cache_relative,
            ledger,
            expected=model_cache_sha256,
        )
        _verified_file(run_dir, progress_relative, ledger)
        summary = load_json(_verified_file(run_dir, summary_relative, ledger))
        if isinstance(summary.get("runtime_environment"), dict):
            ref = summary["runtime_environment"]
            runtime_path = _verified_file(run_dir, ref["path"], ledger, expected=ref["sha256"])
            summary["environment"] = load_json(runtime_path)
        selected_metrics = summary.get("selected_metrics")
        if (
            summary.get("status") != "completed"
            or (summary.get("canonical_mode") is not True
                if full.get("schema_version") != "pipeline-run-2" else
                summary.get("completed_steps") != training_config.get("max_steps"))
            or summary.get("seed") != seed
            or summary.get("test_evaluated") is not False
            or (full.get("schema_version") != "pipeline-run-2" and (
                summary.get("model_name") != checkpoint_manifest.get("base_model")
                or summary.get("model_revision") != checkpoint_manifest.get("base_model_revision")
                or summary.get("selection_metric") != "triple_f1"))
            or summary.get("selected_step")
            != checkpoint_manifest.get("checkpoint_step")
            or not isinstance(selected_metrics, dict)
            or selected_metrics.get("triple_f1")
            != checkpoint_manifest.get("selected_metric_value")
        ):
            raise PhaseEError(f"seed-{seed} training summary is not publication-safe")
        if compatibility_relative is not None:
            compatibility = load_json(compatibility_path)
            fresh_clone_dataset = compatibility.get("fresh_clone_dataset")
            if (
                compatibility.get("status")
                != "partial_match_full_legacy_equivalence_unavailable"
                or not isinstance(fresh_clone_dataset, dict)
                or fresh_clone_dataset.get("dataset_id")
                != pipeline["dataset_id"]
                or fresh_clone_dataset.get("archive_sha256")
                != archive_sha
                or fresh_clone_dataset.get("prepared_dataset_tree_sha256")
                != preparation.get("dataset_tree_sha256")
            ):
                raise PhaseEError(
                    f"seed-{seed} dataset compatibility evidence differs from preparation"
                )
        checkpoint_manifest_hashes[seed] = ledger[checkpoint_manifest_relative]["sha256"]
        encoder_identities.append({
            "base_model": training_config["base_model"],
            "base_model_revision": training_config["base_model_revision"],
        })
        train_manifest = load_json(_verified_file(
            run_dir, f"manifests/model-train-live-seed-{seed}.json", ledger
        ))
        _reject_foreign_run_ids(
            train_manifest, run_id, f"seed-{seed} training manifest"
        )
        _require_fields(
            train_manifest,
            {
                "stage": "model-train",
                "execution_mode": "live",
                "status": "completed",
                "protocol_id": pipeline["protocol_id"],
                "matcher_id": pipeline["matcher_id"],
                "training_seed": seed,
                "expected_checkpoint_dir": f"checkpoints/seed-{seed}",
                "final_test_selection_forbidden": True,
                **({"config": full["config"]} if full.get("schema_version") == "pipeline-run-2" else
                   {"recipe": training_config}),
            },
            f"seed-{seed} training manifest",
        )
        expected_training_inputs = {
            "acquisition_manifest_sha256": ledger[acquisition_relative]["sha256"],
            "checkout_manifest_sha256": ledger[paths["checkout_manifest"]]["sha256"],
            "development_jsonl_sha256": ledger[paths["prepared_development"]]["sha256"],
            "preparation_manifest_sha256": ledger[paths["preparation_manifest"]]["sha256"],
            "split_manifest_sha256": split_manifest_sha,
            "train_jsonl_sha256": ledger[paths["prepared_train"]]["sha256"],
        }
        _validate_training_checkpoint_identity(
            seed=seed,
            train_manifest=train_manifest,
            checkpoint_manifest=checkpoint_manifest,
            training_config=training_config,
            split_seed=pipeline["split_seed"],
            split=split,
            expected_inputs=expected_training_inputs,
        )
        if hardware is None:
            hardware = train_manifest.get("environment")
        for split_name, prepared_key, count_key in (
            ("development", "prepared_development", "development"),
            ("test", "prepared_test", "test"),
        ):
            relative, rows = _validate_candidate_generation(
                run_dir,
                run_id,
                pipeline,
                ledger,
                seed=seed,
                split=split_name,
                checkpoint_manifest_relative=checkpoint_manifest_relative,
                checkpoint_manifest_sha256=checkpoint_manifest_hashes[seed],
                checkpoint_sha256=checkpoint_digest,
                max_span_width=checkpoint_manifest["max_span_width"],
                prepared_relative=paths[prepared_key],
                prepared_sha256=ledger[paths[prepared_key]]["sha256"],
                expected_sentence_count=split[count_key],
                split_manifest_sha256=split_manifest_sha,
                require_seal=not legacy_compatibility,
            )
            generation_rows[split_name].extend(rows)
            generation_files[split_name].append(
                {
                    "training_seed": seed,
                    "path": relative,
                    "sha256": ledger[relative]["sha256"],
                    "candidate_count": len(rows),
                    "candidates": [
                        {
                            "candidate_id": row.get("candidate_id"),
                            "example_id": row.get("example_id"),
                        }
                        for row in rows
                    ],
                }
            )
    encoder_identity = require_consistent_encoder_identity(encoder_identities)

    for split_name, combined_relative in (
        ("development", "predictions/dev/development-candidates.jsonl"),
        ("test", paths["candidates"]),
    ):
        combined_path = _verified_file(run_dir, combined_relative, ledger)
        expected_bytes = b"".join(
            run_file(run_dir, item["path"]).read_bytes()
            for item in generation_files[split_name]
        )
        if combined_path.read_bytes() != expected_bytes:
            raise PhaseEError(
                f"Combined {split_name} candidates differ from ordered seed ledgers"
            )
    expected_index = {
        "schema_version": "phase-b-candidate-index-1.0",
        "protocol_id": pipeline["protocol_id"],
        "workflow_id": pipeline["workflow_id"],
        "split_id": "CODE-SPLIT-1:development",
        "split_manifest_sha256": split_manifest_sha,
        "files": generation_files["development"],
        "candidate_count": len(generation_rows["development"]),
    }
    if load_json(development_index) != expected_index:
        raise PhaseEError("Development candidate index differs from seed ledgers")
    candidate_rows = load_jsonl(run_file(run_dir, paths["candidates"]))
    if candidate_rows != generation_rows["test"]:
        raise PhaseEError("Test candidate ledger differs from per-seed generation")
    for row in candidate_rows:
        seed = row.get("training_seed")
        if seed not in checkpoint_manifest_hashes:
            raise PhaseEError("Candidate ledger contains a foreign training seed")
        inputs = row.get("input_hashes")
        if (
            not isinstance(inputs, dict)
            or inputs.get("checkpoint_manifest") != checkpoint_manifest_hashes[seed]
            or inputs.get("prepared_sentences") != ledger[paths["prepared_test"]]["sha256"]
        ):
            raise PhaseEError("Candidate ledger contains mixed encoder/data lineage")

    simple_manifest, simple_model, simple_rows = _validate_verifier_stage(
        run_dir,
        run_id,
        ledger,
        paths,
        pipeline,
        run_config,
        "simple",
        require_seal=not legacy_compatibility,
    )
    corrective_manifest, corrective_model, corrective_rows = _validate_verifier_stage(
        run_dir,
        run_id,
        ledger,
        paths,
        pipeline,
        run_config,
        "corrective",
        require_seal=not legacy_compatibility,
    )
    qwen_fields = ("name", "tag_digest", "blob_sha256", "details", "model_path")
    if any(simple_model[field] != corrective_model[field] for field in qwen_fields):
        raise PhaseEError("Simple and corrective stages use different Qwen identities")
    if simple_model.get("ollama_runtime") != corrective_model.get("ollama_runtime"):
        raise PhaseEError("Simple and corrective stages use different Ollama server runtimes")
    if full.get("schema_version") == "pipeline-run-2":
        if full.get("ollama_runtime") != corrective_model.get("ollama_runtime"):
            raise PhaseEError("Full-run manifest and verifier stages disagree about Qwen runtime")
        reference = full["ollama_runtime"]
        _verified_file(run_dir, reference["path"], ledger, expected=reference["sha256"])
    elif (
        full.get("ollama_model") != corrective_model["name"]
        or full.get("model_blob_sha256") != corrective_model["blob_sha256"]
        or full.get("run_local_model_blob") != corrective_model["model_path"]
    ):
        raise PhaseEError("Full-run manifest and verifier stages disagree about Qwen")
    _verified_file(
        run_dir,
        corrective_model["model_path"],
        ledger,
        expected=corrective_model["blob_sha256"],
    )

    artifacts_list = [ledger[key] for key in sorted(ledger)]
    artifact_set = sha256_bytes(canonical_json_bytes(artifacts_list, newline=False))
    observed_profile = {
        "hardware": hardware if isinstance(hardware, dict) else {},
        "encoder": encoder_identity,
        "qwen": {key: corrective_model[key] for key in ("name", "tag_digest", "blob_sha256", "details")},
    }
    return {
        "run_id": run_id,
        "protocol_id": pipeline["protocol_id"],
        "workflow_id": pipeline["workflow_id"],
        "matcher_id": pipeline["matcher_id"],
        "selected_threshold": float(selected_threshold),
        "artifact_set_sha256": artifact_set,
        "artifacts": artifacts_list,
        "score_manifest_sha256": ledger[paths["score_manifest"]]["sha256"],
        "prepared_rows": prepared_rows,
        "gold_rows": gold_rows,
        "candidate_rows": candidate_rows,
        "corrective_rows": corrective_rows,
        "model_identity": corrective_model,
        "observed_profile": observed_profile,

    }


def project_evaluator_records(
    prepared_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    count = contract["pipeline"]["test_records"]
    if len(prepared_rows) != count or len(gold_rows) != count:
        raise PhaseEError("Same-run prepared test and gold row counts differ")
    gold_by_id = {row.get("example_id"): row for row in gold_rows}
    prepared_ids = [row.get("example_id") for row in prepared_rows]
    if (
        len(gold_by_id) != len(gold_rows)
        or None in gold_by_id
        or len(set(prepared_ids)) != len(prepared_ids)
        or set(prepared_ids) != set(gold_by_id)
        or set(prepared_ids) != set(split_manifest.get("test_ids", []))
    ):
        raise PhaseEError("Prepared text, private gold, and split identities differ")
    projected = []
    for prepared in prepared_rows:
        example_id = prepared["example_id"]
        triples = []
        for triple in gold_by_id[example_id].get("gold_triples", []):
            try:
                triples.append({
                    "head_text": triple["head"]["text"],
                    "tail_text": triple["tail"]["text"],
                    "relation": triple["relation"],
                })
            except (KeyError, TypeError) as exc:
                raise PhaseEError(f"Private gold triple is malformed for {example_id}") from exc
        projected.append({
            "doc_id": example_id,
            "sentence": prepared["content"],
            "gold_triples": triples,
            "predicted_triples": [],
        })
    expected_count = contract["evaluator"]["part_of_questions"]
    questions = evaluator.generate_questions(projected)
    if len(questions) != expected_count:
        raise PhaseEError(
            f"Projection does not generate exactly {expected_count} approved questions"
        )
    summary = {
        "records": len(projected),
        "questions": len(questions),
        "question_contract": [{"q": row["question"], "gold": row["gold_answer"]} for row in questions],
        "question_sha256": sha256_bytes(
            canonical_json_bytes([row["question"] for row in questions])
        ),
    }
    return projected, canonical_jsonl_bytes(projected), summary


def validate_environment(required_bytes: int) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise PhaseEError("Phase E requires Python 3.10 or newer")
    curl = shutil.which("curl")
    if not curl:
        raise PhaseEError("curl is required by the frozen evaluator")
    storage = shutil.disk_usage(ROOT)
    required_free = max(100 * 1024 * 1024, required_bytes * 4)
    if storage.free < required_free:
        raise PhaseEError("Insufficient free storage for Phase E outputs")
    return {
        "python": platform.python_version(),
        "curl": str(Path(curl).resolve()),
        "free_bytes": storage.free,
        "required_free_bytes": required_free,
    }


def preflight_same_run(
    run_id: str, contract: dict[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, bytes], dict[str, dict[str, Any]]]:
    run_dir = validate_run_id(run_id)
    parent = validate_parent_lineage(run_dir, run_id, contract)
    paths = contract["pipeline"]["artifacts"]
    split_manifest = load_json(run_file(run_dir, paths["split_manifest"]))
    records, record_bytes, record_summary = project_evaluator_records(
        parent["prepared_rows"], parent["gold_rows"], split_manifest, contract
    )
    graphs = build_table_graphs(
        run_id=run_id,
        prepared_rows=parent["prepared_rows"],
        gold_rows=parent["gold_rows"],
        candidate_rows=parent["candidate_rows"],
        corrective_rows=parent["corrective_rows"],
        selected_threshold=parent["selected_threshold"],
        seed=contract["pipeline"]["table2_seed"],
        prepared_sha256=sha256_file(run_file(run_dir, paths["prepared_test"])),
        split_sha256=sha256_file(run_file(run_dir, paths["split_manifest"])),
        gold_sha256=sha256_file(run_file(run_dir, paths["private_gold"])),
        candidate_sha256=sha256_file(run_file(run_dir, paths["candidates"])),
        corrective_sha256=sha256_file(run_file(run_dir, paths["corrective_verdicts"])),
    )
    projections = {"records": record_bytes}
    graph_summaries = {}
    seed = contract["pipeline"]["table2_seed"]
    for name, snapshot in graphs.items():
        relative = contract["pipeline"]["existing_graphs"][name].format(seed=seed)
        existing_path = run_file(run_dir, relative, required=False)
        source = "constructed"
        source_hash = sha256_bytes(canonical_json_bytes(snapshot))
        if existing_path.is_file():
            if existing_path.resolve() != existing_path:
                raise PhaseEError(f"Existing graph traverses a link: {relative}")
            existing = load_json(existing_path)
            if canonical_json(existing) != canonical_json(snapshot):
                raise PhaseEError(f"Existing graph differs from same-run reconstruction: {name}")
            source = "existing"
            source_hash = sha256_file(existing_path)
        graph = project_graph(snapshot)
        projections[name] = canonical_json_bytes(graph)
        graph_summaries[name] = {
            "source": source,
            "source_path": (
                relative
                if source == "existing"
                else f"table2-code-all/canonical-graphs/{name}/manifest.json"
            ),
            "source_sha256": source_hash,
            "graph_id": snapshot["graph_id"],
            "condition": snapshot["condition"],
            "construction_recipe": snapshot["construction_recipe"],
            "entities": len(snapshot["entities"]),
            "relations": len(snapshot["relations"]),
            "triples": len(snapshot["triples"]),
        }
    environment = validate_environment(sum(len(content) for content in projections.values()))
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "lineage_status": "lineage-valid",
        "parent_artifact_set_sha256": parent["artifact_set_sha256"],
        "score_manifest_sha256": parent["score_manifest_sha256"],
        "threshold": parent["selected_threshold"],
        "seed": seed,
        "records": record_summary,
        "graphs": graph_summaries,
        "projection_sha256": {name: sha256_bytes(content) for name, content in projections.items()},
        "model_identity": parent["model_identity"],
        "observed_profile": parent["observed_profile"],
        "environment": environment,
    }
    return run_dir, summary, projections, graphs


def validate_ollama_url(ollama_url: str) -> str:
    parsed = urllib.parse.urlparse(ollama_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PhaseEError("--ollama-url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise PhaseEError("Credentials must not be embedded in --ollama-url")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise PhaseEError("Phase E permits only a loopback Ollama endpoint")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise PhaseEError("--ollama-url must contain only scheme, loopback host, and port")
    return ollama_url.rstrip("/")


def fetch_ollama_json(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> tuple[dict[str, Any], bytes, str]:
    request = urllib.request.Request(
        url,
        data=canonical_json_bytes(payload) if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise PhaseEError(f"Cannot query Ollama model identity at {url}: {exc}") from exc
    if len(raw) > 16 * 1024 * 1024:
        raise PhaseEError("Ollama identity response exceeds the 16 MiB safety limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseEError(f"Ollama identity endpoint returned invalid JSON: {url}") from exc
    if not isinstance(value, dict):
        raise PhaseEError(f"Ollama identity endpoint did not return an object: {url}")
    canonical = canonical_json_bytes(value)
    return value, canonical, sha256_bytes(canonical)


def fetch_matching_model(
    ollama_url: str, expected: dict[str, Any], *, check_runtime: bool = True
) -> tuple[dict[str, Any], bytes, bytes]:
    origin = validate_ollama_url(ollama_url)
    tags, tags_bytes, tags_hash = fetch_ollama_json("GET", origin + "/api/tags")
    models = tags.get("models")
    if not isinstance(models, list):
        raise PhaseEError("Ollama /api/tags response lacks a models array")
    matches = [
        item for item in models
        if isinstance(item, dict) and expected["name"] in {item.get("name"), item.get("model")}
    ]
    if len(matches) != 1:
        raise PhaseEError("Ollama inventory does not uniquely contain the same-run Qwen tag")
    observed_tag = validate_sha256(
        str(matches[0].get("digest", "")).removeprefix("sha256:"), "Ollama tag"
    )
    if observed_tag != expected["tag_digest"]:
        raise PhaseEError("Ollama tag digest differs from the same-run verifier")
    show, show_bytes, show_hash = fetch_ollama_json(
        "POST", origin + "/api/show", {"model": expected["name"]}
    )
    details = show.get("details")
    observed_details = {
        "family": str(details.get("family", "")).lower() if isinstance(details, dict) else "",
        "parameter_size": details.get("parameter_size") if isinstance(details, dict) else None,
        "quantization_level": details.get("quantization_level") if isinstance(details, dict) else None,
    }
    if observed_details != expected["details"]:
        raise PhaseEError("Ollama model details differ from the same-run verifier")
    modelfile = show.get("modelfile")
    observed_blobs = {
        match.lower()
        for line in modelfile.splitlines() if isinstance(modelfile, str)
        for match in MODEL_BLOB_RE.findall(line)
        if line.lstrip().upper().startswith("FROM ")
    } if isinstance(modelfile, str) else set()
    if observed_blobs != {expected["blob_sha256"]}:
        raise PhaseEError("Ollama blob identity differs from the same-run verifier")
    runtime_evidence = {}
    if "ollama_runtime_identity" in expected:
        from utils.verifier.runtime import runtime_identity
        from utils.common.artifact_io import DataContractError
        stored_runtime = expected["ollama_runtime_identity"]
        if check_runtime:
            version, _version_bytes, _version_hash = fetch_ollama_json("GET", origin + "/api/version")
        else:
            # The enclosing stage checks the actual server before new calls.
            # Per-answer checks still authenticate tag/blob/defaults without
            # duplicating the same version request and metadata thousands of times.
            version = {"version": stored_runtime.get("server_version")}
        try:
            observed_runtime = runtime_identity(origin, version, show, expected)
        except DataContractError as exc:
            raise PhaseEError(str(exc)) from exc
        if observed_runtime != stored_runtime:
            raise PhaseEError("Ollama server/model runtime differs from the same-run verifier")
        runtime_evidence = {"ollama_runtime_sha256": expected["ollama_runtime"]["sha256"]}
    evidence = {"tags_response_sha256": tags_hash, "show_response_sha256": show_hash}
    if runtime_evidence:
        evidence.update(runtime_evidence)
    else:
        evidence.update({"identity_verified": True, "name": expected["name"],
            "tag_digest": observed_tag, "blob_sha256": expected["blob_sha256"],
            "details": observed_details,
            "verifier_environment_sha256": expected["verifier_environment_sha256"],
            "verifier_condition_id": expected["verifier_condition_id"],
            "verifier_protocol_id": expected["verifier_protocol_id"]})
    return evidence, tags_bytes, show_bytes


def validate_child_write_surface(child_dir: Path) -> None:
    if not child_dir.exists():
        return
    if not child_dir.is_dir() or child_dir.is_symlink() or child_dir.resolve() != child_dir:
        raise PhaseEError("Table-2 output must be one physical same-run directory")
    for entry in child_dir.rglob("*"):
        if entry.is_symlink() or entry.resolve(strict=False) != entry:
            raise PhaseEError(f"Phase E child contains a link or escaped path: {entry}")


def validate_child_target(child_dir: Path, path: Path) -> None:
    try:
        path.relative_to(child_dir)
    except ValueError as exc:
        raise PhaseEError(f"Phase E write target escapes its child: {path}") from exc
    validate_child_write_surface(child_dir)


def write_bytes_once(path: Path, content: bytes, child_dir: Path) -> None:
    validate_child_target(child_dir, path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise PhaseEError(f"Refusing to replace existing different file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_once(path: Path, value: Any, child_dir: Path) -> None:
    write_bytes_once(path, canonical_json_bytes(value), child_dir)


def write_status(path: Path, status: dict[str, Any], child_dir: Path) -> None:
    validate_child_target(child_dir, path)
    content = canonical_json_bytes(status)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stable_child_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise PhaseEError(f"Evaluator path is outside the source checkout: {path}") from exc


def validate_rag_output(
    path: Path,
    contract: dict[str, Any],
    model_name: str,
    *,
    expected_questions: list[dict[str, str]],
    expected_kg: str,
    expected_count: int | None = None,
) -> dict[str, Any]:
    output = load_json(path)
    modes = list(evaluator.MODES)
    count = (
        contract["evaluator"]["part_of_questions"]
        if expected_count is None
        else expected_count
    )
    if not isinstance(output, dict) or set(output) != {"metadata", "accuracy", "correct_counts", "results"}:
        raise PhaseEError(f"RAG output has changed top-level fields: {path}")
    metadata = output.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("n_questions") != count
        or metadata.get("model") != model_name
        or metadata.get("kg") != expected_kg
        or not isinstance(metadata.get("time_seconds"), (int, float))
    ):
        raise PhaseEError(f"RAG output metadata differs from its same-run stage: {path}")
    for field in ("accuracy", "correct_counts", "results"):
        if not isinstance(output.get(field), dict) or list(output[field]) != modes:
            raise PhaseEError(f"RAG output {field} ordering changed: {path}")
    for mode in modes:
        rows = output["results"][mode]
        if not isinstance(rows, list) or len(rows) != count:
            raise PhaseEError(f"RAG output mode {mode} has the wrong row count")
        for index, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or set(row) != {"q", "pred", "gold", "correct"}
                or not isinstance(row.get("pred"), str)
                or not row["pred"].strip()
                or row["pred"].startswith("ERROR:")
                or not isinstance(row.get("correct"), bool)
                or {"q": row.get("q"), "gold": row.get("gold")} != expected_questions[index]
                or row["correct"] != evaluator.check_answer(row["pred"], row["gold"])
            ):
                raise PhaseEError(f"RAG output mode {mode} contains a failed or changed result")
        correct = sum(row["correct"] for row in rows)
        if output["correct_counts"][mode] != correct or output["accuracy"][mode] != correct / count:
            raise PhaseEError(f"RAG output mode {mode} has inconsistent scores")
    return {"sha256": sha256_file(path), "accuracy": output["accuracy"], "correct_counts": output["correct_counts"]}


def record_model_check(
    child_dir: Path,
    status: dict[str, Any],
    status_path: Path,
    phase: str,
    evidence: dict[str, Any],
    tags_bytes: bytes,
    show_bytes: bytes,
) -> dict[str, Any]:
    checks_root = child_dir / "logs" / "model-identity-checks"
    records = status.setdefault("model_identity_checks", [])
    if not isinstance(records, list):
        raise PhaseEError("run status model_identity_checks must be an array")
    index = len(records) + 1
    check_dir = checks_root / f"{index:02d}-{phase}"
    write_bytes_once(check_dir / "ollama-tags.json", tags_bytes, child_dir)
    write_bytes_once(check_dir / "ollama-show.json", show_bytes, child_dir)
    write_json_once(check_dir / "result.json", {
        "run_id": status["run_id"], "phase": phase, "checked_at": utc_now(), "evidence": evidence
    }, child_dir)
    record = {
        "index": index,
        "phase": phase,
        "result": (check_dir / "result.json").relative_to(child_dir).as_posix(),
        "result_sha256": sha256_file(check_dir / "result.json"),
    }
    records.append(record)
    write_status(status_path, status, child_dir)
    return record


def _validate_stored_model_response(
    tags: dict[str, Any], show: dict[str, Any], expected: dict[str, Any]
) -> None:
    models = tags.get("models")
    matches = [
        item
        for item in (models if isinstance(models, list) else [])
        if isinstance(item, dict)
        and expected["name"] in {item.get("name"), item.get("model")}
    ]
    if len(matches) != 1:
        raise ModelEvidenceError("Stored Ollama tags do not uniquely identify the run model")
    observed_tag = str(matches[0].get("digest", "")).removeprefix("sha256:").lower()
    if observed_tag != expected["tag_digest"]:
        raise ModelEvidenceError("Stored Ollama tag digest differs from the run model")
    details = show.get("details")
    observed_details = {
        "family": str(details.get("family", "")).lower() if isinstance(details, dict) else "",
        "parameter_size": details.get("parameter_size") if isinstance(details, dict) else None,
        "quantization_level": details.get("quantization_level") if isinstance(details, dict) else None,
    }
    if observed_details != expected["details"]:
        raise ModelEvidenceError("Stored Ollama model details differ from the run model")
    modelfile = show.get("modelfile")
    observed_blobs = {
        match.lower()
        for line in modelfile.splitlines() if isinstance(modelfile, str)
        for match in MODEL_BLOB_RE.findall(line)
        if line.lstrip().upper().startswith("FROM ")
    } if isinstance(modelfile, str) else set()
    if observed_blobs != {expected["blob_sha256"]}:
        raise ModelEvidenceError("Stored Ollama blob identity differs from the run model")
    if "ollama_runtime_identity" in expected:
        from utils.verifier.runtime import runtime_identity
        from utils.common.artifact_io import DataContractError
        runtime = expected["ollama_runtime_identity"]
        try:
            observed = runtime_identity(runtime["origin"],
                {"version": runtime["server_version"]}, show, expected)
        except (DataContractError, KeyError) as exc:
            raise ModelEvidenceError("Stored Ollama runtime evidence is malformed") from exc
        if observed != runtime:
            raise ModelEvidenceError("Stored Ollama generation defaults differ from the run runtime")


def validate_model_check(
    child_dir: Path,
    status: dict[str, Any],
    record: Any,
    *,
    expected_phase: str,
    expected_model: dict[str, Any],
    condition: str | None,
) -> dict[str, Any]:
    """Authenticate one persisted pre/post model check and its raw evidence."""

    if not isinstance(record, dict):
        raise ModelEvidenceError(
            f"Missing persisted model check for {expected_phase}", condition
        )
    index = record.get("index")
    checks = status.get("model_identity_checks")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 1
        or not isinstance(checks, list)
        or index > len(checks)
        or checks[index - 1] != record
        or record.get("phase") != expected_phase
    ):
        raise ModelEvidenceError(
            f"Status check index/order is invalid for {expected_phase}", condition
        )
    expected_relative = (
        f"logs/model-identity-checks/{index:02d}-{expected_phase}/result.json"
    )
    if record.get("result") != expected_relative:
        raise ModelEvidenceError(
            f"Stored model-check path is invalid for {expected_phase}", condition
        )
    result_path = child_dir / expected_relative
    if (
        not result_path.is_file()
        or result_path.is_symlink()
        or sha256_file(result_path) != record.get("result_sha256")
    ):
        raise ModelEvidenceError(
            f"Stored model-check result changed for {expected_phase}", condition
        )
    result = load_json(result_path)
    evidence = result.get("evidence") if isinstance(result, dict) else None
    if (
        set(result) != {"run_id", "phase", "checked_at", "evidence"}
        or result.get("run_id") != status.get("run_id")
        or result.get("phase") != expected_phase
        or not isinstance(result.get("checked_at"), str)
        or not isinstance(evidence, dict)
    ):
        raise ModelEvidenceError(
            f"Stored model-check document is malformed for {expected_phase}", condition
        )
    expected_evidence = {
        "identity_verified": True,
        "name": expected_model["name"],
        "tag_digest": expected_model["tag_digest"],
        "blob_sha256": expected_model["blob_sha256"],
        "details": expected_model["details"],
        "verifier_environment_sha256": expected_model["verifier_environment_sha256"],
        "verifier_condition_id": expected_model["verifier_condition_id"],
        "verifier_protocol_id": expected_model["verifier_protocol_id"],
    }
    if "ollama_runtime_identity" in expected_model:
        expected_evidence = {"ollama_runtime_sha256": expected_model["ollama_runtime"]["sha256"]}
    for field, expected in expected_evidence.items():
        if evidence.get(field) != expected:
            raise ModelEvidenceError(
                f"Stored model-check evidence differs at {field} for {expected_phase}",
                condition,
            )
    check_dir = result_path.parent
    tags_path = check_dir / "ollama-tags.json"
    show_path = check_dir / "ollama-show.json"
    if not tags_path.is_file() or not show_path.is_file():
        raise ModelEvidenceError(
            f"Stored raw model evidence is missing for {expected_phase}", condition
        )
    if (
        sha256_file(tags_path) != evidence.get("tags_response_sha256")
        or sha256_file(show_path) != evidence.get("show_response_sha256")
    ):
        raise ModelEvidenceError(
            f"Stored raw model evidence hashes changed for {expected_phase}", condition
        )
    try:
        _validate_stored_model_response(
            load_json(tags_path), load_json(show_path), expected_model
        )
    except ModelEvidenceError as exc:
        raise ModelEvidenceError(str(exc), condition) from exc
    return record


def run_stage(
    name: str,
    execute: Callable[[], dict[str, Any]],
    output_path: Path,
    validator: Callable[[Path], dict[str, Any]],
    child_dir: Path,
    status: dict[str, Any],
    status_path: Path,
) -> dict[str, Any]:
    stage = status.setdefault("stages", {}).setdefault(name, {"attempts": []})
    if stage.get("status") in {"complete", "output-validated"}:
        if not output_path.is_file() or sha256_file(output_path) != stage.get("output_sha256"):
            raise PhaseEError(f"Completed stage output no longer matches: {name}")
        return validator(output_path)
    if output_path.exists():
        recovery_root = child_dir / "recovery" / name
        recoveries = list(recovery_root.glob("*.json")) if recovery_root.is_dir() else []
        recovered = recovery_root / f"unverified-{len(recoveries) + 1:03d}.json"
        validate_child_target(child_dir, recovered)
        recovered.parent.mkdir(parents=True, exist_ok=True)
        os.replace(output_path, recovered)
        stage.setdefault("quarantined_outputs", []).append({
            "path": recovered.relative_to(child_dir).as_posix(),
            "sha256": sha256_file(recovered),
        })
    attempt = {"number": len(stage["attempts"]) + 1, "started_at": utc_now()}
    stage["attempts"].append(attempt)
    stage["status"] = "running"
    write_status(status_path, status, child_dir)
    try:
        output = execute()
        write_bytes_once(output_path, scientific_json_bytes(output), child_dir)
        validation = validator(output_path)
    except BaseException:
        stage["status"] = "failed"
        write_status(status_path, status, child_dir)
        raise
    attempt["finished_at"] = utc_now()
    stage.update(status="output-validated", output_sha256=sha256_file(output_path), validation=validation)
    write_status(status_path, status, child_dir)
    return validation


def persisted_model_identity(model: dict[str, Any]) -> dict[str, Any]:
    """Keep the shared runtime reference; its loaded contents stay in memory."""
    if "ollama_runtime" in model:
        return {key: model[key] for key in ("ollama_runtime", "verifier_environment_sha256") if key in model}
    return {key: value for key, value in model.items() if key != "ollama_runtime_identity"}


def _preflight_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": preflight["run_id"],
        "parent_artifact_set_sha256": preflight["parent_artifact_set_sha256"],
        "score_manifest_sha256": preflight["score_manifest_sha256"],
        "seed": preflight["seed"],
        "threshold": preflight["threshold"],
        "projection_sha256": preflight["projection_sha256"],
        "graphs": {
            name: {
                "graph_id": value["graph_id"],
                "source_sha256": value["source_sha256"],
            }
            for name, value in preflight["graphs"].items()
        },
        "model_identity": persisted_model_identity({
            key: preflight["model_identity"][key]
            for key in (
                "name",
                "tag_digest",
                "blob_sha256",
                "details",
                "verifier_environment_sha256",
                "verifier_condition_id",
                "verifier_protocol_id",
            ) + tuple(key for key in ("ollama_runtime",)
                      if key in preflight["model_identity"])
        }),
    }
