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

import table_evaluator as evaluator
from graph_construction import build_table_graphs, canonical_json, project_graph


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = (ROOT / "output").resolve()
CONTRACT_PATH = ROOT / "table2_contract.json"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MODEL_BLOB_RE = re.compile(r"sha256[-:]([0-9a-f]{64})", re.IGNORECASE)


class PhaseEError(RuntimeError):
    """Raised when a frozen-method or same-run lineage gate fails."""


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


def git_blob(path: str) -> str:
    return run_git("hash-object", f"--path={path}", path).stdout.strip()


def validate_source_contract(
    contract: dict[str, Any], require_clean: bool
) -> dict[str, str]:
    if contract.get("schema_version") != 3:
        raise PhaseEError("Phase E contract schema is not revision 0.8")
    baseline = contract["source_baseline"]["commit"]
    head = run_git("rev-parse", "HEAD").stdout.strip()
    branch = run_git("branch", "--show-current").stdout.strip()
    if run_git("merge-base", "--is-ancestor", baseline, head, check=False).returncode:
        raise PhaseEError(f"Source HEAD {head} does not descend from baseline {baseline}")
    expected_branch = contract["source_baseline"]["branch"]
    if branch != expected_branch:
        raise PhaseEError(
            f"Phase E must run from branch {expected_branch}; current branch is "
            f"{branch or '<detached>'}"
        )
    if require_clean:
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            if run_git(*args, check=False).returncode:
                raise PhaseEError("Tracked source changes exist; commit or discard them first")
        untracked = run_git("ls-files", "--others", "--exclude-standard").stdout.strip()
        if untracked:
            raise PhaseEError("Untracked source files exist; remove or commit them first")
    mismatches = []
    for path, expected in contract["frozen_upstream_blobs"].items():
        actual = git_blob(path)
        if actual != expected:
            mismatches.append({"path": path, "expected": expected, "actual": actual})
    if mismatches:
        raise PhaseEError("Frozen source mismatch: " + json.dumps(mismatches, sort_keys=True))
    return {"head": head, "branch": branch, "baseline": baseline}


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


def load_verifier_model_identity(path: Path, expected_hash: str) -> dict[str, Any]:
    if sha256_file(path) != validate_sha256(expected_hash, "verifier environment hash"):
        raise PhaseEError("Verifier environment differs from its stage manifest")
    manifest = load_json(path)
    if not isinstance(manifest, dict) or manifest.get("execution_mode") != "live":
        raise PhaseEError("Verifier environment must describe a completed live verifier run")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("identity_verified") is not True:
        raise PhaseEError("Verifier environment lacks a verified model identity")
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
    return {
        "name": name,
        "tag_digest": tag,
        "blob_sha256": blob,
        "details": normalized_details,
        "verifier_environment_sha256": expected_hash,
        "verifier_condition_id": manifest.get("condition_id"),
        "verifier_protocol_id": manifest.get("protocol_id"),
    }


def _compare_profile(observed: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    fields = {}
    for key, expected in reference.items():
        actual = observed.get(key)
        if isinstance(expected, dict):
            nested = _compare_profile(actual if isinstance(actual, dict) else {}, expected)
            fields[key] = nested
        else:
            fields[key] = {"observed": actual, "reference": expected, "match": actual == expected}
    matches = [
        value["match"] if "match" in value else value["all_fields_match"]
        for value in fields.values()
    ]
    return {"all_fields_match": all(matches), "fields": fields}


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


def _validate_verifier_stage(
    run_dir: Path,
    ledger: dict[str, dict[str, Any]],
    artifacts: dict[str, str],
    pipeline: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    condition = "VER-" + mode.upper()
    manifest_relative = artifacts[f"{mode}_verifier_manifest"]
    manifest_path = _verified_file(run_dir, manifest_relative, ledger)
    manifest = load_json(manifest_path)
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
    for relative, expected in {**(inputs or {}), **(outputs or {})}.items():
        _verified_file(run_dir, relative, ledger, expected=expected)
    for required in (artifacts["prepared_test"], artifacts["candidates"]):
        _manifest_hash(inputs, required, f"{mode}.inputs")
    verdict_relative = artifacts[f"{mode}_verdicts"]
    environment_relative = artifacts[f"{mode}_environment"]
    verdict_hash = _manifest_hash(outputs, verdict_relative, f"{mode}.outputs")
    environment_hash = _manifest_hash(outputs, environment_relative, f"{mode}.outputs")
    verdicts = load_jsonl(_verified_file(run_dir, verdict_relative, ledger, expected=verdict_hash))
    if manifest.get("verdict_count") != len(verdicts):
        raise PhaseEError(f"{mode} verifier manifest count differs from its ledger")
    identity = load_verifier_model_identity(
        _verified_file(run_dir, environment_relative, ledger, expected=environment_hash),
        environment_hash,
    )
    if identity["verifier_condition_id"] != condition:
        raise PhaseEError(f"{mode} environment carries another condition")
    model_manifest = validate_sha256(str(manifest.get("model_manifest_sha256", "")), "model manifest")
    if model_manifest != identity["tag_digest"]:
        raise PhaseEError(f"{mode} manifest and environment disagree about Qwen")
    if any(row.get("model_manifest_sha256") != model_manifest for row in verdicts):
        raise PhaseEError(f"{mode} verdicts contain a conflicting Qwen digest")
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
    return manifest, {**identity, "model_path": model_path}, verdicts


def validate_parent_lineage(
    run_dir: Path, run_id: str, contract: dict[str, Any]
) -> dict[str, Any]:
    pipeline = contract["pipeline"]
    paths = pipeline["artifacts"]
    ledger: dict[str, dict[str, Any]] = {}

    checkout = load_json(_verified_file(run_dir, paths["checkout_manifest"], ledger))
    _require_fields(
        checkout,
        {
            "schema_version": "phase-b-checkout-manifest-1.0",
            "status": "pass",
            "run_id": run_id,
            "protocol_id": pipeline["protocol_id"],
            "workflow_id": pipeline["workflow_id"],
        },
        "checkout manifest",
    )
    if checkout.get("source", {}).get("worktree_clean") is not True:
        raise PhaseEError("Selected run was not created from a clean source checkout")

    full = load_json(_verified_file(run_dir, paths["full_run_manifest"], ledger))
    _require_fields(
        full,
        {
            "schema_version": "phase-b-debug-full-run-1.0",
            "run_id": run_id,
            "protocol_id": pipeline["protocol_id"],
            "workflow_id": pipeline["workflow_id"],
            "training_seeds": pipeline["training_seeds"],
            "conditional_b07_approval": True,
        },
        "full-run manifest",
    )

    preparation = load_json(_verified_file(run_dir, paths["preparation_manifest"], ledger))
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
    split = preparation.get("split")
    if not isinstance(split, dict) or split.get("seed") != pipeline["split_seed"] or split.get("test") != pipeline["test_records"]:
        raise PhaseEError("Preparation split differs from the frozen table method")
    prep_names = {
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

    score_path = _verified_file(run_dir, paths["score_manifest"], ledger)
    score = load_json(score_path)
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
    for relative, expected in {**(score_inputs or {}), **(score_outputs or {})}.items():
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
    selected_threshold = threshold.get("selected_threshold")
    if (
        isinstance(selected_threshold, bool)
        or not isinstance(selected_threshold, (int, float))
        or not 0 <= float(selected_threshold) <= 1
        or threshold.get("used_test_labels") is not False
        or threshold.get("split_manifest_sha256") != ledger[paths["split_manifest"]]["sha256"]
        or threshold.get("development_gold_sha256") != ledger[paths["development_gold"]]["sha256"]
    ):
        raise PhaseEError("Threshold selection is not bound to development-only same-run data")
    development_index = _verified_file(run_dir, paths["development_candidate_index"], ledger)
    if threshold.get("development_candidate_index_sha256") != ledger[paths["development_candidate_index"]]["sha256"]:
        raise PhaseEError("Threshold selection differs from the same-run development candidates")

    prepared_rows = load_jsonl(run_file(run_dir, paths["prepared_test"]))
    gold_rows = load_jsonl(run_file(run_dir, paths["private_gold"]))
    candidate_rows = load_jsonl(run_file(run_dir, paths["candidates"]))
    if len(prepared_rows) != pipeline["test_records"] or len(gold_rows) != pipeline["test_records"]:
        raise PhaseEError("Prepared test and private gold have the wrong row count")

    encoder_identities = []
    checkpoint_manifest_hashes = {}
    hardware = None
    for seed in pipeline["training_seeds"]:
        checkpoint_manifest_relative = paths["checkpoint_manifest"].format(seed=seed)
        checkpoint_blob_relative = paths["checkpoint_blob"].format(seed=seed)
        generation_relative = paths["candidate_generation_manifest"].format(seed=seed)
        checkpoint_manifest_path = _verified_file(run_dir, checkpoint_manifest_relative, ledger)
        checkpoint_manifest = load_json(checkpoint_manifest_path)
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
        }:
            raise PhaseEError(f"seed-{seed} checkpoint manifest schema is unsupported")
        checkpoint_digest = validate_sha256(
            str(checkpoint_manifest.get("checkpoint_sha256", "")),
            f"seed-{seed} checkpoint",
        )
        _verified_file(run_dir, checkpoint_blob_relative, ledger, expected=checkpoint_digest)
        generation = load_json(_verified_file(run_dir, generation_relative, ledger))
        _require_fields(
            generation,
            {
                "stage": "model-generate-candidates",
                "status": "completed",
                "execution_mode": "live",
                "protocol_id": pipeline["protocol_id"],
                "training_seed": seed,
            },
            f"seed-{seed} candidate generation",
        )
        generation_checkpoint = generation.get("inputs", {}).get("checkpoint_manifest", {})
        if (
            generation_checkpoint.get("path") != checkpoint_manifest_relative
            or generation_checkpoint.get("sha256") != ledger[checkpoint_manifest_relative]["sha256"]
            or generation.get("checkpoint", {}).get("sha256") != checkpoint_digest
            or generation.get("inputs", {}).get("prepared_sentences", {}).get("sha256")
            != ledger[paths["prepared_test"]]["sha256"]
        ):
            raise PhaseEError(f"seed-{seed} candidates are not bound to their checkpoint/data")
        checkpoint_manifest_hashes[seed] = ledger[checkpoint_manifest_relative]["sha256"]
        encoder_identities.append({
            "base_model": checkpoint_manifest.get("base_model"),
            "base_model_revision": checkpoint_manifest.get("base_model_revision"),
        })
        train_manifest = load_json(_verified_file(
            run_dir, f"manifests/model-train-live-seed-{seed}.json", ledger
        ))
        if train_manifest.get("training_seed") != seed or train_manifest.get("status") != "completed":
            raise PhaseEError(f"seed-{seed} training manifest is incomplete")
        recipe = train_manifest.get("recipe")
        outputs = train_manifest.get("outputs")
        if (
            not isinstance(recipe, dict)
            or recipe.get("base_model") != checkpoint_manifest.get("base_model")
            or recipe.get("base_model_revision")
            != checkpoint_manifest.get("base_model_revision")
            or not isinstance(outputs, dict)
            or outputs.get("checkpoint") != checkpoint_blob_relative
            or outputs.get("checkpoint_manifest") != checkpoint_manifest_relative
        ):
            raise PhaseEError(f"seed-{seed} training and checkpoint identities disagree")
        if hardware is None:
            hardware = train_manifest.get("environment")
    encoder_identity = require_consistent_encoder_identity(encoder_identities)
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
        run_dir, ledger, paths, pipeline, "simple"
    )
    corrective_manifest, corrective_model, corrective_rows = _validate_verifier_stage(
        run_dir, ledger, paths, pipeline, "corrective"
    )
    qwen_fields = ("name", "tag_digest", "blob_sha256", "details", "model_path")
    if any(simple_model[field] != corrective_model[field] for field in qwen_fields):
        raise PhaseEError("Simple and corrective stages use different Qwen identities")
    if (
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
    comparisons = {
        key: _compare_profile(observed_profile[key], contract["reference_profile"][key])
        for key in ("hardware", "encoder", "qwen")
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
        "historical_reference_match": {
            **comparisons,
            "overall": all(value["all_fields_match"] for value in comparisons.values()),
            "admission_effect": "none",
        },
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
    questions = evaluator.generate_questions(projected, contract["evaluator"]["max_questions"])
    if len(questions) != contract["evaluator"]["max_questions"]:
        raise PhaseEError("Projection does not generate exactly ten frozen questions")
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
            "source_path": relative if source == "existing" else f"table2/canonical-graphs/{name}/manifest.json",
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
        "historical_reference_match": parent["historical_reference_match"],
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
    ollama_url: str, expected: dict[str, Any]
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
    evidence = {
        "identity_verified": True,
        "name": expected["name"],
        "tag_digest": observed_tag,
        "blob_sha256": expected["blob_sha256"],
        "details": observed_details,
        "verifier_environment_sha256": expected["verifier_environment_sha256"],
        "verifier_condition_id": expected["verifier_condition_id"],
        "verifier_protocol_id": expected["verifier_protocol_id"],
        "tags_response_sha256": tags_hash,
        "show_response_sha256": show_hash,
    }
    return evidence, tags_bytes, show_bytes


def validate_child_write_surface(child_dir: Path) -> None:
    if not child_dir.exists():
        return
    if not child_dir.is_dir() or child_dir.is_symlink() or child_dir.resolve() != child_dir:
        raise PhaseEError("table2 must be one physical same-run directory")
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
) -> dict[str, Any]:
    output = load_json(path)
    modes = contract["evaluator"]["modes"]
    count = contract["evaluator"]["max_questions"]
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
    existing = list(checks_root.glob("*")) if checks_root.is_dir() else []
    check_dir = checks_root / f"{len(existing) + 1:02d}-{phase}"
    write_bytes_once(check_dir / "ollama-tags.json", tags_bytes, child_dir)
    write_bytes_once(check_dir / "ollama-show.json", show_bytes, child_dir)
    write_json_once(check_dir / "result.json", {
        "run_id": status["run_id"], "phase": phase, "checked_at": utc_now(), "evidence": evidence
    }, child_dir)
    record = {
        "phase": phase,
        "result": (check_dir / "result.json").relative_to(child_dir).as_posix(),
        "result_sha256": sha256_file(check_dir / "result.json"),
    }
    status.setdefault("model_identity_checks", []).append(record)
    write_status(status_path, status, child_dir)
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
    if stage.get("status") == "complete":
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
        write_json_once(output_path, output, child_dir)
        validation = validator(output_path)
    except BaseException:
        stage["status"] = "failed"
        write_status(status_path, status, child_dir)
        raise
    attempt["finished_at"] = utc_now()
    stage.update(status="output-validated", output_sha256=sha256_file(output_path), validation=validation)
    write_status(status_path, status, child_dir)
    return validation


def build_child_identity(
    run_id: str, source: dict[str, str], preflight: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "phase-e-same-run-identity-2.0",
        "run_id": run_id,
        "source_head": source["head"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "parent_artifact_set_sha256": preflight["parent_artifact_set_sha256"],
        "score_manifest_sha256": preflight["score_manifest_sha256"],
        "seed": preflight["seed"],
        "threshold": preflight["threshold"],
        "projection_sha256": preflight["projection_sha256"],
        "graph_ids": {name: value["graph_id"] for name, value in preflight["graphs"].items()},
        "encoder_base_model_revision": preflight["observed_profile"]["encoder"]["base_model_revision"],
        "ollama_tag_digest": preflight["model_identity"]["tag_digest"],
        "ollama_model_blob_sha256": preflight["model_identity"]["blob_sha256"],
    }


def validate_child_hashes(child_dir: Path, run_id: str) -> None:
    manifest = load_json(child_dir / "artifact-hashes.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "phase-e-artifact-hashes-1.0"
        or manifest.get("run_id") != run_id
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise PhaseEError("Completed Phase E artifact hash manifest is malformed")
    observed = {
        path.relative_to(child_dir).as_posix(): sha256_file(path)
        for path in sorted(child_dir.rglob("*"))
        if path.is_file() and path.name not in {"artifact-hashes.json", "run-status.json"}
    }
    if manifest["artifacts"] != observed:
        raise PhaseEError("Completed Phase E child artifact hashes differ")


def validate_phase_e_child(
    child_dir: Path,
    status: dict[str, Any],
    contract: dict[str, Any],
    preflight: dict[str, Any],
    projection_paths: dict[str, Path],
) -> dict[str, Any]:
    validate_child_write_surface(child_dir)
    validate_child_hashes(child_dir, preflight["run_id"])
    projection_manifest = load_json(child_dir / "projections" / "manifest.json")
    if (
        projection_manifest.get("run_id") != preflight["run_id"]
        or projection_manifest.get("inputs", {}).get("parent_artifact_set_sha256")
        != preflight["parent_artifact_set_sha256"]
    ):
        raise PhaseEError("Projection manifest differs from the selected run")
    for name, path in projection_paths.items():
        if sha256_file(path) != preflight["projection_sha256"][name]:
            raise PhaseEError(f"Phase E projection changed: {name}")
    results = {}
    for condition in ("confidence", "corrective", "gold"):
        output = child_dir / "rag-results" / f"{condition}.json"
        results[condition] = validate_rag_output(
            output,
            contract,
            preflight["model_identity"]["name"],
            expected_questions=preflight["records"]["question_contract"],
            expected_kg=stable_child_path(projection_paths[condition]),
        )
        stage = status.get("stages", {}).get(f"rag_{condition}")
        if not isinstance(stage, dict) or stage.get("status") != "complete" or stage.get("output_sha256") != results[condition]["sha256"]:
            raise PhaseEError(f"Status does not bind the completed {condition} stage")
    table = load_json(child_dir / "table2-results.json")
    if (
        table.get("run_id") != preflight["run_id"]
        or table.get("new_results") != results
        or table.get("legacy_displayed_accuracy") != contract["legacy_table2_accuracy"]
    ):
        raise PhaseEError("Table 2 summary differs from validated results")
    return results


def run_command(args: argparse.Namespace, contract: dict[str, Any]) -> int:
    source = validate_source_contract(contract, require_clean=not args.dry_run)
    run_dir, preflight, projections, graphs = preflight_same_run(args.run_id, contract)
    child_dir = run_dir / "table2"
    validate_child_write_surface(child_dir)
    if args.dry_run:
        print(json.dumps({"source": source, **preflight}, indent=2, sort_keys=True))
        return 0

    identity = build_child_identity(args.run_id, source, preflight)
    status_path = child_dir / "run-status.json"
    projection_dir = child_dir / "projections"
    projection_paths = {
        "records": projection_dir / "test-with-private-gold.jsonl",
        "confidence": projection_dir / "confidence-seed-42.json",
        "corrective": projection_dir / "corrective-seed-42.json",
        "gold": projection_dir / "gold-oracle.json",
    }
    new_child = not child_dir.exists() or not status_path.exists()
    if child_dir.exists() and not status_path.exists() and any(child_dir.iterdir()):
        raise PhaseEError("table2 exists without a resumable status identity")
    if new_child:
        status = {
            "schema_version": "phase-e-same-run-status-2.0",
            "run_id": args.run_id,
            "identity": identity,
            "status": "running",
            "created_at": utc_now(),
            "stages": {},
        }
    else:
        status = load_json(status_path)
        if status.get("identity") != identity or status.get("run_id") != args.run_id:
            raise PhaseEError("Existing Phase E child identity differs from this run")
        if status.get("status") == "complete":
            validate_phase_e_child(child_dir, status, contract, preflight, projection_paths)
            print(json.dumps({"run_dir": str(run_dir), "status": "complete", "resumed": True}, indent=2))
            return 0

    model_evidence, tags_bytes, show_bytes = fetch_matching_model(
        args.ollama_url, preflight["model_identity"]
    )
    if new_child:
        child_dir.mkdir(exist_ok=True)
        write_status(status_path, status, child_dir)
    record_model_check(child_dir, status, status_path, "before-stages", model_evidence, tags_bytes, show_bytes)

    for name, path in projection_paths.items():
        write_bytes_once(path, projections[name], child_dir)
    for name, graph in graphs.items():
        if preflight["graphs"][name]["source"] == "constructed":
            write_json_once(child_dir / "canonical-graphs" / name / "manifest.json", graph, child_dir)
    write_json_once(projection_dir / "manifest.json", {
        "schema_version": "phase-e-projection-manifest-2.0",
        "run_id": args.run_id,
        "representation_only": True,
        "inputs": {
            "parent_artifact_set_sha256": preflight["parent_artifact_set_sha256"],
            "score_manifest_sha256": preflight["score_manifest_sha256"],
            "seed": preflight["seed"],
            "threshold": preflight["threshold"],
            "graphs": preflight["graphs"],
        },
        "outputs": {path.relative_to(child_dir).as_posix(): sha256_file(path) for path in projection_paths.values()},
        "record_summary": preflight["records"],
    }, child_dir)
    write_json_once(child_dir / "method-manifest.json", {
        "run_id": args.run_id,
        "contract": contract,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "lineage_status": preflight["lineage_status"],
        "observed_profile": preflight["observed_profile"],
        "historical_reference_match": preflight["historical_reference_match"],
    }, child_dir)
    write_json_once(child_dir / "environment.json", {
        "run_id": args.run_id,
        "captured_at": status["created_at"],
        "platform": platform.platform(),
        "python": sys.version,
        "source": source,
        "model_identity": preflight["model_identity"],
    }, child_dir)

    results = {}
    for condition in ("confidence", "corrective", "gold"):
        output_path = child_dir / "rag-results" / f"{condition}.json"
        kg_identity = stable_child_path(projection_paths[condition])
        records = load_jsonl(projection_paths["records"])
        kg = load_json(projection_paths[condition])
        results[condition] = run_stage(
            f"rag_{condition}",
            lambda records=records, kg=kg, kg_identity=kg_identity: evaluator.evaluate(
                records,
                kg,
                kg_identity=kg_identity,
                ollama_url=args.ollama_url.rstrip("/"),
                ollama_model=preflight["model_identity"]["name"],
                max_questions=contract["evaluator"]["max_questions"],
            ),
            output_path,
            lambda path, kg_identity=kg_identity: validate_rag_output(
                path,
                contract,
                preflight["model_identity"]["name"],
                expected_questions=preflight["records"]["question_contract"],
                expected_kg=kg_identity,
            ),
            child_dir,
            status,
            status_path,
        )
        model_evidence, tags_bytes, show_bytes = fetch_matching_model(
            args.ollama_url, preflight["model_identity"]
        )
        check = record_model_check(
            child_dir, status, status_path, f"after-{condition}", model_evidence, tags_bytes, show_bytes
        )
        status["stages"][f"rag_{condition}"].update(status="complete", post_model_identity_check=check)
        write_status(status_path, status, child_dir)

    write_json_once(child_dir / "table2-results.json", {
        "schema_version": "phase-e-same-run-table2-1.0",
        "run_id": args.run_id,
        "lineage_status": preflight["lineage_status"],
        "historical_reference_match": preflight["historical_reference_match"],
        "new_results": results,
        "legacy_displayed_accuracy": contract["legacy_table2_accuracy"],
        "note": "Legacy values are comparison references only and never runtime inputs or tuning targets.",
    }, child_dir)
    hashes = {
        path.relative_to(child_dir).as_posix(): sha256_file(path)
        for path in sorted(child_dir.rglob("*"))
        if path.is_file() and path.name not in {"artifact-hashes.json", "run-status.json"}
    }
    write_json_once(child_dir / "artifact-hashes.json", {
        "schema_version": "phase-e-artifact-hashes-1.0", "run_id": args.run_id, "artifacts": hashes
    }, child_dir)
    validate_parent_lineage(run_dir, args.run_id, contract)
    validate_phase_e_child(child_dir, status, contract, preflight, projection_paths)
    status["status"] = "complete"
    status["finished_at"] = utc_now()
    write_status(status_path, status, child_dir)
    print(json.dumps({"run_dir": str(run_dir), "child": "table2", "status": "complete"}, indent=2))
    return 0
