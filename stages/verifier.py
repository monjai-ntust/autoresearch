"""Verifier-stage controller for development gating and final inference."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from pathlib import Path
import shutil
import subprocess
from typing import Any

from utils.common.artifact_io import (
    DataContractError,
    _load_manifest,
    _next_recovery_path,
    _quarantine_run_artifact,
    _require_fields,
    _same_run_seal_path,
    _validate_recovery_provenance,
    _validate_same_run_seal,
    _write_recovery_provenance,
    _write_same_run_seal,
    atomic_write_json,
    iter_jsonl,
    sha256_file,
)
from utils.common.config import PipelineConfig
from utils.common.paths import RunLayout
from utils.common.records import Candidate
from utils.evaluation.publication import prepare_verifier_pilot
from utils.rag import runner as table2_runner
from utils.verifier.pilot import PilotInputs, run_verifier_pilot
from utils.verifier.threshold import select_threshold
from utils.verifier.verifier import run_verifier, validate_response_cache


@dataclass(frozen=True)
class VerifierContext:
    development: Path
    warmup_candidates: Path
    model_blob: Path


def prepare_development_gate(
    layout: RunLayout,
    config: PipelineConfig,
    development_candidates: Path,
) -> tuple[Path, Path]:
    """Select the development threshold and materialize pilot inputs."""

    threshold_path = layout.resolve("predictions/dev/threshold-selection.json")
    if not threshold_path.is_file():
        select_threshold(
            layout,
            config,
            candidates_path=development_candidates,
            gold_path=layout.resolve(
                "data-prepared/development-gold.jsonl", must_exist=True
            ),
            split_manifest_path=layout.resolve(
                "data-prepared/split-manifest.json", must_exist=True
            ),
            candidate_index_path=layout.resolve(
                "predictions/dev/candidate-index.json", must_exist=True
            ),
            out_path=threshold_path,
        )
    _validate_threshold(layout, config)

    pilot_selection = layout.resolve("predictions/dev/pilot-selection.json")
    if not pilot_selection.is_file():
        prepare_verifier_pilot(layout, config)
    _validate_pilot_inputs(layout, config)
    return threshold_path, pilot_selection


def run_pilot(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    threshold_path: Path,
    pilot_selection: Path,
    ollama_url: str,
    model_blob_source: str | None,
    encoder_identity: dict[str, Any],
    training_hardware: dict[str, object],
) -> VerifierContext:
    """Authenticate the live model and run the development-only verifier gate."""

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

    pilot_candidates = layout.resolve(
        "predictions/dev/pilot-candidates.jsonl", must_exist=True
    )
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
                {
                    "capture_id": capture_id,
                    "mode": mode,
                    "repeat": repeat,
                    "artifact_prefix": prefix,
                }
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
                gold=layout.resolve(
                    "data-prepared/development-gold.jsonl", must_exist=True
                ),
                candidates=pilot_candidates,
                warmup_candidates=warmup_candidates,
                split_manifest=layout.resolve(
                    "data-prepared/split-manifest.json", must_exist=True
                ),
                candidate_index=layout.resolve(
                    "predictions/dev/candidate-index.json", must_exist=True
                ),
                pilot_selection=pilot_selection,
                threshold_selection=threshold_path,
                capture_index=capture_index,
            ),
            evidence_class="development-pilot",
        )
        if (
            audit.get("pilot_status") != "pass"
            or audit.get("material_protocol_review_required") is not False
        ):
            raise DataContractError(
                "Development verifier pilot did not admit final-test execution"
            )
    _validate_pilot_audit(layout, config)
    return VerifierContext(development, warmup_candidates, model_blob)


def run_test(
    layout: RunLayout,
    config: PipelineConfig,
    context: VerifierContext,
    test_candidates: Path,
    *,
    ollama_url: str,
) -> None:
    """Run the simple and corrective verifier over final-test candidates."""

    for mode in ("simple", "corrective"):
        _resume_verifier_stage(
            layout,
            config,
            mode=mode,
            sentences=layout.resolve("data-prepared/test.jsonl", must_exist=True),
            candidates=test_candidates,
            development=context.development,
            warmup_candidates=context.warmup_candidates,
            model_blob=context.model_blob,
            ollama_url=ollama_url,
        )


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
