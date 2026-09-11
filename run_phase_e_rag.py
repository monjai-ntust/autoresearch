"""Fail-closed Phase E runner for the frozen downstream E-T02 workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = (ROOT / "output").resolve()
CONTRACT_PATH = ROOT / "phase_e_rag_contract.json"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MODEL_BLOB_RE = re.compile(r"sha256[-:]([0-9a-f]{64})", re.IGNORECASE)


class PhaseEError(RuntimeError):
    """Raised when a frozen-method or artifact gate fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseEError(f"Cannot read valid JSON from {path}: {exc}") from exc


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


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


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PhaseEError(f"git {' '.join(args)} failed: {detail}")
    return result


def git_blob(path: str) -> str:
    return run_git("hash-object", f"--path={path}", path).stdout.strip()


def validate_source_contract(contract: dict, require_clean: bool) -> dict:
    baseline = contract["source_baseline"]["commit"]
    head = run_git("rev-parse", "HEAD").stdout.strip()
    branch = run_git("branch", "--show-current").stdout.strip()
    ancestor = run_git("merge-base", "--is-ancestor", baseline, head, check=False)
    if ancestor.returncode != 0:
        raise PhaseEError(f"Source HEAD {head} does not descend from baseline {baseline}")
    if branch != "publication-legacy-methodology":
        raise PhaseEError(
            "Phase E must run from branch publication-legacy-methodology; "
            f"current branch is {branch or '<detached>'}"
        )
    if require_clean:
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            if run_git(*args, check=False).returncode != 0:
                raise PhaseEError("Tracked source changes exist; commit or discard them first")

    mismatches = []
    for path, expected in contract["frozen_upstream_blobs"].items():
        actual = git_blob(path)
        if actual != expected:
            mismatches.append({"path": path, "expected": expected, "actual": actual})
    active = {
        "build_kg.py": contract["graph_builder"]["blob"],
        "eval_graph_rag.py": contract["evaluator"]["post_bf02_blob"],
    }
    for path, expected in active.items():
        actual = git_blob(path)
        if actual != expected:
            mismatches.append({"path": path, "expected": expected, "actual": actual})
    if mismatches:
        raise PhaseEError(
            "Frozen source mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    return {"head": head, "branch": branch, "baseline": baseline}


def validate_input_file(path: Path, expected_hash: str, label: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PhaseEError(f"{label} is not a file: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_hash:
        raise PhaseEError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual}"
        )
    return actual


def load_inference(path: Path, contract: dict) -> tuple[list[dict], dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PhaseEError(
                    f"Invalid inference JSON at nonblank line {line_number}: {exc}"
                ) from exc
            required = {
                "doc_id",
                "sentence",
                "gold_triples",
                "predicted_triples",
            }
            missing = sorted(required - set(row))
            if missing:
                raise PhaseEError(
                    f"Inference line {line_number} lacks required keys: {missing}"
                )
            for triple in row["gold_triples"]:
                if not {"head_text", "tail_text", "relation"} <= set(triple):
                    raise PhaseEError(
                        f"Gold triple on line {line_number} has an incompatible schema"
                    )
            for triple in row["predicted_triples"]:
                required_triple = {"head_text", "tail_text", "relation", "triple_conf"}
                if not required_triple <= set(triple):
                    raise PhaseEError(
                        f"Predicted triple on line {line_number} has an incompatible schema"
                    )
            rows.append(row)

    expected_records = contract["artifacts"]["inference"]["records"]
    if len(rows) != expected_records:
        raise PhaseEError(
            f"Inference record count mismatch: expected {expected_records}, got {len(rows)}"
        )
    evaluator = load_evaluator()
    questions = evaluator.generate_questions(
        rows, contract["evaluator"]["max_questions"]
    )
    if len(questions) != contract["evaluator"]["max_questions"]:
        raise PhaseEError(
            "Frozen input does not generate exactly ten table-era questions; "
            f"got {len(questions)}"
        )
    return rows, {
        "records": len(rows),
        "questions": len(questions),
        "question_sha256": hashlib.sha256(
            canonical_json_bytes([q["question"] for q in questions])
        ).hexdigest(),
    }


def validate_graph(
    path: Path,
    expected_nodes: int,
    expected_edges: int,
    expected_metadata: dict | None = None,
) -> dict:
    graph = load_json(path)
    if not isinstance(graph, dict):
        raise PhaseEError(f"Graph root must be an object: {path}")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise PhaseEError(f"Graph must contain node and edge arrays: {path}")
    if len(nodes) != expected_nodes or len(edges) != expected_edges:
        raise PhaseEError(
            f"Graph shape mismatch for {path}: expected "
            f"{expected_nodes} nodes/{expected_edges} edges, got "
            f"{len(nodes)} nodes/{len(edges)} edges"
        )
    for index, edge in enumerate(edges):
        if not {"head", "relation", "tail"} <= set(edge):
            raise PhaseEError(f"Graph edge {index} has an incompatible schema: {path}")
    metadata = graph.get("metadata", {})
    if expected_metadata is not None:
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise PhaseEError(
                f"Graph metadata mismatch for {path}: "
                + json.dumps(mismatches, sort_keys=True)
            )
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "sha256": sha256_file(path),
        "metadata": {key: metadata.get(key) for key in (expected_metadata or {})},
    }


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "phase_e_eval_graph_rag", ROOT / "eval_graph_rag.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_environment(inference_size: int) -> dict:
    if sys.version_info < (3, 10):
        raise PhaseEError("Phase E requires Python 3.10 or newer")
    curl = shutil.which("curl")
    if not curl:
        raise PhaseEError("curl is required by the frozen evaluator but was not found")
    storage = shutil.disk_usage(OUTPUT_ROOT.parent)
    required_free = max(100 * 1024 * 1024, inference_size * 4)
    if storage.free < required_free:
        raise PhaseEError(
            f"Insufficient free storage: require at least {required_free} bytes, "
            f"found {storage.free}"
        )
    return {
        "python": platform.python_version(),
        "curl": str(Path(curl).resolve()),
        "free_bytes": storage.free,
        "required_free_bytes": required_free,
    }


def load_verifier_model_identity(path: Path, expected_hash: str) -> dict:
    expected_hash = validate_sha256(
        expected_hash, "--verifier-environment-sha256"
    )
    validate_input_file(path, expected_hash, "verifier environment manifest")
    manifest = load_json(path)
    if not isinstance(manifest, dict) or manifest.get("execution_mode") != "live":
        raise PhaseEError(
            "Verifier environment manifest must describe a completed live verifier run"
        )
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("identity_verified") is not True:
        raise PhaseEError(
            "Verifier environment manifest lacks a verified model identity"
        )
    name = model.get("name")
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise PhaseEError("Verifier environment manifest has an invalid model tag")

    tag_digest = validate_sha256(
        str(model.get("tag_digest", "")).removeprefix("sha256:"),
        "verifier model tag digest",
    )
    registry_digest = validate_sha256(
        str(model.get("registry_manifest_sha256", "")).removeprefix("sha256:"),
        "verifier registry manifest digest",
    )
    if tag_digest != registry_digest:
        raise PhaseEError(
            "Verifier environment manifest contains conflicting model tag digests"
        )

    blob_sha256 = validate_sha256(
        str(model.get("blob_sha256", "")).removeprefix("sha256:"),
        "verifier model blob SHA-256",
    )
    model_blob_sha256 = validate_sha256(
        str(model.get("model_blob_sha256", "")).removeprefix("sha256:"),
        "verifier recorded model blob SHA-256",
    )
    if blob_sha256 != model_blob_sha256:
        raise PhaseEError(
            "Verifier environment manifest contains conflicting model blob hashes"
        )

    details = model.get("details")
    if not isinstance(details, dict):
        raise PhaseEError("Verifier environment manifest lacks model details")
    expected_details = {
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }
    if any(
        not isinstance(value, str) or not value
        for value in expected_details.values()
    ):
        raise PhaseEError("Verifier environment manifest has incomplete model details")
    expected_details["family"] = expected_details["family"].lower()
    return {
        "name": name,
        "tag_digest": tag_digest,
        "blob_sha256": blob_sha256,
        "details": expected_details,
        "verifier_environment_sha256": expected_hash,
        "verifier_condition_id": manifest.get("condition_id"),
        "verifier_protocol_id": manifest.get("protocol_id"),
    }


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
    method: str, url: str, payload: dict | None = None
) -> tuple[dict, bytes, str]:
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
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseEError(
            f"Ollama identity endpoint returned invalid JSON: {url}"
        ) from exc
    if not isinstance(manifest, dict):
        raise PhaseEError(f"Ollama identity endpoint did not return an object: {url}")
    canonical = canonical_json_bytes(manifest)
    return manifest, canonical, hashlib.sha256(canonical).hexdigest()


def fetch_matching_model(ollama_url: str, expected: dict) -> tuple[dict, bytes, bytes]:
    origin = validate_ollama_url(ollama_url)
    tags, tags_bytes, tags_hash = fetch_ollama_json("GET", origin + "/api/tags")
    models = tags.get("models")
    if not isinstance(models, list):
        raise PhaseEError("Ollama /api/tags response lacks a models array")
    matches = [
        item
        for item in models
        if isinstance(item, dict)
        and expected["name"] in {item.get("name"), item.get("model")}
    ]
    if len(matches) != 1:
        raise PhaseEError(
            "Ollama tag inventory does not uniquely contain the verifier model tag"
        )
    observed_tag_digest = validate_sha256(
        str(matches[0].get("digest", "")).removeprefix("sha256:"),
        "Ollama model tag digest",
    )
    if observed_tag_digest != expected["tag_digest"]:
        raise PhaseEError(
            "Ollama model tag digest differs from the completed verifier run"
        )

    show, show_bytes, show_hash = fetch_ollama_json(
        "POST", origin + "/api/show", {"model": expected["name"]}
    )
    details = show.get("details")
    if not isinstance(details, dict):
        raise PhaseEError("Ollama /api/show response lacks model details")
    observed_details = {
        "family": str(details.get("family", "")).lower(),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }
    if observed_details != expected["details"]:
        raise PhaseEError(
            "Ollama model details differ from the completed verifier run: "
            + json.dumps(
                {"expected": expected["details"], "actual": observed_details},
                sort_keys=True,
            )
        )
    modelfile = show.get("modelfile")
    if not isinstance(modelfile, str):
        raise PhaseEError("Ollama /api/show response lacks a Modelfile")
    from_lines = [
        line
        for line in modelfile.splitlines()
        if line.lstrip().upper().startswith("FROM ")
    ]
    observed_blobs = {
        match.lower()
        for line in from_lines
        for match in MODEL_BLOB_RE.findall(line)
    }
    if observed_blobs != {expected["blob_sha256"]}:
        raise PhaseEError(
            "Ollama Modelfile blob identity differs from the completed verifier run"
        )
    return (
        {
            "identity_verified": True,
            "name": expected["name"],
            "tag_digest": observed_tag_digest,
            "blob_sha256": expected["blob_sha256"],
            "details": observed_details,
            "verifier_environment_sha256": expected[
                "verifier_environment_sha256"
            ],
            "verifier_condition_id": expected["verifier_condition_id"],
            "verifier_protocol_id": expected["verifier_protocol_id"],
            "tags_response_sha256": tags_hash,
            "show_response_sha256": show_hash,
        },
        tags_bytes,
        show_bytes,
    )


def validate_run_id(run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise PhaseEError(
            "--run-id must be 1-80 characters using letters, digits, dot, underscore, "
            "or hyphen, and must start with a letter or digit"
        )
    run_dir = (OUTPUT_ROOT / run_id).resolve()
    if run_dir.parent != OUTPUT_ROOT:
        raise PhaseEError(f"Derived run path escapes output root: {run_dir}")
    return run_dir


def write_bytes_once(path: Path, content: bytes) -> None:
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


def write_json_once(path: Path, value) -> None:
    write_bytes_once(path, canonical_json_bytes(value))


def write_status(path: Path, status: dict) -> None:
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


def copy_input(source: Path, destination: Path, expected_hash: str) -> None:
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_hash:
            raise PhaseEError(f"Existing copied input is incompatible: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        copied_hash = sha256_file(temporary)
        if copied_hash != expected_hash:
            raise PhaseEError(
                f"Copied input hash changed for {source}: got {copied_hash}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_rag_output(path: Path, contract: dict, model_name: str) -> dict:
    output = load_json(path)
    modes = contract["evaluator"]["modes"]
    expected_questions = contract["evaluator"]["max_questions"]
    if output.get("metadata", {}).get("n_questions") != expected_questions:
        raise PhaseEError(f"RAG output has the wrong question count: {path}")
    if output.get("metadata", {}).get("model") != model_name:
        raise PhaseEError(f"RAG output has the wrong model tag: {path}")
    if list(output.get("accuracy", {})) != modes:
        raise PhaseEError(f"RAG output accuracy keys/order changed: {path}")
    if list(output.get("correct_counts", {})) != modes:
        raise PhaseEError(f"RAG output count keys/order changed: {path}")
    if list(output.get("results", {})) != modes:
        raise PhaseEError(f"RAG output result keys/order changed: {path}")
    for mode in modes:
        records = output["results"][mode]
        if len(records) != expected_questions:
            raise PhaseEError(f"RAG output mode {mode} has the wrong record count")
        if any(
            not isinstance(record.get("pred"), str)
            or not record["pred"].strip()
            or record["pred"].startswith("ERROR:")
            for record in records
        ):
            raise PhaseEError(f"RAG output mode {mode} contains failed model calls")
        count = sum(bool(record.get("correct")) for record in records)
        if output["correct_counts"][mode] != count:
            raise PhaseEError(f"RAG output mode {mode} has inconsistent counts")
        if output["accuracy"][mode] != count / expected_questions:
            raise PhaseEError(f"RAG output mode {mode} has inconsistent accuracy")
    return {
        "sha256": sha256_file(path),
        "accuracy": output["accuracy"],
        "correct_counts": output["correct_counts"],
    }


def run_stage(
    name: str,
    command: list[str],
    output_path: Path,
    validator,
    run_dir: Path,
    status: dict,
    status_path: Path,
):
    stages = status.setdefault("stages", {})
    stage = stages.setdefault(name, {"attempts": []})
    if stage.get("status") == "complete":
        expected = stage.get("output_sha256")
        if not output_path.is_file() or sha256_file(output_path) != expected:
            raise PhaseEError(f"Completed stage output no longer matches: {name}")
        validator(output_path)
        return stage["validation"]
    if output_path.exists():
        validation = validator(output_path)
        stage.update(
            status="complete",
            recovered_after_interruption=True,
            output_sha256=sha256_file(output_path),
            validation=validation,
        )
        write_status(status_path, status)
        return validation

    attempt_number = len(stage["attempts"]) + 1
    log_path = run_dir / "logs" / f"{name}-attempt-{attempt_number}.log"
    if log_path.exists():
        raise PhaseEError(f"Attempt log already exists: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = {
        "number": attempt_number,
        "started_at": utc_now(),
        "command": command,
        "log": str(log_path.relative_to(run_dir)),
    }
    stage["attempts"].append(attempt)
    stage["status"] = "running"
    write_status(status_path, status)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    attempt["finished_at"] = utc_now()
    attempt["returncode"] = completed.returncode
    attempt["log_sha256"] = sha256_file(log_path)
    if completed.returncode != 0 or not output_path.is_file():
        stage["status"] = "failed"
        write_status(status_path, status)
        raise PhaseEError(
            f"Stage {name} failed with exit code {completed.returncode}; see {log_path}"
        )
    try:
        validation = validator(output_path)
    except PhaseEError:
        stage["status"] = "failed-validation"
        write_status(status_path, status)
        raise
    stage.update(
        status="complete",
        output_sha256=sha256_file(output_path),
        validation=validation,
    )
    write_status(status_path, status)
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "validate-contract", help="Validate frozen source blobs without running models"
    )

    inspect_model = subparsers.add_parser(
        "inspect-model", help="Verify Ollama against a completed verifier manifest"
    )
    inspect_model.add_argument("--ollama-url", default="http://localhost:11434")
    inspect_model.add_argument(
        "--verifier-environment-manifest", required=True, type=Path
    )
    inspect_model.add_argument("--verifier-environment-sha256", required=True)

    run = subparsers.add_parser("run", help="Run only the downstream E-T02 stages")
    run.add_argument("--run-id", required=True)
    run.add_argument("--inference", required=True, type=Path)
    run.add_argument("--confidence-graph", type=Path)
    verified = run.add_mutually_exclusive_group(required=True)
    verified.add_argument("--verified-graph", type=Path)
    verified.add_argument(
        "--block-verified",
        action="store_true",
        help="Explicitly retain the verified condition as blocked",
    )
    run.add_argument("--verified-graph-sha256")
    run.add_argument("--ollama-url", default="http://localhost:11434")
    run.add_argument("--verifier-environment-manifest", required=True, type=Path)
    run.add_argument("--verifier-environment-sha256", required=True)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source and artifact inputs without contacting Ollama or writing output",
    )
    return parser


def inspect_model_command(args) -> int:
    verifier_environment = args.verifier_environment_manifest.expanduser().resolve()
    expected = load_verifier_model_identity(
        verifier_environment, args.verifier_environment_sha256
    )
    evidence, _tags_bytes, _show_bytes = fetch_matching_model(
        args.ollama_url, expected
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def run_command(args, contract: dict) -> int:
    source = validate_source_contract(contract, require_clean=not args.dry_run)
    run_dir = validate_run_id(args.run_id)
    inference = args.inference.expanduser().resolve()
    inference_hash = contract["artifacts"]["inference"]["sha256"]
    validate_input_file(inference, inference_hash, "inference")
    _rows, inference_summary = load_inference(inference, contract)
    environment_summary = validate_environment(inference.stat().st_size)
    verifier_environment = args.verifier_environment_manifest.expanduser().resolve()
    model_identity = load_verifier_model_identity(
        verifier_environment, args.verifier_environment_sha256
    )

    confidence = None
    confidence_summary = None
    if args.confidence_graph:
        confidence = args.confidence_graph.expanduser().resolve()
        expected = contract["artifacts"]["confidence_graph"]["retained_sha256"]
        validate_input_file(confidence, expected, "confidence graph")
        confidence_summary = validate_graph(
            confidence,
            contract["artifacts"]["confidence_graph"]["nodes"],
            contract["artifacts"]["confidence_graph"]["edges"],
            contract["artifacts"]["confidence_graph"]["metadata"],
        )

    verified = None
    verified_hash = None
    verified_summary = None
    if args.verified_graph:
        if not args.verified_graph_sha256:
            raise PhaseEError(
                "--verified-graph-sha256 is required with --verified-graph because no "
                "canonical repository hash survives"
            )
        verified_hash = validate_sha256(
            args.verified_graph_sha256, "--verified-graph-sha256"
        )
        verified = args.verified_graph.expanduser().resolve()
        validate_input_file(verified, verified_hash, "verified graph")
        verified_summary = validate_graph(
            verified,
            contract["artifacts"]["verified_graph"]["nodes"],
            contract["artifacts"]["verified_graph"]["edges"],
        )
    elif args.verified_graph_sha256:
        raise PhaseEError(
            "--verified-graph-sha256 cannot be used with --block-verified"
        )

    plan = {
        "source": source,
        "run_dir": str(run_dir),
        "environment": environment_summary,
        "inference": inference_summary,
        "confidence": confidence_summary or "build-from-frozen-inference",
        "verified": verified_summary or "blocked-by-explicit-flag",
        "gold": "build-inside-table-era-evaluator",
        "model_identity": model_identity,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    model_evidence, tags_bytes, show_bytes = fetch_matching_model(
        args.ollama_url, model_identity
    )

    contract_hash = sha256_file(CONTRACT_PATH)
    identity = {
        "schema_version": 1,
        "run_id": args.run_id,
        "source_head": source["head"],
        "contract_sha256": contract_hash,
        "inference_sha256": inference_hash,
        "confidence_graph_sha256": confidence_summary["sha256"] if confidence else None,
        "verified_graph_sha256": verified_hash,
        "verified_blocked": bool(args.block_verified),
        "ollama_url": args.ollama_url.rstrip("/"),
        "ollama_model": model_identity["name"],
        "ollama_tag_digest": model_identity["tag_digest"],
        "ollama_model_blob_sha256": model_identity["blob_sha256"],
        "verifier_environment_sha256": model_identity[
            "verifier_environment_sha256"
        ],
    }
    status_path = run_dir / "run-status.json"
    if run_dir.exists():
        if not status_path.is_file():
            raise PhaseEError(
                f"Run directory exists without a resumable status file: {run_dir}"
            )
        status = load_json(status_path)
        if status.get("identity") != identity:
            raise PhaseEError("Existing run identity differs; choose a new --run-id")
    else:
        run_dir.mkdir(parents=True)
        status = {
            "schema_version": 1,
            "identity": identity,
            "status": "running",
            "created_at": utc_now(),
            "stages": {},
        }
        write_status(status_path, status)

    copied_inference = run_dir / "inputs" / "inference.jsonl"
    copy_input(inference, copied_inference, inference_hash)
    copy_input(
        verifier_environment,
        run_dir / "inputs" / "verifier-environment-manifest.json",
        model_identity["verifier_environment_sha256"],
    )
    model_check_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    model_check_dir = run_dir / "logs" / "model-identity-checks" / model_check_id
    write_bytes_once(model_check_dir / "ollama-tags.json", tags_bytes)
    write_bytes_once(model_check_dir / "ollama-show.json", show_bytes)
    write_json_once(model_check_dir / "result.json", model_evidence)
    write_json_once(
        run_dir / "method-manifest.json",
        {
            "contract": contract,
            "contract_sha256": contract_hash,
            "resolved_model_identity": model_identity,
        },
    )
    write_json_once(
        run_dir / "environment.json",
        {
            "captured_at": status["created_at"],
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "preflight": environment_summary,
            "source": source,
            "model_identity": model_identity,
        },
    )

    graph_dir = run_dir / "graphs"
    result_dir = run_dir / "rag-results"
    graph_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    confidence_output = graph_dir / "confidence-0.7.json"
    if confidence:
        copy_input(
            confidence,
            confidence_output,
            contract["artifacts"]["confidence_graph"]["retained_sha256"],
        )
        status["stages"]["confidence_graph"] = {
            "status": "complete",
            "source": "compatible-completed-graph",
            "output_sha256": sha256_file(confidence_output),
            "validation": confidence_summary,
        }
        write_status(status_path, status)
    else:
        run_stage(
            "confidence_graph",
            [
                sys.executable,
                "-B",
                str(ROOT / "build_kg.py"),
                "--input",
                str(copied_inference),
                "--output",
                str(confidence_output),
                "--filter-mode",
                contract["graph_builder"]["confidence_filter_mode"],
                "--conf-threshold",
                str(contract["graph_builder"]["confidence_threshold"]),
            ],
            confidence_output,
            lambda path: validate_graph(
                path,
                contract["artifacts"]["confidence_graph"]["nodes"],
                contract["artifacts"]["confidence_graph"]["edges"],
                contract["artifacts"]["confidence_graph"]["metadata"],
            ),
            run_dir,
            status,
            status_path,
        )

    evaluator_base = [
        sys.executable,
        "-B",
        str(ROOT / "eval_graph_rag.py"),
        "--gold-jsonl",
        str(copied_inference),
        "--ollama-url",
        args.ollama_url.rstrip("/"),
        "--ollama-model",
        model_identity["name"],
        "--max-questions",
        str(contract["evaluator"]["max_questions"]),
    ]
    condition_results = {}
    confidence_rag = result_dir / "confidence.json"
    condition_results["confidence"] = run_stage(
        "rag_confidence",
        evaluator_base
        + ["--kg", str(confidence_output), "--output", str(confidence_rag)],
        confidence_rag,
        lambda path: validate_rag_output(path, contract, model_identity["name"]),
        run_dir,
        status,
        status_path,
    )

    if verified:
        copied_verified = run_dir / "inputs" / "verified-graph.json"
        copy_input(verified, copied_verified, verified_hash)
        verified_rag = result_dir / "verified.json"
        condition_results["verified"] = run_stage(
            "rag_verified",
            evaluator_base
            + ["--kg", str(copied_verified), "--output", str(verified_rag)],
            verified_rag,
            lambda path: validate_rag_output(path, contract, model_identity["name"]),
            run_dir,
            status,
            status_path,
        )
    else:
        status["stages"]["rag_verified"] = {
            "status": "blocked",
            "reason": "No compatible completed 48-node/31-edge verified graph supplied",
        }
        condition_results["verified"] = status["stages"]["rag_verified"]
        write_status(status_path, status)

    gold_rag = result_dir / "gold.json"
    condition_results["gold"] = run_stage(
        "rag_gold",
        evaluator_base
        + [
            "--kg",
            str(confidence_output),
            "--use-gold-kg",
            "--output",
            str(gold_rag),
        ],
        gold_rag,
        lambda path: validate_rag_output(path, contract, model_identity["name"]),
        run_dir,
        status,
        status_path,
    )

    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "new_results": condition_results,
        "legacy_displayed_accuracy": contract["legacy_table2_accuracy"],
        "note": (
            "Legacy values are comparison references only; differences must not trigger "
            "method or prompt tuning."
        ),
    }
    write_json_once(run_dir / "table2-results.json", summary)
    hashes = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"artifact-hashes.json", "run-status.json"}:
            hashes[str(path.relative_to(run_dir)).replace("\\", "/")] = sha256_file(path)
    write_json_once(run_dir / "artifact-hashes.json", hashes)
    status["status"] = (
        "complete_with_verified_blocked" if args.block_verified else "complete"
    )
    status["finished_at"] = utc_now()
    write_status(status_path, status)
    print(json.dumps({"run_dir": str(run_dir), "status": status["status"]}, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        contract = load_json(CONTRACT_PATH)
        if args.command == "validate-contract":
            print(json.dumps(validate_source_contract(contract, False), indent=2))
            return 0
        if args.command == "inspect-model":
            return inspect_model_command(args)
        return run_command(args, contract)
    except PhaseEError as exc:
        print(f"Phase E blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
