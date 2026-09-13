"""Fail-closed same-run Phase E runner for the frozen downstream E-T02 workflow."""

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
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = (ROOT / "output").resolve()
CONTRACT_PATH = ROOT / "phase_e_rag_contract.json"
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
    rows: list[dict[str, Any]] = []
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
    return rows


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + suffix
    ).encode("utf-8")


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
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PhaseEError(f"git {' '.join(args)} failed: {detail}")
    return result


def git_blob(path: str) -> str:
    return run_git("hash-object", f"--path={path}", path).stdout.strip()


def validate_source_contract(
    contract: dict[str, Any], require_clean: bool
) -> dict[str, str]:
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
    frozen = {
        **contract["frozen_upstream_blobs"],
        "eval_graph_rag.py": contract["evaluator"]["post_bf02_blob"],
    }
    frozen.update(
        {
            path: specification["current_blob"]
            for path, specification in contract[
                "same_run_interface_adaptations"
            ].items()
        }
    )
    for path, expected in frozen.items():
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


def validate_file(
    path: Path, expected_hash: str, expected_bytes: int | None, label: str
) -> None:
    if not path.is_file():
        raise PhaseEError(f"{label} is not a file: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise PhaseEError(
            f"{label} byte count mismatch: expected {expected_bytes}, got {path.stat().st_size}"
        )
    actual = sha256_file(path)
    if actual != expected_hash:
        raise PhaseEError(f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual}")


def artifact_index(
    document: dict[str, Any], label: str
) -> dict[str, dict[str, Any]]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise PhaseEError(f"{label} lacks an artifacts array")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict) or not {"path", "sha256", "bytes"} <= set(entry):
            raise PhaseEError(f"{label} has a malformed artifact entry")
        path = entry["path"]
        if not isinstance(path, str) or path in indexed:
            raise PhaseEError(f"{label} has a duplicate or invalid artifact path")
        validate_sha256(str(entry["sha256"]), f"{label} {path} hash")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise PhaseEError(f"{label} {path} has an invalid byte count")
        indexed[path] = entry
    return indexed


def validate_parent_lineage(
    run_dir: Path, contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    same_run = contract["same_run"]
    parent = same_run["parent"]
    ledger_path = run_file(run_dir, parent["artifact_ledger"]["path"])
    validate_file(
        ledger_path,
        parent["artifact_ledger"]["sha256"],
        parent["artifact_ledger"].get("bytes"),
        "parent artifact ledger",
    )
    ledger = load_json(ledger_path)
    if not isinstance(ledger, dict):
        raise PhaseEError("Parent artifact ledger must be an object")
    if (
        ledger.get("schema_version") != "rag-phase-b-parent-artifacts-1.0"
        or ledger.get("run_id") != same_run["expected_run_id"]
        or ledger.get("score_manifest_sha256")
        != parent["score_manifest"]["sha256"]
    ):
        raise PhaseEError("Parent artifact ledger carries a different run identity")
    artifacts = ledger.get("artifacts")
    set_hash = sha256_bytes(canonical_json_bytes(artifacts, newline=False))
    if (
        ledger.get("artifact_set_sha256") != set_hash
        or set_hash != parent["artifact_set_sha256"]
    ):
        raise PhaseEError("Parent artifact-set identity does not match the selected run")
    indexed = artifact_index(ledger, "parent artifact ledger")
    for relative, entry in indexed.items():
        validate_file(
            run_file(run_dir, relative),
            entry["sha256"],
            entry["bytes"],
            f"parent artifact {relative}",
        )

    full_run_path = run_file(run_dir, same_run["full_run"]["path"])
    validate_file(
        full_run_path,
        same_run["full_run"]["sha256"],
        same_run["full_run"].get("bytes"),
        "full-run manifest",
    )
    full_run = load_json(full_run_path)
    if (
        full_run.get("schema_version") != "phase-b-debug-full-run-1.0"
        or full_run.get("run_id") != same_run["expected_run_id"]
        or full_run.get("protocol_id") != same_run["protocol_id"]
        or full_run.get("workflow_id") != same_run["workflow_id"]
        or full_run.get("training_seeds") != list(range(42, 50))
    ):
        raise PhaseEError("Full-run manifest differs from the selected run identity")

    checkout_path = run_file(run_dir, same_run["checkout"]["path"])
    validate_file(
        checkout_path,
        same_run["checkout"]["sha256"],
        same_run["checkout"].get("bytes"),
        "checkout manifest",
    )
    checkout = load_json(checkout_path)
    if (
        checkout.get("run_id") != same_run["expected_run_id"]
        or checkout.get("status") != "pass"
        or checkout.get("source", {}).get("worktree_clean") is not True
    ):
        raise PhaseEError("Selected run lacks its passing same-run checkout identity")

    score_path = run_file(run_dir, parent["score_manifest"]["path"])
    validate_file(
        score_path,
        parent["score_manifest"]["sha256"],
        parent["score_manifest"].get("bytes"),
        "score manifest",
    )
    score = load_json(score_path)
    for field in ("run_id", "protocol_id", "workflow_id", "matcher_id"):
        expected = same_run["expected_run_id"] if field == "run_id" else same_run[field]
        if score.get(field) != expected:
            raise PhaseEError(f"Score manifest {field} differs from same-run identity")

    prep_path = run_file(run_dir, same_run["preparation_manifest"]["path"])
    validate_file(
        prep_path,
        same_run["preparation_manifest"]["sha256"],
        same_run["preparation_manifest"].get("bytes"),
        "data-preparation manifest",
    )
    preparation = load_json(prep_path)
    if (
        preparation.get("schema_version") != "phase-b-data-preparation-manifest-2.0"
        or preparation.get("protocol_id") != same_run["protocol_id"]
        or preparation.get("dataset_id") != "CODE-ACCORD-v1.0.0"
        or preparation.get("split", {}).get("seed") != same_run["seed"]
        or preparation.get("split", {}).get("test") != 173
    ):
        raise PhaseEError("Data-preparation manifest differs from the selected run")
    prep_artifacts = artifact_index(preparation, "data-preparation manifest")
    for name in ("test.jsonl", "test-gold.jsonl", "split-manifest.json"):
        expected = same_run["prepared"][name]
        observed = prep_artifacts.get(name)
        if observed is None or any(
            observed.get(key) != expected[key] for key in ("sha256", "bytes")
        ):
            raise PhaseEError(f"Data-preparation manifest does not bind {name}")

    threshold_path = run_file(run_dir, same_run["threshold"]["path"])
    validate_file(
        threshold_path,
        same_run["threshold"]["sha256"],
        same_run["threshold"].get("bytes"),
        "threshold selection",
    )
    threshold = load_json(threshold_path)
    if (
        threshold.get("selected_threshold") != same_run["threshold"]["selected"]
        or threshold.get("used_test_labels") is not False
        or threshold.get("split_manifest_sha256")
        != same_run["prepared"]["split-manifest.json"]["sha256"]
        or threshold.get("development_candidate_index_sha256")
        != indexed["predictions/dev/candidate-index.json"]["sha256"]
        or threshold.get("development_gold_sha256")
        != indexed["data-prepared/development-gold.jsonl"]["sha256"]
    ):
        raise PhaseEError("Development threshold selection is not the approved same-run value")

    verifier = same_run["corrective_verifier"]
    verifier_path = run_file(run_dir, verifier["manifest"]["path"])
    validate_file(
        verifier_path,
        verifier["manifest"]["sha256"],
        verifier["manifest"].get("bytes"),
        "corrective-verifier stage manifest",
    )
    verifier_manifest = load_json(verifier_path)
    if (
        verifier_manifest.get("condition_id") != "VER-CORRECTIVE"
        or verifier_manifest.get("execution_mode") != "live"
        or verifier_manifest.get("status") != "completed"
    ):
        raise PhaseEError("Corrective-verifier manifest is not one completed live stage")
    for relative, expected in verifier["required_bindings"].items():
        containers = (
            verifier_manifest.get("inputs", {}),
            verifier_manifest.get("outputs", {}),
        )
        if not any(container.get(relative) == expected for container in containers):
            raise PhaseEError(f"Corrective-verifier manifest does not bind {relative}")

    model_bindings = {
        relative: digest
        for relative, digest in verifier["required_bindings"].items()
        if relative.startswith("inputs/ollama/blobs/sha256-")
    }
    if len(model_bindings) != 1:
        raise PhaseEError("Corrective-verifier lineage must bind one model blob")
    model_relative, model_digest = next(iter(model_bindings.items()))
    validate_file(
        run_file(run_dir, model_relative),
        validate_sha256(model_digest, "same-run model blob hash"),
        verifier["model_blob_bytes"],
        "same-run model blob",
    )
    return indexed


def validate_sharded_ledger_artifact(
    run_dir: Path,
    graph_root: str,
    relative: str,
    entry: dict[str, Any],
) -> None:
    """Validate the one approved transport-sharded JSONL as its original bytes."""
    if relative != "traces/retrieval.jsonl":
        raise PhaseEError(f"Graph artifact is missing and has no sharding contract: {relative}")
    manifest_relative = f"{graph_root}/traces/retrieval.parts.json"
    manifest_path = run_file(run_dir, manifest_relative)
    manifest = load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "graph-rag-jsonl-shards-1.0"
        or manifest.get("logical_path") != relative
        or manifest.get("representation") != "transport_sharding_only"
        or manifest.get("join_mode") != "byte_concatenation_in_listed_order"
        or manifest.get("record_format") != "jsonl"
        or manifest.get("split_boundary") != "complete_jsonl_record"
        or manifest.get("encoding") != "utf-8"
    ):
        raise PhaseEError("Retrieval shard manifest has changed its transport contract")
    original = manifest.get("original")
    if not isinstance(original, dict) or any(
        original.get(key) != entry[key] for key in ("sha256", "bytes")
    ):
        raise PhaseEError("Retrieval shard manifest does not bind the graph ledger entry")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise PhaseEError("Retrieval shard manifest has no parts")
    digest = hashlib.sha256()
    observed_bytes = 0
    observed_records = 0
    expected_first = 1
    traces_dir = Path(graph_root) / "traces"
    for part in parts:
        if not isinstance(part, dict) or set(part) != {
            "bytes",
            "first_record",
            "last_record",
            "path",
            "records",
            "sha256",
        }:
            raise PhaseEError("Retrieval shard entry is malformed")
        records = part.get("records")
        first = part.get("first_record")
        last = part.get("last_record")
        if (
            not isinstance(records, int)
            or records <= 0
            or first != expected_first
            or last != first + records - 1
        ):
            raise PhaseEError("Retrieval shard record boundaries are not contiguous")
        part_relative = str(traces_dir / str(part.get("path", ""))).replace("\\", "/")
        part_path = run_file(run_dir, part_relative)
        validate_file(
            part_path,
            validate_sha256(str(part.get("sha256", "")), "retrieval shard hash"),
            part.get("bytes"),
            f"retrieval shard {part_relative}",
        )
        newline_count = 0
        with part_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                newline_count += chunk.count(b"\n")
                observed_bytes += len(chunk)
        if newline_count != records:
            raise PhaseEError(f"Retrieval shard record count differs: {part_relative}")
        observed_records += records
        expected_first = last + 1
    if (
        observed_bytes != entry["bytes"]
        or digest.hexdigest() != entry["sha256"]
        or observed_records != original.get("records")
    ):
        raise PhaseEError("Retrieval shards do not reconstruct the ledger artifact")


def validate_graph_child(
    run_dir: Path, contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    same_run = contract["same_run"]
    child = same_run["graph_child"]
    identity_path = run_file(run_dir, child["identity"]["path"])
    validate_file(
        identity_path,
        child["identity"]["sha256"],
        child["identity"].get("bytes"),
        "graph child identity",
    )
    identity = load_json(identity_path)
    if (
        identity.get("run_id") != same_run["expected_run_id"]
        or identity.get("protocol_id") != child["protocol_id"]
        or identity.get("parent", {}).get("artifact_set_sha256")
        != same_run["parent"]["artifact_set_sha256"]
        or identity.get("parent", {}).get("score_manifest_sha256")
        != same_run["parent"]["score_manifest"]["sha256"]
    ):
        raise PhaseEError("Graph child identity is not bound to the selected parent run")

    complete_path = run_file(run_dir, child["complete"]["path"])
    validate_file(
        complete_path,
        child["complete"]["sha256"],
        child["complete"].get("bytes"),
        "graph completion manifest",
    )
    complete = load_json(complete_path)
    if complete.get("run_id") != same_run["expected_run_id"]:
        raise PhaseEError("Graph completion manifest carries another run identity")

    ledger_path = run_file(run_dir, child["artifact_ledger"]["path"])
    validate_file(
        ledger_path,
        child["artifact_ledger"]["sha256"],
        child["artifact_ledger"].get("bytes"),
        "graph artifact ledger",
    )
    if complete.get("artifact_manifest_sha256") != child["artifact_ledger"]["sha256"]:
        raise PhaseEError("Graph completion manifest does not bind its artifact ledger")
    ledger = load_json(ledger_path)
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != "rag-phase-b-diagnostic-artifacts-1.0"
        or ledger.get("run_id") != same_run["expected_run_id"]
    ):
        raise PhaseEError("Graph artifact ledger carries a different run identity")
    indexed = artifact_index(ledger, "graph artifact ledger")
    graph_root = child["root"]
    for relative, entry in indexed.items():
        artifact_path = run_file(run_dir, f"{graph_root}/{relative}", required=False)
        if artifact_path.is_file() and artifact_path.resolve() == artifact_path:
            validate_file(
                artifact_path,
                entry["sha256"],
                entry["bytes"],
                f"graph artifact {relative}",
            )
        else:
            validate_sharded_ledger_artifact(run_dir, graph_root, relative, entry)
    return indexed


def validate_graph_snapshot(
    run_dir: Path,
    graph_root: str,
    specification: dict[str, Any],
    graph_artifacts: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    relative_manifest = specification["manifest"]
    manifest_path = run_file(run_dir, relative_manifest)
    ledger_manifest = graph_artifacts.get(relative_manifest.removeprefix(graph_root + "/"))
    if ledger_manifest is None:
        raise PhaseEError(f"Graph artifact ledger omits {relative_manifest}")
    validate_file(
        manifest_path,
        specification["manifest_sha256"],
        ledger_manifest["bytes"],
        f"{specification['condition']} graph manifest",
    )
    manifest = load_json(manifest_path)
    expected_scalars = {
        "schema_version": "rag-graph-snapshot-1.0",
        "condition": specification["condition"],
        "construction_recipe": specification["construction_recipe"],
        "graph_id": specification["graph_id"],
    }
    for field, expected in expected_scalars.items():
        if manifest.get(field) != expected:
            raise PhaseEError(f"Graph manifest {field} differs for {specification['condition']}")

    collections: dict[str, list[dict[str, Any]]] = {}
    for collection in ("entities", "relations", "triples"):
        relative = str(
            Path(relative_manifest).parent / f"{collection}.jsonl"
        ).replace("\\", "/")
        entry = graph_artifacts.get(relative.removeprefix(graph_root + "/"))
        if entry is None:
            raise PhaseEError(f"Graph artifact ledger omits {relative}")
        path = run_file(run_dir, relative)
        validate_file(path, entry["sha256"], entry["bytes"], f"graph {collection}")
        rows = load_jsonl(path)
        if rows != manifest.get(collection):
            raise PhaseEError(f"Graph {collection} JSONL differs from its manifest array")
        digest = sha256_bytes(canonical_json_bytes(rows, newline=False))
        digest_field = {
            "entities": "entity_sha256",
            "relations": "relation_sha256",
            "triples": "triple_sha256",
        }[collection]
        if digest != manifest.get(digest_field):
            raise PhaseEError(f"Graph {collection} canonical hash differs from its manifest")
        count_field = {
            "entities": "entity_count",
            "relations": "relation_count",
            "triples": "triple_count",
        }[collection]
        expected_count = specification[count_field]
        if len(rows) != expected_count:
            raise PhaseEError(
                f"Graph {collection} count mismatch: expected {expected_count}, got {len(rows)}"
            )
        collections[collection] = rows

    entities = {row.get("entity_id"): row for row in collections["entities"]}
    relations = {row.get("relation_id"): row for row in collections["relations"]}
    if len(entities) != len(collections["entities"]) or None in entities:
        raise PhaseEError("Graph entities do not have unique nonempty identifiers")
    if len(relations) != len(collections["relations"]) or None in relations:
        raise PhaseEError("Graph relations do not have unique nonempty identifiers")
    nodes = [
        {
            "id": row["canonical_label"],
            "entity_id": row["entity_id"],
            "entity_type": row["entity_type"],
        }
        for row in collections["entities"]
    ]
    edges = []
    for row in collections["triples"]:
        try:
            head = entities[row["head_id"]]
            relation = relations[row["relation_id"]]
            tail = entities[row["tail_id"]]
        except KeyError as exc:
            raise PhaseEError("Graph triple refers to an unknown entity or relation") from exc
        edges.append(
            {
                "head": head["canonical_label"],
                "relation": relation["label"],
                "tail": tail["canonical_label"],
                "triple_id": row["triple_id"],
            }
        )
    projection = {
        "metadata": {
            "schema_version": "phase-e-legacy-graph-projection-1.0",
            "run_id": specification["run_id"],
            "source_graph_id": specification["graph_id"],
            "source_condition": specification["condition"],
            "representation_only": True,
        },
        "nodes": nodes,
        "edges": edges,
    }
    summary = {
        "source_graph_id": specification["graph_id"],
        "source_manifest_sha256": specification["manifest_sha256"],
        "nodes": len(nodes),
        "edges": len(edges),
    }
    return projection, canonical_json_bytes(projection), summary


def validate_contract_same_run_bindings(contract: dict[str, Any]) -> None:
    """Reject a contract that could select a different run or upstream stage."""
    same_run = contract["same_run"]
    run_id = same_run["expected_run_id"]
    seed = same_run["seed"]
    if seed != 42 or contract["evaluator"]["sampling_seed"] != seed:
        raise PhaseEError("Phase E graph and evaluator seed selection must remain seed 42")

    graphs = same_run.get("graphs", {})
    if set(graphs) != {"confidence", "corrective", "gold"}:
        raise PhaseEError("Phase E requires exactly the confidence, corrective, and gold graphs")
    expected_graphs = {
        "confidence": (
            f"confidence_filtered:seed-{seed}",
            f"graph-rag/graphs/confidence_filtered/seed-{seed}/manifest.json",
        ),
        "corrective": (
            f"corrective_verifier:seed-{seed}",
            f"graph-rag/graphs/corrective_verifier/seed-{seed}/manifest.json",
        ),
        "gold": ("gold_graph_oracle", "graph-rag/graphs/gold_graph_oracle/manifest.json"),
    }
    for name, (condition, manifest) in expected_graphs.items():
        specification = graphs[name]
        if (
            specification.get("run_id") != run_id
            or specification.get("condition") != condition
            or specification.get("manifest") != manifest
        ):
            raise PhaseEError(f"Phase E {name} graph selection mixes a run or seed")

    verifier = same_run["corrective_verifier"]
    if verifier.get("model_blob_bytes") != 20201240160:
        raise PhaseEError("Corrective-verifier model-blob byte contract differs")
    bindings = verifier.get("required_bindings", {})
    fixed_paths = {
        same_run["prepared"]["test.jsonl"]["path"],
        "predictions/test/candidates.jsonl",
        verifier["environment"]["path"],
        "verifier/corrective/verdicts.jsonl",
    }
    model_paths = [
        path
        for path in bindings
        if path.startswith("inputs/ollama/blobs/sha256-")
    ]
    if set(bindings) != fixed_paths | set(model_paths) or len(model_paths) != 1:
        raise PhaseEError(
            "Corrective-verifier bindings must name the same-run test, candidates, "
            "environment, verdicts, and one content-addressed model blob"
        )
    model_hash = model_paths[0].removeprefix("inputs/ollama/blobs/sha256-")
    if bindings[model_paths[0]] != model_hash or not SHA256_RE.fullmatch(model_hash):
        raise PhaseEError("Corrective-verifier model-blob path and digest differ")


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "phase_e_eval_graph_rag", ROOT / "eval_graph_rag.py"
    )
    if spec is None or spec.loader is None:
        raise PhaseEError("Cannot load the frozen evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_evaluator_records(
    prepared_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    if len(prepared_rows) != 173 or len(gold_rows) != 173:
        raise PhaseEError("Same-run prepared test and gold must each contain 173 records")
    gold_by_id = {row.get("example_id"): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows) or None in gold_by_id:
        raise PhaseEError("Same-run private gold has duplicate or missing example IDs")
    prepared_ids = [row.get("example_id") for row in prepared_rows]
    if (
        len(set(prepared_ids)) != len(prepared_ids)
        or set(prepared_ids) != set(gold_by_id)
        or set(prepared_ids) != set(split_manifest.get("test_ids", []))
    ):
        raise PhaseEError("Prepared text, private gold, and split test identities differ")

    projected: list[dict[str, Any]] = []
    for prepared in prepared_rows:
        example_id = prepared["example_id"]
        gold = gold_by_id[example_id]
        triples = []
        for triple in gold.get("gold_triples", []):
            try:
                triples.append(
                    {
                        "head_text": triple["head"]["text"],
                        "tail_text": triple["tail"]["text"],
                        "relation": triple["relation"],
                    }
                )
            except (KeyError, TypeError) as exc:
                raise PhaseEError(f"Private gold triple is malformed for {example_id}") from exc
        projected.append(
            {
                "doc_id": example_id,
                "sentence": prepared["content"],
                "gold_triples": triples,
                "predicted_triples": [],
            }
        )
    evaluator = load_evaluator()
    questions = evaluator.generate_questions(
        projected, contract["evaluator"]["max_questions"]
    )
    if len(questions) != contract["evaluator"]["max_questions"]:
        raise PhaseEError(
            "Same-run projection does not generate exactly ten frozen table-era questions"
        )
    summary = {
        "records": len(projected),
        "questions": len(questions),
        "question_contract": [
            {"q": item["question"], "gold": item["gold_answer"]}
            for item in questions
        ],
        "question_sha256": sha256_bytes(
            canonical_json_bytes([item["question"] for item in questions])
        ),
    }
    return projected, canonical_jsonl_bytes(projected), summary


def validate_environment(required_bytes: int) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise PhaseEError("Phase E requires Python 3.10 or newer")
    curl = shutil.which("curl")
    if not curl:
        raise PhaseEError("curl is required by the frozen evaluator but was not found")
    storage = shutil.disk_usage(ROOT)
    required_free = max(100 * 1024 * 1024, required_bytes * 4)
    if storage.free < required_free:
        raise PhaseEError(
            f"Insufficient free storage: require {required_free} bytes, found {storage.free}"
        )
    return {
        "python": platform.python_version(),
        "curl": str(Path(curl).resolve()),
        "free_bytes": storage.free,
        "required_free_bytes": required_free,
    }


def load_verifier_model_identity(path: Path, expected_hash: str) -> dict[str, Any]:
    validate_file(
        path,
        validate_sha256(expected_hash, "verifier environment hash"),
        None,
        "verifier environment manifest",
    )
    manifest = load_json(path)
    if not isinstance(manifest, dict) or manifest.get("execution_mode") != "live":
        raise PhaseEError("Verifier environment must describe a completed live verifier run")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("identity_verified") is not True:
        raise PhaseEError("Verifier environment lacks a verified model identity")
    name = model.get("name")
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise PhaseEError("Verifier environment has an invalid model tag")
    tag_digest = validate_sha256(
        str(model.get("tag_digest", "")).removeprefix("sha256:"),
        "verifier tag digest",
    )
    registry = validate_sha256(
        str(model.get("registry_manifest_sha256", "")).removeprefix("sha256:"),
        "verifier registry digest",
    )
    blob = validate_sha256(
        str(model.get("blob_sha256", "")).removeprefix("sha256:"),
        "verifier model blob",
    )
    recorded_blob = validate_sha256(
        str(model.get("model_blob_sha256", "")).removeprefix("sha256:"),
        "verifier recorded model blob",
    )
    if tag_digest != registry or blob != recorded_blob:
        raise PhaseEError("Verifier environment contains conflicting stable model identities")
    details = model.get("details")
    expected_details = {
        "family": str(details.get("family", "")).lower()
        if isinstance(details, dict)
        else "",
        "parameter_size": details.get("parameter_size")
        if isinstance(details, dict)
        else None,
        "quantization_level": details.get("quantization_level")
        if isinstance(details, dict)
        else None,
    }
    if any(not isinstance(value, str) or not value for value in expected_details.values()):
        raise PhaseEError("Verifier environment has incomplete model details")
    return {
        "name": name,
        "tag_digest": tag_digest,
        "blob_sha256": blob,
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
        item
        for item in models
        if isinstance(item, dict)
        and expected["name"] in {item.get("name"), item.get("model")}
    ]
    if len(matches) != 1:
        raise PhaseEError("Ollama tag inventory does not uniquely contain the verifier model tag")
    observed_tag = validate_sha256(
        str(matches[0].get("digest", "")).removeprefix("sha256:"),
        "Ollama tag digest",
    )
    if observed_tag != expected["tag_digest"]:
        raise PhaseEError("Ollama model tag digest differs from the completed verifier run")
    show, show_bytes, show_hash = fetch_ollama_json(
        "POST", origin + "/api/show", {"model": expected["name"]}
    )
    details = show.get("details")
    observed_details = {
        "family": str(details.get("family", "")).lower()
        if isinstance(details, dict)
        else "",
        "parameter_size": details.get("parameter_size")
        if isinstance(details, dict)
        else None,
        "quantization_level": details.get("quantization_level")
        if isinstance(details, dict)
        else None,
    }
    if observed_details != expected["details"]:
        raise PhaseEError("Ollama model details differ from the completed verifier run")
    modelfile = show.get("modelfile")
    if not isinstance(modelfile, str):
        raise PhaseEError("Ollama /api/show response lacks a Modelfile")
    observed_blobs = {
        match.lower()
        for line in modelfile.splitlines()
        if line.lstrip().upper().startswith("FROM ")
        for match in MODEL_BLOB_RE.findall(line)
    }
    if observed_blobs != {expected["blob_sha256"]}:
        raise PhaseEError("Ollama Modelfile blob identity differs from the completed verifier run")
    return (
        {
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
        },
        tags_bytes,
        show_bytes,
    )


def validate_child_write_surface(child_dir: Path) -> None:
    """Reject any existing link/reparse escape before Phase E reads or writes a child."""
    if not child_dir.exists():
        return
    if not child_dir.is_dir() or child_dir.is_symlink() or child_dir.resolve() != child_dir:
        raise PhaseEError("phase-e-rag must be one physical same-run directory")
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


def validate_rag_output(
    path: Path,
    contract: dict[str, Any],
    model_name: str,
    *,
    expected_questions: list[dict[str, str]] | None = None,
    expected_kg: str | None = None,
) -> dict[str, Any]:
    output = load_json(path)
    modes = contract["evaluator"]["modes"]
    count = contract["evaluator"]["max_questions"]
    if not isinstance(output, dict) or set(output) != {
        "metadata",
        "accuracy",
        "correct_counts",
        "results",
    }:
        raise PhaseEError(f"RAG output has changed top-level fields: {path}")
    metadata = output.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
        "kg",
        "n_questions",
        "model",
        "time_seconds",
    }:
        raise PhaseEError(f"RAG output metadata fields changed: {path}")
    if metadata.get("n_questions") != count:
        raise PhaseEError(f"RAG output has the wrong question count: {path}")
    if metadata.get("model") != model_name:
        raise PhaseEError(f"RAG output has the wrong model tag: {path}")
    if expected_kg is not None and metadata.get("kg") != expected_kg:
        raise PhaseEError(f"RAG output has the wrong same-run graph path: {path}")
    if not isinstance(metadata.get("time_seconds"), (int, float)):
        raise PhaseEError(f"RAG output has an invalid duration: {path}")
    for field in ("accuracy", "correct_counts", "results"):
        if not isinstance(output.get(field), dict) or list(output[field]) != modes:
            raise PhaseEError(f"RAG output {field} keys/order changed: {path}")
    evaluator = load_evaluator()
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
            ):
                raise PhaseEError(
                    f"RAG output mode {mode} contains a malformed or failed model call"
                )
            observed_question = {"q": row.get("q"), "gold": row.get("gold")}
            if (
                expected_questions is not None
                and observed_question != expected_questions[index]
            ):
                raise PhaseEError(
                    f"RAG output mode {mode} changed question/gold order"
                )
            if row["correct"] != evaluator.check_answer(row["pred"], row["gold"]):
                raise PhaseEError(f"RAG output mode {mode} changed scoring semantics")
        correct = sum(row["correct"] for row in rows)
        if (
            output["correct_counts"][mode] != correct
            or output["accuracy"][mode] != correct / count
        ):
            raise PhaseEError(f"RAG output mode {mode} has inconsistent scores")
    return {
        "sha256": sha256_file(path),
        "accuracy": output["accuracy"],
        "correct_counts": output["correct_counts"],
    }


def validate_child_hashes(child_dir: Path, run_id: str) -> None:
    validate_child_write_surface(child_dir)
    manifest_path = child_dir / "artifact-hashes.json"
    if not manifest_path.is_file():
        raise PhaseEError("Completed Phase E child lacks artifact-hashes.json")
    manifest = load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "run_id", "artifacts"}
        or manifest.get("schema_version") != "phase-e-artifact-hashes-1.0"
        or manifest.get("run_id") != run_id
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise PhaseEError("Completed Phase E artifact hash manifest is malformed")
    expected = manifest["artifacts"]
    observed = {
        path.relative_to(child_dir).as_posix(): sha256_file(path)
        for path in sorted(child_dir.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact-hashes.json", "run-status.json"}
    }
    if expected != observed:
        raise PhaseEError("Completed Phase E child artifact hashes differ")


def stable_child_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise PhaseEError(f"Evaluator path is outside the source checkout: {path}") from exc


def validate_execution_surface(
    source: dict[str, str],
    contract: dict[str, Any],
    expected_contract_sha256: str,
    projections: dict[str, bytes],
    projection_paths: dict[str, Path],
    child_dir: Path,
) -> None:
    observed_source = validate_source_contract(contract, require_clean=True)
    if observed_source != source:
        raise PhaseEError("Source identity changed during Phase E execution")
    validate_file(
        CONTRACT_PATH,
        expected_contract_sha256,
        None,
        "Phase E contract",
    )
    validate_child_write_surface(child_dir)
    for name, expected_bytes in projections.items():
        path = projection_paths[name]
        validate_child_target(child_dir, path)
        validate_file(
            path,
            sha256_bytes(expected_bytes),
            len(expected_bytes),
            f"Phase E {name} projection",
        )


def validate_phase_e_child(
    child_dir: Path,
    status: dict[str, Any],
    contract: dict[str, Any],
    preflight: dict[str, Any],
    projection_paths: dict[str, Path],
) -> dict[str, Any]:
    validate_child_hashes(child_dir, preflight["run_id"])
    projection_manifest_path = child_dir / "projections" / "manifest.json"
    projection_manifest = load_json(projection_manifest_path)
    expected_projection_outputs = {
        path.relative_to(child_dir).as_posix(): preflight["projection_sha256"][name]
        for name, path in projection_paths.items()
    }
    expected_projection_manifest = {
        "schema_version": "phase-e-projection-manifest-1.0",
        "run_id": preflight["run_id"],
        "representation_only": True,
        "inputs": {
            "parent_artifact_set_sha256": preflight[
                "parent_artifact_set_sha256"
            ],
            "score_manifest_sha256": preflight["score_manifest_sha256"],
            "seed": preflight["seed"],
            "threshold": preflight["threshold"],
            "graphs": preflight["graphs"],
        },
        "outputs": expected_projection_outputs,
        "record_summary": preflight["records"],
    }
    if projection_manifest != expected_projection_manifest:
        raise PhaseEError(
            "Phase E projection manifest differs from the authenticated preflight"
        )
    for name, path in projection_paths.items():
        validate_file(
            path,
            preflight["projection_sha256"][name],
            None,
            f"Phase E {name} projection",
        )
    expected_questions = preflight["records"]["question_contract"]
    results: dict[str, Any] = {}
    for condition in ("confidence", "corrective", "gold"):
        output_path = child_dir / "rag-results" / f"{condition}.json"
        results[condition] = validate_rag_output(
            output_path,
            contract,
            preflight["model_identity"]["name"],
            expected_questions=expected_questions,
            expected_kg=stable_child_path(projection_paths[condition]),
        )
        stage = status.get("stages", {}).get(f"rag_{condition}")
        if (
            not isinstance(stage, dict)
            or stage.get("status") != "complete"
            or stage.get("output_sha256") != results[condition]["sha256"]
        ):
            raise PhaseEError(f"Phase E status does not bind completed {condition} output")

    table_path = child_dir / "table2-results.json"
    table = load_json(table_path)
    if (
        not isinstance(table, dict)
        or set(table) != {
            "schema_version",
            "run_id",
            "new_results",
            "legacy_displayed_accuracy",
            "note",
        }
        or table.get("schema_version") != "phase-e-same-run-table2-1.0"
        or table.get("run_id") != preflight["run_id"]
        or table.get("new_results") != results
        or table.get("legacy_displayed_accuracy") != contract["legacy_table2_accuracy"]
    ):
        raise PhaseEError("Phase E Table 2 summary differs from validated same-run results")

    expected_model = preflight["model_identity"]
    checks_root = child_dir / "logs" / "model-identity-checks"
    if not checks_root.is_dir():
        raise PhaseEError("Phase E child lacks its model identity check directory")
    check_directories = sorted(checks_root.iterdir())
    checks: list[Path] = []
    for check_dir in check_directories:
        if not check_dir.is_dir() or check_dir.resolve() != check_dir:
            raise PhaseEError(f"Malformed model identity check directory: {check_dir}")
        entries = {entry.name for entry in check_dir.iterdir()}
        if entries != {"ollama-tags.json", "ollama-show.json", "result.json"}:
            raise PhaseEError(f"Incomplete model identity check directory: {check_dir}")
        if any(not entry.is_file() or entry.resolve() != entry for entry in check_dir.iterdir()):
            raise PhaseEError(f"Model identity check contains a non-file: {check_dir}")
        checks.append(check_dir / "result.json")
    phases = set()
    checks_by_relative: dict[str, dict[str, Any]] = {}
    expected_status_checks: list[dict[str, Any]] = []
    for check_path in checks:
        check = load_json(check_path)
        if not isinstance(check, dict):
            raise PhaseEError(f"Malformed model identity check: {check_path}")
        if check.get("run_id") != preflight["run_id"]:
            raise PhaseEError(f"Model identity check carries another run: {check_path}")
        phase = check.get("phase")
        evidence = check.get("evidence")
        if not isinstance(phase, str) or not isinstance(evidence, dict):
            raise PhaseEError(f"Malformed model identity check: {check_path}")
        for key in (
            "identity_verified",
            "name",
            "tag_digest",
            "blob_sha256",
            "details",
            "verifier_environment_sha256",
            "verifier_condition_id",
            "verifier_protocol_id",
        ):
            if evidence.get(key) != (
                True if key == "identity_verified" else expected_model.get(key)
            ):
                raise PhaseEError(f"Model identity check differs at {check_path}")
        check_dir = check_path.parent
        if evidence.get("tags_response_sha256") != sha256_file(
            check_dir / "ollama-tags.json"
        ) or evidence.get("show_response_sha256") != sha256_file(
            check_dir / "ollama-show.json"
        ):
            raise PhaseEError(
                f"Model identity evidence does not bind its raw responses: {check_path}"
            )
        phases.add(phase)
        relative = check_path.relative_to(child_dir).as_posix()
        checks_by_relative[relative] = check
        expected_status_checks.append(
            {
                "phase": phase,
                "result": relative,
                "result_sha256": sha256_file(check_path),
            }
        )
    if status.get("model_identity_checks") != expected_status_checks:
        raise PhaseEError(
            "Phase E status does not exactly bind its model identity check records"
        )
    if phases != {
        "before-stages",
        "after-confidence",
        "after-corrective",
        "after-gold",
    }:
        raise PhaseEError("Phase E child lacks all required model-drift checks")
    for condition in ("confidence", "corrective", "gold"):
        stage = status["stages"][f"rag_{condition}"]
        binding = stage.get("post_model_identity_check")
        if (
            not isinstance(binding, dict)
            or binding.get("phase") != f"after-{condition}"
            or binding.get("result") not in checks_by_relative
            or binding.get("result_sha256")
            != sha256_file(child_dir / binding["result"])
            or checks_by_relative[binding["result"]].get("phase")
            != f"after-{condition}"
        ):
            raise PhaseEError(
                f"Phase E stage {condition} lacks its post-stage model check binding"
            )
    return results


def record_model_check(
    child_dir: Path,
    status: dict[str, Any],
    status_path: Path,
    phase: str,
    evidence: dict[str, Any],
    tags_bytes: bytes,
    show_bytes: bytes,
) -> dict[str, Any]:
    validate_child_write_surface(child_dir)
    checks_root = child_dir / "logs" / "model-identity-checks"
    existing = list(checks_root.glob("*")) if checks_root.is_dir() else []
    check_dir = checks_root / f"{len(existing) + 1:02d}-{phase}"
    write_bytes_once(check_dir / "ollama-tags.json", tags_bytes, child_dir)
    write_bytes_once(check_dir / "ollama-show.json", show_bytes, child_dir)
    write_json_once(
        check_dir / "result.json",
        {
            "run_id": status["run_id"],
            "phase": phase,
            "checked_at": utc_now(),
            "evidence": evidence,
        },
        child_dir,
    )
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
    command: list[str],
    output_path: Path,
    validator,
    child_dir: Path,
    status: dict[str, Any],
    status_path: Path,
) -> Any:
    validate_child_write_surface(child_dir)
    stage = status.setdefault("stages", {}).setdefault(name, {"attempts": []})
    if stage.get("status") == "complete":
        if (
            not output_path.is_file()
            or sha256_file(output_path) != stage.get("output_sha256")
        ):
            raise PhaseEError(f"Completed stage output no longer matches: {name}")
        return validator(output_path)
    if output_path.exists():
        if not output_path.is_file() or output_path.resolve() != output_path:
            raise PhaseEError(f"Unverified stage output is not a physical file: {output_path}")
        recovery_root = child_dir / "recovery" / name
        existing_recoveries = (
            list(recovery_root.glob("*.json")) if recovery_root.is_dir() else []
        )
        recovered_path = recovery_root / f"unverified-{len(existing_recoveries) + 1:03d}.json"
        validate_child_target(child_dir, output_path)
        validate_child_target(child_dir, recovered_path)
        recovered_path.parent.mkdir(parents=True, exist_ok=True)
        recovered_hash = sha256_file(output_path)
        os.replace(output_path, recovered_path)
        validate_child_write_surface(child_dir)
        stage.setdefault("quarantined_outputs", []).append(
            {
                "path": recovered_path.relative_to(child_dir).as_posix(),
                "sha256": recovered_hash,
                "reason": "missing completed contemporaneous post-stage model check",
            }
        )
        stage["status"] = "retrying-after-unverified-output"
        write_status(status_path, status, child_dir)
    attempt_number = len(stage["attempts"]) + 1
    log_path = child_dir / "logs" / f"{name}-attempt-{attempt_number}.log"
    if log_path.exists():
        raise PhaseEError(f"Attempt log already exists: {log_path}")
    attempt = {
        "number": attempt_number,
        "started_at": utc_now(),
        "command": command,
        "log": log_path.relative_to(child_dir).as_posix(),
    }
    stage["attempts"].append(attempt)
    stage["status"] = "running"
    write_status(status_path, status, child_dir)
    validate_child_target(child_dir, log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    validate_child_target(child_dir, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_child_write_surface(child_dir)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
    validate_child_write_surface(child_dir)
    attempt.update(
        finished_at=utc_now(),
        returncode=completed.returncode,
        log_sha256=sha256_file(log_path),
    )
    if completed.returncode or not output_path.is_file():
        stage["status"] = "failed"
        write_status(status_path, status, child_dir)
        raise PhaseEError(f"Stage {name} failed; see {log_path}")
    try:
        validation = validator(output_path)
    except PhaseEError:
        stage["status"] = "failed-validation"
        write_status(status_path, status, child_dir)
        raise
    stage.update(
        status="output-validated",
        output_sha256=sha256_file(output_path),
        validation=validation,
    )
    write_status(status_path, status, child_dir)
    return validation


def preflight_same_run(
    run_id: str, contract: dict[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, bytes]]:
    same_run = contract["same_run"]
    if run_id != same_run["expected_run_id"]:
        raise PhaseEError(
            f"Contract authenticates run {same_run['expected_run_id']}; got {run_id}"
        )
    validate_contract_same_run_bindings(contract)
    run_dir = validate_run_id(run_id)
    validate_parent_lineage(run_dir, contract)
    graph_artifacts = validate_graph_child(run_dir, contract)

    prepared_paths = {
        name: run_file(run_dir, specification["path"])
        for name, specification in same_run["prepared"].items()
    }
    for name, path in prepared_paths.items():
        specification = same_run["prepared"][name]
        validate_file(path, specification["sha256"], specification["bytes"], name)
    split_manifest = load_json(prepared_paths["split-manifest.json"])
    prepared_rows = load_jsonl(prepared_paths["test.jsonl"])
    gold_rows = load_jsonl(prepared_paths["test-gold.jsonl"])
    _records, record_bytes, record_summary = project_evaluator_records(
        prepared_rows, gold_rows, split_manifest, contract
    )

    projections: dict[str, bytes] = {"records": record_bytes}
    graph_summaries: dict[str, dict[str, Any]] = {}
    graph_root = same_run["graph_child"]["root"]
    for name, specification in same_run["graphs"].items():
        _graph, content, summary = validate_graph_snapshot(
            run_dir, graph_root, specification, graph_artifacts
        )
        projections[name] = content
        graph_summaries[name] = summary

    verifier = same_run["corrective_verifier"]
    environment_path = run_file(run_dir, verifier["environment"]["path"])
    model_identity = load_verifier_model_identity(
        environment_path, verifier["environment"]["sha256"]
    )
    environment_summary = validate_environment(
        sum(len(value) for value in projections.values())
    )
    projection_hashes = {name: sha256_bytes(value) for name, value in projections.items()}
    expected_projection_hashes = same_run["projection_expected_sha256"]
    observed_projection_hashes = {
        **projection_hashes,
        "questions": record_summary["question_sha256"],
    }
    if observed_projection_hashes != expected_projection_hashes:
        raise PhaseEError(
            "Deterministic same-run projection hashes differ from the approved contract"
        )
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "parent_artifact_set_sha256": same_run["parent"]["artifact_set_sha256"],
        "score_manifest_sha256": same_run["parent"]["score_manifest"]["sha256"],
        "threshold": same_run["threshold"]["selected"],
        "seed": same_run["seed"],
        "records": record_summary,
        "graphs": graph_summaries,
        "projection_sha256": projection_hashes,
        "model_identity": model_identity,
        "environment": environment_summary,
    }
    return run_dir, summary, projections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-contract", help="validate frozen source blobs")
    inspect_model = subparsers.add_parser(
        "inspect-model", help="verify Ollama against the selected same-run verifier"
    )
    inspect_model.add_argument("--run-id", required=True)
    inspect_model.add_argument("--ollama-url", default="http://localhost:11434")
    run = subparsers.add_parser("run", help="run only same-run downstream E-T02 stages")
    run.add_argument("--run-id", required=True)
    run.add_argument("--ollama-url", default="http://localhost:11434")
    run.add_argument(
        "--dry-run", action="store_true", help="validate without model calls or writes"
    )
    return parser


def inspect_model_command(args: argparse.Namespace, contract: dict[str, Any]) -> int:
    _run_dir, summary, _projections = preflight_same_run(args.run_id, contract)
    evidence, _tags, _show = fetch_matching_model(
        args.ollama_url, summary["model_identity"]
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def build_child_identity(
    run_id: str, source: dict[str, str], preflight: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "phase-e-same-run-identity-1.0",
        "run_id": run_id,
        "source_head": source["head"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "parent_artifact_set_sha256": preflight["parent_artifact_set_sha256"],
        "score_manifest_sha256": preflight["score_manifest_sha256"],
        "seed": preflight["seed"],
        "threshold": preflight["threshold"],
        "projection_sha256": preflight["projection_sha256"],
        "graph_ids": {
            name: value["source_graph_id"] for name, value in preflight["graphs"].items()
        },
        "ollama_model": preflight["model_identity"]["name"],
        "ollama_tag_digest": preflight["model_identity"]["tag_digest"],
        "ollama_model_blob_sha256": preflight["model_identity"]["blob_sha256"],
        "verifier_environment_sha256": preflight["model_identity"][
            "verifier_environment_sha256"
        ],
    }


def run_command(args: argparse.Namespace, contract: dict[str, Any]) -> int:
    source = validate_source_contract(contract, require_clean=not args.dry_run)
    run_dir, preflight, projections = preflight_same_run(args.run_id, contract)
    child_dir = run_dir / "phase-e-rag"
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
    new_child = not child_dir.exists()
    if child_dir.exists():
        if not status_path.is_file():
            if any(child_dir.iterdir()):
                raise PhaseEError("phase-e-rag exists without a resumable status identity")
            new_child = True
        else:
            new_child = False
    if not new_child:
        status = load_json(status_path)
        if (
            not isinstance(status, dict)
            or status.get("schema_version") != "phase-e-same-run-status-1.0"
            or status.get("run_id") != args.run_id
            or status.get("identity") != identity
        ):
            raise PhaseEError("Existing Phase E child identity differs from this same run")
        if status.get("status") == "complete":
            validate_phase_e_child(
                child_dir, status, contract, preflight, projection_paths
            )
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "child": "phase-e-rag",
                        "status": "complete",
                        "resumed": True,
                    },
                    indent=2,
                )
            )
            return 0
        if (child_dir / "artifact-hashes.json").exists():
            validate_phase_e_child(
                child_dir, status, contract, preflight, projection_paths
            )
            status["status"] = "complete"
            status["finished_at"] = utc_now()
            status["recovered_finalization"] = True
            write_status(status_path, status, child_dir)
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "child": "phase-e-rag",
                        "status": "complete",
                        "recovered_finalization": True,
                    },
                    indent=2,
                )
            )
            return 0
    else:
        status = {
            "schema_version": "phase-e-same-run-status-1.0",
            "run_id": args.run_id,
            "identity": identity,
            "status": "running",
            "created_at": utc_now(),
            "stages": {},
        }

    validate_child_write_surface(child_dir)
    model_evidence, tags_bytes, show_bytes = fetch_matching_model(
        args.ollama_url, preflight["model_identity"]
    )
    if new_child:
        child_dir.mkdir(exist_ok=True)
        validate_child_write_surface(child_dir)
        write_status(status_path, status, child_dir)
    record_model_check(
        child_dir,
        status,
        status_path,
        "before-stages",
        model_evidence,
        tags_bytes,
        show_bytes,
    )

    for name, path in projection_paths.items():
        write_bytes_once(path, projections[name], child_dir)
    write_json_once(
        projection_dir / "manifest.json",
        {
            "schema_version": "phase-e-projection-manifest-1.0",
            "run_id": args.run_id,
            "representation_only": True,
            "inputs": {
                "parent_artifact_set_sha256": preflight["parent_artifact_set_sha256"],
                "score_manifest_sha256": preflight["score_manifest_sha256"],
                "seed": preflight["seed"],
                "threshold": preflight["threshold"],
                "graphs": preflight["graphs"],
            },
            "outputs": {
                path.relative_to(child_dir).as_posix(): sha256_file(path)
                for path in projection_paths.values()
            },
            "record_summary": preflight["records"],
        },
        child_dir,
    )
    write_json_once(
        child_dir / "method-manifest.json",
        {
            "run_id": args.run_id,
            "contract": contract,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "resolved_same_run": {
                key: value
                for key, value in preflight.items()
                if key not in {"environment", "run_dir"}
            },
        },
        child_dir,
    )
    write_json_once(
        child_dir / "environment.json",
        {
            "run_id": args.run_id,
            "captured_at": status["created_at"],
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "source": source,
            "model_identity": preflight["model_identity"],
        },
        child_dir,
    )

    evaluator_base = [
        sys.executable,
        "-I",
        "-B",
        str(ROOT / "eval_graph_rag.py"),
        "--gold-jsonl",
        stable_child_path(projection_paths["records"]),
        "--ollama-url",
        args.ollama_url.rstrip("/"),
        "--ollama-model",
        preflight["model_identity"]["name"],
        "--max-questions",
        str(contract["evaluator"]["max_questions"]),
    ]
    results: dict[str, Any] = {}
    result_dir = child_dir / "rag-results"
    for condition in ("confidence", "corrective", "gold"):
        validate_execution_surface(
            source,
            contract,
            identity["contract_sha256"],
            projections,
            projection_paths,
            child_dir,
        )
        output = result_dir / f"{condition}.json"
        results[condition] = run_stage(
            f"rag_{condition}",
            evaluator_base
            + [
                "--kg",
                stable_child_path(projection_paths[condition]),
                "--output",
                stable_child_path(output),
            ],
            output,
            lambda path, condition=condition: validate_rag_output(
                path,
                contract,
                preflight["model_identity"]["name"],
                expected_questions=preflight["records"]["question_contract"],
                expected_kg=stable_child_path(projection_paths[condition]),
            ),
            child_dir,
            status,
            status_path,
        )
        try:
            model_evidence, tags_bytes, show_bytes = fetch_matching_model(
                args.ollama_url, preflight["model_identity"]
            )
            validate_execution_surface(
                source,
                contract,
                identity["contract_sha256"],
                projections,
                projection_paths,
                child_dir,
            )
        except PhaseEError:
            status["stages"][f"rag_{condition}"]["status"] = (
                "failed-post-stage-identity-or-input-check"
            )
            write_status(status_path, status, child_dir)
            raise
        post_check = record_model_check(
            child_dir,
            status,
            status_path,
            f"after-{condition}",
            model_evidence,
            tags_bytes,
            show_bytes,
        )
        status["stages"][f"rag_{condition}"].update(
            status="complete", post_model_identity_check=post_check
        )
        write_status(status_path, status, child_dir)

    validate_execution_surface(
        source,
        contract,
        identity["contract_sha256"],
        projections,
        projection_paths,
        child_dir,
    )
    write_json_once(
        child_dir / "table2-results.json",
        {
            "schema_version": "phase-e-same-run-table2-1.0",
            "run_id": args.run_id,
            "new_results": results,
            "legacy_displayed_accuracy": contract["legacy_table2_accuracy"],
            "note": (
                "Legacy values are comparison references only; differences must not "
                "trigger method or prompt tuning."
            ),
        },
        child_dir,
    )
    hashes = {
        path.relative_to(child_dir).as_posix(): sha256_file(path)
        for path in sorted(child_dir.rglob("*"))
        if path.is_file() and path.name not in {"artifact-hashes.json", "run-status.json"}
    }
    write_json_once(
        child_dir / "artifact-hashes.json",
        {
            "schema_version": "phase-e-artifact-hashes-1.0",
            "run_id": args.run_id,
            "artifacts": hashes,
        },
        child_dir,
    )
    validate_execution_surface(
        source,
        contract,
        identity["contract_sha256"],
        projections,
        projection_paths,
        child_dir,
    )
    # Re-hash the complete selected parent and graph lineages immediately before
    # finalization. Projections alone cannot prove that their upstream tree stayed
    # unchanged during a long model run.
    validate_parent_lineage(run_dir, contract)
    validate_graph_child(run_dir, contract)
    validate_phase_e_child(child_dir, status, contract, preflight, projection_paths)
    status["status"] = "complete"
    status["finished_at"] = utc_now()
    write_status(status_path, status, child_dir)
    print(
        json.dumps(
            {"run_dir": str(run_dir), "child": "phase-e-rag", "status": "complete"},
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        contract = load_json(CONTRACT_PATH)
        if args.command == "validate-contract":
            print(json.dumps(validate_source_contract(contract, False), indent=2))
            return 0
        if args.command == "inspect-model":
            return inspect_model_command(args, contract)
        return run_command(args, contract)
    except PhaseEError as exc:
        print(f"Phase E blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
