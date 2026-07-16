"""Standalone source, environment, ignore, and identity preflight."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .constants import MATCHER_ID, PROTOCOL_ID, WORKFLOW_ID
from .io import atomic_write_json
from .paths import RunLayout, resolve_tracked_path


def _command(source_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=source_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def _uv_version(source_root: Path) -> tuple[str | None, str]:
    try:
        result = _command(source_root, ["uv", "--version"])
    except FileNotFoundError:
        return None, "uv executable was not found"
    text = result.stdout.strip()
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
    if result.returncode != 0 or match is None:
        return None, text or result.stderr.strip() or "could not parse uv version"
    return match.group(1), text


def _git_blob_sha256(source_root: Path, path: Path) -> str:
    relative = path.relative_to(source_root).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=source_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(
            f"could not read committed artifact bytes for {relative}: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return hashlib.sha256(result.stdout).hexdigest()


def run_doctor(layout: RunLayout, config: PipelineConfig) -> tuple[dict[str, Any], bool]:
    """Create one immutable checkout manifest; return it and pass/fail status."""

    source_root = layout.source_root
    checks: list[dict[str, Any]] = []

    git_head = _command(source_root, ["git", "rev-parse", "HEAD"])
    head = git_head.stdout.strip() if git_head.returncode == 0 else None
    checks.append(_check("git-head", bool(head), "resolved" if head else "unavailable"))

    git_status = _command(
        source_root, ["git", "status", "--porcelain", "--untracked-files=normal"]
    )
    clean = git_status.returncode == 0 and not git_status.stdout.strip()
    checks.append(
        _check(
            "clean-tracked-source",
            clean,
            "clean" if clean else "tracked or untracked source changes are present",
        )
    )

    ignore_lines = (source_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    checks.append(
        _check(
            "root-output-ignore-rule",
            "/output/" in ignore_lines,
            "exact /output/ rule present"
            if "/output/" in ignore_lines
            else "exact /output/ rule is missing",
        )
    )

    positive = _command(
        source_root,
        ["git", "check-ignore", "-q", "--", "output/__phase_b_probe__/nested.json"],
    )
    negative = _command(
        source_root,
        ["git", "check-ignore", "-q", "--", "nested/output/__phase_b_probe__.json"],
    )
    checks.append(
        _check(
            "root-output-ignore-positive",
            positive.returncode == 0,
            "root output is ignored" if positive.returncode == 0 else "root output is not ignored",
        )
    )
    checks.append(
        _check(
            "root-output-ignore-negative",
            negative.returncode == 1,
            "nested source path is not overmatched"
            if negative.returncode == 1
            else "nested source path is unexpectedly ignored",
        )
    )

    required_python = config.value["environment"]["python"]
    actual_python = platform.python_version()
    checks.append(
        _check(
            "python-version",
            actual_python == required_python,
            f"required={required_python}; actual={actual_python}",
        )
    )
    checks.append(
        _check(
            "source-bytecode-disabled",
            sys.dont_write_bytecode,
            "Python bytecode writes are disabled"
            if sys.dont_write_bytecode
            else "rerun with python -B so __pycache__ is not created outside output/",
        )
    )
    required_uv = config.value["environment"]["uv"]
    actual_uv, uv_detail = _uv_version(source_root)
    checks.append(
        _check(
            "uv-version",
            actual_uv == required_uv,
            f"required={required_uv}; actual={actual_uv or 'unavailable'}; {uv_detail}",
        )
    )

    artifact_paths = [config.path, config.matrix_path]
    for relative in config.value["tracked_artifacts"]:
        artifact_paths.append(resolve_tracked_path(source_root, relative))
    unique_paths = sorted(set(artifact_paths), key=lambda item: item.relative_to(source_root).as_posix())
    artifact_hashes = {
        path.relative_to(source_root).as_posix(): _git_blob_sha256(source_root, path)
        for path in unique_paths
    }
    checks.append(
        _check("tracked-artifact-hashes", bool(artifact_hashes), f"hashed {len(artifact_hashes)} files")
    )

    layout.create()
    passed = all(item["status"] == "pass" for item in checks)
    manifest = {
        "schema_version": "phase-b-checkout-manifest-1.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "matcher_id": MATCHER_ID,
        "run_id": layout.run_id,
        "status": "pass" if passed else "blocked",
        "publication_execution_admitted": False,
        "publication_execution_gate": "B-07-go-no-go-not-yet-approved",
        "source": {
            "commit": head,
            "worktree_clean": clean,
            "lockfile_sha256": artifact_hashes["uv.lock"],
        },
        "environment": {
            "python": actual_python,
            "uv": actual_uv,
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
        },
        "tracked_artifact_sha256": artifact_hashes,
        "checks": checks,
    }
    manifest_path = layout.resolve("manifests/00-checkout-manifest.json")
    atomic_write_json(manifest_path, manifest)
    return manifest, passed
