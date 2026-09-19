"""Standalone source, environment, ignore, and identity preflight."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils.common.config import PipelineConfig
from utils.common.constants import MATCHER_ID, PROTOCOL_ID, WORKFLOW_ID
from utils.common.artifact_io import atomic_write_json, sha256_file
from utils.common.paths import RunLayout


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


def _document_version(
    check_id: str, *, documented: str, actual: str | None, detail: str = ""
) -> dict[str, Any]:
    """Record a runtime version without treating it as a checkout admission gate."""

    suffix = f"; {detail}" if detail else ""
    return _check(
        check_id,
        True,
        f"documented={documented}; actual={actual or 'unavailable'}{suffix}; not enforced",
    )


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
        ["git", "check-ignore", "-q", "--", "output/__pipeline_probe__/nested.json"],
    )
    negative = _command(
        source_root,
        ["git", "check-ignore", "-q", "--", "nested/output/__pipeline_probe__.json"],
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

    documented_python = config.value["environment"]["python"]
    actual_python = platform.python_version()
    checks.append(
        _document_version(
            "python-version", documented=documented_python, actual=actual_python
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
    documented_uv = config.value["environment"]["uv"]
    actual_uv, uv_detail = _uv_version(source_root)
    checks.append(
        _document_version(
            "uv-version", documented=documented_uv, actual=actual_uv, detail=uv_detail
        )
    )

    layout.create()
    passed = all(item["status"] == "pass" for item in checks)
    config_relative = config.path.relative_to(source_root).as_posix()
    manifest = {
        "schema_version": "phase-b-checkout-manifest-2.0",
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "matcher_id": MATCHER_ID,
        "run_id": layout.run_id,
        "status": "pass" if passed else "blocked",
        "source": {
            "commit": head,
            "worktree_clean": clean,
            "config": {
                "path": config_relative,
                "sha256": sha256_file(config.path),
            },
        },
        "environment": {
            "python": actual_python,
            "uv": actual_uv,
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
        },
        "checks": checks,
    }
    manifest_path = layout.resolve("manifests/00-checkout-manifest.json")
    atomic_write_json(manifest_path, manifest)
    return manifest, passed
