"""Standalone source, environment, ignore, and identity preflight."""

from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from utils.common.config import PipelineConfig
from utils.common.constants import MATCHER_ID, PROTOCOL_ID, WORKFLOW_ID
from utils.common.artifact_io import atomic_write_json
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


def _git_status_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Show the failing checkout evidence without unbounded manifest output."""

    if result.returncode != 0:
        prefix = f"git status failed (exit {result.returncode}): "
        output = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
    elif not result.stdout.strip():
        return "clean"
    else:
        prefix = "tracked or untracked source changes are present; git status --porcelain:\n"
        output = result.stdout.rstrip("\r\n")
    if len(output) > 4000:
        output = output[:4000] + "\n... (diagnostic truncated; run git status --short for full output)"
    return prefix + output


def inspect_checkout(layout: RunLayout, config: PipelineConfig) -> tuple[dict[str, Any], bool]:
    """Inspect checkout identity and preflight checks without writing run files."""

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
            _git_status_detail(git_status),
        )
    )

    positive = _command(
        source_root,
        ["git", "check-ignore", "-q", "--", "output/__pipeline_probe__/nested.json"],
    )
    checks.append(
        _check(
            "root-output-ignore-positive",
            positive.returncode == 0,
            "root output is ignored" if positive.returncode == 0 else "root output is not ignored",
        )
    )
    config_relative = config.path.relative_to(source_root).as_posix()
    tracked = _command(source_root,
        ['git', 'ls-files', '--error-unmatch', '--', config_relative])
    checks.append(_check('selected-config-tracked', tracked.returncode == 0,
        'selected configuration is tracked' if tracked.returncode == 0
        else 'selected configuration is not tracked by the source commit'))
    actual_python = platform.python_version()
    actual_uv, _ = _uv_version(source_root)

    passed = all(item["status"] == "pass" for item in checks)
    manifest = {
        "schema_version": "phase-b-checkout-manifest-2.0",
        "lifecycle_version": 1,
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
    return manifest, passed


def run_doctor(layout: RunLayout, config: PipelineConfig) -> tuple[dict[str, Any], bool]:
    """Create one immutable checkout manifest; return it and pass/fail status."""

    manifest, passed = inspect_checkout(layout, config)
    layout.create(allow_lock_only=True)
    manifest_path = layout.resolve("manifests/00-checkout-manifest.json")
    atomic_write_json(manifest_path, manifest)
    return manifest, passed
