"""Run-local, content-addressed Hugging Face cache contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from utils.common.artifact_io import (
    DataContractError,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
)
from utils.common.paths import RunLayout


CACHE_RELATIVE = "inputs/huggingface"
MANIFEST_RELATIVE = "manifests/huggingface-model-cache.json"


def configure_model_loading() -> None:
    """Keep Hub conversion workers from mutating the frozen run-local cache.

    Transformers can asynchronously download a converted safetensors copy even
    when the pinned model was loaded from its original PyTorch weights. Disable
    that optional conversion without changing which pinned weights are loaded.
    """
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"


def _cache_files(cache_dir: Path) -> list[dict[str, Any]]:
    if not cache_dir.is_dir():
        raise DataContractError("run-local Hugging Face cache directory is missing")
    records: list[dict[str, Any]] = []
    for path in sorted(cache_dir.rglob("*"), key=lambda item: item.relative_to(cache_dir).as_posix()):
        relative = path.relative_to(cache_dir).as_posix()
        if relative == ".locks" or relative.startswith(".locks/"):
            continue
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(cache_dir.resolve()):
                raise DataContractError('model cache link points outside its physical run-local cache')
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(path),
                }
            )
        elif path.resolve() != path.absolute():
            raise DataContractError('model cache entry is reached through a junction')
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not records:
        raise DataContractError("run-local Hugging Face cache contains no model/tokenizer files")
    return records


def _tree_sha256(files: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(files)).hexdigest()


def _inventory_difference(expected: Any, observed: list[dict[str, Any]]) -> str:
    """Describe bounded path differences without weakening the frozen identity."""
    if not isinstance(expected, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("path"), str)
        for item in expected
    ):
        return "; frozen file inventory is malformed"
    before = {item["path"]: item for item in expected}
    after = {item["path"]: item for item in observed}
    differences = (
        ("added", sorted(after.keys() - before.keys())),
        ("removed", sorted(before.keys() - after.keys())),
        ("changed", sorted(path for path in before.keys() & after.keys()
                           if before[path] != after[path])),
    )
    descriptions = []
    for label, paths in differences:
        if paths:
            shown = ", ".join(repr(path[:200]) for path in paths[:5])
            if len(paths) > 5:
                shown += f", ... ({len(paths)} total)"
            descriptions.append(f"{label}: {shown}")
    return "; " + "; ".join(descriptions or ["manifest metadata differs"])


def write_cache_manifest(
    layout: RunLayout, *, model: str, revision: str
) -> dict[str, Any]:
    cache_dir = layout.resolve(CACHE_RELATIVE, must_exist=True)
    files = _cache_files(cache_dir)
    manifest = {
        "schema_version": "phase-b-huggingface-cache-1.0",
        "model": model,
        "revision": revision,
        "cache_root": CACHE_RELATIVE,
        "file_count": len(files),
        "tree_sha256": _tree_sha256(files),
        "files": files,
        "offline_replay_ready": True,
    }
    path = layout.resolve(MANIFEST_RELATIVE)
    if path.exists():
        frozen = load_json(path)
        if frozen != manifest:
            expected = frozen.get("files") if isinstance(frozen, dict) else None
            raise DataContractError(
                "run-local Hugging Face cache differs from its frozen manifest"
                + _inventory_difference(expected, files)
            )
    else:
        atomic_write_json(path, manifest)
    return manifest


def verify_cache_manifest(
    layout: RunLayout, *, model: str, revision: str
) -> dict[str, Any]:
    path = layout.resolve(MANIFEST_RELATIVE, must_exist=True)
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise DataContractError("Hugging Face cache manifest must be an object")
    if manifest.get("schema_version") != "phase-b-huggingface-cache-1.0":
        raise DataContractError("Hugging Face cache manifest schema is unsupported")
    if manifest.get("model") != model or manifest.get("revision") != revision:
        raise DataContractError("Hugging Face cache model/revision differs from configuration")
    if manifest.get("cache_root") != CACHE_RELATIVE:
        raise DataContractError("Hugging Face cache manifest has an unsupported cache root")
    files = _cache_files(layout.resolve(CACHE_RELATIVE, must_exist=True))
    if manifest.get("files") != files or manifest.get("file_count") != len(files):
        raise DataContractError(
            "run-local Hugging Face cache file inventory drifted"
            + _inventory_difference(manifest.get("files"), files)
        )
    if manifest.get("tree_sha256") != _tree_sha256(files):
        raise DataContractError("run-local Hugging Face cache tree hash drifted")
    if manifest.get("offline_replay_ready") is not True:
        raise DataContractError("run-local Hugging Face cache is not marked offline-ready")
    return manifest
