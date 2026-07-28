"""Run-local, content-addressed Hugging Face cache contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from phase_b_io import DataContractError, atomic_write_json, canonical_json_bytes, load_json, sha256_file
from paths import RunLayout


CACHE_RELATIVE = "inputs/huggingface"
MANIFEST_RELATIVE = "manifests/huggingface-model-cache.json"
SCHEMA_VERSION = "phase-b-huggingface-cache-1.1"


def _cache_files(cache_dir: Path) -> list[dict[str, Any]]:
    if not cache_dir.is_dir():
        raise DataContractError("run-local Hugging Face cache directory is missing")
    cache_root = cache_dir.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for path in sorted(cache_dir.rglob("*"), key=lambda item: item.relative_to(cache_dir).as_posix()):
        relative = path.relative_to(cache_dir).as_posix()
        if relative == ".locks" or relative.startswith(".locks/"):
            continue
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise DataContractError(
                    f"run-local Hugging Face cache symlink target is absolute: {relative}"
                )
            try:
                resolved_target = path.resolve(strict=True)
                resolved_target.relative_to(cache_root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise DataContractError(
                    "run-local Hugging Face cache symlink is broken or escapes "
                    f"the cache root: {relative}"
                ) from exc
            if not resolved_target.is_file():
                raise DataContractError(
                    f"run-local Hugging Face cache symlink must target a file: {relative}"
                )
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": target,
                    "bytes": resolved_target.stat().st_size,
                    "sha256": sha256_file(resolved_target),
                }
            )
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


def write_cache_manifest(
    layout: RunLayout, *, model: str, revision: str
) -> dict[str, Any]:
    cache_dir = layout.resolve(CACHE_RELATIVE, must_exist=True)
    files = _cache_files(cache_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
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
        if load_json(path) != manifest:
            raise DataContractError("run-local Hugging Face cache differs from its frozen manifest")
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
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError("Hugging Face cache manifest schema is unsupported")
    if manifest.get("model") != model or manifest.get("revision") != revision:
        raise DataContractError("Hugging Face cache model/revision differs from configuration")
    if manifest.get("cache_root") != CACHE_RELATIVE:
        raise DataContractError("Hugging Face cache manifest has an unsupported cache root")
    files = _cache_files(layout.resolve(CACHE_RELATIVE, must_exist=True))
    if manifest.get("files") != files or manifest.get("file_count") != len(files):
        raise DataContractError("run-local Hugging Face cache file inventory drifted")
    if manifest.get("tree_sha256") != _tree_sha256(files):
        raise DataContractError("run-local Hugging Face cache tree hash drifted")
    if manifest.get("offline_replay_ready") is not True:
        raise DataContractError("run-local Hugging Face cache is not marked offline-ready")
    return manifest
