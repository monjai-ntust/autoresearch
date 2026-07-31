"""Content identities and immutable cache verification."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json

from .contracts import canonical_data, canonical_json


def fingerprint(namespace: str, *parts: Any) -> str:
    payload = {"namespace": namespace, "parts": [canonical_data(part) for part in parts]}
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> tuple[dict[str, Any], ...]:
    if not root.exists():
        return ()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return tuple(rows)


def tree_sha256(root: Path) -> str:
    return fingerprint("tree-v1", tree_manifest(root))


def source_surface_manifest(source_root: Path) -> tuple[dict[str, Any], ...]:
    """Hash the standalone code/schema surface that governs a Graph RAG run.

    The manifest intentionally excludes configurations (bound separately) and
    every generated/cache path. It does not depend on Git metadata, so it works
    in the clean standalone-copy validation lane.
    """

    root = source_root.resolve()
    candidates = [
        root / "graph_rag.py",
        # C-04 parses the frozen Phase B candidate/gold/verdict contracts
        # through their canonical immutable record host.
        root / "records.py",
        root / "constants.py",
        root / "phase_b_io.py",
    ]
    candidates.extend(
        path
        for path in (root / "graph_rag_eval").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    candidates.extend((root / "schemas" / "phase_b").glob("rag-*.schema.json"))
    rows = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return tuple(rows)


def source_surface_sha256(source_root: Path) -> str:
    return fingerprint("graph-rag-source-surface-v1", source_surface_manifest(source_root))


def verify_cache(cache_dir: Path, expected_fingerprint: str) -> bool:
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("fingerprint") != expected_fingerprint:
        return False
    expected_tree = data.get("content_tree_sha256")
    if not isinstance(expected_tree, str):
        return False
    content = cache_dir / "content"
    return tree_sha256(content) == expected_tree


def identity_namespace(
    dataset_id: str,
    snapshot_id: str,
    adapter_id: str,
    adapter_version: str,
    canonical_hash: str,
    config: Any,
) -> str:
    return fingerprint(
        "dataset-run-v1",
        {
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "canonical_hash": canonical_hash,
            "config": config,
        },
    )
