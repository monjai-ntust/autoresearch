"""Deterministic, atomic file helpers for Phase B artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


class DataContractError(ValueError):
    """Raised when machine-readable input violates a frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """Return the upstream-required MD5 identity for a public archive."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"invalid UTF-8 JSON file {path.name}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise DataContractError(
                        f"{path.name}:{line_number}: JSONL record lacks a newline terminator"
                    )
                if not line.strip():
                    raise DataContractError(
                        f"{path.name}:{line_number}: blank JSONL records are forbidden"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataContractError(
                        f"{path.name}:{line_number}: malformed JSON: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise DataContractError(
                        f"{path.name}:{line_number}: each JSONL record must be an object"
                    )
                yield line_number, value
    except (OSError, UnicodeError) as exc:
        raise DataContractError(f"could not read UTF-8 JSONL file {path.name}: {exc}") from exc


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_replace(path, canonical_json_bytes(value))


def atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(value) for value in values)
    _atomic_replace(path, payload)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    _atomic_replace(path, payload)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))
