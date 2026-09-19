"""Shared artifact I/O, same-run seals, and recovery primitives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from utils.common.paths import RunLayout


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


def _load_manifest(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_outputs(layout: RunLayout, manifest: dict) -> dict[str, str]:
    declared = manifest.get("outputs")
    if isinstance(declared, dict):
        outputs = declared
    else:
        outputs = {}
        for field in ("generation_plan_output", "candidates_output"):
            relative = manifest.get(field)
            if isinstance(relative, str):
                path = layout.resolve(relative, must_exist=True)
                outputs[relative] = sha256_file(path)
        if (
            manifest.get("stage") == "model-generate-candidates"
            and manifest.get("execution_mode") == "live"
        ):
            ledger = manifest.get("inputs", {}).get("prediction_ledger")
            if isinstance(ledger, dict) and isinstance(ledger.get("path"), str):
                outputs[ledger["path"]] = ledger.get("sha256")
    if not outputs:
        raise DataContractError("stage producer manifest does not declare an output")
    for relative, expected in outputs.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise DataContractError("stage producer manifest has an invalid output binding")
        path = layout.resolve(relative, must_exist=True)
        if not path.is_file() or sha256_file(path) != expected:
            raise DataContractError(f"stage output differs from its producer: {relative}")
    return outputs


def _same_run_seal_path(producer_manifest: Path) -> Path:
    return producer_manifest.with_name(f"same-run-{producer_manifest.name}")


def _write_same_run_seal(
    layout: RunLayout, producer_manifest: Path, manifest: dict
) -> Path:
    outputs = _manifest_outputs(layout, manifest)
    seal_path = _same_run_seal_path(producer_manifest)
    document = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": layout.run_id,
        "producer_manifest": {
            "path": layout.relative_identity(producer_manifest),
            "sha256": sha256_file(producer_manifest),
        },
        "outputs": outputs,
    }
    atomic_write_json(seal_path, document)
    return seal_path


def _validate_same_run_seal(
    layout: RunLayout, producer_manifest: Path, manifest: dict
) -> None:
    outputs = _manifest_outputs(layout, manifest)
    seal = _load_manifest(
        _same_run_seal_path(producer_manifest), "same-run stage seal"
    )
    expected = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": layout.run_id,
        "producer_manifest": {
            "path": layout.relative_identity(producer_manifest),
            "sha256": sha256_file(producer_manifest),
        },
        "outputs": outputs,
    }
    if seal != expected:
        raise DataContractError("stage seal does not authenticate the selected run")


def _require_fields(value: dict, expected: dict[str, object], label: str) -> None:
    for field, item in expected.items():
        if value.get(field) != item:
            raise DataContractError(f"{label}.{field} differs from this pipeline run")


def _next_recovery_path(layout: RunLayout, category: str, filename: str) -> Path:
    directory = layout.resolve(f"inputs/recovery/{category}")
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob(f"*-{filename}"))
    return directory / f"{len(existing) + 1:03d}-{filename}"


def _quarantine_run_artifact(
    layout: RunLayout, path: Path, category: str
) -> Path | None:
    if not path.exists():
        return None
    layout.relative_identity(path)
    if not path.is_file() or path.is_symlink():
        raise DataContractError(f"recovery target is not a physical file: {path}")
    destination = _next_recovery_path(layout, f"quarantine/{category}", path.name)
    os.replace(path, destination)
    return destination


def _write_recovery_provenance(
    layout: RunLayout,
    cache: Path,
    *,
    kind: str,
    identity: dict[str, object],
    bindings: dict[str, Path],
) -> Path:
    provenance = cache.with_suffix(cache.suffix + ".manifest.json")
    checkout = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    document = {
        "schema_version": "phase-b-same-run-recovery-2.0",
        "run_id": layout.run_id,
        "kind": kind,
        "identity": identity,
        "cache": {
            "path": layout.relative_identity(cache),
            "sha256": sha256_file(cache),
        },
        "checkout_manifest": {
            "path": layout.relative_identity(checkout),
            "sha256": sha256_file(checkout),
        },
        "bindings": {
            name: {
                "path": layout.relative_identity(path),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(bindings.items())
        },
    }
    atomic_write_json(provenance, document)
    return provenance


def _validate_recovery_provenance(
    layout: RunLayout,
    cache: Path,
    *,
    kind: str,
    identity: dict[str, object],
    bindings: dict[str, Path],
) -> None:
    provenance_path = cache.with_suffix(cache.suffix + ".manifest.json")
    provenance = _load_manifest(provenance_path, "recovery provenance")
    checkout = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    expected = {
        "schema_version": "phase-b-same-run-recovery-2.0",
        "run_id": layout.run_id,
        "kind": kind,
        "identity": identity,
        "cache": {
            "path": layout.relative_identity(cache),
            "sha256": sha256_file(cache),
        },
        "checkout_manifest": {
            "path": layout.relative_identity(checkout),
            "sha256": sha256_file(checkout),
        },
        "bindings": {
            name: {
                "path": layout.relative_identity(path),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(bindings.items())
        },
    }
    if provenance != expected:
        raise DataContractError("recovery cache is not bound to the selected run")
