"""Immutable, resumable CODE-ACCORD acquisition inside one Phase B run."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .io import DataContractError, atomic_write_json, load_json, md5_file, sha256_file
from .paths import RunLayout


@dataclass(frozen=True)
class ArchiveContract:
    dataset_id: str
    name: str
    url: str
    expected_bytes: int
    expected_md5: str
    license: str


def archive_contract_from_config(config: PipelineConfig) -> ArchiveContract:
    value = config.value["dataset"]
    return ArchiveContract(
        dataset_id=value["dataset_id"],
        name=value["archive_name"],
        url=value["archive_url"],
        expected_bytes=value["archive_bytes"],
        expected_md5=value["archive_md5"],
        license=value["license"],
    )


def _verify_archive(path: Path, contract: ArchiveContract) -> dict[str, Any]:
    actual_bytes = path.stat().st_size
    if actual_bytes != contract.expected_bytes:
        raise DataContractError(
            f"archive byte-size mismatch: expected {contract.expected_bytes}, "
            f"got {actual_bytes}"
        )
    actual_md5 = md5_file(path)
    if actual_md5 != contract.expected_md5:
        raise DataContractError(
            f"archive MD5 mismatch: expected {contract.expected_md5}, got {actual_md5}"
        )
    return {
        "bytes": actual_bytes,
        "md5": actual_md5,
        "sha256": sha256_file(path),
    }


def download_verified_archive(
    contract: ArchiveContract,
    destination: Path,
    *,
    attempts: int = 3,
    timeout_seconds: int = 120,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any], str]:
    """Download one immutable archive, safely resuming a retained partial file."""

    if attempts < 1:
        raise ValueError("download attempts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return _verify_archive(destination, contract), "preexisting_verified_cache"

    partial = destination.with_name(destination.name + ".partial")
    last_error: BaseException | None = None
    for _ in range(attempts):
        current_size = partial.stat().st_size if partial.exists() else 0
        if current_size > contract.expected_bytes:
            raise DataContractError(
                f"retained partial archive exceeds the expected size: {current_size}"
            )
        if current_size == contract.expected_bytes:
            identity = _verify_archive(partial, contract)
            os.replace(partial, destination)
            return identity, "verified_retained_partial"

        request = urllib.request.Request(contract.url)
        if current_size:
            request.add_header("Range", f"bytes={current_size}-")
        try:
            response = opener(request, timeout=timeout_seconds)
            try:
                status = getattr(response, "status", None)
                resumed = current_size > 0 and status == 206
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        if handle.tell() > contract.expected_bytes:
                            raise DataContractError(
                                "download exceeded the immutable archive byte contract"
                            )
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                response.close()
        except (OSError, urllib.error.URLError, DataContractError) as exc:
            last_error = exc
            if isinstance(exc, DataContractError):
                raise
            continue

        if partial.stat().st_size == contract.expected_bytes:
            identity = _verify_archive(partial, contract)
            os.replace(partial, destination)
            return identity, "downloaded_and_verified"

    retained = partial.stat().st_size if partial.exists() else 0
    detail = f": {last_error}" if last_error else ""
    raise DataContractError(
        f"archive download incomplete after {attempts} attempts; retained "
        f"{retained}/{contract.expected_bytes} bytes in the partial file{detail}"
    )


def fetch_run(layout: RunLayout, config: PipelineConfig) -> dict[str, Any]:
    """Fetch CODE-ACCORD into the selected run and write its acquisition manifest."""

    layout.require_existing()
    checkout = load_json(
        layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    )
    if not isinstance(checkout, dict) or checkout.get("status") != "pass":
        raise DataContractError("fetch requires a passing doctor checkout manifest")
    contract = archive_contract_from_config(config)
    manifest_path = layout.resolve("manifests/02-input-acquisition-manifest.json")
    cache_relative = (
        f"inputs/cache/md5-{contract.expected_md5}/{contract.name}"
    )
    archive_path = layout.resolve(cache_relative)

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        identity = _verify_archive(archive_path, contract)
        archive = manifest.get("archive") if isinstance(manifest, dict) else None
        if not isinstance(archive, dict) or any(
            archive.get(field) != expected
            for field, expected in {
                "path": cache_relative,
                "bytes": identity["bytes"],
                "md5": identity["md5"],
                "sha256": identity["sha256"],
            }.items()
        ):
            raise DataContractError("existing acquisition manifest does not match the cache")
        return manifest

    identity, acquisition_status = download_verified_archive(contract, archive_path)
    manifest = {
        "schema_version": "phase-b-input-acquisition-manifest-1.0",
        "dataset_id": contract.dataset_id,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "url": contract.url,
            "path": cache_relative,
            "bytes": identity["bytes"],
            "md5": identity["md5"],
            "sha256": identity["sha256"],
            "license": contract.license,
            "status": acquisition_status,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
