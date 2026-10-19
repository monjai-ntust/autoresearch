"""Preparation-stage controller for the canonical workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from utils.common.artifact_io import (
    DataContractError,
    _load_manifest,
    _require_fields,
    atomic_write_json,
    load_json,
    md5_file,
    sha256_file,
)
from utils.common.config import PipelineConfig
from utils.common.constants import PROTOCOL_ID, TREE_HASH_REVISION
from utils.common.environment import run_doctor
from utils.common.paths import RunLayout
from utils.preparation.acquisition import fetch_run
from utils.preparation.preparation import _locate_annotation_files, _prepare_corpus, dataset_contract_from_config, extract_zip_safely
from utils.preparation.materialization import _annotation_bundle_sha256, _materialize_dataset, _tree_inventory


def run(
    layout: RunLayout,
    config: PipelineConfig,
) -> None:
    """Validate the checkout, acquire immutable inputs, and prepare data."""

    initialize(layout, config)

    # Both producers authenticate existing artifacts on resume.
    fetch_run(layout, config)
    prepare_run(layout, config)


def initialize(layout: RunLayout, config: PipelineConfig) -> None:
    doctor_path = layout.resolve("manifests/00-checkout-manifest.json")
    if doctor_path.is_file():
        from utils.preparation.checkout import continue_checkout
        continue_checkout(layout, config)
        _validate_existing_checkout(layout, config)
    else:
        manifest, passed = run_doctor(layout, config)
        if not passed:
            raise DataContractError(f"Checkout doctor failed: {manifest}")


def require_canonical_namespace(layout: RunLayout) -> None:
    for relative in ('manifests/comparison-matrix.json',
                     'manifests/comparison-matrix-link.json',
                     'manifests/encoder-comparison-selection.json'):
        if layout.resolve(relative).exists():
            raise DataContractError('canonical workflow cannot consume a comparison namespace')



def _validate_existing_checkout(layout: RunLayout, config) -> dict:
    """Authenticate an existing run namespace before any downstream work."""
    path = layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    manifest = _load_manifest(path, "checkout manifest")
    if manifest.get("schema_version") not in {
        "phase-b-checkout-manifest-1.0",
        "phase-b-checkout-manifest-2.0",
    }:
        raise DataContractError("checkout manifest schema_version is unsupported")
    _require_fields(
        manifest,
        {
            "protocol_id": config.value["protocol_id"],
            "workflow_id": config.value["workflow_id"],
            "matcher_id": config.value["matcher_id"],
            "run_id": layout.run_id,
            "status": "pass",
        },
        "checkout manifest",
    )
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("worktree_clean") is not True:
        raise DataContractError("existing run was not admitted from a clean checkout")
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=layout.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=layout.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source.get("commit") != current or status:
        raise DataContractError(
            "existing run checkout differs from the current clean source commit"
        )
    config_relative = config.path.resolve().relative_to(layout.source_root).as_posix()
    source_config = source.get("config")
    if source_config != {"path": config_relative}:
        raise DataContractError(
            "existing run checkout does not authenticate its selected config"
        )
    return manifest




def prepare_run(layout: RunLayout, config: PipelineConfig) -> dict[str, Any]:
    """Extract, validate, reconstruct, and independently rematerialize CODE-ACCORD."""

    layout.require_existing()
    contract = dataset_contract_from_config(config)
    acquisition_path = layout.resolve(
        "manifests/02-input-acquisition-manifest.json", must_exist=True
    )
    acquisition = load_json(acquisition_path)
    archive_value = acquisition.get("archive") if isinstance(acquisition, dict) else None
    if not isinstance(archive_value, dict) or not isinstance(archive_value.get("path"), str):
        raise DataContractError("input acquisition manifest lacks archive.path")
    archive_path = layout.resolve(archive_value["path"], must_exist=True)
    if (
        archive_path.stat().st_size != contract.archive_bytes
        or md5_file(archive_path) != contract.archive_md5
        or archive_value.get("bytes") != contract.archive_bytes
        or archive_value.get("md5") != contract.archive_md5
        or archive_value.get("sha256") != sha256_file(archive_path)
    ):
        raise DataContractError("acquisition manifest/archive identity mismatch")

    manifest_path = layout.resolve("manifests/03-data-preparation-manifest.json")
    data_root = layout.resolve("data-prepared")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        actual_rows, actual_tree_hash = _tree_inventory(data_root)
        if (manifest.get("dataset_tree_sha256") != actual_tree_hash
                or manifest.get('artifacts') != actual_rows):
            raise DataContractError("existing prepared dataset tree hash mismatch")
        if (manifest.get('protocol_id') != PROTOCOL_ID
                or manifest.get('dataset_id') != contract.dataset_id
                or manifest.get('archive_sha256') != sha256_file(archive_path)
                or manifest.get('acquisition_manifest_sha256') != sha256_file(acquisition_path)
                or manifest.get('annotation_bundle_sha256') != _annotation_bundle_sha256(contract)
                or manifest.get('byte_identical_independent_materializations') is not True):
            raise DataContractError('existing preparation is not bound to its authenticated inputs')
        return manifest

    extracted = extract_zip_safely(
        archive_path,
        layout.resolve("inputs/extracted/CODE-ACCORD-v1.0.0-annotations"),
        include_suffixes=tuple(contract.annotation_files),
    )
    annotation_paths = _locate_annotation_files(extracted, contract)
    (
        entities,
        relations,
        repair_ledger,
        split,
        relation_split_audit,
        alignment_audit,
        raw_relation_rows,
    ) = _prepare_corpus(annotation_paths, contract)

    first = layout.resolve(".prepare-materialization-a.partial")
    second = layout.resolve(".prepare-materialization-b.partial")
    _materialize_dataset(
        first,
        contract=contract,
        entities=entities,
        relations=relations,
        repair_ledger=repair_ledger,
        alignment_audit=alignment_audit,
        raw_relation_rows=raw_relation_rows,
        split=split,
        relation_split_audit=relation_split_audit,
    )
    _materialize_dataset(
        second,
        contract=contract,
        entities=entities,
        relations=relations,
        repair_ledger=repair_ledger,
        alignment_audit=alignment_audit,
        raw_relation_rows=raw_relation_rows,
        split=split,
        relation_split_audit=relation_split_audit,
    )
    first_rows, first_hash = _tree_inventory(first)
    second_rows, second_hash = _tree_inventory(second)
    if first_hash != second_hash or first_rows != second_rows:
        raise DataContractError("independent CODE-ACCORD materializations differ")
    if any(data_root.iterdir()):
        actual_rows, actual_hash = _tree_inventory(data_root)
        if actual_rows != first_rows or actual_hash != first_hash:
            raise DataContractError("unmanifested prepared data differs from deterministic materialization")
        shutil.rmtree(first)
    else:
        data_root.rmdir()
        os.replace(first, data_root)
    shutil.rmtree(second)

    final_rows, final_hash = _tree_inventory(data_root)
    if final_hash != first_hash or final_rows != first_rows:
        raise DataContractError("promoted prepared dataset differs from verified staging")
    manifest = {
        "schema_version": "phase-b-data-preparation-manifest-2.0",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": contract.dataset_id,
        "archive_sha256": sha256_file(archive_path),
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "annotation_bundle_sha256": _annotation_bundle_sha256(contract),
        "tree_hash_revision": TREE_HASH_REVISION,
        "first_materialization_tree_sha256": first_hash,
        "second_materialization_tree_sha256": second_hash,
        "dataset_tree_sha256": final_hash,
        "byte_identical_independent_materializations": True,
        "counts": contract.counts,
        "split": {"train": 586, "development": 103, "test": 173, "seed": 42},
        "repair_rows": 1,
        "unresolved_alignments": alignment_audit["summary"]["unresolved_marker_arguments"],
        "typed_strict_ineligible_rows": alignment_audit["summary"]["affected_positive_relation_rows"],
        "typed_strict_eligible_rows": alignment_audit["summary"]["typed_strict_eligible_positive_rows"],
        "typed_strict_unique_triples": alignment_audit["summary"]["typed_strict_unique_triples"],
        "artifacts": final_rows,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
