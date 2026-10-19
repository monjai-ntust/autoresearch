"""Retain and retry fresh checkout-only failures without admitting old work."""
from __future__ import annotations

import hashlib
import re

from utils.common.artifact_io import (
    DataContractError, atomic_write_bytes, canonical_json_bytes, load_json,
)
from utils.common.environment import inspect_checkout
from utils.common.execution import retain_bytes

CHECKOUT = "manifests/00-checkout-manifest.json"
CONFIG = "inputs/config/pipeline.json"
RECOVERY = "manifests/checkout-recovery"
_SNAPSHOTS = {
    "previous_checkout": "previous-checkout.json",
    "fresh_checkout": "fresh-checkout.json",
    "config": "config.json",
}
_RETRYABLE_CHECKS = {"clean-tracked-source", "root-output-ignore-rule",
                     "root-output-ignore-negative", "source-bytecode-disabled"}


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _blocked_checkout(document, layout, config, config_bytes):
    expected = {
        "schema_version": "phase-b-checkout-manifest-2.0",
        "lifecycle_version": 1,
        "protocol_id": config.value["protocol_id"],
        "workflow_id": config.value["workflow_id"],
        "matcher_id": config.value["matcher_id"],
        "run_id": layout.run_id,
        "status": "blocked",
    }
    if (not isinstance(document, dict)
            or "checkout_recovery" in document
            or type(document.get("lifecycle_version")) is not int
            or any(document.get(key) != value for key, value in expected.items())):
        raise DataContractError("checkout continuation requires a fresh lifecycle-1 blocked run")
    source = document.get("source")
    checks = document.get("checks")
    if (not isinstance(source, dict) or type(source.get("worktree_clean")) is not bool
            or not isinstance(source.get("commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is None
            or not isinstance(checks, list) or not checks
            or any(not isinstance(item, dict) or not isinstance(item.get("check_id"), str)
                   or item.get("status") not in {"pass", "fail"} for item in checks)
            or len({item["check_id"] for item in checks}) != len(checks)):
        raise DataContractError("checkout continuation requires a retryable fresh doctor failure")
    failures = {item["check_id"] for item in checks if item["status"] == "fail"}
    if (not failures or not failures <= _RETRYABLE_CHECKS
            or ("clean-tracked-source" in failures) != (source["worktree_clean"] is False)):
        raise DataContractError("checkout continuation requires a dirty-source or retired-check failure")
    identity = {"path": config.path.relative_to(layout.source_root).as_posix(),
                "sha256": _digest(config_bytes), "run_local_path": CONFIG}
    if source.get("config") != identity or config.path.read_bytes() != config_bytes:
        raise DataContractError("blocked checkout configuration differs from the selected configuration")


def _inventory(layout, *, checkout_only):
    """Reject redirected recovery paths and any pre-admission scientific files."""
    ordinary = {CHECKOUT, CONFIG, "logs/locks/workflow.lock"}
    snapshots = {f"{RECOVERY}/{name}" for name in _SNAPSHOTS.values()}
    receipts = []
    scan_root = layout.run_root if checkout_only else layout.run_root / RECOVERY
    if scan_root.resolve() != scan_root.absolute():
        raise DataContractError("checkout continuation refuses redirected run paths")
    for path in scan_root.rglob("*"):
        relative = path.relative_to(layout.run_root).as_posix()
        in_recovery = relative == RECOVERY or relative.startswith(RECOVERY + "/")
        if (checkout_only or in_recovery) and path.resolve() != path.absolute():
            raise DataContractError("checkout continuation refuses redirected run paths")
        if not path.is_file():
            continue
        if in_recovery:
            match = re.fullmatch(rf"{RECOVERY}/receipt-([0-9a-f]{{64}})\.json", relative)
            if match:
                if _digest(path.read_bytes()) != match[1]:
                    raise DataContractError("checkout recovery receipt hash differs")
                receipts.append(path)
            elif relative not in snapshots:
                raise DataContractError(f"unknown checkout recovery evidence: {relative}")
        elif checkout_only and relative not in ordinary:
            raise DataContractError(f"checkout continuation refuses downstream or unknown artifact: {relative}")
    if len(receipts) > 1:
        raise DataContractError("checkout recovery has multiple promotion receipts")
    return receipts


def _receipt(layout, payloads):
    return {"schema_version": "checkout-recovery-1.0", "run_id": layout.run_id,
            "artifacts": {key: {"path": f"{RECOVERY}/{_SNAPSHOTS[key]}",
                                "sha256": _digest(value)}
                          for key, value in payloads.items()}}


def _validate_retained(layout, config, path):
    receipt = load_json(path)
    payloads = {key: layout.resolve(f"{RECOVERY}/{name}", must_exist=True).read_bytes()
                for key, name in _SNAPSHOTS.items()}
    if receipt != _receipt(layout, payloads):
        raise DataContractError("checkout recovery receipt belongs to different or changed evidence")
    previous = load_json(layout.resolve(f"{RECOVERY}/previous-checkout.json", must_exist=True))
    _blocked_checkout(previous, layout, config, payloads["config"])
    fresh = load_json(layout.resolve(f"{RECOVERY}/fresh-checkout.json", must_exist=True))
    marker = {"receipt_directory": RECOVERY,
              "previous_checkout_sha256": _digest(payloads["previous_checkout"])}
    if (not isinstance(fresh, dict) or fresh.get("checkout_recovery") != marker
            or fresh.get("status") != "pass" or fresh.get("run_id") != layout.run_id):
        raise DataContractError("checkout recovery replacement identity differs")
    return payloads, fresh


def continue_checkout(layout, config):
    """Retry only a fresh failed doctor, retaining both checkout identities.

    The caller owns the run's OS lock. Before replacement, every retained byte
    and its receipt is durable; only the active manifest needs atomic promotion.
    """
    active_path = layout.resolve(CHECKOUT, must_exist=True)
    active_bytes = active_path.read_bytes()
    active = load_json(active_path)
    has_marker = isinstance(active, dict) and "checkout_recovery" in active
    recovery = layout.resolve(RECOVERY)
    if isinstance(active, dict) and active.get("status") == "pass" and not has_marker and not recovery.exists():
        return
    checkout_only = not isinstance(active, dict) or active.get("status") != "pass"
    receipts = _inventory(layout, checkout_only=checkout_only)
    if has_marker and not receipts:
        raise DataContractError("checkout recovery promotion receipt is missing")
    if not receipts:
        config_bytes = layout.resolve(CONFIG, must_exist=True).read_bytes()
        _blocked_checkout(active, layout, config, config_bytes)
        fresh, passed = inspect_checkout(layout, config)
        if not passed:
            raise DataContractError(f"Checkout doctor failed: {fresh}")
        fresh["checkout_recovery"] = {"receipt_directory": RECOVERY,
                                      "previous_checkout_sha256": _digest(active_bytes)}
        payloads = {"previous_checkout": active_bytes,
                    "fresh_checkout": canonical_json_bytes(fresh), "config": config_bytes}
        for key, value in payloads.items():
            retain_bytes(layout.resolve(f"{RECOVERY}/{_SNAPSHOTS[key]}"), value)
        receipt_bytes = canonical_json_bytes(_receipt(layout, payloads))
        receipt_path = layout.resolve(f"{RECOVERY}/receipt-{_digest(receipt_bytes)}.json")
        retain_bytes(receipt_path, receipt_bytes)
        receipts = [receipt_path]
    payloads, fresh = _validate_retained(layout, config, receipts[0])
    if active_bytes not in {payloads["previous_checkout"], payloads["fresh_checkout"]}:
        raise DataContractError("active checkout differs from retained recovery evidence")
    if layout.resolve(CONFIG, must_exist=True).read_bytes() != payloads["config"]:
        raise DataContractError("run-local configuration changed during checkout recovery")
    if active_bytes == payloads["fresh_checkout"]:
        return
    current, passed = inspect_checkout(layout, config)
    if (not passed or current.get("source") != fresh.get("source")
            or any(current.get(key) != fresh.get(key) for key in (
                "schema_version", "lifecycle_version", "protocol_id", "workflow_id", "matcher_id", "run_id"))):
        raise DataContractError(f"Checkout recovery requires the recorded clean source: {current}")
    _inventory(layout, checkout_only=True)
    if active_path.read_bytes() != active_bytes:
        raise DataContractError("blocked checkout changed during recovery")
    atomic_write_bytes(active_path, payloads["fresh_checkout"])
