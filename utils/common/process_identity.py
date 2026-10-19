"""One immutable observation of a producing Python process, never a seed change.

Call ``capture_process_start`` at the entry point before application imports.
Python exposes whether hash randomization and environment isolation are enabled;
it does not expose the original seed environment or the internal hash secret.
Linux's initial exec environment provides separate OS evidence when available.
An ordinary environment snapshot is never substituted for that evidence.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path


_IDENTITY = None


def _setting(value):
    if value is None:
        return {"kind": "unset", "value": None}
    if value == "random":
        return {"kind": "random", "value": value}
    if value == "0":
        return {"kind": "zero", "value": value}
    try:
        numeric = int(value, 10)
    except (TypeError, ValueError):
        numeric = -1
    if 0 <= numeric <= 4294967295 and value.isascii() and value.isdecimal():
        return {"kind": "numeric", "value": value}
    return {"kind": "invalid", "value": value}


def _initial_setting():
    # Linux documents this as the initial exec environment, unaffected by ordinary
    # putenv/os.environ changes. Privileged native code can relocate the kernel's
    # environment range; this is OS provenance, not access to Python's secret.
    # Inspect only this one key and never retain the remaining environment.
    if sys.platform != "linux":
        return {"kind": "unavailable", "value": None,
                "evidence": "initial_environment_unavailable_on_this_platform"}
    try:
        raw = Path("/proc/self/environ").read_bytes()
    except OSError:
        return {"kind": "unavailable", "value": None,
                "evidence": "linux_initial_environment_unreadable"}
    if raw and not raw.endswith(b"\0"):
        return {"kind": "unavailable", "value": None,
                "evidence": "linux_initial_environment_malformed"}
    values = [entry[len(b"PYTHONHASHSEED="):]
              for entry in raw.split(b"\0") if entry.startswith(b"PYTHONHASHSEED=")]
    if len(values) > 1:
        return {"kind": "unavailable", "value": None,
                "evidence": "linux_initial_environment_duplicate_key"}
    try:
        value = values[0].decode("ascii") if values else None
    except UnicodeDecodeError:
        return {"kind": "unavailable", "value": None,
                "evidence": "linux_initial_environment_non_ascii_seed"}
    setting = _setting(value)
    setting["evidence"] = "linux_proc_initial_exec_environment"
    return setting


def _effective_setting(initial):
    # These flags describe this interpreter, even after os.environ is mutated.
    # In particular, -I/-E ignores PYTHONHASHSEED, including a supplied zero.
    if not sys.flags.hash_randomization:
        return {"kind": "zero", "value": "0",
                "evidence": "sys.flags.hash_randomization=0"}
    if sys.flags.ignore_environment:
        return {"kind": "random", "value": None,
                "evidence": "sys.flags.ignore_environment=1",
                "secret": "unavailable_through_supported_runtime_api"}
    if initial["kind"] in {"unset", "random"}:
        return {"kind": "random", "value": None,
                "evidence": "linux_proc_initial_exec_environment_and_runtime_flags",
                "secret": "unavailable_through_supported_runtime_api"}
    if initial["kind"] == "numeric" and int(initial["value"], 10) != 0:
        return {"kind": "numeric", "value": initial["value"],
                "evidence": "linux_proc_initial_exec_environment_and_runtime_flags"}
    return {"kind": "unavailable", "value": None,
            "evidence": "original_environment_not_exposed_by_supported_runtime_api",
            "secret": "unavailable_through_supported_runtime_api"}


def capture_process_start(entrypoint: str | None = None):
    """Freeze the earliest observation once per PID and return an independent copy.

    ``entrypoint`` identifies the capture boundary; it cannot certify an original
    environment value. Linux initial-environment evidence is labelled separately;
    other platforms retain unavailable original settings. Forked children receive
    a new identity instead of inheriting their parent's producing-process identity.
    """
    global _IDENTITY
    pid = os.getpid()
    if _IDENTITY is None or _IDENTITY["pid"] != pid:
        observed_ns = time.time_ns()
        initial = _initial_setting()
        _IDENTITY = {
            "schema_version": "python-process-observation-1",
            "process_id": f"{pid}-{observed_ns}",
            "pid": pid,
            "parent_pid": os.getppid(),
            "observed_unix_ns": observed_ns,
            "capture_scope": "entrypoint" if entrypoint is not None else "late_import",
            "entrypoint": entrypoint,
            "python": {
                "implementation": sys.implementation.name,
                "version": sys.version.split()[0],
                "isolated": bool(sys.flags.isolated),
                "ignore_environment": bool(sys.flags.ignore_environment),
                "hash_randomization": bool(sys.flags.hash_randomization),
                "hash_algorithm": sys.hash_info.algorithm,
                "hash_width": sys.hash_info.width,
            },
            "python_hash_seed": {
                "environment_at_capture": _setting(os.environ.get("PYTHONHASHSEED")),
                "environment_evidence": "os.environ_at_capture_not_original_startup_proof",
                "startup_setting": initial,
                "effective_setting": _effective_setting(initial),
            },
        }
    return copy.deepcopy(_IDENTITY)


def process_identity():
    """Return the frozen process observation, marking an uncaptured caller late."""
    return capture_process_start()
