"""Exclusive run execution and durable, hash-bound invocation accounting."""
from __future__ import annotations

import os
import math
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from utils.common.artifact_io import DataContractError, atomic_write_json, load_json, sha256_file
from utils.common.paths import RunLayout


@contextmanager
def exclusive_run(layout: RunLayout, stage: str = "workflow"):
    """OS locks release on process exit; the retained lock file is never a lease."""
    path = layout.resolve(f"logs/locks/{stage}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            raise DataContractError(f"another command owns {layout.run_id}/{stage}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)


def bind_json(path: Path, value: dict) -> None:
    """Finalize deterministic evidence only if any existing bytes are identical."""
    from utils.common.artifact_io import canonical_json_bytes, atomic_write_bytes
    retain_bytes(path, canonical_json_bytes(value))


def retain_bytes(path: Path, expected: bytes) -> None:
    """Pure-product finalization preserves existing equal bytes and rejects drift."""
    from utils.common.artifact_io import atomic_write_bytes
    if path.is_symlink() or path.parent.resolve() != path.parent.absolute():
        raise DataContractError("durable output is reached through a symlink or junction")
    if path.exists():
        if path.read_bytes() != expected:
            raise DataContractError(f"existing durable evidence differs: {path.name}")
    else:
        atomic_write_bytes(path, expected)


def retain_jsonl(path: Path, rows) -> None:
    from utils.common.artifact_io import canonical_json_bytes
    retain_bytes(path, b''.join(canonical_json_bytes(row) for row in rows))


def retain_text(path: Path, value: str) -> None:
    retain_bytes(path, value.encode('utf-8'))


def checked_process(layout: RunLayout, command: list[str], *, label: str,
                    limit_seconds: float | None = None, command_runner=subprocess.run) -> None:
    """Charge every invocation, including failures and abandoned starts.

    A start without its finish is conservatively charged through the next
    invocation. This bound is elapsed accelerator-job time, not GPU utilization.
    """
    directory = layout.resolve(f"logs/invocations/{label}")
    directory.mkdir(parents=True, exist_ok=True)
    starts = sorted(directory.glob("*-start.json"))
    charged = 0.0
    now = time.time()
    for index, start_path in enumerate(starts, 1):
        if start_path.name != f"{index:04d}-start.json":
            raise DataContractError("invocation accounting is not contiguous")
        start = load_json(start_path)
        if start.get("run_id") != layout.run_id or start.get("label") != label:
            raise DataContractError("invocation accounting belongs to another run")
        finish_path = start_path.with_name(start_path.name.replace("-start", "-finish"))
        if finish_path.exists():
            finish = load_json(finish_path)
            if finish.get("start_sha256") != sha256_file(start_path):
                raise DataContractError("invocation accounting hash differs")
            elapsed = finish.get("elapsed_seconds")
            if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                    or not math.isfinite(elapsed) or elapsed < 0
                    or finish.get("status") not in {"completed", "failed"}):
                raise DataContractError("invocation timing is malformed")
            charged += elapsed
        else:
            charged += max(0.0, now - start["started_unix"])
    remaining = None if limit_seconds is None else limit_seconds - charged
    if remaining is not None and remaining <= 0:
        raise DataContractError(f"{label} cumulative execution limit exhausted")
    ordinal = len(starts) + 1
    start_path = directory / f"{ordinal:04d}-start.json"
    atomic_write_json(start_path, {"run_id": layout.run_id, "label": label,
        "started_unix": now, "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command, "owner_pid": os.getpid(), "previous_charged_seconds": charged})
    started = time.monotonic()
    status = "failed"
    try:
        kwargs = {"cwd": layout.source_root, "check": True}
        if remaining is not None:
            kwargs["timeout"] = remaining
        command_runner(command, **kwargs)
        status = "completed"
    finally:
        atomic_write_json(directory / f"{ordinal:04d}-finish.json", {
            "start_sha256": sha256_file(start_path), "status": status,
            "elapsed_seconds": time.monotonic() - started})


def require_active_dispatch(layout: RunLayout, arguments: list[str], *,
                            label: str = "comparison-full") -> None:
    starts = sorted(layout.resolve(f"logs/invocations/{label}").glob("*-start.json"))
    if not starts:
        raise DataContractError("comparison workers require an accounted matrix dispatch")
    path = starts[-1]
    start = load_json(path)
    finish = path.with_name(path.name.replace("-start", "-finish"))
    if (finish.exists() or start.get("run_id") != layout.run_id
            or start.get("label") != label or start.get("owner_pid") != os.getppid()
            or start.get("command", [])[3:] != arguments):
        raise DataContractError("private worker is outside its active matrix dispatch")
