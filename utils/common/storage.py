"""Advisory remaining-storage estimates scoped to one workflow's outputs."""
from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable


def existing_storage_bytes(filesystem: Path, roots: Iterable[Path], *,
                           excluded: tuple[Path, ...] = ()) -> int:
    """Count allocated regular files once, without credit from other volumes."""
    device = filesystem.stat().st_dev
    seen = set()
    existing = 0
    for root in roots:
        for path in root.rglob("*"):
            if (any(path == skip or skip in path.parents for skip in excluded)
                    or path.is_symlink() or path.resolve() != path.absolute()):
                continue
            info = path.lstat()
            identity = (info.st_dev, info.st_ino)
            if (not stat.S_ISREG(info.st_mode) or info.st_dev != device
                    or identity in seen):
                continue
            seen.add(identity)
            existing += info.st_blocks * 512 if hasattr(info, "st_blocks") else info.st_size
    return existing


def warn_storage(filesystem: Path, *, budget: int, roots: Iterable[Path], label: str,
                 excluded: tuple[Path, ...] = ()) -> tuple[int | None, int | None]:
    """Warn below the outstanding budget; storage inspection never blocks work."""
    try:
        existing = existing_storage_bytes(filesystem, roots, excluded=excluded)
        required = max(0, budget - existing)
        free = shutil.disk_usage(filesystem).free
    except OSError as exc:
        print(f"{label}: warning: cannot estimate storage: {exc}. Continuing execution.", file=sys.stderr)
        return None, None
    if free < required:
        print(f"{label}: warning: storage budget {budget:,} bytes ({budget / 1e9:g} GB) "
            f"minus {existing:,} existing output bytes leaves {required:,} additional "
            f"bytes estimated; {free:,} bytes free. Continuing execution.", file=sys.stderr)
    return free, required
