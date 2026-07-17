"""Path containment for the standalone Phase B output contract."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from constants import RUN_DIRECTORIES


class PathContractError(ValueError):
    """Raised when a workflow path escapes the standalone source contract."""


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def discover_source_root(start: Path | None = None) -> Path:
    """Find the direct ``src`` checkout without consulting its parent repository."""

    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "uv.lock").is_file()
            and (candidate / "phase_b.py").is_file()
        ):
            return candidate.resolve()
    raise PathContractError(
        "could not locate the standalone source root containing pyproject.toml, "
        "uv.lock, and phase_b.py"
    )


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise PathContractError(
            "run ID must be 1-64 characters, start with an alphanumeric "
            "character, and contain only alphanumerics, '.', '_', or '-'"
        )
    return run_id


def resolve_tracked_path(
    source_root: Path, supplied: str | Path, *, must_exist: bool = True
) -> Path:
    """Resolve a tracked workflow resource inside ``src`` and outside output."""

    root = source_root.resolve()
    raw = Path(supplied)
    if ".." in raw.parts:
        raise PathContractError(f"tracked workflow path must not contain '..': {supplied}")
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve(strict=must_exist)
    output_root = (root / "output").resolve(strict=False)
    if not _is_within(resolved, root) or _is_within(resolved, output_root):
        raise PathContractError(
            f"tracked workflow path must resolve inside the source checkout and "
            f"outside output/: {supplied}"
        )
    if must_exist and not resolved.is_file():
        raise PathContractError(f"tracked workflow file does not exist: {supplied}")
    return resolved


@dataclass(frozen=True)
class RunLayout:
    """Resolved, symlink-safe identity of one ignored Phase B run."""

    source_root: Path
    run_id: str

    def __post_init__(self) -> None:
        root = self.source_root.resolve()
        validate_run_id(self.run_id)
        declared_output = root / "output"
        resolved_output = declared_output.resolve(strict=False)
        if not _same_path(resolved_output, declared_output):
            raise PathContractError(
                "source output/ must be a physical directory, not a symlink or junction"
            )
        if not _is_within(resolved_output, root):
            raise PathContractError("source output/ resolves outside the source checkout")
        object.__setattr__(self, "source_root", root)

    @property
    def output_root(self) -> Path:
        return self.source_root / "output"

    @property
    def run_root(self) -> Path:
        return self.output_root / self.run_id

    def create(self) -> Path:
        """Create a new run atomically enough to reject accidental overwrite."""

        self.output_root.mkdir(parents=False, exist_ok=True)
        if not _same_path(self.output_root.resolve(), self.output_root):
            raise PathContractError(
                "source output/ became a symlink or junction during run creation"
            )
        try:
            self.run_root.mkdir()
        except FileExistsError as exc:
            raise PathContractError(
                f"run already exists; choose a new run ID: {self.run_id}"
            ) from exc
        if not _same_path(self.run_root.resolve(), self.run_root):
            raise PathContractError("new run root did not resolve to its declared path")
        for relative in RUN_DIRECTORIES:
            self.resolve(relative).mkdir(parents=True, exist_ok=False)
        return self.run_root

    def require_existing(self) -> Path:
        if not self.run_root.is_dir():
            raise PathContractError(
                f"run does not exist; execute doctor first: {self.run_id}"
            )
        if not _same_path(self.run_root.resolve(), self.run_root):
            raise PathContractError("run root is a symlink, junction, or escaped path")
        return self.run_root

    def resolve(self, relative: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve a run-relative path and reject absolute or traversal escapes."""

        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts:
            raise PathContractError(f"run path must be relative and traversal-free: {relative}")
        candidate = self.run_root / raw
        resolved = candidate.resolve(strict=must_exist)
        root = self.run_root.resolve(strict=False)
        if not _is_within(resolved, root):
            raise PathContractError(f"run path resolves outside selected run: {relative}")
        if must_exist and not resolved.exists():
            raise PathContractError(f"required run path does not exist: {relative}")
        return resolved

    def relative_identity(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        run_root = self.run_root.resolve(strict=False)
        if not _is_within(resolved, run_root):
            raise PathContractError("cannot serialize a path outside the selected run")
        return resolved.relative_to(run_root).as_posix()
