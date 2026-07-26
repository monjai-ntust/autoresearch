"""Dataset adapter protocol and validation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from graph_rag_eval.contracts import CanonicalBundle


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    adapter_id: str
    ready: bool
    issues: tuple[ValidationIssue, ...]
    capabilities: tuple[str, ...]


class DatasetAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def validate(self) -> ValidationReport: ...

    def load(self) -> CanonicalBundle: ...
