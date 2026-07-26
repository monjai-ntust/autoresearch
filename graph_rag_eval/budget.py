"""Shared deterministic item/token budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Protocol


class EvidenceLike(Protocol):
    evidence_id: str
    content: str


def tokenize_budget(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


@dataclass(frozen=True)
class Budget:
    max_items: int
    max_tokens: int
    tokenizer_id: str = "unicode-word-punctuation-v1"

    def __post_init__(self) -> None:
        if self.max_items < 0 or self.max_tokens < 0:
            raise ValueError("budget limits cannot be negative")


@dataclass(frozen=True)
class BudgetDecision:
    selected_ids: tuple[str, ...]
    dropped_duplicate_ids: tuple[str, ...]
    truncated_ids: tuple[str, ...]
    requested_items: int
    realized_items: int
    realized_tokens: int
    max_items: int
    max_tokens: int


def apply_budget(items: Iterable[EvidenceLike], budget: Budget):
    rows = list(items)
    selected = []
    duplicates = []
    truncated = []
    seen = set()
    used_tokens = 0
    for row in rows:
        if row.evidence_id in seen:
            duplicates.append(row.evidence_id)
            continue
        seen.add(row.evidence_id)
        tokens = len(tokenize_budget(row.content))
        if len(selected) >= budget.max_items or used_tokens + tokens > budget.max_tokens:
            truncated.append(row.evidence_id)
            continue
        selected.append(row)
        used_tokens += tokens
    return tuple(selected), BudgetDecision(
        selected_ids=tuple(row.evidence_id for row in selected),
        dropped_duplicate_ids=tuple(duplicates),
        truncated_ids=tuple(truncated),
        requested_items=len(rows),
        realized_items=len(selected),
        realized_tokens=used_tokens,
        max_items=budget.max_items,
        max_tokens=budget.max_tokens,
    )


def assert_matched_budgets(decisions: Iterable[BudgetDecision]) -> None:
    rows = tuple(decisions)
    if not rows:
        return
    requested = {(row.max_items, row.max_tokens) for row in rows}
    if len(requested) != 1:
        raise ValueError("comparison uses unmatched requested budgets")
