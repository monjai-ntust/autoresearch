"""Dataset capability declarations and typed unavailable outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Capability(str, Enum):
    DOCUMENTS = "documents"
    DOCUMENT_CLUSTERS = "document_clusters"
    GOLD_ENTITIES = "gold_entities"
    GOLD_RELATIONS = "gold_relations"
    GOLD_TRIPLES = "gold_triples"
    SOURCE_SPANS = "source_spans"
    QUESTIONS = "questions"
    ANSWER_ALIASES = "answer_aliases"
    EVIDENCE_SETS = "evidence_sets"
    GRADED_RELEVANCE = "graded_relevance"
    UNANSWERABLE_QUESTIONS = "unanswerable_questions"
    GOLD_PATHS = "gold_paths"


@dataclass(frozen=True)
class CapabilityUnavailable:
    """Machine-readable reason a requested operation is not applicable."""

    operation: str
    missing: tuple[str, ...]
    reason: str
    status: str = "not_applicable"

    @classmethod
    def for_required(
        cls,
        operation: str,
        required: Iterable[Capability | str],
        available: Iterable[Capability | str],
        *,
        reason: str | None = None,
    ) -> "CapabilityUnavailable | None":
        have = {str(item.value if isinstance(item, Capability) else item) for item in available}
        missing = tuple(
            sorted(
                str(item.value if isinstance(item, Capability) else item)
                for item in required
                if str(item.value if isinstance(item, Capability) else item) not in have
            )
        )
        if not missing:
            return None
        return cls(
            operation=operation,
            missing=missing,
            reason=reason or f"missing required capabilities: {', '.join(missing)}",
        )


def capability_values(items: Iterable[Capability | str]) -> tuple[str, ...]:
    return tuple(
        sorted({str(item.value if isinstance(item, Capability) else item) for item in items})
    )
