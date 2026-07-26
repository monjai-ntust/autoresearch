"""Common question-only retrieval contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from graph_rag_eval.budget import Budget, BudgetDecision, apply_budget
from graph_rag_eval.contracts import QueryView


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    evidence_type: str
    content: str
    score: float
    rank: int
    provenance_ids: tuple[str, ...] = ()
    path_ids: tuple[str, ...] = ()
    components: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    retriever_id: str
    query: QueryView
    items: tuple[EvidenceItem, ...]
    budget: BudgetDecision
    index_fingerprint: str
    latency_ms: float
    failure: str | None = None
    seed_ids: tuple[str, ...] = ()
    expansion_ids: tuple[str, ...] = ()


class Retriever(Protocol):
    retriever_id: str
    index_fingerprint: str

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult: ...


def finalize_result(
    *,
    retriever_id: str,
    query: QueryView,
    rows: tuple[EvidenceItem, ...],
    budget: Budget,
    index_fingerprint: str,
    latency_ms: float,
    failure: str | None = None,
    seed_ids: tuple[str, ...] = (),
    expansion_ids: tuple[str, ...] = (),
) -> RetrievalResult:
    selected, decision = apply_budget(rows, budget)
    selected = tuple(
        EvidenceItem(
            evidence_id=item.evidence_id,
            evidence_type=item.evidence_type,
            content=item.content,
            score=item.score,
            rank=rank,
            provenance_ids=item.provenance_ids,
            path_ids=item.path_ids,
            components=item.components,
        )
        for rank, item in enumerate(selected, start=1)
    )
    return RetrievalResult(
        retriever_id=retriever_id,
        query=query,
        items=selected,
        budget=decision,
        index_fingerprint=index_fingerprint,
        latency_ms=latency_ms,
        failure=failure,
        seed_ids=seed_ids,
        expansion_ids=expansion_ids,
    )
