"""Evidence-set retrieval metrics."""

from __future__ import annotations

from math import log2

from graph_rag_eval.capabilities import Capability, CapabilityUnavailable
from graph_rag_eval.contracts import EvidenceSet, RelevanceJudgment
from graph_rag_eval.retrieval.base import RetrievalResult


def retrieval_metrics(
    result: RetrievalResult,
    evidence_sets: tuple[EvidenceSet, ...],
    judgments: tuple[RelevanceJudgment, ...],
    *,
    capabilities: tuple[str, ...],
    k: int,
) -> dict[str, float | int | None | str | list[str]]:
    unavailable = CapabilityUnavailable.for_required(
        "retrieval_metrics",
        (Capability.EVIDENCE_SETS,),
        capabilities,
    )
    if unavailable is not None:
        return {
            "status": unavailable.status,
            "missing": list(unavailable.missing),
            "reason": unavailable.reason,
            "denominator": 0,
        }
    relevant_sets = [
        set(item.evidence_ids)
        for item in evidence_sets
        if item.question_id == result.query.question_id
    ]
    if not relevant_sets:
        return {
            "status": "not_applicable",
            "missing": ["question_evidence_set"],
            "reason": "question has no frozen acceptable evidence set",
            "denominator": 0,
        }
    ranked = [item.evidence_id for item in result.items[:k]]
    recall = max(
        len(set(ranked) & relevant) / len(relevant)
        for relevant in relevant_sets
    )
    all_relevant = set().union(*relevant_sets)
    relevant_ranks = [index for index, item in enumerate(ranked, start=1) if item in all_relevant]
    reciprocal_rank = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
    success = float(any(set(ranked) >= relevant for relevant in relevant_sets))
    precision = len(set(ranked) & all_relevant) / len(ranked) if ranked else 0.0
    grades = {
        item.evidence_id: item.grade
        for item in judgments
        if item.question_id == result.query.question_id
    }
    ndcg = None
    if grades:
        gains = [grades.get(item, 0) for item in ranked]
        dcg = sum((2**grade - 1) / log2(index + 1) for index, grade in enumerate(gains, start=1))
        ideal = sorted(grades.values(), reverse=True)[:k]
        idcg = sum((2**grade - 1) / log2(index + 1) for index, grade in enumerate(ideal, start=1))
        ndcg = dcg / idcg if idcg else None
    return {
        "status": "available",
        "k": k,
        "evidence_recall_at_k": recall,
        "mrr": reciprocal_rank,
        "success_at_k": success,
        "precision_at_k": precision,
        "ndcg_at_k": ndcg,
        "denominator": 1,
    }
