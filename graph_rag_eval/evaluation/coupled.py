"""Coupled answer-and-provenance outcomes."""

from __future__ import annotations


def coupled_metrics(
    answer: dict, retrieval: dict, support: dict
) -> dict[str, float | str | list[str]]:
    """Combine answer, evidence-sufficiency, and support outcomes for one question.

    Every component must be `available`; otherwise the record carries no numeric
    value, because a coupled 0.0 derived from a missing prerequisite would be
    indistinguishable from a measured end-to-end failure.
    """

    missing = [
        name
        for name, record in (
            ("answer", answer),
            ("retrieval", retrieval),
            ("support", support),
        )
        if record.get("status") != "available"
    ]
    if missing:
        return {
            "status": "not_applicable",
            "missing": missing,
            "reason": "coupled outcome requires available answer, retrieval, and support metrics",
        }
    answer_correct = float(answer.get("answer_correct", 0.0)) == 1.0
    sufficient = float(retrieval.get("sufficient_evidence_at_k", 0.0)) == 1.0
    supported = float(support.get("support_recall", 0.0)) == 1.0
    return {
        "status": "available",
        "answer_correct": float(answer_correct),
        "sufficient_evidence_retrieved": float(sufficient),
        "fully_supported": float(supported),
        "supported_answer": float(answer_correct and sufficient and supported),
    }
