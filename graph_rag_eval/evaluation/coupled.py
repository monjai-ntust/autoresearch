"""Coupled answer-and-provenance outcomes."""

from __future__ import annotations


def coupled_metrics(answer: dict, retrieval: dict, support: dict) -> dict[str, float | str]:
    if (
        answer.get("status") != "available"
        or retrieval.get("status") != "available"
        or support.get("status") != "available"
    ):
        return {
            "status": "not_applicable",
            "supported_answer": 0.0,
            "reason": "answer, retrieval, and support metrics are not all available",
        }
    answer_correct = float(answer.get("exact_match", 0.0)) == 1.0
    sufficient = float(retrieval.get("success_at_k", 0.0)) == 1.0
    supported = float(support.get("support_recall", 0.0)) == 1.0
    return {
        "status": "available",
        "answer_correct": float(answer_correct),
        "sufficient_evidence_retrieved": float(sufficient),
        "fully_supported": float(supported),
        "supported_answer": float(answer_correct and sufficient and supported),
    }
