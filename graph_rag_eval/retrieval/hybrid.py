"""Deterministic reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter

from graph_rag_eval.budget import Budget
from graph_rag_eval.contracts import QueryView
from graph_rag_eval.identity import fingerprint
from graph_rag_eval.retrieval.base import EvidenceItem, RetrievalResult, finalize_result


class ReciprocalRankFusionRetriever:
    retriever_id = "rrf-v1"

    def __init__(self, *retrievers, rrf_k: int = 60):
        if len(retrievers) < 2:
            raise ValueError("hybrid retrieval requires at least two retrievers")
        self.retrievers = tuple(retrievers)
        self.rrf_k = rrf_k
        self.index_fingerprint = fingerprint(
            "rrf-index-v1",
            {
                "rrf_k": rrf_k,
                "children": [item.index_fingerprint for item in self.retrievers],
            },
        )

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult:
        started = perf_counter()
        child_budget = Budget(max_items=max(100, budget.max_items * 10), max_tokens=10**9)
        child_results = [item.retrieve(query, child_budget) for item in self.retrievers]
        scores = defaultdict(float)
        rows = {}
        components = defaultdict(dict)
        for result in child_results:
            for rank, item in enumerate(result.items, start=1):
                contribution = 1.0 / (self.rrf_k + rank)
                scores[item.evidence_id] += contribution
                components[item.evidence_id][result.retriever_id] = contribution
                rows.setdefault(item.evidence_id, item)
        ranking = sorted(scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id))
        fused = tuple(
            EvidenceItem(
                evidence_id=evidence_id,
                evidence_type=rows[evidence_id].evidence_type,
                content=rows[evidence_id].content,
                score=scores[evidence_id],
                rank=rank,
                provenance_ids=rows[evidence_id].provenance_ids,
                path_ids=rows[evidence_id].path_ids,
                components=components[evidence_id],
            )
            for rank, evidence_id in enumerate(ranking, start=1)
        )
        # A child failure is reported even when the surviving child returns rows;
        # otherwise an empty graph link is invisible in every hybrid condition.
        failures = sorted(
            {
                f"{result.retriever_id}:{result.failure}"
                for result in child_results
                if result.failure
            }
        )
        return finalize_result(
            retriever_id=self.retriever_id,
            query=query,
            rows=fused,
            budget=budget,
            index_fingerprint=self.index_fingerprint,
            latency_ms=(perf_counter() - started) * 1000.0,
            failure=";".join(failures) if failures else None,
            seed_ids=tuple(
                seed for result in child_results for seed in result.seed_ids
            ),
            expansion_ids=tuple(
                item for result in child_results for item in result.expansion_ids
            ),
        )
