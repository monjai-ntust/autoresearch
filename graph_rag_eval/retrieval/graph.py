"""Question-only entity linking and bounded provenance-preserving expansion."""

from __future__ import annotations

from collections import deque
from time import perf_counter
import re

from graph_rag_eval.budget import Budget
from graph_rag_eval.contracts import QueryView
from graph_rag_eval.graphs.snapshots import GraphSnapshot
from graph_rag_eval.identity import fingerprint
from graph_rag_eval.retrieval.base import EvidenceItem, RetrievalResult, finalize_result


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


class GraphRetriever:
    retriever_id = "question-only-graph-v1"

    def __init__(self, graph: GraphSnapshot, *, max_hops: int = 2):
        if max_hops < 0:
            raise ValueError("max_hops cannot be negative")
        self.graph = graph
        self.max_hops = max_hops
        self.entity_by_id = {item.entity_id: item for item in graph.entities}
        self.relation_by_id = {item.relation_id: item for item in graph.relations}
        self.adjacency = graph.adjacency()
        self.index_fingerprint = fingerprint(
            "graph-index-v1",
            {"graph_id": graph.graph_id, "max_hops": max_hops},
        )

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult:
        started = perf_counter()
        query_tokens = _tokens(query.text)
        seeds = []
        for entity in self.graph.entities:
            label_tokens = _tokens(
                " ".join(entity.surface_forms) + " " + entity.canonical_label
            )
            union = query_tokens | label_tokens
            score = len(query_tokens & label_tokens) / len(union) if union else 0.0
            if score > 0.0:
                seeds.append((score, entity.entity_id))
        seeds.sort(key=lambda item: (-item[0], item[1]))
        if not seeds:
            return finalize_result(
                retriever_id=self.retriever_id,
                query=query,
                rows=(),
                budget=budget,
                index_fingerprint=self.index_fingerprint,
                latency_ms=(perf_counter() - started) * 1000.0,
                failure="empty_entity_link",
            )
        queue = deque((entity_id, 0, (entity_id,), score) for score, entity_id in seeds)
        visited_entities = set()
        visited_triples = set()
        rows = []
        expansion_ids = []
        while queue:
            entity_id, hop, path, seed_score = queue.popleft()
            if (entity_id, hop) in visited_entities or hop >= self.max_hops:
                continue
            visited_entities.add((entity_id, hop))
            for triple in self.adjacency.get(entity_id, ()):
                if triple.triple_id in visited_triples:
                    continue
                visited_triples.add(triple.triple_id)
                expansion_ids.append(triple.triple_id)
                head = self.entity_by_id[triple.head_id].canonical_label
                relation = self.relation_by_id[triple.relation_id].label
                tail = self.entity_by_id[triple.tail_id].canonical_label
                score = seed_score / (1 + hop)
                rows.append(
                    EvidenceItem(
                        evidence_id=triple.triple_id,
                        evidence_type="triple",
                        content=f"({head}, {relation}, {tail})",
                        score=score,
                        rank=0,
                        provenance_ids=triple.chunk_ids,
                        path_ids=path + (triple.triple_id,),
                        components={"entity_link": seed_score, "hop": float(hop)},
                    )
                )
                next_entity = (
                    triple.tail_id if triple.head_id == entity_id else triple.head_id
                )
                if hop + 1 < self.max_hops:
                    queue.append(
                        (next_entity, hop + 1, path + (triple.triple_id, next_entity), seed_score)
                    )
        rows.sort(key=lambda item: (-item.score, item.evidence_id))
        ranked = tuple(
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
            for rank, item in enumerate(rows, start=1)
        )
        return finalize_result(
            retriever_id=self.retriever_id,
            query=query,
            rows=ranked,
            budget=budget,
            index_fingerprint=self.index_fingerprint,
            latency_ms=(perf_counter() - started) * 1000.0,
            failure=None if ranked else "empty_graph_expansion",
            seed_ids=tuple(item[1] for item in seeds),
            expansion_ids=tuple(expansion_ids),
        )
