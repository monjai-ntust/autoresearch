"""Genuine corpus-statistics BM25 using the frozen bm25s dependency."""

from __future__ import annotations

from time import perf_counter

import bm25s

from graph_rag_eval.budget import Budget
from graph_rag_eval.contracts import Chunk, QueryView
from graph_rag_eval.identity import fingerprint
from graph_rag_eval.retrieval.base import EvidenceItem, RetrievalResult, finalize_result


class BM25Retriever:
    retriever_id = "bm25s-0.3.9"

    def __init__(
        self,
        chunks: tuple[Chunk, ...],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        method: str = "lucene",
        idf_method: str = "lucene",
        token_pattern: str = r"(?u)\b\w+\b",
    ):
        if not chunks:
            raise ValueError("BM25 requires a non-empty corpus")
        self.chunks = tuple(sorted(chunks, key=lambda item: item.chunk_id))
        self.k1 = k1
        self.b = b
        self.method = method
        self.idf_method = idf_method
        self.token_pattern = token_pattern
        corpus_tokens = bm25s.tokenize(
            [item.text for item in self.chunks],
            lower=True,
            token_pattern=token_pattern,
            stopwords=[],
            return_ids=False,
            show_progress=False,
        )
        self._index = bm25s.BM25(
            k1=k1,
            b=b,
            method=method,
            idf_method=idf_method,
        )
        self._index.index(corpus_tokens, show_progress=False)
        self.index_fingerprint = fingerprint(
            "bm25-index-v1",
            {
                "package": "bm25s==0.3.9",
                "k1": k1,
                "b": b,
                "method": method,
                "idf_method": idf_method,
                "token_pattern": token_pattern,
                "corpus": [(item.chunk_id, item.content_sha256) for item in self.chunks],
            },
        )

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult:
        started = perf_counter()
        query_tokens = bm25s.tokenize(
            query.text,
            lower=True,
            token_pattern=self.token_pattern,
            stopwords=[],
            return_ids=False,
            show_progress=False,
        )[0]
        scores = self._index.get_scores(query_tokens)
        ranking = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id),
        )
        rows = tuple(
            EvidenceItem(
                evidence_id=self.chunks[index].chunk_id,
                evidence_type="chunk",
                content=self.chunks[index].text,
                score=float(scores[index]),
                rank=rank,
                provenance_ids=(self.chunks[index].document_id,),
                components={"bm25": float(scores[index])},
            )
            for rank, index in enumerate(ranking, start=1)
            if float(scores[index]) > 0.0
        )
        return finalize_result(
            retriever_id=self.retriever_id,
            query=query,
            rows=rows,
            budget=budget,
            index_fingerprint=self.index_fingerprint,
            latency_ms=(perf_counter() - started) * 1000.0,
            failure=None if rows else "empty_query_match",
        )
