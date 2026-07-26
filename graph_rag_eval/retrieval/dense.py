"""Frozen dense interface plus deterministic offline fixture implementation."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from time import perf_counter
import re

from graph_rag_eval.budget import Budget
from graph_rag_eval.contracts import Chunk, QueryView
from graph_rag_eval.identity import fingerprint
from graph_rag_eval.retrieval.base import EvidenceItem, RetrievalResult, finalize_result


def _vector(text: str, dimensions: int = 256) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for token in re.findall(r"\w+", text.casefold()):
        index = int.from_bytes(token.encode("utf-8"), "little", signed=False) % dimensions
        counts[index] += 1
    norm = sqrt(sum(value * value for value in counts.values()))
    return {key: value / norm for key, value in counts.items()} if norm else {}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class HashingDenseRetriever:
    """No-model deterministic dense-shaped control for synthetic validation only."""

    retriever_id = "hashing-dense-fixture-v1"

    def __init__(self, chunks: tuple[Chunk, ...], *, dimensions: int = 256):
        self.chunks = tuple(sorted(chunks, key=lambda item: item.chunk_id))
        self.dimensions = dimensions
        self.vectors = tuple(_vector(item.text, dimensions) for item in self.chunks)
        self.index_fingerprint = fingerprint(
            "hashing-dense-index-v1",
            {
                "dimensions": dimensions,
                "corpus": [(item.chunk_id, item.content_sha256) for item in self.chunks],
            },
        )

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult:
        started = perf_counter()
        query_vector = _vector(query.text, self.dimensions)
        scores = [_cosine(query_vector, item) for item in self.vectors]
        ranking = sorted(
            range(len(self.chunks)),
            key=lambda index: (-scores[index], self.chunks[index].chunk_id),
        )
        rows = tuple(
            EvidenceItem(
                evidence_id=self.chunks[index].chunk_id,
                evidence_type="chunk",
                content=self.chunks[index].text,
                score=scores[index],
                rank=rank,
                provenance_ids=(self.chunks[index].document_id,),
                components={"hashing_cosine": scores[index]},
            )
            for rank, index in enumerate(ranking, start=1)
            if scores[index] > 0.0
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


class FrozenTransformerDenseRetriever:
    """Pinned local-cache Transformer encoder for externally approved runs."""

    retriever_id = "frozen-transformer-dense-v1"

    def __init__(
        self,
        chunks: tuple[Chunk, ...],
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str,
        pooling: str = "mean",
        normalize: bool = True,
        device: str = "cpu",
        local_files_only: bool = True,
        max_length: int = 512,
    ):
        if not revision or revision in {"main", "latest"}:
            raise ValueError("dense model requires an immutable revision")
        if pooling not in {"mean", "cls"}:
            raise ValueError("dense pooling must be mean or cls")
        if not chunks:
            raise ValueError("dense retrieval requires a non-empty corpus")
        self.chunks = tuple(sorted(chunks, key=lambda item: item.chunk_id))
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.pooling = pooling
        self.normalize = normalize
        self.device = device
        self.local_files_only = local_files_only
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._corpus_vectors = None
        self.index_fingerprint = fingerprint(
            "frozen-transformer-dense-index-v1",
            {
                "model_id": model_id,
                "revision": revision,
                "pooling": pooling,
                "normalize": normalize,
                "max_length": max_length,
                "corpus": [(item.chunk_id, item.content_sha256) for item in self.chunks],
            },
        )

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self._model.eval()
        self._corpus_vectors = self._encode([item.text for item in self.chunks], torch)

    def _encode(self, texts, torch_module):
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch_module.inference_mode():
            hidden = self._model(**encoded).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if self.normalize:
            pooled = torch_module.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu()

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult:
        started = perf_counter()
        self._load()
        import torch

        query_vector = self._encode([query.text], torch)[0]
        scores = self._corpus_vectors @ query_vector
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
                components={"dense_cosine": float(scores[index])},
            )
            for rank, index in enumerate(ranking, start=1)
        )
        return finalize_result(
            retriever_id=self.retriever_id,
            query=query,
            rows=rows,
            budget=budget,
            index_fingerprint=self.index_fingerprint,
            latency_ms=(perf_counter() - started) * 1000.0,
        )
