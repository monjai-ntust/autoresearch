"""Canonical text encoder and span/relation heads for CODE-ACCORD."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextAdapter(nn.Module):
    """Expose the backbone's shared word embeddings as ``inputs_embeds``."""

    def __init__(self, word_embeddings: nn.Embedding):
        super().__init__()
        # Completed checkpoints contain this historical registered alias.
        self.word_embeddings = word_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return {
            "inputs_embeds": self.word_embeddings(input_ids),
            "attention_mask": attention_mask,
        }


class BertBackbone(nn.Module):
    """Load the pinned Hugging Face encoder and return hidden states."""

    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        model_cache_dir: str | None = None,
        model_local_files_only: bool = False,
    ):
        super().__init__()
        kwargs = {"revision": model_revision} if model_revision else {}
        if model_cache_dir:
            kwargs["cache_dir"] = model_cache_dir
        if model_local_files_only:
            kwargs["local_files_only"] = True
        self.bert = AutoModel.from_pretrained(model_name, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        return self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state


class BertKGExtractor(nn.Module):
    """The span-NER and context-enriched RE architecture used by Table 1."""

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        *,
        num_bio_tags: int,
        num_relations: int,
        num_entity_types: int,
        max_span_width: int = 8,
        model_revision: str | None = None,
        model_cache_dir: str | None = None,
        model_local_files_only: bool = False,
    ):
        super().__init__()
        self.backbone = BertBackbone(
            model_name,
            model_revision=model_revision,
            model_cache_dir=model_cache_dir,
            model_local_files_only=model_local_files_only,
        )
        hidden = self.backbone.hidden_size

        # Preserve construction order and registered names so both seeded
        # initialization and completed checkpoint keyspace remain unchanged.
        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(hidden, num_bio_tags)
        self.max_span_width = max_span_width
        span_width = hidden * 3
        self.span_ner_head = nn.Linear(span_width, num_entity_types + 1)
        self.span_width_emb = nn.Embedding(max_span_width, hidden)
        self.span_width_proj = nn.Linear(hidden, span_width)

        self.re_context_span = False
        self.re_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_relations),
        )
        self.adapters = nn.ModuleDict(
            {"text": TextAdapter(self.backbone.bert.get_input_embeddings())}
        )

    def encode(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        adapted = self.adapters["text"](
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.backbone(**adapted)

    @staticmethod
    def _first_token_for_word(word_ids: list, word_idx: int) -> int:
        for index, candidate in enumerate(word_ids):
            if candidate == word_idx:
                return index
        return 0

    @staticmethod
    def _last_token_for_word(word_ids: list, word_idx: int) -> int:
        last = 0
        for index, candidate in enumerate(word_ids):
            if candidate == word_idx:
                last = index
        return last

    def span_repr(
        self,
        hidden_states: torch.Tensor,
        word_ids: list,
        span: tuple[int, int],
    ) -> torch.Tensor:
        """Max-pool subword states belonging to an inclusive word span."""

        start, end = span
        token_indices = [
            index
            for index, word_id in enumerate(word_ids)
            if word_id is not None and start <= word_id <= end
        ]
        if not token_indices:
            return hidden_states.new_zeros(hidden_states.size(-1))
        return hidden_states[token_indices].max(dim=0).values

    def _span_ner_repr(
        self,
        hidden_states: torch.Tensor,
        word_ids: list,
        span: tuple[int, int],
    ) -> torch.Tensor:
        start, end = span
        return torch.cat(
            [
                hidden_states[self._first_token_for_word(word_ids, start)],
                hidden_states[self._last_token_for_word(word_ids, end)],
                self.span_repr(hidden_states, word_ids, span),
            ],
            dim=-1,
        )

    def forward_span_ner(
        self,
        hidden_states: torch.Tensor,
        word_ids: list,
        num_words: int,
        max_span_width: int = 8,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Enumerate and classify every span up to ``max_span_width``."""

        candidates = [
            (start, end)
            for start in range(num_words)
            for end in range(start, min(start + max_span_width, num_words))
        ]
        if not candidates:
            return hidden_states.new_zeros((0, self.span_ner_head.out_features)), []

        span_vectors = torch.stack(
            [self._span_ner_repr(hidden_states, word_ids, span) for span in candidates]
        )
        widths = torch.tensor(
            [
                min(end - start, self.span_width_emb.num_embeddings - 1)
                for start, end in candidates
            ],
            device=hidden_states.device,
            dtype=torch.long,
        )
        span_vectors = span_vectors + self.span_width_proj(self.span_width_emb(widths))
        return self.span_ner_head(self.dropout(span_vectors)), candidates

    @staticmethod
    def _between_span_repr(
        hidden_states: torch.Tensor,
        word_ids: list,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> torch.Tensor:
        low = min(first[1], second[1]) + 1
        high = max(first[0], second[0]) - 1
        if low > high:
            return hidden_states.new_zeros(hidden_states.size(-1))
        token_indices = [
            index
            for index, word_id in enumerate(word_ids)
            if word_id is not None and low <= word_id <= high
        ]
        if not token_indices:
            return hidden_states.new_zeros(hidden_states.size(-1))
        return hidden_states[token_indices].mean(dim=0)

    def forward_re(
        self,
        hidden_states: torch.Tensor,
        word_ids: list,
        pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> torch.Tensor:
        """Classify ordered entity pairs using span and inter-span context."""

        if not pairs:
            return hidden_states.new_zeros((0, self.re_head[-1].out_features))
        features = []
        for head, tail in pairs:
            parts = [
                self.span_repr(hidden_states, word_ids, head),
                self.span_repr(hidden_states, word_ids, tail),
            ]
            if self.re_context_span:
                parts.append(
                    self._between_span_repr(hidden_states, word_ids, head, tail)
                )
            features.append(torch.cat(parts, dim=-1))
        return self.re_head(self.dropout(torch.stack(features)))
