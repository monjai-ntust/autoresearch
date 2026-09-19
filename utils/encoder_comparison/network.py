"""Bounded historical span/relation network for Phase G comparisons.

The active implementation is transplanted from ``models/bert_kg_encoder.py``
blob ``d05678f00097cb11176dacef6fade0b1b3f557de`` at historical source commit
``9feafa4029e65ab48ecfb2f452f4b0fabbff0826``. Inactive experimental branches
were omitted. Immutable model loading and the explicit output dimensions are
adapters required by the current run contract; they do not change the active
historical text/span/relation computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


HISTORICAL_SOURCE_COMMIT = "9feafa4029e65ab48ecfb2f452f4b0fabbff0826"
HISTORICAL_NETWORK_BLOB = "d05678f00097cb11176dacef6fade0b1b3f557de"


class TextAdapter(nn.Module):
    """Historical shared-word-embedding text adapter."""

    def __init__(self, word_embeddings: nn.Embedding):
        super().__init__()
        self.word_embeddings = word_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return {
            "inputs_embeds": self.word_embeddings(input_ids),
            "attention_mask": attention_mask,
        }


class BertBackbone(nn.Module):
    """Historical AutoModel wrapper with immutable run-local loading controls."""

    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str,
        model_cache_dir: str,
        model_local_files_only: bool,
        expected_hidden_size: int,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_name,
            revision=model_revision,
            cache_dir=model_cache_dir,
            local_files_only=model_local_files_only,
            use_safetensors=False,
        )
        if self.bert.config.hidden_size != expected_hidden_size:
            raise ValueError(
                "loaded encoder hidden size differs from its frozen comparison profile"
            )

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        return self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state


class HistoricalKGExtractor(nn.Module):
    """Historical span-NER and ordered-pair relation extractor."""

    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str,
        model_cache_dir: str,
        model_local_files_only: bool,
        expected_hidden_size: int,
        num_bio_tags: int,
        num_relations: int,
        num_entity_types: int,
        max_span_width: int,
        context_between_spans: bool,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = BertBackbone(
            model_name,
            model_revision=model_revision,
            model_cache_dir=model_cache_dir,
            model_local_files_only=model_local_files_only,
            expected_hidden_size=expected_hidden_size,
        )
        hidden = self.backbone.hidden_size

        # Preserve historical construction order and registered names. The
        # original 2H relation head is constructed before an optional A12 3H
        # replacement, which preserves the seeded initialization sequence.
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
        if context_between_spans:
            self.re_context_span = True
            self.re_head = nn.Sequential(
                nn.Linear(hidden * 3, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_relations),
            )

    def encode(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        adapted = self.adapters["text"](
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.backbone(**adapted)

    def forward_ner(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ner_head(self.dropout(hidden_states))

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
        max_span_width: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
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
