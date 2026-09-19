"""Comparison-only BIO adapter over the unchanged canonical data utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.encoder import data as canonical


ENTITY_TYPES = canonical.ENTITY_TYPES
NUM_BIO_TAGS = canonical.NUM_BIO_TAGS
NO_REL_ID = canonical.NO_REL_ID
REL2ID = canonical.REL2ID
ID2REL = canonical.ID2REL
NUM_RELATIONS = canonical.NUM_RELATIONS
COMPARISON_REL_IDS = canonical.COMPARISON_REL_IDS
ResumableRandomSampler = canonical.ResumableRandomSampler

BIO_TAGS = ("O",) + tuple(
    f"{prefix}-{entity_type}"
    for entity_type in ENTITY_TYPES
    for prefix in ("B", "I")
)
BIO_TAG2ID = {tag: index for index, tag in enumerate(BIO_TAGS)}


def _bio_tags_for_sentence(
    num_words: int, entities: list[tuple[int, int, str]]
) -> list[int]:
    """Build the historical first-subword BIO auxiliary labels."""

    labels = [BIO_TAG2ID["O"]] * num_words
    for start, end, entity_type in entities:
        labels[start] = BIO_TAG2ID[f"B-{entity_type}"]
        for index in range(start + 1, end + 1):
            labels[index] = BIO_TAG2ID[f"I-{entity_type}"]
    return labels


class HistoricalCodeAccordDataset(canonical.CodeAccordDataset):
    """Add only the historical trainer fields to a canonical dataset item."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        example = self.examples[index]
        word_bio = _bio_tags_for_sentence(len(example["words"]), example["ner"])
        token_labels = []
        previous_word_id = None
        for word_id in item["word_ids"]:
            if word_id is None or word_id == previous_word_id:
                token_labels.append(-100)
            else:
                token_labels.append(word_bio[word_id])
            previous_word_id = word_id
        return {
            **item,
            "ner_labels": token_labels,
            "words": example["words"],
            "example_id": example["example_id"],
        }


def collate_fn(batch, pad_token_id: int = 0):
    """Retain the canonical batch exactly and append historical-only fields."""

    result = canonical.collate_fn(batch, pad_token_id=pad_token_id)
    max_length = result["input_ids"].size(1)
    ner_labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for index, item in enumerate(batch):
        length = len(item["ner_labels"])
        ner_labels[index, :length] = torch.tensor(
            item["ner_labels"], dtype=torch.long
        )
    return {
        **result,
        "ner_labels": ner_labels,
        "words": [item["words"] for item in batch],
        "example_ids": [item["example_id"] for item in batch],
    }


def build_dataloaders(
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    seed: int,
    prepared_dir: str | Path,
):
    """Reuse canonical loading/sampling while adapting only comparison batches."""

    prepared_dir = Path(prepared_dir)
    train_examples = canonical._load_prepared_examples(
        prepared_dir / "train.jsonl", "train"
    )
    development_examples = canonical._load_prepared_examples(
        prepared_dir / "development.jsonl", "development"
    )
    pad_token_id = tokenizer.pad_token_id

    def loader(examples, training: bool):
        dataset = HistoricalCodeAccordDataset(examples, tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=ResumableRandomSampler(dataset, seed) if training else None,
            num_workers=0,
            collate_fn=lambda values: collate_fn(
                values, pad_token_id=pad_token_id
            ),
        )

    return loader(train_examples, True), loader(development_examples, False), None
