"""Canonical prepared CODE-ACCORD adapter for span-NER and relation training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from utils.pipeline.common.constants import ENTITY_TYPES, RELATION_TYPES


NUM_BIO_TAGS = 1 + 2 * len(ENTITY_TYPES)
NO_REL_ID = 0
REL2ID = {
    "NO_REL": NO_REL_ID,
    **{relation: index + 1 for index, relation in enumerate(RELATION_TYPES)},
}
ID2REL = {index: relation for relation, index in REL2ID.items()}
NUM_RELATIONS = len(REL2ID)
COMPARISON_REL_IDS = [REL2ID[name] for name in ("equal", "greater", "greater-equal", "less", "less-equal")]


class ResumableRandomSampler(Sampler):
    """Shuffle deterministically while retaining the next unconsumed example."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.order = []
        self.cursor = 0

    def __iter__(self):
        if self.cursor >= len(self.order):
            self.order = torch.randperm(
                len(self.data_source), generator=self.generator
            ).tolist()
            self.cursor = 0
        while self.cursor < len(self.order):
            index = self.order[self.cursor]
            self.cursor += 1
            yield index

    def __len__(self):
        return len(self.data_source)

    def state_dict(self):
        return {
            "order": list(self.order),
            "cursor": self.cursor,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {
            "order",
            "cursor",
            "generator_state",
        }:
            raise ValueError("canonical sampler state has an invalid contract")
        order = list(state["order"])
        if sorted(order) != list(range(len(self.data_source))):
            raise ValueError("canonical sampler order does not match the dataset")
        cursor = state["cursor"]
        if not isinstance(cursor, int) or not 0 <= cursor <= len(order):
            raise ValueError("canonical sampler cursor is out of range")
        self.order = order
        self.cursor = cursor
        self.generator.set_state(state["generator_state"])


def _load_prepared_examples(path: Path, expected_split: str) -> list[dict]:
    """Losslessly adapt canonical prepared JSONL into the trainer shape."""

    examples = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} is not valid JSON") from exc
            if record.get("split") != expected_split:
                raise ValueError(
                    f"{path.name}:{line_number} has split {record.get('split')!r}; "
                    f"expected {expected_split!r}"
                )
            example_id = record.get("example_id")
            if not isinstance(example_id, str) or not example_id or example_id in seen:
                raise ValueError(
                    f"{path.name}:{line_number} has a missing or duplicate example_id"
                )
            seen.add(example_id)
            words = record.get("words")
            if not isinstance(words, list) or not words or not all(
                isinstance(word, str) for word in words
            ):
                raise ValueError(f"{path.name}:{line_number} has invalid words")

            entities = []
            entity_spans = set()
            for entity in record.get("entities", []):
                span = (entity.get("start"), entity.get("end"))
                entity_type = entity.get("type")
                if (
                    not all(isinstance(value, int) for value in span)
                    or not 0 <= span[0] <= span[1] < len(words)
                    or entity_type not in ENTITY_TYPES
                ):
                    raise ValueError(
                        f"{path.name}:{line_number} has an invalid entity span"
                    )
                entity_spans.add(span)
                entities.append((span[0], span[1], entity_type))

            relations = []
            for relation in record.get("relations", []):
                relation_type = relation.get("relation")
                head = relation.get("head", {})
                tail = relation.get("tail", {})
                head_span = (head.get("start"), head.get("end"))
                tail_span = (tail.get("start"), tail.get("end"))
                if (
                    relation_type not in REL2ID
                    or head_span not in entity_spans
                    or tail_span not in entity_spans
                ):
                    raise ValueError(
                        f"{path.name}:{line_number} has a relation outside typed spans"
                    )
                relations.append((head_span, tail_span, REL2ID[relation_type]))

            examples.append(
                {
                    "example_id": example_id,
                    "words": words,
                    "ner": entities,
                    "relations": relations,
                }
            )
    if not examples:
        raise ValueError(f"{path.name} contains no prepared examples")
    return examples


class CodeAccordDataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        encoding = self.tokenizer(
            example["words"],
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "word_ids": encoding.word_ids(),
            "gold_entities": example["ner"],
            "gold_relations": example["relations"],
            "num_words": len(example["words"]),
        }


def collate_fn(batch, pad_token_id: int = 0):
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for index, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[index, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[index, :length] = torch.tensor(
            item["attention_mask"], dtype=torch.long
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_ids": [item["word_ids"] for item in batch],
        "gold_entities": [item["gold_entities"] for item in batch],
        "gold_relations": [item["gold_relations"] for item in batch],
        "num_words": [item["num_words"] for item in batch],
    }


def build_dataloaders(
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    seed: int,
    prepared_dir: str | Path,
):
    """Build train/development loaders without ever loading the test split."""

    prepared_dir = Path(prepared_dir)
    train_examples = _load_prepared_examples(prepared_dir / "train.jsonl", "train")
    development_examples = _load_prepared_examples(
        prepared_dir / "development.jsonl", "development"
    )
    pad_token_id = tokenizer.pad_token_id

    def loader(examples, training: bool):
        dataset = CodeAccordDataset(examples, tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=ResumableRandomSampler(dataset, seed) if training else None,
            num_workers=0,
            collate_fn=lambda values: collate_fn(values, pad_token_id=pad_token_id),
        )

    return loader(train_examples, True), loader(development_examples, False), None
