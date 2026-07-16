"""Deterministic sentence-level iterative multilabel split for ``CODE-SPLIT-1``."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Iterable

from .io import DataContractError


SPLIT_ID = "CODE-SPLIT-1"
SPLIT_ALGORITHM_REVISION = "iterative-multilabel-two-fold-1.0"


@dataclass(frozen=True)
class SplitItem:
    example_id: str
    labels: frozenset[str]


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    assignments: dict[str, str]
    label_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class OfficialCodeSplit:
    train_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    assignments: dict[str, str]
    label_counts: dict[str, dict[str, int]]


def iterative_multilabel_split(
    items: Iterable[SplitItem], *, development_size: int, seed: int
) -> SplitResult:
    """Assign two folds using a deterministic Sechidis-style greedy procedure.

    The input is normalized by ascending example ID. At each iteration the
    least frequent remaining label is chosen (lexical tie), then one carrying
    example is chosen by a seed-derived rank followed by ascending ID. The fold
    with the largest remaining quota for that label wins; remaining fold
    capacity and then a seeded choice break fold ties. Every assignment updates
    quotas for all labels carried by the example. Unlabelled leftovers use the
    same capacity/tie rule. This definition is part of CODE-SPLIT-1 and must not
    be replaced by a library-default variant without a protocol revision.
    """

    normalized = sorted(items, key=lambda item: item.example_id)
    if not normalized:
        raise DataContractError("CODE-SPLIT-1 requires at least one sentence")
    ids = [item.example_id for item in normalized]
    if any(not isinstance(example_id, str) or not example_id for example_id in ids):
        raise DataContractError("CODE-SPLIT-1 example IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise DataContractError("CODE-SPLIT-1 example IDs must be unique")
    if isinstance(development_size, bool) or not isinstance(development_size, int):
        raise DataContractError("CODE-SPLIT-1 development_size must be an integer")
    if development_size <= 0 or development_size >= len(normalized):
        raise DataContractError(
            "CODE-SPLIT-1 development_size must leave nonempty train and development folds"
        )
    for item in normalized:
        if not isinstance(item.labels, frozenset) or not item.labels or any(
            not isinstance(label, str) or not label for label in item.labels
        ):
            raise DataContractError(
                f"CODE-SPLIT-1 labels for {item.example_id!r} must be nonempty strings"
            )

    total = len(normalized)
    capacities = {"train": total - development_size, "development": development_size}
    remaining_capacity = dict(capacities)
    labels_by_id = {item.example_id: item.labels for item in normalized}
    remaining_ids = set(ids)
    label_to_ids: dict[str, set[str]] = {}
    for item in normalized:
        for label in item.labels:
            label_to_ids.setdefault(label, set()).add(item.example_id)

    desired: dict[str, dict[str, float]] = {}
    for label, carrying_ids in label_to_ids.items():
        desired[label] = {
            fold: len(carrying_ids) * capacity / total
            for fold, capacity in capacities.items()
        }

    generator = random.Random(seed)
    seeded_rank = {example_id: generator.random() for example_id in ids}
    assignments: dict[str, str] = {}

    def choose_fold(label: str | None) -> str:
        available = [fold for fold in ("train", "development") if remaining_capacity[fold] > 0]
        if not available:
            raise AssertionError("CODE-SPLIT-1 exhausted both folds early")
        if label is not None:
            best_quota = max(desired[label][fold] for fold in available)
            available = [fold for fold in available if desired[label][fold] == best_quota]
        best_capacity = max(remaining_capacity[fold] for fold in available)
        available = [fold for fold in available if remaining_capacity[fold] == best_capacity]
        if len(available) == 1:
            return available[0]
        return available[generator.randrange(len(available))]

    def assign(example_id: str, fold: str) -> None:
        assignments[example_id] = fold
        remaining_ids.remove(example_id)
        remaining_capacity[fold] -= 1
        for label in labels_by_id[example_id]:
            desired[label][fold] -= 1.0
            label_to_ids[label].discard(example_id)

    while remaining_ids:
        active = [
            (len(carrying_ids), label)
            for label, carrying_ids in label_to_ids.items()
            if carrying_ids
        ]
        if not active:
            break
        _, rarest_label = min(active, key=lambda item: (item[0], item[1]))
        example_id = min(
            label_to_ids[rarest_label], key=lambda value: (seeded_rank[value], value)
        )
        assign(example_id, choose_fold(rarest_label))

    for example_id in sorted(remaining_ids, key=lambda value: (seeded_rank[value], value)):
        assign(example_id, choose_fold(None))

    if remaining_capacity != {"train": 0, "development": 0}:
        raise AssertionError(f"CODE-SPLIT-1 capacity invariant failed: {remaining_capacity}")

    train_ids = tuple(sorted(key for key, fold in assignments.items() if fold == "train"))
    development_ids = tuple(
        sorted(key for key, fold in assignments.items() if fold == "development")
    )
    if set(train_ids) & set(development_ids) or set(train_ids) | set(development_ids) != set(ids):
        raise AssertionError("CODE-SPLIT-1 partition invariant failed")

    counts: dict[str, dict[str, int]] = {}
    for label in sorted(label_to_ids):
        counts[label] = {
            "all": sum(label in labels_by_id[example_id] for example_id in ids),
            "train": sum(label in labels_by_id[example_id] for example_id in train_ids),
            "development": sum(
                label in labels_by_id[example_id] for example_id in development_ids
            ),
        }
    return SplitResult(
        train_ids=train_ids,
        development_ids=development_ids,
        assignments=dict(sorted(assignments.items())),
        label_counts=counts,
    )


def split_manifest(result: SplitResult, *, seed: int) -> dict:
    return {
        "split_id": SPLIT_ID,
        "algorithm_revision": SPLIT_ALGORITHM_REVISION,
        "seed": seed,
        "train_count": len(result.train_ids),
        "development_count": len(result.development_ids),
        "train_ids": list(result.train_ids),
        "development_ids": list(result.development_ids),
        "label_counts": result.label_counts,
    }


def build_official_code_split(
    entity_train_items: Iterable[SplitItem], entity_test_ids: Iterable[str]
) -> OfficialCodeSplit:
    """Apply the fixed 586/103/173 protocol boundary with split seed 42."""

    train_items = list(entity_train_items)
    test_ids = tuple(sorted(entity_test_ids))
    if len(train_items) != 689 or len(test_ids) != 173:
        raise DataContractError(
            "official CODE-SPLIT-1 requires 689 entity-train and 173 entity-test IDs"
        )
    all_ids = [item.example_id for item in train_items] + list(test_ids)
    if len(set(all_ids)) != 862:
        raise DataContractError(
            "official CODE-SPLIT-1 IDs must be unique and disjoint across the entity boundary"
        )
    for example_id in all_ids:
        try:
            parsed = uuid.UUID(example_id)
        except (ValueError, AttributeError) as exc:
            raise DataContractError(
                f"official CODE-SPLIT-1 example ID is not a UUID: {example_id!r}"
            ) from exc
        if str(parsed) != example_id:
            raise DataContractError(
                f"official CODE-SPLIT-1 UUID is not canonical: {example_id!r}"
            )
    result = iterative_multilabel_split(train_items, development_size=103, seed=42)
    require_disjoint_partitions(
        train=result.train_ids,
        development=result.development_ids,
        test=test_ids,
    )
    return OfficialCodeSplit(
        train_ids=result.train_ids,
        development_ids=result.development_ids,
        test_ids=test_ids,
        assignments=result.assignments,
        label_counts=result.label_counts,
    )


def official_split_manifest(result: OfficialCodeSplit) -> dict:
    return {
        "split_id": SPLIT_ID,
        "algorithm_revision": SPLIT_ALGORITHM_REVISION,
        "seed": 42,
        "train_count": len(result.train_ids),
        "development_count": len(result.development_ids),
        "test_count": len(result.test_ids),
        "train_ids": list(result.train_ids),
        "development_ids": list(result.development_ids),
        "test_ids": list(result.test_ids),
        "label_counts": result.label_counts,
        "overlap_count": 0,
    }


def require_disjoint_partitions(**partitions: Iterable[str]) -> None:
    """Fail on any sentence leakage across named train/dev/test partitions."""

    owners: dict[str, str] = {}
    for partition, values in partitions.items():
        for example_id in values:
            if not isinstance(example_id, str) or not example_id:
                raise DataContractError(f"partition {partition!r} has an invalid example ID")
            previous = owners.get(example_id)
            if previous is not None:
                raise DataContractError(
                    f"sentence {example_id!r} leaks across partitions {previous!r} and {partition!r}"
                )
            owners[example_id] = partition
