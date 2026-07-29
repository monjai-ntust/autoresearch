"""Exact mention and end-to-end typed-relation extraction metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from graph_rag_eval.contracts import Entity, Relation, Triple
from graph_rag_eval.graphs.snapshots import GraphSnapshot


@dataclass(frozen=True)
class ExtractionCounts:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def support(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float | None:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        precision = self.precision
        recall = self.recall
        if precision is None or recall is None:
            return None
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator else 0.0


def _counts(predicted: set[Any], gold: set[Any]) -> ExtractionCounts:
    return ExtractionCounts(
        true_positive=len(predicted & gold),
        false_positive=len(predicted - gold),
        false_negative=len(gold - predicted),
    )


def _counts_data(counts: ExtractionCounts) -> dict[str, int | float | None]:
    return {
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "false_negative": counts.false_negative,
        "support": counts.support,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
    }


def _entity_key(entity: Entity) -> tuple[Any, ...]:
    """Exact mention key independent of a producer's opaque entity ID."""

    return (
        entity.dataset_id,
        tuple(entity.source_spans),
        entity.entity_type,
    )


def _triple_key(
    triple: Triple,
    entities: dict[str, Entity],
    relations: dict[str, Relation],
) -> tuple[Any, ...]:
    relation = relations[triple.relation_id]
    head = _entity_key(entities[triple.head_id])
    tail = _entity_key(entities[triple.tail_id])
    if relation.direction == "undirected" and repr(tail) < repr(head):
        head, tail = tail, head
    return (head, relation.label, relation.direction, tail)


def _by_label(
    labels: Iterable[str],
    predicted: set[Any],
    gold: set[Any],
    label_of: Callable[[Any], str],
) -> dict[str, ExtractionCounts]:
    return {
        label: _counts(
            {item for item in predicted if label_of(item) == label},
            {item for item in gold if label_of(item) == label},
        )
        for label in sorted(set(labels))
    }


def evaluate_extraction(
    predicted: GraphSnapshot,
    gold: GraphSnapshot,
) -> dict[str, Any]:
    """Return exact entity and end-to-end typed-relation TP/FP/FN and P/R/F1.

    Entities match by dataset, source span, and type. Relations match only when
    both endpoint mentions and the directed relation label are correct. This is
    deliberately separate from relation classification on gold spans.
    """

    if predicted.dataset_id != gold.dataset_id:
        raise ValueError("cannot score extraction across different datasets")

    predicted_entities = {_entity_key(item) for item in predicted.entities}
    gold_entities = {_entity_key(item) for item in gold.entities}
    predicted_entity_labels = {
        _entity_key(item): item.entity_type or "UNSPECIFIED"
        for item in predicted.entities
    }
    gold_entity_labels = {
        _entity_key(item): item.entity_type or "UNSPECIFIED"
        for item in gold.entities
    }

    predicted_entities_by_id = {item.entity_id: item for item in predicted.entities}
    gold_entities_by_id = {item.entity_id: item for item in gold.entities}
    predicted_relations_by_id = {item.relation_id: item for item in predicted.relations}
    gold_relations_by_id = {item.relation_id: item for item in gold.relations}
    predicted_triples = {
        _triple_key(item, predicted_entities_by_id, predicted_relations_by_id)
        for item in predicted.triples
    }
    gold_triples = {
        _triple_key(item, gold_entities_by_id, gold_relations_by_id)
        for item in gold.triples
    }

    entity_labels = set(predicted_entity_labels.values()) | set(gold_entity_labels.values())
    all_entity_labels = {**gold_entity_labels, **predicted_entity_labels}
    relation_labels = {item[1] for item in predicted_triples | gold_triples}
    entity_by_label = _by_label(
        entity_labels,
        predicted_entities,
        gold_entities,
        lambda item: all_entity_labels[item],
    )
    relation_by_label = _by_label(
        relation_labels,
        predicted_triples,
        gold_triples,
        lambda item: item[1],
    )
    return {
        "matcher": "exact-source-span-and-type/end-to-end-directed-relation-v1",
        "entity": {
            "micro": _counts_data(_counts(predicted_entities, gold_entities)),
            "per_label": {
                label: _counts_data(counts)
                for label, counts in entity_by_label.items()
            },
        },
        "end_to_end_relation": {
            "micro": _counts_data(_counts(predicted_triples, gold_triples)),
            "per_label": {
                label: _counts_data(counts)
                for label, counts in relation_by_label.items()
            },
        },
    }
