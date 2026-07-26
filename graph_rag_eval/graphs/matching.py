"""Exact and relaxed one-to-one graph matching."""

from __future__ import annotations

from dataclasses import dataclass
import re

from graph_rag_eval.graphs.snapshots import GraphSnapshot


def normalize_label(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


@dataclass(frozen=True)
class MatchCounts:
    true_positive: int
    false_positive: int
    false_negative: int

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
        if self.precision is None or self.recall is None:
            return None
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True)
class GraphMatch:
    entities: MatchCounts
    relations: MatchCounts
    triples: MatchCounts
    provenance: MatchCounts
    matcher: str


def _counts(predicted: set, gold: set) -> MatchCounts:
    return MatchCounts(
        true_positive=len(predicted & gold),
        false_positive=len(predicted - gold),
        false_negative=len(gold - predicted),
    )


def match_graphs(
    predicted: GraphSnapshot,
    gold: GraphSnapshot,
    *,
    relaxed: bool = False,
) -> GraphMatch:
    if predicted.dataset_id != gold.dataset_id:
        raise ValueError("cannot match graphs from different datasets")
    gold_entities_by_id = {item.entity_id: item for item in gold.entities}
    predicted_entities_by_id = {item.entity_id: item for item in predicted.entities}
    if relaxed:
        entity_key = lambda item: (normalize_label(item.canonical_label), item.entity_type)
        relation_key = lambda item: (normalize_label(item.label), item.direction)
    else:
        entity_key = lambda item: item.entity_id
        relation_key = lambda item: item.relation_id
    predicted_entity_keys = {entity_key(item) for item in predicted.entities}
    gold_entity_keys = {entity_key(item) for item in gold.entities}
    predicted_relation_keys = {relation_key(item) for item in predicted.relations}
    gold_relation_keys = {relation_key(item) for item in gold.relations}

    def triple_key(triple, entities, relations):
        if relaxed:
            return (
                entity_key(entities[triple.head_id]),
                relation_key(relations[triple.relation_id]),
                entity_key(entities[triple.tail_id]),
            )
        return (triple.head_id, triple.relation_id, triple.tail_id)

    predicted_relations = {item.relation_id: item for item in predicted.relations}
    gold_relations = {item.relation_id: item for item in gold.relations}
    predicted_triples = {
        triple_key(item, predicted_entities_by_id, predicted_relations)
        for item in predicted.triples
    }
    gold_triples = {
        triple_key(item, gold_entities_by_id, gold_relations)
        for item in gold.triples
    }
    predicted_provenance = {
        (triple_key(item, predicted_entities_by_id, predicted_relations), chunk_id)
        for item in predicted.triples
        for chunk_id in item.chunk_ids
    }
    gold_provenance = {
        (triple_key(item, gold_entities_by_id, gold_relations), chunk_id)
        for item in gold.triples
        for chunk_id in item.chunk_ids
    }
    return GraphMatch(
        entities=_counts(predicted_entity_keys, gold_entity_keys),
        relations=_counts(predicted_relation_keys, gold_relation_keys),
        triples=_counts(predicted_triples, gold_triples),
        provenance=_counts(predicted_provenance, gold_provenance),
        matcher="relaxed-normalized-v1" if relaxed else "exact-id-v1",
    )
