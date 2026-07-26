"""Immutable graph snapshots with reconstructible identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from graph_rag_eval.contracts import Entity, Relation, Triple, content_sha256


@dataclass(frozen=True)
class GraphSnapshot:
    dataset_id: str
    graph_id: str
    condition: str
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    triples: tuple[Triple, ...]
    descriptor_sha256: str
    corpus_sha256: str
    adapter_sha256: str
    extractor_sha256: str
    entity_sha256: str
    relation_sha256: str
    triple_sha256: str
    construction_recipe: str
    parent_graph_id: str | None = None
    corruption_recipe: dict | None = None
    schema_version: str = "rag-graph-snapshot-1.0"
    construction_time: str = "deterministic"

    def __post_init__(self) -> None:
        entity_ids = {item.entity_id for item in self.entities}
        relation_ids = {item.relation_id for item in self.relations}
        if len(entity_ids) != len(self.entities) or len(relation_ids) != len(self.relations):
            raise ValueError("graph snapshot contains duplicate entities or relations")
        if any(
            item.head_id not in entity_ids
            or item.tail_id not in entity_ids
            or item.relation_id not in relation_ids
            for item in self.triples
        ):
            raise ValueError("graph snapshot contains a foreign-key triple")
        if self.entity_sha256 != content_sha256(self.entities):
            raise ValueError("graph entity hash mismatch")
        if self.relation_sha256 != content_sha256(self.relations):
            raise ValueError("graph relation hash mismatch")
        if self.triple_sha256 != content_sha256(self.triples):
            raise ValueError("graph triple hash mismatch")
        if self.corruption_recipe is not None and self.parent_graph_id is None:
            raise ValueError("corrupted graph requires a parent graph identity")

    @property
    def counts(self) -> dict[str, int]:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "triples": len(self.triples),
        }

    def adjacency(self) -> dict[str, tuple[Triple, ...]]:
        result: dict[str, list[Triple]] = {item.entity_id: [] for item in self.entities}
        for triple in self.triples:
            result[triple.head_id].append(triple)
            result[triple.tail_id].append(triple)
        return {
            key: tuple(sorted(value, key=lambda item: item.triple_id))
            for key, value in sorted(result.items())
        }


def build_snapshot(
    *,
    dataset_id: str,
    condition: str,
    entities: Iterable[Entity],
    relations: Iterable[Relation],
    triples: Iterable[Triple],
    descriptor_sha256: str,
    corpus_sha256: str,
    adapter_sha256: str,
    extractor_sha256: str,
    construction_recipe: str,
    parent_graph_id: str | None = None,
    corruption_recipe: dict | None = None,
) -> GraphSnapshot:
    entity_rows = tuple(sorted(entities, key=lambda item: item.entity_id))
    relation_rows = tuple(sorted(relations, key=lambda item: item.relation_id))
    triple_rows = tuple(sorted(triples, key=lambda item: item.triple_id))
    identity_payload = {
        "dataset_id": dataset_id,
        "condition": condition,
        "descriptor_sha256": descriptor_sha256,
        "corpus_sha256": corpus_sha256,
        "adapter_sha256": adapter_sha256,
        "extractor_sha256": extractor_sha256,
        "entity_sha256": content_sha256(entity_rows),
        "relation_sha256": content_sha256(relation_rows),
        "triple_sha256": content_sha256(triple_rows),
        "construction_recipe": construction_recipe,
        "parent_graph_id": parent_graph_id,
        "corruption_recipe": corruption_recipe,
    }
    graph_id = content_sha256(identity_payload)
    return GraphSnapshot(
        dataset_id=dataset_id,
        graph_id=graph_id,
        condition=condition,
        entities=entity_rows,
        relations=relation_rows,
        triples=triple_rows,
        descriptor_sha256=descriptor_sha256,
        corpus_sha256=corpus_sha256,
        adapter_sha256=adapter_sha256,
        extractor_sha256=extractor_sha256,
        entity_sha256=content_sha256(entity_rows),
        relation_sha256=content_sha256(relation_rows),
        triple_sha256=content_sha256(triple_rows),
        construction_recipe=construction_recipe,
        parent_graph_id=parent_graph_id,
        corruption_recipe=corruption_recipe,
    )
