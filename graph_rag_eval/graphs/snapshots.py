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


def structure_summary(graph: GraphSnapshot) -> dict[str, float | int]:
    """Reference-free structural description of one graph snapshot.

    Reported alongside gold-referenced precision/recall so a graph condition can
    still be characterised when no gold graph exists, and so corruption severity
    can be read against measured structure rather than only against the recipe.
    """

    adjacency = graph.adjacency()
    neighbours = {entity.entity_id: set() for entity in graph.entities}
    for triple in graph.triples:
        neighbours[triple.head_id].add(triple.tail_id)
        neighbours[triple.tail_id].add(triple.head_id)
    order = len(graph.entities)
    degrees = {key: len(value) for key, value in neighbours.items()}
    unvisited = set(neighbours)
    components = []
    while unvisited:
        stack = [unvisited.pop()]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for other in neighbours[current]:
                if other in unvisited:
                    unvisited.discard(other)
                    stack.append(other)
        components.append(size)
    clustering = []
    for entity_id, adjacent in neighbours.items():
        degree = len(adjacent)
        if degree < 2:
            clustering.append(0.0)
            continue
        links = sum(
            1
            for index, left in enumerate(sorted(adjacent))
            for right in sorted(adjacent)[index + 1 :]
            if right in neighbours[left]
        )
        clustering.append(2 * links / (degree * (degree - 1)))
    return {
        "entities": order,
        "relations": len(graph.relations),
        "triples": len(graph.triples),
        "distinct_endpoint_pairs": len({frozenset((t.head_id, t.tail_id)) for t in graph.triples}),
        "entities_with_edges": sum(1 for value in adjacency.values() if value),
        "isolated_entities": sum(1 for value in degrees.values() if value == 0),
        "average_degree": (sum(degrees.values()) / order) if order else 0.0,
        "connected_components": len(components),
        "largest_component_entities": max(components) if components else 0,
        "average_clustering_coefficient": (sum(clustering) / order) if order else 0.0,
        "triples_with_provenance": sum(1 for item in graph.triples if item.chunk_ids),
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
