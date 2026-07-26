"""Deterministic graph-quality controls with parent/recipe identities."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import random

from graph_rag_eval.contracts import Entity, Provenance, Triple, content_sha256
from graph_rag_eval.graphs.snapshots import GraphSnapshot, build_snapshot


def _rng(seed: int, name: str) -> random.Random:
    digest = sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _derive(
    graph: GraphSnapshot,
    name: str,
    *,
    entities=None,
    relations=None,
    triples=None,
    seed: int,
    severity: float,
    details: dict | None = None,
) -> GraphSnapshot:
    recipe = {
        "name": name,
        "seed": seed,
        "severity": severity,
        "details": details or {},
    }
    return build_snapshot(
        dataset_id=graph.dataset_id,
        condition=f"{graph.condition}:{name}",
        entities=graph.entities if entities is None else entities,
        relations=graph.relations if relations is None else relations,
        triples=graph.triples if triples is None else triples,
        descriptor_sha256=graph.descriptor_sha256,
        corpus_sha256=graph.corpus_sha256,
        adapter_sha256=graph.adapter_sha256,
        extractor_sha256=graph.extractor_sha256,
        construction_recipe=f"deterministic-corruption:{name}",
        parent_graph_id=graph.graph_id,
        corruption_recipe=recipe,
    )


def drop_edges(graph: GraphSnapshot, severity: float, seed: int) -> GraphSnapshot:
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    rows = list(graph.triples)
    count = min(len(rows), round(len(rows) * severity))
    selected = set(_rng(seed, "drop-edges").sample(range(len(rows)), count))
    kept = tuple(row for index, row in enumerate(rows) if index not in selected)
    return _derive(
        graph,
        "edge-drop",
        triples=kept,
        seed=seed,
        severity=severity,
        details={"dropped_triple_ids": sorted(rows[index].triple_id for index in selected)},
    )


def add_edges(graph: GraphSnapshot, severity: float, seed: int) -> GraphSnapshot:
    if len(graph.entities) < 2 or not graph.relations:
        raise ValueError("edge addition requires entities and relations")
    count = max(1, round(max(1, len(graph.triples)) * severity))
    existing = {(row.head_id, row.relation_id, row.tail_id) for row in graph.triples}
    candidates = [
        (head.entity_id, relation.relation_id, tail.entity_id)
        for head in graph.entities
        for relation in graph.relations
        for tail in graph.entities
        if head.entity_id != tail.entity_id
        and (head.entity_id, relation.relation_id, tail.entity_id) not in existing
    ]
    rng = _rng(seed, "add-edges")
    picked = rng.sample(candidates, min(count, len(candidates)))
    additions = []
    for index, (head_id, relation_id, tail_id) in enumerate(picked):
        triple_id = f"corrupt-add-{seed}-{index:04d}"
        additions.append(
            Triple(
                dataset_id=graph.dataset_id,
                triple_id=triple_id,
                head_id=head_id,
                relation_id=relation_id,
                tail_id=tail_id,
                chunk_ids=(),
                confidence=0.0,
                provenance=Provenance(
                    dataset_id=graph.dataset_id,
                    snapshot_id=graph.graph_id,
                    stage="corruption-edge-add",
                    producer_version="1.0.0",
                    content_sha256=sha256(
                        f"{triple_id}:{head_id}:{relation_id}:{tail_id}".encode("utf-8")
                    ).hexdigest(),
                    parent_ids=(graph.graph_id,),
                ),
            )
        )
    return _derive(
        graph,
        "edge-add",
        triples=tuple(graph.triples) + tuple(additions),
        seed=seed,
        severity=severity,
        details={"added_triple_ids": [item.triple_id for item in additions]},
    )


def relabel_relations(graph: GraphSnapshot, severity: float, seed: int) -> GraphSnapshot:
    if len(graph.relations) < 2:
        raise ValueError("relation relabeling requires at least two relations")
    count = min(len(graph.triples), round(len(graph.triples) * severity))
    indexes = set(_rng(seed, "relation-relabel").sample(range(len(graph.triples)), count))
    relation_ids = [item.relation_id for item in graph.relations]
    changed = []
    for index, triple in enumerate(graph.triples):
        if index not in indexes:
            changed.append(triple)
            continue
        current = relation_ids.index(triple.relation_id)
        replacement = relation_ids[(current + 1 + seed) % len(relation_ids)]
        if replacement == triple.relation_id:
            replacement = relation_ids[(current + 1) % len(relation_ids)]
        changed.append(replace(triple, relation_id=replacement))
    return _derive(
        graph,
        "relation-relabel",
        triples=tuple(changed),
        seed=seed,
        severity=severity,
        details={"changed_triple_ids": sorted(graph.triples[i].triple_id for i in indexes)},
    )


def rewire_endpoints(graph: GraphSnapshot, severity: float, seed: int) -> GraphSnapshot:
    if len(graph.entities) < 3:
        raise ValueError("endpoint rewiring requires at least three entities")
    count = min(len(graph.triples), round(len(graph.triples) * severity))
    indexes = set(_rng(seed, "endpoint-rewire").sample(range(len(graph.triples)), count))
    entity_ids = [item.entity_id for item in graph.entities]
    changed = []
    for index, triple in enumerate(graph.triples):
        if index not in indexes:
            changed.append(triple)
            continue
        start = (entity_ids.index(triple.tail_id) + 1 + seed) % len(entity_ids)
        candidates = entity_ids[start:] + entity_ids[:start]
        replacement = next(
            entity_id
            for entity_id in candidates
            if entity_id not in {triple.head_id, triple.tail_id}
        )
        changed.append(replace(triple, tail_id=replacement))
    return _derive(
        graph,
        "endpoint-rewire",
        triples=tuple(changed),
        seed=seed,
        severity=severity,
        details={"changed_triple_ids": sorted(graph.triples[i].triple_id for i in indexes)},
    )


def remove_provenance(graph: GraphSnapshot, severity: float, seed: int) -> GraphSnapshot:
    count = min(len(graph.triples), round(len(graph.triples) * severity))
    indexes = set(_rng(seed, "provenance-remove").sample(range(len(graph.triples)), count))
    changed = tuple(
        replace(triple, chunk_ids=()) if index in indexes else triple
        for index, triple in enumerate(graph.triples)
    )
    return _derive(
        graph,
        "provenance-remove",
        triples=changed,
        seed=seed,
        severity=severity,
        details={"changed_triple_ids": sorted(graph.triples[i].triple_id for i in indexes)},
    )


def merge_entities(graph: GraphSnapshot, seed: int) -> GraphSnapshot:
    if len(graph.entities) < 2:
        raise ValueError("entity merge requires at least two entities")
    first, second = _rng(seed, "entity-merge").sample(list(graph.entities), 2)
    merged_id = f"corrupt-merge-{seed}"
    merged = replace(
        first,
        entity_id=merged_id,
        surface_forms=tuple(sorted(set(first.surface_forms + second.surface_forms))),
        canonical_label=f"{first.canonical_label}|{second.canonical_label}",
        source_spans=tuple(sorted(set(first.source_spans + second.source_spans))),
    )
    entities = tuple(
        item for item in graph.entities if item.entity_id not in {first.entity_id, second.entity_id}
    ) + (merged,)
    triples = []
    for triple in graph.triples:
        head = merged_id if triple.head_id in {first.entity_id, second.entity_id} else triple.head_id
        tail = merged_id if triple.tail_id in {first.entity_id, second.entity_id} else triple.tail_id
        if head == tail:
            continue
        triples.append(replace(triple, head_id=head, tail_id=tail))
    return _derive(
        graph,
        "entity-merge",
        entities=entities,
        triples=tuple(triples),
        seed=seed,
        severity=1 / max(1, len(graph.entities)),
        details={"merged": sorted((first.entity_id, second.entity_id)), "new_id": merged_id},
    )


def split_entity(graph: GraphSnapshot, seed: int) -> GraphSnapshot:
    candidates = [item for item in graph.entities if any(len(form.split()) > 1 for form in item.surface_forms)]
    if not candidates:
        raise ValueError("entity split requires a multi-token entity")
    original = _rng(seed, "entity-split").choice(candidates)
    left_id = f"{original.entity_id}-split-a"
    right_id = f"{original.entity_id}-split-b"
    words = original.canonical_label.split()
    pivot = max(1, len(words) // 2)
    left = replace(
        original,
        entity_id=left_id,
        canonical_label=" ".join(words[:pivot]),
        surface_forms=(" ".join(words[:pivot]),),
    )
    right = replace(
        original,
        entity_id=right_id,
        canonical_label=" ".join(words[pivot:]) or words[-1],
        surface_forms=(" ".join(words[pivot:]) or words[-1],),
    )
    entities = tuple(item for item in graph.entities if item.entity_id != original.entity_id) + (
        left,
        right,
    )
    triples = []
    for index, triple in enumerate(graph.triples):
        head = left_id if triple.head_id == original.entity_id else triple.head_id
        tail = right_id if triple.tail_id == original.entity_id else triple.tail_id
        triples.append(replace(triple, head_id=head, tail_id=tail))
    return _derive(
        graph,
        "entity-split",
        entities=entities,
        triples=tuple(triples),
        seed=seed,
        severity=1 / max(1, len(graph.entities)),
        details={"split": original.entity_id, "new_ids": [left_id, right_id]},
    )
