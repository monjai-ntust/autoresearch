"""Dataset adapter protocol and validation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from graph_rag_eval.contracts import CanonicalBundle, Entity, Relation, Triple


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    adapter_id: str
    ready: bool
    issues: tuple[ValidationIssue, ...]
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedGraph:
    """Adapter-supplied predicted graph, independent of the gold graph.

    A predicted graph may omit gold material and may assert entities, relations,
    and triples that the gold graph does not contain. Deriving it from the gold
    records alone would force intrinsic precision to 1.0 and make the framework
    structurally unable to observe extraction false positives.
    """

    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    triples: tuple[Triple, ...]
    extractor_id: str
    construction_recipe: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entities", tuple(sorted(self.entities, key=lambda item: item.entity_id))
        )
        object.__setattr__(
            self, "relations", tuple(sorted(self.relations, key=lambda item: item.relation_id))
        )
        object.__setattr__(
            self, "triples", tuple(sorted(self.triples, key=lambda item: item.triple_id))
        )
        entity_ids = {item.entity_id for item in self.entities}
        relation_ids = {item.relation_id for item in self.relations}
        if len(entity_ids) != len(self.entities) or len(relation_ids) != len(self.relations):
            raise ValueError("predicted graph contains duplicate entities or relations")
        if any(
            item.head_id not in entity_ids
            or item.tail_id not in entity_ids
            or item.relation_id not in relation_ids
            for item in self.triples
        ):
            raise ValueError("predicted graph triple references an undeclared node or relation")
        if not self.extractor_id.strip() or not self.construction_recipe.strip():
            raise ValueError("predicted graph requires extractor and recipe identities")


def gold_subset_graph(
    bundle: CanonicalBundle,
    triple_ids: Iterable[str],
    *,
    extractor_id: str,
    construction_recipe: str = "gold-triple-subset-v1",
) -> GeneratedGraph:
    """Build a recall-only predicted graph for adapters with no separate extractor.

    The result can only lose gold triples, so intrinsic precision is 1.0 by
    construction. Callers must report that limitation rather than present it as a
    measured precision.
    """

    selected = set(triple_ids)
    unknown = selected - {item.triple_id for item in bundle.triples}
    if unknown:
        raise ValueError(f"predicted subset references unknown triples: {sorted(unknown)}")
    triples = tuple(item for item in bundle.triples if item.triple_id in selected)
    return GeneratedGraph(
        entities=bundle.entities,
        relations=bundle.relations,
        triples=triples,
        extractor_id=extractor_id,
        construction_recipe=construction_recipe,
    )


class DatasetAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def validate(self) -> ValidationReport: ...

    def load(self) -> CanonicalBundle: ...

    def generated_graph(self, bundle: CanonicalBundle) -> GeneratedGraph: ...
