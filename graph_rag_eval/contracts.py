"""Immutable canonical records and leakage-safe question projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import json

from .capabilities import capability_values

CONTRACT_VERSION = "graph-rag-canonical-1.0"


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _require_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty opaque string")
    if "\x00" in value:
        raise ValueError(f"{name} contains a NUL byte")


def canonical_data(value: Any) -> Any:
    """Convert records into stable JSON-compatible data."""

    if is_dataclass(value):
        return {
            item.name: canonical_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): canonical_data(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [canonical_data(item) for item in value]
    if isinstance(value, set):
        return [canonical_data(item) for item in sorted(value, key=str)]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Provenance:
    dataset_id: str
    snapshot_id: str
    stage: str
    producer_version: str
    content_sha256: str
    document_id: str | None = None
    chunk_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    parent_ids: tuple[str, ...] = ()
    timestamp: str = "deterministic"

    def __post_init__(self) -> None:
        for name in ("dataset_id", "snapshot_id", "stage", "producer_version"):
            _require_id(name, getattr(self, name))
        if len(self.content_sha256) != 64:
            raise ValueError("provenance content_sha256 must be a SHA-256 hex digest")
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("provenance span bounds must be both present or both absent")
        if self.span_start is not None and not 0 <= self.span_start <= self.span_end:
            raise ValueError("provenance span bounds are invalid")


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_id: str
    snapshot_id: str
    adapter_id: str
    adapter_version: str
    license: str
    citation: str
    language: str
    capabilities: tuple[str, ...]
    raw_content_sha256: str
    canonical_content_sha256: str
    redistribution: str = "unknown"
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("dataset_id", "snapshot_id", "adapter_id", "adapter_version"):
            _require_id(name, getattr(self, name))
        object.__setattr__(self, "capabilities", capability_values(self.capabilities))
        for name in ("raw_content_sha256", "canonical_content_sha256"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")


@dataclass(frozen=True)
class Document:
    dataset_id: str
    document_id: str
    text: str
    split_id: str
    cluster_id: str
    provenance: Provenance
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("dataset_id", "document_id", "split_id", "cluster_id"):
            _require_id(name, getattr(self, name))
        if not isinstance(self.text, str):
            raise ValueError("document text must be a string")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class Chunk:
    dataset_id: str
    chunk_id: str
    document_id: str
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    chunker_id: str
    content_sha256: str
    provenance: Provenance
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("dataset_id", "chunk_id", "document_id", "chunker_id"):
            _require_id(name, getattr(self, name))
        if not (0 <= self.char_start <= self.char_end and 0 <= self.token_start <= self.token_end):
            raise ValueError("chunk offsets are invalid")
        if self.content_sha256 != sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("chunk content hash does not match text")


@dataclass(frozen=True)
class Entity:
    dataset_id: str
    entity_id: str
    surface_forms: tuple[str, ...]
    canonical_label: str
    entity_type: str | None
    source_spans: tuple[tuple[str, int, int], ...]
    provenance: Provenance
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_id("entity_id", self.entity_id)
        if not self.surface_forms or any(not item for item in self.surface_forms):
            raise ValueError("entity surface_forms must be non-empty")
        for chunk_id, start, end in self.source_spans:
            _require_id("source span chunk_id", chunk_id)
            if not 0 <= start <= end:
                raise ValueError("entity source span is invalid")


@dataclass(frozen=True)
class Relation:
    dataset_id: str
    relation_id: str
    label: str
    direction: str
    provenance: Provenance
    description: str | None = None
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_id("relation_id", self.relation_id)
        _require_id("relation label", self.label)
        if self.direction not in {"directed", "undirected"}:
            raise ValueError("relation direction must be directed or undirected")


@dataclass(frozen=True)
class Triple:
    dataset_id: str
    triple_id: str
    head_id: str
    relation_id: str
    tail_id: str
    chunk_ids: tuple[str, ...]
    confidence: float | None
    provenance: Provenance
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("triple_id", "head_id", "relation_id", "tail_id"):
            _require_id(name, getattr(self, name))
        if self.head_id == self.tail_id:
            raise ValueError("self-loop triples require an explicit later contract")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("triple confidence must be in [0, 1]")


@dataclass(frozen=True)
class Question:
    dataset_id: str
    question_id: str
    text: str
    split_id: str
    cluster_id: str
    regime: str
    authoring_provenance: str
    validation_provenance: str
    public_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("question_id", "split_id", "cluster_id"):
            _require_id(name, getattr(self, name))
        if self.regime not in {"diagnostic_fact_probe", "question_only"}:
            raise ValueError("question regime is invalid")
        if not self.text.strip():
            raise ValueError("question text is empty")
        object.__setattr__(self, "public_metadata", _frozen_mapping(self.public_metadata))

    def public_view(self) -> "QueryView":
        return QueryView(
            dataset_id=self.dataset_id,
            question_id=self.question_id,
            text=self.text,
            regime=self.regime,
            public_metadata=self.public_metadata,
        )


@dataclass(frozen=True)
class QueryView:
    """The only question type accepted by retrieval and generation."""

    dataset_id: str
    question_id: str
    text: str
    regime: str
    public_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_metadata", _frozen_mapping(self.public_metadata))


@dataclass(frozen=True)
class AnswerAlias:
    dataset_id: str
    question_id: str
    normalized_answers: tuple[str, ...]
    answer_type: str
    abstention_expected: bool = False
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_id("question_id", self.question_id)
        if not self.abstention_expected and not self.normalized_answers:
            raise ValueError("answerable question has no accepted answers")


@dataclass(frozen=True)
class EvidenceSet:
    dataset_id: str
    question_id: str
    evidence_set_id: str
    evidence_ids: tuple[str, ...]
    relevance_grade: int
    annotation_provenance: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_id("evidence_set_id", self.evidence_set_id)
        if not self.evidence_ids:
            raise ValueError("evidence set cannot be empty")
        if self.relevance_grade < 0:
            raise ValueError("relevance grade cannot be negative")


@dataclass(frozen=True)
class RelevanceJudgment:
    dataset_id: str
    question_id: str
    evidence_id: str
    grade: int
    assessor: str
    adjudication_state: str
    schema_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class SplitMembership:
    dataset_id: str
    item_id: str
    split_name: str
    grouping_keys: tuple[str, ...]
    algorithm: str
    algorithm_version: str
    seed: int
    schema_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CanonicalBundle:
    descriptor: DatasetDescriptor
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    triples: tuple[Triple, ...]
    questions: tuple[Question, ...]
    answers: tuple[AnswerAlias, ...]
    evidence_sets: tuple[EvidenceSet, ...]
    relevance_judgments: tuple[RelevanceJudgment, ...] = ()
    split_membership: tuple[SplitMembership, ...] = ()

    def __post_init__(self) -> None:
        collections: Sequence[tuple[str, Sequence[Any], str]] = (
            ("documents", self.documents, "document_id"),
            ("chunks", self.chunks, "chunk_id"),
            ("entities", self.entities, "entity_id"),
            ("relations", self.relations, "relation_id"),
            ("triples", self.triples, "triple_id"),
            ("questions", self.questions, "question_id"),
        )
        for label, values, attr in collections:
            ids = [getattr(value, attr) for value in values]
            if ids != sorted(ids):
                raise ValueError(f"{label} must be stably sorted")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} contains duplicate IDs")
        document_ids = {item.document_id for item in self.documents}
        chunk_ids = {item.chunk_id for item in self.chunks}
        entity_ids = {item.entity_id for item in self.entities}
        relation_ids = {item.relation_id for item in self.relations}
        question_ids = {item.question_id for item in self.questions}
        if any(item.document_id not in document_ids for item in self.chunks):
            raise ValueError("chunk references an unknown document")
        if any(
            item.head_id not in entity_ids
            or item.tail_id not in entity_ids
            or item.relation_id not in relation_ids
            or any(chunk_id not in chunk_ids for chunk_id in item.chunk_ids)
            for item in self.triples
        ):
            raise ValueError("triple contains a foreign key")
        if any(item.question_id not in question_ids for item in self.answers):
            raise ValueError("answer references an unknown question")
        if any(item.question_id not in question_ids for item in self.evidence_sets):
            raise ValueError("evidence set references an unknown question")

    def private_tokens(self) -> tuple[str, ...]:
        tokens: set[str] = set()
        for answer in self.answers:
            tokens.update(answer.normalized_answers)
        for evidence in self.evidence_sets:
            tokens.update(evidence.evidence_ids)
        return tuple(sorted(token for token in tokens if token))
