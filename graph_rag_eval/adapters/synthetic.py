"""Deterministic no-network fixture exercising every framework layer."""

from __future__ import annotations

from hashlib import sha256

from graph_rag_eval.capabilities import Capability, capability_values
from graph_rag_eval.contracts import (
    AnswerAlias,
    CanonicalBundle,
    Chunk,
    DatasetDescriptor,
    Document,
    Entity,
    EvidenceSet,
    Provenance,
    Question,
    Relation,
    RelevanceJudgment,
    SplitMembership,
    Triple,
    canonical_data,
    content_sha256,
)
from graph_rag_eval.adapters.base import GeneratedGraph, ValidationReport


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class SyntheticAdapter:
    adapter_id = "synthetic-fixture"
    adapter_version = "1.0.0"

    def __init__(self, dataset_id: str = "synthetic-rag", variant: str = "base"):
        self.dataset_id = dataset_id
        self.variant = variant

    def validate(self) -> ValidationReport:
        return ValidationReport(
            adapter_id=self.adapter_id,
            ready=True,
            issues=(),
            capabilities=capability_values(Capability),
        )

    def load(self) -> CanonicalBundle:
        rows = (
            ("doc-01", "Fire doors require steel frames.", "development"),
            ("doc-02", "Emergency exits require a clear width of 800 mm.", "test"),
            ("doc-03", "Ventilation ducts use aluminium covers.", "test"),
            ("doc-04", "A service hatch has no declared minimum width.", "test"),
        )
        raw_hash = content_sha256({"variant": self.variant, "rows": rows})
        documents = []
        chunks = []
        for document_id, text, split in rows:
            digest = _text_hash(text)
            provenance = Provenance(
                dataset_id=self.dataset_id,
                snapshot_id=f"fixture-{self.variant}",
                document_id=document_id,
                stage="synthetic-source",
                producer_version=self.adapter_version,
                content_sha256=digest,
            )
            documents.append(
                Document(
                    dataset_id=self.dataset_id,
                    document_id=document_id,
                    text=text,
                    split_id=split,
                    cluster_id=document_id,
                    provenance=provenance,
                    metadata={"synthetic": True, "variant": self.variant},
                )
            )
            chunks.append(
                Chunk(
                    dataset_id=self.dataset_id,
                    chunk_id=f"chunk-{document_id[-2:]}",
                    document_id=document_id,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    token_start=0,
                    token_end=len(text.split()),
                    chunker_id="whole-document-v1",
                    content_sha256=digest,
                    provenance=provenance,
                )
            )

        entity_specs = (
            ("e-aluminium", "aluminium covers", "Object", "chunk-03", 22, 38),
            ("e-clear-width", "clear width", "Property", "chunk-02", 26, 37),
            ("e-ducts", "Ventilation ducts", "Object", "chunk-03", 0, 17),
            ("e-emergency-exit", "Emergency exits", "Object", "chunk-02", 0, 15),
            ("e-fire-door", "Fire doors", "Object", "chunk-01", 0, 10),
            ("e-service-hatch", "service hatch", "Object", "chunk-04", 2, 15),
            ("e-steel-frame", "steel frames", "Object", "chunk-01", 19, 31),
            ("e-width-800", "800 mm", "Value", "chunk-02", 41, 47),
        )
        entities = []
        for entity_id, label, entity_type, chunk_id, start, end in entity_specs:
            entities.append(
                Entity(
                    dataset_id=self.dataset_id,
                    entity_id=entity_id,
                    surface_forms=(label,),
                    canonical_label=label.casefold(),
                    entity_type=entity_type,
                    source_spans=((chunk_id, start, end),),
                    provenance=Provenance(
                        dataset_id=self.dataset_id,
                        snapshot_id=f"fixture-{self.variant}",
                        chunk_id=chunk_id,
                        span_start=start,
                        span_end=end,
                        stage="synthetic-gold-entity",
                        producer_version=self.adapter_version,
                        content_sha256=_text_hash(label),
                    ),
                )
            )
        relation_specs = (
            ("r-minimum", "minimum", "minimum threshold"),
            ("r-requires", "requires", "requirement"),
            ("r-uses", "uses", "material use"),
        )
        relations = tuple(
            Relation(
                dataset_id=self.dataset_id,
                relation_id=relation_id,
                label=label,
                description=description,
                direction="directed",
                provenance=Provenance(
                    dataset_id=self.dataset_id,
                    snapshot_id=f"fixture-{self.variant}",
                    stage="synthetic-schema",
                    producer_version=self.adapter_version,
                    content_sha256=_text_hash(f"{relation_id}:{label}"),
                ),
            )
            for relation_id, label, description in relation_specs
        )
        triple_specs = (
            ("t-01", "e-fire-door", "r-requires", "e-steel-frame", "chunk-01"),
            ("t-02", "e-emergency-exit", "r-requires", "e-clear-width", "chunk-02"),
            ("t-03", "e-clear-width", "r-minimum", "e-width-800", "chunk-02"),
            ("t-04", "e-ducts", "r-uses", "e-aluminium", "chunk-03"),
        )
        triples = tuple(
            Triple(
                dataset_id=self.dataset_id,
                triple_id=triple_id,
                head_id=head_id,
                relation_id=relation_id,
                tail_id=tail_id,
                chunk_ids=(chunk_id,),
                confidence=1.0,
                provenance=Provenance(
                    dataset_id=self.dataset_id,
                    snapshot_id=f"fixture-{self.variant}",
                    chunk_id=chunk_id,
                    stage="synthetic-gold-triple",
                    producer_version=self.adapter_version,
                    content_sha256=_text_hash(
                        f"{triple_id}:{head_id}:{relation_id}:{tail_id}"
                    ),
                ),
            )
            for triple_id, head_id, relation_id, tail_id, chunk_id in triple_specs
        )
        questions = (
            Question(
                dataset_id=self.dataset_id,
                question_id="q-01",
                text="What frames do fire doors require?",
                split_id="test",
                cluster_id="doc-01",
                regime="question_only",
                authoring_provenance="synthetic-fixture-author-v1",
                validation_provenance="synthetic-fixture-reviewer-v1",
            ),
            Question(
                dataset_id=self.dataset_id,
                question_id="q-02",
                text="What minimum clear width is required for emergency exits?",
                split_id="test",
                cluster_id="doc-02",
                regime="question_only",
                authoring_provenance="synthetic-fixture-author-v1",
                validation_provenance="synthetic-fixture-reviewer-v1",
            ),
            Question(
                dataset_id=self.dataset_id,
                question_id="q-03",
                text="Which material covers ventilation ducts?",
                split_id="test",
                cluster_id="doc-03",
                regime="diagnostic_fact_probe",
                authoring_provenance="synthetic-template-v1",
                validation_provenance="diagnostic-only",
            ),
            Question(
                dataset_id=self.dataset_id,
                question_id="q-04",
                text="What minimum width is declared for the service hatch?",
                split_id="test",
                cluster_id="doc-04",
                regime="question_only",
                authoring_provenance="synthetic-fixture-author-v1",
                validation_provenance="synthetic-fixture-reviewer-v1",
                public_metadata={"answerability": "unanswerable"},
            ),
        )
        answers = (
            AnswerAlias(self.dataset_id, "q-01", ("steel frames",), "entity"),
            AnswerAlias(self.dataset_id, "q-02", ("800 mm", "800mm"), "value"),
            AnswerAlias(self.dataset_id, "q-03", ("aluminium", "aluminium covers"), "entity"),
            AnswerAlias(self.dataset_id, "q-04", (), "abstention", True),
        )
        evidence_sets = (
            EvidenceSet(
                self.dataset_id,
                "q-01",
                "ev-q01-graph",
                ("t-01",),
                2,
                "synthetic-fixture-reviewer-v1",
            ),
            EvidenceSet(
                self.dataset_id,
                "q-01",
                "ev-q01-text",
                ("chunk-01",),
                2,
                "synthetic-fixture-reviewer-v1",
            ),
            EvidenceSet(
                self.dataset_id,
                "q-02",
                "ev-q02-graph",
                ("t-02", "t-03"),
                2,
                "synthetic-fixture-reviewer-v1",
            ),
            EvidenceSet(
                self.dataset_id,
                "q-02",
                "ev-q02-text",
                ("chunk-02",),
                2,
                "synthetic-fixture-reviewer-v1",
            ),
            EvidenceSet(
                self.dataset_id,
                "q-03",
                "ev-q03-graph",
                ("t-04",),
                2,
                "synthetic-diagnostic-v1",
            ),
            EvidenceSet(
                self.dataset_id,
                "q-03",
                "ev-q03-text",
                ("chunk-03",),
                2,
                "synthetic-diagnostic-v1",
            ),
            EvidenceSet(
                self.dataset_id,
                "q-04",
                "ev-q04-text",
                ("chunk-04",),
                1,
                "synthetic-fixture-reviewer-v1",
            ),
        )
        judgments = tuple(
            RelevanceJudgment(
                self.dataset_id,
                evidence.question_id,
                evidence_id,
                evidence.relevance_grade,
                "synthetic-reviewer",
                "frozen",
            )
            for evidence in evidence_sets
            for evidence_id in evidence.evidence_ids
        )
        splits = tuple(
            SplitMembership(
                self.dataset_id,
                question.question_id,
                question.split_id,
                (question.cluster_id,),
                "fixture-explicit",
                "1.0",
                42,
            )
            for question in questions
        )
        provisional = {
            "documents": canonical_data(tuple(documents)),
            "chunks": canonical_data(tuple(chunks)),
            "entities": canonical_data(tuple(entities)),
            "relations": canonical_data(relations),
            "triples": canonical_data(triples),
            "questions": canonical_data(questions),
        }
        descriptor = DatasetDescriptor(
            dataset_id=self.dataset_id,
            snapshot_id=f"fixture-{self.variant}",
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            license="CC0-1.0 synthetic fixture",
            citation="Repository-authored deterministic test fixture",
            language="en",
            capabilities=capability_values(Capability),
            raw_content_sha256=raw_hash,
            canonical_content_sha256=content_sha256(provisional),
            redistribution="included",
        )
        return CanonicalBundle(
            descriptor=descriptor,
            documents=tuple(sorted(documents, key=lambda item: item.document_id)),
            chunks=tuple(sorted(chunks, key=lambda item: item.chunk_id)),
            entities=tuple(sorted(entities, key=lambda item: item.entity_id)),
            relations=tuple(sorted(relations, key=lambda item: item.relation_id)),
            triples=tuple(sorted(triples, key=lambda item: item.triple_id)),
            questions=tuple(sorted(questions, key=lambda item: item.question_id)),
            answers=tuple(sorted(answers, key=lambda item: item.question_id)),
            evidence_sets=tuple(
                sorted(evidence_sets, key=lambda item: item.evidence_set_id)
            ),
            relevance_judgments=tuple(
                sorted(judgments, key=lambda item: (item.question_id, item.evidence_id))
            ),
            split_membership=tuple(sorted(splits, key=lambda item: item.item_id)),
        )

    def generated_graph(self, bundle: CanonicalBundle) -> GeneratedGraph:
        """Predicted graph with both extraction misses and hallucinations.

        The fixture omits one gold entity and one gold triple and asserts one
        hallucinated entity, relation, and triple, so intrinsic precision and
        recall are both strictly below 1.0 and every matching branch is exercised.
        """

        dropped_entity = "e-service-hatch"
        dropped_triple = "t-03"
        hallucinated_entity = Entity(
            dataset_id=self.dataset_id,
            entity_id="e-hallucinated-vent-cover",
            surface_forms=("vent cover",),
            canonical_label="vent cover",
            entity_type="Object",
            source_spans=(("chunk-03", 0, 17),),
            provenance=Provenance(
                dataset_id=self.dataset_id,
                snapshot_id=f"fixture-{self.variant}",
                chunk_id="chunk-03",
                span_start=0,
                span_end=17,
                stage="synthetic-predicted-entity",
                producer_version=self.adapter_version,
                content_sha256=_text_hash("vent cover"),
            ),
        )
        hallucinated_relation = Relation(
            dataset_id=self.dataset_id,
            relation_id="r-prohibits",
            label="prohibits",
            description="hallucinated prohibition",
            direction="directed",
            provenance=Provenance(
                dataset_id=self.dataset_id,
                snapshot_id=f"fixture-{self.variant}",
                stage="synthetic-predicted-schema",
                producer_version=self.adapter_version,
                content_sha256=_text_hash("r-prohibits:prohibits"),
            ),
        )
        hallucinated_triple = Triple(
            dataset_id=self.dataset_id,
            triple_id="t-90",
            head_id="e-ducts",
            relation_id="r-prohibits",
            tail_id="e-hallucinated-vent-cover",
            chunk_ids=("chunk-03",),
            confidence=0.4,
            provenance=Provenance(
                dataset_id=self.dataset_id,
                snapshot_id=f"fixture-{self.variant}",
                chunk_id="chunk-03",
                stage="synthetic-predicted-triple",
                producer_version=self.adapter_version,
                content_sha256=_text_hash("t-90:e-ducts:r-prohibits:e-hallucinated-vent-cover"),
            ),
        )
        return GeneratedGraph(
            entities=tuple(
                item for item in bundle.entities if item.entity_id != dropped_entity
            )
            + (hallucinated_entity,),
            relations=bundle.relations + (hallucinated_relation,),
            triples=tuple(
                item for item in bundle.triples if item.triple_id != dropped_triple
            )
            + (hallucinated_triple,),
            extractor_id="synthetic-lossy-and-hallucinating-extractor-v1",
            construction_recipe="drop-one-entity-and-triple-add-one-hallucination-v1",
        )
