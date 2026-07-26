"""CODE-ACCORD canonical adapter and explicit Regime-Q readiness gate."""

from __future__ import annotations

from ast import literal_eval
from hashlib import sha256
from pathlib import Path
import csv

from graph_rag_eval.capabilities import Capability, capability_values
from graph_rag_eval.contracts import (
    CanonicalBundle,
    Chunk,
    DatasetDescriptor,
    Document,
    Entity,
    Provenance,
    content_sha256,
)
from graph_rag_eval.adapters.base import (
    ValidationIssue,
    ValidationReport,
)


class CodeAccordAdapter:
    adapter_id = "code-accord-canonical"
    adapter_version = "1.0.0"

    def __init__(
        self,
        data_root: str = "data/code_accord",
        question_set: str = "output/external-inputs/code-accord-regime-q.jsonl",
        dataset_id: str = "code-accord",
    ):
        self.data_root = Path(data_root)
        self.question_set = Path(question_set)
        self.dataset_id = dataset_id

    @property
    def entity_train(self) -> Path:
        return self.data_root / "entities" / "train.csv"

    def validate(self) -> ValidationReport:
        issues = []
        if not self.entity_train.is_file():
            issues.append(
                ValidationIssue(
                    "missing-corpus",
                    "error",
                    "tracked CODE-ACCORD entity training CSV is absent",
                    self.entity_train.as_posix(),
                )
            )
        for relative in ("entities/test.csv", "relations/train.csv", "relations/test.csv"):
            path = self.data_root / relative
            if not path.is_file():
                issues.append(
                    ValidationIssue(
                        "incomplete-corpus",
                        "block",
                        "required gold graph component is not present in this standalone checkout",
                        path.as_posix(),
                    )
                )
        issues.append(
            ValidationIssue(
                "no-domain-expert",
                "block",
                "independently authored and reviewed Regime Q questions/evidence are unavailable",
                self.question_set.as_posix(),
            )
        )
        capabilities = {
            Capability.DOCUMENTS,
            Capability.DOCUMENT_CLUSTERS,
            Capability.GOLD_ENTITIES,
            Capability.SOURCE_SPANS,
        }
        return ValidationReport(
            adapter_id=self.adapter_id,
            ready=not any(item.severity == "error" for item in issues),
            issues=tuple(issues),
            capabilities=capability_values(capabilities),
        )

    def load(self) -> CanonicalBundle:
        report = self.validate()
        if not report.ready:
            raise ValueError("CODE-ACCORD adapter cannot load: corpus source is absent")
        raw_bytes = self.entity_train.read_bytes()
        raw_hash = sha256(raw_bytes).hexdigest()
        documents = []
        chunks = []
        entities = []
        with self.entity_train.open(encoding="utf-8", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                example_id = row["example_id"]
                raw_text = row["content"]
                processed = row["processed_content"]
                text = processed
                words = processed.split()
                labels = row["label"].split()
                if len(words) != len(labels):
                    raise ValueError(f"CODE-ACCORD BIO length mismatch: {example_id}")
                try:
                    metadata = literal_eval(row.get("metadata", "")) or {}
                except (SyntaxError, ValueError):
                    metadata = {}
                source_id = str(metadata.get("ID", "UNKNOWN"))
                document_id = f"doc-{example_id}"
                chunk_id = f"chunk-{example_id}"
                digest = sha256(text.encode("utf-8")).hexdigest()
                provenance = Provenance(
                    dataset_id=self.dataset_id,
                    snapshot_id=f"entity-train-{raw_hash[:12]}",
                    document_id=document_id,
                    stage="code-accord-entity-csv",
                    producer_version=self.adapter_version,
                    content_sha256=digest,
                )
                documents.append(
                    Document(
                        dataset_id=self.dataset_id,
                        document_id=document_id,
                        text=text,
                        split_id="historical-train-source",
                        cluster_id=source_id,
                        provenance=provenance,
                        metadata={
                            "example_id": example_id,
                            "source_document_id": source_id,
                            "row_index": row_index,
                            "raw_text_sha256": sha256(raw_text.encode("utf-8")).hexdigest(),
                        },
                    )
                )
                chunks.append(
                    Chunk(
                        dataset_id=self.dataset_id,
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=text,
                        char_start=0,
                        char_end=len(text),
                        token_start=0,
                        token_end=len(words),
                        chunker_id="official-sentence-v1",
                        content_sha256=digest,
                        provenance=provenance,
                    )
                )
                start = None
                entity_type = None
                spans = []
                for index, label in enumerate(labels + ["O"]):
                    prefix, _, raw_type = label.partition("-")
                    if prefix == "B" or (prefix == "I" and start is None):
                        if start is not None:
                            spans.append((start, index, entity_type))
                        start = index
                        entity_type = raw_type.capitalize()
                    elif prefix == "I" and raw_type.capitalize() == entity_type:
                        continue
                    elif start is not None:
                        spans.append((start, index, entity_type))
                        start = None
                        entity_type = None
                for ordinal, (word_start, word_end, normalized_type) in enumerate(spans):
                    label = " ".join(words[word_start:word_end])
                    char_start = processed.find(label)
                    char_end = char_start + len(label)
                    entity_id = f"entity-{example_id}-{ordinal:03d}"
                    entities.append(
                        Entity(
                            dataset_id=self.dataset_id,
                            entity_id=entity_id,
                            surface_forms=(label,),
                            canonical_label=label.casefold(),
                            entity_type=normalized_type,
                            source_spans=((chunk_id, char_start, char_end),),
                            provenance=Provenance(
                                dataset_id=self.dataset_id,
                                snapshot_id=f"entity-train-{raw_hash[:12]}",
                                document_id=document_id,
                                chunk_id=chunk_id,
                                span_start=char_start,
                                span_end=char_end,
                                stage="code-accord-bio",
                                producer_version=self.adapter_version,
                                content_sha256=sha256(label.encode("utf-8")).hexdigest(),
                            ),
                        )
                    )
        canonical_hash = content_sha256(
            {
                "documents": documents,
                "chunks": chunks,
                "entities": entities,
                "relations": [],
                "triples": [],
            }
        )
        descriptor = DatasetDescriptor(
            dataset_id=self.dataset_id,
            snapshot_id=f"entity-train-{raw_hash[:12]}",
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            license="CODE-ACCORD terms require external review",
            citation="Hettiarachchi et al. (2024), CODE-ACCORD",
            language="en",
            capabilities=report.capabilities,
            raw_content_sha256=raw_hash,
            canonical_content_sha256=canonical_hash,
            redistribution="partial tracked source only",
        )
        return CanonicalBundle(
            descriptor=descriptor,
            documents=tuple(sorted(documents, key=lambda item: item.document_id)),
            chunks=tuple(sorted(chunks, key=lambda item: item.chunk_id)),
            entities=tuple(sorted(entities, key=lambda item: item.entity_id)),
            relations=(),
            triples=(),
            questions=(),
            answers=(),
            evidence_sets=(),
        )

    def generated_triple_ids(self) -> tuple[str, ...]:
        return ()
