"""Faithful SciERC extraction adapter with pinned split and ontology gates."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from graph_rag_eval.adapters.base import (
    GeneratedGraph,
    ValidationIssue,
    ValidationReport,
)
from graph_rag_eval.capabilities import Capability, capability_values
from graph_rag_eval.contracts import (
    CanonicalBundle,
    Chunk,
    DatasetDescriptor,
    Document,
    Entity,
    Provenance,
    Relation,
    SplitMembership,
    Triple,
    content_sha256,
)
from graph_rag_eval.identity import file_sha256


ENTITY_TYPES = (
    "Task",
    "Method",
    "Metric",
    "Material",
    "OtherScientificTerm",
    "Generic",
)
RELATION_TYPES = (
    "USED-FOR",
    "FEATURE-OF",
    "HYPONYM-OF",
    "EVALUATE-FOR",
    "PART-OF",
    "COMPARE",
    "CONJUNCTION",
)
SYMMETRIC_RELATION_TYPES = {"COMPARE", "CONJUNCTION"}
SPLIT_NAMES = ("train", "dev", "test")
OFFICIAL_SPLIT_SHA256 = {
    "train.json": "04970819bd215fce8c60b3a64fccca15388b49eab2010bdc5f6d322b463568c3",
    "dev.json": "61f21b224129e580a654b036ee6fffeeb1a0311f1009a38ea931c13612db040e",
    "test.json": "7424da64e6214a90e39b09e47a74b3ded57dde86b4a7f848b3625d2c8ecdfbe1",
}
OFFICIAL_DOCUMENT_COUNTS = {"train": 350, "dev": 50, "test": 100}


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _relation_id(label: str) -> str:
    return f"scierc:relation:{label}"


def _document_id(doc_key: str) -> str:
    return f"scierc:document:{doc_key}"


def _chunk_id(doc_key: str, sentence_index: int) -> str:
    return f"scierc:document:{doc_key}:sentence:{sentence_index:03d}"


def _entity_id(
    doc_key: str,
    start: int,
    end: int,
    entity_type: str,
) -> str:
    return f"scierc:mention:{doc_key}:{start}:{end}:{entity_type}"


def _triple_id(
    doc_key: str,
    sentence_index: int,
    head_start: int,
    head_end: int,
    tail_start: int,
    tail_end: int,
    relation_type: str,
) -> str:
    return (
        f"scierc:triple:{doc_key}:{sentence_index:03d}:"
        f"{head_start}:{head_end}:{relation_type}:{tail_start}:{tail_end}"
    )


class SciERCAdapter:
    """Map native SciERC mentions/relations without inventing QA or graph labels."""

    adapter_id = "scierc-native-extraction"
    adapter_version = "1.0.0"

    def __init__(
        self,
        run_root: str,
        data_root: str = "inputs/scierc",
        prediction_ledger: str | None = "inputs/scierc-predictions.jsonl",
        dataset_id: str = "scierc",
        *,
        fixture_mode: bool = False,
        expected_split_sha256: dict[str, str] | None = None,
        expected_document_counts: dict[str, int] | None = None,
    ):
        self.run_root = Path(run_root).resolve()
        self.data_root = self._resolve_run_path(data_root)
        self.prediction_ledger = (
            self._resolve_run_path(prediction_ledger)
            if prediction_ledger is not None
            else None
        )
        self.dataset_id = dataset_id
        self.fixture_mode = bool(fixture_mode)
        if self.fixture_mode:
            if expected_split_sha256 is None or expected_document_counts is None:
                raise ValueError(
                    "fixture_mode requires explicit split hashes and document counts"
                )
            self.expected_split_sha256 = dict(expected_split_sha256)
            self.expected_document_counts = dict(expected_document_counts)
        else:
            if (
                expected_split_sha256 is not None
                and dict(expected_split_sha256) != OFFICIAL_SPLIT_SHA256
            ):
                raise ValueError("scientific SciERC runs cannot override official split hashes")
            if (
                expected_document_counts is not None
                and dict(expected_document_counts) != OFFICIAL_DOCUMENT_COUNTS
            ):
                raise ValueError(
                    "scientific SciERC runs cannot override official split counts"
                )
            self.expected_split_sha256 = dict(OFFICIAL_SPLIT_SHA256)
            self.expected_document_counts = dict(OFFICIAL_DOCUMENT_COUNTS)
        if set(self.expected_split_sha256) != {
            f"{split}.json" for split in SPLIT_NAMES
        }:
            raise ValueError("SciERC split hash map must name train.json, dev.json, test.json")
        if set(self.expected_document_counts) != set(SPLIT_NAMES):
            raise ValueError("SciERC count map must name train, dev, test")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.expected_split_sha256.values()
        ):
            raise ValueError("SciERC expected split hashes must be lowercase SHA-256")
        self._bundle: CanonicalBundle | None = None
        self._rows_by_doc_key: dict[str, dict[str, Any]] = {}

    def _resolve_run_path(self, supplied: str) -> Path:
        raw = Path(supplied)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError("SciERC input paths must be run-relative and traversal-free")
        candidate = (self.run_root / raw).resolve()
        if self.run_root != candidate and self.run_root not in candidate.parents:
            raise ValueError("SciERC input path escapes the declared run root")
        return candidate

    def _split_paths(self) -> dict[str, Path]:
        return {
            split: self.data_root / f"{split}.json"
            for split in SPLIT_NAMES
        }

    def _read_rows(self) -> tuple[dict[str, Any], ...]:
        rows = []
        seen: dict[str, str] = {}
        for split, path in self._split_paths().items():
            with path.open(encoding="utf-8") as handle:
                split_rows = [
                    json.loads(line)
                    for line in handle
                    if line.strip()
                ]
            if len(split_rows) != self.expected_document_counts[split]:
                raise ValueError(
                    f"SciERC {split} document count mismatch: "
                    f"{len(split_rows)} != {self.expected_document_counts[split]}"
                )
            for row in split_rows:
                required = {"doc_key", "sentences", "ner", "relations"}
                missing = sorted(required - set(row))
                if missing:
                    raise ValueError(
                        f"SciERC {split} record is missing keys: {missing}"
                    )
                doc_key = row["doc_key"]
                if not isinstance(doc_key, str) or not doc_key:
                    raise ValueError("SciERC doc_key must be a non-empty string")
                if doc_key in seen:
                    raise ValueError(
                        f"SciERC document overlaps {seen[doc_key]} and {split}: {doc_key}"
                    )
                seen[doc_key] = split
                if not (
                    len(row["sentences"])
                    == len(row["ner"])
                    == len(row["relations"])
                ):
                    raise ValueError(
                        f"SciERC sentence/NER/relation alignment mismatch: {doc_key}"
                    )
                rows.append({"split": split, "record": row})
        return tuple(rows)

    @staticmethod
    def _sentence_offsets(sentences: list[list[str]]) -> list[int]:
        offsets = [0]
        for sentence in sentences:
            offsets.append(offsets[-1] + len(sentence))
        return offsets

    @staticmethod
    def _token_char_spans(words: list[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
        text = " ".join(words)
        spans = []
        cursor = 0
        for word in words:
            spans.append((cursor, cursor + len(word)))
            cursor += len(word) + 1
        return text, tuple(spans)

    def _build_relations(self, snapshot_id: str) -> tuple[Relation, ...]:
        return tuple(
            Relation(
                dataset_id=self.dataset_id,
                relation_id=_relation_id(label),
                label=label,
                direction=(
                    "undirected"
                    if label in SYMMETRIC_RELATION_TYPES
                    else "directed"
                ),
                description="Native SciERC typed relation",
                provenance=Provenance(
                    dataset_id=self.dataset_id,
                    snapshot_id=snapshot_id,
                    stage="scierc-native-ontology",
                    producer_version=self.adapter_version,
                    content_sha256=_sha256_text(label),
                ),
            )
            for label in sorted(RELATION_TYPES)
        )

    def _build_bundle(self) -> CanonicalBundle:
        if self._bundle is not None:
            return self._bundle
        rows = self._read_rows()
        split_hashes = {
            f"{split}.json": file_sha256(path)
            for split, path in self._split_paths().items()
        }
        raw_hash = content_sha256(split_hashes)
        snapshot_id = (
            f"fixture-{raw_hash[:12]}"
            if self.fixture_mode
            else f"official-processed-{raw_hash[:12]}"
        )
        documents = []
        chunks = []
        entities = []
        triples = []
        memberships = []
        rows_by_doc_key = {}

        for item in rows:
            split = item["split"]
            row = item["record"]
            doc_key = row["doc_key"]
            sentences = row["sentences"]
            offsets = self._sentence_offsets(sentences)
            sentence_texts = [" ".join(words) for words in sentences]
            document_text = "\n".join(sentence_texts)
            document_id = _document_id(doc_key)
            document_hash = _sha256_text(document_text)
            document_provenance = Provenance(
                dataset_id=self.dataset_id,
                snapshot_id=snapshot_id,
                document_id=document_id,
                stage="scierc-processed-jsonl",
                producer_version=self.adapter_version,
                content_sha256=document_hash,
            )
            documents.append(
                Document(
                    dataset_id=self.dataset_id,
                    document_id=document_id,
                    text=document_text,
                    split_id=split,
                    cluster_id=doc_key,
                    provenance=document_provenance,
                    metadata={
                        "doc_key": doc_key,
                        "sentence_count": len(sentences),
                        "coreference_cluster_count": len(row.get("clusters", ())),
                        "native_split": split,
                    },
                )
            )
            memberships.append(
                SplitMembership(
                    dataset_id=self.dataset_id,
                    item_id=document_id,
                    split_name=split,
                    grouping_keys=(doc_key,),
                    algorithm="official-scierc-split",
                    algorithm_version="processed-data-pinned-2026-07-29",
                    seed=0,
                )
            )
            mention_ids: dict[tuple[int, int], str] = {}
            document_char_start = 0
            sentence_index_data = []
            for sentence_index, words in enumerate(sentences):
                if not isinstance(words, list) or any(
                    not isinstance(word, str) for word in words
                ):
                    raise ValueError(f"SciERC sentence tokens are malformed: {doc_key}")
                sentence_text, token_char_spans = self._token_char_spans(words)
                chunk_id = _chunk_id(doc_key, sentence_index)
                sentence_hash = _sha256_text(sentence_text)
                chunks.append(
                    Chunk(
                        dataset_id=self.dataset_id,
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=sentence_text,
                        char_start=document_char_start,
                        char_end=document_char_start + len(sentence_text),
                        token_start=offsets[sentence_index],
                        token_end=offsets[sentence_index + 1],
                        chunker_id="scierc-native-sentence-v1",
                        content_sha256=sentence_hash,
                        provenance=Provenance(
                            dataset_id=self.dataset_id,
                            snapshot_id=snapshot_id,
                            document_id=document_id,
                            chunk_id=chunk_id,
                            stage="scierc-native-sentence",
                            producer_version=self.adapter_version,
                            content_sha256=sentence_hash,
                        ),
                    )
                )
                sentence_index_data.append(
                    {
                        "words": words,
                        "token_char_spans": token_char_spans,
                        "chunk_id": chunk_id,
                        "token_offset": offsets[sentence_index],
                    }
                )
                for start, end, entity_type in row["ner"][sentence_index]:
                    if entity_type not in ENTITY_TYPES:
                        raise ValueError(
                            f"unknown SciERC entity type {entity_type}: {doc_key}"
                        )
                    local_start = start - offsets[sentence_index]
                    local_end = end - offsets[sentence_index]
                    if not (0 <= local_start <= local_end < len(words)):
                        raise ValueError(
                            f"SciERC entity crosses or escapes its sentence: {doc_key}"
                        )
                    if (start, end) in mention_ids:
                        raise ValueError(
                            f"SciERC span has duplicate/ambiguous entity labels: "
                            f"{doc_key}:{start}:{end}"
                        )
                    char_start = token_char_spans[local_start][0]
                    char_end = token_char_spans[local_end][1]
                    surface = sentence_text[char_start:char_end]
                    entity_id = _entity_id(doc_key, start, end, entity_type)
                    mention_ids[(start, end)] = entity_id
                    entities.append(
                        Entity(
                            dataset_id=self.dataset_id,
                            entity_id=entity_id,
                            surface_forms=(surface,),
                            canonical_label=surface.casefold(),
                            entity_type=entity_type,
                            source_spans=((chunk_id, char_start, char_end),),
                            provenance=Provenance(
                                dataset_id=self.dataset_id,
                                snapshot_id=snapshot_id,
                                document_id=document_id,
                                chunk_id=chunk_id,
                                span_start=char_start,
                                span_end=char_end,
                                stage="scierc-gold-entity",
                                producer_version=self.adapter_version,
                                content_sha256=_sha256_text(
                                    f"{entity_id}:{surface}"
                                ),
                            ),
                        )
                    )
                document_char_start += len(sentence_text)
                if sentence_index + 1 < len(sentences):
                    document_char_start += 1

            for sentence_index, relation_rows in enumerate(row["relations"]):
                chunk_id = _chunk_id(doc_key, sentence_index)
                for head_start, head_end, tail_start, tail_end, relation_type in relation_rows:
                    if relation_type not in RELATION_TYPES:
                        raise ValueError(
                            f"unknown SciERC relation type {relation_type}: {doc_key}"
                        )
                    head_id = mention_ids.get((head_start, head_end))
                    tail_id = mention_ids.get((tail_start, tail_end))
                    if head_id is None or tail_id is None:
                        raise ValueError(
                            f"SciERC relation endpoint lacks an entity: {doc_key}"
                        )
                    triple_id = _triple_id(
                        doc_key,
                        sentence_index,
                        head_start,
                        head_end,
                        tail_start,
                        tail_end,
                        relation_type,
                    )
                    triples.append(
                        Triple(
                            dataset_id=self.dataset_id,
                            triple_id=triple_id,
                            head_id=head_id,
                            relation_id=_relation_id(relation_type),
                            tail_id=tail_id,
                            chunk_ids=(chunk_id,),
                            confidence=1.0,
                            provenance=Provenance(
                                dataset_id=self.dataset_id,
                                snapshot_id=snapshot_id,
                                document_id=document_id,
                                chunk_id=chunk_id,
                                stage="scierc-gold-relation",
                                producer_version=self.adapter_version,
                                content_sha256=_sha256_text(triple_id),
                            ),
                        )
                    )
            rows_by_doc_key[doc_key] = {
                "split": split,
                "document_id": document_id,
                "sentences": sentence_index_data,
                "offsets": offsets,
            }

        relations = self._build_relations(snapshot_id)
        provisional = {
            "documents": tuple(sorted(documents, key=lambda value: value.document_id)),
            "chunks": tuple(sorted(chunks, key=lambda value: value.chunk_id)),
            "entities": tuple(sorted(entities, key=lambda value: value.entity_id)),
            "relations": relations,
            "triples": tuple(sorted(triples, key=lambda value: value.triple_id)),
            "split_membership": tuple(
                sorted(memberships, key=lambda value: value.item_id)
            ),
        }
        descriptor = DatasetDescriptor(
            dataset_id=self.dataset_id,
            snapshot_id=snapshot_id,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            license=(
                "Repository-authored synthetic fixture"
                if self.fixture_mode
                else "No explicit SciERC redistribution license found; review required"
            ),
            citation="Luan et al. (2018), Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction",
            language="en",
            capabilities=capability_values(
                (
                    Capability.DOCUMENTS,
                    Capability.DOCUMENT_CLUSTERS,
                    Capability.GOLD_ENTITIES,
                    Capability.GOLD_RELATIONS,
                    Capability.GOLD_TRIPLES,
                    Capability.SOURCE_SPANS,
                    *(
                        (Capability.PREDICTED_GRAPH,)
                        if self.prediction_ledger is not None
                        and self.prediction_ledger.is_file()
                        else ()
                    ),
                )
            ),
            raw_content_sha256=raw_hash,
            canonical_content_sha256=content_sha256(provisional),
            redistribution=(
                "included synthetic fixture"
                if self.fixture_mode
                else "blocked pending explicit license or permission"
            ),
        )
        self._rows_by_doc_key = rows_by_doc_key
        self._bundle = CanonicalBundle(
            descriptor=descriptor,
            documents=provisional["documents"],
            chunks=provisional["chunks"],
            entities=provisional["entities"],
            relations=relations,
            triples=provisional["triples"],
            questions=(),
            answers=(),
            evidence_sets=(),
            split_membership=provisional["split_membership"],
        )
        return self._bundle

    def _prediction_rows(self) -> tuple[dict[str, Any], ...]:
        if self.prediction_ledger is None or not self.prediction_ledger.is_file():
            return ()
        with self.prediction_ledger.open(encoding="utf-8") as handle:
            rows = tuple(json.loads(line) for line in handle if line.strip())
        for row in rows:
            if row.get("schema_version") != "rag-extraction-prediction-1.0":
                raise ValueError("SciERC prediction ledger has an unsupported schema")
            if row.get("record_type") not in {"entity", "relation"}:
                raise ValueError("SciERC prediction row has an invalid record_type")
        return rows

    def _prediction_span(
        self,
        doc_key: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> tuple[str, str, int, int, str]:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"unknown predicted SciERC entity type: {entity_type}")
        document = self._rows_by_doc_key.get(doc_key)
        if document is None:
            raise ValueError(f"prediction references unknown SciERC document: {doc_key}")
        sentence_indexes = [
            index
            for index, sentence in enumerate(document["sentences"])
            if sentence["token_offset"] <= start
            and end < document["offsets"][index + 1]
        ]
        if len(sentence_indexes) != 1:
            raise ValueError("predicted SciERC entity crosses or escapes a sentence")
        sentence_index = sentence_indexes[0]
        sentence = document["sentences"][sentence_index]
        local_start = start - sentence["token_offset"]
        local_end = end - sentence["token_offset"]
        if not (0 <= local_start <= local_end < len(sentence["words"])):
            raise ValueError("predicted SciERC entity has invalid token bounds")
        char_start = sentence["token_char_spans"][local_start][0]
        char_end = sentence["token_char_spans"][local_end][1]
        surface = " ".join(sentence["words"])[char_start:char_end]
        return sentence["chunk_id"], surface, char_start, char_end, document["document_id"]

    def validate(self) -> ValidationReport:
        issues = []
        for split, path in self._split_paths().items():
            if not path.is_file():
                issues.append(
                    ValidationIssue(
                        "missing-scierc-split",
                        "error",
                        f"pinned SciERC {split} split is absent",
                        path.as_posix(),
                    )
                )
                continue
            observed = file_sha256(path)
            expected = self.expected_split_sha256[f"{split}.json"]
            if observed != expected:
                issues.append(
                    ValidationIssue(
                        "scierc-split-hash-mismatch",
                        "error",
                        f"SciERC {split} SHA-256 {observed} != {expected}",
                        path.as_posix(),
                    )
                )
        if not any(item.severity == "error" for item in issues):
            try:
                bundle = self._build_bundle()
                self._prediction_rows()
                if (
                    self.prediction_ledger is not None
                    and self.prediction_ledger.is_file()
                ):
                    self.generated_graph(bundle)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
            ) as error:
                issues.append(
                    ValidationIssue(
                        "invalid-scierc-content",
                        "error",
                        str(error),
                        self.data_root.as_posix(),
                    )
                )
        if self.prediction_ledger is None or not self.prediction_ledger.is_file():
            issues.append(
                ValidationIssue(
                    "missing-scierc-predictions",
                    "block",
                    "a SciERC-specific prediction ledger/checkpoint run is unavailable",
                    (
                        None
                        if self.prediction_ledger is None
                        else self.prediction_ledger.as_posix()
                    ),
                )
            )
        if not self.fixture_mode:
            issues.append(
                ValidationIssue(
                    "scierc-redistribution-license-unresolved",
                    "warning",
                    "raw/text-bearing artifacts cannot be packaged until explicit redistribution permission is established",
                    self.data_root.as_posix(),
                )
            )
        issues.append(
            ValidationIssue(
                "scierc-qa-not-applicable",
                "info",
                "SciERC has no independent questions, answers, or evidence judgments; Graph RAG QA is out of scope",
            )
        )
        capabilities = {
            Capability.DOCUMENTS,
            Capability.DOCUMENT_CLUSTERS,
            Capability.GOLD_ENTITIES,
            Capability.GOLD_RELATIONS,
            Capability.GOLD_TRIPLES,
            Capability.SOURCE_SPANS,
        }
        if (
            self.prediction_ledger is not None
            and self.prediction_ledger.is_file()
            and not any(item.severity == "error" for item in issues)
        ):
            capabilities.add(Capability.PREDICTED_GRAPH)
        return ValidationReport(
            adapter_id=self.adapter_id,
            ready=not any(item.severity == "error" for item in issues),
            issues=tuple(issues),
            capabilities=capability_values(capabilities),
        )

    def load(self) -> CanonicalBundle:
        report = self.validate()
        if not report.ready:
            raise ValueError("SciERC adapter cannot load until split validation passes")
        return self._build_bundle()

    def generated_graph(self, bundle: CanonicalBundle) -> GeneratedGraph:
        rows = self._prediction_rows()
        if not rows:
            return GeneratedGraph(
                entities=(),
                relations=(),
                triples=(),
                extractor_id="scierc-dataset-specific-checkpoint-unavailable",
                construction_recipe="blocked-no-prediction-ledger-v1",
            )
        if not self._rows_by_doc_key:
            self._build_bundle()
        snapshot_id = bundle.descriptor.snapshot_id
        entities = []
        entity_ids: dict[tuple[str, int, int, str], str] = {}
        entity_chunk_ids: dict[tuple[str, int, int, str], str] = {}
        for row in rows:
            if row["record_type"] != "entity":
                continue
            required = {
                "doc_key",
                "start",
                "end",
                "entity_type",
                "confidence",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"SciERC predicted entity is missing keys: {missing}")
            if set(row) != required | {"schema_version", "record_type"}:
                raise ValueError("SciERC predicted entity contains unknown keys")
            doc_key = row["doc_key"]
            start = int(row["start"])
            end = int(row["end"])
            entity_type = row["entity_type"]
            confidence = row["confidence"]
            if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
                raise ValueError("SciERC predicted entity confidence must be in [0, 1]")
            key = (doc_key, start, end, entity_type)
            if key in entity_ids:
                raise ValueError(f"duplicate SciERC predicted entity: {key}")
            chunk_id, surface, char_start, char_end, document_id = self._prediction_span(
                doc_key,
                start,
                end,
                entity_type,
            )
            entity_id = _entity_id(doc_key, start, end, entity_type)
            entity_ids[key] = entity_id
            entity_chunk_ids[key] = chunk_id
            entities.append(
                Entity(
                    dataset_id=self.dataset_id,
                    entity_id=entity_id,
                    surface_forms=(surface,),
                    canonical_label=surface.casefold(),
                    entity_type=entity_type,
                    source_spans=((chunk_id, char_start, char_end),),
                    provenance=Provenance(
                        dataset_id=self.dataset_id,
                        snapshot_id=snapshot_id,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        span_start=char_start,
                        span_end=char_end,
                        stage="scierc-predicted-entity",
                        producer_version=self.adapter_version,
                        content_sha256=content_sha256(row),
                    ),
                )
            )
        triples = []
        for row in rows:
            if row["record_type"] != "relation":
                continue
            required = {
                "doc_key",
                "sentence_index",
                "head",
                "tail",
                "relation_type",
                "confidence",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"SciERC predicted relation is missing keys: {missing}")
            if set(row) != required | {"schema_version", "record_type"}:
                raise ValueError("SciERC predicted relation contains unknown keys")
            relation_type = row["relation_type"]
            if relation_type not in RELATION_TYPES:
                raise ValueError(
                    f"unknown predicted SciERC relation type: {relation_type}"
                )
            doc_key = row["doc_key"]
            head = row["head"]
            tail = row["tail"]
            endpoint_keys = {"start", "end", "entity_type"}
            if not isinstance(head, dict) or set(head) != endpoint_keys:
                raise ValueError("SciERC predicted relation head is malformed")
            if not isinstance(tail, dict) or set(tail) != endpoint_keys:
                raise ValueError("SciERC predicted relation tail is malformed")
            head_key = (
                doc_key,
                int(head["start"]),
                int(head["end"]),
                head["entity_type"],
            )
            tail_key = (
                doc_key,
                int(tail["start"]),
                int(tail["end"]),
                tail["entity_type"],
            )
            if head_key not in entity_ids or tail_key not in entity_ids:
                raise ValueError(
                    "SciERC predicted relation endpoint is not a declared predicted entity"
                )
            confidence = row["confidence"]
            if confidence is not None:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        "SciERC predicted relation confidence must be in [0, 1]"
                    )
            sentence_index = int(row["sentence_index"])
            chunk_id = _chunk_id(doc_key, sentence_index)
            if (
                entity_chunk_ids[head_key] != chunk_id
                or entity_chunk_ids[tail_key] != chunk_id
            ):
                raise ValueError(
                    "SciERC predicted relation endpoints do not belong to sentence_index"
                )
            triple_id = _triple_id(
                doc_key,
                sentence_index,
                head_key[1],
                head_key[2],
                tail_key[1],
                tail_key[2],
                relation_type,
            )
            if any(item.triple_id == triple_id for item in triples):
                raise ValueError(f"duplicate SciERC predicted relation: {triple_id}")
            triples.append(
                Triple(
                    dataset_id=self.dataset_id,
                    triple_id=triple_id,
                    head_id=entity_ids[head_key],
                    relation_id=_relation_id(relation_type),
                    tail_id=entity_ids[tail_key],
                    chunk_ids=(chunk_id,),
                    confidence=confidence,
                    provenance=Provenance(
                        dataset_id=self.dataset_id,
                        snapshot_id=snapshot_id,
                        document_id=_document_id(doc_key),
                        chunk_id=chunk_id,
                        stage="scierc-predicted-relation",
                        producer_version=self.adapter_version,
                        content_sha256=content_sha256(row),
                    ),
                )
            )
        ledger_hash = file_sha256(self.prediction_ledger)
        return GeneratedGraph(
            entities=tuple(entities),
            relations=self._build_relations(snapshot_id),
            triples=tuple(triples),
            extractor_id=f"scierc-prediction-ledger-sha256:{ledger_hash}",
            construction_recipe="native-document-global-spans-to-canonical-graph-v1",
        )
