"""Regime-D Graph RAG diagnostic over one immutable, scored Phase B run."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Iterable, Mapping
import hashlib
import json
import os
import platform
import re
import sys
import tempfile

from graph_rag_eval.budget import Budget, assert_matched_budgets
from graph_rag_eval.contracts import (
    AnswerAlias,
    Chunk,
    Document,
    Entity,
    EvidenceSet,
    Provenance,
    Question,
    Relation,
    Triple,
    canonical_data,
    canonical_json,
    content_sha256,
)
from graph_rag_eval.evaluation.statistics import (
    PairedObservation,
    clustered_paired_bootstrap,
    percentile,
)
from graph_rag_eval.graphs.snapshots import (
    GraphSnapshot,
    build_snapshot,
    structure_summary,
)
from graph_rag_eval.identity import (
    file_sha256,
    fingerprint,
    source_surface_manifest,
    source_surface_sha256,
)
from graph_rag_eval.phase_b_artifacts import (
    PhaseBArtifactError,
    PhaseBRunData,
    StrictKey,
    load_phase_b_run,
    strict_counts,
    strict_triple_from_key,
)
from graph_rag_eval.retrieval.bm25 import BM25Retriever
from graph_rag_eval.retrieval.graph import GraphRetriever
from graph_rag_eval.retrieval.hybrid import ReciprocalRankFusionRetriever
from graph_rag_eval.trace import (
    OutputContainmentError,
    resolve_within,
    validate_schema,
)


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class PhaseBDiagnosticError(ValueError):
    """Raised before overwrite or scientific execution on a contract failure."""


@dataclass(frozen=True)
class DiagnosticContext:
    source_root: Path
    config_path: Path
    config: dict[str, Any]
    run_id: str
    parent_root: Path
    child_root: Path


@dataclass(frozen=True)
class ProbeTarget:
    question: Question
    answer: AnswerAlias
    evidence: EvidenceSet
    strict_key: StrictKey
    gold_triple_id: str
    chunk_id: str
    cluster_id: str
    source_document_id: str
    relation: str
    head: str
    tail: str


@dataclass(frozen=True)
class CanonicalDiagnostic:
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]
    probes: tuple[ProbeTarget, ...]
    exclusions: tuple[dict[str, Any], ...]
    chunk_by_example: Mapping[str, str]


def _load_config(path: Path, source_root: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else source_root / path
    resolved = resolved.resolve()
    if source_root != resolved and source_root not in resolved.parents:
        raise PhaseBDiagnosticError("diagnostic config must remain inside src")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhaseBDiagnosticError("diagnostic config is unreadable") from error
    schema = json.loads(
        (
            source_root
            / "schemas"
            / "phase_b"
            / "rag-phase-b-diagnostic-config.schema.json"
        ).read_text(encoding="utf-8")
    )
    validate_schema(value, schema)
    expected_seeds = tuple(range(42, 50))
    if tuple(value["parent"]["training_seeds"]) != expected_seeds:
        raise PhaseBDiagnosticError("canonical diagnostic requires ordered seeds 42-49")
    if value["probe"]["public_query_fields"] != ["text", "public_metadata"]:
        raise PhaseBDiagnosticError("public query projection must be text + public_metadata")
    expected_conditions = {
        "text_bm25",
        "confidence_filtered_graph",
        "confidence_filtered_hybrid",
        "simple_verifier_graph",
        "simple_verifier_hybrid",
        "corrective_verifier_graph",
        "corrective_verifier_hybrid",
        "gold_graph_oracle",
    }
    if set(value["retrieval"]["conditions"]) != expected_conditions:
        raise PhaseBDiagnosticError("diagnostic retrieval condition set differs from C-04")
    confidence = float(value["statistics"]["confidence"])
    if not 0.0 < confidence < 1.0:
        raise PhaseBDiagnosticError("bootstrap confidence must be between zero and one")
    if set(value["probe"]["forbidden_query_fields"]) != {
        "answer",
        "gold_tail",
        "gold_relation",
        "gold_strict_key",
        "source_document_id",
        "evidence_id",
        "training_seed",
        "condition",
    }:
        raise PhaseBDiagnosticError("private query-field sentinel differs from C-04")
    return value


def create_phase_b_diagnostic_context(
    config_path: str | Path,
    *,
    run_id: str,
    source_root: str | Path | None = None,
) -> DiagnosticContext:
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    if not RUN_ID_RE.fullmatch(run_id):
        raise PhaseBDiagnosticError("invalid Phase B run ID")
    config_file = Path(config_path)
    config = _load_config(config_file, root)
    if not config_file.is_absolute():
        config_file = root / config_file
    unresolved_parent = root / "output" / run_id
    if unresolved_parent.is_symlink():
        raise OutputContainmentError("Phase B run root must not be a symlink")
    parent = unresolved_parent.resolve()
    output_root = (root / "output").resolve()
    if output_root not in parent.parents:
        raise OutputContainmentError("Phase B run escapes src/output")
    child = (unresolved_parent / "graph-rag").resolve()
    if child.parent != parent:
        raise OutputContainmentError("Graph RAG child is not nested under its Phase B run")
    return DiagnosticContext(
        source_root=root,
        config_path=config_file.resolve(),
        config=config,
        run_id=run_id,
        parent_root=parent,
        child_root=child,
    )


def _schema_manifest(source_root: Path) -> tuple[dict[str, Any], ...]:
    schema_root = source_root / "schemas" / "phase_b"
    return tuple(
        {
            "path": path.relative_to(source_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(schema_root.glob("rag-*.schema.json"))
    )


def _diagnostic_output_schema(source_root: Path) -> dict[str, Any]:
    return json.loads(
        (
            source_root
            / "schemas"
            / "phase_b"
            / "rag-phase-b-diagnostic-output.schema.json"
        ).read_text(encoding="utf-8")
    )


def _runtime_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_package = str(config["retrieval"]["bm25"]["package"])
    package_name, separator, expected_version = expected_package.partition("==")
    if not separator or not package_name or not expected_version:
        raise PhaseBDiagnosticError("BM25 package identity must use name==version")
    try:
        observed_version = metadata.version(package_name)
    except metadata.PackageNotFoundError as error:
        raise PhaseBDiagnosticError(
            f"required retrieval package is unavailable: {package_name}"
        ) from error
    if observed_version != expected_version:
        raise PhaseBDiagnosticError(
            f"retrieval package version mismatch: expected {expected_package}, "
            f"observed {package_name}=={observed_version}"
        )
    return {
        "schema_version": "rag-phase-b-diagnostic-environment-1.0",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_basename": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {package_name: observed_version},
        "network_used": False,
        "model_calls": 0,
    }


def _expected_identity(
    context: DiagnosticContext,
    parent: PhaseBRunData,
) -> dict[str, Any]:
    schemas = _schema_manifest(context.source_root)
    source = source_surface_manifest(context.source_root)
    return {
        "schema_version": "rag-phase-b-child-identity-1.0",
        "run_id": context.run_id,
        "protocol_id": context.config["protocol"]["protocol_id"],
        "parent": {
            "protocol_id": parent.protocol_id,
            "workflow_id": parent.workflow_id,
            "matcher_id": parent.matcher_id,
            "score_manifest_sha256": parent.score_manifest_sha256,
            "artifact_set_sha256": parent.artifact_set_sha256,
        },
        "config": {
            "path": context.config_path.relative_to(context.source_root).as_posix(),
            "sha256": file_sha256(context.config_path),
        },
        "source_surface_sha256": source_surface_sha256(context.source_root),
        "source_surface": list(source),
        "schema_set_sha256": content_sha256(schemas),
        "schemas": list(schemas),
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            canonical_data(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Iterable[Any]) -> bytes:
    lines = [
        json.dumps(
            canonical_data(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for value in values
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _child_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or value.is_absolute()
        or ".." in value.parts
    ):
        raise PhaseBDiagnosticError(f"invalid child artifact path: {relative!r}")
    unresolved = root.joinpath(*value.parts)
    if unresolved.is_symlink() or any(
        item.is_symlink()
        for item in unresolved.parents
        if item != root.parent
    ):
        raise PhaseBDiagnosticError(f"child artifact traverses a symlink: {relative}")
    return resolve_within(root, relative)


def _write_once(root: Path, relative: str, payload: bytes) -> dict[str, Any]:
    path = _child_path(root, relative)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise PhaseBDiagnosticError(
                f"occupied child artifact is not a regular file: {relative}"
            )
        if path.read_bytes() != payload:
            raise PhaseBDiagnosticError(
                f"resume refuses to overwrite differing child artifact: {relative}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_name, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise PhaseBDiagnosticError(
                        f"concurrent child artifact differs: {relative}"
                    )
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _write_json(root: Path, relative: str, value: Any) -> dict[str, Any]:
    return _write_once(root, relative, _json_bytes(value))


def _write_jsonl(
    root: Path,
    relative: str,
    values: Iterable[Any],
) -> dict[str, Any]:
    return _write_once(root, relative, _jsonl_bytes(values))


def _write_text(root: Path, relative: str, value: str) -> dict[str, Any]:
    return _write_once(root, relative, value.encode("utf-8"))


def _bind_child(
    context: DiagnosticContext,
    parent: PhaseBRunData,
) -> dict[str, Any]:
    expected = _expected_identity(context, parent)
    identity_path = context.child_root / "manifests" / "run-identity.json"
    if context.child_root.exists():
        if context.child_root.is_symlink() or not context.child_root.is_dir():
            raise PhaseBDiagnosticError("Graph RAG child root is not a real directory")
        if not identity_path.is_file() or identity_path.is_symlink():
            raise PhaseBDiagnosticError(
                "occupied Graph RAG child has no immutable identity manifest"
            )
        try:
            observed = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PhaseBDiagnosticError("Graph RAG child identity is unreadable") from error
        schema = json.loads(
            (
                context.source_root
                / "schemas"
                / "phase_b"
                / "rag-phase-b-child-identity.schema.json"
            ).read_text(encoding="utf-8")
        )
        validate_schema(observed, schema)
        if canonical_json(observed) != canonical_json(expected):
            raise PhaseBDiagnosticError(
                "Graph RAG child identity differs from parent/config/source/schema"
            )
    else:
        context.child_root.mkdir(parents=False, exist_ok=False)
        _write_json(
            context.child_root,
            "manifests/run-identity.json",
            expected,
        )
    return expected


def _verify_child_file_set(
    context: DiagnosticContext,
    artifacts: Iterable[Mapping[str, Any]],
    *,
    completed: bool,
) -> None:
    expected = {
        "manifests/run-identity.json",
        "manifests/artifacts.json",
        *(str(item["path"]) for item in artifacts),
    }
    if completed:
        expected.add("manifests/complete.json")
    observed = set()
    for path in context.child_root.rglob("*"):
        if path.is_symlink():
            raise PhaseBDiagnosticError("Graph RAG child tree contains a symlink")
        if path.is_file():
            observed.add(path.relative_to(context.child_root).as_posix())
    if observed != expected:
        raise PhaseBDiagnosticError(
            "Graph RAG child file set differs from its artifact contract"
        )


def _verify_completed_child(context: DiagnosticContext) -> dict[str, Any] | None:
    complete_path = context.child_root / "manifests" / "complete.json"
    if not complete_path.exists():
        return None
    if complete_path.is_symlink() or not complete_path.is_file():
        raise PhaseBDiagnosticError(
            "Graph RAG completion manifest is not a regular file"
        )
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhaseBDiagnosticError(
            "Graph RAG completion manifest is unreadable"
        ) from error
    if (
        complete.get("schema_version") != "rag-phase-b-diagnostic-complete-1.0"
        or complete.get("run_id") != context.run_id
        or complete.get("status") != "diagnostic_complete"
    ):
        raise PhaseBDiagnosticError("Graph RAG completion manifest is invalid")
    validate_schema(complete, _diagnostic_output_schema(context.source_root))
    artifact_path = context.child_root / "manifests" / "artifacts.json"
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise PhaseBDiagnosticError(
            "Graph RAG artifact manifest is not a regular file"
        )
    artifact_digest = file_sha256(artifact_path)
    if artifact_digest != complete.get("artifact_manifest_sha256"):
        raise PhaseBDiagnosticError("Graph RAG artifact manifest hash mismatch")
    try:
        artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhaseBDiagnosticError(
            "Graph RAG artifact manifest is unreadable"
        ) from error
    artifacts = artifact_manifest.get("artifacts")
    if (
        artifact_manifest.get("schema_version")
        != "rag-phase-b-diagnostic-artifacts-1.0"
        or artifact_manifest.get("run_id") != context.run_id
        or not isinstance(artifacts, list)
        or len(artifacts) != complete.get("artifact_count")
    ):
        raise PhaseBDiagnosticError("Graph RAG artifact manifest is invalid")
    validate_schema(
        artifact_manifest,
        _diagnostic_output_schema(context.source_root),
    )
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("bytes"), int)
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise PhaseBDiagnosticError(
                "Graph RAG artifact descriptor is invalid"
            )
        path = _child_path(context.child_root, artifact["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or file_sha256(path) != artifact["sha256"]
        ):
            raise PhaseBDiagnosticError(
                f"completed Graph RAG artifact is missing or changed: {artifact['path']}"
            )
    _verify_child_file_set(context, artifacts, completed=True)
    return complete


def _provenance(
    parent: PhaseBRunData,
    *,
    stage: str,
    payload: Any,
    document_id: str | None = None,
    chunk_id: str | None = None,
    span: tuple[int, int] | None = None,
    parents: tuple[str, ...] = (),
) -> Provenance:
    return Provenance(
        dataset_id=parent.dataset_id,
        snapshot_id=f"{parent.run_id}:CODE-SPLIT-1:test",
        stage=stage,
        producer_version="C-04-PHASE-B-REGIME-D-1.0",
        content_sha256=content_sha256(payload),
        document_id=document_id,
        chunk_id=chunk_id,
        span_start=span[0] if span else None,
        span_end=span[1] if span else None,
        parent_ids=parents,
    )


def _chunk_id(example_id: str) -> str:
    return "chunk-" + fingerprint("phase-b-example-chunk-v1", example_id)


def _entity_id(key: StrictKey, endpoint: str) -> str:
    if endpoint == "head":
        payload = (key[0], key[1], key[2], key[3])
    else:
        payload = (key[0], key[5], key[6], key[7])
    return "entity-" + fingerprint("phase-b-strict-entity-v1", payload)


def _relation_id(label: str) -> str:
    return "relation-" + fingerprint("phase-b-strict-relation-v1", label)


def _triple_id(key: StrictKey) -> str:
    return "triple-" + fingerprint("phase-b-strict-triple-v1", key)


def _build_canonical(parent: PhaseBRunData, config: Mapping[str, Any]) -> CanonicalDiagnostic:
    chunk_by_example = {
        example_id: _chunk_id(example_id) for example_id in sorted(parent.prepared)
    }
    chunks = []
    for example_id, prepared in sorted(parent.prepared.items()):
        text = prepared.content
        chunk_id = chunk_by_example[example_id]
        chunks.append(
            Chunk(
                dataset_id=parent.dataset_id,
                chunk_id=chunk_id,
                document_id=prepared.source_document_id,
                text=text,
                char_start=0,
                char_end=len(text),
                token_start=0,
                token_end=len(prepared.words),
                chunker_id="phase-b-prepared-sentence-v1",
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                provenance=_provenance(
                    parent,
                    stage="phase-b-prepared-test",
                    payload={
                        "example_id": example_id,
                        "text": text,
                    },
                    document_id=prepared.source_document_id,
                    chunk_id=chunk_id,
                    parents=(example_id,),
                ),
            )
        )
    chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk)
    documents = []
    for document_id, document_chunks in sorted(chunks_by_document.items()):
        text = "\n".join(item.text for item in sorted(document_chunks, key=lambda x: x.chunk_id))
        documents.append(
            Document(
                dataset_id=parent.dataset_id,
                document_id=document_id,
                text=text,
                split_id=config["protocol"]["evaluation_split_id"],
                cluster_id="cluster-" + fingerprint(
                    "phase-b-source-document-cluster-v1", document_id
                ),
                provenance=_provenance(
                    parent,
                    stage="phase-b-source-document",
                    payload={"document_id": document_id, "text": text},
                    document_id=document_id,
                    parents=tuple(item.chunk_id for item in document_chunks),
                ),
            )
        )

    probes = []
    exclusions = []
    template = config["probe"]["template"]
    condition_values = tuple(config["conditions"].values())
    query_occurrences: dict[str, int] = defaultdict(int)
    for example_id, gold in sorted(parent.gold.items()):
        prepared = parent.prepared[example_id]
        chunk_id = chunk_by_example[example_id]
        cluster_id = "cluster-" + fingerprint(
            "phase-b-source-document-cluster-v1", prepared.source_document_id
        )
        for triple in sorted(gold.triples):
            key = triple.key()
            head = triple.head.text or prepared.span_text(triple.head.start, triple.head.end)
            tail = triple.tail.text or prepared.span_text(triple.tail.start, triple.tail.end)
            relation = triple.relation
            triple_id = _triple_id(key)
            query = template.format(head=head)
            normalized_query = " ".join(query.casefold().split())
            occurrence = query_occurrences[normalized_query]
            query_occurrences[normalized_query] += 1
            question_id = "probe-" + fingerprint(
                "phase-b-regime-d-public-probe-v1",
                {
                    "template_id": config["probe"]["template_id"],
                    "normalized_query": normalized_query,
                    "occurrence": occurrence,
                },
            )
            normalized_head = " ".join(head.casefold().split())
            normalized_tail = " ".join(tail.casefold().split())
            private_values = (
                normalized_tail,
                relation.casefold(),
                prepared.source_document_id.casefold(),
                chunk_id.casefold(),
                triple_id.casefold(),
                *(item.casefold() for item in condition_values),
            )
            reason = None
            if not normalized_head or not normalized_tail:
                reason = "empty_head_or_tail"
            elif normalized_head == normalized_tail:
                reason = "head_tail_surface_collision"
            elif any(value and value in normalized_query for value in private_values):
                reason = "private_value_surface_collision"
            if reason is not None:
                exclusions.append(
                    {
                        "schema_version": "rag-phase-b-probe-exclusion-1.0",
                        "question_id": question_id,
                        "reason": reason,
                        "private_target_sha256": content_sha256(
                            {
                                "strict_key": key,
                                "source_document_id": prepared.source_document_id,
                            }
                        ),
                    }
                )
                continue
            question = Question(
                dataset_id=parent.dataset_id,
                question_id=question_id,
                text=query,
                split_id=config["protocol"]["evaluation_split_id"],
                cluster_id=cluster_id,
                regime="diagnostic_fact_probe",
                authoring_provenance="mechanical-private-gold-head-only-v1",
                validation_provenance="C-04-leakage-sentinel-v1",
                public_metadata={"template_id": config["probe"]["template_id"]},
            )
            answer = AnswerAlias(
                dataset_id=parent.dataset_id,
                question_id=question_id,
                normalized_answers=(tail,),
                answer_type=triple.tail.entity_type,
            )
            evidence = EvidenceSet(
                dataset_id=parent.dataset_id,
                question_id=question_id,
                evidence_set_id="evidence-set-" + fingerprint(
                    "phase-b-probe-evidence-v1", key
                ),
                evidence_ids=(chunk_id, triple_id),
                relevance_grade=1,
                annotation_provenance="private-phase-b-gold-and-prepared-source-v1",
            )
            probes.append(
                ProbeTarget(
                    question=question,
                    answer=answer,
                    evidence=evidence,
                    strict_key=key,
                    gold_triple_id=triple_id,
                    chunk_id=chunk_id,
                    cluster_id=cluster_id,
                    source_document_id=prepared.source_document_id,
                    relation=relation,
                    head=head,
                    tail=tail,
                )
            )
    probes.sort(key=lambda item: item.question.question_id)
    exclusions.sort(key=lambda item: item["question_id"])
    if not probes:
        raise PhaseBDiagnosticError("no leakage-safe Regime-D probes are available")
    return CanonicalDiagnostic(
        documents=tuple(documents),
        chunks=tuple(sorted(chunks, key=lambda item: item.chunk_id)),
        probes=tuple(probes),
        exclusions=tuple(exclusions),
        chunk_by_example=chunk_by_example,
    )


def _graph_from_keys(
    parent: PhaseBRunData,
    canonical: CanonicalDiagnostic,
    keys: Iterable[StrictKey],
    *,
    condition: str,
    extractor_identity: Any,
    input_identity: Mapping[str, Any],
    recipe: str,
) -> GraphSnapshot:
    entities: dict[str, Entity] = {}
    relations: dict[str, Relation] = {}
    triples = []
    for key in sorted(set(keys)):
        strict = strict_triple_from_key(key, parent.prepared)
        prepared = parent.prepared[strict.example_id]
        chunk_id = canonical.chunk_by_example[strict.example_id]
        head_id = _entity_id(key, "head")
        tail_id = _entity_id(key, "tail")
        relation_id = _relation_id(strict.relation)
        for entity_id, span in ((head_id, strict.head), (tail_id, strict.tail)):
            char_span = prepared.span_char_range(span.start, span.end)
            candidate = Entity(
                dataset_id=parent.dataset_id,
                entity_id=entity_id,
                surface_forms=(span.text or prepared.span_text(span.start, span.end),),
                canonical_label=span.text or prepared.span_text(span.start, span.end),
                entity_type=span.entity_type,
                source_spans=((chunk_id, char_span[0], char_span[1]),),
                provenance=_provenance(
                    parent,
                    stage="phase-b-strict-entity",
                    payload=(strict.example_id, span.key()),
                    document_id=prepared.source_document_id,
                    chunk_id=chunk_id,
                    span=char_span,
                    parents=(strict.example_id,),
                ),
            )
            previous = entities.get(entity_id)
            if previous is not None and canonical_json(previous) != canonical_json(candidate):
                raise PhaseBDiagnosticError("entity identity collision during graph construction")
            entities[entity_id] = candidate
        if relation_id not in relations:
            relations[relation_id] = Relation(
                dataset_id=parent.dataset_id,
                relation_id=relation_id,
                label=strict.relation,
                direction="directed",
                provenance=_provenance(
                    parent,
                    stage="phase-b-strict-relation-vocabulary",
                    payload=strict.relation,
                ),
            )
        triples.append(
            Triple(
                dataset_id=parent.dataset_id,
                triple_id=_triple_id(key),
                head_id=head_id,
                relation_id=relation_id,
                tail_id=tail_id,
                chunk_ids=(chunk_id,),
                confidence=None,
                provenance=_provenance(
                    parent,
                    stage="phase-b-condition-emission",
                    payload={"condition": condition, "strict_key": key},
                    document_id=prepared.source_document_id,
                    chunk_id=chunk_id,
                    parents=(strict.example_id,),
                ),
            )
        )
    descriptor_hash = content_sha256(
        {
            "dataset_id": parent.dataset_id,
            "run_id": parent.run_id,
            "split": "CODE-SPLIT-1:test",
            **input_identity,
        }
    )
    corpus_hash = content_sha256(canonical.chunks)
    return build_snapshot(
        dataset_id=parent.dataset_id,
        condition=condition,
        entities=entities.values(),
        relations=relations.values(),
        triples=triples,
        descriptor_sha256=descriptor_hash,
        corpus_sha256=corpus_hash,
        adapter_sha256=fingerprint(
            "phase-b-output-adapter-v1",
            input_identity,
        ),
        extractor_sha256=fingerprint(
            "phase-b-condition-extractor-v1", extractor_identity
        ),
        construction_recipe=recipe,
    )


def build_phase_b_graphs(
    parent: PhaseBRunData,
    canonical: CanonicalDiagnostic,
    config: Mapping[str, Any],
) -> tuple[GraphSnapshot, dict[str, dict[int, GraphSnapshot]]]:
    input_identity = {
        key: {
            "path": config["artifacts"][config_key],
            "sha256": next(
                item.sha256
                for item in parent.verified_artifacts
                if item.path == config["artifacts"][config_key]
            ),
        }
        for key, config_key in (
            ("prepared", "prepared_test"),
            ("split_identity", "split_manifest"),
        )
    }
    gold_keys = {
        triple.key()
        for record in parent.gold.values()
        for triple in record.triples
    }
    gold = _graph_from_keys(
        parent,
        canonical,
        gold_keys,
        condition="gold_graph_oracle",
        extractor_identity={
            "private_gold_sha256": next(
                item.sha256
                for item in parent.verified_artifacts
                if item.path == config["artifacts"]["private_gold"]
            )
        },
        input_identity=input_identity,
        recipe="private-phase-b-gold-oracle-v1",
    )
    predicted: dict[str, dict[int, GraphSnapshot]] = {}
    for phase_condition, label in config["conditions"].items():
        predicted[label] = {}
        for seed in parent.seeds:
            keys = {
                key
                for example_keys in parent.emitted[phase_condition][seed].values()
                for key in example_keys
            }
            predicted[label][seed] = _graph_from_keys(
                parent,
                canonical,
                keys,
                condition=f"{label}:seed-{seed}",
                extractor_identity={
                    "phase_b_condition": phase_condition,
                    "training_seed": seed,
                    "candidate_sha256": next(
                        item.sha256
                        for item in parent.verified_artifacts
                        if item.path == config["artifacts"]["candidates"]
                    ),
                    "verdict_sha256": (
                        None
                        if phase_condition == "VER-CONFIDENCE"
                        else next(
                            item.sha256
                            for item in parent.verified_artifacts
                            if item.path
                            == config["artifacts"][
                                "simple_verdicts"
                                if phase_condition == "VER-SIMPLE"
                                else "corrective_verdicts"
                            ]
                        )
                    ),
                    "threshold": parent.selected_threshold,
                },
                input_identity=input_identity,
                recipe=(
                    "phase-b-confidence-threshold-emissions-v1"
                    if phase_condition == "VER-CONFIDENCE"
                    else "phase-b-simple-verdict-emissions-v1"
                    if phase_condition == "VER-SIMPLE"
                    else "phase-b-corrective-verdict-emissions-v1"
                ),
            )
    return gold, predicted


def _prf(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )
    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,
    }


def _intrinsic_rows(
    parent: PhaseBRunData,
    predicted: Mapping[str, Mapping[int, GraphSnapshot]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, int]]]]:
    counts_by_phase = strict_counts(parent.emitted, parent.gold)
    labels = config["conditions"]
    rows = []
    by_label = {}
    for phase_condition, by_seed in counts_by_phase.items():
        label = labels[phase_condition]
        by_label[label] = by_seed
        for seed, counts in by_seed.items():
            rows.append(
                {
                    "schema_version": "rag-phase-b-intrinsic-metrics-1.0",
                    "run_id": parent.run_id,
                    "regime": "diagnostic_fact_probe",
                    "condition": label,
                    "training_seed": seed,
                    "strict_triple": _prf(counts),
                    "structure": structure_summary(predicted[label][seed]),
                    "claims_boundary": config["protocol"]["claims_boundary"],
                }
            )
    return rows, by_label


def _score_result(
    result,
    probe: ProbeTarget,
    *,
    graph: GraphSnapshot | None,
) -> dict[str, Any]:
    ranked = tuple(result.items)
    fact_ranks = [
        index
        for index, item in enumerate(ranked, start=1)
        if item.evidence_id == probe.gold_triple_id
    ]
    source_ranks = [
        index
        for index, item in enumerate(ranked, start=1)
        if item.evidence_id == probe.chunk_id
        or probe.chunk_id in item.provenance_ids
    ]
    fact_available = graph is not None
    graph_ids = {item.triple_id for item in graph.triples} if graph is not None else set()
    return {
        "fact_preserved": (
            {"status": "available", "value": int(probe.gold_triple_id in graph_ids)}
            if fact_available
            else {
                "status": "not_applicable",
                "value": None,
                "reason": "text control has no predicted graph",
            }
        ),
        "fact_retrieved_at_k": (
            {
                "status": "available",
                "value": int(bool(fact_ranks)),
                "reciprocal_rank": 1.0 / min(fact_ranks) if fact_ranks else 0.0,
            }
            if fact_available
            else {
                "status": "not_applicable",
                "value": None,
                "reciprocal_rank": None,
                "reason": "text control cannot return a graph triple",
            }
        ),
        "source_support_at_k": {
            "status": "available",
            "value": int(bool(source_ranks)),
            "reciprocal_rank": 1.0 / min(source_ranks) if source_ranks else 0.0,
        },
        "support_or_fact_at_k": {
            "status": "available",
            "value": int(bool(fact_ranks or source_ranks)),
        },
    }


def _retrieval_matrix(
    parent: PhaseBRunData,
    canonical: CanonicalDiagnostic,
    gold: GraphSnapshot,
    predicted: Mapping[str, Mapping[int, GraphSnapshot]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    budget_config = config["retrieval"]["budget"]
    budget = Budget(
        max_items=int(budget_config["max_items"]),
        max_tokens=int(budget_config["max_tokens"]),
    )
    bm25_config = config["retrieval"]["bm25"]
    bm25 = BM25Retriever(
        canonical.chunks,
        k1=float(bm25_config["k1"]),
        b=float(bm25_config["b"]),
        method=bm25_config["method"],
        idf_method=bm25_config["idf_method"],
        token_pattern=bm25_config["token_pattern"],
    )
    max_hops = int(config["retrieval"]["graph"]["max_hops"])
    oracle = GraphRetriever(gold, max_hops=max_hops)
    rrf_k = int(config["retrieval"]["hybrid"]["rrf_k"])
    traces = []
    failures = []
    metrics = []
    for seed in parent.seeds:
        retrievers: dict[str, tuple[Any, GraphSnapshot | None]] = {
            "text_bm25": (bm25, None),
            "gold_graph_oracle": (oracle, gold),
        }
        for label, by_seed in predicted.items():
            graph = by_seed[seed]
            graph_retriever = GraphRetriever(graph, max_hops=max_hops)
            retrievers[f"{label}_graph"] = (graph_retriever, graph)
            retrievers[f"{label}_hybrid"] = (
                ReciprocalRankFusionRetriever(graph_retriever, bm25, rrf_k=rrf_k),
                graph,
            )
        if set(retrievers) != set(config["retrieval"]["conditions"]):
            raise PhaseBDiagnosticError("runtime retriever matrix differs from config")
        for probe in canonical.probes:
            query = probe.question.public_view()
            expected_metadata = {"template_id": config["probe"]["template_id"]}
            if dict(query.public_metadata) != expected_metadata:
                raise PhaseBDiagnosticError(
                    "tested query public metadata differs from the fixed template "
                    f"contract: {probe.question.question_id}"
                )
            # Opaque question IDs and JSON field names can contain incidental
            # substrings of short private values. The fixed metadata equality
            # above proves that metadata is config-derived; assess target
            # leakage only on the actual free-text retrieval feature.
            normalized_query = " ".join(query.text.casefold().split())
            private_values = (
                probe.tail,
                probe.relation,
                probe.source_document_id,
                probe.chunk_id,
                probe.gold_triple_id,
                *config["conditions"].values(),
            )
            if any(
                value
                and " ".join(str(value).casefold().split())
                in normalized_query
                for value in private_values
            ):
                raise PhaseBDiagnosticError(
                    f"private target reached tested query: {probe.question.question_id}"
                )
            decisions = []
            for condition in config["retrieval"]["conditions"]:
                retriever, graph = retrievers[condition]
                result = retriever.retrieve(query, budget)
                decisions.append(result.budget)
                traces.append(
                    {
                        "schema_version": "rag-phase-b-retrieval-trace-1.0",
                        "run_id": parent.run_id,
                        "regime": "diagnostic_fact_probe",
                        "training_seed": seed,
                        "question_id": query.question_id,
                        "condition": condition,
                        "query_sha256": content_sha256(query),
                        "query": canonical_data(query),
                        "index_fingerprint": result.index_fingerprint,
                        "ranked_evidence": canonical_data(result.items),
                        "budget": canonical_data(result.budget),
                        "failure": result.failure,
                        "seed_ids": list(result.seed_ids),
                        "expansion_ids": list(result.expansion_ids),
                    }
                )
                if result.failure:
                    failures.append(
                        {
                            "schema_version": "rag-phase-b-retrieval-failure-1.0",
                            "run_id": parent.run_id,
                            "training_seed": seed,
                            "question_id": query.question_id,
                            "condition": condition,
                            "failure": result.failure,
                            "denominator_retained": True,
                        }
                    )
                metrics.append(
                    {
                        "schema_version": "rag-phase-b-probe-metrics-1.0",
                        "run_id": parent.run_id,
                        "regime": "diagnostic_fact_probe",
                        "training_seed": seed,
                        "question_id": query.question_id,
                        "cluster_id": probe.cluster_id,
                        "condition": condition,
                        "metrics": _score_result(result, probe, graph=graph),
                        "failure": result.failure,
                        "realized_items": result.budget.realized_items,
                        "realized_tokens": result.budget.realized_tokens,
                    }
                )
            assert_matched_budgets(decisions)
    return traces, failures, metrics


def recompute_diagnostic_metrics(
    per_probe: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Independently aggregate stored per-probe numerators and denominators."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    failures: dict[str, list[float]] = defaultdict(list)
    for row in per_probe:
        condition = str(row["condition"])
        failures[condition].append(float(row.get("failure") is not None))
        metrics = row["metrics"]
        for metric in (
            "fact_preserved",
            "fact_retrieved_at_k",
            "source_support_at_k",
            "support_or_fact_at_k",
        ):
            value = metrics[metric]
            if value.get("status") == "available":
                grouped[(condition, metric)].append(float(value["value"]))
        reciprocal = metrics["fact_retrieved_at_k"]
        if reciprocal.get("status") == "available":
            grouped[(condition, "fact_reciprocal_rank")].append(
                float(reciprocal["reciprocal_rank"])
            )
        support_rr = metrics["source_support_at_k"]
        grouped[(condition, "source_support_reciprocal_rank")].append(
            float(support_rr["reciprocal_rank"])
        )
    rows = []
    for (condition, metric), values in sorted(grouped.items()):
        rows.append(
            {
                "schema_version": "rag-phase-b-aggregate-metrics-1.0",
                "condition": condition,
                "metric": metric,
                "value": sum(values) / len(values),
                "numerator": sum(values),
                "denominator": len(values),
                "status": "available",
            }
        )
    for condition, values in sorted(failures.items()):
        rows.append(
            {
                "schema_version": "rag-phase-b-aggregate-metrics-1.0",
                "condition": condition,
                "metric": "retrieval_failure_rate",
                "value": sum(values) / len(values),
                "numerator": sum(values),
                "denominator": len(values),
                "status": "available",
            }
        )
    return rows


def _per_seed_metrics(per_probe: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_probe:
        grouped[int(row["training_seed"])].append(row)
    return [
        {**row, "training_seed": seed}
        for seed, values in sorted(grouped.items())
        for row in recompute_diagnostic_metrics(values)
    ]


def _retrieval_comparisons(
    per_probe: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_key = {
        (row["training_seed"], row["question_id"], row["condition"]): row
        for row in per_probe
    }
    rows = []
    for comparison in config["statistics"]["comparison_family"]:
        baseline = comparison["baseline"]
        treatment = comparison["treatment"]
        for mode in ("graph", "hybrid"):
            for metric in ("fact_retrieved_at_k", "source_support_at_k"):
                observations = []
                for key, baseline_row in sorted(by_key.items()):
                    seed, question_id, condition = key
                    if condition != f"{baseline}_{mode}":
                        continue
                    treatment_row = by_key[
                        (seed, question_id, f"{treatment}_{mode}")
                    ]
                    left = baseline_row["metrics"][metric]
                    right = treatment_row["metrics"][metric]
                    if left["status"] == right["status"] == "available":
                        observations.append(
                            PairedObservation(
                                item_id=f"{seed}:{question_id}",
                                cluster_id=baseline_row["cluster_id"],
                                baseline=float(left["value"]),
                                treatment=float(right["value"]),
                            )
                        )
                rows.append(
                    {
                        "schema_version": "rag-phase-b-comparison-1.0",
                        "family": "confidence_referenced_phase_b_graph_conditions",
                        "baseline": f"{baseline}_{mode}",
                        "treatment": f"{treatment}_{mode}",
                        "metric": metric,
                        "bootstrap": clustered_paired_bootstrap(
                            observations,
                            resamples=int(
                                config["statistics"]["bootstrap_resamples"]
                            ),
                            seed=int(config["statistics"]["seed"]),
                            confidence=float(config["statistics"]["confidence"]),
                        ),
                        "p_value": None,
                        "p_value_status": "not_predeclared",
                    }
                )
    return rows


def _document_counts(
    parent: PhaseBRunData,
    config: Mapping[str, Any],
) -> dict[str, dict[int, dict[str, dict[str, int]]]]:
    result: dict[str, dict[int, dict[str, dict[str, int]]]] = {}
    for phase_condition, label in config["conditions"].items():
        result[label] = {}
        for seed in parent.seeds:
            result[label][seed] = {}
            for example_id, predicted in parent.emitted[phase_condition][seed].items():
                document_id = parent.gold[example_id].source_document_id
                counts = result[label][seed].setdefault(
                    document_id, {"tp": 0, "fp": 0, "fn": 0}
                )
                target = {item.key() for item in parent.gold[example_id].triples}
                counts["tp"] += len(predicted & target)
                counts["fp"] += len(predicted - target)
                counts["fn"] += len(target - predicted)
    return result


def _intrinsic_bootstrap(
    parent: PhaseBRunData,
    config: Mapping[str, Any],
    baseline: str,
    treatment: str,
) -> dict[str, Any]:
    by_document = _document_counts(parent, config)
    documents = sorted(
        {
            record.source_document_id
            for record in parent.gold.values()
        }
    )
    if len(documents) < 2:
        return {
            "status": "descriptive_only",
            "reason": "fewer than two source-document clusters",
        }
    import random

    def estimate(sampled: list[str]) -> float:
        values = []
        for seed in parent.seeds:
            condition_values = []
            for condition in (baseline, treatment):
                counts = {"tp": 0, "fp": 0, "fn": 0}
                for document_id in sampled:
                    row = by_document[condition][seed].get(
                        document_id, {"tp": 0, "fp": 0, "fn": 0}
                    )
                    for key in counts:
                        counts[key] += row[key]
                condition_values.append(_prf(counts)["f1"] or 0.0)
            values.append(condition_values[1] - condition_values[0])
        return mean(values)

    observed = estimate(documents)
    rng = random.Random(int(config["statistics"]["seed"]))
    resamples = int(config["statistics"]["bootstrap_resamples"])
    estimates = [
        estimate([rng.choice(documents) for _ in documents])
        for _ in range(resamples)
    ]
    confidence = float(config["statistics"]["confidence"])
    alpha = (1.0 - confidence) / 2.0
    return {
        "status": "available",
        "estimate": observed,
        "lower": percentile(estimates, alpha),
        "upper": percentile(estimates, 1.0 - alpha),
        "confidence": confidence,
        "resamples": resamples,
        "seed": int(config["statistics"]["seed"]),
        "clusters": len(documents),
        "training_seeds": len(parent.seeds),
    }


def _intrinsic_comparisons(
    parent: PhaseBRunData,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "rag-phase-b-comparison-1.0",
            "family": "confidence_referenced_phase_b_graph_conditions",
            "baseline": item["baseline"],
            "treatment": item["treatment"],
            "metric": "intrinsic_strict_triple_mean_seed_f1",
            "bootstrap": _intrinsic_bootstrap(
                parent,
                config,
                item["baseline"],
                item["treatment"],
            ),
            "p_value": None,
            "p_value_status": "not_predeclared",
        }
        for item in config["statistics"]["comparison_family"]
    ]


def _summary_table(
    aggregate: Iterable[Mapping[str, Any]],
    intrinsic: Iterable[Mapping[str, Any]],
) -> str:
    lines = [
        "scope\tcondition\tmetric\tvalue\tnumerator\tdenominator\tstatus"
    ]
    for row in aggregate:
        lines.append(
            "\t".join(
                (
                    "retrieval",
                    str(row["condition"]),
                    str(row["metric"]),
                    "" if row["value"] is None else f"{float(row['value']):.12g}",
                    "" if row["numerator"] is None else f"{float(row['numerator']):.12g}",
                    str(row["denominator"]),
                    str(row["status"]),
                )
            )
        )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in intrinsic:
        grouped[str(row["condition"])].append(row)
    for condition, rows in sorted(grouped.items()):
        for metric in ("precision", "recall", "f1"):
            values = [
                float(row["strict_triple"][metric])
                for row in rows
                if row["strict_triple"][metric] is not None
            ]
            lines.append(
                "\t".join(
                    (
                        "intrinsic",
                        condition,
                        f"mean_seed_{metric}",
                        "" if not values else f"{mean(values):.12g}",
                        "" if not values else f"{sum(values):.12g}",
                        str(len(values)),
                        "available" if values else "not_applicable",
                    )
                )
            )
    return "\n".join(lines) + "\n"


def _write_graph(
    root: Path,
    prefix: str,
    graph: GraphSnapshot,
) -> list[dict[str, Any]]:
    return [
        _write_json(root, f"{prefix}/manifest.json", graph),
        _write_jsonl(root, f"{prefix}/entities.jsonl", graph.entities),
        _write_jsonl(root, f"{prefix}/relations.jsonl", graph.relations),
        _write_jsonl(root, f"{prefix}/triples.jsonl", graph.triples),
    ]


def run_phase_b_diagnostic(context: DiagnosticContext) -> dict[str, Any]:
    """Validate parent first, then create/resume only its Graph RAG child."""

    try:
        parent = load_phase_b_run(
            context.parent_root,
            context.config,
            run_id=context.run_id,
        )
    except PhaseBArtifactError as error:
        raise PhaseBDiagnosticError(str(error)) from error
    environment = _runtime_contract(context.config)
    validate_schema(
        environment,
        _diagnostic_output_schema(context.source_root),
    )
    identity = _bind_child(context, parent)
    completed = _verify_completed_child(context)
    if completed is not None:
        return {
            **completed,
            "resume_status": "already_complete_verified",
        }

    canonical = _build_canonical(parent, context.config)
    gold, predicted = build_phase_b_graphs(parent, canonical, context.config)
    intrinsic, _ = _intrinsic_rows(parent, predicted, context.config)
    traces, failures, per_probe = _retrieval_matrix(
        parent,
        canonical,
        gold,
        predicted,
        context.config,
    )
    aggregate = recompute_diagnostic_metrics(per_probe)
    per_seed = _per_seed_metrics(per_probe)
    comparisons = _retrieval_comparisons(per_probe, context.config)
    comparisons.extend(_intrinsic_comparisons(parent, context.config))

    artifacts = []
    artifacts.append(
        _write_json(
            context.child_root,
            "manifests/config.json",
            {
                "schema_version": "rag-phase-b-diagnostic-config-manifest-1.0",
                "config_sha256": file_sha256(context.config_path),
                "config": context.config,
            },
        )
    )
    artifacts.append(
        _write_json(
            context.child_root,
            "manifests/environment.json",
            environment,
        )
    )
    artifacts.append(
        _write_json(
            context.child_root,
            "manifests/parent-artifacts.json",
            {
                "schema_version": "rag-phase-b-parent-artifacts-1.0",
                "run_id": parent.run_id,
                "score_manifest_sha256": parent.score_manifest_sha256,
                "artifact_set_sha256": parent.artifact_set_sha256,
                "artifacts": parent.verified_artifacts,
            },
        )
    )
    artifacts.append(
        _write_json(
            context.child_root,
            "manifests/protocol.json",
            {
                "schema_version": "rag-phase-b-diagnostic-protocol-1.0",
                **context.config["protocol"],
                "conditions": context.config["conditions"],
                "retrieval": context.config["retrieval"],
                "statistics": context.config["statistics"],
                "selected_phase_b_threshold": parent.selected_threshold,
                "eligible_probes": len(canonical.probes),
                "excluded_probes": len(canonical.exclusions),
                "training_seeds": list(parent.seeds),
            },
        )
    )
    artifacts.append(
        _write_jsonl(
            context.child_root,
            "canonical/documents.jsonl",
            canonical.documents,
        )
    )
    artifacts.append(
        _write_jsonl(context.child_root, "canonical/chunks.jsonl", canonical.chunks)
    )
    artifacts.append(
        _write_jsonl(
            context.child_root,
            "canonical/probes-public.jsonl",
            (item.question.public_view() for item in canonical.probes),
        )
    )
    artifacts.append(
        _write_jsonl(
            context.child_root,
            "private/probe-targets.jsonl",
            (
                {
                    "schema_version": "rag-phase-b-private-target-1.0",
                    "question_id": item.question.question_id,
                    "strict_key": item.strict_key,
                    "gold_triple_id": item.gold_triple_id,
                    "answer": item.tail,
                    "gold_relation": item.relation,
                    "source_document_id": item.source_document_id,
                    "evidence_id": item.chunk_id,
                    "cluster_id": item.cluster_id,
                    "scoring_only": True,
                }
                for item in canonical.probes
            ),
        )
    )
    artifacts.append(
        _write_jsonl(
            context.child_root,
            "private/probe-exclusions.jsonl",
            canonical.exclusions,
        )
    )
    artifacts.extend(
        _write_graph(context.child_root, "graphs/gold_graph_oracle", gold)
    )
    for condition, by_seed in sorted(predicted.items()):
        for seed, graph in sorted(by_seed.items()):
            artifacts.extend(
                _write_graph(
                    context.child_root,
                    f"graphs/{condition}/seed-{seed}",
                    graph,
                )
            )
    artifacts.append(
        _write_jsonl(
            context.child_root, "traces/retrieval.jsonl", traces
        )
    )
    artifacts.append(
        _write_jsonl(context.child_root, "traces/failures.jsonl", failures)
    )
    artifacts.append(
        _write_jsonl(
            context.child_root, "metrics/per-probe.jsonl", per_probe
        )
    )
    artifacts.append(
        _write_jsonl(
            context.child_root, "metrics/per-seed.jsonl", per_seed
        )
    )
    artifacts.append(
        _write_jsonl(
            context.child_root, "metrics/intrinsic.jsonl", intrinsic
        )
    )
    artifacts.append(
        _write_jsonl(
            context.child_root, "metrics/aggregate.jsonl", aggregate
        )
    )
    artifacts.append(
        _write_jsonl(
            context.child_root, "metrics/comparisons.jsonl", comparisons
        )
    )
    artifacts.append(
        _write_text(
            context.child_root,
            "tables/diagnostic-summary.tsv",
            _summary_table(aggregate, intrinsic),
        )
    )
    run_manifest = {
        "schema_version": "rag-phase-b-diagnostic-run-1.0",
        "run_id": parent.run_id,
        "stage": "phase_b_graph_rag_diagnostic",
        "status": "diagnostic_complete",
        "regime": "diagnostic_fact_probe",
        "parent_score_manifest_sha256": parent.score_manifest_sha256,
        "child_identity_sha256": content_sha256(identity),
        "eligible_probes": len(canonical.probes),
        "excluded_probes": len(canonical.exclusions),
        "training_seeds": list(parent.seeds),
        "retrieval_conditions": list(context.config["retrieval"]["conditions"]),
        "retrieval_observations": len(per_probe),
        "failures": len(failures),
        "scientific_claims_enabled": True,
        "claims_boundary": context.config["protocol"]["claims_boundary"],
        "regime_q_executed": False,
        "network_used": False,
        "model_calls": 0,
        "phase_b_parent_writes": 0,
    }
    validate_schema(
        run_manifest,
        _diagnostic_output_schema(context.source_root),
    )
    artifacts.append(
        _write_json(context.child_root, "manifests/run.json", run_manifest)
    )
    artifact_manifest = {
        "schema_version": "rag-phase-b-diagnostic-artifacts-1.0",
        "run_id": parent.run_id,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    validate_schema(
        artifact_manifest,
        _diagnostic_output_schema(context.source_root),
    )
    artifact_descriptor = _write_json(
        context.child_root,
        "manifests/artifacts.json",
        artifact_manifest,
    )
    _verify_child_file_set(context, artifacts, completed=False)
    complete = {
        "schema_version": "rag-phase-b-diagnostic-complete-1.0",
        "run_id": parent.run_id,
        "status": "diagnostic_complete",
        "regime": "diagnostic_fact_probe",
        "artifact_manifest_sha256": artifact_descriptor["sha256"],
        "artifact_count": len(artifacts),
        "eligible_probes": len(canonical.probes),
        "excluded_probes": len(canonical.exclusions),
        "retrieval_observations": len(per_probe),
        "claims_boundary": context.config["protocol"]["claims_boundary"],
    }
    validate_schema(
        complete,
        _diagnostic_output_schema(context.source_root),
    )
    _write_json(context.child_root, "manifests/complete.json", complete)
    return {**complete, "resume_status": "completed_or_resumed_partial"}
