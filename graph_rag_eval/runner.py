"""Standalone orchestration for readiness, preparation, and offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import platform, python_version
from statistics import mean
from time import perf_counter
from typing import Any
import json
import sys

from graph_rag_eval.budget import Budget, assert_matched_budgets
from graph_rag_eval.contracts import (
    CanonicalBundle,
    QueryView,
    canonical_data,
    canonical_json,
    content_sha256,
)
from graph_rag_eval.evaluation.coupled import coupled_metrics
from graph_rag_eval.evaluation.extraction import evaluate_extraction
from graph_rag_eval.evaluation.generation import answer_metrics, support_metrics
from graph_rag_eval.evaluation.intrinsic import evaluate_intrinsic
from graph_rag_eval.evaluation.retrieval import retrieval_metrics
from graph_rag_eval.evaluation.statistics import (
    PairedObservation,
    clustered_paired_bootstrap,
    mcnemar_counts,
)
from graph_rag_eval.graphs.corruptions import (
    add_edges,
    drop_edges,
    merge_entities,
    relabel_relations,
    remove_provenance,
    rewire_endpoints,
    split_entity,
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
from graph_rag_eval.registry import load_adapter, load_generator
from graph_rag_eval.retrieval.base import (
    EvidenceItem,
    RetrievalResult,
    finalize_result,
)
from graph_rag_eval.retrieval.bm25 import BM25Retriever
from graph_rag_eval.retrieval.dense import HashingDenseRetriever
from graph_rag_eval.retrieval.graph import GraphRetriever
from graph_rag_eval.retrieval.hybrid import ReciprocalRankFusionRetriever
from graph_rag_eval.trace import (
    confined_path,
    validate_schema,
    write_json,
    write_jsonl,
    write_text,
)


class ConfigError(ValueError):
    pass


class RunIdentityError(ValueError):
    """Raised before writes when an occupied run cannot be safely resumed."""


class LeakageError(ValueError):
    """Raised when private material could reach a tested query or prompt."""


@dataclass(frozen=True)
class RunnerContext:
    source_root: Path
    config_path: Path
    config: dict[str, Any]
    run_id: str
    run_root: Path
    adapter: Any


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run_id",
        "adapter",
        "protocol",
        "retrieval",
        "generator",
        "statistics",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ConfigError(f"run config is missing keys: {missing}")
    if data["schema_version"] != "rag-run-config-1.0":
        raise ConfigError("unsupported run config schema_version")
    adapter = data["adapter"]
    for key in ("import_path", "adapter_id", "adapter_version"):
        if key not in adapter:
            raise ConfigError(f"adapter config is missing {key}")
    protocol = data["protocol"]
    if protocol.get("evaluation_scope", "graph_rag_qa") not in {
        "graph_rag_qa",
        "intrinsic_graph_only",
    }:
        raise ConfigError("unsupported evaluation_scope")
    if protocol.get("question_query_fields") != ["text", "public_metadata"]:
        raise ConfigError("question query projection must be exactly text + public_metadata")
    forbidden = set(protocol.get("forbidden_query_fields", []))
    required_forbidden = {
        "answers",
        "answer_aliases",
        "gold_relation",
        "gold_triple",
        "gold_path",
        "evidence_ids",
        "source_ids",
        "condition",
    }
    if not required_forbidden <= forbidden:
        raise ConfigError("forbidden query field contract is incomplete")
    budget = data["retrieval"].get("budget", {})
    if budget.get("max_items", 0) <= 0 or budget.get("max_tokens", 0) <= 0:
        raise ConfigError("retrieval budget must be positive")
    model_revision = data["generator"].get("revision")
    if not model_revision or model_revision in {"main", "latest"}:
        raise ConfigError("generator requires an immutable revision")
    if not data["generator"].get("import_path"):
        raise ConfigError("generator requires an explicit import_path")
    return data


def _runtime_options(value: Any, run_root: Path) -> Any:
    """Resolve the one generic run-root token without mutating stored config."""

    if value == "${RUN_ROOT}":
        return str(run_root)
    if isinstance(value, dict):
        return {key: _runtime_options(item, run_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_runtime_options(item, run_root) for item in value]
    return value


def create_context(
    config_path: str | Path,
    *,
    source_root: str | Path | None = None,
    run_id: str | None = None,
) -> RunnerContext:
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    if root != path and root not in path.parents:
        raise ConfigError("config path must remain inside the standalone source checkout")
    config = _load_config(path)
    run_schema = json.loads(
        (root / "schemas/phase_b/rag-run-config.schema.json").read_text(encoding="utf-8")
    )
    validate_schema(config, run_schema)
    checkpoint = config.get("checkpoint")
    if checkpoint is not None:
        checkpoint_schema = json.loads(
            (
                root
                / "schemas"
                / "phase_b"
                / "rag-checkpoint-manifest.schema.json"
            ).read_text(encoding="utf-8")
        )
        validate_schema(checkpoint, checkpoint_schema)
        if checkpoint["status"] == "ready":
            if checkpoint["checkpoint_sha256"] is None:
                raise ConfigError("ready checkpoint requires an exact SHA-256")
            if checkpoint["blocked_reasons"]:
                raise ConfigError("ready checkpoint cannot retain blocked reasons")
        elif not checkpoint["blocked_reasons"]:
            raise ConfigError("blocked checkpoint requires at least one reason")
    selected_run_id = run_id or config["run_id"]
    run_root = confined_path(root, selected_run_id)
    adapter_options = _runtime_options(
        config["adapter"].get("options", {}),
        run_root,
    )
    adapter = load_adapter(config["adapter"]["import_path"], adapter_options)
    if adapter.adapter_id != config["adapter"]["adapter_id"]:
        raise ConfigError("loaded adapter ID does not match config")
    if adapter.adapter_version != config["adapter"]["adapter_version"]:
        raise ConfigError("loaded adapter version does not match config")
    return RunnerContext(
        source_root=root,
        config_path=path,
        config=config,
        run_id=selected_run_id,
        run_root=run_root,
        adapter=adapter,
    )


def _rag_schema_manifest(source_root: Path) -> list[dict[str, Any]]:
    schema_root = source_root / "schemas" / "phase_b"
    return [
        {
            "path": path.relative_to(source_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(schema_root.glob("rag-*.schema.json"))
    ]


def _expected_run_identity(context: RunnerContext) -> dict[str, Any]:
    schemas = _rag_schema_manifest(context.source_root)
    checkpoint = context.config.get("checkpoint")
    return {
        "schema_version": "rag-run-identity-1.0",
        "run_id": context.run_id,
        "experiment_group_id": context.config.get("experiment_group_id"),
        "config_sha256": content_sha256(context.config),
        "adapter": {
            "adapter_id": context.config["adapter"]["adapter_id"],
            "adapter_version": context.config["adapter"]["adapter_version"],
            "import_path": context.config["adapter"]["import_path"],
            "options_sha256": content_sha256(
                context.config["adapter"].get("options", {})
            ),
        },
        "checkpoint_sha256": (
            content_sha256(checkpoint) if checkpoint is not None else None
        ),
        "source_surface_sha256": source_surface_sha256(context.source_root),
        "source_surface": list(source_surface_manifest(context.source_root)),
        "schema_set_sha256": content_sha256(schemas),
        "schemas": schemas,
    }


def _bind_run_identity(context: RunnerContext) -> dict[str, Any]:
    """Claim a new run ID or verify an exact matching staged resume."""

    expected = _expected_run_identity(context)
    identity_path = context.run_root / "manifests" / "run-identity.json"
    if not context.run_root.exists():
        # `graph-rag/` is the owned namespace. A sibling Phase B run may already
        # occupy its parent; claiming a fresh child cannot overwrite that parent.
        # Phase-B-output diagnostics additionally verify the parent byte contract
        # before calling their dedicated child binder.
        context.run_root.mkdir(parents=True, exist_ok=False)
        artifact = write_json(
            context.run_root,
            "manifests/run-identity.json",
            expected,
        )
        return artifact
    if not context.run_root.is_dir() or not identity_path.is_file():
        raise RunIdentityError(
            f"occupied run has no Graph RAG identity manifest: {context.run_id}"
        )
    try:
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunIdentityError(
            f"run identity manifest is unreadable: {context.run_id}"
        ) from error
    identity_schema = json.loads(
        (
            context.source_root
            / "schemas"
            / "phase_b"
            / "rag-run-identity.schema.json"
        ).read_text(encoding="utf-8")
    )
    try:
        validate_schema(observed, identity_schema)
    except (OSError, ValueError) as error:
        raise RunIdentityError(
            f"run identity manifest is invalid: {context.run_id}"
        ) from error
    if canonical_json(observed) != canonical_json(expected):
        raise RunIdentityError(
            f"run identity mismatch; choose a new run ID: {context.run_id}"
        )
    return {
        "path": "manifests/run-identity.json",
        "bytes": identity_path.stat().st_size,
        "sha256": file_sha256(identity_path),
    }


def _assert_existing_json_matches(
    path: Path,
    expected: Any,
    *,
    label: str,
) -> None:
    if not path.exists():
        return
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunIdentityError(f"existing {label} manifest is unreadable") from error
    if canonical_json(observed) != canonical_json(expected):
        raise RunIdentityError(
            f"existing {label} identity does not match the resumed run"
        )


def _environment_manifest(context: RunnerContext) -> dict[str, Any]:
    return {
        "schema_version": "rag-environment-1.0",
        "python": python_version(),
        "platform": platform(),
        "implementation": sys.implementation.name,
        "packages": {
            "bm25s": _package_version("bm25s"),
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "network_used": False,
        "gpu_execution": False,
        "hardware_note": "offline lane; GPU presence is not assumed",
    }


def _scientific_gates(context: RunnerContext, report) -> list[dict[str, Any]]:
    gates = [
        {
            "code": issue.code,
            "status": "blocked" if issue.severity in {"block", "error"} else issue.severity,
            "reason": issue.message,
            "path": issue.path,
        }
        for issue in report.issues
    ]
    required = set(context.config["protocol"].get("required_capabilities", []))
    missing = sorted(required - set(report.capabilities))
    if missing:
        gates.append(
            {
                "code": "missing-required-capabilities",
                "status": "blocked",
                "reason": f"adapter lacks required capabilities: {', '.join(missing)}",
                "missing": missing,
            }
        )
    if context.config["protocol"].get("regime_q", {}).get("required"):
        regime = context.config["protocol"]["regime_q"]
        if not regime.get("independent_author_available") or not regime.get(
            "independent_reviewer_available"
        ):
            gates.append(
                {
                    "code": "regime-q-human-independence-unavailable",
                    "status": "blocked",
                    "reason": (
                        "no domain expert is available to independently author and "
                        "review questions and acceptable evidence"
                    ),
                }
            )
    if context.config["generator"].get("judge_required") and not context.config[
        "generator"
    ].get("human_calibration_set"):
        gates.append(
            {
                "code": "judge-calibration-unavailable",
                "status": "blocked",
                "reason": "model-based judging is disabled without frozen human calibration",
            }
        )
    checkpoint = context.config.get("checkpoint")
    if checkpoint is not None and checkpoint.get("status") != "ready":
        gates.append(
            {
                "code": "checkpoint-not-ready",
                "status": "blocked",
                "reason": "; ".join(checkpoint.get("blocked_reasons", ()))
                or "checkpoint identity is not ready for scientific execution",
            }
        )
    return gates


def doctor(context: RunnerContext) -> dict[str, Any]:
    identity_artifact = _bind_run_identity(context)
    report = context.adapter.validate()
    gates = _scientific_gates(context, report)
    blocked = [item for item in gates if item["status"] == "blocked"]
    config_hash = content_sha256(context.config)
    artifacts = [identity_artifact]
    artifacts.append(write_json(context.run_root, "manifests/environment.json", _environment_manifest(context)))
    artifacts.append(
        write_json(
            context.run_root,
            "manifests/config.json",
            {
                "schema_version": "rag-config-manifest-1.0",
                "config_sha256": config_hash,
                "config": context.config,
            },
        )
    )
    checkpoint = context.config.get("checkpoint")
    if checkpoint is not None:
        checkpoint_schema = json.loads(
            (
                context.source_root
                / "schemas"
                / "phase_b"
                / "rag-checkpoint-manifest.schema.json"
            ).read_text(encoding="utf-8")
        )
        validate_schema(checkpoint, checkpoint_schema)
        artifacts.append(
            write_json(
                context.run_root,
                "manifests/checkpoint.json",
                checkpoint,
            )
        )
    result = {
        "schema_version": "rag-run-manifest-1.0",
        "run_id": context.run_id,
        "stage": "doctor",
        "status": "blocked" if blocked else "ready",
        "adapter_id": report.adapter_id,
        "adapter_ready": report.ready,
        "capabilities": list(report.capabilities),
        "gates": gates,
        "artifacts": artifacts,
        "no_domain_expert_assumption": not context.config["protocol"]
        .get("regime_q", {})
        .get("independent_author_available", False),
    }
    write_json(context.run_root, "manifests/run.json", result)
    write_jsonl(context.run_root, "logs/doctor.jsonl", (result,))
    return result


def _build_graphs(context: RunnerContext, bundle: CanonicalBundle) -> dict[str, GraphSnapshot]:
    descriptor_hash = content_sha256(bundle.descriptor)
    corpus_hash = content_sha256(bundle.chunks)
    adapter_hash = content_sha256(
        {
            "adapter_id": bundle.descriptor.adapter_id,
            "adapter_version": bundle.descriptor.adapter_version,
        }
    )
    gold = build_snapshot(
        dataset_id=bundle.descriptor.dataset_id,
        condition="gold_graph_oracle",
        entities=bundle.entities,
        relations=bundle.relations,
        triples=bundle.triples,
        descriptor_sha256=descriptor_hash,
        corpus_sha256=corpus_hash,
        adapter_sha256=adapter_hash,
        extractor_sha256=fingerprint("gold-source", bundle.descriptor.snapshot_id),
        construction_recipe="adapter-declared-gold-v1",
    )
    predicted = context.adapter.generated_graph(bundle)
    generated = build_snapshot(
        dataset_id=bundle.descriptor.dataset_id,
        condition="generated_graph",
        entities=predicted.entities,
        relations=predicted.relations,
        triples=predicted.triples,
        descriptor_sha256=descriptor_hash,
        corpus_sha256=corpus_hash,
        adapter_sha256=adapter_hash,
        extractor_sha256=fingerprint(
            "adapter-generated-graph",
            context.adapter.adapter_id,
            context.adapter.adapter_version,
            predicted.extractor_id,
            content_sha256(
                {
                    "entities": predicted.entities,
                    "relations": predicted.relations,
                    "triples": predicted.triples,
                }
            ),
        ),
        construction_recipe=predicted.construction_recipe,
    )
    graphs = {
        "gold_graph_oracle": gold,
        "generated_graph": generated,
    }
    if (
        generated.triples
        and "corruptions" in context.config["retrieval"].get("conditions", ())
    ):
        corruption = context.config["retrieval"].get("corruption", {})
        seed = int(corruption.get("seed", 42))
        severity = float(corruption.get("severity", 0.5))
        candidates = (
            drop_edges(generated, severity, seed),
            add_edges(generated, severity, seed),
            relabel_relations(generated, severity, seed),
            rewire_endpoints(generated, severity, seed),
            remove_provenance(generated, severity, seed),
            merge_entities(generated, seed),
            split_entity(generated, seed),
        )
        graphs.update({item.condition.split(":")[-1]: item for item in candidates})
    return graphs


def _write_graph(context: RunnerContext, label: str, graph: GraphSnapshot) -> list[dict]:
    prefix = f"graphs/{label}"
    schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-graph-snapshot.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate_schema(graph, schema)
    return [
        write_json(context.run_root, f"{prefix}/manifest.json", graph),
        write_jsonl(context.run_root, f"{prefix}/entities.jsonl", graph.entities),
        write_jsonl(context.run_root, f"{prefix}/relations.jsonl", graph.relations),
        write_jsonl(context.run_root, f"{prefix}/triples.jsonl", graph.triples),
    ]


def prepare(context: RunnerContext) -> dict[str, Any]:
    doctor_result = doctor(context)
    report = context.adapter.validate()
    if not report.ready:
        return doctor_result
    bundle = context.adapter.load()
    _assert_existing_json_matches(
        context.run_root / "manifests" / "dataset.json",
        bundle.descriptor,
        label="dataset",
    )
    graphs = _build_graphs(context, bundle)
    dataset_schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-dataset-descriptor.schema.json").read_text(
            encoding="utf-8"
        )
    )
    canonical_schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-canonical-record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate_schema(bundle.descriptor, dataset_schema)
    for record in (
        bundle.documents
        + bundle.chunks
        + bundle.entities
        + bundle.relations
        + bundle.triples
        + bundle.questions
    ):
        validate_schema(record, canonical_schema)
    question_manifest = {
        "schema_version": "rag-question-set-1.0",
        "question_set_sha256": content_sha256(
            {
                "questions": bundle.questions,
                "answers": bundle.answers,
                "evidence_sets": bundle.evidence_sets,
                "relevance_judgments": bundle.relevance_judgments,
            }
        ),
        "questions": len(bundle.questions),
        "regimes": sorted({item.regime for item in bundle.questions}),
        "independence": context.config["protocol"].get("regime_q", {}),
    }
    question_schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-question-set.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validate_schema(question_manifest, question_schema)
    artifacts = [
        write_json(context.run_root, "manifests/dataset.json", bundle.descriptor),
        write_json(
            context.run_root,
            "manifests/question-set.json",
            question_manifest,
        ),
        write_jsonl(context.run_root, "canonical/documents.jsonl", bundle.documents),
        write_jsonl(context.run_root, "canonical/chunks.jsonl", bundle.chunks),
        write_jsonl(context.run_root, "canonical/entities.jsonl", bundle.entities),
        write_jsonl(context.run_root, "canonical/relations.jsonl", bundle.relations),
        write_jsonl(context.run_root, "canonical/triples.jsonl", bundle.triples),
        write_jsonl(context.run_root, "canonical/questions.jsonl", bundle.questions),
        write_jsonl(context.run_root, "canonical/answer-aliases.jsonl", bundle.answers),
        write_jsonl(context.run_root, "canonical/evidence-sets.jsonl", bundle.evidence_sets),
        write_jsonl(
            context.run_root,
            "canonical/relevance-judgments.jsonl",
            bundle.relevance_judgments,
        ),
    ]
    for label, graph in graphs.items():
        artifacts.extend(_write_graph(context, label, graph))
    graph_manifest = {
        label: {
            "graph_id": graph.graph_id,
            "condition": graph.condition,
            "counts": graph.counts,
            "parent_graph_id": graph.parent_graph_id,
            "corruption_recipe": graph.corruption_recipe,
        }
        for label, graph in sorted(graphs.items())
    }
    artifacts.append(write_json(context.run_root, "manifests/graph.json", graph_manifest))
    blocked = [
        item
        for item in doctor_result["gates"]
        if item["status"] == "blocked"
    ]
    result = {
        "schema_version": "rag-run-manifest-1.0",
        "run_id": context.run_id,
        "stage": "prepare",
        "status": "prepared_with_blockers" if blocked else "prepared",
        "dataset_id": bundle.descriptor.dataset_id,
        "dataset_sha256": bundle.descriptor.canonical_content_sha256,
        "counts": {
            "documents": len(bundle.documents),
            "chunks": len(bundle.chunks),
            "entities": len(bundle.entities),
            "relations": len(bundle.relations),
            "triples": len(bundle.triples),
            "questions": len(bundle.questions),
        },
        "gates": doctor_result["gates"],
        "artifacts": artifacts,
    }
    write_json(context.run_root, "manifests/run.json", result)
    write_jsonl(context.run_root, "logs/prepare.jsonl", (result,))
    return result


class _NoContextRetriever:
    retriever_id = "no-context-v1"
    index_fingerprint = fingerprint("no-context-v1")

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult:
        return finalize_result(
            retriever_id=self.retriever_id,
            query=query,
            rows=(),
            budget=budget,
            index_fingerprint=self.index_fingerprint,
            latency_ms=0.0,
            failure="intentional_no_context",
        )


class _RandomContextRetriever:
    retriever_id = "random-context-v1"

    def __init__(self, bundle: CanonicalBundle, seed: int):
        self.chunks = bundle.chunks
        self.seed = seed
        self.index_fingerprint = fingerprint(
            "random-context-v1",
            seed,
            [(item.chunk_id, item.content_sha256) for item in self.chunks],
        )

    def retrieve(self, query: QueryView, budget: Budget) -> RetrievalResult:
        digest = sha256(f"{self.seed}:{query.question_id}".encode("utf-8")).digest()
        offset = int.from_bytes(digest[:4], "big") % len(self.chunks)
        ordered = self.chunks[offset:] + self.chunks[:offset]
        rows = tuple(
            EvidenceItem(
                item.chunk_id,
                "chunk",
                item.text,
                0.0,
                rank,
                (item.document_id,),
            )
            for rank, item in enumerate(ordered, start=1)
        )
        return finalize_result(
            retriever_id=self.retriever_id,
            query=query,
            rows=rows,
            budget=budget,
            index_fingerprint=self.index_fingerprint,
            latency_ms=0.0,
        )


def _generator(context: RunnerContext):
    config = context.config["generator"]
    generator = load_generator(config["import_path"], config.get("options"))
    if generator.generator_id != config["generator_id"]:
        raise ConfigError("loaded generator ID does not match config")
    return generator


def _assert_no_private_leakage(
    bundle: CanonicalBundle,
    query: QueryView,
    question,
    forbidden_fields: set[str],
) -> None:
    """Fail closed when private material can reach the tested query.

    Answers and evidence identifiers legitimately appear inside retrieved
    evidence; they must never appear in the query projection that retrieval,
    index fingerprints, and prompt construction consume.
    """

    if query.text != question.text or query.question_id != question.question_id:
        raise LeakageError(
            f"query projection diverges from the public question: {question.question_id}"
        )
    exposed = sorted(forbidden_fields & set(query.public_metadata))
    if exposed:
        raise LeakageError(f"public metadata exposes forbidden fields: {exposed}")
    serialized = canonical_json(query)
    for token in bundle.private_tokens():
        if token and token in serialized:
            raise LeakageError(
                f"private answer or evidence identity reached the query: {question.question_id}"
            )


def _retrievers(
    context: RunnerContext,
    bundle: CanonicalBundle,
    graphs: dict[str, GraphSnapshot],
) -> dict[str, Any]:
    retrieval_config = context.config["retrieval"]
    bm25_config = retrieval_config["bm25"]
    bm25 = BM25Retriever(
        bundle.chunks,
        k1=float(bm25_config["k1"]),
        b=float(bm25_config["b"]),
        method=bm25_config["method"],
        idf_method=bm25_config["idf_method"],
    )
    dense = HashingDenseRetriever(bundle.chunks)
    graph = GraphRetriever(
        graphs["generated_graph"],
        max_hops=int(retrieval_config["graph"]["max_hops"]),
    )
    gold = GraphRetriever(
        graphs["gold_graph_oracle"],
        max_hops=int(retrieval_config["graph"]["max_hops"]),
    )
    result = {
        "no_context": _NoContextRetriever(),
        "random_context": _RandomContextRetriever(bundle, retrieval_config["random_seed"]),
        "bm25_text": bm25,
        "dense_text_fixture": dense,
        "hybrid_text": ReciprocalRankFusionRetriever(bm25, dense),
        "generated_graph": graph,
        "graph_plus_bm25": ReciprocalRankFusionRetriever(graph, bm25),
        "gold_graph_oracle": gold,
    }
    for label, snapshot in graphs.items():
        if label in {"generated_graph", "gold_graph_oracle"}:
            continue
        result[f"corruption_{label}"] = GraphRetriever(
            snapshot,
            max_hops=int(retrieval_config["graph"]["max_hops"]),
        )
    return result


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    excluded_keys = {"denominator", "k", "abstention_expected", "abstained"}
    for row in rows:
        condition = row["condition"]
        for family in ("retrieval", "answer", "support", "coupled"):
            record = row[family]
            # A family that is not `available` contributes no observation. Pooling
            # its placeholder values would turn an unmet prerequisite into a
            # measured zero and silently change the denominator.
            if record.get("status") != "available":
                continue
            for key, value in record.items():
                if isinstance(value, bool) or key in excluded_keys:
                    continue
                if isinstance(value, (int, float)):
                    grouped.setdefault((condition, f"{family}.{key}"), []).append(float(value))
    return [
        {
            "schema_version": "rag-metrics-1.0",
            "condition": condition,
            "metric": metric,
            "value": mean(values),
            "numerator": sum(values),
            "denominator": len(values),
            "status": "available",
        }
        for (condition, metric), values in sorted(grouped.items())
    ]


def _table_csv(aggregate: list[dict[str, Any]]) -> str:
    lines = ["condition,metric,value,numerator,denominator,status"]
    for row in aggregate:
        lines.append(
            ",".join(
                (
                    str(row["condition"]),
                    str(row["metric"]),
                    f"{row['value']:.9f}",
                    f"{row['numerator']:.9f}",
                    str(row["denominator"]),
                    str(row["status"]),
                )
            )
        )
    return "\n".join(lines) + "\n"


def _figure_svg(aggregate: list[dict[str, Any]]) -> str:
    rows = [
        item
        for item in aggregate
        if item["metric"] == "coupled.supported_answer"
    ]
    width = 640
    height = 40 + 28 * max(1, len(rows))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font:12px monospace}.bar{fill:#315f72}</style>',
    ]
    for index, row in enumerate(rows):
        y = 25 + index * 28
        bar_width = int(300 * row["value"])
        elements.append(f'<text x="5" y="{y}">{row["condition"]}</text>')
        elements.append(f'<rect class="bar" x="300" y="{y-12}" width="{bar_width}" height="16"/>')
        elements.append(f'<text x="{305+bar_width}" y="{y}">{row["value"]:.3f}</text>')
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _graph_metric_rows(
    context: RunnerContext,
    bundle: CanonicalBundle,
    graphs: dict[str, GraphSnapshot],
    *,
    include_extraction: bool,
) -> list[dict[str, Any]]:
    rows = []
    for label, snapshot in sorted(graphs.items()):
        if label == "gold_graph_oracle":
            continue
        intrinsic = evaluate_intrinsic(
            snapshot,
            graphs["gold_graph_oracle"],
            capabilities=bundle.descriptor.capabilities,
        )
        rows.append(
            {
                "schema_version": "rag-metrics-1.0",
                "run_id": context.run_id,
                "dataset_id": bundle.descriptor.dataset_id,
                "condition": label,
                "metric": "intrinsic_graph",
                "value": canonical_data(intrinsic),
                "status": getattr(intrinsic, "status", "available"),
            }
        )
    if include_extraction:
        rows.append(
            {
                "schema_version": "rag-metrics-1.0",
                "run_id": context.run_id,
                "dataset_id": bundle.descriptor.dataset_id,
                "condition": "generated_graph",
                "metric": "exact_extraction",
                "value": evaluate_extraction(
                    graphs["generated_graph"],
                    graphs["gold_graph_oracle"],
                ),
                "status": "available",
            }
        )
    rows.extend(
        {
            "schema_version": "rag-metrics-1.0",
            "run_id": context.run_id,
            "dataset_id": bundle.descriptor.dataset_id,
            "condition": label,
            "metric": "graph_structure",
            "value": structure_summary(snapshot),
            "status": "available",
        }
        for label, snapshot in sorted(graphs.items())
    )
    return rows


def _extraction_table(extraction: dict[str, Any]) -> str:
    rows = [
        "task\tlabel\ttrue_positive\tfalse_positive\tfalse_negative"
        "\tsupport\tprecision\trecall\tf1"
    ]
    for task in ("entity", "end_to_end_relation"):
        values = extraction[task]
        labeled_counts = [("micro", values["micro"])]
        labeled_counts.extend(values["per_label"].items())
        for label, counts in labeled_counts:
            rows.append(
                "\t".join(
                    [
                        task,
                        label,
                        str(counts["true_positive"]),
                        str(counts["false_positive"]),
                        str(counts["false_negative"]),
                        str(counts["support"]),
                        "" if counts["precision"] is None else f"{counts['precision']:.12g}",
                        "" if counts["recall"] is None else f"{counts['recall']:.12g}",
                        "" if counts["f1"] is None else f"{counts['f1']:.12g}",
                    ]
                )
            )
    return "\n".join(rows) + "\n"


def _evaluate_intrinsic_only(
    context: RunnerContext,
    bundle: CanonicalBundle,
    graphs: dict[str, GraphSnapshot],
    *,
    smoke: bool,
) -> dict[str, Any]:
    metric_rows = _graph_metric_rows(
        context,
        bundle,
        graphs,
        include_extraction=True,
    )
    metric_schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-metrics.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for row in metric_rows:
        validate_schema(row, metric_schema)
    extraction = next(
        row["value"] for row in metric_rows if row["metric"] == "exact_extraction"
    )
    write_json(context.run_root, "manifests/index.json", {})
    write_jsonl(context.run_root, "traces/retrieval.jsonl", ())
    write_jsonl(context.run_root, "traces/generation.jsonl", ())
    write_jsonl(context.run_root, "traces/failures.jsonl", ())
    write_jsonl(context.run_root, "metrics/per-question.jsonl", ())
    write_jsonl(context.run_root, "metrics/per-document.jsonl", ())
    write_jsonl(context.run_root, "metrics/per-seed.jsonl", ())
    write_jsonl(context.run_root, "metrics/aggregate.jsonl", metric_rows)
    write_jsonl(context.run_root, "metrics/comparisons.jsonl", ())
    write_text(
        context.run_root,
        "tables/extraction-results.tsv",
        _extraction_table(extraction),
    )
    result = {
        "schema_version": "rag-run-manifest-1.0",
        "run_id": context.run_id,
        "stage": "evaluate",
        "status": "intrinsic_evaluation_complete",
        "smoke": smoke,
        "dataset_id": bundle.descriptor.dataset_id,
        "evaluation_scope": "intrinsic_graph_only",
        "scientific_claims_enabled": (
            not smoke
            and bool(
                context.config["protocol"].get(
                    "extraction_publication_claim_allowed",
                    False,
                )
            )
        ),
    }
    write_json(context.run_root, "manifests/run.json", result)
    write_jsonl(context.run_root, "logs/evaluate.jsonl", (result,))
    return result


def evaluate(context: RunnerContext, *, smoke: bool = False) -> dict[str, Any]:
    prepare_result = prepare(context)
    blocked = [
        item for item in prepare_result.get("gates", []) if item["status"] == "blocked"
    ]
    if blocked:
        failures = [
            {
                "schema_version": "rag-failure-1.0",
                "run_id": context.run_id,
                "stage": "evaluate",
                **item,
            }
            for item in blocked
        ]
        write_jsonl(context.run_root, "traces/failures.jsonl", failures)
        unavailable = [
            {
                "schema_version": "rag-metrics-1.0",
                "run_id": context.run_id,
                "status": "not_applicable",
                "metric": metric,
                "value": None,
                "numerator": None,
                "denominator": 0,
                "reason": "Regime Q and/or required gold inputs are unavailable",
                "gates": [item["code"] for item in blocked],
            }
            for metric in (
                "regime_q.retrieval",
                "regime_q.generation",
                "regime_q.coupled",
                "judge.support",
                "paired_comparison",
            )
        ]
        metric_schema = json.loads(
            (context.source_root / "schemas/phase_b/rag-metrics.schema.json").read_text(
                encoding="utf-8"
            )
        )
        for row in unavailable:
            validate_schema(row, metric_schema)
        write_jsonl(context.run_root, "metrics/aggregate.jsonl", unavailable)
        result = {
            "schema_version": "rag-run-manifest-1.0",
            "run_id": context.run_id,
            "stage": "evaluate",
            "status": "blocked",
            "smoke": smoke,
            "gates": blocked,
            "scientific_claims_enabled": False,
        }
        write_json(context.run_root, "manifests/run.json", result)
        write_jsonl(context.run_root, "logs/evaluate.jsonl", (result,))
        return result

    bundle = context.adapter.load()
    graphs = _build_graphs(context, bundle)
    if (
        context.config["protocol"].get("evaluation_scope", "graph_rag_qa")
        == "intrinsic_graph_only"
    ):
        return _evaluate_intrinsic_only(
            context,
            bundle,
            graphs,
            smoke=smoke,
        )
    retrievers = _retrievers(context, bundle, graphs)
    budget_config = context.config["retrieval"]["budget"]
    budget = Budget(
        max_items=int(budget_config["max_items"]),
        max_tokens=int(budget_config["max_tokens"]),
    )
    generator = _generator(context)
    answer_by_question = {item.question_id: item for item in bundle.answers}
    forbidden_fields = set(context.config["protocol"]["forbidden_query_fields"])
    retrieval_traces = []
    generation_traces = []
    failures = []
    metric_rows = []
    fixture_mode = bool(
        context.config["protocol"].get("regime_q", {}).get("synthetic_fixture_only")
    )
    for question in bundle.questions:
        query = question.public_view()
        _assert_no_private_leakage(bundle, query, question, forbidden_fields)
        question_decisions = []
        for condition, retriever in sorted(retrievers.items()):
            result = retriever.retrieve(query, budget)
            question_decisions.append(result.budget)
            retrieval_traces.append(
                {
                    "schema_version": "rag-retrieval-trace-1.0",
                    "run_id": context.run_id,
                    "dataset_id": bundle.descriptor.dataset_id,
                    "question_id": query.question_id,
                    "regime": query.regime,
                    "condition": condition,
                    "query_sha256": content_sha256(query),
                    "query": canonical_data(query),
                    "index_fingerprint": result.index_fingerprint,
                    "ranked_evidence": canonical_data(result.items),
                    "budget": canonical_data(result.budget),
                    "latency_ms": 0.0 if fixture_mode else result.latency_ms,
                    "latency_mode": "fixture_normalized" if fixture_mode else "measured",
                    "failure": result.failure,
                    "seed_ids": list(result.seed_ids),
                    "expansion_ids": list(result.expansion_ids),
                }
            )
            if result.failure and result.failure != "intentional_no_context":
                failures.append(
                    {
                        "schema_version": "rag-failure-1.0",
                        "run_id": context.run_id,
                        "question_id": query.question_id,
                        "condition": condition,
                        "failure": result.failure,
                    }
                )
            generation_started = perf_counter()
            prediction, record = generator.generate(query, result)
            generation_latency = (perf_counter() - generation_started) * 1000.0
            prompt = record["prompt"]
            if prompt != generator.build_prompt(query, result):
                raise LeakageError(
                    "prompt is not a function of the public question and retrieved evidence "
                    f"alone: {query.question_id}/{condition}"
                )
            generation_traces.append(
                {
                    "schema_version": "rag-generation-trace-1.0",
                    "run_id": context.run_id,
                    "dataset_id": bundle.descriptor.dataset_id,
                    "question_id": query.question_id,
                    "condition": condition,
                    "retrieved_evidence_ids": [item.evidence_id for item in result.items],
                    "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_record": {
                        "question": query.text,
                        "evidence": [item.content for item in result.items],
                    },
                    "model": generator.generator_id,
                    "revision": context.config["generator"]["revision"],
                    "decoding": context.config["generator"]["decoding"],
                    "response": prediction,
                    "input_tokens": record.get("input_tokens"),
                    "output_tokens": record.get("output_tokens"),
                    "latency_ms": 0.0 if fixture_mode else generation_latency,
                    "latency_mode": "fixture_normalized" if fixture_mode else "measured",
                    "error": None,
                }
            )
            retrieval_score = retrieval_metrics(
                result,
                bundle.evidence_sets,
                bundle.relevance_judgments,
                capabilities=bundle.descriptor.capabilities,
                k=budget.max_items,
            )
            answer_score = answer_metrics(prediction, answer_by_question[query.question_id])
            support_score = support_metrics(
                tuple(item.evidence_id for item in result.items),
                bundle.evidence_sets,
                query.question_id,
            )
            coupled_score = coupled_metrics(answer_score, retrieval_score, support_score)
            metric_rows.append(
                {
                    "schema_version": "rag-metrics-1.0",
                    "run_id": context.run_id,
                    "dataset_id": bundle.descriptor.dataset_id,
                    "question_id": query.question_id,
                    "cluster_id": question.cluster_id,
                    "regime": question.regime,
                    "condition": condition,
                    "retrieval": retrieval_score,
                    "answer": answer_score,
                    "support": support_score,
                    "coupled": coupled_score,
                    "latency_ms": 0.0 if fixture_mode else result.latency_ms,
                    "realized_tokens": result.budget.realized_tokens,
                    "realized_items": result.budget.realized_items,
                }
            )
        assert_matched_budgets(question_decisions)
    # Every non-oracle graph condition is scored intrinsically so a corruption's
    # retrieval effect can be read against its measured graph quality instead of
    # only against its recipe.
    intrinsic_rows = _graph_metric_rows(
        context,
        bundle,
        graphs,
        include_extraction=False,
    )
    aggregate = _aggregate(metric_rows)
    per_document = []
    grouped_documents: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped_documents.setdefault((row["cluster_id"], row["condition"]), []).append(row)
    for (cluster_id, condition), rows in sorted(grouped_documents.items()):
        values = [
            float(item["coupled"].get("supported_answer", 0.0))
            for item in rows
            if item["coupled"].get("status") == "available"
        ]
        per_document.append(
            {
                "schema_version": "rag-metrics-1.0",
                "run_id": context.run_id,
                "dataset_id": bundle.descriptor.dataset_id,
                "cluster_id": cluster_id,
                "condition": condition,
                "metric": "coupled.supported_answer",
                "value": mean(values) if values else None,
                "numerator": sum(values) if values else None,
                "denominator": len(values),
                "status": "available" if values else "not_applicable",
            }
        )
    per_seed = [
        {
            **row,
            "seed": int(context.config["generator"]["decoding"]["seed"]),
        }
        for row in aggregate
    ]
    by_key = {
        (row["question_id"], row["condition"]): row for row in metric_rows
    }
    comparisons = []
    for baseline, treatment in (
        ("bm25_text", "generated_graph"),
        ("generated_graph", "gold_graph_oracle"),
    ):
        observations = []
        for question in bundle.questions:
            left = by_key[(question.question_id, baseline)]["coupled"]
            right = by_key[(question.question_id, treatment)]["coupled"]
            if left.get("status") == right.get("status") == "available":
                observations.append(
                    PairedObservation(
                        question.question_id,
                        question.cluster_id,
                        float(left["supported_answer"]),
                        float(right["supported_answer"]),
                    )
                )
        comparisons.append(
            {
                "schema_version": "rag-metrics-1.0",
                "baseline": baseline,
                "treatment": treatment,
                "metric": "coupled.supported_answer",
                "bootstrap": clustered_paired_bootstrap(
                    observations,
                    resamples=int(context.config["statistics"]["bootstrap_resamples"]),
                    seed=int(context.config["statistics"]["seed"]),
                ),
                # `supported_answer` is binary and paired, so the discordant
                # counts are recorded. No p-value is emitted: a test needs a
                # predeclared comparison family, which no approved run has.
                "mcnemar": mcnemar_counts(
                    [item.baseline == 1.0 for item in observations],
                    [item.treatment == 1.0 for item in observations],
                ),
            }
        )
    index_manifest = {
        condition: {
            "retriever_id": retriever.retriever_id,
            "index_fingerprint": retriever.index_fingerprint,
        }
        for condition, retriever in sorted(retrievers.items())
    }
    retrieval_schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-retrieval-trace.schema.json").read_text(
            encoding="utf-8"
        )
    )
    generation_schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-generation-trace.schema.json").read_text(
            encoding="utf-8"
        )
    )
    metric_schema = json.loads(
        (context.source_root / "schemas/phase_b/rag-metrics.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for row in retrieval_traces:
        validate_schema(row, retrieval_schema)
    for row in generation_traces:
        validate_schema(row, generation_schema)
    for row in metric_rows + per_document + per_seed + intrinsic_rows + aggregate + comparisons:
        validate_schema(row, metric_schema)
    write_json(context.run_root, "manifests/index.json", index_manifest)
    for condition, manifest in sorted(index_manifest.items()):
        write_json(
            context.run_root,
            (
                f"indexes/{condition}/"
                f"{manifest['index_fingerprint']}/manifest.json"
            ),
            manifest,
        )
    write_jsonl(context.run_root, "traces/retrieval.jsonl", retrieval_traces)
    write_jsonl(context.run_root, "traces/generation.jsonl", generation_traces)
    write_jsonl(context.run_root, "traces/failures.jsonl", failures)
    write_jsonl(context.run_root, "metrics/per-question.jsonl", metric_rows)
    write_jsonl(context.run_root, "metrics/per-document.jsonl", per_document)
    write_jsonl(context.run_root, "metrics/per-seed.jsonl", per_seed)
    write_jsonl(context.run_root, "metrics/aggregate.jsonl", intrinsic_rows + aggregate)
    write_jsonl(context.run_root, "metrics/comparisons.jsonl", comparisons)
    write_text(context.run_root, "tables/synthetic-results.csv", _table_csv(aggregate))
    write_text(context.run_root, "figures/synthetic-supported-answer.svg", _figure_svg(aggregate))
    result = {
        "schema_version": "rag-run-manifest-1.0",
        "run_id": context.run_id,
        "stage": "evaluate",
        "status": "synthetic_validation_complete",
        "smoke": smoke,
        "dataset_id": bundle.descriptor.dataset_id,
        "questions": len(bundle.questions),
        "conditions": len(retrievers),
        "retrieval_traces": len(retrieval_traces),
        "generation_traces": len(generation_traces),
        "failures": len(failures),
        "scientific_claims_enabled": False,
        "claims_boundary": (
            "synthetic contract validation only; no CODE-ACCORD Regime Q or "
            "publication result"
        ),
    }
    write_json(context.run_root, "manifests/run.json", result)
    write_jsonl(context.run_root, "logs/evaluate.jsonl", (result,))
    return result
