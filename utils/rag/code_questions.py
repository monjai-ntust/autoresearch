"""Original-CODE question coverage from authenticated same-run parents.

The replacement freshly evaluates the complete expanded panel and writes its
own projections and results beneath ``table2-code-all``. Historical
``table2-q105`` output is an external comparison oracle, never a runtime input.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from utils.rag import evaluator, runner


PROFILE_ID = "code-original"
OUTPUT_NAMESPACE = "table2-code-all"
EXPECTED_ADDITIONAL_TEMPLATES = (
    ("necessity", "What is required for {head}?", "{tail}"),
    ("selection", "What is selected or specified for {head}?", "{tail}"),
    ("equal", "What is {head} equal to?", "{tail}"),
    ("greater-equal", "What is the minimum value for {head}?", "{tail}"),
    ("less-equal", "What is the maximum value for {head}?", "{tail}"),
    ("greater", "What value is {head} greater than?", "{tail}"),
    ("less", "What value is {head} less than?", "{tail}"),
)


def replacement_method() -> dict[str, Any]:
    """Describe the source-defined replacement method for emitted provenance."""

    return {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "output_namespace": OUTPUT_NAMESPACE,
        "additional_templates": [
            {"relation": relation, "question": question, "answer": answer}
            for relation, question, answer in EXPECTED_ADDITIONAL_TEMPLATES
        ],
    }


def _template_mapping() -> dict[str, tuple[str, str]]:
    return {
        relation: (question, answer)
        for relation, question, answer in EXPECTED_ADDITIONAL_TEMPLATES
    }


def _question_digest(questions: list[dict[str, Any]]) -> str:
    content = json.dumps(
        questions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return runner.sha256_bytes(content)


def build_question_panels(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build protected, added, and combined panels with exact ordered removal."""

    protected = evaluator.generate_questions(records)
    combined = evaluator.generate_questions(
        records, additional_templates=_template_mapping()
    )
    protected_subset = [row for row in combined if row["relation"] == "part-of"]
    if protected != protected_subset:
        raise runner.PhaseEError(
            "Protected questions are not the ordered full-record part-of subset"
        )
    protected_keys = {runner.canonical_json(row) for row in protected}
    if len(protected_keys) != len(protected):
        raise runner.PhaseEError("Protected question records are not unique")
    added = [row for row in combined if runner.canonical_json(row) not in protected_keys]
    if len(combined) != len(protected) + len(added):
        raise runner.PhaseEError("Protected questions were not removed exactly once")
    relation_counts = Counter(row["relation"] for row in combined)
    for relation, _question, _answer in EXPECTED_ADDITIONAL_TEMPLATES:
        relation_counts.setdefault(relation, 0)
    return {
        "protected": protected,
        "added": added,
        "combined": combined,
        "summary": {
            "protected_count": len(protected),
            "added_count": len(added),
            "combined_count": len(combined),
            "protected_sha256": _question_digest(protected),
            "added_sha256": _question_digest(added),
            "combined_sha256": _question_digest(combined),
            "relation_counts": dict(sorted(relation_counts.items())),
        },
    }


def _records_from_bytes(content: bytes) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise runner.PhaseEError("Authenticated evaluator projection is invalid JSONL") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise runner.PhaseEError("Authenticated evaluator projection is empty or malformed")
    return rows


def _projection_paths(child_dir: Path) -> dict[str, Path]:
    projection_dir = child_dir / "projections"
    return {
        "records": projection_dir / "test-with-private-gold.jsonl",
        "confidence": projection_dir / "confidence-seed-42.json",
        "corrective": projection_dir / "corrective-seed-42.json",
        "gold": projection_dir / "gold-oracle.json",
    }


def _question_contract(questions: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"q": row["question"], "gold": row["gold_answer"]} for row in questions]


def _base_preflight_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    return runner._preflight_identity(preflight)


def preflight_code_questions(
    run_id: str,
    base_contract: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Authenticate same-run parents and derive replacement-local projections."""

    run_dir, base_preflight, projections, graphs = runner.preflight_same_run(
        run_id, base_contract
    )
    records = _records_from_bytes(projections["records"])
    panels = build_question_panels(records)
    method = replacement_method()
    graph_summaries = {
        name: {
            **value,
            "source_path": (
                f"{OUTPUT_NAMESPACE}/canonical-graphs/{name}/manifest.json"
                if value["source"] == "constructed"
                else value["source_path"]
            ),
        }
        for name, value in base_preflight["graphs"].items()
    }
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "lineage_status": "lineage-valid",
        "profile_id": PROFILE_ID,
        "method_sha256": runner.sha256_bytes(runner.canonical_json_bytes(method)),
        "base_contract_sha256": runner.sha256_file(runner.CONTRACT_PATH),
        "base_preflight_identity": _base_preflight_identity(base_preflight),
        "projection_sha256": base_preflight["projection_sha256"],
        "graphs": graph_summaries,
        "record_summary": base_preflight["records"],
        "questions": panels["summary"],
        "method": method,
        "model_identity": base_preflight["model_identity"],
        "environment": base_preflight["environment"],
    }
    runtime = {
        "base_preflight": base_preflight,
        "projections": projections,
        "graphs": graphs,
        "records": records,
        "panels": panels,
    }
    return run_dir, summary, runtime


def build_identity(
    run_id: str, source: dict[str, str], summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "phase-g-code-questions-identity-1.0",
        "run_id": run_id,
        "source": source,
        "profile_id": summary["profile_id"],
        "method_sha256": summary["method_sha256"],
        "base_contract_sha256": summary["base_contract_sha256"],
        "base_preflight_identity": summary["base_preflight_identity"],
        "projection_sha256": summary["projection_sha256"],
        "question_sha256": {
            name: summary["questions"][f"{name}_sha256"]
            for name in ("protected", "added", "combined")
        },
        "ollama_tag_digest": summary["model_identity"]["tag_digest"],
        "ollama_model_blob_sha256": summary["model_identity"]["blob_sha256"],
    }


def _validate_tracked_extension_source(source: dict[str, str]) -> None:
    """Require the replacement controller to be bytes from source HEAD."""

    path = Path(__file__).resolve()
    relative = path.relative_to(runner.ROOT).as_posix()
    if runner._git_file_bytes(source["head"], relative) != path.read_bytes():
        raise runner.PhaseEError(
            f"CODE question source is not identical to source HEAD: {relative}"
        )


def validate_execution_snapshot(
    identity: dict[str, Any],
    source: dict[str, str],
    summary: dict[str, Any],
    runtime: dict[str, Any],
    base_contract: dict[str, Any],
) -> None:
    """Recheck source, method, parent lineage, projections, and questions."""

    if runner.validate_source_contract(base_contract, require_clean=True) != source:
        raise runner.PhaseEError("Source identity changed after CODE-question preflight")
    _validate_tracked_extension_source(source)
    method = replacement_method()
    if method != summary["method"]:
        raise runner.PhaseEError("CODE question method changed after preflight")
    if (
        runner.sha256_bytes(runner.canonical_json_bytes(method))
        != summary["method_sha256"]
    ):
        raise runner.PhaseEError("CODE question method hash changed after preflight")
    if runner.sha256_file(runner.CONTRACT_PATH) != summary["base_contract_sha256"]:
        raise runner.PhaseEError("Base Table-2 contract changed after preflight")
    if build_identity(summary["run_id"], source, summary) != identity:
        raise runner.PhaseEError("CODE question child identity changed after preflight")
    _run_dir, current_preflight, current_projections, _graphs = (
        runner.preflight_same_run(summary["run_id"], base_contract)
    )
    if _base_preflight_identity(current_preflight) != summary["base_preflight_identity"]:
        raise runner.PhaseEError("Same-run parent lineage changed after preflight")
    if current_preflight["projection_sha256"] != summary["projection_sha256"]:
        raise runner.PhaseEError("Replacement projections changed after preflight")
    rebuilt = build_question_panels(_records_from_bytes(current_projections["records"]))
    if rebuilt["summary"] != summary["questions"] or rebuilt != runtime["panels"]:
        raise runner.PhaseEError("CODE question panel changed after preflight")


def question_ledger(run_id: str, panels: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase-g-code-question-ledger-1.0",
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "hash_serialization": (
            "UTF-8 JSON, ensure_ascii=false, sort_keys=true, separators=(',', ':'), "
            "no trailing newline"
        ),
        "summary": panels["summary"],
        "panels": {
            name: panels[name] for name in ("protected", "added", "combined")
        },
    }


def _panel_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "n": count,
        "correct_count": correct,
        "answer_hit_rate": correct / count if count else None,
    }


def _validate_result_rows(
    rows: list[dict[str, Any]], questions: list[dict[str, Any]], label: str
) -> None:
    if len(rows) != len(questions):
        raise runner.PhaseEError(f"{label} row count differs from its question panel")
    for row, question in zip(rows, questions):
        if (
            not isinstance(row, dict)
            or set(row) != {"q", "pred", "gold", "correct"}
            or row.get("q") != question["question"]
            or row.get("gold") != question["gold_answer"]
            or not isinstance(row.get("pred"), str)
            or not row["pred"].strip()
            or row["pred"].startswith("ERROR:")
            or not isinstance(row.get("correct"), bool)
            or row["correct"] != evaluator.check_answer(row["pred"], row["gold"])
        ):
            raise runner.PhaseEError(f"{label} contains a failed or changed result row")


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    return runner.sha256_bytes(runner.canonical_json_bytes(rows, newline=False))


def _panel_indices(panels: dict[str, Any]) -> tuple[list[int], list[int]]:
    protected_questions = panels["protected"]
    combined_questions = panels["combined"]
    protected_indices = [
        index
        for index, question in enumerate(combined_questions)
        if question["relation"] == "part-of"
    ]
    if [combined_questions[index] for index in protected_indices] != protected_questions:
        raise runner.PhaseEError("Combined panel no longer contains the protected panel in order")
    protected_set = set(protected_indices)
    additional_indices = [
        index for index in range(len(combined_questions)) if index not in protected_set
    ]
    if [combined_questions[index] for index in additional_indices] != panels["added"]:
        raise runner.PhaseEError("Combined panel additional-question order changed")
    return protected_indices, additional_indices


def build_part_of_document(
    *,
    run_id: str,
    condition: str,
    panels: dict[str, Any],
    replacement_output: dict[str, Any],
    replacement_path: str,
    replacement_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract freshly executed part-of rows for later external-oracle comparison."""

    protected_indices, additional_indices = _panel_indices(panels)
    statistics: dict[str, Any] = {}
    results: dict[str, Any] = {}
    row_hashes: dict[str, str] = {}
    for mode in evaluator.MODES:
        combined_rows = replacement_output["results"][mode]
        part_of_rows = [combined_rows[index] for index in protected_indices]
        additional_rows = [combined_rows[index] for index in additional_indices]
        _validate_result_rows(
            combined_rows, panels["combined"], f"replacement {condition}/{mode}"
        )
        _validate_result_rows(
            part_of_rows, panels["protected"], f"part-of {condition}/{mode}"
        )
        results[mode] = part_of_rows
        row_hashes[mode] = _rows_digest(part_of_rows)
        statistics[mode] = {
            "part_of": _panel_statistics(part_of_rows),
            "additional": _panel_statistics(additional_rows),
            "combined": _panel_statistics(combined_rows),
        }
    document = {
        "schema_version": "phase-g-part-of-results-1.0",
        "run_id": run_id,
        "condition": condition,
        "source": {
            "path": replacement_path,
            "sha256": replacement_sha256,
            "time_seconds": replacement_output["metadata"]["time_seconds"],
        },
        "question_sha256": _question_digest(panels["protected"]),
        "combined_indices": protected_indices,
        "row_sha256": row_hashes,
        "results": results,
        "oracle_comparison": {
            "status": "pending-external-oracle",
            "runtime_input": False,
        },
    }
    return document, statistics


def _validate_replacement_output(
    path: Path,
    child_dir: Path,
    runtime: dict[str, Any],
    condition: str,
    model_name: str,
) -> dict[str, Any]:
    return runner.validate_rag_output(
        path,
        {},
        model_name,
        expected_questions=_question_contract(runtime["panels"]["combined"]),
        expected_kg=runner.stable_child_path(_projection_paths(child_dir)[condition]),
        expected_count=len(runtime["panels"]["combined"]),
    )


def _projection_manifest(
    child_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    paths = _projection_paths(child_dir)
    return {
        "schema_version": "phase-g-code-question-projections-1.0",
        "run_id": summary["run_id"],
        "representation_only": True,
        "historical_child_used": False,
        "inputs": {
            "base_preflight_identity": summary["base_preflight_identity"],
            "graphs": summary["graphs"],
        },
        "outputs": {
            path.relative_to(child_dir).as_posix(): runner.sha256_file(path)
            for path in paths.values()
        },
        "record_summary": summary["record_summary"],
    }


def _artifact_hash_document(child_dir: Path, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "phase-g-code-question-artifact-hashes-1.0",
        "run_id": run_id,
        "artifacts": {
            path.relative_to(child_dir).as_posix(): runner.sha256_file(path)
            for path in sorted(child_dir.rglob("*"))
            if path.is_file()
            and path.name not in {"artifact-hashes.json", "run-status.json"}
        },
    }


def validate_extension_child(
    child_dir: Path,
    status: dict[str, Any],
    identity: dict[str, Any],
    summary: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Validate every independently generated replacement artifact."""

    runner.validate_child_write_surface(child_dir)
    if (
        status.get("schema_version") != "phase-g-code-question-status-1.0"
        or status.get("run_id") != summary["run_id"]
        or status.get("identity") != identity
        or status.get("status") != "complete"
        or not isinstance(status.get("stages"), dict)
    ):
        raise runner.PhaseEError("CODE question status identity differs")
    runner.validate_model_check(
        child_dir,
        status,
        status.get("pre_model_identity_check"),
        expected_phase="before-stages",
        expected_model=summary["model_identity"],
        condition=None,
    )
    expected_ledger = question_ledger(summary["run_id"], runtime["panels"])
    if runner.load_json(child_dir / "question-ledger.json") != expected_ledger:
        raise runner.PhaseEError("CODE question ledger changed")
    projection_paths = _projection_paths(child_dir)
    for name, path in projection_paths.items():
        if (
            not path.is_file()
            or runner.sha256_file(path) != summary["projection_sha256"][name]
        ):
            raise runner.PhaseEError(f"Replacement projection changed: {name}")
    expected_projection_manifest = _projection_manifest(child_dir, summary)
    if (
        runner.load_json(child_dir / "projections" / "manifest.json")
        != expected_projection_manifest
    ):
        raise runner.PhaseEError("Replacement projection manifest changed")
    for name, graph in runtime["graphs"].items():
        if summary["graphs"][name]["source"] == "constructed":
            path = child_dir / "canonical-graphs" / name / "manifest.json"
            if runner.load_json(path) != graph:
                raise runner.PhaseEError(f"Replacement canonical graph changed: {name}")
    expected_method = {
        "schema_version": "phase-g-code-question-method-1.0",
        "run_id": summary["run_id"],
        "identity": identity,
        "method": summary["method"],
        "base_contract": runner.load_json(runner.CONTRACT_PATH),
        "reuse": {
            "historical_results": False,
            "historical_projections": False,
            "same_run_graph_construction": True,
            "evaluation_kernel": True,
        },
    }
    if runner.load_json(child_dir / "method-manifest.json") != expected_method:
        raise runner.PhaseEError("CODE question method manifest changed")
    environment = runner.load_json(child_dir / "environment.json")
    if (
        not isinstance(environment, dict)
        or set(environment)
        != {
            "schema_version",
            "run_id",
            "captured_at",
            "platform",
            "python",
            "source",
            "model_identity",
        }
        or environment.get("schema_version")
        != "phase-g-code-question-environment-1.0"
        or environment.get("run_id") != summary["run_id"]
        or environment.get("captured_at") != status.get("created_at")
        or environment.get("source") != identity["source"]
        or environment.get("model_identity") != summary["model_identity"]
        or not isinstance(environment.get("platform"), str)
        or not isinstance(environment.get("python"), str)
    ):
        raise runner.PhaseEError("CODE question environment manifest changed")
    condition_summaries = {}
    part_of_hashes = {}
    for condition in ("confidence", "corrective", "gold"):
        replacement_path = child_dir / "rag-results" / f"{condition}.json"
        replacement_validation = _validate_replacement_output(
            replacement_path,
            child_dir,
            runtime,
            condition,
            summary["model_identity"]["name"],
        )
        stage = status.get("stages", {}).get(f"rag_{condition}")
        if (
            not isinstance(stage, dict)
            or stage.get("status") != "complete"
            or stage.get("output_sha256") != replacement_validation["sha256"]
        ):
            raise runner.PhaseEError(f"CODE question stage is incomplete: {condition}")
        runner.validate_model_check(
            child_dir,
            status,
            stage.get("post_model_identity_check"),
            expected_phase=f"after-{condition}",
            expected_model=summary["model_identity"],
            condition=condition,
        )
        replacement_output = runner.load_json(replacement_path)
        part_of, statistics = build_part_of_document(
            run_id=summary["run_id"],
            condition=condition,
            panels=runtime["panels"],
            replacement_output=replacement_output,
            replacement_path=runner.stable_child_path(replacement_path),
            replacement_sha256=runner.sha256_file(replacement_path),
        )
        part_of_path = child_dir / "part-of-results" / f"{condition}.json"
        if runner.load_json(part_of_path) != part_of:
            raise runner.PhaseEError(f"CODE question part-of evidence changed: {condition}")
        condition_summaries[condition] = statistics
        part_of_hashes[condition] = runner.sha256_file(part_of_path)
    result_summary = runner.load_json(child_dir / "code-question-results.json")
    expected_summary = {
        "schema_version": "phase-g-code-question-results-1.0",
        "run_id": summary["run_id"],
        "profile_id": PROFILE_ID,
        "question_summary": summary["questions"],
        "conditions": condition_summaries,
        "historical_part_of_comparison": {
            "status": "pending-external-oracle",
            "runtime_input": False,
            "result_artifact_sha256": part_of_hashes,
        },
        "note": (
            "Expanded Table-2-format diagnostic on canonical upstream outputs; "
            "not the paper's original n=10 Table 2."
        ),
    }
    if result_summary != expected_summary:
        raise runner.PhaseEError("CODE question result summary changed")
    hash_document = runner.load_json(child_dir / "artifact-hashes.json")
    if hash_document != _artifact_hash_document(child_dir, summary["run_id"]):
        raise runner.PhaseEError("CODE question artifact hashes differ")
    return expected_summary


def run_command(
    args: argparse.Namespace,
    base_contract: dict[str, Any],
) -> int:
    """Validate or execute the full-panel same-run replacement RAG."""

    source = runner.validate_source_contract(
        base_contract, require_clean=not args.dry_run
    )
    run_dir, summary, runtime = preflight_code_questions(args.run_id, base_contract)
    if args.dry_run:
        print(json.dumps({"source": source, **summary}, indent=2, sort_keys=True))
        return 0

    child_dir = run_dir / OUTPUT_NAMESPACE
    status_path = child_dir / "run-status.json"
    runner.validate_child_write_surface(child_dir)
    identity = build_identity(args.run_id, source, summary)
    new_child = not child_dir.exists() or not status_path.exists()
    if child_dir.exists() and not status_path.exists() and any(child_dir.iterdir()):
        raise runner.PhaseEError(
            f"{OUTPUT_NAMESPACE} exists without a resumable status identity"
        )
    if new_child:
        status = {
            "schema_version": "phase-g-code-question-status-1.0",
            "run_id": args.run_id,
            "identity": identity,
            "status": "running",
            "created_at": runner.utc_now(),
            "stages": {},
        }
        child_dir.mkdir(exist_ok=True)
        runner.write_status(status_path, status, child_dir)
    else:
        status = runner.load_json(status_path)
        if status.get("identity") != identity or status.get("run_id") != args.run_id:
            raise runner.PhaseEError("Existing CODE question child identity differs")
        if status.get("status") == "complete":
            validate_extension_child(
                child_dir, status, identity, summary, runtime
            )
            print(
                json.dumps(
                    {"run_dir": str(run_dir), "status": "complete", "resumed": True},
                    indent=2,
                )
            )
            return 0

    try:
        runner.validate_model_check(
            child_dir,
            status,
            status.get("pre_model_identity_check"),
            expected_phase="before-stages",
            expected_model=summary["model_identity"],
            condition=None,
        )
    except runner.ModelEvidenceError:
        if any(stage.get("attempts") for stage in status.get("stages", {}).values()):
            raise runner.PhaseEError(
                "Started CODE question child lacks valid pre-stage model evidence"
            )
        evidence, tags_bytes, show_bytes = runner.fetch_matching_model(
            args.ollama_url, summary["model_identity"]
        )
        status["pre_model_identity_check"] = runner.record_model_check(
            child_dir,
            status,
            status_path,
            "before-stages",
            evidence,
            tags_bytes,
            show_bytes,
        )
        runner.write_status(status_path, status, child_dir)

    projection_paths = _projection_paths(child_dir)
    for name, path in projection_paths.items():
        runner.write_bytes_once(path, runtime["projections"][name], child_dir)
    for name, graph in runtime["graphs"].items():
        if summary["graphs"][name]["source"] == "constructed":
            runner.write_json_once(
                child_dir / "canonical-graphs" / name / "manifest.json",
                graph,
                child_dir,
            )
    runner.write_json_once(
        child_dir / "projections" / "manifest.json",
        _projection_manifest(child_dir, summary),
        child_dir,
    )
    runner.write_json_once(
        child_dir / "question-ledger.json",
        question_ledger(args.run_id, runtime["panels"]),
        child_dir,
    )
    runner.write_json_once(
        child_dir / "method-manifest.json",
        {
            "schema_version": "phase-g-code-question-method-1.0",
            "run_id": args.run_id,
            "identity": identity,
            "method": summary["method"],
            "base_contract": base_contract,
            "reuse": {
                "historical_results": False,
                "historical_projections": False,
                "same_run_graph_construction": True,
                "evaluation_kernel": True,
            },
        },
        child_dir,
    )
    runner.write_json_once(
        child_dir / "environment.json",
        {
            "schema_version": "phase-g-code-question-environment-1.0",
            "run_id": args.run_id,
            "captured_at": status["created_at"],
            "platform": platform.platform(),
            "python": sys.version,
            "source": source,
            "model_identity": summary["model_identity"],
        },
        child_dir,
    )

    replacement_outputs: dict[str, dict[str, Any]] = {}
    for condition in ("confidence", "corrective", "gold"):
        stage_name = f"rag_{condition}"
        validate_execution_snapshot(
            identity, source, summary, runtime, base_contract
        )
        stage = status.get("stages", {}).get(stage_name)
        reuse_completed = False
        if isinstance(stage, dict) and stage.get("status") == "complete":
            runner.validate_model_check(
                child_dir,
                status,
                stage.get("post_model_identity_check"),
                expected_phase=f"after-{condition}",
                expected_model=summary["model_identity"],
                condition=condition,
            )
            reuse_completed = True
        output_path = child_dir / "rag-results" / f"{condition}.json"
        kg_path = projection_paths[condition]
        records = runtime["records"]
        kg = runner.load_json(kg_path)
        runner.run_stage(
            stage_name,
            lambda records=records, kg=kg, kg_path=kg_path: evaluator.evaluate(
                records,
                kg,
                kg_identity=runner.stable_child_path(kg_path),
                ollama_url=args.ollama_url.rstrip("/"),
                ollama_model=summary["model_identity"]["name"],
                question_panel=runtime["panels"]["combined"],
            ),
            output_path,
            lambda path, condition=condition: _validate_replacement_output(
                path,
                child_dir,
                runtime,
                condition,
                summary["model_identity"]["name"],
            ),
            child_dir,
            status,
            status_path,
        )
        if not reuse_completed:
            validate_execution_snapshot(
                identity, source, summary, runtime, base_contract
            )
            evidence, tags_bytes, show_bytes = runner.fetch_matching_model(
                args.ollama_url, summary["model_identity"]
            )
            check = runner.record_model_check(
                child_dir,
                status,
                status_path,
                f"after-{condition}",
                evidence,
                tags_bytes,
                show_bytes,
            )
            status["stages"][stage_name].update(
                status="complete", post_model_identity_check=check
            )
            runner.write_status(status_path, status, child_dir)
        replacement_outputs[condition] = runner.load_json(output_path)

    condition_summaries = {}
    part_of_hashes = {}
    for condition in ("confidence", "corrective", "gold"):
        replacement_path = child_dir / "rag-results" / f"{condition}.json"
        part_of, statistics = build_part_of_document(
            run_id=args.run_id,
            condition=condition,
            panels=runtime["panels"],
            replacement_output=replacement_outputs[condition],
            replacement_path=runner.stable_child_path(replacement_path),
            replacement_sha256=runner.sha256_file(replacement_path),
        )
        part_of_path = child_dir / "part-of-results" / f"{condition}.json"
        runner.write_json_once(
            part_of_path,
            part_of,
            child_dir,
        )
        condition_summaries[condition] = statistics
        part_of_hashes[condition] = runner.sha256_file(part_of_path)

    result_summary = {
        "schema_version": "phase-g-code-question-results-1.0",
        "run_id": args.run_id,
        "profile_id": PROFILE_ID,
        "question_summary": summary["questions"],
        "conditions": condition_summaries,
        "historical_part_of_comparison": {
            "status": "pending-external-oracle",
            "runtime_input": False,
            "result_artifact_sha256": part_of_hashes,
        },
        "note": (
            "Expanded Table-2-format diagnostic on canonical upstream outputs; "
            "not the paper's original n=10 Table 2."
        ),
    }
    runner.write_json_once(
        child_dir / "code-question-results.json", result_summary, child_dir
    )
    runner.write_json_once(
        child_dir / "artifact-hashes.json",
        _artifact_hash_document(child_dir, args.run_id),
        child_dir,
    )
    validate_execution_snapshot(
        identity, source, summary, runtime, base_contract
    )
    _final_run, final_summary, _final_runtime = preflight_code_questions(
        args.run_id, base_contract
    )
    if (
        final_summary["base_preflight_identity"] != summary["base_preflight_identity"]
        or final_summary["projection_sha256"] != summary["projection_sha256"]
        or final_summary["questions"] != summary["questions"]
    ):
        raise runner.PhaseEError("Same-run parent changed during CODE question execution")
    completed = dict(status)
    completed["status"] = "complete"
    completed["finished_at"] = runner.utc_now()
    validate_extension_child(
        child_dir, completed, identity, summary, runtime
    )
    status.clear()
    status.update(completed)
    runner.write_status(status_path, status, child_dir)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "child": OUTPUT_NAMESPACE,
                "status": "complete",
            },
            indent=2,
        )
    )
    return 0
