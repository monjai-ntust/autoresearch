"""Authenticated same-run RAG controller and durable continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from utils.rag import evaluator, runner
from utils.common.paths import RunLayout
from utils.rag.journal import evaluate_ordered
from utils.rag.code_questions import OUTPUT_NAMESPACE, PROFILE_ID, _artifact_hash_document, _projection_manifest, _projection_paths, _validate_replacement_output, build_identity, build_part_of_document, preflight_code_questions, question_ledger, validate_execution_snapshot, validate_extension_child


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
        print(json.dumps({"source": source, **summary,
                          "model_identity": runner.persisted_model_identity(summary["model_identity"])},
                         indent=2, sort_keys=True))
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
            from utils.evaluation.reporting import report_rag
            report_rag(RunLayout(runner.ROOT, args.run_id), summary, runtime, child_dir)
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
        child_dir / "environment.json",
        {
            "run_id": args.run_id,
            "source": source,
            "model_identity": runner.persisted_model_identity(summary["model_identity"]),
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
        output_path = child_dir / "rag-results" / f"{condition}.json"
        kg_path = projection_paths[condition]
        records = runtime["records"]
        kg = runner.load_json(kg_path)
        reuse_completed = False
        if isinstance(stage, dict) and stage.get("status") in {"complete", "output-validated"}:
            runner.validate_model_check(
                child_dir,
                status,
                stage.get("post_model_identity_check"),
                expected_phase=f"after-{condition}",
                expected_model=summary["model_identity"],
                condition=condition,
            )
            restored = evaluate_ordered(records, kg,
                questions=runtime["panels"]["combined"], condition=condition,
                identity=identity, child_dir=child_dir, status=status,
                status_path=status_path, model_identity=summary["model_identity"],
                ollama_url=args.ollama_url.rstrip("/"), completed_only=True)
            if not output_path.is_file() or output_path.read_bytes() != runner.scientific_json_bytes(restored):
                raise runner.PhaseEError("completed RAG output differs from its original occurrence evidence")
            reuse_completed = True
        runner.run_stage(
            stage_name,
            lambda records=records, kg=kg, kg_path=kg_path: evaluate_ordered(
                records,
                kg,
                questions=runtime["panels"]["combined"], condition=condition,
                identity=identity, child_dir=child_dir, status=status,
                status_path=status_path, model_identity=summary["model_identity"],
                ollama_url=args.ollama_url.rstrip("/"),
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
        elif stage.get("status") != "complete":
            stage["status"] = "complete"
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
    from utils.evaluation.reporting import report_rag
    report_rag(RunLayout(runner.ROOT, args.run_id), summary, runtime, child_dir)
    return 0
