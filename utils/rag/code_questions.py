"""Additive original-CODE question coverage for an authenticated Table-2 child.

The replacement deliberately treats ``table2-q105`` as an immutable, separately
authenticated parity oracle. It freshly evaluates the complete expanded panel
and writes every new artifact to the sibling ``table2-code-all`` namespace.
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


PROFILE_PATH = runner.ROOT / "resources/contracts/code-questions.json"
PROFILE_ID = "code-original"
EXPECTED_ADDITIONAL_TEMPLATES = (
    ("necessity", "What is required for {head}?", "{tail}"),
    ("selection", "What is selected or specified for {head}?", "{tail}"),
    ("equal", "What is {head} equal to?", "{tail}"),
    ("greater-equal", "What is the minimum value for {head}?", "{tail}"),
    ("less-equal", "What is the maximum value for {head}?", "{tail}"),
    ("greater", "What value is {head} greater than?", "{tail}"),
    ("less", "What value is {head} less than?", "{tail}"),
)


def load_profile() -> dict[str, Any]:
    """Load and validate the tracked extension profile."""

    return validate_profile(runner.load_json(PROFILE_PATH))


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the profile is the approved literal extension."""

    expected_keys = {
        "schema_version",
        "profile_id",
        "base_contract",
        "base_contract_sha256",
        "protected_namespace",
        "output_namespace",
        "additional_templates",
    }
    if not isinstance(profile, dict) or set(profile) != expected_keys:
        raise runner.PhaseEError("CODE question profile fields differ from revision 0.1")
    if (
        profile.get("schema_version") != 1
        or profile.get("profile_id") != PROFILE_ID
        or profile.get("base_contract") != "resources/contracts/table2.json"
        or profile.get("protected_namespace") != "table2-q105"
        or profile.get("output_namespace") != "table2-code-all"
    ):
        raise runner.PhaseEError("CODE question profile identity is invalid")
    if profile.get("base_contract_sha256") != runner.sha256_file(runner.CONTRACT_PATH):
        raise runner.PhaseEError("CODE question profile does not bind the frozen base contract")
    templates = profile.get("additional_templates", [])
    if not isinstance(templates, list) or any(
        not isinstance(item, dict) or set(item) != {"relation", "question", "answer"}
        for item in templates
    ):
        raise runner.PhaseEError("CODE question template records are malformed")
    observed = tuple(
        (item.get("relation"), item.get("question"), item.get("answer"))
        for item in templates
    )
    if observed != EXPECTED_ADDITIONAL_TEMPLATES:
        raise runner.PhaseEError("CODE question templates differ from the approved literals")
    return profile


def _template_mapping(profile: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        item["relation"]: (item["question"], item["answer"])
        for item in profile["additional_templates"]
    }


def _question_digest(questions: list[dict[str, Any]]) -> str:
    content = json.dumps(
        questions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return runner.sha256_bytes(content)


def build_question_panels(
    records: list[dict[str, Any]], profile: dict[str, Any]
) -> dict[str, Any]:
    """Build protected, added, and combined panels with exact ordered removal."""

    protected = evaluator.generate_questions(records)
    combined = evaluator.generate_questions(
        records, additional_templates=_template_mapping(profile)
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


def _protected_projection_paths(child_dir: Path) -> dict[str, Path]:
    projection_dir = child_dir / "projections"
    return {
        "records": projection_dir / "test-with-private-gold.jsonl",
        "confidence": projection_dir / "confidence-seed-42.json",
        "corrective": projection_dir / "corrective-seed-42.json",
        "gold": projection_dir / "gold-oracle.json",
    }


def child_inventory(child_dir: Path) -> dict[str, Any]:
    """Hash every protected file, including its status and hash manifest."""

    runner.validate_child_write_surface(child_dir)
    files = {
        path.relative_to(child_dir).as_posix(): {
            "sha256": runner.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(child_dir.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise runner.PhaseEError("Protected Table-2 child has no files")
    return {
        "files": files,
        "inventory_sha256": runner.sha256_bytes(runner.canonical_json_bytes(files)),
    }


def validate_child_inventory(child_dir: Path, expected: dict[str, Any]) -> None:
    if child_inventory(child_dir) != expected:
        raise runner.PhaseEError("Protected Table-2 child inventory changed")


def _question_contract(questions: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"q": row["question"], "gold": row["gold_answer"]} for row in questions]


def _validate_protected_child(
    run_dir: Path,
    base_preflight: dict[str, Any],
    base_contract: dict[str, Any],
) -> dict[str, Any]:
    child_dir = run_dir / base_contract["evaluator"]["output_namespace"]
    paths = _protected_projection_paths(child_dir)
    status = runner.load_json(child_dir / "run-status.json")
    results = runner.validate_phase_e_child(
        child_dir,
        status,
        base_contract,
        base_preflight,
        paths,
    )
    environment = runner.load_json(child_dir / "environment.json")
    protected_source = environment.get("source") if isinstance(environment, dict) else None
    if (
        not isinstance(protected_source, dict)
        or set(protected_source) != {"head", "branch", "baseline"}
        or status.get("identity")
        != runner.build_child_identity(
            base_preflight["run_id"], protected_source, base_preflight
        )
    ):
        raise runner.PhaseEError(
            "Protected Table-2 child identity does not match its recorded source"
        )
    method = runner.load_json(child_dir / "method-manifest.json")
    if (
        method.get("run_id") != base_preflight["run_id"]
        or method.get("contract") != base_contract
        or method.get("contract_sha256") != runner.sha256_file(runner.CONTRACT_PATH)
    ):
        raise runner.PhaseEError("Protected Table-2 method provenance changed")
    inventory = child_inventory(child_dir)
    return {
        "dir": child_dir,
        "paths": paths,
        "status": status,
        "identity": status.get("identity"),
        "inventory": inventory,
        "results": results,
        "result_documents": {
            condition: runner.load_json(child_dir / "rag-results" / f"{condition}.json")
            for condition in ("confidence", "corrective", "gold")
        },
    }


def _base_preflight_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    return runner._preflight_identity(preflight)


def preflight_code_questions(
    run_id: str,
    base_contract: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Authenticate the same-run parent and immutable protected child."""

    run_dir, base_preflight, projections, graphs = runner.preflight_same_run(
        run_id, base_contract
    )
    protected = _validate_protected_child(run_dir, base_preflight, base_contract)
    records = _records_from_bytes(projections["records"])
    panels = build_question_panels(records, profile)
    protected_count = base_contract["evaluator"]["max_questions"]
    if panels["summary"]["protected_count"] != protected_count:
        raise runner.PhaseEError(
            f"Protected child must contain exactly {protected_count} questions"
        )
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "lineage_status": "lineage-valid",
        "profile_id": profile["profile_id"],
        "profile_sha256": runner.sha256_file(PROFILE_PATH),
        "base_contract_sha256": runner.sha256_file(runner.CONTRACT_PATH),
        "base_preflight_identity": _base_preflight_identity(base_preflight),
        "protected_child": {
            "namespace": profile["protected_namespace"],
            "identity": protected["identity"],
            "artifact_hashes_sha256": runner.sha256_file(
                protected["dir"] / "artifact-hashes.json"
            ),
            "inventory_sha256": protected["inventory"]["inventory_sha256"],
            "files": protected["inventory"]["files"],
        },
        "questions": panels["summary"],
        "model_identity": base_preflight["model_identity"],
        "environment": base_preflight["environment"],
    }
    runtime = {
        "base_preflight": base_preflight,
        "projections": projections,
        "graphs": graphs,
        "records": records,
        "panels": panels,
        "protected": protected,
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
        "profile_sha256": summary["profile_sha256"],
        "base_contract_sha256": summary["base_contract_sha256"],
        "base_preflight_identity": summary["base_preflight_identity"],
        "protected_identity": summary["protected_child"]["identity"],
        "protected_inventory_sha256": summary["protected_child"][
            "inventory_sha256"
        ],
        "question_sha256": {
            name: summary["questions"][f"{name}_sha256"]
            for name in ("protected", "added", "combined")
        },
        "ollama_tag_digest": summary["model_identity"]["tag_digest"],
        "ollama_model_blob_sha256": summary["model_identity"]["blob_sha256"],
    }


def _validate_tracked_extension_source(source: dict[str, str]) -> None:
    """Require the new controller and profile to be bytes from source HEAD."""

    for path in (Path(__file__).resolve(), PROFILE_PATH):
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
    profile: dict[str, Any],
) -> None:
    """Recheck source, profile, projections, and the complete protected child."""

    if runner.validate_source_contract(base_contract, require_clean=True) != source:
        raise runner.PhaseEError("Source identity changed after CODE-question preflight")
    _validate_tracked_extension_source(source)
    if validate_profile(runner.load_json(PROFILE_PATH)) != profile:
        raise runner.PhaseEError("CODE question profile changed after preflight")
    if runner.sha256_file(PROFILE_PATH) != summary["profile_sha256"]:
        raise runner.PhaseEError("CODE question profile hash changed after preflight")
    if runner.sha256_file(runner.CONTRACT_PATH) != summary["base_contract_sha256"]:
        raise runner.PhaseEError("Base Table-2 contract changed after preflight")
    if build_identity(summary["run_id"], source, summary) != identity:
        raise runner.PhaseEError("CODE question child identity changed after preflight")
    protected = runtime["protected"]
    validate_child_inventory(protected["dir"], protected["inventory"])
    runner.validate_child_hashes(protected["dir"], summary["run_id"])
    for name, path in protected["paths"].items():
        if (
            not path.is_file()
            or runner.sha256_file(path)
            != runtime["base_preflight"]["projection_sha256"][name]
        ):
            raise runner.PhaseEError(f"Protected projection changed: {name}")
    rebuilt = build_question_panels(runtime["records"], profile)
    if rebuilt["summary"] != summary["questions"]:
        raise runner.PhaseEError("CODE question panel changed after preflight")


def question_ledger(
    run_id: str, panels: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "phase-g-code-question-ledger-1.0",
        "run_id": run_id,
        "profile_id": profile["profile_id"],
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


def build_parity_document(
    *,
    run_id: str,
    condition: str,
    panels: dict[str, Any],
    protected_output: dict[str, Any],
    protected_path: str,
    protected_sha256: str,
    replacement_output: dict[str, Any],
    replacement_path: str,
    replacement_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare newly executed part-of rows with the authenticated historical rows."""

    protected_indices, additional_indices = _panel_indices(panels)
    statistics: dict[str, Any] = {}
    modes: dict[str, Any] = {}
    for mode in evaluator.MODES:
        historical_rows = protected_output["results"][mode]
        combined_rows = replacement_output["results"][mode]
        part_of_rows = [combined_rows[index] for index in protected_indices]
        additional_rows = [combined_rows[index] for index in additional_indices]
        _validate_result_rows(
            historical_rows, panels["protected"], f"historical {condition}/{mode}"
        )
        _validate_result_rows(
            combined_rows, panels["combined"], f"replacement {condition}/{mode}"
        )
        equal = part_of_rows == historical_rows
        first_mismatch = None
        if not equal:
            for index, (historical, replacement) in enumerate(
                zip(historical_rows, part_of_rows)
            ):
                if historical != replacement:
                    first_mismatch = {
                        "protected_index": index,
                        "combined_index": protected_indices[index],
                        "historical": historical,
                        "replacement": replacement,
                    }
                    break
        modes[mode] = {
            "equal": equal,
            "compared_rows": len(historical_rows),
            "historical_rows_sha256": _rows_digest(historical_rows),
            "replacement_part_of_rows_sha256": _rows_digest(part_of_rows),
            "first_mismatch": first_mismatch,
        }
        statistics[mode] = {
            "part_of": _panel_statistics(part_of_rows),
            "additional": _panel_statistics(additional_rows),
            "combined": _panel_statistics(combined_rows),
        }
    parity = {
        "schema_version": "phase-g-code-question-parity-1.0",
        "run_id": run_id,
        "condition": condition,
        "historical": {
            "path": protected_path,
            "sha256": protected_sha256,
            "time_seconds": protected_output["metadata"]["time_seconds"],
        },
        "replacement": {
            "path": replacement_path,
            "sha256": replacement_sha256,
            "time_seconds": replacement_output["metadata"]["time_seconds"],
        },
        "protected_indices": protected_indices,
        "modes": modes,
        "overall_equal": all(item["equal"] for item in modes.values()),
    }
    return parity, statistics


def _validate_replacement_output(
    path: Path,
    runtime: dict[str, Any],
    condition: str,
    model_name: str,
) -> dict[str, Any]:
    return runner.validate_rag_output(
        path,
        {},
        model_name,
        expected_questions=_question_contract(runtime["panels"]["combined"]),
        expected_kg=runner.stable_child_path(runtime["protected"]["paths"][condition]),
        expected_count=len(runtime["panels"]["combined"]),
    )


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
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Validate every replacement artifact and its immutable parity oracle."""

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
    validate_child_inventory(
        runtime["protected"]["dir"], runtime["protected"]["inventory"]
    )
    expected_ledger = question_ledger(summary["run_id"], runtime["panels"], profile)
    if runner.load_json(child_dir / "question-ledger.json") != expected_ledger:
        raise runner.PhaseEError("CODE question ledger changed")
    expected_parent = {
        "schema_version": "phase-g-protected-table2-parent-1.0",
        "run_id": summary["run_id"],
        **summary["protected_child"],
    }
    if runner.load_json(child_dir / "protected-parent.json") != expected_parent:
        raise runner.PhaseEError("Protected Table-2 parent binding changed")
    expected_method = {
        "schema_version": "phase-g-code-question-method-1.0",
        "run_id": summary["run_id"],
        "identity": identity,
        "profile": profile,
        "base_contract": runner.load_json(runner.CONTRACT_PATH),
        "reuse": {
            "protected_results": "comparison-only",
            "projection": True,
            "graphs": True,
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
            "protected_environment_path",
        }
        or environment.get("schema_version")
        != "phase-g-code-question-environment-1.0"
        or environment.get("run_id") != summary["run_id"]
        or environment.get("captured_at") != status.get("created_at")
        or environment.get("source") != identity["source"]
        or environment.get("model_identity") != summary["model_identity"]
        or environment.get("protected_environment_path")
        != runner.stable_child_path(runtime["protected"]["dir"] / "environment.json")
        or not isinstance(environment.get("platform"), str)
        or not isinstance(environment.get("python"), str)
    ):
        raise runner.PhaseEError("CODE question environment manifest changed")
    condition_summaries = {}
    parity_summaries = {}
    for condition in ("confidence", "corrective", "gold"):
        protected_path = runtime["protected"]["dir"] / "rag-results" / f"{condition}.json"
        replacement_path = child_dir / "rag-results" / f"{condition}.json"
        replacement_validation = _validate_replacement_output(
            replacement_path,
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
        parity, statistics = build_parity_document(
            run_id=summary["run_id"],
            condition=condition,
            panels=runtime["panels"],
            protected_output=runtime["protected"]["result_documents"][condition],
            protected_path=runner.stable_child_path(protected_path),
            protected_sha256=runner.sha256_file(protected_path),
            replacement_output=replacement_output,
            replacement_path=runner.stable_child_path(replacement_path),
            replacement_sha256=runner.sha256_file(replacement_path),
        )
        if runner.load_json(child_dir / "part-of-parity" / f"{condition}.json") != parity:
            raise runner.PhaseEError(f"CODE question parity evidence changed: {condition}")
        condition_summaries[condition] = statistics
        parity_summaries[condition] = {
            "overall_equal": parity["overall_equal"],
            "modes": {
                mode: {
                    "equal": value["equal"],
                    "compared_rows": value["compared_rows"],
                    "historical_rows_sha256": value["historical_rows_sha256"],
                    "replacement_part_of_rows_sha256": value[
                        "replacement_part_of_rows_sha256"
                    ],
                }
                for mode, value in parity["modes"].items()
            },
        }
    result_summary = runner.load_json(child_dir / "code-question-results.json")
    expected_summary = {
        "schema_version": "phase-g-code-question-results-1.0",
        "run_id": summary["run_id"],
        "profile_id": profile["profile_id"],
        "question_summary": summary["questions"],
        "conditions": condition_summaries,
        "historical_part_of_parity": {
            "overall_equal": all(
                value["overall_equal"] for value in parity_summaries.values()
            ),
            "conditions": parity_summaries,
        },
        "note": (
            "Expanded Table-2-format diagnostic on canonical upstream outputs; "
            "not the paper's original n=10 Table 2."
        ),
    }
    if result_summary != expected_summary:
        raise runner.PhaseEError("CODE question result summary changed")
    if not expected_summary["historical_part_of_parity"]["overall_equal"]:
        raise runner.PhaseEError(
            "Replacement part-of results differ from the authenticated historical run"
        )
    hash_document = runner.load_json(child_dir / "artifact-hashes.json")
    if hash_document != _artifact_hash_document(child_dir, summary["run_id"]):
        raise runner.PhaseEError("CODE question artifact hashes differ")
    return expected_summary


def run_command(
    args: argparse.Namespace,
    base_contract: dict[str, Any],
    profile: dict[str, Any] | None = None,
) -> int:
    """Validate or execute the full-panel same-run replacement RAG."""

    profile = load_profile() if profile is None else validate_profile(profile)
    source = runner.validate_source_contract(
        base_contract, require_clean=not args.dry_run
    )
    run_dir, summary, runtime = preflight_code_questions(
        args.run_id, base_contract, profile
    )
    if args.dry_run:
        print(json.dumps({"source": source, **summary}, indent=2, sort_keys=True))
        return 0

    child_dir = run_dir / profile["output_namespace"]
    status_path = child_dir / "run-status.json"
    runner.validate_child_write_surface(child_dir)
    identity = build_identity(args.run_id, source, summary)
    new_child = not child_dir.exists() or not status_path.exists()
    if child_dir.exists() and not status_path.exists() and any(child_dir.iterdir()):
        raise runner.PhaseEError(
            f"{profile['output_namespace']} exists without a resumable status identity"
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
                child_dir, status, identity, summary, runtime, profile
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

    runner.write_json_once(
        child_dir / "question-ledger.json",
        question_ledger(args.run_id, runtime["panels"], profile),
        child_dir,
    )
    runner.write_json_once(
        child_dir / "protected-parent.json",
        {
            "schema_version": "phase-g-protected-table2-parent-1.0",
            "run_id": args.run_id,
            **summary["protected_child"],
        },
        child_dir,
    )
    runner.write_json_once(
        child_dir / "method-manifest.json",
        {
            "schema_version": "phase-g-code-question-method-1.0",
            "run_id": args.run_id,
            "identity": identity,
            "profile": profile,
            "base_contract": base_contract,
            "reuse": {
                "protected_results": "comparison-only",
                "projection": True,
                "graphs": True,
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
            "protected_environment_path": runner.stable_child_path(
                runtime["protected"]["dir"] / "environment.json"
            ),
        },
        child_dir,
    )

    replacement_outputs: dict[str, dict[str, Any]] = {}
    for condition in ("confidence", "corrective", "gold"):
        stage_name = f"rag_{condition}"
        validate_execution_snapshot(
            identity, source, summary, runtime, base_contract, profile
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
        kg_path = runtime["protected"]["paths"][condition]
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
                identity, source, summary, runtime, base_contract, profile
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
    parity_summaries = {}
    for condition in ("confidence", "corrective", "gold"):
        protected_path = runtime["protected"]["dir"] / "rag-results" / f"{condition}.json"
        replacement_path = child_dir / "rag-results" / f"{condition}.json"
        parity, statistics = build_parity_document(
            run_id=args.run_id,
            condition=condition,
            panels=runtime["panels"],
            protected_output=runtime["protected"]["result_documents"][condition],
            protected_path=runner.stable_child_path(protected_path),
            protected_sha256=runner.sha256_file(protected_path),
            replacement_output=replacement_outputs[condition],
            replacement_path=runner.stable_child_path(replacement_path),
            replacement_sha256=runner.sha256_file(replacement_path),
        )
        runner.write_json_once(
            child_dir / "part-of-parity" / f"{condition}.json", parity, child_dir
        )
        condition_summaries[condition] = statistics
        parity_summaries[condition] = {
            "overall_equal": parity["overall_equal"],
            "modes": {
                mode: {
                    "equal": value["equal"],
                    "compared_rows": value["compared_rows"],
                    "historical_rows_sha256": value["historical_rows_sha256"],
                    "replacement_part_of_rows_sha256": value[
                        "replacement_part_of_rows_sha256"
                    ],
                }
                for mode, value in parity["modes"].items()
            },
        }

    result_summary = {
        "schema_version": "phase-g-code-question-results-1.0",
        "run_id": args.run_id,
        "profile_id": profile["profile_id"],
        "question_summary": summary["questions"],
        "conditions": condition_summaries,
        "historical_part_of_parity": {
            "overall_equal": all(
                value["overall_equal"] for value in parity_summaries.values()
            ),
            "conditions": parity_summaries,
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
        identity, source, summary, runtime, base_contract, profile
    )
    _final_run, final_summary, _final_runtime = preflight_code_questions(
        args.run_id, base_contract, profile
    )
    if (
        final_summary["base_preflight_identity"] != summary["base_preflight_identity"]
        or final_summary["protected_child"] != summary["protected_child"]
        or final_summary["questions"] != summary["questions"]
    ):
        raise runner.PhaseEError("Same-run parent changed during CODE question execution")
    if not result_summary["historical_part_of_parity"]["overall_equal"]:
        status["status"] = "parity-failed"
        status["finished_at"] = runner.utc_now()
        status["historical_part_of_parity"] = result_summary[
            "historical_part_of_parity"
        ]
        runner.write_status(status_path, status, child_dir)
        raise runner.PhaseEError(
            "Replacement part-of results differ from the authenticated historical run; "
            "outputs were retained for review"
        )
    completed = dict(status)
    completed["status"] = "complete"
    completed["finished_at"] = runner.utc_now()
    validate_extension_child(
        child_dir, completed, identity, summary, runtime, profile
    )
    status.clear()
    status.update(completed)
    runner.write_status(status_path, status, child_dir)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "child": profile["output_namespace"],
                "status": "complete",
            },
            indent=2,
        )
    )
    return 0
