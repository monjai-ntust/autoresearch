"""Focused tests for the Phase G original-CODE replacement RAG path."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from utils.rag import code_questions, evaluator, runner


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "resources/contracts/table2.json").read_text(encoding="utf-8")
)
RUN_ID = "code-questions-test"
ORIGINAL_CODE_EVALUATOR_BLOB = "c06b753433681f13140d288c7e642c20e1eab851"


def original_code_evaluator():
    source = subprocess.run(
        ["git", "cat-file", "blob", ORIGINAL_CODE_EVALUATOR_BLOB],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    module = types.ModuleType("original_code_evaluator")
    exec(compile(source, "<original-code-evaluator>", "exec"), module.__dict__)
    return module


def record(index: int, triples: list[tuple[str, str, str]], sentence: str | None = None):
    return {
        "doc_id": f"doc-{index}",
        "sentence": sentence or f"Sentence {index}",
        "gold_triples": [
            {"head_text": head, "relation": relation, "tail_text": tail}
            for head, relation, tail in triples
        ],
        "predicted_triples": [],
    }


def model_identity() -> dict:
    return {
        "name": "example-model:tag",
        "tag_digest": "4" * 64,
        "blob_sha256": "5" * 64,
        "details": {
            "family": "qwen3",
            "parameter_size": "32.8B",
            "quantization_level": "Q4_K_M",
        },
        "verifier_environment_sha256": "6" * 64,
        "verifier_condition_id": "VER-CORRECTIVE",
        "verifier_protocol_id": "B04-PATH-A-1.3",
    }


def model_capture(identity: dict) -> tuple[dict, bytes, bytes]:
    tags = {"models": [{"name": identity["name"], "digest": identity["tag_digest"]}]}
    show = {
        "details": identity["details"],
        "modelfile": f"FROM /models/blobs/sha256-{identity['blob_sha256']}\n",
    }
    tags_bytes = runner.canonical_json_bytes(tags)
    show_bytes = runner.canonical_json_bytes(show)
    return (
        {
            **identity,
            "identity_verified": True,
            "tags_response_sha256": runner.sha256_bytes(tags_bytes),
            "show_response_sha256": runner.sha256_bytes(show_bytes),
        },
        tags_bytes,
        show_bytes,
    )


def rag_output(
    questions: list[dict], *, kg: str, model: str, prefix: str
) -> dict:
    results = {
        mode: [
            {
                "q": question["question"],
                "pred": question["gold_answer"],
                "gold": question["gold_answer"],
                "correct": True,
            }
            for question in questions
        ]
        for mode in evaluator.MODES
    }
    return {
        "metadata": {
            "kg": kg,
            "n_questions": len(questions),
            "model": model,
            "time_seconds": 1.0 if prefix == "protected" else 2.0,
        },
        "accuracy": {mode: 1.0 for mode in evaluator.MODES},
        "correct_counts": {mode: len(questions) for mode in evaluator.MODES},
        "results": results,
    }


class CodeQuestionTests(unittest.TestCase):
    def setUp(self):
        self.method = code_questions.replacement_method()

    def test_method_matches_historical_templates_and_keeps_protected_generator(self):
        records = [
            record(
                1,
                [
                    ("Part", "part-of", "Whole"),
                    ("Need", "NeCeSsItY", "First"),
                    ("Need", "necessity", "Second"),
                    ("Choice", "selection", "Selected"),
                    ("Equal", "equal", "Peer"),
                    ("Floor", "greater-equal", "10"),
                    ("Ceiling", "less-equal", "20"),
                    ("Bound", "greater", "5"),
                    ("Ignored", "conjunction", "Other"),
                ],
            )
        ]
        protected = evaluator.generate_questions(records)
        panels = code_questions.build_question_panels(records)
        self.assertEqual([row["relation"] for row in protected], ["part-of"])
        self.assertEqual(protected, panels["protected"])
        self.assertEqual(panels["summary"]["relation_counts"]["less"], 0)
        necessity = [
            row for row in panels["combined"] if row["relation"] == "necessity"
        ]
        self.assertEqual(len(necessity), 1)
        self.assertEqual(necessity[0]["gold_answer"], "First")
        self.assertEqual(
            tuple(
                (item["relation"], item["question"], item["answer"])
                for item in self.method["additional_templates"]
            ),
            code_questions.EXPECTED_ADDITIONAL_TEMPLATES,
        )
        historical = original_code_evaluator().generate_questions(records, 0)
        self.assertEqual(panels["combined"], historical)

    def test_protected_and_replacement_share_one_evaluator_kernel(self):
        self.assertIs(runner.evaluator, evaluator)
        self.assertIs(code_questions.evaluator, evaluator)
        replacement_source = (ROOT / "utils/rag/code_questions.py").read_text(
            encoding="utf-8"
        )
        for duplicated_primitive in (
            "def call_llm(",
            "def retrieve_from_kg(",
            "def retrieve_sentences(",
            "def check_answer(",
            "Answer in 1-2 words.",
        ):
            self.assertNotIn(duplicated_primitive, replacement_source)

    def test_added_panel_uses_full_record_pool_and_frozen_mode_order(self):
        records = [
            record(
                1,
                [("Component", "part-of", "Assembly")],
                sentence="Widget Power appears in protected context.",
            ),
            record(
                2,
                [("Widget", "necessity", "Power")],
                sentence="Unrelated source sentence.",
            ),
        ]
        panels = code_questions.build_question_panels(records)
        prompts = []

        def call(_url, _model, prompt):
            prompts.append(prompt)
            return "Power"

        output = evaluator.evaluate(
            records,
            {"edges": []},
            kg_identity="output/example/graph.json",
            ollama_url="http://localhost:11434",
            ollama_model="example-model:tag",
            llm_call=call,
            question_panel=panels["added"],
        )
        self.assertEqual(len(prompts), 5)
        self.assertEqual(prompts[0], "Answer in 1-2 words. What is required for Widget?")
        self.assertIn("Widget Power appears in protected context.", prompts[1])
        self.assertEqual(list(output["results"]), list(evaluator.MODES))

    def test_method_is_source_defined_without_a_duplicate_resource(self):
        changed = code_questions.replacement_method()
        changed["additional_templates"][0]["question"] = "Changed {head}?"
        self.assertEqual(
            code_questions.replacement_method()["additional_templates"][0]["question"],
            "What is required for {head}?",
        )
        self.assertFalse((ROOT / "resources/contracts/code-questions.json").exists())

    def test_parity_extracts_fresh_part_of_rows_in_combined_order(self):
        records = [
            record(1, [("Need", "necessity", "Power")]),
            record(2, [("Part", "part-of", "Whole")]),
            record(3, [("Choice", "selection", "Option")]),
            record(4, [("Piece", "part-of", "Set")]),
        ]
        panels = code_questions.build_question_panels(records)
        protected = rag_output(
            panels["protected"], kg="protected-graph", model="model", prefix="protected"
        )
        replacement = rag_output(
            panels["combined"], kg="protected-graph", model="model", prefix="added"
        )
        parity, statistics = code_questions.build_parity_document(
            run_id=RUN_ID,
            condition="gold",
            panels=panels,
            protected_output=protected,
            protected_path="output/run/table2-q105/rag-results/gold.json",
            protected_sha256="a" * 64,
            replacement_output=replacement,
            replacement_path="output/run/table2-code-all/rag-results/gold.json",
            replacement_sha256="b" * 64,
        )
        self.assertTrue(parity["overall_equal"])
        self.assertEqual(parity["protected_indices"], [1, 3])
        self.assertTrue(parity["modes"]["llm_only"]["equal"])
        self.assertEqual(statistics["llm_only"]["combined"]["n"], 4)
        self.assertEqual(statistics["llm_only"]["part_of"]["n"], 2)

        replacement["results"]["hybrid"][1]["pred"] = "different"
        replacement["results"]["hybrid"][1]["correct"] = False
        parity, _statistics = code_questions.build_parity_document(
            run_id=RUN_ID,
            condition="gold",
            panels=panels,
            protected_output=protected,
            protected_path="historical.json",
            protected_sha256="a" * 64,
            replacement_output=replacement,
            replacement_path="replacement.json",
            replacement_sha256="b" * 64,
        )
        self.assertFalse(parity["overall_equal"])
        self.assertEqual(
            parity["modes"]["hybrid"]["first_mismatch"]["combined_index"], 1
        )

    def test_complete_protected_inventory_detects_any_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            child = Path(temporary)
            path = child / "rag-results" / "confidence.json"
            path.parent.mkdir()
            path.write_text("original", encoding="utf-8")
            inventory = code_questions.child_inventory(child)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(runner.PhaseEError, "inventory changed"):
                code_questions.validate_child_inventory(child, inventory)

    def test_dry_run_has_no_model_call_or_output_write(self):
        args = argparse.Namespace(
            run_id=RUN_ID, ollama_url="http://localhost:11434", dry_run=True
        )
        summary = {"run_id": RUN_ID, "questions": {"added_count": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with mock.patch.object(
                runner, "validate_source_contract", return_value={"head": "a" * 40}
            ), mock.patch.object(
                code_questions,
                "preflight_code_questions",
                return_value=(run_dir, summary, {}),
            ), mock.patch.object(
                runner, "fetch_matching_model"
            ) as fetch, redirect_stdout(io.StringIO()):
                self.assertEqual(
                    code_questions.run_command(args, CONTRACT), 0
                )
            fetch.assert_not_called()
            self.assertFalse((run_dir / code_questions.OUTPUT_NAMESPACE).exists())

    def test_replacement_executes_full_panel_and_resumes(self):
        records = [
            record(1, [("Need", "necessity", "Power")], "Need Power context."),
            record(2, [("Part", "part-of", "Whole")], "Part Whole context."),
        ]
        panels = code_questions.build_question_panels(records)
        output_root = ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as temporary:
            run_dir = Path(temporary).resolve()
            protected_dir = run_dir / code_questions.PROTECTED_NAMESPACE
            projection_dir = protected_dir / "projections"
            result_dir = protected_dir / "rag-results"
            projection_dir.mkdir(parents=True)
            result_dir.mkdir()
            paths = {
                "records": projection_dir / "test-with-private-gold.jsonl",
                "confidence": projection_dir / "confidence-seed-42.json",
                "corrective": projection_dir / "corrective-seed-42.json",
                "gold": projection_dir / "gold-oracle.json",
            }
            paths["records"].write_bytes(runner.canonical_jsonl_bytes(records))
            for condition in ("confidence", "corrective", "gold"):
                paths[condition].write_bytes(
                    runner.canonical_json_bytes({"edges": [], "condition": condition})
                )
                output = rag_output(
                    panels["protected"],
                    kg=runner.stable_child_path(paths[condition]),
                    model=model_identity()["name"],
                    prefix="protected",
                )
                (result_dir / f"{condition}.json").write_bytes(
                    runner.scientific_json_bytes(output)
                )
            (protected_dir / "environment.json").write_bytes(
                runner.canonical_json_bytes({"protected": True})
            )
            inventory = code_questions.child_inventory(protected_dir)
            protected_results = {
                condition: runner.load_json(result_dir / f"{condition}.json")
                for condition in ("confidence", "corrective", "gold")
            }
            identity = model_identity()
            base_preflight = {
                "run_id": RUN_ID,
                "parent_artifact_set_sha256": "1" * 64,
                "score_manifest_sha256": "2" * 64,
                "seed": 42,
                "threshold": 0.25,
                "projection_sha256": {
                    name: runner.sha256_file(path) for name, path in paths.items()
                },
                "graphs": {
                    name: {"graph_id": character * 64, "source_sha256": character * 64}
                    for name, character in zip(("confidence", "corrective", "gold"), "789")
                },
                "model_identity": identity,
                "observed_profile": {
                    "encoder": {"base_model_revision": "a" * 40}
                },
            }
            summary = {
                "run_id": RUN_ID,
                "run_dir": str(run_dir),
                "lineage_status": "lineage-valid",
                "profile_id": code_questions.PROFILE_ID,
                "method_sha256": runner.sha256_bytes(
                    runner.canonical_json_bytes(self.method)
                ),
                "base_contract_sha256": runner.sha256_file(runner.CONTRACT_PATH),
                "base_preflight_identity": runner._preflight_identity(base_preflight),
                "protected_child": {
                    "namespace": code_questions.PROTECTED_NAMESPACE,
                    "identity": {"source_head": "old-source"},
                    "artifact_hashes_sha256": "3" * 64,
                    "inventory_sha256": inventory["inventory_sha256"],
                    "files": inventory["files"],
                },
                "questions": panels["summary"],
                "method": self.method,
                "model_identity": identity,
                "environment": {},
            }
            runtime = {
                "base_preflight": base_preflight,
                "projections": {},
                "graphs": {},
                "records": records,
                "panels": panels,
                "protected": {
                    "dir": protected_dir,
                    "paths": paths,
                    "identity": summary["protected_child"]["identity"],
                    "inventory": inventory,
                    "result_documents": protected_results,
                },
            }
            args = argparse.Namespace(
                run_id=RUN_ID,
                ollama_url="http://localhost:11434",
                dry_run=False,
            )
            source = {
                "head": "a" * 40,
                "branch": "publication-refactored-rag",
                "baseline": "b" * 40,
            }
            capture = model_capture(identity)
            calls = []

            def evaluate(all_records, _kg, *, kg_identity, question_panel, **_kwargs):
                calls.append((all_records, question_panel, kg_identity))
                return rag_output(
                    question_panel,
                    kg=kg_identity,
                    model=identity["name"],
                    prefix="added",
                )

            protected_before = code_questions.child_inventory(protected_dir)
            with mock.patch.object(
                runner, "validate_source_contract", return_value=source
            ), mock.patch.object(
                code_questions,
                "preflight_code_questions",
                return_value=(run_dir, summary, runtime),
            ), mock.patch.object(
                code_questions, "validate_execution_snapshot"
            ), mock.patch.object(
                runner, "fetch_matching_model", return_value=capture
            ) as fetch, mock.patch.object(
                evaluator, "evaluate", side_effect=evaluate
            ) as execute, redirect_stdout(io.StringIO()):
                self.assertEqual(
                    code_questions.run_command(args, CONTRACT), 0
                )
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(item[0] == records for item in calls))
                self.assertTrue(all(item[1] == panels["combined"] for item in calls))
                self.assertEqual(fetch.call_count, 4)
                calls.clear()
                fetch.reset_mock()
                execute.reset_mock()
                self.assertEqual(
                    code_questions.run_command(args, CONTRACT), 0
                )
                fetch.assert_not_called()
                execute.assert_not_called()
            self.assertEqual(
                code_questions.child_inventory(protected_dir), protected_before
            )
            combined = runner.load_json(
                run_dir
                / code_questions.OUTPUT_NAMESPACE
                / "rag-results/confidence.json"
            )
            historical = protected_results["confidence"]["results"]["llm_only"][0]
            self.assertIn(historical, combined["results"]["llm_only"])

    def test_public_runner_dispatches_to_replacement(self):
        args = argparse.Namespace(
            run_id=RUN_ID, ollama_url="http://localhost:11434", dry_run=True
        )
        with mock.patch.object(code_questions, "run_command", return_value=7) as call:
            self.assertEqual(runner.run_command(args, CONTRACT), 7)
        call.assert_called_once_with(args, CONTRACT)


if __name__ == "__main__":
    unittest.main()
