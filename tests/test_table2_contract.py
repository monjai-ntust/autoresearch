"""Characterization tests for the frozen Table-2 evaluator method."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from utils.pipeline.rag import evaluator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "resources/contracts/table2.json").read_text(encoding="utf-8")
)
TABLE_ERA_EVALUATOR_BLOB = "964893546b28f04e34e8c546bcbbfd4cfbc27354"


def historical_evaluator():
    source = subprocess.run(
        ["git", "cat-file", "blob", TABLE_ERA_EVALUATOR_BLOB],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    module = types.ModuleType("historical_table_evaluator")
    exec(compile(source, "<historical-table-evaluator>", "exec"), module.__dict__)
    return module


class Table2EvaluatorContractTests(unittest.TestCase):
    def test_all_table_era_relation_questions_are_exact(self):
        expected = [
            ("used-for", "What is Head used for?", "Tail"),
            ("feature-of", "What is a feature of Tail?", "Head"),
            ("hyponym-of", "What is Head a type of?", "Tail"),
            ("evaluate-for", "What is Head evaluated for?", "Tail"),
            ("evaluated-with", "What is Head evaluated with?", "Tail"),
            ("part-of", "What is Head part of?", "Tail"),
            ("compare", "What is Head compared with?", "Tail"),
            ("compare-with", "What is Head compared with?", "Tail"),
            ("trained-with", "What is Head trained with?", "Tail"),
            ("subclass-of", "What is Head a subclass of?", "Tail"),
            ("subtask-of", "What is Head a subtask of?", "Tail"),
            ("synonym-of", "What is a synonym of Head?", "Tail"),
            ("benchmark-for", "What is Head a benchmark for?", "Tail"),
        ]
        for index, (relation, question, answer) in enumerate(expected):
            record = {
                "doc_id": index,
                "sentence": f"Exact contract record {index}.",
                "gold_triples": [{
                    "head_text": "Head",
                    "tail_text": "Tail",
                    "relation": relation.upper(),
                }],
            }
            with self.subTest(relation=relation):
                generated = evaluator.generate_questions([record])
                self.assertEqual(
                    (
                        generated[0]["relation"],
                        generated[0]["question"],
                        generated[0]["gold_answer"],
                    ),
                    (relation, question, answer),
                )

    def test_later_code_specific_templates_remain_excluded(self):
        records = [
            {
                "doc_id": index,
                "sentence": "Excluded post-table template.",
                "gold_triples": [{
                    "head_text": "Head",
                    "tail_text": "Tail",
                    "relation": relation,
                }],
            }
            for index, relation in enumerate(
                ["necessity", "selection", "equal", "greater-equal", "less-equal", "greater", "less"]
            )
        ]
        self.assertEqual(evaluator.generate_questions(records), [])

    def test_full_question_panel_keeps_every_fixture_question(self):
        fixture = json.loads(
            (ROOT / "tests/fixtures/accord_table2_part_of_projection.json").read_text(
                encoding="utf-8"
            )
        )
        questions = evaluator.generate_questions(fixture["records"])
        self.assertEqual(CONTRACT["evaluator"]["max_questions"], 105)
        self.assertEqual(len(questions), len(fixture["records"]))

    def test_full_question_panel_includes_all_105_eligible_questions(self):
        records = [
            {
                "doc_id": index,
                "sentence": f"Sentence {index}.",
                "gold_triples": [{
                    "head_text": f"Head {index}",
                    "tail_text": f"Tail {index}",
                    "relation": "part-of",
                }],
            }
            for index in range(105)
        ]
        actual = evaluator.generate_questions(records)
        self.assertEqual(len(actual), 105)

    def test_legacy_question_limit_is_removed(self):
        records = [
            {
                "doc_id": index,
                "sentence": f"Sentence {index}.",
                "gold_triples": [{
                    "head_text": f"Head {index}",
                    "tail_text": f"Tail {index}",
                    "relation": "part-of",
                }],
            }
            for index in range(106)
        ]
        questions = evaluator.generate_questions(records)
        self.assertEqual(len(questions), 106)
        with mock.patch.object(evaluator, "call_llm") as call:
            with self.assertRaisesRegex(ValueError, "complete 105-question panel"):
                evaluator.evaluate(
                    records,
                    {"nodes": [], "edges": []},
                    kg_identity="fixture",
                    ollama_url="http://localhost:11434",
                    ollama_model="qwen3:32b",
                    max_questions=CONTRACT["evaluator"]["max_questions"],
                )
        call.assert_not_called()

    def test_internal_transform_matches_authoritative_historical_execution(self):
        historical = historical_evaluator()
        sentence = (
            "If insulating material is inserted into a cavity in a cavity wall , "
            "reasonable precautions shall be taken to prevent the subsequent permeation "
            "of any toxic fumes from that material into any part of the building occupied "
            "by people ."
        )
        record = {
            "doc_id": 5,
            "sentence": sentence,
            "gold_triples": [{
                "head_text": "cavity",
                "tail_text": "cavity wall",
                "relation": "part-of",
            }],
        }
        graph = {
            "nodes": [{"id": "cavity"}, {"id": "cavity wall"}],
            "edges": [{"head": "cavity", "relation": "part-of", "tail": "cavity wall"}],
        }
        historical_prompts: list[str] = []
        current_prompts: list[str] = []

        def historical_call(_url, _model, prompt):
            historical_prompts.append(prompt)
            return "cavity wall"

        def current_call(_url, _model, prompt):
            current_prompts.append(prompt)
            return "cavity wall"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path = root / "records.jsonl"
            graph_path = root / "graph.json"
            output_path = root / "result.json"
            records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            argv = [
                "historical-evaluator",
                "--kg", str(graph_path),
                "--gold-jsonl", str(records_path),
                "--max-questions", "1",
                "--output", str(output_path),
            ]
            historical.call_llm = historical_call
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                historical.main()
            expected = json.loads(output_path.read_text(encoding="utf-8"))

        actual = evaluator.evaluate(
            [record],
            graph,
            kg_identity=str(graph_path),
            ollama_url="http://localhost:11434",
            ollama_model="qwen3:32b",
            max_questions=1,
            llm_call=current_call,
        )
        self.assertEqual(current_prompts, historical_prompts)
        expected["metadata"].pop("time_seconds")
        actual["metadata"].pop("time_seconds")
        self.assertEqual(actual, expected)

    def test_call_payload_is_byte_exact(self):
        completed = mock.Mock(stdout='{"message":{"content":" answer "}}')
        with mock.patch.object(evaluator.subprocess, "run", return_value=completed) as run:
            answer = evaluator.call_llm(
                "http://localhost:11434", "qwen3:32b", "PROMPT"
            )
        self.assertEqual(answer, "answer")
        args, kwargs = run.call_args
        self.assertEqual(
            args[0],
            [
                "curl", "-s", "http://localhost:11434/api/chat", "-d",
                '{"model": "qwen3:32b", "messages": [{"role": "user", '
                '"content": "PROMPT"}], "stream": false, "think": false, '
                '"options": {"temperature": 0.0, "num_predict": 50}}',
            ],
        )
        self.assertEqual(kwargs, {"capture_output": True, "text": True, "timeout": 60})

    def test_empty_question_set_fails_before_a_model_call(self):
        record = {
            "doc_id": 1,
            "sentence": "Unsupported relation only.",
            "gold_triples": [{
                "head_text": "Head", "tail_text": "Tail", "relation": "necessity"
            }],
        }
        with mock.patch.object(evaluator, "call_llm") as call:
            with self.assertRaisesRegex(ValueError, "No supported questions"):
                evaluator.evaluate(
                    [record],
                    {"nodes": [], "edges": []},
                    kg_identity="fixture",
                    ollama_url="http://localhost:11434",
                    ollama_model="qwen3:32b",
                    max_questions=CONTRACT["evaluator"]["max_questions"],
                )
        call.assert_not_called()

    def test_internal_modules_expose_no_file_selector_or_entrypoint(self):
        for relative in (
            "utils/pipeline/rag/evaluator.py",
            "utils/pipeline/rag/graph.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("ArgumentParser", source)
            self.assertNotIn("__main__", source)


if __name__ == "__main__":
    unittest.main()
