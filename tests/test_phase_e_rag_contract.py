import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "phase_e_rag_contract.json").read_text(encoding="utf-8"))


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "phase_e_eval_graph_rag", ROOT / "eval_graph_rag.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhaseERagContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

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
        for index, expected_item in enumerate(expected):
            relation, question, answer = expected_item
            record = {
                "doc_id": index,
                "sentence": f"Exact contract record {index}.",
                "gold_triples": [
                    {
                        "head_text": "Head",
                        "tail_text": "Tail",
                        "relation": relation.upper(),
                    }
                ],
            }

            with self.subTest(relation=relation):
                generated = self.evaluator.generate_questions([record], max_q=0)
                self.assertEqual(len(generated), 1)
                self.assertEqual(
                    (
                        generated[0]["relation"],
                        generated[0]["question"],
                        generated[0]["gold_answer"],
                    ),
                    (relation, question, answer),
                )

    def test_later_code_specific_templates_are_rejected(self):
        excluded = [
            "necessity",
            "selection",
            "equal",
            "greater-equal",
            "less-equal",
            "greater",
            "less",
        ]
        records = [
            {
                "doc_id": index,
                "sentence": "Excluded post-table template.",
                "gold_triples": [
                    {
                        "head_text": "Head",
                        "tail_text": "Tail",
                        "relation": relation,
                    }
                ],
            }
            for index, relation in enumerate(excluded)
        ]

        self.assertEqual(self.evaluator.generate_questions(records, max_q=0), [])

    def test_seeded_table_question_selection_matches_real_projection(self):
        fixture_path = ROOT / "tests" / "fixtures" / "accord_table2_part_of_projection.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        questions = self.evaluator.generate_questions(
            fixture["records"], CONTRACT["evaluator"]["max_questions"]
        )

        self.assertEqual(
            [q["question"] for q in questions], fixture["expected_seed42_questions"]
        )

    def test_five_runtime_prompts_are_byte_exact(self):
        sentence = (
            "If insulating material is inserted into a cavity in a cavity wall , "
            "reasonable precautions shall be taken to prevent the subsequent permeation "
            "of any toxic fumes from that material into any part of the building occupied "
            "by people ."
        )
        record = {
            "doc_id": 5,
            "sentence": sentence,
            "gold_triples": [
                {
                    "head_text": "cavity",
                    "tail_text": "cavity wall",
                    "relation": "part-of",
                }
            ],
        }
        graph = {
            "nodes": [{"id": "cavity"}, {"id": "cavity wall"}],
            "edges": [
                {
                    "head": "cavity",
                    "relation": "part-of",
                    "tail": "cavity wall",
                }
            ],
        }
        question = "What is cavity part of?"
        triple = "(cavity, part-of, cavity wall)"
        clipped = sentence[:200]
        expected = [
            f"Answer in 1-2 words. {question}",
            f"Based on these scientific sentences:\n- {clipped}\n\n"
            f"Answer in 1-2 words: {question}",
            f"Based on this knowledge graph:\n- {triple}\n\n"
            f"Answer in 1-2 words: {question}",
            f"Based on this knowledge graph:\n- {triple}\n\n"
            f"Answer in 1-2 words: {question}",
            f"Based on this evidence:\nKnowledge graph:\n- {triple}\n\n"
            f"Source text:\n- {clipped}\n\nAnswer in 1-2 words: {question}",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            graph_path = tmp_path / "graph.json"
            output_path = tmp_path / "result.json"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            captured = []

            def fake_call(_url, _model, prompt):
                captured.append(prompt)
                return "cavity wall"

            argv = [
                "eval_graph_rag.py",
                "--kg",
                str(graph_path),
                "--gold-jsonl",
                str(input_path),
                "--max-questions",
                "1",
                "--output",
                str(output_path),
            ]
            with mock.patch.object(self.evaluator, "call_llm", side_effect=fake_call):
                with mock.patch.object(sys, "argv", argv):
                    with redirect_stdout(io.StringIO()):
                        self.evaluator.main()

            self.assertEqual(captured, expected)
            self.assertTrue(output_path.is_file())

    def test_call_payload_is_exact(self):
        completed = mock.Mock(stdout='{"message":{"content":" answer "}}')
        with mock.patch.object(
            self.evaluator.subprocess, "run", return_value=completed
        ) as run:
            answer = self.evaluator.call_llm(
                "http://localhost:11434", "qwen3:32b", "PROMPT"
            )

        self.assertEqual(answer, "answer")
        args, kwargs = run.call_args
        self.assertEqual(
            args[0],
            [
                "curl",
                "-s",
                "http://localhost:11434/api/chat",
                "-d",
                '{"model": "qwen3:32b", "messages": [{"role": "user", '
                '"content": "PROMPT"}], "stream": false, "think": false, '
                '"options": {"temperature": 0.0, "num_predict": 50}}',
            ],
        )
        self.assertEqual(
            kwargs, {"capture_output": True, "text": True, "timeout": 60}
        )

    def test_empty_question_set_fails_without_model_or_output(self):
        record = {
            "doc_id": 1,
            "sentence": "Unsupported relation only.",
            "gold_triples": [
                {
                    "head_text": "Head",
                    "tail_text": "Tail",
                    "relation": "necessity",
                }
            ],
        }
        graph = {"nodes": [], "edges": []}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            graph_path = tmp_path / "graph.json"
            output_path = tmp_path / "result.json"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            argv = [
                "eval_graph_rag.py",
                "--kg",
                str(graph_path),
                "--gold-jsonl",
                str(input_path),
                "--output",
                str(output_path),
            ]

            with mock.patch.object(self.evaluator, "call_llm") as call_llm:
                with mock.patch.object(sys, "argv", argv):
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(
                            SystemExit, "No supported questions were generated"
                        ):
                            self.evaluator.main()

            call_llm.assert_not_called()
            self.assertFalse(output_path.exists())


class PhaseENoTouchTests(unittest.TestCase):
    def test_frozen_upstream_git_blobs_are_unchanged(self):
        for path, expected in CONTRACT["frozen_upstream_blobs"].items():
            actual = subprocess.run(
                ["git", "hash-object", f"--path={path}", path],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(actual, expected, path)

    def test_graph_builder_git_blob_is_unchanged(self):
        actual = subprocess.run(
            ["git", "hash-object", "--path=build_kg.py", "build_kg.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(actual, CONTRACT["graph_builder"]["blob"])

    def test_evaluator_has_only_the_approved_post_restore_guard(self):
        actual = subprocess.run(
            ["git", "hash-object", "--path=eval_graph_rag.py", "eval_graph_rag.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(actual, CONTRACT["evaluator"]["post_bf02_blob"])

        guard = (
            "    if not questions:\n"
            "        raise SystemExit(\n"
            '            "No supported questions were generated; refusing to write an empty RAG result."\n'
            "        )\n"
        )
        current = (ROOT / "eval_graph_rag.py").read_text(encoding="utf-8")
        self.assertEqual(current.count(guard), 1)
        table_era = subprocess.run(
            ["git", "cat-file", "blob", CONTRACT["evaluator"]["table_era_blob"]],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")
        self.assertEqual(current.replace(guard, "", 1), table_era)


if __name__ == "__main__":
    unittest.main()
