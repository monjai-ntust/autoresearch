"""Fresh-process retrieval is stable before graph and hybrid fact limits."""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import rag
from utils.rag import evaluator, requests


ROOT = Path(__file__).resolve().parents[1]


def edge(head, tail, relation="part-of"):
    return {"head": head, "relation": relation, "tail": tail}


class RagDeterminismTests(unittest.TestCase):
    def _validate_options(self, contract):
        evaluator._configured_options.cache_clear()
        self.addCleanup(evaluator._configured_options.cache_clear)
        configuration = mock.Mock()
        configuration.read_text.return_value = json.dumps(contract)
        path = mock.MagicMock()
        path.return_value.resolve.return_value.parents.__getitem__.return_value.__truediv__.return_value = configuration
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(evaluator, "Path", path), \
             mock.patch.object(rag.runner, "validate_source_contract", return_value={"head": "fixture"}), \
             mock.patch.object(rag, "RunLayout") as layout, \
             mock.patch.object(rag.runner, "fetch_ollama_json") as server, \
             mock.patch.object(evaluator, "call_llm") as model, \
             redirect_stdout(stdout), redirect_stderr(stderr):
            result = rag.main(["validate"])
            layout.assert_not_called(); server.assert_not_called(); model.assert_not_called()
        return result, stdout.getvalue(), stderr.getvalue()

    def test_public_validate_checks_configured_seed_without_copying_settings_or_accessing_server(self):
        options = {"temperature": 0.0, "num_predict": 50, "seed": 17}
        result, stdout, stderr = self._validate_options({"generation_options": options})
        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "RAG configuration valid.\n")

    def test_public_validate_rejects_missing_or_malformed_generation_options(self):
        valid = {"temperature": 0.0, "num_predict": 50, "seed": 17}
        for options in (None, {"temperature": 0.0, "num_predict": 50},
                        {**valid, "seed": False}, {**valid, "seed": -1},
                        {**valid, "seed": 2**31}, {**valid, "temperature": 0},
                        {**valid, "num_predict": 51}, {**valid, "extra": "unsupported"}):
            with self.subTest(options=options):
                result, stdout, stderr = self._validate_options(
                    {} if options is None else {"generation_options": options})
                self.assertEqual(result, 2)
                self.assertEqual(stdout, "")
                self.assertIn("RAG generation configuration", stderr)

    def test_matching_nodes_and_adjacent_edges_keep_first_appearance(self):
        graph = {"edges": [edge("Core Z", "first"), edge("Core A", "second"),
                           edge("Core Z", "third")]}
        original = copy.deepcopy(graph)
        self.assertEqual(evaluator.retrieve_from_kg(graph, " Core "), [
            "(Core Z, part-of, first)", "(Core Z, part-of, third)",
            "(Core A, part-of, second)"])
        self.assertEqual(graph, original)

    def test_duplicates_self_loops_and_cycles_do_not_repeat_facts(self):
        graph = {"edges": [edge("Anchor", "Bridge"), edge("Bridge", "Far"),
                           edge("Far", "Anchor"), edge("Anchor", "Bridge"),
                           edge("Anchor", "Anchor")]}
        self.assertEqual(evaluator.retrieve_from_kg(graph, "anchor", hops=8), [
            "(Anchor, part-of, Bridge)", "(Far, part-of, Anchor)",
            "(Anchor, part-of, Anchor)", "(Bridge, part-of, Far)"])
        self.assertEqual(evaluator.retrieve_from_kg(graph, "anchor", hops=0), [])

    def test_second_hop_keeps_endpoint_discovery_order(self):
        graph = {"edges": [edge("Anchor", "Z bridge"), edge("Anchor", "A bridge"),
                           edge("A bridge", "A far"), edge("Z bridge", "Z far")]}
        self.assertEqual(evaluator.retrieve_from_kg(graph, "anchor", hops=2), [
            "(Anchor, part-of, Z bridge)", "(Anchor, part-of, A bridge)",
            "(Z bridge, part-of, Z far)", "(A bridge, part-of, A far)"])

    def test_word_overlap_fallback_preserves_graph_order_and_unicode(self):
        graph = {"edges": [edge("β core", "First"), edge("α core", "Second"),
                           edge("irrelevant", "Third")]}
        self.assertEqual(evaluator.retrieve_from_kg(graph, "missing core phrase"), [
            "(β core, part-of, First)", "(α core, part-of, Second)"])
        self.assertEqual(evaluator.retrieve_from_kg(graph, "unmatched-zqx"), [])

    def test_caps_select_first_discovered_facts_before_hybrid_inclusion(self):
        graph = {"edges": [edge(f"Core {index}", f"Tail {index}") for index in range(21)]}
        questions = [{"question": "What is Core part of?", "gold_answer": "Tail",
                      "head": "Core", "tail": "Tail"}]
        plan = requests.capture_request_plan([], graph, questions, "gold", "fixture", "fixture")
        expected = [f"(Core {index}, part-of, Tail {index})" for index in range(15)]
        self.assertEqual(plan["requests"][2]["context"]["included_triples"], expected)
        self.assertEqual(plan["requests"][3]["context"]["included_triples"], expected)
        self.assertEqual(plan["requests"][4]["context"]["included_triples"], expected[:5])

    def test_text_ties_and_duplicate_questions_keep_input_order(self):
        records = [{"sentence": f"Core Tail {name}", "doc_id": name,
                    "gold_triples": [{"head_text": "Core", "relation": "part-of",
                                      "tail_text": name}]}
                   for name in ("Z", "A", "B", "C")]
        self.assertEqual(evaluator.retrieve_sentences(records, "Core Tail"),
                         [record["sentence"] for record in records[:3]])
        self.assertEqual(evaluator.generate_questions(records)[0]["gold_answer"], "Z")
        self.assertEqual(len(evaluator.generate_questions(records)), 1)

    def test_exact_context_prompt_and_wire_bytes_ignore_hash_seed_and_isolation(self):
        fixture = {
            "records": [{"sentence": f"Core Tail α {index}"} for index in range(5)],
            "kg": {"edges": [edge(f"Core {index}", f"Bridge {index}") for index in range(19)]
                   + [edge(f"Bridge {index}", f"Far {index}") for index in range(19)]},
            "questions": [{"question": "What is Core part of?", "gold_answer": "Tail",
                           "head": "Core", "tail": "Tail"},
                          {"question": "What is missing part of?", "gold_answer": "Far",
                           "head": "unmatched-zqx", "tail": "Bridge 0"}],
        }
        program = f"""import json,sys
sys.path.insert(0,{str(ROOT)!r})
from utils.rag import requests,runner
fixture={fixture!r}
plan=requests.capture_request_plan(fixture['records'],fixture['kg'],fixture['questions'],'gold','fixture','fixture')
print(json.dumps({{'input_sha256':runner.sha256_bytes(runner.canonical_json_bytes(fixture)),
                  'plan':plan,'wire':[requests.request_bytes(plan,i).hex() for i in range(len(plan['prompts']))]}},sort_keys=True))
"""
        expected = None
        for seed, flags in ((None, ()), ("random", ()), ("0", ()), ("42", ()),
                            ("4294967295", ()), ("0", ("-I",)), ("42", ("-I",)),
                            (None, ("-I",))):
            with self.subTest(seed=seed, flags=flags):
                environment = dict(os.environ)
                environment.pop("PYTHONHASHSEED", None)
                if seed is not None:
                    environment["PYTHONHASHSEED"] = seed
                completed = subprocess.run([sys.executable, *flags, "-B", "-c", program],
                    cwd=ROOT, env=environment, check=True, capture_output=True,
                    text=True, encoding="utf-8")
                observed = json.loads(completed.stdout)
                if expected is None:
                    expected = observed
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
