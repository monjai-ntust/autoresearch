"""Observability retains literal RAG requests and the original selected order."""
from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from utils.rag import evaluator, requests, runner


class RagRequestTests(unittest.TestCase):
    def _fixture(self):
        questions = [{"question": "What is Core part of?", "gold_answer": "Body",
                      "head": "Core", "tail": "Body", "relation": "part-of",
                      "sentence": "Core is part of Body.", "doc_id": "fixture"}]
        records = [{"sentence": f"Core Body α {index} " + "x" * 220}
                   for index in range(5)]
        graph = {"edges": [{"head": f"Core {index}", "relation": "part-of",
                            "tail": f"Tail {index}"} for index in range(20)]}
        return records, graph, questions

    def _plan(self):
        records, graph, questions = self._fixture()
        return (requests.capture_request_plan(records, graph, questions, "gold", "qwen3:32b",
                                             "http://fixture.invalid"), questions)

    def test_plan_records_actual_caps_order_and_one_retrieval_pass(self):
        records, graph, questions = self._fixture()
        with (mock.patch.object(evaluator, "retrieve_from_kg", wraps=evaluator.retrieve_from_kg) as kg,
              mock.patch.object(evaluator, "retrieve_sentences", wraps=evaluator.retrieve_sentences) as text):
            plan = requests.capture_request_plan(records, graph, questions, "gold", "qwen3:32b",
                                                 "http://fixture.invalid")
        self.assertEqual(kg.call_count, 2)
        self.assertEqual(text.call_count, 1)
        self.assertEqual([item["mode"] for item in plan["requests"]], list(evaluator.MODES))
        one_hop = plan["requests"][2]["context"]
        hybrid = plan["requests"][4]["context"]
        self.assertEqual(len(one_hop["retrieved_triples"]), 15)
        self.assertEqual(one_hop["retrieved_triples"], one_hop["included_triples"])
        self.assertEqual(hybrid["retrieved_triples"], one_hop["retrieved_triples"])
        self.assertEqual(hybrid["included_triples"], one_hop["retrieved_triples"][:5])
        self.assertEqual(len(plan["requests"][1]["context"]["retrieved_sentences"]), 3)
        self.assertTrue(all(len(value) == 200
                            for value in plan["requests"][1]["context"]["source_excerpts"]))
        self.assertEqual(hybrid["source_excerpts"], [records[0]["sentence"][:200]])
        with (mock.patch.object(evaluator, "retrieve_from_kg", side_effect=AssertionError("reretrieval")),
              mock.patch.object(evaluator, "retrieve_sentences", side_effect=AssertionError("reretrieval"))):
            self.assertEqual(requests.validate_request_plan(plan, questions, "qwen3:32b"), plan["requests"])

    def test_reconstructed_wire_bytes_match_frozen_actual_dispatch(self):
        plan, _questions = self._plan()
        restored = json.loads(runner.canonical_json_bytes(plan))
        requests.validate_request_plan(restored, _questions, "qwen3:32b")
        self.assertEqual([requests.request_bytes(plan, ordinal) for ordinal in range(5)],
                         [requests.request_bytes(restored, ordinal) for ordinal in range(5)])
        dispatched = []

        def curl(command, **_kwargs):
            dispatched.append(command[-1].encode("utf-8"))
            return SimpleNamespace(stdout=json.dumps({"message": {"content": "Body"}}))

        with mock.patch.object(evaluator.subprocess, "run", side_effect=curl):
            for ordinal, prompt in enumerate(plan["prompts"]):
                self.assertEqual(evaluator.call_llm("http://fixture.invalid", "qwen3:32b", prompt), "Body")
                self.assertEqual(dispatched[-1], requests.request_bytes(plan, ordinal))
                self.assertEqual(runner.sha256_bytes(dispatched[-1]),
                                 plan["requests"][ordinal]["request_sha256"])
        self.assertIn(b"\\u03b1", dispatched[1])
        self.assertNotIn("α".encode("utf-8"), dispatched[1])

    def test_observer_precedes_dispatch_without_changing_prompts_answers_or_scores(self):
        records, graph, questions = self._fixture()
        original = copy.deepcopy((records, graph, questions))
        prompts = []
        with mock.patch.object(evaluator.time, "time", side_effect=[10.0, 12.5]):
            baseline = evaluator.evaluate(records, graph, kg_identity="gold", ollama_url="fixture",
                ollama_model="qwen3:32b", question_panel=questions,
                llm_call=lambda _url, _model, prompt: prompts.append(prompt) or "Body")
        events = []

        def observe(**occurrence):
            events.append(("observe", occurrence["prompt"]))
            occurrence["question"]["question"] = "Changed only inside observer"
            for field in requests.CONTEXT_FIELDS:
                occurrence[field].append("Changed only inside observer")

        def call(_url, _model, prompt):
            events.append(("dispatch", prompt))
            return "Body"

        with mock.patch.object(evaluator.time, "time", side_effect=[10.0, 12.5]):
            observed = evaluator.evaluate(records, graph, kg_identity="gold", ollama_url="fixture",
                ollama_model="qwen3:32b", question_panel=questions, llm_call=call,
                request_observer=observe)
        self.assertEqual(baseline, observed)
        self.assertEqual(events, [event for prompt in prompts
                                  for event in (("observe", prompt), ("dispatch", prompt))])
        self.assertEqual((records, graph, questions), original)

    def test_empty_context_retains_hybrid_empty_source_bullet(self):
        questions = [{"question": "Empty?", "gold_answer": "none", "head": "absent", "tail": "missing"}]
        plan = requests.capture_request_plan([], {"edges": []}, questions,
                                             "gold", "qwen3:32b", "fixture")
        for item in plan["requests"][:-1]:
            self.assertTrue(all(not values for values in item["context"].values()))
        self.assertEqual(plan["requests"][-1]["context"]["source_excerpts"], [""])
        self.assertIn("Source text:\n- \n\nAnswer", plan["prompts"][-1])
        requests.validate_request_plan(plan, questions, "qwen3:32b")

    def test_saved_context_preserves_tail_fallback_and_distinct_second_hop(self):
        questions = [{"question": "Fallback?", "gold_answer": "anchor",
                      "head": "unmatched-zqx", "tail": "anchor"}]
        graph = {"edges": [{"head": "anchor", "relation": "part-of", "tail": "bridge"},
                           {"head": "bridge", "relation": "part-of", "tail": "far end"}]}
        with mock.patch.object(evaluator, "retrieve_from_kg", wraps=evaluator.retrieve_from_kg) as retrieve:
            plan = requests.capture_request_plan([], graph, questions, "gold", "qwen3:32b", "fixture")
        self.assertEqual(retrieve.call_args_list, [mock.call(graph, "unmatched-zqx", hops=1),
            mock.call(graph, "anchor", hops=1), mock.call(graph, "unmatched-zqx", hops=2),
            mock.call(graph, "anchor", hops=2)])
        self.assertEqual(plan["requests"][2]["context"]["included_triples"],
                         ["(anchor, part-of, bridge)"])
        self.assertEqual(plan["requests"][3]["context"]["included_triples"],
                         ["(anchor, part-of, bridge)", "(bridge, part-of, far end)"])
        self.assertEqual(plan["requests"][4]["context"]["included_triples"],
                         plan["requests"][2]["context"]["included_triples"])

    def test_missing_or_changed_request_context_question_and_payload_fail(self):
        plan, questions = self._plan()
        for field in ("prompt_sha256", "context_sha256", "request_sha256", "question_sha256", "mode"):
            with self.subTest(field=field):
                changed = copy.deepcopy(plan)
                changed["requests"][2][field] = "changed"
                with self.assertRaises(runner.PhaseEError):
                    requests.validate_request_plan(changed, questions, "qwen3:32b")
        for change in (lambda value: value.pop("requests"),
                       lambda value: value["requests"].pop(),
                       lambda value: value["prompts"].__setitem__(2, "changed"),
                       lambda value: value["requests"][2]["context"]["included_triples"].reverse(),
                       lambda value: value["request_template"]["options"].__setitem__("num_predict", 51),
                       lambda value: value["request_template"]["options"].__setitem__("temperature", False)):
            changed = copy.deepcopy(plan)
            change(changed)
            with self.assertRaises(runner.PhaseEError):
                requests.validate_request_plan(changed, questions, "qwen3:32b")

    def test_repeated_question_occurrences_have_distinct_ordinals(self):
        records, graph, questions = self._fixture()
        questions = questions * 2
        plan = requests.capture_request_plan(records, graph, questions,
                                             "gold", "qwen3:32b", "fixture")
        self.assertEqual(len(plan["requests"]), 10)
        self.assertEqual([item["question_index"] for item in plan["requests"]], [0] * 5 + [1] * 5)
        self.assertEqual(plan["requests"][0]["request_sha256"], plan["requests"][5]["request_sha256"])
        self.assertNotEqual(plan["requests"][0]["ordinal"], plan["requests"][5]["ordinal"])


if __name__ == "__main__":
    unittest.main()
