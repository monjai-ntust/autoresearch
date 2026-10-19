"""Exact RAG request and original process evidence survives pure fixture replay."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.common.process_identity import process_identity
from utils.rag import evaluator, journal, requests, runner
from utils.verifier.runtime import runtime_identity


ROOT = Path(__file__).resolve().parents[1]


class RagJournalIdentityTests(unittest.TestCase):
    def _fixture(self, directory):
        record = {"doc_id": "fixture", "sentence": "Fixture head part of tail.",
            "gold_triples": [{"head_text": "head", "relation": "part-of", "tail_text": "tail"}],
            "predicted_triples": []}
        model = {"name": "qwen3:32b", "tag_digest": "a" * 64, "blob_sha256": "b" * 64,
            "details": {"family": "qwen3", "parameter_size": "32.8B", "quantization_level": "Q4_K_M"}}
        tags = runner.canonical_json_bytes({"models": [{"name": model["name"], "digest": model["tag_digest"]}]})
        show = runner.canonical_json_bytes({"details": model["details"], "modelfile": "FROM sha256-" + model["blob_sha256"]})
        capture = ({**model, "tags_response_sha256": runner.sha256_bytes(tags),
                    "show_response_sha256": runner.sha256_bytes(show)}, tags, show)
        arguments = {"records": [record], "kg": {"nodes": [], "edges": []},
            "questions": evaluator.generate_questions([record]), "condition": "gold",
            "identity": {"run_id": "fixture", "source": "authenticated-fixture"},
            "child_dir": directory, "status": {"stages": {}},
            "status_path": directory / "run-status.json", "ollama_url": "http://localhost:11434",
            "model_identity": model}
        return arguments, capture

    def _producer_process(self, pid):
        process = copy.deepcopy(process_identity())
        process.update(pid=pid, observed_unix_ns=pid, process_id=f"{pid}-{pid}")
        return process

    def _tree_bytes(self, directory):
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def test_request_and_producer_are_durable_before_each_exact_dispatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            calls = []

            def call(url, model, prompt):
                ordinal = len(calls)
                plan = runner.load_json(args["child_dir"] / "calls/gold/requests.json")
                self.assertNotIn("schema_version", plan)
                self.assertEqual(set(plan), {"binding", "producer", "prompts", "requests", "model"})
                path = args["child_dir"] / f"calls/gold/{ordinal:06d}/001-request.json"
                request = runner.load_json(path)
                self.assertEqual(request["request"], requests.request_identity(plan, ordinal))
                self.assertEqual(args["status"]["stages"]["rag_gold"]["requests"][f"{ordinal}:1"], runner.sha256_file(path))
                producer = runner.load_json(args["child_dir"] / request["producer"]["path"])
                self.assertEqual(set(producer["process"]), {"process_id", "pid", "observed_unix_ns"})
                self.assertEqual(producer["binding"], {"identity": args["identity"],
                    "model_identity": args["model_identity"], "ollama_url": url})
                wire = json.loads(requests.request_bytes(plan, ordinal))
                self.assertEqual(wire["model"], model)
                self.assertEqual(wire["messages"], [{"role": "user", "content": prompt}])
                calls.append(prompt)
                return "tail"

            with mock.patch.object(evaluator, "call_llm", side_effect=call), \
                 mock.patch.object(runner, "fetch_matching_model", return_value=capture):
                result = journal.evaluate_ordered(**args)
            self.assertEqual(len(calls), 5)
            self.assertTrue(all(value == 1 for value in result["accuracy"].values()))
            self.assertEqual(len(list((args["child_dir"] / "calls/producers").glob("*.json"))), 1)
            for path in (args["child_dir"] / "calls/gold").rglob("*-answer.json"):
                request = path.with_name(path.name.replace("-answer", "-request"))
                self.assertEqual(runner.load_json(path)["request_record_sha256"], runner.sha256_file(request))
                proof = runner.load_json(path.with_name(path.name.replace("-answer", "-model")))
                self.assertEqual(proof["request_record_sha256"], runner.sha256_file(request))

    def test_interrupted_unindexed_request_retries_under_new_original_producer(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            write_status = runner.write_status
            writes = []

            def interrupt_status(path, status, directory):
                writes.append(path)
                if len(writes) == 2:
                    raise RuntimeError("fixture crash after request before status")
                write_status(path, status, directory)

            with mock.patch.object(journal, "process_identity", return_value=self._producer_process(101)), \
                 mock.patch.object(runner, "write_status", side_effect=interrupt_status), \
                 mock.patch.object(evaluator, "call_llm") as call:
                with self.assertRaisesRegex(RuntimeError, "fixture crash"):
                    journal.evaluate_ordered(**args)
                call.assert_not_called()
            first = args["child_dir"] / "calls/gold/000000/001-request.json"
            original = first.read_bytes()
            args["status"] = runner.load_json(args["status_path"])
            with mock.patch.object(journal, "process_identity", return_value=self._producer_process(202)), \
                 mock.patch.object(requests, "capture_request_plan") as retrieve, \
                 mock.patch.object(evaluator, "call_llm", return_value="tail") as call, \
                 mock.patch.object(runner, "fetch_matching_model", return_value=capture):
                completed = journal.evaluate_ordered(**args)
                self.assertEqual(call.call_count, 5)
                retrieve.assert_not_called()
            self.assertEqual(first.read_bytes(), original)
            second = runner.load_json(first.with_name("002-request.json"))
            self.assertNotEqual(second["producer"], runner.load_json(first)["producer"])
            self.assertTrue(second["producer"]["path"].endswith("202-202.json"))
            recorded = self._tree_bytes(args["child_dir"])
            with mock.patch.object(journal, "process_identity") as process, \
                 mock.patch.object(requests, "capture_request_plan") as retrieve, \
                 mock.patch.object(evaluator, "call_llm") as call, \
                 mock.patch.object(runner, "fetch_matching_model") as fetch:
                replay = journal.evaluate_ordered(**args, completed_only=True)
                process.assert_not_called(); retrieve.assert_not_called()
                call.assert_not_called(); fetch.assert_not_called()
            self.assertEqual(runner.scientific_json_bytes(replay), runner.scientific_json_bytes(completed))
            self.assertEqual(self._tree_bytes(args["child_dir"]), recorded)

    def test_corrupt_or_missing_later_evidence_blocks_an_earlier_retry(self):
        for mutation in ("request-delete", "request-hash", "request-shape", "producer", "cross-run", "answer-request"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
                args, capture = self._fixture(Path(temporary))
                with mock.patch.object(evaluator, "call_llm", side_effect=["ERROR: fixture", *["tail"] * 4]), \
                     mock.patch.object(runner, "fetch_matching_model", return_value=capture):
                    journal.evaluate_ordered(**args)
                path = args["child_dir"] / "calls/gold/000004/001-request.json"
                request = runner.load_json(path)
                if mutation == "request-delete":
                    path.unlink()
                elif mutation == "request-hash":
                    request["request"]["prompt_sha256"] = "0" * 64
                    path.write_bytes(runner.canonical_json_bytes(request))
                elif mutation == "request-shape":
                    path.write_bytes(b"[]\n")
                elif mutation in {"producer", "cross-run"}:
                    producer_path = args["child_dir"] / request["producer"]["path"]
                    producer = runner.load_json(producer_path)
                    if mutation == "producer":
                        producer_path.unlink()
                    else:
                        producer["binding"]["identity"]["run_id"] = "foreign"
                        producer_path.write_bytes(runner.canonical_json_bytes(producer))
                else:
                    answer_path = path.with_name("001-answer.json")
                    answer = runner.load_json(answer_path)
                    answer["request_record_sha256"] = "0" * 64
                    answer_path.write_bytes(runner.canonical_json_bytes(answer))
                with mock.patch.object(evaluator, "call_llm") as call, \
                     mock.patch.object(runner, "fetch_matching_model") as fetch:
                    with self.assertRaises(runner.PhaseEError):
                        journal.evaluate_ordered(**args)
                    call.assert_not_called(); fetch.assert_not_called()

    def test_hash_valid_foreign_producers_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            root = Path(temporary)
            args, _capture = self._fixture(root / "child")
            reference = journal._producer(args["child_dir"], args["identity"],
                args["model_identity"], args["ollama_url"])
            path = args["child_dir"] / reference["path"]
            original = runner.load_json(path)
            self.assertEqual(journal._validate_producer(reference, args["child_dir"],
                args["identity"], args["model_identity"]), original)
            for mismatch in ("run", "model"):
                with self.subTest(mismatch=mismatch):
                    foreign = copy.deepcopy(original)
                    if mismatch == "run":
                        foreign["binding"]["identity"]["run_id"] = "foreign-run"
                    else:
                        foreign["binding"]["model_identity"]["name"] = "foreign-model:tag"
                    path.write_bytes(runner.canonical_json_bytes(foreign))
                    hash_valid = {**reference, "sha256": runner.sha256_file(path)}
                    with self.assertRaisesRegex(runner.PhaseEError, "different run or model"):
                        journal._validate_producer(hash_valid, args["child_dir"],
                            args["identity"], args["model_identity"])
            outside = root / "outside-producer.json"
            outside.write_bytes(runner.canonical_json_bytes(original))
            for escape in ("../outside-producer.json", str(outside),
                           "calls/producers/../../../outside-producer.json"):
                with self.subTest(escape=escape):
                    escaping = {"path": escape, "sha256": runner.sha256_file(outside)}
                    with self.assertRaises(runner.PhaseEError):
                        journal._validate_producer(escaping, args["child_dir"],
                            args["identity"], args["model_identity"])

    def test_minimal_and_original_legacy_producer_records_validate_without_rewrites(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, _capture = self._fixture(Path(temporary))
            reference = journal._producer(args["child_dir"], args["identity"],
                args["model_identity"], args["ollama_url"])
            path = args["child_dir"] / reference["path"]
            producer = runner.load_json(path)
            self.assertEqual(set(producer["process"]), {"process_id", "pid", "observed_unix_ns"})
            self.assertEqual(journal._validate_producer(reference, args["child_dir"],
                args["identity"], args["model_identity"]), producer)
            producer["process"].update({"schema_version": "python-process-observation-1",
                "parent_pid": 10, "capture_scope": "entrypoint", "entrypoint": "rag.py",
                "python": {"version": "fixture historical runtime"},
                "python_hash_seed": {name: {"kind": "random", "value": None}
                    for name in ("environment_at_capture", "startup_setting", "effective_setting")}})
            path.write_bytes(runner.canonical_json_bytes(producer))
            original = path.read_bytes()
            reference["sha256"] = runner.sha256_file(path)
            self.assertEqual(journal._validate_producer(reference, args["child_dir"],
                args["identity"], args["model_identity"]), producer)
            self.assertEqual(path.read_bytes(), original)
            del producer["process"]["python_hash_seed"]["effective_setting"]
            path.write_bytes(runner.canonical_json_bytes(producer))
            reference["sha256"] = runner.sha256_file(path)
            with self.assertRaisesRegex(runner.PhaseEError, "legacy producing process"):
                journal._validate_producer(reference, args["child_dir"],
                    args["identity"], args["model_identity"])

    def test_producer_uses_shared_runtime_reference_without_repeated_model_definition(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, _capture = self._fixture(Path(temporary))
            model = args["model_identity"]
            model.update({"ollama_runtime": {"path": "runtime/ollama.json", "sha256": "c" * 64},
                          "ollama_runtime_identity": {"fixture": "loaded shared runtime"}})
            reference = journal._producer(args["child_dir"], args["identity"], model, args["ollama_url"])
            path = args["child_dir"] / reference["path"]
            producer = runner.load_json(path)
            self.assertEqual(producer["binding"]["model_identity"], {"ollama_runtime": model["ollama_runtime"]})
            self.assertEqual(journal._validate_producer(reference, args["child_dir"], args["identity"], model), producer)
            producer["binding"]["model_identity"] = model
            path.write_bytes(runner.canonical_json_bytes(producer))
            original = path.read_bytes()
            reference["sha256"] = runner.sha256_file(path)
            self.assertEqual(journal._validate_producer(reference, args["child_dir"], args["identity"], model), producer)
            self.assertEqual(path.read_bytes(), original)

    def test_same_byte_model_evidence_outside_the_child_cannot_be_consumed(self):
        for field, evidence_index in (("tags", 1), ("show", 2)):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
                root = Path(temporary)
                args, capture = self._fixture(root / "child")
                with mock.patch.object(evaluator, "call_llm", side_effect=["ERROR: fixture", *["tail"] * 4]), \
                     mock.patch.object(runner, "fetch_matching_model", return_value=capture):
                    journal.evaluate_ordered(**args)
                external = root / f"foreign-{field}.json"
                external.write_bytes(capture[evidence_index])
                proof_path = args["child_dir"] / "calls/gold/000004/001-model.json"
                proof = runner.load_json(proof_path)
                proof[field] = f"../{external.name}"
                proof_path.write_bytes(runner.canonical_json_bytes(proof))
                # Keep the digest valid to exercise physical same-run containment,
                # rather than merely rejecting a changed proof hash.
                args["status"]["stages"]["rag_gold"]["calls"]["4:1"]["model"] = runner.sha256_file(proof_path)
                with mock.patch.object(evaluator, "call_llm") as call, \
                     mock.patch.object(runner, "fetch_matching_model") as fetch:
                    with self.assertRaisesRegex(runner.PhaseEError, "Run-relative path is unsafe"):
                        journal.evaluate_ordered(**args)
                    call.assert_not_called(); fetch.assert_not_called()

    def _legacy_plan(self, args, capture, *, unfinished=False):
        directory = args["child_dir"] / "calls/gold"
        prompts = []
        evaluator.evaluate(args["records"], args["kg"], kg_identity="gold",
            ollama_url=args["ollama_url"], ollama_model=args["model_identity"]["name"],
            question_panel=args["questions"], llm_call=lambda _url, _model, prompt: prompts.append(prompt) or "")
        plan = {"binding": {"identity": args["identity"], "condition": "gold",
            "questions_sha256": runner.sha256_bytes(runner.canonical_json_bytes(args["questions"])),
            "graph_sha256": runner.sha256_bytes(runner.canonical_json_bytes(args["kg"]))}, "prompts": prompts}
        plan_path = directory / "requests.json"
        runner.write_json_once(plan_path, plan, args["child_dir"])
        stage = {"attempts": [], "plan_sha256": runner.sha256_file(plan_path), "time_seconds": 12.0, "calls": {}}
        evidence, tags, show = capture
        tags_path = args["child_dir"] / "calls/model-evidence/tags.json"
        show_path = args["child_dir"] / "calls/model-evidence/show.json"
        runner.write_bytes_once(tags_path, tags, args["child_dir"])
        runner.write_bytes_once(show_path, show, args["child_dir"])
        for ordinal in range(len(prompts)):
            answer_path = directory / f"{ordinal:06d}/001-answer.json"
            proof_path = answer_path.with_name("001-model.json")
            runner.write_json_once(answer_path, {"binding": {"plan_sha256": stage["plan_sha256"], "ordinal": ordinal},
                "answer": "ERROR: fixture" if unfinished and ordinal == 0 else "tail", "recorded_at": "fixture"}, args["child_dir"])
            runner.write_json_once(proof_path, {"answer_sha256": runner.sha256_file(answer_path), "evidence": evidence,
                "checked_at": "fixture", "condition_time_seconds": 10.0,
                "tags": tags_path.relative_to(args["child_dir"]).as_posix(),
                "show": show_path.relative_to(args["child_dir"]).as_posix()}, args["child_dir"])
            stage["calls"][f"{ordinal}:1"] = {"answer": runner.sha256_file(answer_path), "model": runner.sha256_file(proof_path)}
        args["status"]["stages"]["rag_gold"] = stage
        runner.write_status(args["status_path"], args["status"], args["child_dir"])

    def test_completed_legacy_plan_stays_read_only_without_invented_context(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            self._legacy_plan(args, capture)
            original = self._tree_bytes(args["child_dir"])
            with mock.patch.object(journal, "process_identity") as process, \
                 mock.patch.object(requests, "capture_request_plan") as retrieve, \
                 mock.patch.object(evaluator, "call_llm") as call, \
                 mock.patch.object(runner, "fetch_matching_model") as fetch, \
                 mock.patch.object(runner, "write_status") as write:
                result = journal.evaluate_ordered(**args)
                self.assertEqual(result["metadata"]["time_seconds"], 12.0)
                process.assert_not_called(); retrieve.assert_not_called()
                call.assert_not_called(); fetch.assert_not_called(); write.assert_not_called()
            self.assertEqual(self._tree_bytes(args["child_dir"]), original)

    def test_unfinished_legacy_plan_fails_without_retrieval_or_model_work(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            self._legacy_plan(args, capture, unfinished=True)
            original = self._tree_bytes(args["child_dir"])
            with mock.patch.object(requests, "capture_request_plan") as retrieve, \
                 mock.patch.object(evaluator, "call_llm") as call, \
                 mock.patch.object(runner, "fetch_matching_model") as fetch:
                with self.assertRaisesRegex(runner.PhaseEError, "historical RAG plan"):
                    journal.evaluate_ordered(**args)
                retrieve.assert_not_called(); call.assert_not_called(); fetch.assert_not_called()
            self.assertEqual(self._tree_bytes(args["child_dir"]), original)

    def _template_plan(self, args, capture, *, unfinished=False, seeded=False):
        capture_plan = requests.capture_request_plan

        def legacy_plan(*positional, **keywords):
            plan = capture_plan(*positional, **keywords)
            plan["request_template"] = requests._template(plan.pop("model"), legacy=not seeded)
            for ordinal, request in enumerate(plan["requests"]):
                request["request_sha256"] = runner.sha256_bytes(requests.request_bytes(plan, ordinal))
            return plan

        # Produce a complete fixture with precisely the previous wire payload.
        # Mark it current only during fixture setup; validation below uses the
        # actual production distinction between seeded and seedless plans.
        with mock.patch.object(requests, "capture_request_plan", side_effect=legacy_plan), \
             mock.patch.object(requests, "is_current_request_plan", return_value=True), \
             mock.patch.object(evaluator, "call_llm", side_effect=[
                 "ERROR: fixture" if unfinished else "tail", *["tail"] * 4]), \
             mock.patch.object(runner, "fetch_matching_model", return_value=capture):
            journal.evaluate_ordered(**args)

    def test_complete_seedless_enriched_plan_replays_read_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            self._template_plan(args, capture)
            original = self._tree_bytes(args["child_dir"])
            with mock.patch.object(requests, "capture_request_plan") as retrieve, \
                 mock.patch.object(evaluator, "call_llm") as call, \
                 mock.patch.object(runner, "fetch_matching_model") as fetch, \
                 mock.patch.object(runner, "write_status") as write:
                result = journal.evaluate_ordered(**args)
                self.assertTrue(all(value == 1.0 for value in result["accuracy"].values()))
                retrieve.assert_not_called(); call.assert_not_called()
                fetch.assert_not_called(); write.assert_not_called()
            self.assertEqual(self._tree_bytes(args["child_dir"]), original)

    def test_incomplete_seedless_enriched_plan_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            self._template_plan(args, capture, unfinished=True)
            original = self._tree_bytes(args["child_dir"])
            with mock.patch.object(requests, "capture_request_plan") as retrieve, \
                 mock.patch.object(evaluator, "call_llm") as call, \
                 mock.patch.object(runner, "fetch_matching_model") as fetch:
                with self.assertRaisesRegex(runner.PhaseEError, "original source workflow"):
                    journal.evaluate_ordered(**args)
                retrieve.assert_not_called(); call.assert_not_called(); fetch.assert_not_called()
            self.assertEqual(self._tree_bytes(args["child_dir"]), original)

    def test_seeded_template_plans_replay_complete_only_under_original_recorded_bytes(self):
        for unfinished in (False, True):
            with self.subTest(unfinished=unfinished), tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
                args, capture = self._fixture(Path(temporary))
                self._template_plan(args, capture, unfinished=unfinished, seeded=True)
                original = self._tree_bytes(args["child_dir"])
                with mock.patch.object(requests, "capture_request_plan") as retrieve, \
                     mock.patch.object(evaluator, "call_llm") as call, \
                     mock.patch.object(runner, "fetch_matching_model") as fetch, \
                     mock.patch.object(runner, "write_status") as write:
                    if unfinished:
                        with self.assertRaisesRegex(runner.PhaseEError, "original source workflow"):
                            journal.evaluate_ordered(**args)
                    else:
                        result = journal.evaluate_ordered(**args)
                        self.assertTrue(all(value == 1.0 for value in result["accuracy"].values()))
                    retrieve.assert_not_called(); call.assert_not_called()
                    fetch.assert_not_called(); write.assert_not_called()
                self.assertEqual(self._tree_bytes(args["child_dir"]), original)

    def test_resumed_missing_call_checks_server_before_dispatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            args["model_identity"]["ollama_runtime_identity"] = runtime_identity(
                args["ollama_url"], {"version": "fixture-runtime"},
                json.loads(capture[2]), args["model_identity"])
            args["model_identity"]["ollama_runtime"] = {"path": "runtime/ollama.json", "sha256": "c" * 64}
            capture = ({"tags_response_sha256": capture[0]["tags_response_sha256"],
                "show_response_sha256": capture[0]["show_response_sha256"], "ollama_runtime_sha256": "c" * 64},
                *capture[1:])
            with mock.patch.object(evaluator, "call_llm", side_effect=["ERROR: fixture", *["tail"] * 4]), \
                 mock.patch.object(runner, "fetch_matching_model", return_value=capture):
                journal.evaluate_ordered(**args)
            original = self._tree_bytes(args["child_dir"])
            with mock.patch.object(runner, "fetch_matching_model",
                                   side_effect=runner.ModelEvidenceError("runtime changed", "gold")) as fetch, \
                 mock.patch.object(evaluator, "call_llm") as call:
                with self.assertRaisesRegex(runner.ModelEvidenceError, "runtime changed"):
                    journal.evaluate_ordered(**args)
                fetch.assert_called_once_with(args["ollama_url"], args["model_identity"])
                call.assert_not_called()
            self.assertEqual(self._tree_bytes(args["child_dir"]), original)

    def test_runtime_proof_hash_omission_and_mutation_block_reuse(self):
        for field in ("ollama_runtime_sha256",):
            for mutation in ("omit", "change"):
                with self.subTest(field=field, mutation=mutation), \
                     tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
                    args, capture = self._fixture(Path(temporary))
                    model = args["model_identity"]
                    model["ollama_runtime_identity"] = runtime_identity(args["ollama_url"],
                        {"version": "fixture-runtime"}, json.loads(capture[2]), model)
                    model["ollama_runtime"] = {"path": "runtime/ollama.json", "sha256": "c" * 64}
                    capture = ({"tags_response_sha256": capture[0]["tags_response_sha256"],
                        "show_response_sha256": capture[0]["show_response_sha256"], "ollama_runtime_sha256": "c" * 64},
                        *capture[1:])
                    with mock.patch.object(evaluator, "call_llm", return_value="tail"), \
                         mock.patch.object(runner, "fetch_matching_model", return_value=capture):
                        journal.evaluate_ordered(**args)
                    proof_path = args["child_dir"] / "calls/gold/000000/001-model.json"
                    proof = runner.load_json(proof_path)
                    if mutation == "omit":
                        del proof["evidence"][field]
                    else:
                        proof["evidence"][field] = "0" * 64
                    proof_path.write_bytes(runner.canonical_json_bytes(proof))
                    args["status"]["stages"]["rag_gold"]["calls"]["0:1"]["model"] = runner.sha256_file(proof_path)
                    with mock.patch.object(evaluator, "call_llm") as call, \
                         mock.patch.object(runner, "fetch_matching_model") as fetch:
                        with self.assertRaisesRegex(runner.ModelEvidenceError, "runtime identity differs"):
                            journal.evaluate_ordered(**args)
                        call.assert_not_called(); fetch.assert_not_called()

    def test_incomplete_context_plan_is_not_mistaken_for_a_legacy_plan(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as temporary:
            args, capture = self._fixture(Path(temporary))
            with mock.patch.object(evaluator, "call_llm", return_value="tail"), \
                 mock.patch.object(runner, "fetch_matching_model", return_value=capture):
                journal.evaluate_ordered(**args)
            path = args["child_dir"] / "calls/gold/requests.json"
            plan = runner.load_json(path)
            del plan["model"]
            path.write_bytes(runner.canonical_json_bytes(plan))
            args["status"]["stages"]["rag_gold"]["plan_sha256"] = runner.sha256_file(path)
            with mock.patch.object(evaluator, "call_llm") as call, \
                 mock.patch.object(runner, "fetch_matching_model") as fetch, \
                 mock.patch.object(requests, "capture_request_plan") as retrieve:
                with self.assertRaisesRegex(runner.PhaseEError, "fields.*incomplete"):
                    journal.evaluate_ordered(**args, completed_only=True)
                call.assert_not_called(); fetch.assert_not_called(); retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
