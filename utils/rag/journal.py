"""Ordered RAG occurrences with original answers and contemporaneous identity proof."""
from __future__ import annotations

import math
import re
import time

from utils.common.process_identity import process_identity
from utils.rag import evaluator, runner, requests as request_evidence


def _producer(child_dir, identity, model_identity, ollama_url):
    process = process_identity()
    path = child_dir / "calls" / "producers" / f"{process['process_id']}.json"
    runner.write_json_once(path, {"binding": {"identity": identity,
        "model_identity": runner.persisted_model_identity(model_identity),
        "ollama_url": ollama_url}, "process": process}, child_dir)
    return {"path": path.relative_to(child_dir).as_posix(), "sha256": runner.sha256_file(path)}


def _validate_producer(reference, child_dir, identity, model_identity):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference.get("path"), str)):
        raise runner.PhaseEError("RAG producer reference is malformed")
    path = runner.run_file(child_dir, reference["path"])
    if runner.sha256_file(path) != reference["sha256"]:
        raise runner.PhaseEError("RAG producer evidence changed")
    producer = runner.load_json(path)
    if not isinstance(producer, dict) or set(producer) != {"binding", "process"}:
        raise runner.PhaseEError("RAG producer evidence is malformed")
    binding = producer["binding"]
    if (not isinstance(binding, dict) or set(binding) != {"identity", "model_identity", "ollama_url"}
            or binding.get("identity") != identity
            or binding.get("model_identity") not in (runner.persisted_model_identity(model_identity), model_identity)
            or not isinstance(binding.get("ollama_url"), str) or not binding["ollama_url"]):
        raise runner.PhaseEError("RAG producer belongs to different run or model inputs")
    process = producer["process"]
    fields = {"process_id", "pid", "observed_unix_ns"}
    if (not isinstance(process, dict)
            or not isinstance(process.get("process_id"), str)
            or not re.fullmatch(r"[1-9][0-9]*-[0-9]+", process["process_id"])
            or type(process.get("pid")) is not int or process["pid"] <= 0
            or type(process.get("observed_unix_ns")) is not int or process["observed_unix_ns"] <= 0
            or process["process_id"] != f"{process['pid']}-{process['observed_unix_ns']}"
            or reference["path"] != f"calls/producers/{process['process_id']}.json"):
        raise runner.PhaseEError("RAG producing process identity is malformed")
    if set(process) != fields:
        seed_fields = {"environment_at_capture", "startup_setting", "effective_setting"}
        if (process.get("schema_version") != "python-process-observation-1"
                or not isinstance(process.get("python_hash_seed"), dict)
                or not seed_fields.issubset(process["python_hash_seed"])):
            raise runner.PhaseEError("RAG legacy producing process evidence is malformed")
    return producer


def _recorded_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise runner.PhaseEError("RAG journal lacks valid original condition timing")
    return value


def _load_record(path, label):
    document = runner.load_json(path)
    if not isinstance(document, dict):
        raise runner.PhaseEError(f"RAG {label} evidence is malformed")
    return document


def _saved_occurrences(directory, plan, stage, child_dir, condition, identity, model_identity):
    """Authenticate every saved call before issuing any unfinished occurrence."""
    prompts = plan["prompts"]
    enriched = "requests" in plan
    saved_calls = stage.get("calls", {})
    saved_requests = stage.get("requests", {})
    if not isinstance(saved_calls, dict) or not isinstance(saved_requests, dict):
        raise runner.PhaseEError("RAG occurrence hash index is malformed")
    observed_calls = {}
    observed_requests = {}
    answers = {}
    counts = {}
    elapsed = _recorded_seconds(stage.get("time_seconds", 0.0))
    expected_dirs = {f"{ordinal:06d}" for ordinal in range(len(prompts))}
    if any(path.is_dir() and path.name not in expected_dirs for path in directory.iterdir()):
        raise runner.PhaseEError("RAG journal contains an unplanned occurrence")
    for ordinal in range(len(prompts)):
        occurrence = directory / f"{ordinal:06d}"
        prior = sorted(occurrence.glob("*-request.json" if enriched else "*-answer.json"))
        counts[ordinal] = len(prior)
        expected_answers = {path.name.replace("-request", "-answer") for path in prior} if enriched else {path.name for path in prior}
        expected_proofs = {name.replace("-answer", "-model") for name in expected_answers}
        if (any(path.name not in expected_answers for path in occurrence.glob("*-answer.json"))
                or any(path.name not in expected_proofs for path in occurrence.glob("*-model.json"))):
            raise runner.PhaseEError("saved RAG occurrence request or answer is missing")
        for index, original_path in enumerate(prior, 1):
            suffix = "request" if enriched else "answer"
            if original_path.name != f"{index:03d}-{suffix}.json":
                raise runner.PhaseEError("RAG occurrence attempts are not contiguous")
            if ordinal in answers:
                raise runner.PhaseEError("RAG journal continues after a completed occurrence")
            key = f"{ordinal}:{index}"
            request_hash = None
            if enriched:
                request = _load_record(original_path, "request")
                if (request.get("binding") != {"plan_sha256": stage["plan_sha256"],
                        "ordinal": ordinal, "attempt": index}
                        or request.get("request") != request_evidence.request_identity(plan, ordinal)):
                    raise runner.PhaseEError("RAG attempt belongs to a different exact request")
                _validate_producer(request.get("producer"), child_dir, identity, model_identity)
                request_hash = runner.sha256_file(original_path)
                if key in saved_requests and saved_requests[key] != request_hash:
                    raise runner.PhaseEError("saved RAG request occurrence changed")
                observed_requests[key] = request_hash
                answer_path = original_path.with_name(original_path.name.replace("-request", "-answer"))
            else:
                answer_path = original_path
            proof_path = answer_path.with_name(answer_path.name.replace("-answer", "-model"))
            if not answer_path.exists():
                if proof_path.exists():
                    raise runner.PhaseEError("saved RAG occurrence answer is missing")
                continue  # An interrupted dispatch is retained; its retry uses a new ordinal.
            response = _load_record(answer_path, "answer")
            if response.get("binding") != {"plan_sha256": stage["plan_sha256"], "ordinal": ordinal}:
                raise runner.PhaseEError("RAG answer occurrence identity differs")
            if enriched and response.get("request_record_sha256") != request_hash:
                raise runner.PhaseEError("RAG answer differs from its original request evidence")
            if not proof_path.exists():
                raise runner.ModelEvidenceError("completed RAG call lacks contemporaneous post-model evidence", condition)
            proof = _load_record(proof_path, "model proof")
            if enriched and proof.get("request_record_sha256") != request_hash:
                raise runner.PhaseEError("RAG model proof differs from its original request evidence")
            if proof.get("answer_sha256") != runner.sha256_file(answer_path):
                raise runner.PhaseEError("RAG answer changed after its model proof")
            evidence = proof.get("evidence", {})
            if "ollama_runtime_identity" not in model_identity:
                for field in ("name", "tag_digest", "blob_sha256", "details"):
                    if evidence.get(field) != model_identity[field]:
                        raise runner.ModelEvidenceError("RAG occurrence model identity differs", condition)
            if ("ollama_runtime_identity" in model_identity
                    and evidence.get("ollama_runtime_sha256") != model_identity["ollama_runtime"]["sha256"]):
                raise runner.ModelEvidenceError("RAG occurrence runtime identity differs", condition)
            tags = runner.run_file(child_dir, proof["tags"])
            show = runner.run_file(child_dir, proof["show"])
            if (runner.sha256_file(tags) != evidence.get("tags_response_sha256")
                    or runner.sha256_file(show) != evidence.get("show_response_sha256")):
                raise runner.ModelEvidenceError("RAG occurrence model evidence changed", condition)
            runner._validate_stored_model_response(runner.load_json(tags), runner.load_json(show), model_identity)
            hashes = {"answer": runner.sha256_file(answer_path), "model": runner.sha256_file(proof_path)}
            if key in saved_calls and saved_calls[key] != hashes:
                raise runner.PhaseEError("saved RAG occurrence hashes changed")
            observed_calls[key] = hashes
            elapsed = max(elapsed, _recorded_seconds(proof.get("condition_time_seconds")))
            candidate = response.get("answer")
            if not isinstance(candidate, str):
                raise runner.PhaseEError("saved RAG answer is malformed")
            if candidate.strip() and not candidate.startswith("ERROR:"):
                answers[ordinal] = candidate
    if set(saved_calls) - set(observed_calls) or set(saved_requests) - set(observed_requests):
        raise runner.PhaseEError("saved RAG occurrence evidence is missing")
    stage["calls"] = observed_calls
    if enriched:
        stage["requests"] = observed_requests
    stage["time_seconds"] = elapsed
    return answers, counts, elapsed


def evaluate_ordered(records, kg, *, questions, condition, identity, child_dir,
                     status, status_path, ollama_url, model_identity, completed_only=False):
    stage = status.setdefault("stages", {}).setdefault(f"rag_{condition}", {"attempts": []})
    directory = child_dir / "calls" / condition
    plan_path = directory / "requests.json"
    runner.validate_child_target(child_dir, plan_path)
    binding = {"identity": identity, "condition": condition,
               "questions_sha256": runner.sha256_bytes(runner.canonical_json_bytes(questions)),
               "graph_sha256": runner.sha256_bytes(runner.canonical_json_bytes(kg))}
    if plan_path.exists():
        plan = _load_record(plan_path, "request plan")
        if plan.get("binding") != binding:
            raise runner.PhaseEError("saved RAG request plan belongs to different inputs")
        if stage.get("plan_sha256") != runner.sha256_file(plan_path):
            raise runner.PhaseEError("saved RAG request plan lacks its original hash binding")
    else:
        if completed_only or stage.get("plan_sha256") or stage.get("calls") or stage.get("requests"):
            raise runner.PhaseEError("saved RAG request plan is missing")
        producer = _producer(child_dir, identity, model_identity, ollama_url)
        plan = {"binding": binding, "producer": producer,
            **request_evidence.capture_request_plan(records, kg, questions, condition,
                model_identity["name"], ollama_url)}
        runner.write_json_once(plan_path, plan, child_dir)
        stage["plan_sha256"] = runner.sha256_file(plan_path)
        runner.write_status(status_path, status, child_dir)
    prompts = plan.get("prompts")
    if (not questions or not isinstance(prompts, list)
            or len(prompts) != len(questions)*len(evaluator.MODES)
            or not all(isinstance(prompt, str) for prompt in prompts)):
        raise runner.PhaseEError("RAG request plan occurrence coverage differs")
    enriched = set(plan) in ({"binding", "producer", "prompts", "requests", "model"},
                             {"binding", "producer", "prompts", "requests", "request_template"})
    current = enriched and request_evidence.is_current_request_plan(plan, model_identity["name"])
    if enriched:
        request_evidence.validate_request_plan(plan, questions, model_identity["name"], allow_legacy=True)
        _validate_producer(plan.get("producer"), child_dir, identity, model_identity)
    elif set(plan) != {"binding", "prompts"}:
        raise runner.PhaseEError("RAG request plan evidence fields are unsupported or incomplete")
    answers, counts, previous_seconds = _saved_occurrences(
        directory, plan, stage, child_dir, condition, identity, model_identity)
    if not enriched and len(answers) != len(prompts):
        raise runner.PhaseEError("unfinished historical RAG plan lacks recorded ordered request context")
    if enriched and not current and len(answers) != len(prompts):
        raise runner.PhaseEError("unfinished historical RAG plan requires its original source workflow")
    if completed_only and len(answers) != len(prompts):
        raise runner.PhaseEError("completed RAG stage lacks its successful occurrence evidence")
    results = {mode: [] for mode in evaluator.MODES}
    correct = {mode: 0 for mode in evaluator.MODES}
    started = time.time()
    called = False
    producer = None
    for ordinal, prompt in enumerate(prompts):
        occurrence = directory / f"{ordinal:06d}"
        answer = answers.get(ordinal)
        if answer is None:
            index = counts[ordinal]+1
            request_path = occurrence / f"{index:03d}-request.json"
            answer_path = occurrence / f"{index:03d}-answer.json"
            proof_path = occurrence / f"{index:03d}-model.json"
            if producer is None:
                if "ollama_runtime_identity" in model_identity:
                    runner.fetch_matching_model(ollama_url, model_identity)
                producer = _producer(child_dir, identity, model_identity, ollama_url)
            runner.write_json_once(request_path, {"binding": {"plan_sha256": stage["plan_sha256"],
                "ordinal": ordinal, "attempt": index},
                "request": request_evidence.request_identity(plan, ordinal), "producer": producer}, child_dir)
            request_hash = runner.sha256_file(request_path)
            stage.setdefault("requests", {})[f"{ordinal}:{index}"] = request_hash
            runner.write_status(status_path, status, child_dir)
            answer = evaluator.call_llm(ollama_url, model_identity["name"], prompt)
            called = True
            runner.write_json_once(answer_path, {"binding": {"plan_sha256": stage["plan_sha256"], "ordinal": ordinal},
                "request_record_sha256": request_hash,
                "answer": answer, "recorded_at": runner.utc_now()}, child_dir)
            # Once the answer is durable, absence of this contemporaneous proof
            # must fail closed on resume; a later query cannot replace it.
            evidence, tags_bytes, show_bytes = runner.fetch_matching_model(ollama_url, model_identity,
                                                                         check_runtime=False)
            evidence_dir = child_dir / "calls" / "model-evidence"
            tags = evidence_dir / (runner.sha256_bytes(tags_bytes)+".json")
            show = evidence_dir / (runner.sha256_bytes(show_bytes)+".json")
            runner.write_bytes_once(tags, tags_bytes, child_dir)
            runner.write_bytes_once(show, show_bytes, child_dir)
            stage["time_seconds"] = previous_seconds + time.time() - started
            runner.write_json_once(proof_path, {"answer_sha256": runner.sha256_file(answer_path),
                "request_record_sha256": request_hash,
                "evidence": evidence, "checked_at": runner.utc_now(),
                "condition_time_seconds": stage["time_seconds"],
                "tags": tags.relative_to(child_dir).as_posix(), "show": show.relative_to(child_dir).as_posix()}, child_dir)
            stage.setdefault("calls", {})[f"{ordinal}:{index}"] = {"answer": runner.sha256_file(answer_path),
                                                          "model": runner.sha256_file(proof_path)}
            runner.write_status(status_path, status, child_dir)
        question = questions[ordinal//len(evaluator.MODES)]
        mode = evaluator.MODES[ordinal%len(evaluator.MODES)]
        hit = evaluator.check_answer(answer, question["gold_answer"])
        correct[mode] += int(hit)
        results[mode].append({"q": question["question"], "pred": answer,
                             "gold": question["gold_answer"], "correct": hit})
    if called:
        stage["time_seconds"] = previous_seconds + time.time() - started
    if not completed_only and current:
        runner.write_status(status_path, status, child_dir)
    count = len(questions)
    return {"metadata": {"kg": runner.stable_child_path(child_dir/"projections"/f"{condition}-seed-42.json")
                if condition != "gold" else runner.stable_child_path(child_dir/"projections/gold-oracle.json"),
                "n_questions": count, "model": model_identity["name"], "time_seconds": stage["time_seconds"]},
            "accuracy": {mode: correct[mode]/count for mode in correct},
            "correct_counts": correct, "results": results}
