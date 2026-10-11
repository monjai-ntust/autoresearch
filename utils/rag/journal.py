"""Ordered RAG occurrences with original answers and contemporaneous identity proof."""
from __future__ import annotations

import math
import time

from utils.rag import evaluator, runner


def _recorded_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise runner.PhaseEError("RAG journal lacks valid original condition timing")
    return value


def _saved_occurrences(directory, prompts, stage, child_dir, condition, model_identity):
    """Authenticate every saved call before issuing any unfinished occurrence."""
    saved_calls = stage.get("calls", {})
    if not isinstance(saved_calls, dict):
        raise runner.PhaseEError("RAG occurrence hash index is malformed")
    observed_calls = {}
    answers = {}
    counts = {}
    elapsed = _recorded_seconds(stage.get("time_seconds", 0.0))
    expected_dirs = {f"{ordinal:06d}" for ordinal in range(len(prompts))}
    if any(path.is_dir() and path.name not in expected_dirs for path in directory.iterdir()):
        raise runner.PhaseEError("RAG journal contains an unplanned occurrence")
    for ordinal in range(len(prompts)):
        occurrence = directory / f"{ordinal:06d}"
        prior = sorted(occurrence.glob("*-answer.json"))
        counts[ordinal] = len(prior)
        expected_proofs = {path.name.replace("-answer", "-model") for path in prior}
        if any(path.name not in expected_proofs for path in occurrence.glob("*-model.json")):
            raise runner.PhaseEError("saved RAG occurrence answer is missing")
        for index, answer_path in enumerate(prior, 1):
            if answer_path.name != f"{index:03d}-answer.json":
                raise runner.PhaseEError("RAG occurrence attempts are not contiguous")
            response = runner.load_json(answer_path)
            proof_path = answer_path.with_name(answer_path.name.replace("-answer", "-model"))
            if response.get("binding") != {"plan_sha256": stage["plan_sha256"], "ordinal": ordinal}:
                raise runner.PhaseEError("RAG answer occurrence identity differs")
            if not proof_path.exists():
                raise runner.ModelEvidenceError("completed RAG call lacks contemporaneous post-model evidence", condition)
            proof = runner.load_json(proof_path)
            if proof.get("answer_sha256") != runner.sha256_file(answer_path):
                raise runner.PhaseEError("RAG answer changed after its model proof")
            evidence = proof.get("evidence", {})
            for key in ("name", "tag_digest", "blob_sha256", "details"):
                if evidence.get(key) != model_identity[key]:
                    raise runner.ModelEvidenceError("RAG occurrence model identity differs", condition)
            tags = child_dir / proof["tags"]
            show = child_dir / proof["show"]
            runner.validate_child_target(child_dir, tags)
            runner.validate_child_target(child_dir, show)
            if (runner.sha256_file(tags) != evidence.get("tags_response_sha256")
                    or runner.sha256_file(show) != evidence.get("show_response_sha256")):
                raise runner.ModelEvidenceError("RAG occurrence model evidence changed", condition)
            runner._validate_stored_model_response(runner.load_json(tags), runner.load_json(show), model_identity)
            key = f"{ordinal}:{index}"
            hashes = {"answer": runner.sha256_file(answer_path), "model": runner.sha256_file(proof_path)}
            if key in saved_calls and saved_calls[key] != hashes:
                raise runner.PhaseEError("saved RAG occurrence hashes changed")
            observed_calls[key] = hashes
            elapsed = max(elapsed, _recorded_seconds(proof.get("condition_time_seconds")))
            candidate = response.get("answer")
            if not isinstance(candidate, str):
                raise runner.PhaseEError("saved RAG answer is malformed")
            if ordinal in answers:
                raise runner.PhaseEError("RAG journal continues after a completed occurrence")
            if candidate.strip() and not candidate.startswith("ERROR:"):
                answers[ordinal] = candidate
    if set(saved_calls) - set(observed_calls):
        raise runner.PhaseEError("saved RAG occurrence evidence is missing")
    stage["calls"] = observed_calls
    stage["time_seconds"] = elapsed
    return answers, counts, elapsed


def evaluate_ordered(records, kg, *, questions, condition, identity, child_dir,
                     status, status_path, ollama_url, model_identity, completed_only=False):
    stage = status.setdefault("stages", {}).setdefault(f"rag_{condition}", {"attempts": []})
    directory = child_dir / "calls" / condition
    plan_path = directory / "requests.json"
    binding = {"identity": identity, "condition": condition,
               "questions_sha256": runner.sha256_bytes(runner.canonical_json_bytes(questions)),
               "graph_sha256": runner.sha256_bytes(runner.canonical_json_bytes(kg))}
    if plan_path.exists():
        plan = runner.load_json(plan_path)
        if plan.get("binding") != binding:
            raise runner.PhaseEError("saved RAG request plan belongs to different inputs")
        if stage.get("plan_sha256") != runner.sha256_file(plan_path):
            raise runner.PhaseEError("saved RAG request plan lacks its original hash binding")
    else:
        if completed_only or stage.get("plan_sha256") or stage.get("calls"):
            raise runner.PhaseEError("saved RAG request plan is missing")
        prompts = []
        evaluator.evaluate(records, kg, kg_identity=condition, ollama_url=ollama_url,
            ollama_model=model_identity["name"], question_panel=questions,
            llm_call=lambda _url, _model, prompt: prompts.append(prompt) or "")
        plan = {"binding": binding, "prompts": prompts}
        runner.write_json_once(plan_path, plan, child_dir)
        stage["plan_sha256"] = runner.sha256_file(plan_path)
        runner.write_status(status_path, status, child_dir)
    prompts = plan.get("prompts")
    if (not questions or not isinstance(prompts, list)
            or len(prompts) != len(questions)*len(evaluator.MODES)
            or not all(isinstance(prompt, str) for prompt in prompts)):
        raise runner.PhaseEError("RAG request plan occurrence coverage differs")
    answers, counts, previous_seconds = _saved_occurrences(
        directory, prompts, stage, child_dir, condition, model_identity)
    if completed_only and len(answers) != len(prompts):
        raise runner.PhaseEError("completed RAG stage lacks its successful occurrence evidence")
    results = {mode: [] for mode in evaluator.MODES}
    correct = {mode: 0 for mode in evaluator.MODES}
    started = time.time()
    called = False
    for ordinal, prompt in enumerate(prompts):
        occurrence = directory / f"{ordinal:06d}"
        answer = answers.get(ordinal)
        if answer is None:
            index = counts[ordinal]+1
            answer_path = occurrence / f"{index:03d}-answer.json"
            proof_path = occurrence / f"{index:03d}-model.json"
            answer = evaluator.call_llm(ollama_url, model_identity["name"], prompt)
            called = True
            runner.write_json_once(answer_path, {"binding": {"plan_sha256": stage["plan_sha256"], "ordinal": ordinal},
                "answer": answer, "recorded_at": runner.utc_now()}, child_dir)
            # Once the answer is durable, absence of this contemporaneous proof
            # must fail closed on resume; a later query cannot replace it.
            evidence, tags_bytes, show_bytes = runner.fetch_matching_model(ollama_url, model_identity)
            evidence_dir = child_dir / "calls" / "model-evidence"
            tags = evidence_dir / (runner.sha256_bytes(tags_bytes)+".json")
            show = evidence_dir / (runner.sha256_bytes(show_bytes)+".json")
            runner.write_bytes_once(tags, tags_bytes, child_dir)
            runner.write_bytes_once(show, show_bytes, child_dir)
            stage["time_seconds"] = previous_seconds + time.time() - started
            runner.write_json_once(proof_path, {"answer_sha256": runner.sha256_file(answer_path),
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
    if not completed_only:
        runner.write_status(status_path, status, child_dir)
    count = len(questions)
    return {"metadata": {"kg": runner.stable_child_path(child_dir/"projections"/f"{condition}-seed-42.json")
                if condition != "gold" else runner.stable_child_path(child_dir/"projections/gold-oracle.json"),
                "n_questions": count, "model": model_identity["name"], "time_seconds": stage["time_seconds"]},
            "accuracy": {mode: correct[mode]/count for mode in correct},
            "correct_counts": correct, "results": results}
