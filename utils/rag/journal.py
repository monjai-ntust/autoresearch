"""Ordered RAG occurrences with original answers and contemporaneous identity proof."""
from __future__ import annotations

import time
from pathlib import Path

from utils.rag import evaluator, runner


def evaluate_ordered(records, kg, *, questions, condition, identity, child_dir,
                     status, status_path, ollama_url, model_identity):
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
        prompts = []
        evaluator.evaluate(records, kg, kg_identity=condition, ollama_url=ollama_url,
            ollama_model=model_identity["name"], question_panel=questions,
            llm_call=lambda _url, _model, prompt: prompts.append(prompt) or "")
        plan = {"binding": binding, "prompts": prompts}
        runner.write_json_once(plan_path, plan, child_dir)
        stage["plan_sha256"] = runner.sha256_file(plan_path)
        runner.write_status(status_path, status, child_dir)
    prompts = plan.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != len(questions)*len(evaluator.MODES):
        raise runner.PhaseEError("RAG request plan occurrence coverage differs")
    results = {mode: [] for mode in evaluator.MODES}
    correct = {mode: 0 for mode in evaluator.MODES}
    started = time.time()
    for ordinal, prompt in enumerate(prompts):
        occurrence = directory / f"{ordinal:06d}"
        prior = sorted(occurrence.glob("*-answer.json")) if occurrence.exists() else []
        answer = None
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
            saved = stage.setdefault("calls", {}).get(f"{ordinal}:{index}")
            if saved is not None and saved != {"answer": runner.sha256_file(answer_path), "model": runner.sha256_file(proof_path)}:
                raise runner.PhaseEError("saved RAG occurrence hashes changed")
            stage.setdefault("calls", {})[f"{ordinal}:{index}"] = {"answer": runner.sha256_file(answer_path), "model": runner.sha256_file(proof_path)}
            runner.write_status(status_path, status, child_dir)
            candidate = response.get("answer")
            if not isinstance(candidate, str):
                raise runner.PhaseEError("saved RAG answer is malformed")
            if candidate.strip() and not candidate.startswith("ERROR:"):
                if answer is not None:
                    raise runner.PhaseEError("RAG journal repeats a completed occurrence")
                answer = candidate
        if answer is None:
            index = len(prior)+1
            answer_path = occurrence / f"{index:03d}-answer.json"
            proof_path = occurrence / f"{index:03d}-model.json"
            answer = evaluator.call_llm(ollama_url, model_identity["name"], prompt)
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
            runner.write_json_once(proof_path, {"answer_sha256": runner.sha256_file(answer_path),
                "evidence": evidence, "checked_at": runner.utc_now(),
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
    count = len(questions)
    return {"metadata": {"kg": runner.stable_child_path(child_dir/"projections"/f"{condition}-seed-42.json")
                if condition != "gold" else runner.stable_child_path(child_dir/"projections/gold-oracle.json"),
                "n_questions": count, "model": model_identity["name"], "time_seconds": time.time()-started},
            "accuracy": {mode: correct[mode]/count for mode in correct},
            "correct_counts": correct, "results": results}
