"""Exact request and ordered context evidence for the configured RAG evaluator."""
from __future__ import annotations

import json

from utils.rag import evaluator, runner


SERIALIZATION = "json.dumps defaults, insertion-order keys, ASCII escapes, UTF-8, no trailing newline"
CONTEXT_FIELDS = ("retrieved_triples", "included_triples", "retrieved_sentences", "source_excerpts")


def _template(model, *, legacy=False):
    payload = evaluator.request_payload(model, "")
    if legacy:
        del payload["options"]["seed"]
    return {"model": payload["model"], "role": payload["messages"][0]["role"],
            "stream": payload["stream"], "think": payload["think"], "options": payload["options"],
            "serialization": SERIALIZATION}


def _digest(value):
    return runner.sha256_bytes(runner.canonical_json_bytes(value, newline=False))


def request_bytes(plan, ordinal):
    """Reconstruct the literal JSON passed to curl, without retrieval or a model."""
    if "request_template" in plan:
        template = plan["request_template"]
        payload = evaluator.request_payload(template["model"], plan["prompts"][ordinal],
                                            options=template["options"])
    else:
        payload = evaluator.request_payload(plan["model"], plan["prompts"][ordinal])
    return json.dumps(payload).encode("utf-8")


def request_identity(plan, ordinal):
    """The immutable per-occurrence identities referenced by attempt records."""
    return {key: plan["requests"][ordinal][key]
            for key in ("prompt_sha256", "context_sha256", "request_sha256")}


def capture_request_plan(records, kg, questions, condition, model, ollama_url):
    """Observe each original retrieval once while preparing the existing plan."""
    plan = {"prompts": [], "requests": [], "model": model}

    def observe(*, question, mode, prompt, **context):
        ordinal = len(plan["prompts"])
        question_index = ordinal // len(evaluator.MODES)
        if (question_index >= len(questions) or question != questions[question_index]
                or mode != evaluator.MODES[ordinal % len(evaluator.MODES)]):
            raise runner.PhaseEError("observed RAG occurrence differs from its question panel")
        plan["prompts"].append(prompt)
        plan["requests"].append({
            "ordinal": ordinal, "question_index": question_index,
            "question_sha256": _digest(question), "mode": mode, "context": context,
            "prompt_sha256": runner.sha256_bytes(prompt.encode("utf-8")),
            "context_sha256": _digest(context),
            "request_sha256": runner.sha256_bytes(request_bytes(plan, ordinal)),
        })

    evaluator.evaluate(records, kg, kg_identity=condition, ollama_url=ollama_url,
        ollama_model=model, question_panel=questions, request_observer=observe,
        llm_call=lambda _url, _model, _prompt: "")
    validate_request_plan(plan, questions, model)
    return plan


def _context_prompt(question, mode, context):
    """Check the recorded inclusion boundaries without selecting facts again."""
    retrieved = context["retrieved_triples"]
    included = context["included_triples"]
    sentences = context["retrieved_sentences"]
    excerpts = context["source_excerpts"]
    if len(retrieved) > 15 or len(sentences) > 3:
        raise runner.PhaseEError("saved RAG context exceeds its retrieval cap")
    if mode == "llm_only":
        if any(context.values()):
            raise runner.PhaseEError("LLM-only occurrence contains retrieved context")
        return f"Answer in 1-2 words. {question['question']}"
    if mode == "text_retrieval":
        if retrieved or included or excerpts != [sentence[:200] for sentence in sentences]:
            raise runner.PhaseEError("saved text context differs from its included excerpts")
        context_text = "\n".join(f"- {excerpt}" for excerpt in excerpts)
        return (f"Based on these scientific sentences:\n{context_text}\n\n"
                f"Answer in 1-2 words: {question['question']}")
    if mode in {"kg_1hop", "kg_2hop"}:
        if included != retrieved or sentences or excerpts:
            raise runner.PhaseEError("saved graph context differs from its included facts")
        context_text = "\n".join(f"- {fact}" for fact in included)
        return (f"Based on this knowledge graph:\n{context_text}\n\n"
                f"Answer in 1-2 words: {question['question']}")
    if included != retrieved[:5] or excerpts != [sentences[0][:200] if sentences else ""]:
        raise runner.PhaseEError("saved hybrid context differs from its first-five boundary")
    context_text = "Knowledge graph:\n" + "\n".join(f"- {fact}" for fact in included)
    context_text += f"\n\nSource text:\n- {excerpts[0]}"
    return (f"Based on this evidence:\n{context_text}\n\n"
            f"Answer in 1-2 words: {question['question']}")


def is_current_request_plan(plan, model):
    fields = {"prompts", "requests", "model"}
    return (isinstance(plan, dict) and plan.get("model") == model
            and set(plan) in (fields, fields | {"binding", "producer"}))


def _is_legacy_request_plan(plan, model):
    fields = {"prompts", "requests", "request_template"}
    if not isinstance(plan, dict) or set(plan) not in (fields, fields | {"binding", "producer"}):
        return False
    template = plan["request_template"]
    if not isinstance(template, dict) or not isinstance(template.get("options"), dict):
        return False
    expected = _template(model, legacy=True)
    if "seed" in template["options"]:
        seed = template["options"]["seed"]
        if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
            return False
        expected["options"]["seed"] = seed
    return runner.canonical_json_bytes(template) == runner.canonical_json_bytes(expected)


def validate_request_plan(plan, questions, model, *, allow_legacy=False):
    """Authenticate recorded prompts, ordered contexts and wire payload hashes."""
    if not is_current_request_plan(plan, model) and not (
            allow_legacy and _is_legacy_request_plan(plan, model)):
        raise runner.PhaseEError("saved RAG model or payload identity differs from the configured source")
    prompts = plan.get("prompts")
    requests = plan.get("requests")
    count = len(questions) * len(evaluator.MODES)
    if (not questions or not isinstance(prompts, list) or not isinstance(requests, list)
            or len(prompts) != count or len(requests) != count
            or not all(isinstance(prompt, str) for prompt in prompts)):
        raise runner.PhaseEError("saved RAG request evidence occurrence coverage differs")
    fields = {"ordinal", "question_index", "question_sha256", "mode", "context",
              "prompt_sha256", "context_sha256", "request_sha256"}
    for ordinal, (prompt, request) in enumerate(zip(prompts, requests)):
        index = ordinal // len(evaluator.MODES)
        mode = evaluator.MODES[ordinal % len(evaluator.MODES)]
        if (not isinstance(request, dict) or set(request) != fields
                or type(request.get("ordinal")) is not int or request["ordinal"] != ordinal
                or type(request.get("question_index")) is not int or request["question_index"] != index
                or request.get("question_sha256") != _digest(questions[index])
                or request.get("mode") != mode):
            raise runner.PhaseEError("saved RAG request question or mode identity differs")
        context = request.get("context")
        if (not isinstance(context, dict) or set(context) != set(CONTEXT_FIELDS)
                or any(not isinstance(context[field], list)
                       or not all(isinstance(value, str) for value in context[field])
                       for field in CONTEXT_FIELDS)):
            raise runner.PhaseEError("saved RAG ordered context is malformed")
        if prompt != _context_prompt(questions[index], mode, context):
            raise runner.PhaseEError("saved RAG prompt differs from its ordered context")
        if (request["prompt_sha256"] != runner.sha256_bytes(prompt.encode("utf-8"))
                or request["context_sha256"] != _digest(context)
                or request["request_sha256"] != runner.sha256_bytes(request_bytes(plan, ordinal))):
            raise runner.PhaseEError("saved RAG request or context hash differs")
        if mode == "hybrid":
            graph_context = requests[ordinal - 2]["context"]
            text_context = requests[ordinal - 3]["context"]
            if (context["retrieved_triples"] != graph_context["retrieved_triples"]
                    or context["retrieved_sentences"] != text_context["retrieved_sentences"]):
                raise runner.PhaseEError("saved hybrid context differs from its original retrieval")
    return requests
