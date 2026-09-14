"""Internal, frozen Table-2 Graph RAG evaluator.

This module intentionally has no command-line parser or file-selection surface.
The same-run table pipeline supplies already authenticated records and graphs.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from typing import Any


MODES = ("llm_only", "text_retrieval", "kg_1hop", "kg_2hop", "hybrid")


def call_llm(ollama_url, model, prompt):
    """On-premise LLM call via Ollama."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 50},
    })
    try:
        result = subprocess.run(
            ["curl", "-s", f"{ollama_url}/api/chat", "-d", payload],
            capture_output=True, text=True, timeout=60,
        )
        resp = json.loads(result.stdout)
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def generate_questions(records, max_q):
    """Generate QA pairs from gold triples."""
    questions = []
    rel_templates = {
        "used-for": ("What is {head} used for?", "{tail}"),
        "feature-of": ("What is a feature of {tail}?", "{head}"),
        "hyponym-of": ("What is {head} a type of?", "{tail}"),
        "evaluate-for": ("What is {head} evaluated for?", "{tail}"),
        "evaluated-with": ("What is {head} evaluated with?", "{tail}"),
        "part-of": ("What is {head} part of?", "{tail}"),
        "compare": ("What is {head} compared with?", "{tail}"),
        "compare-with": ("What is {head} compared with?", "{tail}"),
        "trained-with": ("What is {head} trained with?", "{tail}"),
        "subclass-of": ("What is {head} a subclass of?", "{tail}"),
        "subtask-of": ("What is {head} a subtask of?", "{tail}"),
        "synonym-of": ("What is a synonym of {head}?", "{tail}"),
        "benchmark-for": ("What is {head} a benchmark for?", "{tail}"),
    }

    for rec in records:
        for t in rec.get("gold_triples", []):
            rel = t["relation"].lower()
            if rel not in rel_templates or rel == "conjunction":
                continue
            template, answer_template = rel_templates[rel]
            q = template.format(head=t["head_text"], tail=t["tail_text"])
            a = answer_template.format(head=t["head_text"], tail=t["tail_text"])
            questions.append({
                "question": q,
                "gold_answer": a,
                "relation": rel,
                "sentence": rec["sentence"],
                "doc_id": rec["doc_id"],
                "head": t["head_text"],
                "tail": t["tail_text"],
            })

    seen = set()
    unique = []
    for q in questions:
        key = q["question"]
        if key not in seen:
            seen.add(key)
            unique.append(q)
    if max_q > 0 and len(unique) > max_q:
        import random
        random.seed(42)
        unique = random.sample(unique, max_q)
    return unique


def retrieve_from_kg(kg, query_entity, hops=1):
    """Retrieve KG subgraph around an entity (multi-hop neighborhood)."""
    query_norm = query_entity.lower().strip()
    adj = {}
    for edge in kg["edges"]:
        h, t = edge["head"].lower(), edge["tail"].lower()
        adj.setdefault(h, []).append(edge)
        adj.setdefault(t, []).append(edge)

    visited = set()
    frontier = set()
    for node in adj:
        if query_norm in node or node in query_norm:
            frontier.add(node)
    if not frontier:
        query_words = set(query_norm.split())
        for node in adj:
            node_words = set(node.split())
            if len(query_words & node_words) >= max(1, len(query_words) // 2):
                frontier.add(node)

    context_triples = []
    seen_edges = set()
    for _hop in range(hops):
        next_frontier = set()
        for entity in frontier:
            if entity in visited:
                continue
            visited.add(entity)
            for edge in adj.get(entity, []):
                edge_key = (edge["head"], edge["relation"], edge["tail"])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    context_triples.append(
                        f"({edge['head']}, {edge['relation']}, {edge['tail']})"
                    )
                    next_frontier.add(edge["head"].lower())
                    next_frontier.add(edge["tail"].lower())
        frontier = next_frontier - visited
    return context_triples[:15]


def retrieve_sentences(records, query_words, top_k=3):
    """Simple BM25-style text retrieval: word overlap scoring."""
    query_set = set(query_words.lower().split())
    scored = []
    for rec in records:
        sent_words = set(rec["sentence"].lower().split())
        overlap = len(query_set & sent_words)
        if overlap > 0:
            scored.append((overlap, rec["sentence"]))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_k]]


def check_answer(predicted, gold):
    """Check if gold answer appears in predicted answer (soft match)."""
    pred_lower = predicted.lower().strip()
    gold_lower = gold.lower().strip()
    gold_words = set(gold_lower.split())
    pred_words = set(pred_lower.split())
    if not gold_words:
        return False
    overlap = len(gold_words & pred_words) / len(gold_words)
    return overlap >= 0.5


def build_gold_kg(records):
    """Build KG directly from gold triples (ceiling test)."""
    edges = []
    nodes = {}
    for rec in records:
        for t in rec.get("gold_triples", []):
            h = t["head_text"].lower().strip()
            tl = t["tail_text"].lower().strip()
            rel = t["relation"]
            if not h or not tl or rel == "CONJUNCTION":
                continue
            edge = {"head": h, "relation": rel, "tail": tl, "confidence": 1.0, "n_sources": 1}
            edges.append(edge)
            for e in [h, tl]:
                if e not in nodes:
                    nodes[e] = {"id": e, "frequency": 0}
                nodes[e]["frequency"] += 1
    return {"nodes": list(nodes.values()), "edges": edges}


def evaluate(
    records: list[dict[str, Any]],
    kg: dict[str, Any],
    *,
    kg_identity: str,
    ollama_url: str,
    ollama_model: str,
    max_questions: int,
    llm_call: Callable[[str, str, str], str] = call_llm,
) -> dict[str, Any]:
    """Run the frozen five-mode transform over authenticated in-memory inputs."""
    questions = generate_questions(records, max_questions)
    if not questions:
        raise ValueError(
            "No supported questions were generated; refusing to write an empty RAG result."
        )

    results = {mode: [] for mode in MODES}
    correct = {mode: 0 for mode in MODES}
    t0 = time.time()
    for q in questions:
        prompt1 = f"Answer in 1-2 words. {q['question']}"
        ans1 = llm_call(ollama_url, ollama_model, prompt1)

        retrieved_sents = retrieve_sentences(records, q["head"] + " " + q["tail"])
        context2 = "\n".join(f"- {s[:200]}" for s in retrieved_sents)
        prompt2 = (f"Based on these scientific sentences:\n{context2}\n\n"
                   f"Answer in 1-2 words: {q['question']}")
        ans2 = llm_call(ollama_url, ollama_model, prompt2)

        kg_1hop = retrieve_from_kg(kg, q["head"], hops=1)
        if not kg_1hop:
            kg_1hop = retrieve_from_kg(kg, q["tail"], hops=1)
        context3 = "\n".join(f"- {t}" for t in kg_1hop)
        prompt3 = (f"Based on this knowledge graph:\n{context3}\n\n"
                   f"Answer in 1-2 words: {q['question']}")
        ans3 = llm_call(ollama_url, ollama_model, prompt3)

        kg_2hop = retrieve_from_kg(kg, q["head"], hops=2)
        if not kg_2hop:
            kg_2hop = retrieve_from_kg(kg, q["tail"], hops=2)
        context4 = "\n".join(f"- {t}" for t in kg_2hop)
        prompt4 = (f"Based on this knowledge graph:\n{context4}\n\n"
                   f"Answer in 1-2 words: {q['question']}")
        ans4 = llm_call(ollama_url, ollama_model, prompt4)

        top_sent = retrieved_sents[0][:200] if retrieved_sents else ""
        context5 = f"Knowledge graph:\n" + "\n".join(f"- {t}" for t in kg_1hop[:5])
        context5 += f"\n\nSource text:\n- {top_sent}"
        prompt5 = (f"Based on this evidence:\n{context5}\n\n"
                   f"Answer in 1-2 words: {q['question']}")
        ans5 = llm_call(ollama_url, ollama_model, prompt5)

        answers = [ans1, ans2, ans3, ans4, ans5]
        for mode, answer in zip(MODES, answers):
            is_correct = check_answer(answer, q["gold_answer"])
            correct[mode] += int(is_correct)
            results[mode].append({
                "q": q["question"],
                "pred": answer,
                "gold": q["gold_answer"],
                "correct": is_correct,
            })

    elapsed = time.time() - t0
    count = len(questions)
    return {
        "metadata": {
            "kg": kg_identity,
            "n_questions": count,
            "model": ollama_model,
            "time_seconds": elapsed,
        },
        "accuracy": {mode: correct[mode] / count for mode in correct},
        "correct_counts": correct,
        "results": results,
    }
