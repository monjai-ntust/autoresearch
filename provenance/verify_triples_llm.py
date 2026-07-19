"""
LLM-as-Verifier: Use on-premise Qwen3:32b to verify encoder-predicted triples.

For each predicted triple, asks the LLM:
"Given this sentence, is the relation (head, RELATION, tail) correct?"

This is Layer 2 of the KG quality pipeline:
  Layer 1: Confidence filtering (softmax threshold) — done in inference_kg.py
  Layer 2: LLM semantic verification — this script
  Layer 3: KG consistency check — todo

All processing runs on-premise (Ollama). No data leaves the local network.

Usage:
    uv run python verify_triples_llm.py \
        --input results/kg_inference_scierc_test.jsonl \
        --output results/kg_verified_scierc_test.jsonl \
        --min-triple-conf 0.3
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


VERIFY_PROMPT_SIMPLE = """You are a scientific knowledge graph expert. Given a sentence from a scientific paper and a predicted relation triple, determine if the triple is correct.

Sentence: "{sentence}"

Predicted triple: ({head}, {relation}, {tail})

Is this triple correctly extracted from the sentence? Consider:
1. Are both entities actually mentioned in the sentence?
2. Does the stated relation accurately describe how they relate in the sentence?
3. Is the direction of the relation correct?

Answer with ONLY one word: Yes, No, or Uncertain."""

VERIFY_PROMPT_CORRECT = """You are a scientific knowledge graph expert. Given a sentence and a predicted relation triple, choose one action:

Sentence: "{sentence}"

Predicted triple: ({head}, {relation}, {tail})

Available relations: USED-FOR, FEATURE-OF, HYPONYM-OF, EVALUATE-FOR, PART-OF, COMPARE, CONJUNCTION

Choose ONE action:
- KEEP: The triple is correct as-is.
- CORRECT: The triple has an error. Provide the corrected triple.
- DISCARD: The triple is wrong and cannot be fixed.

Answer in this exact format (one line only):
KEEP
or
CORRECT: (corrected_head, corrected_relation, corrected_tail)
or
DISCARD"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input JSONL from inference_kg.py")
    p.add_argument("--output", required=True, help="Output JSONL with LLM verification")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-model", default="qwen3:32b")
    p.add_argument("--min-triple-conf", type=float, default=0.0,
                   help="Only verify triples with confidence >= this threshold. "
                        "Saves LLM calls on obviously bad triples.")
    p.add_argument("--mode", default="correct", choices=["simple", "correct"],
                   help="simple: Yes/No/Uncertain. correct: Keep/Correct/Discard.")
    p.add_argument("--max-docs", type=int, default=0, help="0 = all")
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-request Ollama timeout in seconds.")
    p.add_argument("--retries", type=int, default=3,
                   help="Number of Ollama attempts per triple.")
    p.add_argument("--retry-backoff", type=float, default=2.0,
                   help="Initial retry delay in seconds; doubles after each failure.")
    return p.parse_args()


def call_ollama(ollama_url, model, prompt, timeout=300, retries=3, retry_backoff=2.0):
    """Call Ollama chat API with think:false for fast inference."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 80},
    })
    delay = retry_backoff
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["curl", "-sS", f"{ollama_url}/api/chat", "-d", payload],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"curl exited {result.returncode}")
            resp = json.loads(result.stdout)
            text = resp.get("message", {}).get("content", "").strip()
            if not text:
                raise RuntimeError(f"empty Ollama response: {result.stdout[:200]}")
            return text
        except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            if attempt < retries:
                print(f"  Ollama attempt {attempt}/{retries} failed: {e}; retrying in {delay:.1f}s")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"  Ollama failed after {retries} attempts: {e}")
    return f"Error: {last_error}"


def parse_verdict_simple(response):
    """Parse LLM response to Yes/No/Uncertain."""
    r = response.lower().strip().rstrip(".").strip()
    if r.startswith("yes"):
        return {"action": "keep"}
    elif r.startswith("no"):
        return {"action": "discard"}
    elif r.startswith("uncertain"):
        return {"action": "uncertain"}
    return {"action": "unknown"}


def parse_verdict_correct(response):
    """Parse Keep/Correct/Discard response with optional correction."""
    r = response.strip()
    first_line = r.split("\n")[0].strip()

    if first_line.upper().startswith("KEEP"):
        return {"action": "keep"}
    elif first_line.upper().startswith("DISCARD"):
        return {"action": "discard"}
    elif first_line.upper().startswith("CORRECT"):
        # Try to parse: CORRECT: (head, relation, tail)
        rest = first_line[len("CORRECT"):].strip().lstrip(":").strip()
        rest = rest.strip("()").strip()
        parts = [p.strip().strip("'\"") for p in rest.split(",")]
        if len(parts) == 3:
            return {
                "action": "correct",
                "corrected_head": parts[0],
                "corrected_relation": parts[1].upper().replace(" ", "-"),
                "corrected_tail": parts[2],
            }
        return {"action": "correct_failed", "raw": rest}
    return {"action": "unknown"}


def main():
    args = parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    if args.max_docs > 0:
        records = records[:args.max_docs]

    total_triples = sum(len(r["predicted_triples"]) for r in records)
    eligible = sum(
        1 for r in records for t in r["predicted_triples"]
        if t["triple_conf"] >= args.min_triple_conf
    )
    use_correct = args.mode == "correct"
    prompt_template = VERIFY_PROMPT_CORRECT if use_correct else VERIFY_PROMPT_SIMPLE
    parse_fn = parse_verdict_correct if use_correct else parse_verdict_simple

    print(f"=== LLM Triple Verification (On-Premise) ===")
    print(f"  Model: {args.ollama_model} via {args.ollama_url}")
    print(f"  Mode: {args.mode}")
    print(f"  Input: {args.input} ({len(records)} docs, {total_triples} triples)")
    print(f"  Min confidence: {args.min_triple_conf} → {eligible} triples to verify")
    print(f"  Timeout/retry: {args.timeout}s, retries={args.retries}, backoff={args.retry_backoff}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_verified = 0
    counts = {"keep": 0, "discard": 0, "correct": 0, "correct_failed": 0,
              "uncertain": 0, "unknown": 0, "skip": 0}
    t0 = time.time()

    with open(out_path, "w") as fout:
        for rec in records:
            verified_triples = []

            for triple in rec["predicted_triples"]:
                if triple["triple_conf"] < args.min_triple_conf:
                    triple["llm_verdict"] = "skipped"
                    verified_triples.append(triple)
                    counts["skip"] += 1
                    continue

                prompt = prompt_template.format(
                    sentence=rec["sentence"],
                    head=triple["head_text"],
                    relation=triple["relation"],
                    tail=triple["tail_text"],
                )
                response = call_ollama(
                    args.ollama_url,
                    args.ollama_model,
                    prompt,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_backoff=args.retry_backoff,
                )
                verdict = parse_fn(response)

                triple["llm_verdict"] = verdict["action"]
                triple["llm_raw"] = response

                # Apply correction if available
                if verdict["action"] == "correct" and "corrected_head" in verdict:
                    triple["corrected_head"] = verdict["corrected_head"]
                    triple["corrected_relation"] = verdict["corrected_relation"]
                    triple["corrected_tail"] = verdict["corrected_tail"]

                verified_triples.append(triple)
                counts[verdict["action"]] = counts.get(verdict["action"], 0) + 1
                n_verified += 1

            rec["predicted_triples"] = verified_triples
            fout.write(json.dumps(rec) + "\n")

            if (rec["doc_id"] + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{rec['doc_id']+1}/{len(records)}] "
                      f"verified={n_verified} keep={counts['keep']} "
                      f"correct={counts['correct']} discard={counts['discard']} "
                      f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n=== Verification Complete ===")
    print(f"  Verified: {n_verified}, Skipped: {counts['skip']}")
    for action in ["keep", "correct", "discard", "correct_failed", "uncertain", "unknown"]:
        if counts[action] > 0:
            print(f"  {action:15s}: {counts[action]} ({counts[action]/max(n_verified,1):.1%})")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(n_verified,1):.1f}s/triple)")
    print(f"  Output: {out_path}")

    # Evaluate: compare different filters vs gold
    with open(out_path) as f:
        verified_records = [json.loads(line) for line in f]

    has_gold = any(r["gold_triples"] for r in verified_records)
    if has_gold:
        def make_triple_set(triples, filter_fn, use_corrections=False):
            result = set()
            for t in triples:
                if not filter_fn(t):
                    continue
                if use_corrections and t.get("llm_verdict") == "correct" and "corrected_head" in t:
                    # Use corrected triple — match against gold by text
                    h = t["corrected_head"].lower().strip()
                    tl = t["corrected_tail"].lower().strip()
                    rel = t["corrected_relation"]
                else:
                    h = t.get("head_text", "").lower().strip()
                    tl = t.get("tail_text", "").lower().strip()
                    rel = t["relation"]
                result.add((h, tl, rel))
            return result

        def make_gold_set(triples):
            return {
                (t["head_text"].lower().strip(), t["tail_text"].lower().strip(), t["relation"])
                for t in triples
            }

        print(f"\n  Quality comparison (with gold labels):")
        filters = [
            ("All predicted", lambda t: True, False),
            ("Confidence >= 0.5", lambda t: t["triple_conf"] >= 0.5, False),
            ("LLM keep", lambda t: t.get("llm_verdict") == "keep", False),
            ("LLM keep+correct (orig)", lambda t: t.get("llm_verdict") in ("keep", "correct"), False),
            ("LLM keep+correct (fixed)", lambda t: t.get("llm_verdict") in ("keep", "correct"), True),
            ("Conf>=0.5 + keep", lambda t: t["triple_conf"] >= 0.5 and t.get("llm_verdict") == "keep", False),
            ("Conf>=0.5 + keep+corr", lambda t: t["triple_conf"] >= 0.5 and t.get("llm_verdict") in ("keep", "correct"), True),
        ]
        for label, filter_fn, use_corr in filters:
            tp = fp = fn = 0
            for rec in verified_records:
                pred_set = make_triple_set(rec["predicted_triples"], filter_fn, use_corr)
                gold_set = make_gold_set(rec["gold_triples"])
                tp += len(pred_set & gold_set)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"    {label:30s}: P={p:.3f} R={r:.3f} F1={f1:.3f} (kept={tp+fp})")


if __name__ == "__main__":
    main()
