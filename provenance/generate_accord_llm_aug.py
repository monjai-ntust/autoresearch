"""
Generate schema-aware LLM augmentation for CODE-ACCORD relation extraction.

This is intentionally not relation replay: it asks the local LLM to create a
new regulatory sentence while preserving exact head/tail strings and the
ACCORD relation label. Outputs are compatible with data/synth_loader.py.
"""
import argparse
import importlib
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from train_span import DATASET_REGISTRY


REL_DEFINITIONS = {
    "selection": "the head is the item/scope and the tail is a selected, specified, or applicable requirement, method, component, or condition",
    "necessity": "the head requires, must have, should have, or shall satisfy the tail",
    "part-of": "the head is a component, portion, or member of the tail",
    "equal": "the head has a value equal to the tail",
    "greater": "the head has a value greater than the tail",
    "greater-equal": "the head has a value at least the tail",
    "less": "the head has a value less than the tail",
    "less-equal": "the head has a value no greater than the tail",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="accord", choices=list(DATASET_REGISTRY))
    p.add_argument("--model-name", default="microsoft/deberta-large")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=120)
    p.add_argument("--target-relations", default="selection,part-of,necessity,equal,greater-equal")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-model", default="qwen3:32b")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    return p.parse_args()


def call_ollama(args, prompt):
    payload = json.dumps({
        "model": args.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "num_predict": 120},
    })
    delay = args.retry_backoff
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            result = subprocess.run(
                ["curl", "-sS", f"{args.ollama_url}/api/chat", "-d", payload],
                capture_output=True, text=True, timeout=args.timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"curl exited {result.returncode}")
            resp = json.loads(result.stdout)
            text = resp.get("message", {}).get("content", "").strip()
            if not text:
                raise RuntimeError("empty Ollama response")
            return text
        except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            if attempt < args.retries:
                print(f"  Ollama attempt {attempt}/{args.retries} failed: {e}; retrying in {delay:.1f}s")
                time.sleep(delay)
                delay *= 2
    print(f"  Ollama failed: {last_error}")
    return ""


def parse_sentence(text):
    text = text.strip()
    try:
        obj = json.loads(text)
        sent = obj.get("sentence", "")
        if isinstance(sent, str):
            return sent.strip()
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            sent = obj.get("sentence", "")
            if isinstance(sent, str):
                return sent.strip()
        except json.JSONDecodeError:
            pass
    return text.splitlines()[0].strip().strip('"')


def contains_exact(sentence, phrase):
    return phrase.lower() in sentence.lower()


def collect_train_relations(args, ds_mod):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, _, _ = ds_mod.build_dataloaders(
        tokenizer, batch_size=16, max_length=128, seed=args.seed,
    )
    by_rel = defaultdict(list)
    for batch in train_loader:
        for words, ents, rels in zip(
            batch["words"], batch["gold_entities"], batch["gold_relations"]
        ):
            type_by_span = {(s, e): t for (s, e, t) in ents}
            for h_span, t_span, rel_id in rels:
                if rel_id == ds_mod.NO_REL_ID:
                    continue
                rel = ds_mod.ID2REL[int(rel_id)]
                hs, he = h_span
                ts, te = t_span
                by_rel[rel].append({
                    "source_sentence": " ".join(words),
                    "head": " ".join(words[hs:he + 1]),
                    "tail": " ".join(words[ts:te + 1]),
                    "rel": rel,
                    "rel_id": int(rel_id),
                    "entity_type": type_by_span.get(h_span, ds_mod.ENTITY_TYPES[0]),
                    "tail_entity_type": type_by_span.get(t_span, ds_mod.ENTITY_TYPES[0]),
                })
    return by_rel


def make_prompt(rec):
    rel_def = REL_DEFINITIONS.get(rec["rel"], rec["rel"])
    return f"""Create one new construction-regulation sentence for relation extraction training.

Relation label: {rec["rel"]}
Relation meaning: {rel_def}
Head phrase, preserve exactly: {rec["head"]}
Tail phrase, preserve exactly: {rec["tail"]}

Rules:
- The sentence must contain the exact head phrase and exact tail phrase.
- Do not wrap the phrases in tags.
- Express the relation label clearly.
- Keep the sentence realistic for building code or construction requirements.
- Return JSON only: {{"sentence": "..."}}
"""


def main():
    args = parse_args()
    random.seed(args.seed)
    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])
    target_rels = [r.strip() for r in args.target_relations.split(",") if r.strip()]
    by_rel = collect_train_relations(args, ds_mod)

    candidates = []
    per_rel = max(1, args.max_examples // max(1, len(target_rels)))
    for rel in target_rels:
        rel_records = list(by_rel.get(rel, []))
        random.shuffle(rel_records)
        candidates.extend(rel_records[:per_rel])
    random.shuffle(candidates)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    attempted = 0
    rel_kept = defaultdict(int)
    with out_path.open("w") as fout:
        for rec in candidates:
            attempted += 1
            response = call_ollama(args, make_prompt(rec))
            sent = parse_sentence(response)
            if not sent or not contains_exact(sent, rec["head"]) or not contains_exact(sent, rec["tail"]):
                continue
            out = {
                "synth_sentence": sent,
                "head": rec["head"],
                "tail": rec["tail"],
                "rel": rec["rel"],
                "rel_id": rec["rel_id"],
                "entity_type": rec["entity_type"],
                "tail_entity_type": rec["tail_entity_type"],
                "containment": 1.0,
                "source_sentence": rec["source_sentence"],
            }
            fout.write(json.dumps(out) + "\n")
            kept += 1
            rel_kept[rec["rel"]] += 1
            if kept >= args.max_examples:
                break
            if attempted % 25 == 0:
                print(f"attempted={attempted} kept={kept}")

    print(f"attempted={attempted} kept={kept} output={out_path}")
    for rel, count in sorted(rel_kept.items()):
        print(f"  {rel}: {count}")


if __name__ == "__main__":
    main()
