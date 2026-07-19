"""
EntiGraph-style entity-centric synthetic data generation for domain pre-training.

Adapted from "Synthetic Continued Pretraining" (ICLR 2025 Oral).

Pipeline:
  1. Extract salient entities from each ACCORD sentence
  2. For each entity pair, generate a diverse synthetic sentence
     describing their relationship
  3. Output JSONL for continued MLM pre-training

Uses on-premise Qwen3:32b via Ollama. No data leaves local network.

Usage:
    uv run python generate_entigraph.py \
        --input data/code_accord/entities/train.csv \
        --output data/accord_entigraph.jsonl \
        --max-pairs-per-doc 5
"""
import argparse
import csv
import json
import subprocess
import time
import random
from pathlib import Path


ENTITY_EXTRACT_PROMPT = """Read this sentence from a building regulation document and list all important entities (objects, properties, values, technical terms). Return ONLY a JSON list of strings.

Sentence: "{sentence}"

Entities:"""

RELATION_SYNTH_PROMPT = """You are an expert in building regulations and construction standards. Given two entities from a building regulation document, write ONE new sentence that describes how they relate in the context of building codes. The sentence should be natural, technically accurate, and different from the original.

Entity 1: {entity1}
Entity 2: {entity2}
Original context: {context}

New sentence:"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="ACCORD entities train CSV")
    p.add_argument("--output", required=True, help="Output JSONL for pre-training")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-model", default="qwen3:32b")
    p.add_argument("--max-pairs-per-doc", type=int, default=5,
                   help="Max entity pairs to synthesize per source sentence")
    p.add_argument("--max-docs", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def call_ollama(ollama_url, model, prompt, max_tokens=100):
    """On-premise LLM call."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "num_predict": max_tokens},
    })
    try:
        result = subprocess.run(
            ["curl", "-s", f"{ollama_url}/api/chat", "-d", payload],
            capture_output=True, text=True, timeout=60,
        )
        resp = json.loads(result.stdout)
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        return ""


def extract_entities_from_bio(processed_content, label):
    """Extract entity strings from BIO tags (faster than LLM extraction)."""
    words = processed_content.split()
    tags = label.split()
    entities = []
    current = []
    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current:
                entities.append(" ".join(current))
            current = [word]
        elif tag.startswith("I-") and current:
            current.append(word)
        else:
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))
    return list(set(entities))


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load ACCORD sentences with entities
    sentences = []
    with open(args.input) as f:
        for row in csv.DictReader(f):
            content = row["content"]
            processed = row.get("processed_content", content)
            label = row.get("label", "")
            entities = extract_entities_from_bio(processed, label) if label else []
            if len(entities) >= 2:
                sentences.append({
                    "content": content,
                    "entities": entities,
                })

    if args.max_docs > 0:
        sentences = sentences[:args.max_docs]

    print(f"=== EntiGraph Synthesis (On-Premise) ===")
    print(f"  Source sentences: {len(sentences)} (with ≥2 entities)")
    print(f"  Max pairs/doc: {args.max_pairs_per_doc}")
    print(f"  Model: {args.ollama_model}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_generated = 0
    t0 = time.time()

    with open(out_path, "w") as fout:
        # Also write original sentences (for diversity)
        for sent in sentences:
            fout.write(json.dumps({"text": sent["content"]}) + "\n")
            n_generated += 1

        # Generate synthetic sentences from entity pairs
        for i, sent in enumerate(sentences):
            entities = sent["entities"]
            pairs = [(a, b) for a in entities for b in entities if a != b]
            random.shuffle(pairs)
            pairs = pairs[:args.max_pairs_per_doc]

            for e1, e2 in pairs:
                prompt = RELATION_SYNTH_PROMPT.format(
                    entity1=e1, entity2=e2, context=sent["content"][:200],
                )
                synth = call_ollama(args.ollama_url, args.ollama_model, prompt)
                if synth and len(synth.split()) >= 5:
                    # Clean: take first sentence only
                    synth = synth.split("\n")[0].strip().strip('"')
                    fout.write(json.dumps({"text": synth}) + "\n")
                    n_generated += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(sentences)}] generated={n_generated} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    expansion = n_generated / len(sentences) if sentences else 0
    print(f"\n=== Done ===")
    print(f"  Source: {len(sentences)} sentences")
    print(f"  Generated: {n_generated} sentences ({expansion:.1f}x expansion)")
    print(f"  Output: {out_path}")
    print(f"  Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
