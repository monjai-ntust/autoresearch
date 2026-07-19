"""
CycleGT: Generate round-trip consistency data from gold triples.

For each gold (head, rel, tail) triple in SciERC train, generate a
synthetic sentence using either Qwen-0.5B locally or Qwen3:32b via Ollama.
Output is a JSONL compatible with synth_loader.py for cycle training.

This implements the G2T (graph-to-text) direction of the CycleGT closed loop:
  gold triple → decoder generates sentence → [saved to JSONL]
  training: encoder re-extracts triple from generated sentence → cycle loss

Usage:
    # Qwen-0.5B local (fast, lower quality)
    uv run python generate_cycle_data.py \
        --backend qwen-local \
        --out-jsonl data/cycle_data.jsonl

    # Qwen3:32b via Ollama (slower, higher quality)
    uv run python generate_cycle_data.py \
        --backend ollama --ollama-model qwen3:32b \
        --out-jsonl data/cycle_data_32b.jsonl
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.scierc import build_dataloaders, ID2REL, NO_REL_ID


# Reuse prompt template from decoder_d.py
PROMPT_TEMPLATE = (
    "Write one short scientific English sentence that expresses this "
    "knowledge graph triple. Do not add explanations, lists, or bullet "
    "points. Use natural prose.\n\n"
    "Triple: ({head}, {rel}, {tail})\n"
    "Sentence:"
)

RELATION_PHRASES = {
    "USED-FOR":     "is used for",
    "FEATURE-OF":   "is a feature of",
    "HYPONYM-OF":   "is a kind of",
    "EVALUATE-FOR": "is evaluated for",
    "PART-OF":      "is part of",
    "COMPARE":      "is compared with",
    "CONJUNCTION":  "and",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["qwen-local", "ollama"], default="qwen-local")
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--ollama-model", default="qwen3:32b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased",
                   help="SciBERT tokenizer for loading SciERC data.")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=60,
                   help="Max tokens for local Qwen. For Ollama with thinking models, "
                        "use --ollama-max-tokens (default 500) to account for thinking overhead.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-containment", type=float, default=0.5)
    p.add_argument("--max-triples", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def iter_train_triples(train_loader):
    """
    Yield dicts for every gold relation in SciERC train split.
    Includes tail_entity_type for synth_loader compatibility.
    """
    for batch in train_loader:
        words_batch = batch["words"]
        ents_batch = batch["gold_entities"]
        rels_batch = batch["gold_relations"]
        for sent_idx in range(len(words_batch)):
            words = words_batch[sent_idx]
            ents = ents_batch[sent_idx]
            type_by_span = {(s, e): t for (s, e, t) in ents}
            for (h_span, t_span, rel_id) in rels_batch[sent_idx]:
                if rel_id == NO_REL_ID:
                    continue
                hs, he = h_span
                ts, te = t_span
                head = " ".join(words[hs:he + 1])
                tail = " ".join(words[ts:te + 1])
                head_type = type_by_span.get(h_span, "Method")
                tail_type = type_by_span.get(t_span, "Method")
                yield {
                    "head": head,
                    "rel_str": ID2REL[rel_id],
                    "tail": tail,
                    "rel_id": int(rel_id),
                    "entity_type": head_type,
                    "tail_entity_type": tail_type,
                    "source_sentence": " ".join(words),
                }


def check_containment(sentence, head, tail):
    """Check if head and tail appear in the generated sentence."""
    s_lower = sentence.lower()
    h_found = head.lower() in s_lower
    t_found = tail.lower() in s_lower
    if h_found and t_found:
        return 1.0
    elif h_found or t_found:
        return 0.5
    return 0.0


def humanize_rel(rel):
    return RELATION_PHRASES.get(rel, rel.lower().replace("-", " "))


def build_prompt(head, rel_str, tail):
    return PROMPT_TEMPLATE.format(
        head=head, rel=humanize_rel(rel_str), tail=tail,
    )


def generate_qwen_local(decoder, triples_batch, max_new_tokens, temperature):
    """Use FrozenQwenDecoder for batch generation."""
    triple_tuples = [(t["head"], humanize_rel(t["rel_str"]), t["tail"])
                     for t in triples_batch]
    sentences = decoder.generate_batch(
        triple_tuples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return sentences


def generate_ollama(ollama_url, model_name, triples_batch, max_new_tokens, temperature):
    """Generate one-by-one via Ollama chat API (required for Qwen3 chat models)."""
    import subprocess
    sentences = []
    for t in triples_batch:
        prompt = build_prompt(t["head"], t["rel_str"], t["tail"])
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": 100},
        })
        try:
            result = subprocess.run(
                ["curl", "-s", f"{ollama_url}/api/chat", "-d", payload],
                capture_output=True, text=True, timeout=300,
            )
            resp = json.loads(result.stdout)
            text = resp.get("message", {}).get("content", "")
            text = text.strip().strip('"').strip("'").split("\n")[0]
            sentences.append(text)
        except Exception as e:
            print(f"  Ollama error: {e}")
            sentences.append("")
    return sentences


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== CycleGT: Generate Round-Trip Data ===")
    print(f"  backend: {args.backend}")
    print(f"  min_containment: {args.min_containment}")
    print(f"  output: {args.out_jsonl}")

    # Load SciERC train
    sci_tok = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, _, _ = build_dataloaders(
        sci_tok, batch_size=args.batch_size, max_length=128,
    )

    # Collect all triples
    all_triples = list(iter_train_triples(train_loader))
    if args.max_triples > 0:
        all_triples = all_triples[:args.max_triples]
    print(f"  gold triples: {len(all_triples)}")

    # Init decoder
    decoder = None
    if args.backend == "qwen-local":
        from models.decoder_d import FrozenQwenDecoder
        decoder = FrozenQwenDecoder(args.decoder_name, device=device)
        print(f"  decoder: {args.decoder_name} (local)")
    else:
        print(f"  decoder: {args.ollama_model} via Ollama ({args.ollama_url})")

    # Generate
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_accepted = 0
    n_total = 0
    t0 = time.time()

    with open(out_path, "w") as fout:
        for i in range(0, len(all_triples), args.batch_size):
            batch = all_triples[i:i + args.batch_size]

            if args.backend == "qwen-local":
                sentences = generate_qwen_local(
                    decoder, batch, args.max_new_tokens, args.temperature,
                )
            else:
                sentences = generate_ollama(
                    args.ollama_url, args.ollama_model, batch,
                    args.max_new_tokens, args.temperature,
                )

            for triple, sent in zip(batch, sentences):
                n_total += 1
                if not sent.strip():
                    continue
                cont = check_containment(sent, triple["head"], triple["tail"])
                if cont < args.min_containment:
                    continue

                record = {
                    "synth_sentence": sent.strip(),
                    "head": triple["head"],
                    "tail": triple["tail"],
                    "rel": triple["rel_str"],
                    "rel_id": triple["rel_id"],
                    "entity_type": triple["entity_type"],
                    "tail_entity_type": triple["tail_entity_type"],
                    "containment": cont,
                    "source_sentence": triple["source_sentence"],
                }
                fout.write(json.dumps(record) + "\n")
                n_accepted += 1

            if (i // args.batch_size + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{n_total}/{len(all_triples)}] accepted={n_accepted} "
                      f"rate={n_accepted/n_total:.1%} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"  Total triples: {n_total}")
    print(f"  Accepted: {n_accepted} ({n_accepted/max(n_total,1):.1%})")
    print(f"  Output: {out_path}")
    print(f"  Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
