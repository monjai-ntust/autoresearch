"""
Stage 2e v3: generate paraphrases of gold SciERC sentences, preserving
entity spans verbatim.

Instead of triple→sentence (v1/v2, which produced structurally wrong
sentences), we do sentence→paraphrase. The gold sentence provides
structural complexity; Qwen provides lexical variation.

Output format (same as generate_synth_dataset.py — compatible with
synth_loader.py):
    {
      "synth_sentence": "...",
      "source_sentence": "...",
      "head": "...", "rel": "USED-FOR", "tail": "...",
      "rel_id": 3,
      "entity_type": "Method",
      "tail_entity_type": "Task",
      "containment": 1.0
    }

One output row per gold RELATION (not per sentence), so sentences with
multiple relations produce multiple rows with the same synth_sentence.

Usage:
    uv run python generate_paraphrase_dataset.py \
        --out-jsonl data/stage2e_paraphrase.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.scierc import build_dataloaders, ID2REL, NO_REL_ID
from models.decoder_d import FrozenQwenDecoder


PARAPHRASE_TEMPLATE = (
    "Paraphrase the following scientific sentence. "
    "You MUST keep these exact phrases unchanged: {entity_list}. "
    "Do not add explanations. Write one sentence only.\n\n"
    "Original: {sentence}\n"
    "Paraphrase:"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--lora-dir", default="",
                   help="If set, load LoRA adapters. Empty = use base Qwen.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _entity_phrases(words, ner_spans):
    """Extract unique entity phrases from NER span list."""
    phrases = []
    seen = set()
    for s, e, t in ner_spans:
        phrase = " ".join(words[s:e + 1])
        if phrase not in seen:
            phrases.append((phrase, t, s, e))
            seen.add(phrase)
    return phrases


def _check_containment(paraphrase, phrase):
    return phrase.lower() in paraphrase.lower()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Stage 2e v3: paraphrase generation ===")
    print(f"  decoder: {args.decoder_name}")
    print(f"  LoRA:    {args.lora_dir or '(base, no LoRA)'}")

    # Sci data
    sci_tok = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, _, _ = build_dataloaders(
        sci_tok, data_dir=data_dir,
        batch_size=1, max_length=args.max_length,
    )
    print(f"  train sentences: {len(train_loader.dataset)}")

    # Decoder
    if args.lora_dir:
        from models.decoder_d_lora import LoRAQwenDecoder
        decoder = LoRAQwenDecoder(args.decoder_name, device=device)
        decoder.load_adapters(args.lora_dir)
        decoder.model.eval()
        for p in decoder.model.parameters():
            p.requires_grad = False
    else:
        decoder = FrozenQwenDecoder(args.decoder_name, device=device)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_sentences = 0
    n_paraphrased = 0
    n_relations_written = 0
    n_all_entities_kept = 0
    n_some_entities_lost = 0
    n_failed = 0

    # Process sentence by sentence (batch_size=1 in loader)
    # Build prompts in batches of BATCH for efficiency.
    BATCH = 8
    prompt_buf = []       # (prompt_str, sentence_str, words, ner, rels, entities)
    results_buf = []

    def flush():
        nonlocal n_paraphrased, n_relations_written, n_all_entities_kept
        nonlocal n_some_entities_lost, n_failed
        if not prompt_buf:
            return
        prompts_only = [p[0] for p in prompt_buf]
        # Use the decoder's underlying model to generate from arbitrary prompts.
        # We build the generation manually because build_prompts expects triples.
        enc = decoder.tokenizer(
            prompts_only,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            out = decoder.model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=decoder.tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        for i, (prompt_str, sent_str, words, ner, rels, entities) in enumerate(prompt_buf):
            new_tokens = out[i][prompt_len:]
            para = decoder.tokenizer.decode(new_tokens, skip_special_tokens=True)
            para = para.strip().split("\n")[0][:500]
            if not para:
                n_failed += 1
                continue
            n_paraphrased += 1

            # Check entity containment
            kept = [e for e in entities if _check_containment(para, e[0])]
            if len(kept) == len(entities):
                n_all_entities_kept += 1
            else:
                n_some_entities_lost += 1

            # Write one row per gold relation where BOTH head and tail are kept
            kept_phrases = {e[0].lower() for e in kept}
            for (h_span, t_span, rel_id) in rels:
                if rel_id == NO_REL_ID:
                    continue
                hs, he = h_span
                ts, te = t_span
                head_phrase = " ".join(words[hs:he + 1])
                tail_phrase = " ".join(words[ts:te + 1])
                h_found = head_phrase.lower() in kept_phrases
                t_found = tail_phrase.lower() in kept_phrases
                containment = 0.5 * float(h_found) + 0.5 * float(t_found)
                if containment < 1.0:
                    continue
                # Find entity types
                type_by_span = {(s, e): t for (s, e, t) in ner}
                h_type = type_by_span.get(h_span, "Method")
                t_type = type_by_span.get(t_span, "Method")
                record = {
                    "synth_sentence": para,
                    "source_sentence": sent_str,
                    "head": head_phrase,
                    "rel": ID2REL[rel_id],
                    "tail": tail_phrase,
                    "rel_id": int(rel_id),
                    "entity_type": h_type,
                    "tail_entity_type": t_type,
                    "containment": containment,
                }
                results_buf.append(record)
                n_relations_written += 1
        prompt_buf.clear()

    with out_path.open("w") as fout:
        for batch in train_loader:
            words = batch["words"][0]
            ner = batch["gold_entities"][0]
            rels = batch["gold_relations"][0]
            n_sentences += 1

            # Skip sentences with no relations
            real_rels = [(h, t, r) for (h, t, r) in rels if r != NO_REL_ID]
            if not real_rels:
                continue

            entities = _entity_phrases(words, ner)
            if not entities:
                continue

            sent_str = " ".join(words)
            entity_list = ", ".join(f'"{e[0]}"' for e in entities)
            prompt = PARAPHRASE_TEMPLATE.format(
                entity_list=entity_list,
                sentence=sent_str,
            )
            prompt_buf.append((prompt, sent_str, words, ner, rels, entities))

            if len(prompt_buf) >= BATCH:
                flush()
                for rec in results_buf:
                    fout.write(json.dumps(rec) + "\n")
                results_buf.clear()

            if n_sentences % 200 == 0:
                print(f"  [{n_sentences}] paraphrased={n_paraphrased} "
                      f"relations={n_relations_written} "
                      f"all_kept={n_all_entities_kept} "
                      f"some_lost={n_some_entities_lost}")

        # Flush remaining
        flush()
        for rec in results_buf:
            fout.write(json.dumps(rec) + "\n")

    print(f"\n=== DONE ===")
    print(f"  total sentences:    {n_sentences}")
    print(f"  paraphrased:        {n_paraphrased}")
    print(f"  all entities kept:  {n_all_entities_kept} ({100*n_all_entities_kept/max(n_paraphrased,1):.1f}%)")
    print(f"  some entities lost: {n_some_entities_lost}")
    print(f"  failed:             {n_failed}")
    print(f"  relations written:  {n_relations_written}")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
