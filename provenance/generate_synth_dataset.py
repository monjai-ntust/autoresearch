"""
Stage 2e pre-flight: generate a synthetic paraphrase dataset from a trained
LoRA decoder.

For every gold (head, rel, tail) triple in the SciERC train split, sample
one paraphrase from the frozen LoRA-tuned Qwen, check entity containment,
and write the output to a jsonl for inspection + later use by Stage 2e
encoder training.

Output format (one line per accepted synth example):
    {
      "synth_sentence": "...",
      "source_sentence": "...",      // the gold SciERC sentence this triple came from
      "head": "...", "rel": "USED-FOR", "tail": "...",
      "containment": 1.0,            // 0.0 / 0.5 / 1.0
      "head_found": true, "tail_found": true,
      "rel_id": 3, "entity_type": "Method"
    }

Usage:
    uv run python generate_synth_dataset.py \
        --lora-dir checkpoints/stage2_009_lora_final \
        --out-jsonl data/stage2e_synth_v8.jsonl \
        --min-containment 0.5 \
        --max-per-triple 1
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.scierc import build_dataloaders, ID2REL, NO_REL_ID
from models.decoder_d_lora import LoRAQwenDecoder
from models.encoder_reward import string_containment_reward_single


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased",
                   help="Only used to tokenize the sci dataset pipeline.")
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--lora-dir", required=True,
                   help="peft adapter directory (e.g. checkpoints/stage2_009_lora_final)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--min-containment", type=float, default=0.5,
                   help="Accept synth examples with containment >= this.")
    p.add_argument("--max-per-triple", type=int, default=1,
                   help="How many synth sentences to sample per gold triple.")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-triples", type=int, default=0,
                   help="Cap total triples processed (0 = all). Used for smoke tests.")
    return p.parse_args()


def iter_train_triples(train_loader):
    """
    Yield (head_str, rel_str, tail_str, source_sentence, entity_type) for every
    gold relation in the train split.
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
                yield {
                    "head": head,
                    "rel_str": ID2REL[rel_id],
                    "tail": tail,
                    "rel_id": int(rel_id),
                    "entity_type": head_type,
                    "source_sentence": " ".join(words),
                }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 2e: synth generation ===")
    print(f"  LoRA adapters:   {args.lora_dir}")
    print(f"  min containment: {args.min_containment}")
    print(f"  per triple:      {args.max_per_triple}")
    print(f"  output:          {args.out_jsonl}")

    # Sci tokenizer + loader (we only need train words/triples; dev ignored)
    sci_tok = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, _, _ = build_dataloaders(
        sci_tok, data_dir=data_dir,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"  sci train sentences: {len(train_loader.dataset)}")

    # LoRA decoder — load adapters onto a fresh LoRAQwenDecoder instance.
    # We reuse the same helper as Stage 2d so the adapter loading path is
    # exactly the one the training loop used.
    decoder = LoRAQwenDecoder(args.decoder_name, device=device)
    decoder.load_adapters(args.lora_dir)
    decoder.model.eval()
    for p in decoder.model.parameters():
        p.requires_grad = False

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_kept = 0
    containment_hist = [0, 0, 0]   # 0.0, 0.5, 1.0

    # Gather triples first so we can batch the generate calls.
    all_triples = list(iter_train_triples(train_loader))
    if args.max_triples > 0:
        all_triples = all_triples[: args.max_triples]
    print(f"  total train triples: {len(all_triples)}")

    BATCH = 16
    with out_path.open("w") as fout:
        for start in range(0, len(all_triples), BATCH):
            chunk = all_triples[start : start + BATCH]
            triples_tuples = [
                (c["head"], c["rel_str"], c["tail"]) for c in chunk
            ]
            # Each gold triple gets `max_per_triple` samples (default 1).
            for k in range(args.max_per_triple):
                with torch.no_grad():
                    synth_sentences = decoder.generate_batch(
                        triples_tuples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                for ctx, synth in zip(chunk, synth_sentences):
                    n_total += 1
                    containment = string_containment_reward_single(
                        synth, (ctx["head"], ctx["rel_str"], ctx["tail"]),
                    )
                    idx = {0.0: 0, 0.5: 1, 1.0: 2}.get(containment, 0)
                    containment_hist[idx] += 1
                    if containment < args.min_containment:
                        continue
                    n_kept += 1
                    record = {
                        "synth_sentence": synth,
                        "source_sentence": ctx["source_sentence"],
                        "head": ctx["head"],
                        "rel": ctx["rel_str"],
                        "tail": ctx["tail"],
                        "rel_id": ctx["rel_id"],
                        "entity_type": ctx["entity_type"],
                        "containment": containment,
                    }
                    fout.write(json.dumps(record) + "\n")
            if (start // BATCH) % 10 == 0:
                print(f"  [{start + len(chunk)}/{len(all_triples)}] kept={n_kept}/{n_total}")

    print(f"\n=== DONE ===")
    print(f"  total generated: {n_total}")
    print(f"  kept (containment≥{args.min_containment}): {n_kept}")
    print(f"  containment distribution:")
    print(f"    0.0: {containment_hist[0]} ({100*containment_hist[0]/max(n_total,1):.1f}%)")
    print(f"    0.5: {containment_hist[1]} ({100*containment_hist[1]/max(n_total,1):.1f}%)")
    print(f"    1.0: {containment_hist[2]} ({100*containment_hist[2]/max(n_total,1):.1f}%)")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
