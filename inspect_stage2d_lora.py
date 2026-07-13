"""
Stage 2d qualitative audit: compare base Qwen output vs v4 LoRA output
on the same 16 SciERC triples. Check:
1. Does the LoRA produce different text from base?
2. Are entities still mentioned?
3. Is the text more "arXiv-like" (the critic's reward signal)?

Run on DGX:
    uv run python inspect_stage2d_lora.py
"""
import random
from transformers import AutoTokenizer

from data.scierc import build_dataloaders, ID2REL, NO_REL_ID
from models.decoder_d import FrozenQwenDecoder
from models.decoder_d_lora import LoRAQwenDecoder

SEED = 42
N_TRIPLES = 16
LORA_PATH = "checkpoints/stage2_009_lora_final"

def main():
    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    train_loader, _, _ = build_dataloaders(tokenizer, batch_size=16, max_length=128)

    pool = []
    for batch in train_loader:
        for sent_idx, rels in enumerate(batch["gold_relations"]):
            words = batch["words"][sent_idx]
            for (h_span, t_span, rel_id) in rels:
                if rel_id == NO_REL_ID:
                    continue
                hs, he = h_span
                ts, te = t_span
                pool.append((
                    " ".join(words[hs:he + 1]),
                    ID2REL[rel_id],
                    " ".join(words[ts:te + 1]),
                ))
        if len(pool) > 500:
            break

    print(f"Pool size: {len(pool)} gold triples")
    sample = random.sample(pool, N_TRIPLES)

    # Base Qwen (frozen, no LoRA)
    print("\n--- Loading base Qwen ---")
    base_decoder = FrozenQwenDecoder("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
    base_synth = base_decoder.generate_batch(sample, max_new_tokens=40, temperature=0.8, top_p=0.9)

    # LoRA Qwen (v4 checkpoint)
    print("\n--- Loading LoRA Qwen (v4) ---")
    lora_decoder = LoRAQwenDecoder("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
    lora_decoder.load_adapters(LORA_PATH)
    lora_result = lora_decoder.sample_with_logprob(sample, temperature=0.8, top_p=0.9)
    lora_synth = lora_result["sentences"]
    report(sample, base_synth, lora_synth)

# Check entity containment
def check_entities(sentence, head, tail):
    s = sentence.lower()
    h = head.lower() in s
    t = tail.lower() in s
    return ("✓" if h else "✗") + "H " + ("✓" if t else "✗") + "T"

def report(sample, base_synth, lora_synth):
    print("\n" + "=" * 80)
    print("Stage 2d LoRA vs Base audit")
    print("=" * 80)
    base_score = 0
    lora_score = 0
    for i, ((h, r, t), bs, ls) in enumerate(zip(sample, base_synth, lora_synth)):
        be = check_entities(bs, h, t)
        le = check_entities(ls, h, t)
        base_score += (h.lower() in bs.lower()) + (t.lower() in bs.lower())
        lora_score += (h.lower() in ls.lower()) + (t.lower() in ls.lower())
        print(f"\n[{i+1}] Triple: ({h!r}, {r}, {t!r})")
        print(f"  Base:  [{be}] {bs!r}")
        print(f"  LoRA:  [{le}] {ls!r}")

    print(f"\n{'='*80}")
    print(f"Entity containment: base={base_score}/{2*N_TRIPLES} ({100*base_score/(2*N_TRIPLES):.0f}%) | "
          f"lora={lora_score}/{2*N_TRIPLES} ({100*lora_score/(2*N_TRIPLES):.0f}%)")


if __name__ == "__main__":
    main()
