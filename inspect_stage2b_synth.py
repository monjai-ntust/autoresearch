"""
Stage 2b sanity check: dump a handful of Qwen-0.5B synth sentences for
real SciERC gold triples, so we can *read* what the critic was actually
distinguishing from arXiv text.

If these sentences look like "X is used for Y." with the right content,
then the critic signal is meaningful and Stage 2c can trust it.

If they look like gibberish, list bullets, or Chinese translation, then
the critic was learning "English vs garbage" and Stage 2c needs to
filter the decoder output (or reprompt).

Run on DGX:
    uv run python inspect_stage2b_synth.py
"""
import random
from transformers import AutoTokenizer

from data.scierc import build_dataloaders, ID2REL, NO_REL_ID
from models.decoder_d import FrozenQwenDecoder

SEED = 42
N_TRIPLES = 16

def main():
    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    train_loader, _, _ = build_dataloaders(tokenizer, batch_size=16, max_length=128)

    # Pool all gold relations across first few batches
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

    decoder = FrozenQwenDecoder("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
    synth = decoder.generate_batch(sample, max_new_tokens=40, temperature=0.8, top_p=0.9)

    print("\n" + "=" * 80)
    print("Stage 2b synth sentence audit")
    print("=" * 80)
    for (h, r, t), s in zip(sample, synth):
        print(f"\nTriple:  ({h!r}, {r}, {t!r})")
        print(f"  Synth: {s!r}")


if __name__ == "__main__":
    main()
