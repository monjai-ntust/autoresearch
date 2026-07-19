"""
Generate entity token masks for entity-aware ELECTRA pre-training.

Uses the trained span NER model to predict entity spans on unlabeled text,
then marks which tokens belong to predicted entities. Output is a JSONL file
where each line has {"text": ..., "entity_token_mask": [0,0,1,1,1,0,0,...]}.

The entity_token_mask aligns with the tokenizer output (input_ids), where
1 = token belongs to a predicted entity span, 0 = non-entity.

Usage:
    uv run python generate_entity_masks.py \
        --checkpoint checkpoints/span_teacher_v10.pt \
        --input data/arxiv_real/cs_validation.jsonl \
        --output data/arxiv_real/cs_validation_entity_masks.jsonl

    uv run python generate_entity_masks.py \
        --checkpoint checkpoints/span_teacher_v10.pt \
        --input data/conll04_raw.jsonl \
        --output data/conll04_raw_entity_masks.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.bert_kg_encoder import BertKGExtractor
from data.scierc import ENTITY_TYPES, NUM_BIO_TAGS, NUM_RELATIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Span NER model checkpoint")
    p.add_argument("--input", required=True, help="Input JSONL (each line: {\"text\": ...})")
    p.add_argument("--output", required=True, help="Output JSONL with entity_token_mask")
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-span-width", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Minimum softmax probability for entity prediction")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model
    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=NUM_BIO_TAGS,
        num_relations=NUM_RELATIONS,
        num_entity_types=len(ENTITY_TYPES),
        use_span_ner=True,
        max_span_width=args.max_span_width,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "encoder" in ckpt:
        model.load_state_dict(ckpt["encoder"])
    elif "discriminator" in ckpt:
        model.load_state_dict(ckpt["discriminator"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Read input
    with open(args.input) as f:
        texts = [json.loads(line)["text"] for line in f]
    print(f"Input: {len(texts)} documents from {args.input}")

    # Process
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_with_entities = 0

    with open(out_path, "w") as fout:
        for i, text in enumerate(texts):
            enc = tokenizer(
                text, truncation=True, max_length=args.max_length,
                padding=False, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            T = input_ids.size(1)

            # Build word_ids mapping
            word_ids = enc.word_ids(0)

            # Count words
            valid_word_ids = [w for w in word_ids if w is not None]
            num_words = max(valid_word_ids) + 1 if valid_word_ids else 0

            # Entity mask at token level
            entity_token_mask = [0] * T

            if num_words > 0:
                with torch.no_grad():
                    hidden = model.encode(
                        modality="text", input_ids=input_ids, attention_mask=attention_mask,
                    )
                    span_logits, candidates = model.forward_span_ner(
                        hidden[0], word_ids, num_words, args.max_span_width,
                    )

                if len(candidates) > 0:
                    probs = torch.softmax(span_logits, dim=-1)
                    # Class 0 = NONE; entity = argmax > 0 with prob > threshold
                    pred_classes = probs.argmax(dim=-1)  # (N,)
                    pred_probs = probs.max(dim=-1).values  # (N,)

                    entity_word_set = set()
                    for idx, (s, e) in enumerate(candidates):
                        if pred_classes[idx].item() > 0 and pred_probs[idx].item() > args.threshold:
                            for w in range(s, e + 1):
                                entity_word_set.add(w)

                    # Map entity words back to token positions
                    for tok_idx, wid in enumerate(word_ids):
                        if wid is not None and wid in entity_word_set:
                            entity_token_mask[tok_idx] = 1

                    if entity_word_set:
                        n_with_entities += 1

            fout.write(json.dumps({
                "text": text,
                "entity_token_mask": entity_token_mask,
            }) + "\n")

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(texts)} docs, {n_with_entities} with entities")

    entity_rate = n_with_entities / len(texts) if texts else 0
    print(f"\nDone. {len(texts)} docs → {out_path}")
    print(f"  {n_with_entities}/{len(texts)} docs have predicted entities ({entity_rate:.1%})")


if __name__ == "__main__":
    main()
