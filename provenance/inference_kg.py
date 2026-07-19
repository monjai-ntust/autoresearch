"""
KG Inference Pipeline: Extract triples with confidence scores from text.

Runs the best encoder (span NER + BIO multi-task) on a dataset and outputs
per-triple predictions with confidence scores for downstream quality pipeline.

Output JSONL format per document:
{
  "doc_id": 0,
  "words": ["the", "model", ...],
  "sentence": "the model is used for ...",
  "predicted_entities": [
    {"span": [0, 2], "type": "Method", "confidence": 0.95}
  ],
  "predicted_triples": [
    {"head": [0, 2], "tail": [4, 5], "relation": "USED-FOR",
     "head_text": "the model", "tail_text": "classification",
     "ner_conf": 0.92, "re_conf": 0.87, "triple_conf": 0.80}
  ],
  "gold_entities": [...],  // if available
  "gold_triples": [...]    // if available
}

Usage:
    uv run python inference_kg.py --checkpoint checkpoints/best.pt --dataset scierc --split test
    uv run python inference_kg.py --checkpoint checkpoints/best.pt --input raw_text.jsonl
"""
import argparse
import inspect
import importlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.bert_kg_encoder import BertKGExtractor


DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "scier": "data.scier",
    "conll04": "data.conll04",
    "ade": "data.ade",
    "accord": "data.code_accord",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="scierc", choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--model-name", default=None)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-span-width", type=int, default=8)
    p.add_argument("--span-threshold", type=float, default=0.0,
                   help="Minimum NER confidence to keep a span. 0 = keep all predicted entities.")
    p.add_argument("--use-gold-entities", action="store_true",
                   help="Oracle diagnostic: run relation inference over gold entity spans instead of predicted spans.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for datasets with runtime train/dev splits.")
    p.add_argument("--doc-window-size", type=int, default=1,
                   help="For document-aware datasets, join N consecutive sentences before extraction.")
    p.add_argument("--doc-window-stride", type=int, default=1,
                   help="Stride for --doc-window-size sliding windows.")
    p.add_argument("--out-jsonl", default="results/kg_inference.jsonl")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])
    if args.model_name is None:
        args.model_name = {
            "scierc": "allenai/scibert_scivocab_uncased",
            "scier": "allenai/scibert_scivocab_uncased",
            "conll04": "bert-base-uncased",
            "ade": "allenai/scibert_scivocab_uncased",
            "accord": "allenai/scibert_scivocab_uncased",
        }.get(args.dataset, "bert-base-uncased")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dl_kwargs = dict(batch_size=1, max_length=args.max_length)
    dl_params = inspect.signature(ds_mod.build_dataloaders).parameters
    if "seed" in dl_params:
        dl_kwargs["seed"] = args.seed
    if "doc_window_size" in dl_params:
        dl_kwargs["doc_window_size"] = args.doc_window_size
    if "doc_window_stride" in dl_params:
        dl_kwargs["doc_window_stride"] = args.doc_window_stride
    train_loader, dev_loader, test_loader = ds_mod.build_dataloaders(
        tokenizer, **dl_kwargs,
    )
    loader = {"train": train_loader, "dev": dev_loader, "test": test_loader}[args.split]

    # Load model
    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=ds_mod.NUM_BIO_TAGS,
        num_relations=ds_mod.NUM_RELATIONS,
        num_entity_types=len(ds_mod.ENTITY_TYPES),
        use_span_ner=True,
        max_span_width=args.max_span_width,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "encoder" in ckpt:
        model.load_state_dict(ckpt["encoder"])
    elif "discriminator" in ckpt:
        model.load_state_dict(ckpt["discriminator"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"Dataset: {args.dataset} ({args.split}), {len(loader.dataset)} examples")

    NO_REL = ds_mod.NO_REL_ID
    id2entity = {i + 1: t for i, t in enumerate(ds_mod.ENTITY_TYPES)}
    id2rel = ds_mod.ID2REL

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_entities = 0
    total_triples = 0
    doc_id = 0

    with open(out_path, "w") as fout:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids_list = batch["word_ids"]
            num_words_list = batch["num_words"]
            words_list = batch.get("words", [None])
            gold_entities_list = batch.get("gold_entities", [None])
            gold_relations_list = batch.get("gold_relations", [None])
            doc_ids = batch.get("doc_ids", [None])
            example_ids = batch.get("example_ids", [None])

            with torch.no_grad():
                hidden = model.encode(
                    modality="text", input_ids=input_ids, attention_mask=attention_mask,
                )

            for b_idx in range(input_ids.size(0)):
                n_words = num_words_list[b_idx]
                words = words_list[b_idx] if words_list[b_idx] is not None else []
                source_doc_id = doc_ids[b_idx] if doc_ids[b_idx] else doc_id
                source_example_id = example_ids[b_idx] if example_ids[b_idx] else str(doc_id)

                with torch.no_grad():
                    span_logits, candidates = model.forward_span_ner(
                        hidden[b_idx], word_ids_list[b_idx], n_words, args.max_span_width,
                    )

                if not candidates:
                    doc_id += 1
                    continue

                # NER predictions with confidence
                span_probs = torch.softmax(span_logits, dim=-1)
                pred_types = span_logits.argmax(dim=-1).tolist()
                pred_confs = span_probs.max(dim=-1).values.tolist()

                # Filter and deduplicate
                pred_entities = []
                for (s, e), etype_id, conf in zip(candidates, pred_types, pred_confs):
                    if etype_id > 0 and conf >= args.span_threshold:
                        pred_entities.append({
                            "span": [s, e],
                            "type": id2entity.get(etype_id, "Unknown"),
                            "confidence": round(conf, 4),
                        })

                # Greedy non-overlapping (highest confidence first)
                pred_entities.sort(key=lambda x: -x["confidence"])
                taken = set()
                filtered_entities = []
                for ent in pred_entities:
                    s, e = ent["span"]
                    overlap = any(not (e < ts or te < s) for (ts, te) in taken)
                    if not overlap:
                        filtered_entities.append(ent)
                        taken.add((s, e))
                pred_entities = filtered_entities

                gold_ents = gold_entities_list[b_idx] if gold_entities_list[b_idx] is not None else []
                if args.use_gold_entities and gold_ents:
                    pred_entities = [
                        {
                            "span": [s, e],
                            "type": t,
                            "confidence": 1.0,
                        }
                        for (s, e, t) in gold_ents
                    ]

                # RE predictions with confidence
                pred_triples = []
                if len(pred_entities) >= 2:
                    spans = [(ent["span"][0], ent["span"][1]) for ent in pred_entities]
                    pairs = [(h, t) for h in spans for t in spans if h != t]
                    if pairs:
                        with torch.no_grad():
                            re_logits = model.forward_re(
                                hidden[b_idx], word_ids_list[b_idx], pairs,
                            )
                        re_probs = torch.softmax(re_logits, dim=-1)
                        re_preds = re_logits.argmax(dim=-1).tolist()
                        re_confs = re_probs.max(dim=-1).values.tolist()

                        for (h_span, t_span), rel_id, re_conf in zip(pairs, re_preds, re_confs):
                            if rel_id == NO_REL:
                                continue
                            # Find NER confidence for head and tail
                            h_conf = next(
                                (e["confidence"] for e in pred_entities
                                 if e["span"] == list(h_span)), 0.0,
                            )
                            t_conf = next(
                                (e["confidence"] for e in pred_entities
                                 if e["span"] == list(t_span)), 0.0,
                            )
                            ner_conf = min(h_conf, t_conf)
                            triple_conf = ner_conf * re_conf

                            h_text = " ".join(words[h_span[0]:h_span[1]+1]) if words else ""
                            t_text = " ".join(words[t_span[0]:t_span[1]+1]) if words else ""

                            pred_triples.append({
                                "head": list(h_span),
                                "tail": list(t_span),
                                "relation": id2rel.get(rel_id, f"REL_{rel_id}"),
                                "head_text": h_text,
                                "tail_text": t_text,
                                "ner_conf": round(ner_conf, 4),
                                "re_conf": round(re_conf, 4),
                                "triple_conf": round(triple_conf, 4),
                            })

                # Gold data (if available)
                gold_rels = gold_relations_list[b_idx] if gold_relations_list[b_idx] is not None else []

                gold_ents_out = [
                    {"span": [s, e], "type": t} for (s, e, t) in gold_ents
                ] if gold_ents else []

                gold_triples_out = []
                for (h_span, t_span, rel_id) in gold_rels:
                    if rel_id == NO_REL:
                        continue
                    h_text = " ".join(words[h_span[0]:h_span[1]+1]) if words else ""
                    t_text = " ".join(words[t_span[0]:t_span[1]+1]) if words else ""
                    gold_triples_out.append({
                        "head": list(h_span),
                        "tail": list(t_span),
                        "relation": id2rel.get(rel_id, f"REL_{rel_id}"),
                        "head_text": h_text,
                        "tail_text": t_text,
                    })

                record = {
                    "doc_id": source_doc_id,
                    "example_id": source_example_id,
                    "words": words,
                    "sentence": " ".join(words) if words else "",
                    "predicted_entities": pred_entities,
                    "predicted_triples": pred_triples,
                    "gold_entities": gold_ents_out,
                    "gold_triples": gold_triples_out,
                }
                fout.write(json.dumps(record) + "\n")

                total_entities += len(pred_entities)
                total_triples += len(pred_triples)
                doc_id += 1

    # Summary statistics
    print(f"\n=== Inference Complete ===")
    print(f"  Documents: {doc_id}")
    print(f"  Predicted entities: {total_entities}")
    print(f"  Predicted triples: {total_triples}")
    print(f"  Output: {out_path}")

    # Confidence distribution analysis
    all_confs = []
    with open(out_path) as f:
        for line in f:
            rec = json.loads(line)
            for t in rec["predicted_triples"]:
                all_confs.append(t["triple_conf"])

    if all_confs:
        all_confs.sort()
        n = len(all_confs)
        print(f"\n  Triple confidence distribution:")
        print(f"    min={all_confs[0]:.3f}, max={all_confs[-1]:.3f}")
        print(f"    p25={all_confs[n//4]:.3f}, p50={all_confs[n//2]:.3f}, p75={all_confs[3*n//4]:.3f}")

        # Precision at different thresholds (if gold data available)
        with open(out_path) as f:
            records = [json.loads(line) for line in f]

        has_gold = any(r["gold_triples"] for r in records)
        if has_gold:
            print(f"\n  Precision/Recall at confidence thresholds:")
            for thresh in [0.0, 0.3, 0.5, 0.7, 0.9]:
                tp = fp = fn = 0
                for rec in records:
                    pred_set = {
                        (tuple(t["head"]), tuple(t["tail"]), t["relation"])
                        for t in rec["predicted_triples"]
                        if t["triple_conf"] >= thresh
                    }
                    gold_set = {
                        (tuple(t["head"]), tuple(t["tail"]), t["relation"])
                        for t in rec["gold_triples"]
                    }
                    tp += len(pred_set & gold_set)
                    fp += len(pred_set - gold_set)
                    fn += len(gold_set - pred_set)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                print(f"    thresh={thresh:.1f}: P={p:.3f} R={r:.3f} F1={f1:.3f} (kept={tp+fp})")


if __name__ == "__main__":
    main()
