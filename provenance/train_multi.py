"""
Multi-dataset training script for Stage 2 pipeline validation.

Supports SciERC, CoNLL04, and ADE with the same encoder architecture.
Dynamically configures NER/RE head dimensions based on the selected dataset.

Usage:
    # SciERC baseline (same as train_stage2e.py --synth-jsonl '')
    uv run python train_multi.py --dataset scierc

    # CoNLL04 baseline
    uv run python train_multi.py --dataset conll04

    # CoNLL04 with CAST pseudo-labels
    uv run python train_multi.py --dataset conll04 --synth-jsonl data/conll04_pseudo.jsonl
"""
import argparse
import importlib
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.bert_kg_encoder import BertKGExtractor


# Dataset registry: maps name → module with build_dataloaders, NUM_BIO_TAGS, NUM_RELATIONS, etc.
DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "conll04": "data.conll04",
    "ade": "data.ade",
}


def load_dataset_module(name):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return importlib.import_module(DATASET_REGISTRY[name])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scierc", choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--model-name", default=None,
                   help="Encoder backbone. Default: scibert for scierc, bert-base for conll04.")
    p.add_argument("--synth-jsonl", default="",
                   help="Pseudo-label jsonl for augmentation. Empty = gold-only.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--gold-only-steps", type=int, default=250)
    p.add_argument("--synth-weight", type=float, default=0.3)
    p.add_argument("--min-containment", type=float, default=1.0)
    p.add_argument("--re-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best-to", default=None)
    return p.parse_args()


def compute_loss(model, batch, device, ds_mod, re_weight=1.0):
    """Dataset-agnostic compute_loss. Uses the dataset module's NO_REL_ID."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    ner_labels = batch["ner_labels"].to(device)
    word_ids_list = batch["word_ids"]
    gold_entities_list = batch["gold_entities"]
    gold_relations_list = batch["gold_relations"]

    hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)
    ner_logits = model.forward_ner(hidden)
    ner_loss = F.cross_entropy(
        ner_logits.view(-1, ner_logits.size(-1)),
        ner_labels.view(-1),
        ignore_index=-100,
    )

    NO_REL = ds_mod.NO_REL_ID
    re_losses = []
    for b_idx in range(len(gold_entities_list)):
        ents = gold_entities_list[b_idx]
        rels = gold_relations_list[b_idx]
        if len(ents) < 2:
            continue
        rel_lookup = {(h, t): rid for (h, t, rid) in rels}
        spans = [(s, e) for (s, e, _) in ents]
        pairs = []
        targets = []
        for h in spans:
            for t in spans:
                if h == t:
                    continue
                pairs.append((h, t))
                targets.append(rel_lookup.get((h, t), NO_REL))
        if not pairs:
            continue
        targets_t = torch.tensor(targets, device=device, dtype=torch.long)
        re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pairs)
        re_losses.append(F.cross_entropy(re_logits, targets_t))

    if re_losses:
        re_loss = torch.stack(re_losses).mean()
    else:
        re_loss = ner_loss.new_tensor(0.0)

    total = ner_loss + re_weight * re_loss
    return total, ner_loss.detach(), re_loss.detach(), ner_logits


def evaluate(model, dataloader, device, ds_mod):
    """Dataset-agnostic evaluate. Uses ds_mod for NO_REL_ID and ID2BIO."""
    from eval.triple_f1 import _bio_to_spans, _word_level_bio_from_token_logits, _prf
    model.eval()
    NO_REL = ds_mod.NO_REL_ID

    ner_tp = ner_fp = ner_fn = 0
    re_tp = re_fp = re_fn = 0
    triple_tp = triple_fp = triple_fn = 0
    n_examples = n_gold_entities = n_gold_triples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_ids_list = batch["word_ids"]
        gold_entities_list = batch["gold_entities"]
        gold_relations_list = batch["gold_relations"]

        hidden = model.encode(modality="text", input_ids=input_ids, attention_mask=attention_mask)
        ner_logits = model.forward_ner(hidden)

        for b_idx in range(input_ids.size(0)):
            n_examples += 1
            gold_ents = gold_entities_list[b_idx]
            gold_rels = gold_relations_list[b_idx]
            n_gold_entities += len(gold_ents)
            n_gold_triples += len(gold_rels)

            # NER — use the dataset's ID2BIO for decoding
            word_bio = _word_level_bio_from_token_logits(
                ner_logits[b_idx], word_ids_list[b_idx],
            )
            pred_spans = _bio_to_spans(word_bio)
            pred_ent_set = {(s, e, t) for (s, e, t) in pred_spans}
            gold_ent_set = {(s, e, t) for (s, e, t) in gold_ents}
            ner_tp += len(pred_ent_set & gold_ent_set)
            ner_fp += len(pred_ent_set - gold_ent_set)
            ner_fn += len(gold_ent_set - pred_ent_set)

            # RE on gold spans
            gold_span_list = [(s, e) for (s, e, _) in gold_ents]
            gold_pairs = [(a, b) for a in gold_span_list for b in gold_span_list if a != b]
            if gold_pairs:
                re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], gold_pairs)
                re_pred = re_logits.argmax(dim=-1).tolist()
                pred_rel = {(h, t, p) for (h, t), p in zip(gold_pairs, re_pred) if p != NO_REL}
                gold_rel = {(h, t, r) for (h, t, r) in gold_rels}
                re_tp += len(pred_rel & gold_rel)
                re_fp += len(pred_rel - gold_rel)
                re_fn += len(gold_rel - pred_rel)

            # Triple F1 (full pipeline)
            pred_span_list = [(s, e) for (s, e, _) in pred_spans]
            pred_pairs = [(a, b) for a in pred_span_list for b in pred_span_list if a != b]
            if pred_pairs:
                pred_re_logits = model.forward_re(hidden[b_idx], word_ids_list[b_idx], pred_pairs)
                pred_re_ids = pred_re_logits.argmax(dim=-1).tolist()
                pred_full = {(h, t, p) for (h, t), p in zip(pred_pairs, pred_re_ids) if p != NO_REL}
            else:
                pred_full = set()
            gold_full = {(h, t, r) for (h, t, r) in gold_rels}
            triple_tp += len(pred_full & gold_full)
            triple_fp += len(pred_full - gold_full)
            triple_fn += len(gold_full - pred_full)

    _, _, nf = _prf(ner_tp, ner_fp, ner_fn)
    _, _, rf = _prf(re_tp, re_fp, re_fn)
    _, _, tf = _prf(triple_tp, triple_fp, triple_fn)
    return {
        "ner_f1": nf, "re_f1": rf, "triple_f1": tf,
        "n_examples": n_examples, "n_gold_entities": n_gold_entities,
        "n_gold_triples": n_gold_triples,
    }


def cycle(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ds_mod = load_dataset_module(args.dataset)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_synth = bool(args.synth_jsonl)

    # Default model per dataset
    if args.model_name is None:
        args.model_name = {
            "scierc": "allenai/scibert_scivocab_uncased",
            "conll04": "bert-base-uncased",
            "ade": "allenai/scibert_scivocab_uncased",
        }.get(args.dataset, "bert-base-uncased")

    mode = f"{args.dataset} gold+synth" if use_synth else f"{args.dataset} gold-only"
    print(f"=== Multi-dataset training ({mode}) ===")
    print(f"  dataset:  {args.dataset}")
    print(f"  encoder:  {args.model_name}")
    print(f"  NER tags: {ds_mod.NUM_BIO_TAGS}  REL types: {ds_mod.NUM_RELATIONS}")
    print(f"  steps:    {args.max_steps}  lr: {args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, dev_loader, test_loader = ds_mod.build_dataloaders(
        tokenizer, data_dir=data_dir,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"  train: {len(train_loader.dataset)} | dev: {len(dev_loader.dataset)}"
          f" | test: {len(test_loader.dataset)}")

    synth_loader = None
    if use_synth:
        from data.synth_loader import build_synth_loader
        synth_loader = build_synth_loader(
            tokenizer, args.synth_jsonl,
            batch_size=args.batch_size, max_length=args.max_length,
            min_containment=args.min_containment,
        )
        print(f"  synth: {len(synth_loader.dataset)}")

    # Build model with dataset-specific head dimensions via constructor args.
    # Also patch data.scierc dicts in-place so that triple_f1's decode
    # functions (which import ID2BIO at module level) use the right tags.
    import data.scierc as scierc_mod
    scierc_mod.NO_REL_ID = ds_mod.NO_REL_ID
    scierc_mod.ID2BIO.clear()
    scierc_mod.ID2BIO.update(ds_mod.ID2BIO)
    scierc_mod.BIO_TAG2ID.clear()
    scierc_mod.BIO_TAG2ID.update(ds_mod.BIO_TAG2ID)
    model = BertKGExtractor(
        args.model_name,
        num_bio_tags=ds_mod.NUM_BIO_TAGS,
        num_relations=ds_mod.NUM_RELATIONS,
    ).to(device)

    print(f"  NER head: {model.ner_head.out_features}  RE head: {model.re_head[-1].out_features}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps,
    )

    # Monkey-patch eval's ID2BIO for the selected dataset
    import eval.triple_f1 as eval_mod
    orig_id2bio = eval_mod.ID2BIO
    eval_mod.ID2BIO = ds_mod.ID2BIO

    gold_iter = cycle(train_loader)
    synth_iter = cycle(synth_loader) if synth_loader else None
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    model.train()
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        optimizer.zero_grad()
        gold_batch = next(gold_iter)
        gold_loss, ner_loss, re_loss, _ = compute_loss(
            model, gold_batch, device, ds_mod, re_weight=args.re_weight,
        )

        synth_loss_val = 0.0
        if use_synth and step >= args.gold_only_steps and synth_iter:
            synth_batch = next(synth_iter)
            try:
                s_loss, _, _, _ = compute_loss(
                    model, synth_batch, device, ds_mod, re_weight=args.re_weight,
                )
                synth_loss_val = s_loss.item()
                total = gold_loss + args.synth_weight * s_loss
            except Exception:
                total = gold_loss
        else:
            total = gold_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {step:04d}] L={gold_loss.item():.4f} "
                f"NER={ner_loss.item():.4f} RE={re_loss.item():.4f} "
                f"synth={synth_loss_val:.4f} lr={cur_lr:.2e} | {dt:.0f}ms/step"
            )

        if step > 0 and step % args.eval_every == 0:
            metrics = evaluate(model, dev_loader, device, ds_mod)
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " *"
                if args.save_best_to:
                    save_path = Path(args.save_best_to)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"encoder": model.state_dict(), "step": step,
                                "metrics": metrics}, save_path)
            print(
                f"[Eval @ {step}] NER={metrics['ner_f1']:.4f} RE={metrics['re_f1']:.4f} "
                f"Triple={metrics['triple_f1']:.4f}{star}"
            )
            model.train()

        step += 1

    # Final eval on dev + test
    for split_name, loader in [("dev", dev_loader), ("test", test_loader)]:
        metrics = evaluate(model, loader, device, ds_mod)
        if split_name == "dev" and metrics["triple_f1"] > best_metrics["triple_f1"]:
            best_metrics = dict(metrics)
            best_step = step
        print(f"\n=== {split_name.upper()} (step {step}) ===")
        print(f"  NER={metrics['ner_f1']:.4f} RE={metrics['re_f1']:.4f} Triple={metrics['triple_f1']:.4f}")
    print(f"=== BEST DEV (step {best_step}) ===")
    print(f"  NER={best_metrics['ner_f1']:.4f} RE={best_metrics['re_f1']:.4f} "
          f"Triple={best_metrics['triple_f1']:.4f}")
    print(f"  time={time.time()-t0:.1f}s")

    eval_mod.ID2BIO = orig_id2bio


if __name__ == "__main__":
    main()
