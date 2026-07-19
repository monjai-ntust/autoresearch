"""
Stage 2e: Encoder training with gold + LoRA-synth data augmentation.

This is the thesis experiment. The question:
    Does training the encoder on a mix of gold SciERC sentences and
    LoRA-generated paraphrases improve Triple F1 beyond training on
    gold-only?

Protocol:
    1. Fresh SciBERT encoder (no stage2b checkpoint — clean comparison).
    2. Train for --max-steps steps with curriculum:
       - Phase 1 (steps 0..G): gold-only (warmup, same as stage2-006).
       - Phase 2 (steps G..T): alternating gold + synth batches.
    3. Evaluate dev Triple F1 every --eval-every steps.

Must run a **gold-only baseline** with the same config (--synth-jsonl "")
to compare. The Δ Triple F1 is Stage 2e's signal.

Usage:
    # Stage 2e experiment (gold + synth)
    uv run python train_stage2e.py --synth-jsonl data/stage2e_synth_v8.jsonl \
        --max-steps 1500 --gold-only-steps 250

    # Baseline (gold only, same --max-steps / --seed)
    uv run python train_stage2e.py --synth-jsonl "" --max-steps 1500
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.scierc import build_dataloaders
from models.bert_kg_encoder import BertKGExtractor, compute_loss
from eval.triple_f1 import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--synth-jsonl", default="",
                   help="Path to synth jsonl from generate_synth_dataset.py. "
                        "Empty string = gold-only baseline.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--gold-only-steps", type=int, default=250,
                   help="Train on gold-only for this many steps before "
                        "mixing in synth. Ignored if --synth-jsonl is empty.")
    p.add_argument("--re-weight", type=float, default=1.0)
    p.add_argument("--synth-weight", type=float, default=0.2,
                   help="Weight on synth loss. total = gold + w*synth. "
                        "Default 0.2 (gold-dominant).")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-containment", type=float, default=1.0,
                   help="Load-time filter on synth containment score.")
    p.add_argument("--use-crf", action="store_true",
                   help="Add CRF layer on NER head (stage2-012). Negative result — not recommended.")
    p.add_argument("--adv-epsilon", type=float, default=0.0,
                   help="FGM adversarial perturbation epsilon on word embeddings. "
                        "0 = disabled. Recommended: 0.5-1.0 (READ-style).")
    p.add_argument("--save-best-to", default=None)
    return p.parse_args()


def cycle(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_synth = bool(args.synth_jsonl)

    mode = "gold+synth" if use_synth else "gold-only (baseline)"
    print(f"=== Stage 2e — Encoder data augmentation ({mode}) ===")
    print(f"  encoder:  {args.model_name}")
    print(f"  synth:    {args.synth_jsonl or '(none — baseline)'}")
    print(f"  steps:    {args.max_steps}")
    print(f"  lr:       {args.lr}")
    print(f"  warmup:   {args.warmup_steps}")
    if use_synth:
        print(f"  gold-only phase: {args.gold_only_steps} steps")

    # ── Tokenizer + data ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, dev_loader, _ = build_dataloaders(
        tokenizer, data_dir=data_dir,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"  sci train: {len(train_loader.dataset)} sentences")
    print(f"  sci dev:   {len(dev_loader.dataset)} sentences")

    synth_loader = None
    if use_synth:
        from data.synth_loader import build_synth_loader
        synth_loader = build_synth_loader(
            tokenizer, args.synth_jsonl,
            batch_size=args.batch_size, max_length=args.max_length,
            min_containment=args.min_containment,
        )
        print(f"  synth:     {len(synth_loader.dataset)} sentences")

    # ── Model ───────────────────────────────────────────────────
    model = BertKGExtractor(args.model_name, use_crf=args.use_crf).to(device)
    if args.use_crf:
        print(f"  CRF:       enabled ({sum(p.numel() for p in model.crf.parameters())} params)")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    gold_iter = cycle(train_loader)
    synth_iter = cycle(synth_loader) if synth_loader else None
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    model.train()
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        optimizer.zero_grad()

        # ── Gold batch (always) ─────────────────────────────────
        gold_batch = next(gold_iter)
        gold_loss, ner_loss, re_loss, _ = compute_loss(
            model, gold_batch, device, re_weight=args.re_weight,
        )

        # ── Synth batch (after gold-only phase) ────────────────
        synth_loss_val = 0.0
        if use_synth and step >= args.gold_only_steps and synth_iter:
            synth_batch = next(synth_iter)
            try:
                s_loss, _, _, _ = compute_loss(
                    model, synth_batch, device, re_weight=args.re_weight,
                )
                synth_loss_val = s_loss.item()
                total = gold_loss + args.synth_weight * s_loss
            except Exception:
                total = gold_loss
        else:
            total = gold_loss

        total.backward()

        # ── FGM adversarial perturbation (READ-style) ──────────
        if args.adv_epsilon > 0:
            # Collect word embedding params and their gradients
            emb = model.backbone.bert.embeddings.word_embeddings.weight
            if emb.grad is not None:
                # FGM: perturbation = epsilon * grad / ||grad||
                norm = emb.grad.norm()
                if norm > 0 and not torch.isnan(norm):
                    r_adv = args.adv_epsilon * emb.grad / norm
                    emb.data.add_(r_adv)
                    # Re-forward on gold batch with perturbed embeddings
                    adv_loss, _, _, _ = compute_loss(
                        model, gold_batch, device, re_weight=args.re_weight,
                    )
                    adv_loss.backward()
                    # Restore original embeddings
                    emb.data.sub_(r_adv)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            phase = "gold+synth" if (use_synth and step >= args.gold_only_steps) else "gold"
            print(
                f"[Step {step:04d} {phase}] "
                f"L_gold={gold_loss.item():.4f} L_synth={synth_loss_val:.4f} "
                f"L_ner={ner_loss.item():.4f} L_re={re_loss.item():.4f} "
                f"lr={cur_lr:.2e} | {dt:.1f}ms/step"
            )

        if step > 0 and step % args.eval_every == 0:
            model.eval()
            metrics = evaluate(model, dev_loader, device)
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " ★ NEW BEST"
                if args.save_best_to:
                    save_path = Path(args.save_best_to)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "encoder": model.state_dict(),
                        "step": step,
                        "metrics": metrics,
                        "args": vars(args),
                    }, save_path)
                    star += f" (ckpt→{save_path})"
            print(
                f"[Eval @ step {step}] "
                f"NER F1={metrics['ner_f1']:.4f} | "
                f"RE F1={metrics['re_f1']:.4f} | "
                f"Triple F1={metrics['triple_f1']:.4f} | "
                f"({metrics['n_gold_triples']} gold triples){star}"
            )
            model.train()

        step += 1

    # Final eval
    model.eval()
    metrics = evaluate(model, dev_loader, device)
    if metrics["triple_f1"] > best_metrics["triple_f1"]:
        best_metrics = dict(metrics)
        best_step = step
    print(f"\n=== FINAL (step {step}) ===")
    print(f"  NER F1    = {metrics['ner_f1']:.4f}")
    print(f"  RE F1     = {metrics['re_f1']:.4f}")
    print(f"  Triple F1 = {metrics['triple_f1']:.4f}")
    print(f"=== BEST DEV (step {best_step}, by Triple F1) ===")
    print(f"  NER F1    = {best_metrics['ner_f1']:.4f}")
    print(f"  RE F1     = {best_metrics['re_f1']:.4f}")
    print(f"  Triple F1 = {best_metrics['triple_f1']:.4f}")
    print(f"  Total time = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
