"""
Stage 2b training: Encoder + RealismCritic + frozen Decoder D (Qwen-0.5B).

The first experiment in this project where the adversarial loop is alive.

Loop structure:
    Phase 1 (warmup): supervised only — same as stage2-006.
    Phase 2 (adversarial): supervised + critic loss every step.

In Phase 2 each step:
    1. Forward sci batch through encoder → NER+RE supervised loss
    2. Forward arXiv batch through encoder → real CLS hidden
    3. Sample triples from sci batch's gold relations
    4. Decoder D paraphrases triples → list of synth sentences
    5. Tokenize synth → forward through encoder → fake CLS hidden
    6. critic_loss = BCE(real, 1) + BCE(fake, 0)
    7. total = sup_loss + lambda_real * critic_loss
    8. Backprop, optimizer step

The encoder body is shared between supervised and critic forward passes,
so gradient from critic_loss flows into the BERT body. The critic head
itself has its own params (small MLP). Decoder D is frozen (no_grad).

Run:
    bash scripts/run_train_stage2b.sh
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.scierc import build_dataloaders, NUM_BIO_TAGS, NUM_RELATIONS, ID2REL, NO_REL_ID
from data.arxiv_real import build_arxiv_loader, cycle
from models.bert_kg_encoder import BertKGExtractor, compute_loss
from models.critic import RealismCritic, critic_loss
from models.decoder_d import FrozenQwenDecoder
from eval.triple_f1 import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=250,
                   help="Supervised-only warmup before adversarial phase begins.")
    p.add_argument("--adversarial-start", type=int, default=None,
                   help="Step at which adversarial loss kicks in. "
                        "Defaults to --warmup-steps. Set higher to give the "
                        "supervised baseline more head start.")
    p.add_argument("--lambda-real", type=float, default=0.3,
                   help="Weight on critic loss in the total objective.")
    p.add_argument("--re-weight", type=float, default=1.0)
    p.add_argument("--synth-batch-size", type=int, default=16,
                   help="Number of synth sentences per training step (Q4 = 16 = full coverage).")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--arxiv-jsonl", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best-to", default=None,
                   help="If set, save encoder+critic weights to this path "
                        "whenever dev Triple F1 is a new best. Used by Stage "
                        "2c to get a frozen recovery encoder.")
    return p.parse_args()


def sample_triples_from_batch(batch, n: int):
    """
    Pool all gold relations across the sci batch and sample `n` triples
    with replacement, returning [(head_str, rel_str, tail_str), ...].

    If the batch has zero gold relations (very rare), returns [].
    """
    pool = []
    words_per_sentence = batch["words"]
    rels_per_sentence = batch["gold_relations"]
    for sent_idx, rels in enumerate(rels_per_sentence):
        words = words_per_sentence[sent_idx]
        for (h_span, t_span, rel_id) in rels:
            if rel_id == NO_REL_ID:
                continue
            hs, he = h_span
            ts, te = t_span
            head_str = " ".join(words[hs:he + 1])
            tail_str = " ".join(words[ts:te + 1])
            rel_str = ID2REL[rel_id]
            pool.append((head_str, rel_str, tail_str))
    if not pool:
        return []
    if len(pool) >= n:
        return random.sample(pool, n)
    # If pool < n, sample with replacement
    return [random.choice(pool) for _ in range(n)]


def tokenize_synth(synth_sentences, tokenizer, max_length: int, device):
    """Tokenize a list of synth sentences into a padded BERT batch on device."""
    enc = tokenizer(
        synth_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.adversarial_start is None:
        args.adversarial_start = args.warmup_steps

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Stage 2b — Encoder + Critic + frozen Qwen Decoder ===")
    print(f"  encoder:  {args.model_name}")
    print(f"  decoder:  {args.decoder_name}")
    print(f"  device:   {device}")
    print(f"  bs:       {args.batch_size}")
    print(f"  lr:       {args.lr}")
    print(f"  steps:    {args.max_steps}")
    print(f"  warmup:   {args.warmup_steps} (adversarial starts at step {args.adversarial_start})")
    print(f"  λ_real:   {args.lambda_real}")
    print(f"  synth bs: {args.synth_batch_size}")

    # ── Tokenizer + data ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, dev_loader, _ = build_dataloaders(
        tokenizer, data_dir=data_dir,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    arxiv_loader = build_arxiv_loader(
        tokenizer,
        batch_size=args.synth_batch_size,
        max_length=args.max_length,
        jsonl_path=args.arxiv_jsonl,
    )
    print(f"  sci train sentences:  {len(train_loader.dataset)}")
    print(f"  sci dev   sentences:  {len(dev_loader.dataset)}")
    print(f"  arxiv real corpus:    {len(arxiv_loader.dataset)} documents")

    # ── Model + critic + decoder ─────────────────────────────────────
    model = BertKGExtractor(args.model_name).to(device)
    hidden = model.backbone.hidden_size
    critic = RealismCritic(hidden).to(device)
    decoder = FrozenQwenDecoder(args.decoder_name, device=device)

    # Critic + encoder share params via the encoder body. AdamW gets all
    # trainable params from both modules.
    trainable_params = list(model.parameters()) + list(critic.parameters())
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    arxiv_iter = cycle(arxiv_loader)
    sci_iter = iter(train_loader)
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    model.train()
    critic.train()

    t0 = time.time()
    step = 0
    while step < args.max_steps:
        try:
            sci_batch = next(sci_iter)
        except StopIteration:
            sci_iter = iter(train_loader)
            sci_batch = next(sci_iter)

        optimizer.zero_grad()

        # ── Phase 1: supervised loss (always on) ─────────────────────
        sup_loss, ner_loss, re_loss, _ = compute_loss(
            model, sci_batch, device, re_weight=args.re_weight,
        )

        # ── Phase 2: adversarial critic loss (after warmup) ──────────
        crit_loss_val = torch.tensor(0.0, device=device)
        real_score_mean = float("nan")
        fake_score_mean = float("nan")

        if step >= args.adversarial_start:
            # 1. Real arXiv text → encoder → CLS → critic
            real_batch = next(arxiv_iter)
            real_input_ids = real_batch["input_ids"].to(device)
            real_attn = real_batch["attention_mask"].to(device)
            real_hidden = model.encode(
                modality="text",
                input_ids=real_input_ids,
                attention_mask=real_attn,
            )
            real_cls = real_hidden[:, 0, :]
            real_logits = critic(real_cls)

            # 2. Sample triples → Decoder D → synth text → encoder → CLS → critic
            triples = sample_triples_from_batch(sci_batch, n=args.synth_batch_size)
            if triples:
                synth_sentences = decoder.generate_batch(triples)
                synth_enc = tokenize_synth(synth_sentences, tokenizer, args.max_length, device)
                synth_hidden = model.encode(
                    modality="text",
                    input_ids=synth_enc["input_ids"],
                    attention_mask=synth_enc["attention_mask"],
                )
                synth_cls = synth_hidden[:, 0, :]
                fake_logits = critic(synth_cls)
                crit_loss_val = critic_loss(real_logits, fake_logits)
                real_score_mean = real_logits.mean().item()
                fake_score_mean = fake_logits.mean().item()

        total = sup_loss + args.lambda_real * crit_loss_val
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            phase = "adv" if step >= args.adversarial_start else "sup"
            crit_str = (
                f" L_crit={crit_loss_val.item():.4f} real={real_score_mean:+.2f} fake={fake_score_mean:+.2f}"
                if step >= args.adversarial_start else ""
            )
            print(
                f"[Step {step:04d} {phase}] L_sup={sup_loss.item():.4f} "
                f"L_ner={ner_loss.item():.4f} L_re={re_loss.item():.4f}{crit_str} "
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
                        "critic": critic.state_dict(),
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
                f"({metrics['n_gold_triples']} gold triples on {metrics['n_examples']} sentences){star}"
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
    print(f"  NER F1     = {metrics['ner_f1']:.4f}")
    print(f"  RE F1      = {metrics['re_f1']:.4f}")
    print(f"  Triple F1  = {metrics['triple_f1']:.4f}")
    print(f"=== BEST DEV (step {best_step}, by Triple F1) ===")
    print(f"  NER F1     = {best_metrics['ner_f1']:.4f}")
    print(f"  RE F1      = {best_metrics['re_f1']:.4f}")
    print(f"  Triple F1  = {best_metrics['triple_f1']:.4f}")
    print(f"  Total time = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
