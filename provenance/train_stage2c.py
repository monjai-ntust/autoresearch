"""
Stage 2c training: LoRA-tuned Decoder D + REINFORCE against (critic,
triple_recovery, KL) reward.

See reports/stage2/stage2_008_DESIGN.md for the full design rationale.
Short version:

    Per step:
      1. Forward sci batch through encoder → sup loss (NER+RE), optional.
      2. Sample N triples from sci batch.
      3. LoRA-D samples one sentence per triple (with per-token logprobs).
      4. Each sentence → encoder → CLS → critic logit       (α term)
         Each sentence → frozen recovery encoder → triple    (β term)
         Per-sentence KL(LoRA‖base)                          (γ term, penalty)
      5. reward_i = α·critic_i + β·recovery_i − γ·kl_i
      6. EMA baseline b; REINFORCE loss = −mean((reward−b) · lora_logprob)
      7. total = L_sup + L_critic + L_reinforce
      8. Backprop: grads into encoder, critic, LoRA. Base Qwen frozen.

By default (design Q4) the encoder and critic are ALSO frozen at a
stage2-007 checkpoint — only LoRA trains. Use --train-encoder to enable
joint mode.
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.scierc import build_dataloaders, ID2REL, NO_REL_ID
from data.arxiv_real import build_arxiv_loader, cycle
from models.bert_kg_encoder import BertKGExtractor, compute_loss
from models.critic import RealismCritic, critic_loss
from models.decoder_d_lora import LoRAQwenDecoder
from models import triple_recovery as tr_mod
from eval.triple_f1 import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--stage2b-ckpt", default="checkpoints/stage2_007_best.pt",
                   help="Path to stage2-007 encoder+critic checkpoint. Used as "
                        "(a) init for encoder/critic, (b) frozen triple recovery "
                        "encoder.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--synth-batch-size", type=int, default=16)
    # Reward weights (design defaults)
    p.add_argument("--alpha", type=float, default=1.0, help="critic reward weight")
    p.add_argument("--beta", type=float, default=2.0, help="triple recovery reward weight")
    p.add_argument("--gamma", type=float, default=0.1, help="KL penalty weight")
    p.add_argument("--ema-decay", type=float, default=0.9, help="baseline EMA")
    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Who trains
    p.add_argument("--train-encoder", action="store_true",
                   help="If set, continue training encoder+critic (joint mode). "
                        "Default (Q4 A) freezes them at stage2b ckpt.")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--arxiv-jsonl", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-adapters-to", default="checkpoints/stage2_008_lora")
    return p.parse_args()


def sample_triples_from_batch(batch, n: int):
    """Same logic as train_stage2b.py:sample_triples_from_batch."""
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
    return [random.choice(pool) for _ in range(n)]


def tokenize_synth(synth_sentences, tokenizer, max_length: int, device):
    enc = tokenizer(
        synth_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def load_stage2b_ckpt(path, model, critic):
    """Load encoder+critic weights from stage2-007 checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["encoder"])
    critic.load_state_dict(ckpt["critic"])
    print(f"[stage2c] loaded stage2b ckpt step={ckpt['step']} "
          f"triple_f1={ckpt['metrics']['triple_f1']:.4f}")
    return ckpt


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Stage 2c — LoRA Decoder + REINFORCE ===")
    print(f"  encoder:          {args.model_name}")
    print(f"  decoder:          {args.decoder_name}")
    print(f"  stage2b ckpt:     {args.stage2b_ckpt}")
    print(f"  device:           {device}")
    print(f"  steps:            {args.max_steps}")
    print(f"  LoRA:             r={args.lora_r} α={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"  reward weights:   α={args.alpha}  β={args.beta}  γ={args.gamma}")
    print(f"  train encoder:    {args.train_encoder}")

    # ── Tokenizer + data ────────────────────────────────────────────
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
    print(f"  sci train: {len(train_loader.dataset)} | sci dev: {len(dev_loader.dataset)} "
          f"| arxiv: {len(arxiv_loader.dataset)}")

    # ── Encoder + critic (trainable or frozen) ──────────────────────
    model = BertKGExtractor(args.model_name).to(device)
    hidden = model.backbone.hidden_size
    critic = RealismCritic(hidden).to(device)
    load_stage2b_ckpt(args.stage2b_ckpt, model, critic)

    # Frozen recovery encoder — a second copy of the same stage2b ckpt,
    # used only by models.triple_recovery. Always frozen regardless of --train-encoder.
    recovery_encoder = BertKGExtractor(args.model_name).to(device)
    rec_critic_unused = RealismCritic(hidden).to(device)   # only so load_state_dict works
    load_stage2b_ckpt(args.stage2b_ckpt, recovery_encoder, rec_critic_unused)
    recovery_encoder.eval()
    for p in recovery_encoder.parameters():
        p.requires_grad = False

    if not args.train_encoder:
        model.eval()
        critic.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in critic.parameters():
            p.requires_grad = False

    # ── LoRA decoder ────────────────────────────────────────────────
    decoder = LoRAQwenDecoder(
        args.decoder_name, device=device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── Optimizer ──────────────────────────────────────────────────
    lora_params = [p for p in decoder.model.parameters() if p.requires_grad]
    param_groups = [{"params": lora_params, "lr": args.lora_lr}]
    if args.train_encoder:
        param_groups.append({"params": list(model.parameters()) + list(critic.parameters()),
                             "lr": args.encoder_lr})
    optimizer = AdamW(param_groups, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    arxiv_iter = cycle(arxiv_loader)
    sci_iter = iter(train_loader)
    best_metrics = {"triple_f1": -1.0}
    best_step = -1
    ema_baseline = 0.0

    t0 = time.time()
    step = 0
    while step < args.max_steps:
        try:
            sci_batch = next(sci_iter)
        except StopIteration:
            sci_iter = iter(train_loader)
            sci_batch = next(sci_iter)

        optimizer.zero_grad()

        total = torch.tensor(0.0, device=device)

        # ── Supervised (optional, only if encoder is training) ─────
        sup_loss_val = 0.0
        if args.train_encoder:
            sup_loss, _, _, _ = compute_loss(model, sci_batch, device)
            total = total + sup_loss
            sup_loss_val = sup_loss.item()

        # ── Critic step (optional) ─────────────────────────────────
        crit_val = 0.0
        if args.train_encoder:
            real_batch = next(arxiv_iter)
            real_hidden = model.encode(
                modality="text",
                input_ids=real_batch["input_ids"].to(device),
                attention_mask=real_batch["attention_mask"].to(device),
            )
            real_cls = real_hidden[:, 0, :]
            real_logits = critic(real_cls)

        # ── REINFORCE on LoRA decoder ───────────────────────────────
        triples = sample_triples_from_batch(sci_batch, n=args.synth_batch_size)
        reinforce_val = 0.0
        reward_val = 0.0
        critic_reward_val = 0.0
        recovery_reward_val = 0.0
        kl_val = 0.0

        if triples:
            sampled = decoder.sample_with_logprob(triples)
            synth_sentences = sampled["sentences"]
            lora_lp = sampled["lora_logprob"]         # (B,) with grad
            kl = sampled["kl"]                         # (B,) with grad

            # Critic reward: run synth through encoder → CLS → critic
            synth_enc = tokenize_synth(synth_sentences, tokenizer, args.max_length, device)
            # For critic reward we need a detached forward (LoRA grad goes
            # through logprob path, not through encoder here).
            with torch.no_grad():
                encoder_for_critic = model if not args.train_encoder else model
                synth_hidden = encoder_for_critic.encode(
                    modality="text",
                    input_ids=synth_enc["input_ids"],
                    attention_mask=synth_enc["attention_mask"],
                )
                synth_cls = synth_hidden[:, 0, :]
                critic_logit_detached = critic(synth_cls).detach()  # (B,)

            # Triple recovery reward
            recovery_scores = tr_mod.score_batch(
                synth_sentences, triples, recovery_encoder, tokenizer, device,
            )  # (B,)

            # Composite reward (no grad w.r.t. rewards themselves)
            critic_reward = critic_logit_detached
            recovery_reward = recovery_scores
            rewards = (
                args.alpha * critic_reward
                + args.beta * recovery_reward
            )  # (B,), no grad

            # KL penalty: γ * KL, penalty only (no baseline)
            kl_penalty = args.gamma * kl.mean()    # grad into LoRA via lora_lp in KL

            # Baseline (EMA over reward mean)
            mean_reward = rewards.mean().item()
            ema_baseline = args.ema_decay * ema_baseline + (1 - args.ema_decay) * mean_reward
            advantage = rewards - ema_baseline       # (B,)

            reinforce_loss = -(advantage.detach() * lora_lp).mean() + kl_penalty
            total = total + reinforce_loss

            reinforce_val = reinforce_loss.item()
            reward_val = mean_reward
            critic_reward_val = critic_reward.mean().item()
            recovery_reward_val = recovery_reward.mean().item()
            kl_val = kl.mean().item()

            # Critic optimization (train D *and* C if --train-encoder)
            if args.train_encoder:
                # Re-forward synth WITH grad for critic update
                synth_hidden_grad = encoder_for_critic.encode(
                    modality="text",
                    input_ids=synth_enc["input_ids"],
                    attention_mask=synth_enc["attention_mask"],
                )
                fake_logits = critic(synth_hidden_grad[:, 0, :])
                c_loss = critic_loss(real_logits, fake_logits)
                total = total + 0.3 * c_loss
                crit_val = c_loss.item()

        total.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        if args.train_encoder:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(critic.parameters()), 1.0,
            )
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {step:04d}] L_sup={sup_loss_val:.3f} "
                f"L_reinforce={reinforce_val:+.3f} "
                f"reward={reward_val:+.3f} (α·crit={args.alpha * critic_reward_val:+.2f} "
                f"β·rec={args.beta * recovery_reward_val:+.3f}) "
                f"kl={kl_val:+.2f} L_crit={crit_val:.3f} "
                f"lr={cur_lr:.2e} | {dt:.0f}ms/step"
            )

        if step > 0 and step % args.eval_every == 0:
            model.eval()
            metrics = evaluate(model, dev_loader, device)
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " ★ NEW BEST"
                decoder.save_adapters(args.save_adapters_to)
                star += f" (lora→{args.save_adapters_to})"
            print(
                f"[Eval @ step {step}] "
                f"NER F1={metrics['ner_f1']:.4f} | "
                f"RE F1={metrics['re_f1']:.4f} | "
                f"Triple F1={metrics['triple_f1']:.4f} | "
                f"EMA baseline={ema_baseline:+.3f}{star}"
            )
            if args.train_encoder:
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
    print(f"=== BEST DEV (step {best_step}) ===")
    print(f"  NER F1    = {best_metrics['ner_f1']:.4f}")
    print(f"  RE F1     = {best_metrics['re_f1']:.4f}")
    print(f"  Triple F1 = {best_metrics['triple_f1']:.4f}")
    print(f"  Total time = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
