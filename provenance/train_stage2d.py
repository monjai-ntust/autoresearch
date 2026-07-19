"""
Stage 2d training: LoRA Decoder + REINFORCE with the plan's §4.3 reward.

Fork of train_stage2c.py. Key differences, all motivated by the stage2c
mode-collapse post-mortem in reports/stage2/stage2_008.md and the
redesign in stage2_009_DESIGN.md:

1. Reward formula returns to migration plan §4.3:
       reward = α · tanh(critic_logit / 3)
              + β · clamp(1 − L_rec(E(synth), source_triple) / scale, 0, 1)
2. Curriculum: Phase A (α=0, β only) → Phase B (both). Prevents the
   critic from dominating while LoRA is still learning to paraphrase.
3. Per-batch advantage (not EMA) — mode collapse no longer absorbs the
   baseline into the reward constant.
4. Entropy bonus η — keeps sampling diverse.
5. Hard KL clip — if kl.mean() > KL_MAX, skip REINFORCE update and only
   apply γ·kl penalty. Stops runaway drift from base distribution.
6. LoRA lr 1e-4 → 1e-5, flat (no peak during the fragile early window).
"""
import argparse
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.scierc import build_dataloaders, ID2REL, NO_REL_ID
from models.bert_kg_encoder import BertKGExtractor
from models.critic import RealismCritic
from models.decoder_d_lora import LoRAQwenDecoder
from models import encoder_reward as er_mod
from eval.triple_f1 import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--stage2b-ckpt", default="checkpoints/stage2_007_best.pt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    # LoRA / optimizer
    p.add_argument("--lora-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--phase-a-steps", type=int, default=500,
                   help="Encoder-loss-only phase. α=0, β term only.")
    p.add_argument("--phase-b-steps", type=int, default=1500,
                   help="Full reward phase. α+β.")
    p.add_argument("--synth-batch-size", type=int, default=16)
    # Reward
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Weight on tanh(critic_logit/3). Zeroed in Phase A.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Weight on β term (string containment or L_rec).")
    p.add_argument("--gamma", type=float, default=0.05,
                   help="KL penalty weight (weak; hard clip is the main safeguard).")
    p.add_argument("--eta", type=float, default=0.01,
                   help="Entropy bonus coefficient.")
    p.add_argument("--l-rec-scale", type=float, default=4.0)
    p.add_argument("--kl-max", type=float, default=10.0,
                   help="Hard KL constraint: skip REINFORCE step if kl.mean() exceeds this.")
    p.add_argument("--beta-mode", default="string",
                   choices=["string", "encoder"],
                   help="β term: 'string' = entity containment (v3), 'encoder' = L_rec (v1/v2).")
    # LoRA config
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Eval / IO
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--arxiv-jsonl", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-adapters-to", default=None,
                   help="Where to save LoRA adapters. If None, defaults to "
                        "checkpoints/stage2_009_lora_<tag> where <tag> is the "
                        "CLI-derived variant tag. Pass explicit path to override.")
    p.add_argument("--variant-tag", default="default",
                   help="Variant identifier (e.g. v7, r16_lr1e6). Used only "
                        "when --save-adapters-to is None to build the default path.")
    return p.parse_args()


def sample_triples_from_batch(batch, n: int):
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


def load_stage2b_ckpt(path, model, critic):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["encoder"])
    critic.load_state_dict(ckpt["critic"])
    print(f"[stage2d] loaded stage2b ckpt step={ckpt['step']} "
          f"triple_f1={ckpt['metrics']['triple_f1']:.4f}")
    return ckpt


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Fill in variant-aware default for adapter save path so that
    # ablation runs (v5, v6, v7...) don't silently overwrite each other.
    if args.save_adapters_to is None:
        args.save_adapters_to = f"checkpoints/stage2_009_lora_{args.variant_tag}"

    total_steps = args.phase_a_steps + args.phase_b_steps
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Stage 2d — LoRA Decoder + REINFORCE v2 ===")
    print(f"  encoder:          {args.model_name}")
    print(f"  decoder:          {args.decoder_name}")
    print(f"  stage2b ckpt:     {args.stage2b_ckpt}")
    print(f"  phases:           A={args.phase_a_steps} (β-only) | B={args.phase_b_steps} (α+β) | total={total_steps}")
    print(f"  LoRA:             r={args.lora_r} α={args.lora_alpha} dropout={args.lora_dropout}")
    beta_desc = (f"β={args.beta}·string_containment" if args.beta_mode == "string"
                 else f"β={args.beta}·(1−L_rec/{args.l_rec_scale})")
    print(f"  reward:           α={args.alpha}·tanh(crit/3) + {beta_desc}")
    print(f"  β mode:           {args.beta_mode}")
    print(f"  KL:               γ={args.gamma}  KL_MAX={args.kl_max}")
    print(f"  entropy bonus η:  {args.eta}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_dir = Path(args.data_dir) if args.data_dir else None
    train_loader, dev_loader, _ = build_dataloaders(
        tokenizer, data_dir=data_dir,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"  sci train: {len(train_loader.dataset)} | sci dev: {len(dev_loader.dataset)}")

    # Encoder + critic — ALWAYS frozen in Stage 2d (Q4 A).
    model = BertKGExtractor(args.model_name).to(device)
    hidden = model.backbone.hidden_size
    critic = RealismCritic(hidden).to(device)
    load_stage2b_ckpt(args.stage2b_ckpt, model, critic)
    model.eval()
    critic.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in critic.parameters():
        p.requires_grad = False

    # LoRA decoder
    decoder = LoRAQwenDecoder(
        args.decoder_name, device=device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    lora_params = [p for p in decoder.model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lora_lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    sci_iter = iter(train_loader)
    best_metrics = {"triple_f1": -1.0}
    best_step = -1
    kl_clip_count = 0

    t0 = time.time()
    step = 0
    while step < total_steps:
        try:
            sci_batch = next(sci_iter)
        except StopIteration:
            sci_iter = iter(train_loader)
            sci_batch = next(sci_iter)

        optimizer.zero_grad()

        phase = "A" if step < args.phase_a_steps else "B"
        cur_alpha = 0.0 if phase == "A" else args.alpha

        triples = sample_triples_from_batch(sci_batch, n=args.synth_batch_size)
        if not triples:
            step += 1
            continue

        sampled = decoder.sample_with_logprob(triples)
        synth_sentences = sampled["sentences"]
        lora_lp = sampled["lora_logprob"]     # (B,) grad
        kl = sampled["kl"]                     # (B,) grad

        # ── β term ──
        if args.beta_mode == "string":
            # v3: direct string containment — robust, no encoder dependency
            rec_reward = er_mod.string_containment_batch(
                synth_sentences, triples, device,
            )
            l_rec_diag = rec_reward.clone()   # diagnostic only (1.0 = both found)
        else:
            # v1/v2: encoder-based L_rec (fragile on Qwen paraphrases)
            l_rec = er_mod.l_rec_batch(
                model, synth_sentences, triples, tokenizer, device,
                max_length=args.max_length, max_loss=args.l_rec_scale,
            )
            rec_reward = torch.clamp(1.0 - l_rec / args.l_rec_scale, min=0.0, max=1.0)
            l_rec_diag = l_rec.clone()

        # ── α term: tanh(critic_logit / 3) ────────────────────────
        with torch.no_grad():
            synth_enc = tokenizer(
                synth_sentences, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_length,
            ).to(device)
            synth_hidden = model.encode(
                modality="text",
                input_ids=synth_enc["input_ids"],
                attention_mask=synth_enc["attention_mask"],
            )
            critic_logit = critic(synth_hidden[:, 0, :])
            critic_reward = torch.tanh(critic_logit / 3.0)

        rewards = cur_alpha * critic_reward + args.beta * rec_reward   # (B,)

        # Per-batch advantage (no EMA)
        advantage = rewards - rewards.mean()

        # Hard KL clip
        kl_mean = kl.mean()
        if kl_mean.item() > args.kl_max:
            # Policy drift too far — only apply KL penalty, skip the REINFORCE term.
            loss = args.gamma * kl_mean
            kl_clip_count += 1
        else:
            reinforce = -(advantage.detach() * lora_lp).mean()
            entropy_bonus = -lora_lp.mean()          # higher lp = lower entropy; we REWARD entropy
            loss = reinforce + args.gamma * kl_mean - args.eta * entropy_bonus

        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {step:04d} {phase}] "
                f"reward={rewards.mean().item():+.3f} "
                f"(α·crit={cur_alpha * critic_reward.mean().item():+.3f} "
                f"β·rec={args.beta * rec_reward.mean().item():+.3f}) "
                f"diag={l_rec_diag.mean().item():.3f} "
                f"kl={kl_mean.item():+.2f} "
                f"loss={loss.item():+.3f} "
                f"lr={cur_lr:.2e} | {dt:.0f}ms/step "
                f"| kl_clips={kl_clip_count}"
            )

        if step > 0 and step % args.eval_every == 0:
            # model is frozen; we still call eval to compute Triple F1
            # since stage2d's goal is "F1 doesn't drop" even though the
            # encoder itself doesn't change. The F1 is a constant under
            # this training regime, but it's worth printing for sanity.
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
                f"Triple F1={metrics['triple_f1']:.4f}"
                f"{star}"
            )

        step += 1

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
    print(f"  kl_clips  = {kl_clip_count}")
    print(f"  total time= {time.time() - t0:.1f}s")
    # Always save the final LoRA (the "best" save only fires on encoder
    # F1 improvement, which never happens when the encoder is frozen).
    final_path = args.save_adapters_to + "_final"
    decoder.save_adapters(final_path)
    print(f"  final lora saved → {final_path}")


if __name__ == "__main__":
    main()
