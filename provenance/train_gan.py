"""
Stage 2-029: GAN-style alternating Critic + Decoder + Encoder training.

The thesis experiment. Three-step loop per iteration:
  Step A: Train Critic on real arXiv vs Decoder-generated text
  Step B: Train Decoder (LoRA REINFORCE) to fool the live Critic
  Step C: Train Encoder (span NER + RE) on gold data

This is the first time the adversarial loop runs with ALL components
updating. Prior stages froze the critic (2d) or the encoder (2d).

Usage:
    uv run python train_gan.py --dataset scierc --max-steps 2500
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
from models.critic import RealismCritic, critic_loss
from models.decoder_d_lora import LoRAQwenDecoder
from models.encoder_reward import string_containment_batch
from data.arxiv_real import build_arxiv_loader, cycle as arxiv_cycle


DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "conll04": "data.conll04",
    "ade": "data.ade",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scierc", choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--model-name", default=None)
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--synth-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    # Learning rates
    p.add_argument("--encoder-lr", type=float, default=5e-5)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--lora-lr", type=float, default=2e-6)
    # GAN-specific
    p.add_argument("--n-critic", type=int, default=3,
                   help="Critic updates per decoder update.")
    p.add_argument("--encoder-update-every", type=int, default=1,
                   help="Encoder updates every N steps.")
    p.add_argument("--adv-stop-step", type=int, default=0,
                   help="Stop critic+decoder updates after this step. 0 = never stop. "
                        "Encoder continues training on gold only after this point.")
    # Reward
    p.add_argument("--alpha", type=float, default=1.0, help="Critic reward weight.")
    p.add_argument("--beta", type=float, default=1.0, help="Containment reward weight.")
    p.add_argument("--gamma", type=float, default=0.05, help="KL penalty.")
    p.add_argument("--kl-max", type=float, default=10.0)
    # Span NER
    p.add_argument("--max-span-width", type=int, default=8)
    p.add_argument("--re-weight", type=float, default=0.5)
    p.add_argument("--neg-sample-ratio", type=float, default=2.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # IO
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best-to", default=None)
    return p.parse_args()


def sample_triples(batch, ds_mod, n):
    """Sample (head, rel, tail) triples from a gold batch."""
    pool = []
    for sent_idx, rels in enumerate(batch["gold_relations"]):
        words = batch["words"][sent_idx]
        for (h_span, t_span, rel_id) in rels:
            if rel_id == ds_mod.NO_REL_ID:
                continue
            hs, he = h_span
            ts, te = t_span
            pool.append((" ".join(words[hs:he+1]), ds_mod.ID2REL[rel_id], " ".join(words[ts:te+1])))
    if not pool:
        return []
    if len(pool) >= n:
        return random.sample(pool, n)
    return [random.choice(pool) for _ in range(n)]


def cycle(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ds_mod = importlib.import_module(DATASET_REGISTRY[args.dataset])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name is None:
        args.model_name = {
            "scierc": "allenai/scibert_scivocab_uncased",
            "conll04": "bert-base-uncased",
            "ade": "allenai/scibert_scivocab_uncased",
        }.get(args.dataset, "bert-base-uncased")

    print("=== Stage 2-029: GAN-style Alternating Training ===")
    print(f"  dataset:    {args.dataset}")
    print(f"  encoder:    {args.model_name}")
    print(f"  decoder:    {args.decoder_name}")
    print(f"  n_critic:   {args.n_critic}")
    print(f"  LRs:        encoder={args.encoder_lr} critic={args.critic_lr} lora={args.lora_lr}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Patch scierc dicts if needed
    import data.scierc as scierc_mod
    if args.dataset != "scierc":
        scierc_mod.ID2BIO.clear()
        scierc_mod.ID2BIO.update(ds_mod.ID2BIO)
        scierc_mod.BIO_TAG2ID.clear()
        scierc_mod.BIO_TAG2ID.update(ds_mod.BIO_TAG2ID)
        scierc_mod.NO_REL_ID = ds_mod.NO_REL_ID

    train_loader, dev_loader, test_loader = ds_mod.build_dataloaders(
        tokenizer, batch_size=args.batch_size, max_length=args.max_length,
    )
    arxiv_loader = build_arxiv_loader(tokenizer, batch_size=args.synth_batch_size, max_length=args.max_length)
    print(f"  train: {len(train_loader.dataset)} | dev: {len(dev_loader.dataset)} | arXiv: {len(arxiv_loader.dataset)}")

    # ── Encoder (span NER v10 config) ──────────────────
    entity_types = ds_mod.ENTITY_TYPES
    entity_type2id = {t: i+1 for i, t in enumerate(entity_types)}
    id2entity_type = {i+1: t for i, t in enumerate(entity_types)}

    encoder = BertKGExtractor(
        args.model_name,
        num_bio_tags=ds_mod.NUM_BIO_TAGS,
        num_relations=ds_mod.NUM_RELATIONS,
        num_entity_types=len(entity_types),
        use_span_ner=True,
        max_span_width=args.max_span_width,
    ).to(device)

    # ── Critic ─────────────────────────────────────────
    hidden = encoder.backbone.hidden_size
    critic = RealismCritic(hidden).to(device)

    # ── Decoder (LoRA) ─────────────────────────────────
    decoder = LoRAQwenDecoder(
        args.decoder_name, device=device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── Optimizers ─────────────────────────────────────
    encoder_optimizer = AdamW(encoder.parameters(), lr=args.encoder_lr, weight_decay=0.01)
    critic_optimizer = AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=0.01)
    lora_params = [p for p in decoder.model.parameters() if p.requires_grad]
    lora_optimizer = AdamW(lora_params, lr=args.lora_lr, weight_decay=0.01)

    encoder_scheduler = get_linear_schedule_with_warmup(
        encoder_optimizer, args.warmup_steps, args.max_steps,
    )

    # ── Iterators ──────────────────────────────────────
    gold_iter = cycle(train_loader)
    arxiv_iter = arxiv_cycle(arxiv_loader)
    best_metrics = {"triple_f1": -1.0}
    best_step = -1
    kl_clip_count = 0

    # Import span loss + eval from train_span.py
    from train_span import compute_span_loss, evaluate_span

    encoder.train()
    critic.train()
    t0 = time.time()

    for step in range(args.max_steps):
        gold_batch = next(gold_iter)
        triples = sample_triples(gold_batch, ds_mod, args.synth_batch_size)

        # Check if adversarial phase should stop
        adv_active = (args.adv_stop_step == 0) or (step < args.adv_stop_step)

        # ══════════════════════════════════════════════════
        # Step A: Train Critic (n_critic times)
        # ══════════════════════════════════════════════════
        critic_loss_val = 0.0
        real_mean = 0.0
        fake_mean = 0.0

        if adv_active:
            for _ in range(args.n_critic):
                critic_optimizer.zero_grad()

                real_batch = next(arxiv_iter)
                with torch.no_grad():
                    real_hidden = encoder.encode(
                        modality="text",
                        input_ids=real_batch["input_ids"].to(device),
                        attention_mask=real_batch["attention_mask"].to(device),
                    )
                real_cls = real_hidden[:, 0, :].detach()
                real_logits = critic(real_cls)

                if triples:
                    with torch.no_grad():
                        fake_sentences = decoder.generate_batch(triples)
                    fake_enc = tokenizer(
                        fake_sentences, return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_length,
                    ).to(device)
                    with torch.no_grad():
                        fake_hidden = encoder.encode(
                            modality="text",
                            input_ids=fake_enc["input_ids"],
                            attention_mask=fake_enc["attention_mask"],
                        )
                    fake_cls = fake_hidden[:, 0, :].detach()
                    fake_logits = critic(fake_cls)
                    c_loss = critic_loss(real_logits, fake_logits)
                else:
                    c_loss = F.binary_cross_entropy_with_logits(
                        real_logits, torch.ones_like(real_logits),
                    )

                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()
                critic_loss_val = c_loss.item()
                real_mean = real_logits.mean().item()
                fake_mean = fake_logits.mean().item() if triples else 0.0

        # ══════════════════════════════════════════════════
        # Step B: Train Decoder (REINFORCE with live critic)
        # ══════════════════════════════════════════════════
        decoder_loss_val = 0.0
        reward_val = 0.0
        kl_val = 0.0
        containment_val = 0.0

        if adv_active and triples:
            lora_optimizer.zero_grad()
            sampled = decoder.sample_with_logprob(triples)
            synth_sentences = sampled["sentences"]
            lora_lp = sampled["lora_logprob"]
            kl = sampled["kl"]

            # Critic reward (live, not frozen)
            synth_enc = tokenizer(
                synth_sentences, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_length,
            ).to(device)
            with torch.no_grad():
                synth_hidden = encoder.encode(
                    modality="text",
                    input_ids=synth_enc["input_ids"],
                    attention_mask=synth_enc["attention_mask"],
                )
                synth_cls = synth_hidden[:, 0, :]
                critic_logit = critic(synth_cls)
            critic_reward = torch.tanh(critic_logit / 3.0)

            # String containment reward
            containment = string_containment_batch(synth_sentences, triples, device)

            rewards = args.alpha * critic_reward + args.beta * containment
            advantage = rewards - rewards.mean()

            kl_mean = kl.mean()
            if kl_mean.item() > args.kl_max:
                loss_d = args.gamma * kl_mean
                kl_clip_count += 1
            else:
                loss_d = -(advantage.detach() * lora_lp).mean() + args.gamma * kl_mean

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            lora_optimizer.step()
            decoder_loss_val = loss_d.item()
            reward_val = rewards.mean().item()
            kl_val = kl_mean.item()
            containment_val = containment.mean().item()

        # ══════════════════════════════════════════════════
        # Step C: Train Encoder (span NER + RE on gold)
        # ══════════════════════════════════════════════════
        encoder_loss_val = 0.0
        if step % args.encoder_update_every == 0:
            encoder_optimizer.zero_grad()
            enc_loss, ner_loss, re_loss = compute_span_loss(
                encoder, gold_batch, device, ds_mod, entity_type2id,
                re_weight=args.re_weight, neg_sample_ratio=args.neg_sample_ratio,
                max_span_width=args.max_span_width, focal_gamma=args.focal_gamma,
            )
            enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            encoder_optimizer.step()
            encoder_scheduler.step()
            encoder_loss_val = enc_loss.item()

        # ── Logging ────────────────────────────────────
        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            print(
                f"[Step {step:04d}] "
                f"L_crit={critic_loss_val:.3f} real={real_mean:+.2f} fake={fake_mean:+.2f} | "
                f"L_dec={decoder_loss_val:+.3f} reward={reward_val:+.3f} "
                f"contain={containment_val:.2f} kl={kl_val:+.2f} | "
                f"L_enc={encoder_loss_val:.3f} | "
                f"{dt:.0f}ms/step clips={kl_clip_count}"
            )

        # ── Eval ───────────────────────────────────────
        if step > 0 and step % args.eval_every == 0:
            metrics = evaluate_span(
                encoder, dev_loader, device, ds_mod,
                entity_type2id, id2entity_type,
                max_span_width=args.max_span_width,
            )
            star = ""
            if metrics["triple_f1"] > best_metrics["triple_f1"]:
                best_metrics = dict(metrics)
                best_step = step
                star = " *"
                if args.save_best_to:
                    save_path = Path(args.save_best_to)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "encoder": encoder.state_dict(),
                        "critic": critic.state_dict(),
                        "step": step, "metrics": metrics,
                    }, save_path)
            print(f"[Eval @ {step}] NER={metrics['ner_f1']:.4f} "
                  f"Triple={metrics['triple_f1']:.4f}{star}")
            encoder.train()
            critic.train()

    # Final eval
    for name, loader in [("dev", dev_loader), ("test", test_loader)]:
        metrics = evaluate_span(
            encoder, loader, device, ds_mod,
            entity_type2id, id2entity_type,
            max_span_width=args.max_span_width,
        )
        if name == "dev" and metrics["triple_f1"] > best_metrics["triple_f1"]:
            best_metrics = dict(metrics)
            best_step = args.max_steps
        print(f"\n=== {name.upper()} ===")
        print(f"  NER={metrics['ner_f1']:.4f} Triple={metrics['triple_f1']:.4f}")
    print(f"=== BEST DEV (step {best_step}) ===")
    print(f"  NER={best_metrics['ner_f1']:.4f} Triple={best_metrics['triple_f1']:.4f}")
    print(f"  kl_clips={kl_clip_count}")
    print(f"  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
