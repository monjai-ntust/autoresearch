"""
Stage 2-030: Gumbel-STE adversarial training.

Port of Stage 1's Gumbel-Softmax STE mechanism to Stage 2's real-data
pipeline. REINFORCE failed 14 times across 4 stages (2c/2d/2e/2-029)
because it's too high-variance for the decoder to chase a live critic.
Gumbel-STE provides direct gradient flow from critic to decoder params.

Architecture:
  1. Decoder (LoRA Qwen) teacher-forces on generated text -> logits (B,T,V_qwen)
  2. Gumbel-Softmax(logits, tau, hard=True) -> soft one-hot (B,T,V_qwen)
  3. soft_onehot @ Qwen.wte -> embeddings (B,T, H_qwen=896)
  4. Linear(H_qwen, H_bert) -> projected (B,T, H_bert=768)
  5. Encoder backbone(inputs_embeds=projected) -> hidden (B,T, H_bert)
  6. CLS = hidden[:,0,:] -> Critic(CLS) -> BCE loss
  7. Gradient flows: Critic -> encoder -> projection -> Gumbel-STE -> LoRA params

Usage:
    uv run python train_gumbel.py --dataset scierc --max-steps 2500
    uv run python train_gumbel.py --dataset scierc --tau 0.5 --n-critic 1
"""
import argparse
import importlib
import os
import random
import time
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.bert_kg_encoder import BertKGExtractor
from models.critic import RealismCritic, critic_loss
from models.decoder_d_lora import LoRAQwenDecoder
from data.arxiv_real import build_arxiv_loader, cycle as arxiv_cycle


DATASET_REGISTRY = {
    "scierc": "data.scierc",
    "conll04": "data.conll04",
    "ade": "data.ade",
}


class VocabProjection(nn.Module):
    """Project from Qwen hidden space to SciBERT hidden space.

    The Gumbel-STE path produces embeddings in Qwen's hidden dim (896).
    The encoder backbone expects inputs_embeds in SciBERT's hidden dim (768).
    This learned linear projection bridges the two spaces.
    """
    def __init__(self, qwen_hidden: int, bert_hidden: int):
        super().__init__()
        self.proj = nn.Linear(qwen_hidden, bert_hidden)

    def forward(self, x):
        return self.proj(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scierc", choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--model-name", default=None)
    p.add_argument("--decoder-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--synth-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    # Learning rates
    p.add_argument("--encoder-lr", type=float, default=5e-5)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--lora-lr", type=float, default=2e-6)
    p.add_argument("--proj-lr", type=float, default=1e-4)
    # Gumbel-STE specific
    p.add_argument("--tau", type=float, default=1.0,
                   help="Gumbel-Softmax temperature. Lower = harder, more discrete.")
    p.add_argument("--tau-anneal-to", type=float, default=0.0,
                   help="If >0, anneal tau linearly to this value over training.")
    p.add_argument("--n-critic", type=int, default=1,
                   help="Critic updates per decoder update. With Gumbel-STE, "
                        "decoder gets exact gradients so n_critic=1 should suffice.")
    p.add_argument("--adv-stop-step", type=int, default=0,
                   help="Stop adversarial updates after this step. 0=never.")
    p.add_argument("--feature-matching", action="store_true",
                   help="Use feature matching loss instead of BCE for decoder update. "
                        "Matches mean intermediate critic features between real and fake.")
    p.add_argument("--adv-start-step", type=int, default=200,
                   help="Start adversarial updates after this many supervised steps.")
    p.add_argument("--max-gen-tokens", type=int, default=40)
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
            pool.append((" ".join(words[hs:he+1]), ds_mod.ID2REL[rel_id],
                         " ".join(words[ts:te+1])))
    if not pool:
        return []
    if len(pool) >= n:
        return random.sample(pool, n)
    return [random.choice(pool) for _ in range(n)]


def cycle(loader):
    while True:
        yield from loader


def gumbel_ste_cls(decoder, triples, encoder, projection,
                   device, tau=1.0, max_gen_tokens=40):
    """Like gumbel_ste_forward but returns CLS hidden state for feature matching."""
    if not triples:
        return None
    with torch.no_grad():
        sentences = decoder.generate_batch(triples, max_new_tokens=max_gen_tokens)
    prompts = decoder.build_prompts(triples)
    full_texts = [p + s for p, s in zip(prompts, sentences)]
    full_enc = decoder.tokenizer(
        full_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)
    prompt_enc = decoder.tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]
    outputs = decoder.model(
        input_ids=full_enc["input_ids"],
        attention_mask=full_enc["attention_mask"],
    )
    gen_logits = outputs.logits[:, prompt_len - 1:-1, :]
    if gen_logits.shape[1] == 0:
        return None
    gen_logits_f32 = gen_logits.float()
    fake_soft = F.gumbel_softmax(gen_logits_f32, tau=tau, hard=True, dim=-1)
    qwen_wte = decoder.model.get_input_embeddings().weight
    fake_embedded = fake_soft @ qwen_wte.float()
    projected = projection(fake_embedded)
    attn_mask = full_enc["attention_mask"][:, prompt_len:]
    if attn_mask.shape[1] > projected.shape[1]:
        attn_mask = attn_mask[:, :projected.shape[1]]
    elif attn_mask.shape[1] < projected.shape[1]:
        projected = projected[:, :attn_mask.shape[1], :]
    hidden = encoder.backbone(
        inputs_embeds=projected,
        attention_mask=attn_mask,
    )
    return hidden[:, 0, :]  # (B, H_bert) — CLS


def gumbel_ste_forward(decoder, triples, encoder, projection, critic,
                       device, tau=1.0, max_gen_tokens=40):
    """
    Gumbel-STE forward pass: decoder -> Gumbel -> embeddings -> encoder -> critic.

    Returns critic logits (B,) with gradient flowing back to decoder LoRA params.

    Steps:
      1. Generate target tokens (no grad) to use as teacher-forcing input
      2. Teacher-force through LoRA model to get logits
      3. Apply Gumbel-Softmax STE on generated portion
      4. Multiply soft one-hot with Qwen word embeddings
      5. Project to SciBERT hidden dim
      6. Feed to encoder backbone -> CLS -> critic
    """
    if not triples:
        return None, 0

    # 1. Generate target tokens (no grad) for teacher-forcing
    with torch.no_grad():
        sentences = decoder.generate_batch(triples, max_new_tokens=max_gen_tokens)

    # 2. Build full sequences: prompt + generated text
    prompts = decoder.build_prompts(triples)
    full_texts = [p + s for p, s in zip(prompts, sentences)]

    # Tokenize the full sequences
    full_enc = decoder.tokenizer(
        full_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)

    # Also tokenize just the prompts to know where generation starts
    prompt_enc = decoder.tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    # 3. Teacher-force through LoRA model (WITH grad for LoRA params)
    outputs = decoder.model(
        input_ids=full_enc["input_ids"],
        attention_mask=full_enc["attention_mask"],
    )
    logits = outputs.logits  # (B, L, V_qwen)

    # Take logits for the generated portion only
    # Logits at position t predict token at t+1, so for generated tokens
    # starting at prompt_len, we want logits at positions [prompt_len-1, ...)
    gen_logits = logits[:, prompt_len - 1:-1, :]  # (B, T_gen, V_qwen)

    if gen_logits.shape[1] == 0:
        return None, 0

    # 4. Gumbel-Softmax STE: hard=True for discrete forward, soft backward
    # Cast to float32 for Gumbel (bf16 causes numerical issues with log)
    gen_logits_f32 = gen_logits.float()
    fake_soft = F.gumbel_softmax(gen_logits_f32, tau=tau, hard=True, dim=-1)
    # (B, T_gen, V_qwen) — one-hot forward, soft gradient backward

    # 5. Multiply with Qwen's word embedding table to get continuous repr
    # Get the embedding weight (V_qwen, H_qwen)
    qwen_wte = decoder.model.get_input_embeddings().weight  # (V_qwen, H_qwen)
    fake_embedded = fake_soft @ qwen_wte.float()  # (B, T_gen, H_qwen)

    # 6. Project to SciBERT hidden dim (stay in float32 for numerical stability)
    projected = projection(fake_embedded)  # (B, T_gen, H_bert), float32

    # 7. Feed to encoder backbone (bypass adapter, use inputs_embeds directly)
    attn_mask = full_enc["attention_mask"][:, prompt_len:]  # (B, T_gen)
    # Trim to match gen_logits length
    if attn_mask.shape[1] > projected.shape[1]:
        attn_mask = attn_mask[:, :projected.shape[1]]
    elif attn_mask.shape[1] < projected.shape[1]:
        projected = projected[:, :attn_mask.shape[1], :]

    hidden = encoder.backbone(
        inputs_embeds=projected,
        attention_mask=attn_mask,
    )  # (B, T_gen, H_bert)

    # 8. CLS token -> critic
    cls_hidden = hidden[:, 0, :]  # (B, H_bert)
    critic_logits = critic(cls_hidden)  # (B,)

    return critic_logits, len(triples)


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

    print("=== Stage 2-030: Gumbel-STE Adversarial Training ===")
    print(f"  dataset:     {args.dataset}")
    print(f"  encoder:     {args.model_name}")
    print(f"  decoder:     {args.decoder_name}")
    print(f"  tau:         {args.tau}")
    print(f"  n_critic:    {args.n_critic}")
    print(f"  adv_start:   {args.adv_start_step}")
    print(f"  adv_stop:    {args.adv_stop_step}")
    print(f"  feat_match:  {args.feature_matching}")
    print(f"  LRs:         enc={args.encoder_lr} crit={args.critic_lr} "
          f"lora={args.lora_lr} proj={args.proj_lr}")

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
    arxiv_loader = build_arxiv_loader(
        tokenizer, batch_size=args.synth_batch_size, max_length=args.max_length,
    )
    print(f"  train: {len(train_loader.dataset)} | dev: {len(dev_loader.dataset)} "
          f"| arXiv: {len(arxiv_loader.dataset)}")

    # -- Encoder (span NER v10 config) --
    entity_types = ds_mod.ENTITY_TYPES
    entity_type2id = {t: i + 1 for i, t in enumerate(entity_types)}
    id2entity_type = {i + 1: t for i, t in enumerate(entity_types)}

    encoder = BertKGExtractor(
        args.model_name,
        num_bio_tags=ds_mod.NUM_BIO_TAGS,
        num_relations=ds_mod.NUM_RELATIONS,
        num_entity_types=len(entity_types),
        use_span_ner=True,
        max_span_width=args.max_span_width,
    ).to(device)

    # -- Critic --
    hidden_size = encoder.backbone.hidden_size
    critic_model = RealismCritic(hidden_size).to(device)

    # -- Decoder (LoRA) --
    decoder = LoRAQwenDecoder(
        args.decoder_name, device=device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # -- Vocab projection (Qwen hidden -> SciBERT hidden) --
    # Get Qwen hidden size from the already-loaded model (no extra HF call)
    qwen_hidden = decoder.model.config.hidden_size
    projection = VocabProjection(qwen_hidden, hidden_size).to(device)
    print(f"  projection:  {qwen_hidden} -> {hidden_size}")

    # -- Optimizers --
    encoder_optimizer = AdamW(encoder.parameters(), lr=args.encoder_lr, weight_decay=0.01)
    critic_optimizer = AdamW(critic_model.parameters(), lr=args.critic_lr, weight_decay=0.01)

    lora_params = [p for p in decoder.model.parameters() if p.requires_grad]
    # Decoder optimizer includes projection params (both are updated in Step B)
    decoder_optimizer = AdamW(
        [{"params": lora_params, "lr": args.lora_lr},
         {"params": projection.parameters(), "lr": args.proj_lr}],
        weight_decay=0.01,
    )

    encoder_scheduler = get_linear_schedule_with_warmup(
        encoder_optimizer, args.warmup_steps, args.max_steps,
    )

    # -- Iterators --
    gold_iter = cycle(train_loader)
    arxiv_iter = arxiv_cycle(arxiv_loader)
    best_metrics = {"triple_f1": -1.0}
    best_step = -1

    from train_span import compute_span_loss, evaluate_span

    encoder.train()
    critic_model.train()
    projection.train()
    t0 = time.time()

    for step in range(args.max_steps):
        gold_batch = next(gold_iter)
        triples = sample_triples(gold_batch, ds_mod, args.synth_batch_size)

        adv_active = (step >= args.adv_start_step and
                      (args.adv_stop_step == 0 or step < args.adv_stop_step))

        # Compute current tau (with optional annealing)
        if args.tau_anneal_to > 0 and adv_active:
            progress = (step - args.adv_start_step) / max(
                (args.adv_stop_step or args.max_steps) - args.adv_start_step, 1)
            current_tau = args.tau + (args.tau_anneal_to - args.tau) * min(progress, 1.0)
        else:
            current_tau = args.tau

        # ============================================================
        # Step A: Train Critic (n_critic times)
        # ============================================================
        critic_loss_val = 0.0
        real_mean = 0.0
        fake_mean = 0.0

        if adv_active:
            for _ in range(args.n_critic):
                critic_optimizer.zero_grad()

                # Real text from arXiv
                real_batch = next(arxiv_iter)
                with torch.no_grad():
                    real_hidden = encoder.encode(
                        modality="text",
                        input_ids=real_batch["input_ids"].to(device),
                        attention_mask=real_batch["attention_mask"].to(device),
                    )
                real_cls = real_hidden[:, 0, :].detach()
                real_logits = critic_model(real_cls)

                # Fake via Gumbel-STE path (same path as Step B)
                # Critical: critic must see the same embedding path the
                # decoder uses, otherwise the critic can't provide useful
                # gradient signal. Prior v1/v2 used generate+tokenize here
                # which is a different embedding space → decoder "won" trivially.
                if triples:
                    with torch.no_grad():
                        fake_cls_for_critic = gumbel_ste_cls(
                            decoder, triples, encoder, projection,
                            device, tau=current_tau,
                            max_gen_tokens=args.max_gen_tokens,
                        )
                    if fake_cls_for_critic is not None:
                        fake_cls = fake_cls_for_critic.detach()
                        fake_logits = critic_model(fake_cls)
                        c_loss = critic_loss(real_logits, fake_logits)
                    else:
                        c_loss = F.binary_cross_entropy_with_logits(
                            real_logits, torch.ones_like(real_logits),
                        )
                        fake_logits = real_logits  # placeholder
                else:
                    c_loss = F.binary_cross_entropy_with_logits(
                        real_logits, torch.ones_like(real_logits),
                    )
                    fake_logits = real_logits  # placeholder

                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
                critic_optimizer.step()
                critic_loss_val = c_loss.item()
                real_mean = real_logits.mean().item()
                fake_mean = fake_logits.mean().item() if triples else 0.0

        # ============================================================
        # Step B: Train Decoder via Gumbel-STE (NOT REINFORCE)
        # ============================================================
        decoder_loss_val = 0.0

        if adv_active and triples:
            decoder_optimizer.zero_grad()

            # Freeze encoder and critic for this step — gradient flows
            # through them but only decoder+projection params are updated
            encoder.requires_grad_(False)
            critic_model.requires_grad_(False)

            # Single Gumbel-STE path for CLS (gradient flows to decoder LoRA)
            fake_cls_b = gumbel_ste_cls(
                decoder, triples, encoder, projection,
                device, tau=current_tau, max_gen_tokens=args.max_gen_tokens,
            )

            if fake_cls_b is not None:
                if args.feature_matching:
                    # Feature matching: match mean intermediate critic features
                    fake_feat = critic_model.features(fake_cls_b)
                    real_feat = critic_model.features(real_cls.detach())
                    loss_d = F.mse_loss(fake_feat.mean(0), real_feat.mean(0))
                else:
                    # Standard: decoder wants critic to think fake is real
                    fake_logits_b = critic_model(fake_cls_b)
                    loss_d = F.binary_cross_entropy_with_logits(
                        fake_logits_b,
                        torch.ones_like(fake_logits_b),
                    )
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
                decoder_optimizer.step()
                decoder_loss_val = loss_d.item()

            # Unfreeze encoder and critic
            encoder.requires_grad_(True)
            critic_model.requires_grad_(True)

        # ============================================================
        # Step C: Train Encoder (span NER + RE on gold data)
        # ============================================================
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

        # -- Logging --
        if step % 10 == 0:
            dt = (time.time() - t0) * 1000 / max(step, 1)
            tau_str = f"tau={current_tau:.2f}" if adv_active else "tau=off"
            print(
                f"[Step {step:04d}] "
                f"L_crit={critic_loss_val:.3f} real={real_mean:+.2f} fake={fake_mean:+.2f} | "
                f"L_dec={decoder_loss_val:.3f} {tau_str} | "
                f"L_enc={encoder_loss_val:.3f} NER={ner_loss.item():.3f} RE={re_loss.item():.3f} | "
                f"{dt:.0f}ms/step"
            )

        # -- Eval --
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
                        "critic": critic_model.state_dict(),
                        "projection": projection.state_dict(),
                        "step": step, "metrics": metrics,
                    }, save_path)
            print(f"[Eval @ {step}] NER={metrics['ner_f1']:.4f} "
                  f"Triple={metrics['triple_f1']:.4f}{star}")
            encoder.train()
            critic_model.train()
            projection.train()

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
    print(f"  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
