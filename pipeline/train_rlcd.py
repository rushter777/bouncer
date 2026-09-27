#!/usr/bin/env python3
"""RLCD fine-tuning for Laya on local JSONL data — adapted from Kaggle DDP notebook.

Key deviations from the original notebook:
- Single-GPU MPS instead of 2×T4 DDP (no torch.distributed, no NCCL)
- fp32 training on MPS (bf16 autocast not supported; see amp_dtype comment below)
- Gradient checkpointing disabled for MPS compatibility
- Uses collate_items from laya.common instead of custom collate_train_batch
- CLI args via argparse instead of hardcoded constants
- Temperature calibration runs after training (kept same logic as notebook)

Usage:
    python train_rlcd.py --data data/router_train.jsonl --output-dir checkpoints/laya-ft \
                         --epochs 4 --batch-size 8 --grad-accum 4
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import (
    QTYPES,
    amp_dtype,
    build_model,
    collate_items,
    proper_reward,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load JSONL dataset; each row must have {state, questions, labels}."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_training_item(state, q, gold_label):
    """Convert one (state, question, label) into a training item dict.

    Adapted from the notebook's build_training_item but simplified for our format:
    - Our dataset has hard labels (strings/ints/floats), not teacher probability distributions.
    - We still normalise target to a distribution for the proper scoring rules.
    """
    t = q["type"]
    crit = q.get("criteria")

    if t == "choice":
        keys = list(crit) if isinstance(crit, (list, tuple)) else list(crit.keys())
        # Build soft target: 1.0 at the gold label index, uniform elsewhere
        gold_idx = keys.index(gold_label) if gold_label in keys else 0
        target = [1.0 / len(keys)] * len(keys)
        target[gold_idx] = 1.0 - (1.0 - 1.0 / len(keys))  # concentrate mass on correct
    elif t == "noul":
        # Binary: false=0, true=1
        is_true = str(gold_label).lower() in ("true", "1", "yes")
        target = [1.0 - is_true, is_true]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        gold_idx = int(float(gold_label)) if not isinstance(gold_label, str) else 0
        target = [1.0 / n_levels] * n_levels
        target[min(gold_idx, n_levels - 1)] = 1.0
    else:
        return None

    # Normalise to valid distribution
    s = sum(target)
    if s > 0:
        target = [v / s for v in target]
    else:
        target = [1.0 / len(target)] * len(target)

    label = target.index(max(target))

    return {
        "qtype": QTYPES[t],
        "target": target,
        "label": label,
    }


def prepare_items(data_rows, tok, cfg, model_dir):
    """Tokenise all (state, questions, labels) into model-ready items."""
    from laya.common import build_sequence, render_options

    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)

    items = []
    for row in data_rows:
        state = row["state"]
        if isinstance(state, str) and state[:1] in ("{", "["):
            # notebook-format rows carry JSON documents as strings; ours are plain text
            state = json.loads(state)
        questions = row["questions"]
        if isinstance(questions, str):
            questions = json.loads(questions)
        labels = row.get("labels", {})
        if isinstance(labels, str):
            labels = json.loads(labels)

        for qid, q in questions.items():
            if qid not in labels:
                continue
            gold_label = labels[qid]
            item_meta = build_training_item(state, q, gold_label)
            if item_meta is None:
                continue

            # normalize the same way laya.Agent._to_internal does:
            # criteria as a list -> {option: None} dict
            crit = q.get("criteria")
            if isinstance(crit, list):
                crit = {c: None for c in crit}
            internal_q = {"t": q["type"], "ins": q.get("instructions", ""), "crit": crit}

            seq, markers = build_sequence(
                tok, state,
                internal_q,
                max_len, head_max_len,
            )
            if len(markers) != len(render_options(internal_q)):
                continue

            item_meta["ids"] = seq
            item_meta["markers"] = markers
            items.append(item_meta)

    print(f"Prepared {len(items)} training items from {len(data_rows)} rows.")
    return items


# ── Temperature fitting (same logic as notebook) ─────────────────────────────

def fit_one_temp(sel):
    """Fit a single temperature via LBFGS on cross-entropy.

    sel: list of (logits_array, target_array) tuples.
    Returns fitted temperature clamped to [0.1, 10.0].
    """
    if len(sel) < 10:
        return 1.0

    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


# ── Training loop (single-GPU MPS adaptation of notebook's DDP main) ─────────

def main():
    parser = argparse.ArgumentParser(description="RLCD fine-tune Laya on JSONL data")
    parser.add_argument("--data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to base laya checkpoint (default: downloads from HF)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/laya-ft",
                        help="Directory to save fine-tuned checkpoint")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Micro-batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Number of noisy samples for RL baseline (GROUP_SIZE in notebook)")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1.0e-4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device setup ───────────────────────────────────────────────────────
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # NOTE: bf16 autocast is NOT supported on MPS in PyTorch. We train in fp32.
    # Deviation from notebook: original used torch.autocast("cuda", dtype=torch.float16).
    # On MPS we must use pure fp32; the model is small enough (421M) to fit.
    use_amp = False  # amp_dtype on MPS forces fp32 anyway

    # ── Load base checkpoint ───────────────────────────────────────────────
    if args.model_dir is None:
        from huggingface_hub import snapshot_download
        print("Downloading base checkpoint convaiinnovations/laya...")
        args.model_dir = snapshot_download("convaiinnovations/laya")

    _fix_tokenizer_config(args.model_dir)

    with open(os.path.join(args.model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))

    enc_dir = os.path.join(args.model_dir, "encoder")
    encoder_dir = enc_dir if os.path.exists(enc_dir) else None
    model = build_model(cfg, encoder_dir=encoder_dir)

    weights_path = os.path.join(args.model_dir, "model.safetensors")
    weights = load_file(weights_path)
    model.load_state_dict(weights, strict=True)

    # NOTE: gradient checkpointing disabled on MPS — it hangs with ModernBERT in PyTorch < 2.9.
    # Deviation from notebook which set cfg["gradient_checkpointing"] = True and called
    # model.encoder.gradient_checkpointing_enable(). We skip this for MPS compatibility.
    print("Gradient checkpointing: DISABLED (MPS compatibility)")

    model.to(device)
    model.train()

    # ── Load & prepare data ────────────────────────────────────────────────
    data_rows = load_jsonl(args.data)
    items = prepare_items(data_rows, tok, cfg, args.model_dir)

    EPOCHS = args.epochs
    MICRO_BATCH = args.batch_size
    GRAD_ACCUM = args.grad_accum
    GROUP_SIZE = args.group_size

    # ── Optimizer ──────────────────────────────────────────────────────────
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    total_updates = (len(items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )

    # NOTE: No GradScaler on MPS — it's CUDA-only. Deviation from notebook line 230.
    t0 = time.time()

    print(f"\nTraining: {len(items)} items | micro_batch={MICRO_BATCH} | grad_accum={GRAD_ACCUM}")
    print(f"Effective batch={MICRO_BATCH * GRAD_ACCUM} | epochs={EPOCHS} | updates~{total_updates}\n")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        random.seed(args.seed + epoch)
        random.shuffle(items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, EPOCHS - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress

        for b_idx in range(0, len(items), MICRO_BATCH):
            chunk = items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_items([chunk], tok.pad_token_id)
            if batch is None:
                continue

            # Forward pass (no autocast on MPS — fp32 only)
            logits, act = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # NaN guard: skip any batch whose logits are already non-finite
            # (one bad batch used to poison the weights permanently)
            if not torch.isfinite(logits).all():
                print(f"  [skip] non-finite logits at batch {b_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Evaluate proper scoring reward
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient loss + soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            # NaN guard: never backprop a non-finite loss
            if not torch.isfinite(loss):
                print(f"  [skip] non-finite loss at batch {b_idx} (rl={loss_rl.item():.3g})")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            accum_step += 1

            # capture scalars BEFORE releasing tensors (printout uses them below)
            loss_val = loss.item() * GRAD_ACCUM
            reward_val = r.mean().item()

            # MPS allocator hygiene: heterogeneous batch shapes make the private
            # pool grow unboundedly without periodic release (leak seen on v3 data)
            del eps, z, q, r, adv, logp, loss_rl, loss_ce, loss, logits, act
            if b_idx % (MICRO_BATCH * 10) == 0:
                torch.mps.empty_cache()

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss_val
            n_batches += 1

            if (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | Loss: {loss_val:.4f} | Reward: {reward_val:.3f} | LR: {cur_lr:.2e}")

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"=== Epoch {epoch+1}/{EPOCHS} Completed in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f} ===\n")

    # ── Post-training temperature calibration ──────────────────────────────
    print("Fitting post-training calibration temperatures...")
    model.eval()

    calib_items = items[::15][:400]  # same sampling as notebook (every 15th, first 400)
    calib_preds = []

    with torch.no_grad():
        for c_idx in range(0, len(calib_items), 16):
            c_chunk = calib_items[c_idx:c_idx + 16]
            cb = collate_items([c_chunk], tok.pad_token_id)
            if cb is None:
                continue

            l_sub, _ = model(
                cb["input_ids"].to(device),
                cb["attention_mask"].to(device),
                cb["marker_pos"].to(device),
                cb["marker_mask"].to(device),
                cb["qtype"].to(device),
            )

            l_np = l_sub.float().cpu().numpy()
            for r_idx, it in enumerate(c_chunk):
                k = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[r_idx, :k], it["target"]))

    fitted_temps = [1.2, 1.2, 1.2]
    try:
        for qt in range(3):
            sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
            if sel:
                fitted_temps[qt] = fit_one_temp(sel)
        print(f"Fitted calibration temperatures (choice, score, noul): {[round(t, 3) for t in fitted_temps]}")
    except Exception as e:
        print(f"Temperature fitting fallback: {e}")

    # ── Save checkpoint ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    sd = {k: v.float().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(args.output_dir, "model.safetensors"))

    enc_save_dir = os.path.join(args.output_dir, "encoder")
    model.encoder.config.save_pretrained(enc_save_dir)
    tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-finetuned"
    cfg["temperature"] = fitted_temps
    with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nModel saved to {args.output_dir}")
    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
