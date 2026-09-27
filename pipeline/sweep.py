#!/usr/bin/env python3
"""Threshold sweep for v3d difficulty: precision/coverage of flagging score>=1 tasks.

Mirrors eval_laya.py's loading + inference, but extracts the difficulty question's
per-level probabilities and computes P(score>=1) = sum of probs for levels 1..4.
Then sweeps cutoffs.
"""

import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from eval_laya import _fix_tokenizer_config, build_model, load_jsonl, QTYPES
from laya.common import build_sequence, collate_items

CKPT = "checkpoints/laya-v3d"
DATA = "data/router_v3d_eval.jsonl"


def main():
    device = torch.device("mps")
    _fix_tokenizer_config(CKPT)
    cfg = json.load(open(os.path.join(CKPT, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(CKPT, "tokenizer"))
    enc_dir = os.path.join(CKPT, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    model.load_state_dict(load_file(os.path.join(CKPT, "model.safetensors")), strict=True)
    model.to(device).eval()
    temp_score = cfg.get("temperature", [1, 1, 1])[1]
    print(f"Model loaded. score temperature={temp_score:.3f}")

    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)

    rows = load_jsonl(DATA)
    print(f"{len(rows)} eval tasks")

    probs_hard, golds = [], []
    bs = 16
    for row in rows:
        state = row["state"]
        if isinstance(state, str) and state[:1] in ("{", "["):
            state = json.loads(state)
        q = row["questions"]["difficulty"]
        crit = q["criteria"] if isinstance(row["questions"]["difficulty"]["criteria"], list) else list(q["criteria"].keys())
        n_opts = len(crit)
        seq, markers = build_sequence(
            tok, state,
            {"t": q["type"], "ins": q.get("instructions", ""),
             "crit": ({c: None for c in crit} if isinstance(crit, list) else q["criteria"])},
            max_len, head_max_len,
        )
        if len(markers) != n_opts:
            continue
        cb = collate_items([[{"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]]}]], tok.pad_token_id)
        with torch.no_grad():
            logits, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                              cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                              cb["qtype"].to(device))
        lg = logits.float().cpu().numpy()[0, :n_opts] / temp_score
        p = np.exp(lg - lg.max())
        p = p / p.sum()
        probs_hard.append(float(p[1:].sum()))  # P(score>=1)
        golds.append(row["labels"]["difficulty"])

    probs = np.array(probs_hard)
    golds = np.array(golds)
    hard = golds >= 1
    print(f"\nGold: {int(hard.sum())} hard / {len(golds)} tasks ({hard.mean():.0%})")
    print(f"Model P(hard) distribution: mean={probs.mean():.3f} median={np.median(probs):.3f} max={probs.max():.3f}")
    print(f"\n{'cutoff':>7} {'flagged':>8} {'coverage':>9} {'precision':>10} {'recall':>8}")
    for c in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        flagged = probs >= c
        cov = flagged.mean()
        prec = (flagged & hard).sum() / flagged.sum() if flagged.sum() else float("nan")
        rec = (flagged & hard).sum() / hard.sum() if hard.sum() else float("nan")
        print(f"{c:>7.2f} {int(flagged.sum()):>8} {cov:>9.1%} {prec:>10.1%} {rec:>8.1%}")
    np.save("data/v3d_sweep_probs.npy", np.vstack([probs, golds]))
    print("\nSaved probs+golds to data/v3d_sweep_probs.npy")


if __name__ == "__main__":
    main()
