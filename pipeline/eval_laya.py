#!/usr/bin/env python3
"""Blind evaluation of a fine-tuned Laya checkpoint on JSONL eval data.

Computes per-question-id metrics: accuracy, soft accuracy, Brier score, ECE,
score MAE, and per-class confusion matrices — matching the notebook's metric
computation exactly.

Usage:
    python eval_laya.py --checkpoint checkpoints/laya-ft --data data/router_eval.jsonl \
                        --device mps
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import (
    QTYPES,
    QTYPE_NAMES,
    build_model,
    collate_items,
    confidence_from_probs,
    ece_score,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Metric accumulators ──────────────────────────────────────────────────────

class MetricsAccumulator:
    """Collects metrics per question id and aggregates at the end."""

    def __init__(self):
        self.accuracy = defaultdict(list)       # qid -> [0/1]
        self.soft_accuracy = defaultdict(list)   # qid -> [dot product]
        self.brier = defaultdict(list)           # qid -> [Brier score]
        self.score_mae = defaultdict(list)       # qid -> [MAE] (score type only)
        self.within_one = defaultdict(list)      # qid -> [0/1] (score type only)
        self.confidence = []                     # global list for ECE
        self.correct = []                        # global list for ECE

        # Confusion matrices per qtype: (qtype, true_label, pred_label) -> count
        self.confusion = defaultdict(int)
        self.label_sets = defaultdict(set)

    def record_choice(self, qid, pred_probs, gold_probs, pred_label, gold_label):
        """Record metrics for a choice-type question."""
        p = np.array(pred_probs)
        g = np.array(gold_probs)
        if p.sum() > 0:
            p = p / p.sum()
        if g.sum() > 0:
            g = g / g.sum()

        self.accuracy[qid].append(float(pred_label == gold_label))
        self.soft_accuracy[qid].append(float((p * g).sum()))
        self.brier[qid].append(float(((p - g) ** 2).sum()))

        conf = float(p.max())
        is_corr = float(pred_label == gold_label)
        self.confidence.append(conf)
        self.correct.append(is_corr)

        # Confusion matrix — per QUESTION, never pooled across qids
        self.label_sets[qid].add(gold_label)
        self.label_sets[qid].add(pred_label)
        self.confusion[(qid, gold_label, pred_label)] += 1

    def record_noul(self, qid, p_val, g_val, pred_label, gold_label):
        """Record metrics for a noul-type question."""
        p_dist = np.array([1.0 - p_val, p_val])
        g_dist = np.array([1.0 - g_val, g_val])

        self.accuracy[qid].append(float(pred_label == gold_label))
        self.soft_accuracy[qid].append(float((p_dist * g_dist).sum()))
        self.brier[qid].append(float(((p_dist - g_dist) ** 2).sum()))

        conf = float(max(p_val, 1.0 - p_val))
        is_corr = float(pred_label == gold_label)
        self.confidence.append(conf)
        self.correct.append(is_corr)

        self.label_sets[qid].add(gold_label)
        self.label_sets[qid].add(pred_label)
        self.confusion[(qid, gold_label, pred_label)] += 1

    def record_score(self, qid, p_score, g_score, p_probs, n_levels, pred_lvl, gold_lvl):
        """Record metrics for a score-type question."""
        self.score_mae[qid].append(abs(p_score - g_score))
        self.within_one[qid].append(float(abs(p_score - g_score) <= 1.0))

        p_arr = np.array(p_probs)
        if p_arr.sum() > 0:
            p_arr = p_arr / p_arr.sum()

        conf = float(p_arr.max()) if p_arr.sum() > 0 else 0.5
        is_corr = float(pred_lvl == gold_lvl)
        self.accuracy[qid].append(is_corr)
        self.confidence.append(conf)
        self.correct.append(is_corr)

        # Soft accuracy: dot product of predicted vs gold distribution
        g_arr = np.zeros(n_levels)
        g_arr[gold_lvl] = 1.0
        self.soft_accuracy[qid].append(float((p_arr * g_arr).sum()))
        self.brier[qid].append(float(((p_arr - g_arr) ** 2).sum()))

        self.label_sets["score"].add(str(gold_lvl))
        self.label_sets["score"].add(str(pred_lvl))
        self.confusion[("score", str(gold_lvl), str(pred_lvl))] += 1

    def summary(self):
        """Return aggregated metric dict."""
        result = {}

        # Global metrics
        if self.accuracy:
            all_acc = []
            for qid in self.accuracy:
                all_acc.extend(self.accuracy[qid])
            result["accuracy"] = float(np.mean(all_acc)) if all_acc else 0.0

            all_soft = []
            for qid in self.soft_accuracy:
                all_soft.extend(self.soft_accuracy[qid])
            result["soft_accuracy"] = float(np.mean(all_soft)) if all_soft else 0.0

            all_brier = []
            for qid in self.brier:
                all_brier.extend(self.brier[qid])
            result["brier_score"] = float(np.mean(all_brier)) if all_brier else 0.0

            if self.confidence and self.correct:
                result["ece"] = ece_score(np.array(self.confidence), np.array(self.correct))
            else:
                result["ece"] = float("nan")

        # Score MAE (score-type only)
        all_mae = []
        for qid in self.score_mae:
            all_mae.extend(self.score_mae[qid])
        result["score_mae"] = float(np.mean(all_mae)) if all_mae else 0.0

        # Within 1 level (score-type only)
        all_w1 = []
        for qid in self.within_one:
            all_w1.extend(self.within_one[qid])
        result["within_1_level"] = float(np.mean(all_w1)) if all_w1 else 0.0

        # Per-qid metrics
        result["per_qid"] = {}
        for qid in sorted(set(list(self.accuracy.keys()) + list(self.score_mae.keys()))):
            qid_metrics = {}
            if qid in self.accuracy:
                qid_metrics["accuracy"] = float(np.mean(self.accuracy[qid]))
                qid_metrics["count"] = len(self.accuracy[qid])
            if qid in self.soft_accuracy:
                qid_metrics["soft_accuracy"] = float(np.mean(self.soft_accuracy[qid]))
            if qid in self.brier:
                qid_metrics["brier_score"] = float(np.mean(self.brier[qid]))
            if qid in self.score_mae:
                qid_metrics["score_mae"] = float(np.mean(self.score_mae[qid]))
            result["per_qid"][qid] = qid_metrics

        # Confusion matrices
        result["confusion"] = dict(self.confusion)
        result["label_sets"] = {k: sorted(v) for k, v in self.label_sets.items()}

        return result


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Laya on JSONL eval data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned checkpoint directory")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to eval JSONL file")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cpu"], help="Device to run on")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Micro-batch size for inference")
    parser.add_argument("--output", type=str, default=None,
                        help="Save metrics JSON to this path")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    _fix_tokenizer_config(args.checkpoint)

    cfg_path = os.path.join(args.checkpoint, "rl_agent_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    tok = AutoTokenizer.from_pretrained(os.path.join(args.checkpoint, "tokenizer"))
    enc_dir = os.path.join(args.checkpoint, "encoder")
    encoder_dir = enc_dir if os.path.exists(enc_dir) else None

    model = build_model(cfg, encoder_dir=encoder_dir)
    weights_path = os.path.join(args.checkpoint, "model.safetensors")
    weights = load_file(weights_path)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    # Load calibrated temperatures from config
    temperature = cfg.get("temperature", [1.0, 1.0, 1.0])
    temperature_by_options = cfg.get("temperature_by_options", {})

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Temperatures: choice={temperature[0]:.3f}, score={temperature[1]:.3f}, noul={temperature[2]:.3f}")

    # ── Load eval data ─────────────────────────────────────────────────────
    data_rows = load_jsonl(args.data)
    print(f"Loaded {len(data_rows)} eval rows from {args.data}\n")

    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)

    from laya.common import build_sequence, render_options, temp_bucket

    metrics = MetricsAccumulator()
    latencies_ms = []

    t0_eval = time.perf_counter()

    # ── Evaluate each row ──────────────────────────────────────────────────
    for row_idx, row in enumerate(data_rows):
        state = row["state"]
        if isinstance(state, str) and state[:1] in ("{", "["):
            state = json.loads(state)
        questions = row["questions"]
        if isinstance(questions, str):
            questions = json.loads(questions)
        labels = row.get("labels", {})
        if isinstance(labels, str):
            labels = json.loads(labels)

        t_start = time.perf_counter()

        # Build batch items
        batch_items = []
        qid_order = list(questions.keys())

        for qid in qid_order:
            q = questions[qid]
            t = q["type"]
            crit = q.get("criteria")
            n_opts = len(crit) if isinstance(crit, (list, tuple)) else len(crit.keys()) if crit else 2

            seq, markers = build_sequence(
                tok, state,
                {"t": t, "ins": q.get("instructions", ""),
                 "crit": ({c: None for c in crit} if isinstance(crit, list) else crit)},
                max_len, head_max_len,
            )
            if len(markers) != n_opts:
                continue

            batch_items.append({
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[t],
                "_qid": qid,
            })

        if not batch_items:
            continue

        # Forward pass in micro-batches
        all_logits = []
        for b_start in range(0, len(batch_items), args.batch_size):
            b_chunk = batch_items[b_start:b_start + args.batch_size]
            cb = collate_items([b_chunk], tok.pad_token_id)
            if cb is None:
                continue

            with torch.no_grad():
                logits, _ = model(
                    cb["input_ids"].to(device),
                    cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device),
                    cb["marker_mask"].to(device),
                    cb["qtype"].to(device),
                )
            all_logits.append(logits.float().cpu().numpy())

        if not all_logits:
            continue

        logits_all = np.concatenate(all_logits, axis=0)

        # Process each prediction
        logit_idx = 0
        for b_start in range(0, len(batch_items), args.batch_size):
            b_chunk = batch_items[b_start:b_start + args.batch_size]
            n_in_batch = len(b_chunk)

            for r_idx, item in enumerate(b_chunk):
                qid = item["_qid"]
                k = len(item["markers"])
                qt = item["qtype"]
                z = logits_all[logit_idx, :k]
                logit_idx += 1

                # Apply temperature scaling
                bucket_key = temp_bucket(qt, k)
                t_scale = temperature_by_options.get(bucket_key, temperature[qt])
                z_calibrated = z / max(1e-3, float(t_scale))

                # Convert logits to probabilities
                p = np.exp(z_calibrated - z_calibrated.max())
                p = p / p.sum()

                qdef = questions[qid]
                gold = labels.get(qid)

                if gold is not None and qid in labels:
                    if qdef["type"] == "choice":
                        keys = list(qdef["criteria"].keys()) if isinstance(qdef["criteria"], dict) else list(qdef["criteria"])
                        pred_label = keys[int(p.argmax())]
                        gold_label = str(gold)

                        # Build gold probability distribution for soft metrics
                        g_probs = np.zeros(len(keys))
                        if gold_label in keys:
                            g_probs[keys.index(gold_label)] = 1.0

                        metrics.record_choice(qid, list(p), list(g_probs), pred_label, gold_label)

                    elif qdef["type"] == "noul":
                        p_val = float(p[1])  # probability of "true"
                        pred_label = "true" if p_val >= 0.5 else "false"
                        # normalize gold: ints 0/1, bools, and "true"/"false"/"1"/"0" all valid
                        try:
                            g_val = float(gold) if gold not in (None, "") else 0.0
                        except (TypeError, ValueError):
                            g_val = 1.0 if str(gold).lower() in ("true", "yes") else 0.0
                        gold_label = "true" if g_val >= 0.5 else "false"

                        metrics.record_noul(qid, p_val, g_val, pred_label, gold_label)

                    elif qdef["type"] == "score":
                        n_levels = len(qdef.get("criteria", [])) if isinstance(qdef.get("criteria"), list) else 4
                        p_score = float((np.arange(n_levels) * p).sum())
                        g_score = float(gold) if not isinstance(gold, str) else 0.0

                        pred_lvl = int(np.argmax(p))
                        try:
                            gold_lvl = int(float(gold))
                        except (ValueError, TypeError):
                            gold_lvl = 0

                        metrics.record_score(qid, p_score, g_score, list(p), n_levels, pred_lvl, gold_lvl)

        dt_ms = (time.perf_counter() - t_start) * 1000
        latencies_ms.append(dt_ms)

    total_time = time.perf_counter() - t0_eval

    # ── Print results ──────────────────────────────────────────────────────
    summary = metrics.summary()

    print("=" * 60)
    print("LAYA EVALUATION RESULTS")
    print("=" * 60)
    print(f"Cases evaluated: {len(data_rows)}")
    print(f"Total decisions: {sum(len(v) for v in metrics.accuracy.values())}")
    print(f"Eval time: {total_time:.1f}s ({np.mean(latencies_ms):.1f} ms/case)")
    print()
    print(f"  Accuracy:       {summary.get('accuracy', float('nan')):.4f}")
    print(f"  Soft Accuracy:  {summary.get('soft_accuracy', float('nan')):.4f}")
    print(f"  Brier Score:    {summary.get('brier_score', float('nan')):.4f}")
    print(f"  ECE:            {summary.get('ece', float('nan')):.4f}")
    print(f"  Score MAE:      {summary.get('score_mae', float('nan')):.4f}")
    print(f"  Within 1 Level: {summary.get('within_1_level', float('nan')):.4f}")

    # Per-qid breakdown
    print("\n--- Per-Question Metrics ---")
    for qid in sorted(summary.get("per_qid", {}).keys()):
        qm = summary["per_qid"][qid]
        acc = qm.get("accuracy", float('nan'))
        cnt = qm.get("count", 0)
        mae = qm.get("score_mae", float('nan'))
        line = f"  {qid:>25s}: accuracy={acc:.4f} (n={cnt})"
        if not np.isnan(mae):
            line += f"  score_mae={mae:.4f}"
        print(line)

    # Confusion matrices — per qid (keys are now qids, not types)
    print("\n--- Confusion Matrices ---")
    confusion = summary.get("confusion", {})
    label_sets = summary.get("label_sets", {})
    for qid in sorted(set(k[0] for k in confusion.keys())):
        type_conf = {(tl, pl): c for (q, tl, pl), c in confusion.items() if q == qid}
        if not type_conf:
            continue
        labels = sorted(label_sets.get(qid, []))
        print(f"\n  {qid}:")
        header = f"    {'':>18s}" + "".join(f"{l[:12]:>13s}" for l in labels)
        print(header)
        for tl in labels:
            row = f"    {tl[:18]:>18s}" + "".join(f"{type_conf.get((tl, pl), 0):>13d}" for pl in labels)
            print(row)

    # ── Save metrics ───────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        def _jsonable(o):
            if isinstance(o, tuple):
                return "|".join(str(x) for x in o)
            if isinstance(o, set):
                return sorted(o)
            raise TypeError(repr(o))
        with open(args.output, "w") as f:
            dumpable = dict(summary)
            if "confusion" in dumpable:
                dumpable["confusion"] = {f"{q}|{gl}|{pl}": c for (q, gl, pl), c in dumpable["confusion"].items()}
            json.dump(dumpable, f, indent=2, default=_jsonable)
        print(f"\nMetrics saved to {args.output}")

    return summary


if __name__ == "__main__":
    main()
