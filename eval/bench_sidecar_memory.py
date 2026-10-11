#!/usr/bin/env python3
"""Reproduction benchmark: memory behavior of the Laya MLX sidecar inference
loop across 100 requests — does Metal-side fragmentation require an explicit
GC sweep after batch predictions?

Mirrors serving/sidecar.py's loop exactly: one agent loaded once, one
predict() per request under serialized access, the sidecar's exact QUESTION_SET.
No FastAPI layer — that would measure uvicorn, not the inference loop.

Phases (each N requests, sampled every 10):
  A   varied-length task texts, NO explicit sweep   -> leak/fragmentation probe
  A2  identical task text,      NO explicit sweep   -> shape-control: fixed
                                                       buffer sizes should be
                                                       reused, not accumulated
  B   varied-length task texts, mx.clear_cache()+gc.collect() every batch
                                                       -> does the sweep help,
                                                       and what does it cost?

Metrics per sample:
  rss      process resident set (ps, MB)          wired-in RAM including Metal
  active   mx.metal.get_active_memory()  (MB)     live Metal buffers (weights+transients)
  peak     mx.metal.get_peak_memory()    (MB)     high-water mark since last reset
  cache    mx.metal.get_cache_memory()   (MB)     MLX allocator's free-buffer pool
  pyobj    len(gc.get_objects())                  Python object count (host-side leak proxy)
  lat      mean ms/request over the last batch

Verdict logic (printed at the end):
  - Python-side leak       : pyobj or active grows monotonically in phase A
  - Fragmentation          : cache/RSS grows in A but not in A2
  - Sweep required         : phase B keeps RSS/cache flat where A did not
  - Sweep cost             : latency delta A vs B

Env:
  LAYA_MODEL_PATH  checkpoint (default: repo checkpoints/cold-test)
  N_REQUESTS       requests per phase (default 100)

Stdlib + mlx + laya_mlx only.
"""

import gc
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL_PATH = os.environ.get("LAYA_MODEL_PATH", os.path.join(REPO, "checkpoints", "cold-test"))
N = int(os.environ.get("N_REQUESTS", "100"))
SAMPLE_EVERY = 10

import mlx.core as mx  # noqa: E402
import laya_mlx as laya  # noqa: E402

# EXACT question text from serving/sidecar.py — paraphrases shift the
# probability distribution, so the benchmark must use the real set.
QUESTION_SET = {
    "tools_file": {
        "type": "noul",
        "instructions": "Does this task involve reading or editing files?",
        "criteria": {"true": "needs file access to do", "false": "no files involved"},
    },
    "tools_terminal": {
        "type": "noul",
        "instructions": "Does this task involve running shell commands or system operations?",
        "criteria": {"true": "needs a terminal or system tools", "false": "no shell needed"},
    },
    "tools_web": {
        "type": "noul",
        "instructions": "Does this task involve looking things up online or fetching web content?",
        "criteria": {"true": "needs web search or fetching", "false": "no web access needed"},
    },
    "needs_verification": {
        "type": "noul",
        "instructions": "Does an answer to this task need double-checking before it is used?",
        "criteria": {"true": "the stakes are high or errors would cost something",
                     "false": "a quick answer is fine and low-stakes"},
    },
}


def rss_mb():
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout.strip()
    return (int(out) / 1024.0) if out else -1.0


def metal_stats():
    a = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    p = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    c = getattr(mx, "get_cache_memory", None) or mx.metal.get_cache_memory
    return {
        "active": a() / 2**20,
        "peak": p() / 2**20,
        "cache": c() / 2**20,
    }


def sample(tag, lat_ms=None):
    m = metal_stats()
    row = {
        "tag": tag,
        "rss_mb": round(rss_mb(), 1),
        "active_mb": round(m["active"], 1),
        "peak_mb": round(m["peak"], 1),
        "cache_mb": round(m["cache"], 1),
        "pyobj": len(gc.get_objects()),
    }
    if lat_ms is not None:
        row["lat_ms"] = round(lat_ms, 1)
    print(json.dumps(row), flush=True)
    return row


WORDS = ("rotate the weekly log archive then verify the cron entry fired "
         "read config edit deploy search fetch file shell terminal web url "
         "summarize translate debug trace refactor lint format build test").split()


def make_tasks(n, varied=True, seed=42):
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        if varied:
            length = rng.randint(8, 320)
        else:
            length = 60
        tasks.append(" ".join(rng.choice(WORDS) for _ in range(length)))
    return tasks


def run_phase(agent, tag, tasks, sweep_every=None):
    """Run len(tasks) predictions; sample every SAMPLE_EVERY; optional sweep."""
    rows, lats = [], []
    t_last = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        result = agent.predict(task, QUESTION_SET)
        lats.append((time.perf_counter() - t_last) * 1000.0)
        t_last = time.perf_counter()
        if sweep_every and i % sweep_every == 0:
            mx.clear_cache()
            gc.collect()
        if i % SAMPLE_EVERY == 0:
            rows.append(sample(f"{tag}:{i}", statistics.mean(lats[-SAMPLE_EVERY:])))
    return rows


def drift(rows, key):
    """Signed growth from first to last sample of a phase."""
    if len(rows) < 2:
        return 0.0
    return rows[-1][key] - rows[0][key]


def main():
    print(f"# model={MODEL_PATH} n_per_phase={N}", flush=True)
    agent = laya.load(MODEL_PATH)
    # Warm-up mirrors sidecar startup.
    agent.predict("warmup task", QUESTION_SET)
    if hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()
    sample("load")

    phase_a = run_phase(agent, "A-varied-nosweep", make_tasks(N, varied=True))
    if hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()
    phase_a2 = run_phase(agent, "A2-identical-nosweep", make_tasks(N, varied=False))

    # What does each half of a sweep release, on already-accumulated state?
    pre = sample("postA2-pre-sweep")
    mx.clear_cache()
    post_cc = sample("post-clear_cache")
    gc.collect()
    sample("post-gc_collect")
    if hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()

    phase_b = run_phase(
        agent, "B-varied-sweep10", make_tasks(N, varied=True, seed=99), sweep_every=SAMPLE_EVERY
    )

    # ---------------- verdict ----------------
    def flat(x, tol):
        return abs(x) <= tol

    rss_a, rss_a2, rss_b = drift(phase_a, "rss_mb"), drift(phase_a2, "rss_mb"), drift(phase_b, "rss_mb")
    act_a = drift(phase_a, "active_mb")
    cache_a, cache_a2, cache_b = (
        drift(phase_a, "cache_mb"),
        drift(phase_a2, "cache_mb"),
        drift(phase_b, "cache_mb"),
    )
    pyobj_a = drift(phase_a, "pyobj")
    released_by_cc = pre["cache_mb"] - post_cc["cache_mb"]

    print("\n# --- verdict inputs ---")
    print(f"# pyobj drift phase A       : {pyobj_a:+d} objects")
    print(f"# active drift phase A      : {act_a:+.1f} MB")
    print(f"# cache drift  A / A2 / B    : {cache_a:+.1f} / {cache_a2:+.1f} / {cache_b:+.1f} MB")
    print(f"# rss drift    A / A2 / B    : {rss_a:+.1f} / {rss_a2:+.1f} / {rss_b:+.1f} MB")
    print(f"# clear_cache released      : {released_by_cc:.1f} MB of pooled buffers")
    lat_a = phase_a[-1].get("lat_ms", 0)
    lat_b = phase_b[-1].get("lat_ms", 0)
    print(f"# latency last batch A vs B : {lat_a:.1f} ms vs {lat_b:.1f} ms")

    python_leak = (not flat(pyobj_a, 2000)) or (not flat(act_a, 10.0))
    fragmentation = (not flat(cache_a, 10.0)) and flat(cache_a2, 10.0)
    sweep_fixes = flat(rss_b, 10.0) and (not flat(rss_a, 10.0))

    print("\n# --- verdict ---")
    print(f"# python-side leak            : {'YES' if python_leak else 'NO'}")
    print(f"# metal-side fragmentation    : {'YES' if fragmentation else 'NO'}"
          f" (varied-shape grows, fixed-shape {'flat' if flat(cache_a2, 10.0) else 'grows'})")
    print(f"# explicit sweep required     : {'YES' if sweep_fixes else 'NO'}")
    if not sweep_fixes:
        print("#   (allocator either self-manages, or RSS growth is outside the cache's control)")


if __name__ == "__main__":
    main()
