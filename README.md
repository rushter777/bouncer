# The Router That Earns Its Keep

![Banner — the bouncer at the door](assets/banner.png)

**A practical pipeline for fine-tuning a small decision model (Laya) on your own tasks — including what failed, and why.**

*Helix Max — technical write-up, September 2026. This repo is the pipeline described below.*

---

## Contents

- [1. The problem](#1-the-problem)
- [2. The method (the part worth stealing)](#2-the-method-the-part-worth-stealing)
- [3. The scoreboard](#3-the-scoreboard)
- [4. Serving architecture](#4-serving-architecture)
  - [4.1 The router (Bouncer proxy)](#41-the-router-bouncer-proxy--and-how-to-see-who-answered)
  - [4.2 The savings dashboard](#42-the-savings-dashboard)
- [5. Repository structure](#5-repository-structure)
- [6. Status and open problems](#6-status-and-open-problems)
- [7. The four-model decomposition experiment](#7-the-four-model-decomposition-experiment)
- [8. Acknowledging the lineage](#8-acknowledging-the-lineage)

---

## 1. The problem

Big LLMs are expensive and slow for one job they're bad at anyway: **fast, consistent yes/no judgments.** "Does this task need terminal access?" "Is this answer high-stakes enough to double-check?" A 400M-parameter classifier can answer those in 40ms with better consistency than a 70B chat model — if it's trained on the right data.

This project built that classifier for a real workload: ~1,500 actual user tasks given to an AI assistant (file work, shell commands, web lookups, debugging). The stack: **Laya** (Apache-2.0 open decision model, 322–421M params) fine-tuned locally via RLCD, served behind a FastAPI sidecar, consulted by a stdlib-only hook with a hard fail-safe. Everything runs on one Mac Studio. No cloud, no API keys.

## 2. The method (the part worth stealing)

The single most important lesson: **subjective questions are untrainable; objective questions are easy.** Everything else in this document is a consequence of that.

### 2.1 Decompose subjective → objective

"Is this task difficult?" is an opinion — two readers disagree, the training labels disagree with themselves, and the model learns noise. Instead, decompose into text-visible facts:

1. **multi_step** — does the task require ≥2 sequential actions where later ones depend on earlier results?
2. **discovery** — must the assistant first find an unknown (which file, which bug), rather than execute a named procedure?
3. **design** — must it choose among plausible approaches?
4. **exec_verify** — does "done" require execution/testing to prove it works?

(This exact set was derived independently by four different AI models asked blind — Gemini, ChatGPT, Grok, and a fourth assistant — and three axes converged across all four. See §7.)

A difficulty score is then the **count of yes answers (0–4)**, not a majority vote. Ordinal counts beat named buckets by a wide margin in our experiments: an ordinal "rate 1–5" formulation scored 71% where the equivalent "trivial/moderate/complex" bucket question scored 31%.

### 2.2 Gate on teacher self-agreement — before labeling anything

You will label your corpus with a bigger model (a "teacher"). Before spending hours on labeling, run this test: **ask the teacher the same question twice on ~100 real tasks (temperature > 0) and measure agreement.**

- ≥85% self-agreement → the question is labelable
- <85% → the question is an opinion, not a fact — cut it or reformulate

In our run: multi_step 96%, design 96%, exec_verify 93%, discovery 92% — all passed. "Is there more than one reasonable interpretation of done?" came in at **79% — cut**, per the rule set before the test. No renegotiating the bar after the numbers come in.

This test costs an afternoon and saved us from training on noise twice.

### 2.3 Train, then eval against the dumb strategy

RLCD fine-tuning (proper scoring rules + policy gradient) worked fine on ~1,200 rows on a single machine (MPS, ~88 min). The trainer is the easy part.

The eval bar matters more: **the baseline is "always answer the majority class."** For a difficulty distribution that's 77% routine, a model must beat 77% — not "random chance," not "the leaderboard," the *majority class*. A router question is only useful if it beats the strategy that costs zero compute.

### 2.4 When you can't beat the average, sweep for precision

Exact prediction tied the baseline (78.0% vs 77.6%) — because most tasks genuinely are easy, and the hard ones are rare. The fix: stop asking for exact scores and sweep the **precision/coverage curve**. Flag a task as hard only when P(hard) ≥ 0.8:

| Cutoff | Coverage | Precision |
|---|---|---|
| P ≥ 0.8 | 6.3% | **81.2%** |
| P ≥ 0.7 | 10.6% | 77.8% |

Against a 22% base rate, that's 3.7× the dumb strategy on flagged tasks. **An imprecise question became a deployable high-precision flag** — which is the operationally useful shape anyway ("which tasks deserve escalation-consideration?").

### 2.5 Ship advisory-first, retire honestly

- The flag ships **advisory-only**: logged, shadow-recorded, nothing gates on it.
- Questions that never beat their baseline get **retired with evidence**, not nursed along. Our "tier" question (which model should handle this task — local vs cloud) failed every formulation: 3-class 29%, binary 59.6% (vs 64.7% baseline), teacher self-agreement 81%, and a post-training probability head that output a constant. It became a **rule layer** computed from the questions that do work — difficulty flag + tool gates + escalation-on-measured-failure. Independent benchmark literature (LLMRouterBench, ACL 2026) reaches the same conclusion: most learned routers lose to "best cheap model + escalate on failure." We reproduced that result from scratch, on our own data, before finding theirs.

## 3. The scoreboard

| Question | First honest eval | Final state |
|---|---|---|
| tools_file | 78.0% | 79.6% — production gate |
| tools_terminal | 86.3% | 85.5% — production gate |
| tools_web | 87.1% | 85.1% — production gate |
| needs_verification | 72.2% | 72.9% — production gate |
| difficulty (buckets) | 31% — *below random* | retired, reformulated |
| difficulty (0–4 count) | 78.0% (tie vs 77.6%) | **precision flag: 81.2% @ 6.3% coverage** |
| tier | 29% | retired → rule layer |

Gate accuracy was verified equivalent between checkpoints in a head-to-head on 255 labeled tasks (323.1 vs 323.6 summed points — a statistical wash) before any production swap.

## 4. Serving architecture

```
prompt → hook (stdlib, 2s timeout)
          → sidecar (FastAPI, loopback :8777, MLX, ~40ms)
          → judgment dict {tools, verification, difficulty flag}
          → shadow log (append-only JSONL, never raises)
fail-safe: ANY failure → all tools exposed, verification on, source=fallback
```

The fail-safe is the load-bearing wall: the classifier may be wrong, dead, or slow — the assistant must never notice. Verified by killing the sidecar mid-operation.

### 4.1 The router (Bouncer proxy) — and how to see who answered

The judgment layer ships with an OpenAI-compatible **proxy** (`serving/proxy.py`) that does the actual routing: point any harness's `base_url` at it and easy prompts are answered by your local model, hard ones pass to the frontier model, tool-carrying prompts are judged too (simple tool work stays local; deep chains go to the frontier). The proxy is harness-agnostic — anything that speaks the OpenAI wire format can sit in front of it.

Every routed reply carries a **tag** so the human always knows who answered. Tags are config, not code — the defaults describe one machine's stack, yours will differ:

```bash
BOUNCER_TAG_LOCAL="[🟢 my-local-model]"   # green: local model answered
BOUNCER_TAG_ESC="[⚡ my-frontier-model]"   # yellow: escalated to frontier
# icons-only works too: BOUNCER_TAG_LOCAL="🟢" BOUNCER_TAG_ESC="⚡"
```

Two hard-won tag rules, learned from the first real session:

1. **Tags are for the human, not the model.** The proxy strips every tag line from conversation history before forwarding upstream. Without this, the model sees tagged history, starts imitating the tag (double-tagging), and — asked about routing — confabulates explanations about a layer it cannot see.
2. **Never hardcode model names.** Multi-frontier setups change the escalation target per workload (`BOUNCER_ESC_MODEL`); a tag that names a model goes stale the first time you swap. Icons + a naming scheme you control beats echoing whatever tonight's stack was.

### 4.2 The savings dashboard

Open the proxy's root URL (`http://127.0.0.1:8788/`) and you get a live dashboard: estimated dollars saved (tokens answered locally × your frontier's price), route mix, actual frontier spend, and the last ten decisions with their p_hard scores. One env var to set your price:

```bash
BOUNCER_FRONTIER_PRICE_PER_MTOK=3.0   # blended $/1M tokens of your frontier model
```

The estimate is honest about its basis: it counts only usage captured from upstream responses (the log line includes the token counts), and the page says so. It starts counting from when you deployed usage logging — it does not invent history.

## 5. Repository structure

```
bouncer/
├── README.md                 # this write-up
├── assets/
│   └── banner.png            # the bouncer at the door (README header)
├── docs/
│   └── method.md             # §2 expanded, with the four-model comparison
├── pipeline/
│   ├── agreement_test.py     # teacher self-agreement gate (≥85%)
│   ├── tier_agreement_test.py# same gate, applied to a tier question (it failed — see §3)
│   ├── label_corpus.py       # rubric-based teacher labeling (JSON-forced output)
│   ├── train_rlcd.py         # RLCD fine-tune (MPS, NaN-guarded, CLI)
│   ├── eval_laya.py          # accuracy + confusion vs majority baseline
│   └── sweep.py              # precision/coverage curves for flags
├── serving/
│   ├── sidecar.py            # FastAPI, health endpoint, env-config checkpoint
│   ├── hook.py               # stdlib client, fail-safe, env thresholds
│   ├── shadow.py             # append-only flight recorder
│   ├── proxy.py              # Bouncer: the routing switch + tags (see §4.1)
│   └── dashboard.html        # savings dashboard served at the proxy root
├── tests/
│   └── test_hook.py          # 14 tests incl. fail-safe + advisory flag contract
├── data/
│   └── synthetic/            # toy corpus so the repo runs out of the box
│       └── generate.py       # generator for the toy corpus
│       # (real task corpora stay private — they contain your data)
├── eval/
│   └── cold_metrics.json     # output of the cold-run smoke test (synthetic
│       # corpus, toy scale — proves the pipeline runs, NOT real accuracy;
│       # real numbers live in §3 and come from private eval data)
├── examples/
│   └── demo_batch.py         # batch-judge a JSONL of tasks
├── checkpoints/              # .gitignored — models encode your corpus
└── LICENSE                   # MIT
```

**What stays private:** real task corpora, shadow logs, trained checkpoints. They are *your data, shaped*. The repo ships the machine that makes your-data-shaped models — which is the whole point.

## 6. Status and open problems

**This repo works as-is.** The full chain — generate a toy corpus, train, calibrate, evaluate, serve, and a discriminating hook test — was verified end-to-end from a clean clone of this exact tree (training run: 64 seconds on Apple Silicon, 14/14 hook tests passing). No private files needed: it runs out of the box on the included synthetic corpus.

**And it's running for real.** The full stack is live on the author's machine — the proxy as a persistent service, wired into a test profile of his assistant (not the daily driver), routing real traffic: easy prompts to the local model, flagged-hard prompts to the frontier model, tags on every reply, all decisions logged. First real session ran from a phone, with green (local) and yellow (escalated) tags landing on the correct prompts and a live savings dashboard. Fail-open throughout: if the judge or the proxy dies, chat degrades to the plain local model, it never breaks.

**Open problems — ideas welcome.** These are deliberate gaps, not oversights. If you're playing with the repo and have ideas (or want to PR a fix), these are the ones that matter:

- **Outcome labels** — the shadow log records judgments, not what happened next. Closing that loop (did the task actually need retries? did a human redo it?) is the path to labels that aren't anyone's opinion at all. Ideas for cheap, honest outcome capture on a live proxy are the single most valuable contribution this repo could receive.
- **Automated retraining** — a nightly retrain is 88 minutes of GPU producing an identical model when the corpus hasn't changed. The right automation: accumulate new labeled rows, retrain only when there's something new to learn, swap only when the eval beats production. Data-triggered, not clock-triggered. Triggers and eval-gates that make that safe are an open design question.

## 7. The four-model decomposition experiment

Asking four different large models, blind and independently, to "turn subjective difficulty into five objective yes/no questions" produced three convergent axes across all four (dependent multi-step, diagnosis-vs-procedure, open-ended done criteria) plus four unique contributions that survived into the final set. Orthogonality mattered: questions that are three rewordings of "is it complex?" give false confidence in a vote. The merged set, plus per-question contrastive examples in the rubric, is what the teacher labeled against — and the agreement test (§2.2) is what certified it.

## 8. Acknowledging the lineage

Laya is an open-weight (Apache 2.0) small decision model in the family popularized by Jev-style classifiers. JevBench-style leaderboards score these models on generic question suites. This project's core finding is that **leaderboard scores don't transfer to your tasks** — and the fix isn't a better base model, it's your own labels, tested for consistency, evaluated against the dumb strategy, and shipped honestly.
