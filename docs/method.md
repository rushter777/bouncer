# The Router That Earns Its Keep

**A practical pipeline for fine-tuning a small decision model (Laya) on your own tasks — including what failed, and why.**

*Helix Max — technical write-up, September 2026. This repo is the write-up and the pipeline.*

---

## 1. The problem

Big LLMs are expensive and slow for one job they're bad at anyway: **fast, consistent yes/no judgments.** "Does this task need terminal access?" "Is this answer high-stakes enough to double-check?" A 400M-parameter classifier can answer those in 40ms with better consistency than a 70B chat model — if it's trained on the right data.

This project built that classifier for a real workload: ~1,500 actual user tasks given to an AI assistant (file work, shell commands, web lookups, debugging). The stack: **Laya** (Apache-2.0 open decision model, 322–421M params) fine-tuned locally via RLCD, served behind a FastAPI sidecar, consulted by a stdlib-only hook with a hard fail-safe. Everything runs on a single Mac. No cloud, no API keys.

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

## 5. Repository structure

```
bouncer/
├── README.md                 # this write-up
├── docs/
│   ├── method.md             # §2 expanded, with the four-model comparison
│   ├── results.md            # §3 expanded: every eval, sweep, parity run
│   └── lessons.md            # teacher noise floor, buckets vs ordinal, baselines
├── pipeline/
│   ├── agreement_test.py     # teacher self-agreement gate (≥85%)
│   ├── label_corpus.py       # rubric-based teacher labeling (JSON-forced output)
│   ├── train_rlcd.py         # RLCD fine-tune (MPS, NaN-guarded, CLI)
│   ├── eval_laya.py          # accuracy + confusion vs majority baseline
│   └── sweep.py              # precision/coverage curves for flags
├── serving/
│   ├── sidecar.py            # FastAPI, health endpoint, env-config checkpoint
│   ├── hook.py               # stdlib client, fail-safe, env thresholds
│   └── shadow.py             # append-only flight recorder
├── tests/
│   └── test_hook.py          # 14 tests incl. fail-safe + advisory flag contract
├── data/
│   └── synthetic_example/    # toy corpus so the repo runs out of the box
│       # (real task corpora stay private — they contain your data)
├── checkpoints/              # .gitignored — models encode your corpus
└── LICENSE                   # MIT
```

**What stays private:** real task corpora, shadow logs, trained checkpoints. They are *your data, shaped*. The repo ships the machine that makes your-data-shaped models — which is the whole point.

## 6. What's deliberately not built

- **Intake/routing** — the judgment exists and logs, but nothing routes prompts between models yet. Build it after shadow data proves the flags fire correctly in the wild.
- **Outcome labels** — the shadow log records judgments, not what happened next. Closing that loop (did the task actually need retries? did a human redo it?) is the path to labels that aren't anyone's opinion at all.
- **Automated retraining** — a nightly retrain is 88 minutes of GPU producing an identical model when the corpus hasn't changed. The right automation: accumulate new labeled rows, retrain only when there's something new to learn, swap only when the eval beats production. Data-triggered, not clock-triggered.

## 7. The four-model decomposition experiment

Asking four different large models, blind and independently, to "turn subjective difficulty into five objective yes/no questions" produced three convergent axes across all four (dependent multi-step, diagnosis-vs-procedure, open-ended done criteria) plus four unique contributions that survived into the final set. Orthogonality mattered: questions that are three rewordings of "is it complex?" give false confidence in a vote. The merged set, plus per-question contrastive examples in the rubric, is what the teacher labeled against — and the agreement test (§2.2) is what certified it.

## 8. Acknowledging the lineage

Laya is an open-weight (Apache 2.0) small decision model in the family popularized by Jev-style classifiers. JevBench-style leaderboards score these models on generic question suites. This project's core finding is that **leaderboard scores don't transfer to your tasks** — and the fix isn't a better base model, it's your own labels, tested for consistency, evaluated against the dumb strategy, and shipped honestly.
