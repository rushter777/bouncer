# Outcome-Capture Strategy for the Shadow Log
**Target:** `serving/shadow.py` + `serving/proxy.py` (routing decision site)  
**Goal:** connect what *actually happened after* a routing decision back to the decision record, so you can measure "did the local tier get it wrong?" per decision — the training signal Laya currently lacks.

---

## 1. The core problem: no join key

Today a shadow entry has `task_text` + `ts` and nothing that survives into the *next* turn. Follow-up signals (retries, tool failures, errors) happen seconds-to-minutes later, in a different process (the worker/agent), with no reference to the decision that routed them. Text-matching task_text is fragile (paraphrased retries, common prompts like "hello").

**Fix: a `decision_id` minted at routing time and threaded through everything downstream.**

- `proxy.py` mints `decision_id = uuid4().hex[:12]` at the moment it logs the judgment to shadow.
- It injects the id as an HTTP header (`X-Decision-Id: ...`) or system-prompt stanza into the worker request, whichever surface the worker runtime exposes.
- Every outcome event *emitted later* carries the same id. Joining is `decision_id` equality — O(1), no heuristics.

Fallback for surfaces that can't thread the header: `(task_text_hash, 5-min bucket)` as a weak secondary key, used only for offline joins, never trusted as ground truth.

## 2. Record schema

### Shadow decision record (extend `_process_record` output):
```json
{
  "ts": "2026-10-11T02:14:33Z",
  "decision_id": "a3f9c2e11b07",
  "task_text": "...",
  "task_text_hash": "sha256:9f2c...",
  "routed_to": "local",                  // NEW: local | frontier
  "tools_file": 0.93, "tools_terminal": 0.29, "tools_web": null,
  "needs_verification": true,
  "p_hard": 0.79,
  "execution_outcome": null,             // still null at write time — filled by joiner (§4)
  "source": "laya"
}
```

### Outcome event record (new file, `data/outcome_events.jsonl`):
```json
{
  "event_ts": "2026-10-11T02:14:41Z",
  "decision_id": "a3f9c2e11b07",         // REQUIRED — events without it are dropped
  "event_type": "tool_error",            // see taxonomy below
  "signal": {
    "tool_name": "terminal",
    "error_class": "exit_code",          // exit_code | timeout | exception | permission
    "excerpt": "ModuleNotFoundError: mx..."   // first 200 chars, redacted
  },
  "latency_ms": 8214,                    // ms after the decision ts
  "confidence": 0.9                      // how sure the extractor is (see §3)
}
```

### Event taxonomy (extraction rules)
| event_type | trigger | confidence heuristic |
|---|---|---|
| `user_retry` | user sends near-identical message (cosine ≥ 0.9 or edit-distance ≤ 15% of length) within 120s of previous assistant completion | 0.8 — retries can also be "do another one" |
| `user_retry_after_error` | retry AND prior reply contained error signals | 0.95 |
| `tool_error` | tool result with non-zero exit / exception / timeout marker | 0.95 |
| `assistant_error_msg` | reply matches error patterns (`Traceback`, `failed`, `I couldn't`, `exit code`) | 0.7 — natural language, fuzzy |
| `human_redo` | user explicitly says "no, do it like X" / rewrites the task with negation of previous approach | 0.6 — weakest, keep but downweight |
| `explicit_success` | user says thanks/confirms OR session ends without any error/retry within a grace window | 0.7 |

**The event extractor is a separate, cheap consumer** — a post-hoc hook on the worker transcript stream (Hermes session events / tool results), stdlib-only, never raises, same discipline as shadow.py.

## 3. Matching key resolution order

1. `decision_id` present on both sides → direct join. Done.
2. Event missing `decision_id` → try `(task_text_hash, ±5min ts window)`. At most one shadow record must match; ambiguity → event quarantined to `data/outcome_unmatched.jsonl`, not force-joined.
3. No match → quarantine. **Never guess a join for training data** — a wrong label is worse than a missing one (this is the same principle as the 85% agreement gate).

## 4. Asynchronous resolution — ledger + joiner, zero blocking

The design principle: **shadow.py stays write-only and dumb.** The active request path never waits for outcome data — outcomes arrive after the reply is already delivered.

```
proxy (decision time)                worker/agent (after the fact)
   │  writes decision record            │  emits outcome events
   │  to shadow_log.jsonl               │  (retry detector, tool-error
   ▼                                    ▼  hook, error-pattern matcher)
data/shadow_log.jsonl              data/outcome_events.jsonl
        └────────────┬─────────────────┘
                     ▼        (async, offline)
              joiner (cron / sidecar tick):
              for each decision with execution_outcome == null:
                  - collect events with matching decision_id
                  - apply precedence: tool_error > user_retry_after_error
                    > assistant_error_msg > user_retry > explicit_success
                  - write backfill: append *updated* record to
                    data/shadow_log_enriched.jsonl (never rewrite history —
                    the original file is append-only and immutable)
                  - decisions with zero events after 30-min watermark →
                    outcome "success" *only if* explicit_success heuristic
                    fired; otherwise null ("unknown"), and null records
                    are excluded from training, not imputed
```

Key timing decisions:

- **30-minute watermark.** A conversation follow-up that arrives 30+ minutes after the decision is almost certainly about something else (topic change). Past the watermark, the decision's outcome is frozen — whatever it is, it is.
- **Precedence over recency.** One tool error beats a later polite "thanks" — the user thanking the *final* attempt after a retry chain is not success of the original decision.
- **`retried` state:** a `user_retry` event followed by *no* further error events marks the original decision as a probable miss (`retried`), because the user asking again means the first answer didn't land. That's the single highest-value signal for the router: "I sent this to local, the user resent it — wrong tier call."
- **Backpressure:** the joiner runs on a timer (cron tick or sidecar idle loop), reads only *new* lines (byte-offset checkpoint in a state file), and appends enriched records. Worst case it lags — nobody in the request path notices.

## 5. What this buys you in training terms

Once `shadow_log_enriched.jsonl` accumulates, the RLCD label pipeline gets a new, *objective* per-deision signal — immune to the subjective-ambiguity problem that forced the 85% agreement gate, because `user_retry` and `tool_error` are text-visible facts, not opinions:

- **Positive labels** (local tier handled it): success, no retry.
- **Negative labels** (should have escalated): `retried` (user re-asked), `tool_error` on tasks where the router had said tools-file at p>0.9 and routed local anyway.
- **Bandit-style credit** (frontier overkill): the inverse — p_hard was 0.95, task succeeded on local *shadow* inference anyway.

The joiner can additionally compute per-source regret (`source: laya` vs `source: fallback`) directly from the enriched file — that's your shadow-mode "would the router have gotten this right" metric, per tier, without ever touching production routing.

## 6. Implementation order

1. `decision_id` mint + header injection in `proxy.py` (5 lines).
2. `decision_id` + `task_text_hash` + `routed_to` into `_process_record` (3 lines).
3. Outcome-event extractors as standalone post-hoc hooks on worker transcripts (the bulk of the work — the retry-similarity matcher is the only nontrivial piece).
4. Joiner script + 30-min watermark + enriched output file (stdlib, ~150 lines).
5. Only after enrichment shows stable volume: wire into the RLCD label pipeline behind the same "never force-join" discipline.
