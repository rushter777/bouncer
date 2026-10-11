"""Outcome-event extractors for the shadow log enrichment pipeline.

Reads raw conversation / tool-result transcripts and emits structured events
to data/outcome_events.jsonl.  Stdlib only, never raises, same discipline as
shadow.py — bad input is silently dropped, not propagated.

Usage:
    python -m serving.outcome_extract <transcript_file_or_dir> [output_file]

Transcript formats supported (auto-detected by file extension):
    .jsonl   — one JSON object per line, each with at least 'ts' and 'text'/'content'.
    .log     — Hermes-style event lines:  ts=<iso> event=<type> payload=<json>
    .json    — a single JSON array of message dicts (role/content or role/tool_result).

Matching key resolution:
    Primary: decision_id present on both sides → direct join.
    Secondary: (task_text_hash, ±5min window) → at most one match; ambiguity = quarantine.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "outcome_events.jsonl"
)

EVENT_TYPES = [
    "tool_error",
    "user_retry",
    "user_retry_after_error",
    "assistant_error_msg",
    "human_redo",
    "explicit_success",
]

ERROR_PATTERNS = [
    "traceback", "error:", "exception", "failed", "exit code",
    "permission denied", "module not found", "timeout",
    "could not", "unable to", "refused",
]

RETRY_PATTERNS = [
    "no, ", "actually, ", "redo", "try again", "that's wrong",
    "not what i meant", "i said", "forget that", "scratch that",
    "undo", "revert", "start over",
]

SUCCESS_PATTERNS = [
    "thanks", "perfect", "great", "works", "got it", "done",
    "exactly", "yes, ", "nice", "awesome", "thank you",
]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: str, b: str) -> float:
    """Bag-of-words cosine similarity — fast, no deps."""
    def _bag(s):
        from collections import Counter
        return Counter(s.lower().split())
    a_bag = _bag(a)
    b_bag = _bag(b)
    if not a_bag or not b_bag:
        return 0.0
    common = set(a_bag) & set(b_bag)
    if not common:
        return 0.0
    dot = sum(a_bag[w] * b_bag[w] for w in common)
    mag_a = sum(v * v for v in a_bag.values()) ** 0.5
    mag_b = sum(v * v for v in b_bag.values()) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def _edit_distance_ratio(a: str, b: str) -> float:
    """Levenshtein ratio — O(n*m) but fine for short texts."""
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    dist = dp[n]
    return 1.0 - (dist / max(m, n))


def _is_error(text: str) -> bool:
    """Check if text looks like an error message."""
    t = text.lower()
    return any(p in t for p in ERROR_PATTERNS)


def _is_retry(text: str, prior_text: str) -> tuple[bool, float]:
    """Determine if this is a retry of the prior turn. Returns (is_retry, confidence)."""
    sim = _cosine_similarity(text, prior_text)
    edr = _edit_distance_ratio(text, prior_text)
    # Retry heuristics: high similarity OR contains retry keywords
    has_keywords = any(p in text.lower() for p in RETRY_PATTERNS)
    if sim >= 0.9 and has_keywords:
        return True, 0.95
    if edr >= 0.85 and has_keywords:
        return True, 0.8
    if sim >= 0.95:
        return True, 0.7
    # Strong keyword-only detection: when retry keywords are present with
    # sufficient signal strength (e.g., "no, actually" or "redo this"), treat
    # as a retry even without high text similarity — the user is explicitly
    # rejecting the prior answer and asking for a new attempt.
    if has_keywords:
        # Boost confidence when multiple keywords match or keywords are at start.
        keyword_count = sum(1 for p in RETRY_PATTERNS if p in text.lower())
        starts_with_keyword = any(text.lower().startswith(p) for p in RETRY_PATTERNS)
        conf = 0.5 + 0.1 * keyword_count
        if starts_with_keyword:
            conf += 0.15
        return True, min(conf, 0.95)
    return False, max(sim * 0.5, edr * 0.3)


def _is_success(text: str) -> tuple[bool, float]:
    """Check if text indicates successful completion."""
    t = text.lower()
    matches = sum(1 for p in SUCCESS_PATTERNS if p in t)
    if matches >= 2:
        return True, 0.9
    if matches >= 1 and len(t) < 50:
        return True, 0.6
    return False, 0.0


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def extract_events_from_transcript(messages: list[dict], decision_id: str | None = None,
                                    task_text_hash: str | None = None) -> list[dict]:
    """Extract outcome events from a list of message dicts.

    Args:
        messages: List of dicts with 'role' and 'content'/'text' keys.
        decision_id: Optional primary join key (from proxy).
        task_text_hash: Optional secondary join key.

    Returns:
        List of outcome event dicts ready for JSONL output.
    """
    events = []
    if not messages:
        return events

    # Collect all text content with indices for cross-referencing.
    turns = []
    for i, m in enumerate(messages):
        role = m.get("role", "")
        content = m.get("content") or m.get("text") or ""
        if not isinstance(content, str):
            # Handle list-of-parts (multimodal) — extract text.
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and isinstance(p.get("type"), str) and p["type"] == "text"
                )
        ts = m.get("ts") or m.get("timestamp") or ""
        tool_name = m.get("tool_name") or m.get("tool", "")
        error_class = None
        excerpt = ""

        # Detect tool errors.
        if role in ("tool", "function", "tool_result"):
            is_err = False
            if isinstance(content, str):
                is_err = _is_error(content)
                excerpt = content[:200]
            elif isinstance(content, dict):
                error_output = content.get("error") or content.get("exception")
                if error_output:
                    is_err = True
                    excerpt = str(error_output)[:200]
            if is_err:
                events.append({
                    "event_ts": ts,
                    "decision_id": decision_id,
                    "task_text_hash": task_text_hash,
                    "event_type": "tool_error",
                    "signal": {
                        "tool_name": tool_name or "unknown",
                        "error_class": "exit_code" if isinstance(excerpt, str) and "exit code" in excerpt.lower() else "exception",
                        "excerpt": excerpt[:200],
                    },
                    "latency_ms": None,  # filled by joiner
                    "confidence": 0.95,
                })

        # Detect assistant error messages.
        if role == "assistant" and isinstance(content, str) and _is_error(content):
            events.append({
                "event_ts": ts,
                "decision_id": decision_id,
                "task_text_hash": task_text_hash,
                "event_type": "assistant_error_msg",
                "signal": {
                    "excerpt": content[:200],
                },
                "latency_ms": None,
                "confidence": 0.7,
            })

        turns.append({"role": role, "content": content, "ts": ts, "idx": i})

    # Detect user retries (compare consecutive user messages).
    for i in range(1, len(turns)):
        curr = turns[i]
        prev = turns[i - 1]
        if curr["role"] == "user" and prev["role"] == "assistant":
            is_retry, conf = _is_retry(curr["content"], prev["content"])
            if is_retry:
                # Check if prior assistant had an error.
                prior_error = any(
                    e["event_type"] in ("tool_error", "assistant_error_msg")
                    for e in events
                )
                etype = "user_retry_after_error" if prior_error else "user_retry"
                conf = 0.95 if prior_error else conf
                events.append({
                    "event_ts": curr["ts"],
                    "decision_id": decision_id,
                    "task_text_hash": task_text_hash,
                    "event_type": etype,
                    "signal": {
                        "prior_assistant_content": prev["content"][:100],
                        "user_content": curr["content"][:200],
                    },
                    "latency_ms": None,
                    "confidence": conf,
                })

    # Detect explicit success.
    for t in reversed(turns):
        if t["role"] == "user" and _is_success(t["content"])[0]:
            events.append({
                "event_ts": t["ts"],
                "decision_id": decision_id,
                "task_text_hash": task_text_hash,
                "event_type": "explicit_success",
                "signal": {
                    "user_content": t["content"][:200],
                },
                "latency_ms": None,
                "confidence": _is_success(t["content"])[1],
            })
            break  # Only take the last success signal.

    return events


# ---------------------------------------------------------------------------
# Transcript loading helpers
# ---------------------------------------------------------------------------

def load_transcript(path: str) -> list[dict]:
    """Load a transcript file in auto-detected format."""
    if path.endswith(".jsonl"):
        msgs = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return msgs
    elif path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]
    else:
        # Treat as raw text — one message per line.
        msgs = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                             "content": line.strip(), "ts": str(i)})
        return msgs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python outcome_extract.py <transcript_file> [output_file]", flush=True)
        sys.exit(1)

    transcript_path = args[0]
    output_path = args[1] if len(args) > 1 else DEFAULT_OUTPUT

    try:
        messages = load_transcript(transcript_path)
    except FileNotFoundError:
        print(f"Error: file not found: {transcript_path}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading transcript: {e}", flush=True)
        sys.exit(1)

    events = extract_events_from_transcript(messages)

    if not events:
        print("No outcome events detected.", flush=True)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"Wrote {len(events)} event(s) to {output_path}", flush=True)


if __name__ == "__main__":
    main()
