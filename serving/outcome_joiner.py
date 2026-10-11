"""Shadow-log joiner — async enrichment of decision records with outcome events.

Reads decisions from data/shadow_log.jsonl (or shadow_log_proxy.jsonl),
outcome events from data/outcome_events.jsonl, joins on decision_id or
(task_text_hash ±5min window), resolves via precedence rules, and writes
enriched copies to data/shadow_log_enriched.jsonl.

The original log files are NEVER rewritten — append-only discipline.
This is a cron / sidecar-tick process, never blocking the request path.

Usage:
    python -m serving.outcome_joiner [--shadow <file>] [--events <file>] [--output <file>] [--watermark-mins 30]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SHADOW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shadow_log.jsonl"
)
DEFAULT_EVENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "outcome_events.jsonl"
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shadow_log_enriched.jsonl"
)
STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "joiner_state.json"
)

# Precedence order: higher number wins. One tool_error beats many retries.
EVENT_PRECEDENCE = {
    "tool_error": 10,
    "user_retry_after_error": 9,
    "assistant_error_msg": 5,
    "human_redo": 3,
    "user_retry": 2,
    "explicit_success": 1,
}

# Confidence threshold: events below this are ignored.
MIN_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string."""
    if not ts_str:
        return None
    try:
        # Handle both 'Z' suffix and '+00:00'.
        s = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _ts_diff_minutes(ts_a: str, ts_b: str) -> float | None:
    """Return difference in minutes between two ISO timestamps. None if unparseable."""
    da = _parse_ts(ts_a)
    db = _parse_ts(ts_b)
    if da and db:
        return abs((da - db).total_seconds()) / 60.0
    return None


# ---------------------------------------------------------------------------
# State management (byte-offset checkpoint for resume-ability)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load joiner state from disk."""
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"shadow_offset": 0, "events_offset": 0, "last_run": None}


def save_state(state: dict):
    """Persist joiner state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# Core join logic
# ---------------------------------------------------------------------------

def _resolve_outcome(events: list[dict]) -> str | None:
    """Resolve a set of events to a single outcome using precedence rules.

    Precedence (highest wins):
        tool_error > user_retry_after_error > assistant_error_msg > human_redo > user_retry > explicit_success

    If the winning event is explicit_success and there are no higher-precedence
    events, outcome = "success". Otherwise outcome = the event type name.
    If no events pass confidence threshold, return None.
    """
    # Filter by confidence.
    valid = [e for e in events if e.get("confidence", 0) >= MIN_CONFIDENCE]
    if not valid:
        return None

    # Sort by precedence (descending), then by confidence (descending).
    valid.sort(key=lambda e: (EVENT_PRECEDENCE.get(e["event_type"], 0), e.get("confidence", 0)), reverse=True)
    winner = valid[0]
    etype = winner["event_type"]

    if etype == "explicit_success":
        return "success"
    elif etype == "tool_error":
        return "failed"
    elif etype in ("user_retry", "user_retry_after_error"):
        return "retried"
    elif etype == "human_redo":
        return "human_redo"
    else:
        return None


def _find_events_for_decision(decision_id: str | None, task_text_hash: str | None,
                               decision_ts: str, watermark_mins: int,
                               all_events: list[dict]) -> list[dict]:
    """Find outcome events matching a decision record.

    Resolution order:
        1. Primary: decision_id exact match (within watermark window).
        2. Secondary: task_text_hash ±5min window.
        3. No match → quarantine.
    """
    matched = []

    # Primary: decision_id.
    if decision_id:
        for e in all_events:
            if e.get("decision_id") == decision_id:
                diff = _ts_diff_minutes(decision_ts, e.get("event_ts", ""))
                if diff is not None and diff <= watermark_mins:
                    matched.append(e)

    # Secondary: task_text_hash ±5min.
    if not matched and task_text_hash:
        for e in all_events:
            if e.get("task_text_hash") == task_text_hash:
                diff = _ts_diff_minutes(decision_ts, e.get("event_ts", ""))
                if diff is not None and 5.0 >= diff >= 0:
                    matched.append(e)

    return matched


# ---------------------------------------------------------------------------
# Main joiner loop
# ---------------------------------------------------------------------------

def run_joiner(shadow_path: str = DEFAULT_SHADOW, events_path: str = DEFAULT_EVENTS,
               output_path: str = DEFAULT_OUTPUT, watermark_mins: int = 30):
    """Run the enrichment pipeline.

    Reads only new lines since last run (byte-offset checkpoint).
    Writes enriched records to a *separate* output file — originals untouched.
    """
    state = load_state()
    shadow_offset = state.get("shadow_offset", 0)
    events_offset = state.get("events_offset", 0)

    # Load new outcome events since last run.
    new_events = []
    try:
        with open(events_path, "r") as f:
            f.seek(events_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    new_events.append(evt)
                except json.JSONDecodeError:
                    pass
            events_offset = f.tell()
    except FileNotFoundError:
        pass

    # Load decisions to enrich.
    new_decisions = []
    try:
        with open(shadow_path, "r") as f:
            f.seek(shadow_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    new_decisions.append(rec)
                except json.JSONDecodeError:
                    pass
            shadow_offset = f.tell()
    except FileNotFoundError:
        print(f"Shadow log not found at {shadow_path} — nothing to enrich.", flush=True)
        return

    if not new_decisions and not new_events:
        print("Nothing new to process.", flush=True)
        save_state({"shadow_offset": shadow_offset, "events_offset": events_offset,
                     "last_run": datetime.now(timezone.utc).isoformat()})
        return

    # Build a lookup from existing enriched records (to avoid duplicates).
    enriched_ids = set()
    try:
        with open(output_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("decision_id"):
                        enriched_ids.add(rec["decision_id"])
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass

    # Enrich decisions with outcome events.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enriched_count = 0
    with open(output_path, "a", encoding="utf-8") as out:
        for rec in new_decisions:
            did = rec.get("decision_id")
            if did and did in enriched_ids:
                continue  # Already enriched.

            outcome_events = _find_events_for_decision(
                decision_id=did,
                task_text_hash=rec.get("task_text_hash"),
                decision_ts=rec.get("ts", ""),
                watermark_mins=watermark_mins,
                all_events=new_events,
            )

            # Resolve to a single outcome.
            outcome = _resolve_outcome(outcome_events)
            enriched_rec = dict(rec)
            enriched_rec["execution_outcome"] = outcome
            enriched_rec["_event_count"] = len(outcome_events)
            enriched_rec["_events_matched"] = [e.get("event_type") for e in outcome_events]

            out.write(json.dumps(enriched_rec, ensure_ascii=False) + "\n")
            enriched_ids.add(did)
            enriched_count += 1

    # Also write events that have no matching decision (quarantine).
    quarantine_path = os.path.join(os.path.dirname(output_path), "outcome_unmatched.jsonl")
    matched_decision_ids = set()
    for evt in new_events:
        if evt.get("decision_id"):
            matched_decision_ids.add(evt["decision_id"])
    unmatched = [e for e in new_events if not e.get("decision_id") or e["decision_id"] not in matched_decision_ids]
    if unmatched:
        with open(quarantine_path, "a", encoding="utf-8") as qf:
            for evt in unmatched:
                qf.write(json.dumps(evt, ensure_ascii=False) + "\n")

    # Persist state.
    save_state({
        "shadow_offset": shadow_offset,
        "events_offset": events_offset,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "enriched_count": enriched_count,
        "quarantined_events": len(unmatched),
    })

    print(f"Enriched {enriched_count} decision(s), quarantined {len(unmatched)} event(s).", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    shadow_path = DEFAULT_SHADOW
    events_path = DEFAULT_EVENTS
    output_path = DEFAULT_OUTPUT
    watermark_mins = 30

    i = 0
    while i < len(args):
        if args[i] == "--shadow" and i + 1 < len(args):
            shadow_path = args[i + 1]; i += 2
        elif args[i] == "--events" and i + 1 < len(args):
            events_path = args[i + 1]; i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]; i += 2
        elif args[i] == "--watermark-mins" and i + 1 < len(args):
            watermark_mins = int(args[i + 1]); i += 2
        else:
            print(f"Unknown arg: {args[i]}", flush=True)
            return
        i += 0

    run_joiner(shadow_path, events_path, output_path, watermark_mins)


if __name__ == "__main__":
    main()
