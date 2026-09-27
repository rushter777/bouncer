"""Shadow-mode logger for the Laya router.

Reads judgment JSON (one per line) from stdin or a file argument,
appends timestamped entries to /Users/apollo/laya-router/data/shadow_log.jsonl.

Fields written per entry:
    ts             – ISO-8601 UTC timestamp
    task_text      – raw task text from the judgment record
    tools_file     – list of file-tool probabilities (floats)
    tools_terminal – list of terminal-tool probabilities (floats)
    tools_web      – list of web-tool probabilities (floats)
    needs_verification – boolean derived from judgment fields

Stdlib only. Never raises. Never modifies anything outside the log file.
"""

import json
import os
import sys
from datetime import datetime, timezone

LOG_DIR = "/Users/apollo/laya-router/data"
LOG_FILE = os.path.join(LOG_DIR, "shadow_log.jsonl")


def _ensure_dir():
    """Create the log directory if it does not exist."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError:
        pass


def _extract_prob(record, qid):
    """Pull the raw noul probability for *qid* from a judgment record.

    Handles both shapes we produce:
      - hook full output:  {"tools": {...}, "raw": {"tools_file": {"noul": p}}}
      - bare answers map:  {"tools_file": {"noul": p, "type": "noul", ...}}
      - legacy scalar:     {"tools_file": p}
    Returns None if not found (entry fields may then be null).
    """
    for source in (record.get("raw"), record, record.get("answers")):
        if not isinstance(source, dict):
            continue
        val = source.get(qid)
        if isinstance(val, dict):
            p = val.get("noul")
            if isinstance(p, (int, float)) and not isinstance(p, bool):
                return float(p)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


def _extract_task_text(record):
    """Pull the best-effort task text from a judgment record."""
    for key in ("task_text", "task", "query", "prompt"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return ""


def _extract_p_hard(record):
    """Pull P(hard) from a hook output: top-level p_hard, or derive from raw
    difficulty probabilities (1 - p('0-none')). Returns None if unavailable."""
    for source in (record, record.get("raw") if isinstance(record.get("raw"), dict) else None):
        if not isinstance(source, dict):
            continue
        if isinstance(source.get("p_hard"), (int, float)) and not isinstance(source.get("p_hard"), bool):
            return float(source["p_hard"])
        diff = source.get("difficulty")
        if isinstance(diff, dict):
            probs = diff.get("probabilities")
            if isinstance(probs, dict) and "0-none" in probs:
                try:
                    return 1.0 - float(probs["0-none"])
                except (TypeError, ValueError):
                    return None
    return None


def _process_record(record):
    """Build a shadow-log entry from one judgment JSON object."""
    source = record.get("source", "unknown")
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_text": _extract_task_text(record),
        "tools_file": _extract_prob(record, "tools_file"),
        "tools_terminal": _extract_prob(record, "tools_terminal"),
        "tools_web": _extract_prob(record, "tools_web"),
        "needs_verification": _extract_prob(record, "needs_verification"),
        "p_hard": _extract_p_hard(record),
        "source": source,   # 'laya' | 'fallback' | sidecar model name if present
    }


def main():
    _ensure_dir()

    # Determine input source: first argv (file path) or stdin
    if len(sys.argv) > 1:
        try:
            fh = open(sys.argv[1], "r", encoding="utf-8")
        except OSError:
            return  # Silently ignore unreadable file arguments
    else:
        fh = sys.stdin

    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # Skip malformed lines silently
            if not isinstance(record, dict):
                continue
            try:
                entry = _process_record(record)
                with open(LOG_FILE, "a", encoding="utf-8") as out:
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass  # Never raise — silently drop bad entries
    finally:
        if fh is not sys.stdin:
            try:
                fh.close()
            except OSError:
                pass


if __name__ == "__main__":
    main()
