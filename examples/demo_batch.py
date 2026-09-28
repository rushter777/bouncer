#!/usr/bin/env python3
"""Demo batch: run a representative task spread through the live pipeline
(judge -> apply -> shadow log), then print the resulting shadow_log entries.

Runs against the sidecar at LAYA_SIDECAR_URL (default http://127.0.0.1:8777/predict).
Nothing is blocked or routed — this is the exact path an intake hook would use.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving import hook as laya_hook  # noqa: E402

TASKS = [
    # (task text, expected family — for YOUR eyes, not Laya's)
    ("What's the difference between TCP and UDP?", "pure QA, no tools"),
    ("Check the free RAM on this machine and tell me what's using the most", "read-only system check"),
    ("Look up the latest version of llama.cpp and tell me what's new", "web lookup"),
    ("Add a comment header to the top of the project README explaining what the project is", "single file edit"),
    ("Find why the log rotation job keeps failing and fix it", "discovery/debugging"),
    ("Design an intake system that routes prompts between local and cloud models based on difficulty", "design/open-ended"),
    ("Run the test suite and paste the results", "terminal, known procedure"),
    ("The integration tests fail after the last commit. Diagnose and fix, make the suite pass", "multi-step: diagnose -> fix -> verify"),
    ("Convert the dataset's row counts into a markdown table", "transform/format"),
    ("Check today's USD/JPY rate and write it into a note", "web + file, chained"),
]

def main():
    print(f"Running {len(TASKS)} tasks through the live pipeline...\n")
    entries = []
    for text, family in TASKS:
        j = laya_hook.judge(text)
        cfg = laya_hook.apply(j)
        # feed the full hook output (with task text) to the shadow logger
        payload = dict(cfg)
        payload["task_text"] = text
        p = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serving", "shadow.py")],
            input=json.dumps(payload) + "\n",
            capture_output=True, text=True)
        if p.returncode != 0:
            print(f"  shadow logger error on: {text[:40]}")
        entries.append((text, family, cfg))

    print("=" * 100)
    print(f"{'p_hard':>6} {'flag':>5} {'file':>5} {'term':>5} {'web':>4} {'verify':>6}  task (expected family)")
    print("-" * 100)
    for text, family, cfg in entries:
        t = cfg["tools"]
        print(f"{cfg['p_hard'] if cfg['p_hard'] is None else round(cfg['p_hard'],2)!s:>6} "
              f"{str(cfg['difficulty_flag']):>5} "
              f"{str(t['file']):>5} {str(t['terminal']):>5} {str(t['web']):>4} "
              f"{str(cfg['needs_verification']):>6}  {text[:48]} ({family})")
    print("=" * 100)
    print(f"\nsource on all: {[e[2]['source'] for e in entries][0]}")
    print("Entries appended to data/shadow_log.jsonl — the real file, exactly as intake would write it.")


if __name__ == "__main__":
    main()
