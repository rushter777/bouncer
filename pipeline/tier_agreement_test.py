#!/usr/bin/env python3
"""Tier agreement test (difficulty protocol, applied to tier).

Binary formulation: local (the local model can complete it alone) vs escalate
(needs a bigger/cloud model). 100 sample tasks, 2 passes each, temp 0.7.
Gate: >=85% self-agreement to justify consensus relabeling.
Teacher: any OpenAI-compatible local server (BOUNCER_TEACHER_URL, default
http://127.0.0.1:8081/v1/chat/completions; model name via BOUNCER_TEACHER_MODEL).
"""

import json
import os
import random
import sys
import time
import urllib.request

API = os.environ.get("BOUNCER_TEACHER_URL", "http://127.0.0.1:8081/v1/chat/completions")
MODEL = os.environ.get("BOUNCER_TEACHER_MODEL", "local-model")
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic/synthetic_corpus.jsonl"
OUT = os.environ.get("BOUNCER_TIER_OUT", "data/tier_agreement.jsonl")

RUBRIC = """You are labeling a task that a user gave to an AI assistant. The assistant runs on a single local machine and uses a local small model (strong at routine coding, file work, shell tasks, and lookups; weak at long multi-step debugging chains, deep architectural reasoning, and very obscure domain knowledge). Cloud models are stronger but cost money and leak data.

Question: judging only from the task text, can the local model complete this task well on its own?

Answer yes if the task is within the local model's demonstrated strengths: routine code edits, file reads/writes, standard shell commands, straightforward lookups, simple scripts, formatting, questions answerable from the text or a quick search.
Answer no if the task plausibly needs more: multi-system debugging with unknown causes, deep design decisions, safety-critical changes, or heavily specialized domain expertise.

Respond with ONLY this JSON, no other text:
{"local_ok": true|false}

Task text:
"""


def ask(task_text, retries=3):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": RUBRIC + task_text}],
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 30,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            raw = resp["choices"][0]["message"]["content"].strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1:
                raise ValueError("no JSON")
            obj = json.loads(raw[start:end + 1])
            if not isinstance(obj.get("local_ok"), bool):
                raise ValueError("bad field")
            return obj["local_ok"]
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows = [json.loads(l) for l in open(DATA)]
    random.seed(42)
    sample = random.sample(rows, min(n, len(rows)))
    print(f"Asking {len(sample)} tasks x 2 passes...", flush=True)

    results = []
    for i, r in enumerate(sample):
        text = r.get("state") or r.get("task_text") or ""
        if not text:
            continue
        a1 = ask(text)
        a2 = ask(text)
        if a1 is None or a2 is None:
            continue
        results.append({"task": text[:200], "pass1": a1, "pass2": a2})
        if (i + 1) % 10 == 0:
            agree = sum(1 for x in results if x["pass1"] == x["pass2"])
            print(f"  {i+1}/{len(sample)} done | agreement so far {agree}/{len(results)} = {agree/len(results):.1%}", flush=True)

    with open(OUT, "w") as f:
        for x in results:
            f.write(json.dumps(x) + "\n")

    agree = sum(1 for x in results if x["pass1"] == x["pass2"])
    yes_rate = sum(1 for x in results if x["pass1"]) / len(results)
    verdict = "PASS" if agree / len(results) >= 0.85 else "FAIL"
    print(f"\n=== FINAL: {len(results)} tasks ===")
    print(f"  tier(local_ok) self-agreement: {agree}/{len(results)} = {agree/len(results):.1%}  local-yes-rate {yes_rate:.0%}  [{verdict} >=85%]")


if __name__ == "__main__":
    main()
