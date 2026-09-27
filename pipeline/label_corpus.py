#!/usr/bin/env python3
"""Label full v2q corpus with the 4-question difficulty rubric (post-agreement-test).

Questions that passed the >=85% gate: multi_step, discovery, design, exec_verify.
(done_open CUT at 79% agreement.) Single pass — teacher agreement proven at 92-96%.
Labels written as a new difficulty_v3 field; raw JSON per task to data/difficulty_v3_labels.jsonl.
"""

import json
import random
import sys
import time
import urllib.request

API = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = os.environ.get("BOUNCER_TEACHER_MODEL", "http://127.0.0.1:8081/v1/chat/completions")  # point at any OpenAI-compatible local server
DATA = "data/router_v2q_train.jsonl"
EVAL = "data/router_v2q_eval.jsonl"
OUT = "data/difficulty_v3_labels.jsonl"

RUBRIC = """You are labeling a task that a user gave to an AI assistant (which has file, shell, and web tools). Answer 4 yes/no questions using ONLY the request text. Answer strictly as JSON.

q1 multi_step: Would a correct solution require at least two sequential actions where a later action depends on the result of an earlier one? (Two independent asks in one message do NOT count. "Write this function in chat" = false. "Find the config file, change the variable, run tests, fix failures" = true.)

q2 discovery: Must the assistant first find an unknown (which file, which bug, why it fails, which source is authoritative) rather than execute a procedure the user already named? ("Run pytest and paste output" = false. "Something in CI is red, fix it" = true.)

q4 design: Must the assistant choose among plausible approaches that could meaningfully affect the result (not merely fill in trivial defaults)? ("Convert this CSV to JSON" = false. "Design an authentication system for this app" = true.)

q5 exec_verify: Does successful completion require verifying the result by execution, testing, external observation, or comparison against an objective criterion — is a plausible-looking answer alone insufficient? ("Write a regex matching these examples" (regex only requested) = false. "Fix this parser and make all tests pass" = true.)

Respond with ONLY this JSON, no other text:
{"q1": true|false, "q2": true|false, "q4": true|false, "q5": true|false}

Task text:
"""


def ask(task_text, retries=3):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": RUBRIC + task_text}],
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 60,
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
            vals = {}
            for q in ("q1", "q2", "q4", "q5"):
                if not isinstance(obj.get(q), bool):
                    raise ValueError(f"bad {q}")
                vals[q] = obj[q]
            return vals
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2)


def label_file(path, tag):
    rows = [json.loads(l) for l in open(path)]
    out = open(OUT, "a")
    done = 0
    for i, r in enumerate(rows):
        text = r.get("state") or r.get("task_text") or ""
        if not text:
            continue
        vals = ask(text)
        if vals is None:
            print(f"  [{tag}{i}] parse failure, skipped", flush=True)
            continue
        score = sum(vals.values())  # 0-4
        out.write(json.dumps({"src": tag, "idx": i, "task": text[:200],
                              "answers": vals, "difficulty_v3": score}) + "\n")
        out.flush()
        done += 1
        if done % 10 == 0:
            print(f"  {tag}: {done}/{len(rows)} labeled", flush=True)
    out.close()
    print(f"{tag} complete: {done}/{len(rows)}")


if __name__ == "__main__":
    label_file(DATA, "train")
    label_file(EVAL, "eval")
    print("ALL DONE")
