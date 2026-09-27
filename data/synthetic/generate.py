"""Synthetic example corpus — toy data so the pipeline runs cold.

These are 40 fabricated tasks (NOT real user data) with the full question
battery labeled: the four tool/verification gates plus the difficulty score.
Use them to smoke-test the pipeline: agreement gate -> labeling -> training
(1 epoch) -> eval -> sweep -> serve -> hook.

Scale notes: the real project used ~1,245 real tasks; 40 rows is enough to
exercise the code paths, not enough to train anything meaningful. Don't
read accuracy numbers off this file.

Format: one JSON object per line, matching what pipeline/train_rlcd.py and
pipeline/eval_laya.py expect.
"""
import json
import sys

# (task text, difficulty 0-4, needs_file, needs_terminal, needs_web, needs_verify)
# Hand-assigned labels for smoke-testing, not learned.
TASKS = [
    ("What's the difference between TCP and UDP?", 0, 0, 0, 0, 1),
    ("Explain how a hash map works.", 0, 0, 0, 0, 1),
    ("What year did the Apollo program end?", 0, 0, 0, 0, 0),
    ("Summarize the plot of The Great Gatsby.", 0, 0, 0, 0, 0),
    ("What does the chmod command do?", 0, 0, 0, 0, 1),
    ("Rename the variable 'foo' to 'bar' in utils.py.", 1, 1, 0, 0, 1),
    ("Add a comment header to the top of main.py.", 1, 1, 0, 0, 0),
    ("Change the port number in config.json from 3000 to 8080.", 1, 1, 0, 0, 1),
    ("Delete the unused imports in app.py.", 1, 1, 0, 0, 0),
    ("Convert this CSV to JSON.", 1, 0, 1, 0, 1),
    ("Write a Python function that reverses a string. Show it in chat.", 0, 0, 0, 0, 0),
    ("Draft a haiku about databases.", 0, 0, 0, 0, 0),
    ("List the steps to deploy a Docker container.", 0, 0, 0, 0, 1),
    ("Look up the latest version of llama.cpp and tell me what's new.", 1, 0, 0, 1, 1),
    ("Check today's USD/JPY exchange rate.", 1, 0, 0, 1, 1),
    ("Read the README and summarize the install steps.", 1, 1, 0, 0, 0),
    ("Install the requests package with pip.", 1, 0, 1, 0, 0),
    ("Create a new git branch called feature-test.", 1, 0, 1, 0, 0),
    ("Write these five lines to notes.txt.", 1, 1, 0, 0, 0),
    ("Run the test suite and paste the results.", 1, 0, 1, 0, 0),
    ("Fetch the docs from that site, write a script against that API, and run its test suite.", 3, 1, 1, 1, 1),
    ("Find the config file, change the variable, run the tests, and fix any failures.", 3, 1, 1, 0, 1),
    ("Read package.json, bump the version, update the changelog, commit, and tag.", 2, 0, 1, 0, 1),
    ("Why does this single SQL query return duplicate rows? Explain and fix.", 2, 1, 0, 0, 1),
    ("The build breaks at the pytest step with an import error. Diagnose and fix it.", 3, 1, 1, 0, 1),
    ("Integration tests fail after the last commit. Find the root cause, fix it, make the suite pass.", 3, 1, 1, 0, 1),
    ("Profile the slow endpoint and refactor the query to speed it up.", 3, 1, 1, 0, 1),
    ("Something in CI is red. Fix it.", 3, 1, 1, 0, 1),
    ("Design an authentication system for this application.", 2, 0, 0, 0, 1),
    ("Propose a caching strategy for our API.", 2, 0, 0, 0, 1),
    ("Review this architecture and suggest improvements.", 2, 0, 0, 0, 1),
    ("Refactor this module to be cleaner.", 2, 1, 0, 0, 1),
    ("Compare SQLite and Postgres for my use case and recommend one.", 2, 0, 0, 1, 1),
    ("Debug the memory spike on launch and fix the leak.", 3, 1, 1, 0, 1),
    ("Migrate the database schema without downtime.", 3, 1, 1, 0, 1),
    ("Set up a cron job that restarts the service if it dies.", 2, 0, 1, 0, 1),
    ("Wire this new endpoint into the existing auth middleware.", 2, 1, 0, 0, 1),
    ("Find why the sidecar keeps restarting and fix it.", 3, 1, 1, 0, 1),
    ("Check today's rate and write it into rate.txt, then convert the file to CSV.", 2, 1, 1, 1, 1),
    ("Write a script that validates every JSON file in the repo and reports the bad ones.", 2, 0, 1, 0, 1),
]

# EXACT question text the serving stack asks — serving/hook.py QUESTION_SET must
# match these byte-for-byte (paraphrased criteria shift probability distributions).
QUESTIONS = {
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
        "criteria": {"true": "the stakes are high or errors would cost something", "false": "a quick answer is fine and low-stakes"},
    },
    "difficulty": {
        "type": "score",
        "instructions": "Rate how demanding this task is for an assistant, as a count of objective demands: dependent multi-step, discovery/diagnosis, design choice, execution verification.",
        "criteria": ["0-none", "1-one demand", "2-two demands", "3-three demands", "4-four demands"],
    },
}


def main():
    out = []
    for text, diff, f, t, w, v in TASKS:
        labels = {
            "tools_file": f,
            "tools_terminal": t,
            "tools_web": w,
            "needs_verification": v,
            "difficulty": diff,
        }
        out.append({"state": text, "questions": QUESTIONS, "labels": labels})

    path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/synthetic_corpus.jsonl"
    with open(path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
