"""Synthetic example corpus — toy data so the pipeline runs cold.

These are 40 fabricated tasks (NOT real user data) with the four objective
difficulty sub-questions labeled. Use them to smoke-test the pipeline:
agreement gate → labeling → training (1 epoch) → eval → sweep.

Scale notes: the real project used ~1,245 real tasks; 40 rows is enough to
exercise the code paths, not enough to train anything meaningful. Don't
read accuracy numbers off this file.

Format: one JSON object per line, {"state": <task text>, "questions": {...}, "labels": {...}}
matching the format pipeline/train_rlcd.py and pipeline/eval_laya.py expect.
"""
import json
import random
import sys

random.seed(7)

TASKS = [
    ("What's the difference between TCP and UDP?", 0),
    ("Explain how a hash map works.", 0),
    ("What year did the Apollo program end?", 0),
    ("Summarize the plot of The Great Gatsby.", 0),
    ("What does the chmod command do?", 0),
    ("Rename the variable 'foo' to 'bar' in utils.py.", 1),
    ("Add a comment header to the top of main.py.", 1),
    ("Change the port number in config.json from 3000 to 8080.", 1),
    ("Delete the unused imports in app.py.", 1),
    ("Convert this CSV to JSON.", 1),
    ("Write a Python function that reverses a string. Show it in chat.", 0),
    ("Draft a haiku about databases.", 0),
    ("List the steps to deploy a Docker container.", 0),
    ("Look up the latest version of llama.cpp and tell me what's new.", 1),
    ("Check today's USD/JPY exchange rate.", 1),
    ("Read the README and summarize the install steps.", 1),
    ("Install the requests package with pip.", 1),
    ("Create a new git branch called feature-test.", 1),
    ("Write these five lines to notes.txt.", 1),
    ("Run the test suite and paste the results.", 1),
    ("Fetch the docs from that site, write a script against that API, and run its test suite.", 3),
    ("Find the config file, change the variable, run the tests, and fix any failures.", 3),
    ("Read package.json, bump the version, update the changelog, commit, and tag.", 2),
    ("Why does this single SQL query return duplicate rows? Explain and fix.", 2),
    ("The build breaks at the pytest step with an import error. Diagnose and fix it.", 3),
    ("Integration tests fail after the last commit. Find the root cause, fix it, and make the suite pass.", 3),
    ("Profile the slow endpoint and refactor the query to speed it up.", 3),
    ("Something in CI is red. Fix it.", 3),
    ("Design an authentication system for this application.", 2),
    ("Propose a caching strategy for our API.", 2),
    ("Review this architecture and suggest improvements.", 2),
    ("Refactor this module to be cleaner.", 2),
    ("Compare SQLite and Postgres for my use case and recommend one.", 2),
    ("Debug the memory spike on launch and fix the leak.", 3),
    ("Migrate the database schema without downtime.", 3),
    ("Set up a cron job that restarts the service if it dies.", 2),
    ("Wire this new endpoint into the existing auth middleware.", 2),
    ("Find why the sidecar keeps restarting and fix it.", 3),
    ("Check today's rate and write it into rate.txt, then convert the file to CSV.", 2),
    ("Write a script that validates every JSON file in the repo and reports the bad ones.", 2),
]

RUBRIC = {
    "multi_step": {"instructions": "Would a correct solution require at least two sequential actions where a later action depends on the result of an earlier one?",
                    "examples": {True: "'Find the config, change it, run tests, fix failures'",
                                 False: "'Rename foo to bar'"}},
    "discovery": {"instructions": "Must the assistant first find an unknown (which file, which bug) rather than execute a named procedure?",
                   "examples": {True: "'Something in CI is red, fix it'",
                                False: "'Run pytest and paste output'"}},
    "design": {"instructions": "Must the assistant choose among plausible approaches that could meaningfully affect the result?",
                "examples": {True: "'Design an authentication system'",
                             False: "'Convert this CSV to JSON'"}},
    "exec_verify": {"instructions": "Does successful completion require verifying the result by execution, testing, or external observation?",
                     "examples": {True: "'Fix this parser and make all tests pass'",
                                  False: "'Write a regex matching these examples'"}},
}

QUESTION_TEXT = {
    "difficulty": {
        "type": "score",
        "instructions": "Rate how demanding this task is for an assistant, as a count of objective demands: dependent multi-step, discovery/diagnosis, design choice, execution verification.",
        "criteria": ["0-none", "1-one demand", "2-two demands", "3-three demands", "4-four demands"],
    },
}

out = []
for text, score in TASKS:
    labels = {"difficulty": score}
    questions = dict(QUESTION_TEXT)
    out.append({"state": text, "questions": questions, "labels": labels})

path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/synthetic_corpus.jsonl"
with open(path, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(out)} rows to {path}")
