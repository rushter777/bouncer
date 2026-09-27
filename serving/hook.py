"""
Laya hook — stdlib-only client for the Laya MLX sidecar.

Provides two public functions:

    judge(task_text) -> dict | None
        POSTs task_text to http://127.0.0.1:8777/predict with the fixed
        question set.  On ANY failure (connection refused, timeout, 5xx,
        bad JSON) returns None — never raises, never blocks the caller.

    apply(decisions) -> dict
        Describes the dispatch configuration derived from a judge()
        result.  Uses threshold 0.5 by default but reads overrides from
        env vars LAYA_THRESH_TOOLS and LAYA_THRESH_VERIFY.

        Output format:
            {
                "tools": {"file": bool, "terminal": bool, "web": bool},
                "needs_verification": bool,
                "raw": <raw probability dict>,
                "source": "laya" | "fallback",
            }

        When judge() returns None, apply() returns the FAIL-SAFE config:
        all tools True (assume everything), needs_verification=True,
        source='fallback'.

NOTE on thresholds:
    The noul head has a +0.03 systematic offset vs PyTorch outputs.
    Thresholds must be tuned on MLX outputs before deployment.
"""

import json
import os
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Fixed question set (must match the sidecar's QUESTION_SET)
# ---------------------------------------------------------------------------

# EXACT question text from the training corpus (must match sidecar's copy).
QUESTION_SET = {
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
    # Difficulty flag (v3d, Sept 26): score-type count of objective demands.
    # Advisory only — nothing gates on it. P(hard) = sum of levels 1-4;
    # flag fires at >= LAYA_THRESH_HARD (default 0.8, ~81% precision at 6% coverage
    # on the v3d eval). Meaningful only when the sidecar runs laya-v3d-mlx;
    # a v2q sidecar answers this out-of-distribution (pre-swap noise).
    "difficulty": {
        "type": "score",
        "instructions": "Rate how demanding this task is for an assistant, as a count of objective demands: dependent multi-step, discovery/diagnosis, design choice, execution verification.",
        "criteria": ["0-none", "1-one demand", "2-two demands", "3-three demands", "4-four demands"],
    },
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SIDECAR_URL = os.environ.get("LAYA_SIDECAR_URL", "http://127.0.0.1:8777/predict")
_TIMEOUT_S = 2


def judge(task_text: str) -> dict | None:
    """Send task_text to the Laya sidecar and return the answers dict.

    Returns None on any failure (connection refused, timeout, HTTP error,
    malformed JSON).  Never raises and never blocks the caller.
    """
    try:
        body = json.dumps({"state": task_text, "questions": QUESTION_SET}).encode("utf-8")
        req = urllib.request.Request(
            _SIDECAR_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            if resp.status != 200:
                return None
            raw = json.loads(resp.read().decode("utf-8"))
            return raw.get("answers", raw)
    except Exception:
        return None


def apply(decisions: dict | None) -> dict:
    """Derive dispatch config from a judge() result.

    Args:
        decisions: The dict returned by judge(), or None on failure.

    Returns:
        A dict with keys "tools", "needs_verification", "raw", "source".
    """
    # Default thresholds (can be overridden via env vars)
    tools_threshold = float(os.environ.get("LAYA_THRESH_TOOLS", "0.5"))
    verify_threshold = float(os.environ.get("LAYA_THRESH_VERIFY", "0.5"))

    if decisions is None:
        # FAIL-SAFE: assume everything, needs verification
        return {
            "tools": {"file": True, "terminal": True, "web": True},
            "needs_verification": True,
            "difficulty_flag": False,   # advisory; no data => no flag
            "p_hard": None,
            "raw": {},
            "source": "fallback",
        }

    raw = decisions.get("answers", decisions) if isinstance(decisions, dict) and "answers" in decisions else decisions

    def _noul(qid):
        # answers are dicts like {"type":"noul","noul":0.31,"confidence":...};
        # tolerate a bare scalar if the sidecar shape ever changes.
        a = raw.get(qid, 0)
        if isinstance(a, dict):
            return float(a.get("noul", 0.0))
        try:
            return float(a)
        except (TypeError, ValueError):
            return 0.0

    tools = {
        "file": _noul("tools_file") >= tools_threshold,
        "terminal": _noul("tools_terminal") >= tools_threshold,
        "web": _noul("tools_web") >= tools_threshold,
    }

    needs_verify = _noul("needs_verification") >= verify_threshold

    # Difficulty advisory flag — P(hard) from the score question's probabilities.
    # Never blocks anything; consumed only by the shadow log (and, someday, the
    # escalation rule layer). Missing/malformed difficulty answer => None/False.
    hard_threshold = float(os.environ.get("LAYA_THRESH_HARD", "0.8"))
    p_hard = None
    difficulty_flag = False
    diff_ans = raw.get("difficulty")
    if isinstance(diff_ans, dict):
        probs = diff_ans.get("probabilities")
        if isinstance(probs, dict):
            # sidecar keys by level index ("0"); corpus-style keys use labels
            p_none = probs.get("0", probs.get("0-none"))
            if p_none is not None:
                try:
                    p_hard = 1.0 - float(p_none)
                    difficulty_flag = p_hard >= hard_threshold
                except (TypeError, ValueError):
                    pass

    return {
        "tools": tools,
        "needs_verification": needs_verify,
        "difficulty_flag": difficulty_flag,   # ADVISORY ONLY — do not gate on this
        "p_hard": p_hard,                     # raw probability, None if unavailable
        "raw": raw,
        "source": "laya",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python laya_hook.py 'task description'", flush=True)
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    result = judge(task)
    config = apply(result)
    print(json.dumps(config, indent=2), flush=True)
