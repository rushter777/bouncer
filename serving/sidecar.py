"""
Laya MLX sidecar — FastAPI server on 127.0.0.1:8777.

Loads the Laya v2q-mlx model once at startup, guards every prediction
with a threading.Lock (single forward pass at a time), and serves a
/health endpoint plus a /predict POST endpoint.
"""

import json
import threading
import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "LAYA_MODEL_PATH",
    # v3d-mlx (Sept 27): gate-accuracy parity with v2q proven head-to-head
    # (323.1 vs 323.6 summed accuracy — a wash), plus a real difficulty head.
    # Swap back = change this path back to laya-v2q-mlx2.
    "checkpoints/laya-v3d-mlx",
)

# EXACT question text from the training corpus — paraphrased criteria shift the
# probability distribution (measured: paraphrase dropped gate firing 3x).
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
}

# ---------------------------------------------------------------------------
# Model loading (once, under lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_agent = None
_load_error: str | None = None

def _load_model():
    """Load the Laya model into global _agent. Must be called under _lock."""
    global _agent, _load_error
    try:
        import laya_mlx as laya
        _agent = laya.load(MODEL_PATH)
        _load_error = None
    except Exception as exc:
        _agent = None
        _load_error = str(exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Laya MLX Sidecar")


@app.on_event("startup")
def startup():
    """Load model at startup. If it fails, /health returns ok=false."""
    _load_model()
    # Warm-up predict if model loaded successfully
    if _agent is not None:
        try:
            _agent.predict(
                "warmup task",
                QUESTION_SET,
            )
        except Exception:
            pass  # warm-up failure is non-fatal


@app.get("/health")
def health():
    """Return model load status."""
    if _agent is not None:
        return {"ok": True, "model": MODEL_PATH}
    return JSONResponse(
        status_code=503,
        content={"ok": False, "model": MODEL_PATH, "error": _load_error or "model failed to load"},
    )


@app.post("/predict")
def predict(payload: dict):
    """Run a single forward pass through the Laya model."""
    if _agent is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not loaded"},
        )

    state = payload.get("state", "")
    questions = payload.get("questions", QUESTION_SET)

    try:
        with _lock:
            result = _agent.predict(state, questions)
        # predict() returns {"answers": {qid: {...}}, "usage": {...}} — surface
        # just the answers map under the same key (no double nesting).
        return {"answers": result.get("answers", result)}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


def main():
    port = int(os.environ.get("LAYA_SIDECAR_PORT", "8777"))
    uvicorn.run(app, host="127.0.0.1", port=port)


@app.on_event("shutdown")
def shutdown():
    """Graceful shutdown hook."""
    global _agent
    _agent = None


if __name__ == "__main__":
    main()
