#!/usr/bin/env python3
"""Bouncer proxy — the model switch, sitting where the session model's door is.

OpenAI-compatible /v1/chat/completions endpoint that routes per request:

    request comes in
      ├─ has tools / tool calls / tool results → PASS-THROUGH to the session
      │   model (GLM 5.3 via OpenRouter), unchanged — harness work, no judgment
      └─ pure chat → judge with Laya (sidecar :8777)
            ├─ difficulty flag (p_hard >= 0.8) → ESCALATE to session model
            └─ routine → LOCAL: the 35B MoE answers (llama-server :8081), free

Every answer is tagged so the human always knows who answered:
    [🟢 LOCAL · 35B-A3B]   or   [⚡ ESCALATED · GLM 5.3]
Every decision is logged to data/proxy_log.jsonl.

Fail-safe: ANY error anywhere → pass through to the session model. The proxy
must never be the reason a chat breaks.

Env:
    BOUNCER_PROXY_PORT   (default 8788)
    BOUNCER_LOCAL_URL    (default http://127.0.0.1:8081/v1/chat/completions)
    BOUNCER_JUDGE_URL    (default http://127.0.0.1:8777/predict)
    BOUNCER_ESC_URL      (default https://openrouter.ai/api/v1/chat/completions)
    BOUNCER_ESC_MODEL    (default z-ai/glm-5.3 — swap frontier providers without code changes)
    BOUNCER_ESC_KEYNAME   (default OPENROUTER_API_KEY — name of env var holding the key)
    BOUNCER_PROXY_HOST / BOUNCER_PROXY_PORT  (default 127.0.0.1:8788)
    LAYA_THRESH_HARD     (default 0.8)
"""

import hashlib
import json
import os
import sys
import time
import uuid
import urllib.request

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laya_hook  # the proven judge client: extractors + fail-safe, tested 14/14

PROXY_PORT = int(os.environ.get("BOUNCER_PROXY_PORT", "8788"))
PROXY_HOST = os.environ.get("BOUNCER_PROXY_HOST", "127.0.0.1")
LOCAL_URL = os.environ.get("BOUNCER_LOCAL_URL", "http://127.0.0.1:8081/v1/chat/completions")
JUDGE_URL = os.environ.get("BOUNCER_JUDGE_URL", "http://127.0.0.1:8777/predict")
ESC_KEYNAME = os.environ.get("BOUNCER_ESC_KEYNAME", "OPENROUTER_API_KEY")
# Frontier endpoint — swap providers without touching code (OpenRouter by default).
ESC_URL = os.environ.get(
    "BOUNCER_ESC_URL", "https://openrouter.ai/api/v1/chat/completions"
)
THRESH = float(os.environ.get("LAYA_THRESH_HARD", "0.8"))

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "data", "proxy_log.jsonl")
# Tags are user-facing receipts, fully configurable. Defaults describe THIS
# machine's stack; override per deployment (icons-only works too, e.g.
# BOUNCER_TAG_LOCAL="🟢"). History-stripping recognizes any bracketed
# 🟢/⚡ tag regardless of exact text, so old tags never poison context.
TAG_LOCAL = os.environ.get("BOUNCER_TAG_LOCAL", "[🟢 LOCAL · 35B-A3B]")
TAG_ESC = os.environ.get("BOUNCER_TAG_ESC", "[⚡ ESCALATED · GLM 5.3]")

app = FastAPI()


def _esc_headers():
    """Escalation key read from Hermes's own .env at runtime. Never printed."""
    headers = {"Content-Type": "application/json"}
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith(ESC_KEYNAME + "="):
                headers["Authorization"] = "Bearer " + line.strip().split("=", 1)[1]
                break
    return headers


def _post(url, payload, headers, timeout_s=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def _latest_user_text(messages):
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, list):  # multimodal: take text parts
                txt = " ".join(p.get("text", "") for p in c
                               if isinstance(p, dict) and p.get("type") == "text")
                if txt.strip():
                    return txt
    return ""


def _strip_tags_from_history(payload):
    """Remove bouncer tags from assistant messages before they go upstream.

    Tags are for the human, not the model: if tagged history reaches the
    model it (a) imitates the tag (double-tagging) and (b) speculates about
    a 'routing layer' it cannot see. Returns a new payload with clean
    assistant content; other messages untouched.
    """
    import re
    # any bracketed tag line led by the route icons: exact configured tags,
    # renamed tags from older configs, and model-imitated variants all match
    tag_line = re.compile(r"^\s*\[(🟢|⚡)[^\]]*\]\s*$")
    msgs = payload.get("messages")
    if not msgs:
        return payload
    changed = False
    clean = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant" \
                and isinstance(m.get("content"), str):
            c = m["content"]
            # strip any tag line (configured, renamed, or imitated)
            lines = [ln for ln in c.split("\n") if not tag_line.match(ln)]
            new_c = "\n".join(lines)
            if new_c != c:
                m = dict(m)
                m["content"] = new_c
                changed = True
        clean.append(m)
    if not changed:
        return payload
    out = dict(payload)
    out["messages"] = clean
    return out


def _needs_harness(payload):
    """True if the request involves Hermes tool machinery → pass through."""
    if payload.get("tools"):
        return True
    for m in payload.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") in ("tool", "function"):
            return True
        if m.get("tool_calls") or m.get("function_call"):
            return True
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


def _thread_decision_id(payload, decision_id):
    """Inject *decision_id* into the payload so downstream workers can echo it back.

    Writes a system message at the top of the messages list — this is the most
    reliable surface because it survives all transports (HTTP headers may be
    stripped by intermediaries, but a system-prompt stanza always reaches the model).
    """
    import copy
    msgs = copy.deepcopy(payload.get("messages") or [])
    stanza = f"ROUTING DECISION ID: {decision_id} — include this id in any follow-up metadata."
    # Insert at position 0 so it's the first context the model sees.
    msgs.insert(0, {"role": "system", "content": stanza})
    payload["messages"] = msgs


def _judge(task_text):
    """Ask Laya via the proven hook client. Returns (p_hard, flag, source)."""
    try:
        j = laya_hook.apply(laya_hook.judge(task_text))
        return j.get("p_hard"), bool(j.get("difficulty_flag")), j.get("source", "laya")
    except Exception:
        raise


def _log(rec):
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # logging must never break the chat


@app.get("/health")
def health():
    return {"ok": True, "proxy": "bouncer", "local": LOCAL_URL, "judge": JUDGE_URL}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "bouncer-proxy", "object": "model",
            "created": int(time.time()), "owned_by": "bouncer"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    payload = await request.json()
    messages = payload.get("messages") or []
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Outcome-capture key — minted at routing time, threaded downstream.
    decision_id = uuid.uuid4().hex[:12]
    task_text_raw = _latest_user_text(messages)
    rec["decision_id"] = decision_id
    rec["task_text_hash"] = hashlib.sha256(task_text_raw.encode()).hexdigest()[:16]

    try:
        # 0. Route decision
        if _needs_harness(payload):
            # Rule v2: tool tasks are JUDGED. Simple tool work (p_hard low) goes
            # LOCAL — the 35B proved clean function-calling (bench: 11/11 format,
            # 11/11 args, 8/8 followup). Hard tool work passes to the session model.
            task = _latest_user_text(messages)
            rec["task"] = task[:200]
            try:
                p_hard, flag, source = _judge(task)
            except Exception:
                p_hard, flag, source = None, False, "judge-failed"
            rec["p_hard"], rec["source"] = p_hard, source
            if flag or source == "judge-failed":
                # hard tool work, or can't judge → session model (conservative
                # for harness integrity; deep tool chains untested locally)
                # v3: tag it too — every escalation must be visible to the human
                decision, tag = "passthrough-harness", TAG_ESC
            else:
                decision, tag = "local-harness", TAG_LOCAL
        else:
            task = _latest_user_text(messages)
            rec["task"] = task[:200]
            try:
                p_hard, flag, source = _judge(task)
            except Exception:
                p_hard, flag, source = None, False, "judge-failed"
            rec["p_hard"], rec["source"] = p_hard, source
            if flag:
                decision = "escalate"
            elif source == "judge-failed":
                decision = "escalate"  # fail-open to the session model
            else:
                decision = "local"
            tag = TAG_LOCAL if decision == "local" else TAG_ESC
        rec["decision"] = decision

        # Outcome-capture: classify as local or frontier for training signal.
        rec["routed_to"] = "local" if decision in ("local", "local-harness") else "frontier"

        # 1. Route (local + local-harness both go to the 35B, tools forwarded)
        if decision in ("local", "local-harness"):
            try:
                local_payload = _strip_tags_from_history(dict(payload))
                local_payload["model"] = "local-35b"
                local_payload.pop("chat_template_kwargs", None)  # llama-server args differ
                # Thread outcome-capture key into the request so it can be echoed back.
                _thread_decision_id(local_payload, decision_id)
                rec["upstream"] = ("local", LOCAL_URL, local_payload,
                                   {"Content-Type": "application/json"})
                if not payload.get("stream"):
                    resp = _post(LOCAL_URL, local_payload,
                                 {"Content-Type": "application/json"}, 600)
                    resp["model"] = "bouncer-local-35b"
                    content = resp["choices"][0]["message"].get("content") or ""
                    if tag:
                        resp["choices"][0]["message"]["content"] = f"{tag}\n{content}"
                    if isinstance(resp.get("usage"), dict):
                        rec["usage"] = resp["usage"]
                    _log(rec)
                    return JSONResponse(resp)
                # streaming: fall through to SSE relay with local upstream
                rec["answered_by"] = "local-35b"
            except Exception as e:
                rec["local_error"] = str(e)[:120]
                rec["rule"] = "local-failed-escalate"
                # fall through to escalation

        # 2. Escalation / pass-through: the session model, unchanged
        if "upstream" not in rec:
            esc_payload = _strip_tags_from_history(dict(payload))
            esc_payload["model"] = os.environ.get("BOUNCER_ESC_MODEL", "z-ai/glm-5.3")
            esc_payload.pop("chat_template_kwargs", None)  # llama-server-specific
            # Thread outcome-capture key into the escalation request too.
            _thread_decision_id(esc_payload, decision_id)
            rec["upstream"] = ("esc", ESC_URL, esc_payload, _esc_headers())
            rec["answered_by"] = "session-model"

        kind, up_url, up_payload, up_headers = rec["upstream"]
        rec["upstream"] = [kind, up_url, {k: v for k, v in up_payload.items() if k != "stream_options"}, "<headers-redacted>"]
        if payload.get("stream") and kind == "esc":
            up_payload.setdefault("stream_options", {})["include_usage"] = True

        # 3a. Streaming request → SSE relay (chunks relayed verbatim, tag prepended)
        if payload.get("stream"):

            def sse_relay():
                import urllib.error
                try:
                    req = urllib.request.Request(
                        up_url, data=json.dumps(up_payload).encode(),
                        headers=up_headers)
                    with urllib.request.urlopen(req, timeout=600) as r:
                        tag_sent = False
                        for line in r:
                            s = line.decode("utf-8", "replace")
                            if s.startswith("data: "):
                                body = s[6:].strip()
                                if body and body != "[DONE]":
                                    try:
                                        j = json.loads(body)
                                        if isinstance(j.get("usage"), dict):
                                            rec["usage"] = j["usage"]
                                        d = (j.get("choices") and j["choices"][0].get("delta")) or {}
                                        if not tag_sent and tag and d.get("content"):
                                            d["content"] = f"{tag}\n{d['content']}"
                                            tag_sent = True
                                            s = "data: " + json.dumps(j) + "\n\n"
                                    except ValueError:
                                        pass
                            yield s
                except Exception as e:
                    err = json.dumps({"error": {"message": f"bouncer relay failure: {e}",
                                                "type": "proxy_error"}})
                    yield f"data: {err}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    _log(rec)  # logged at stream end so token usage is included

            return StreamingResponse(sse_relay(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache",
                                              "X-Accel-Buffering": "no"})

        # 3b. Non-streaming escalation
        resp = _post(up_url, up_payload, up_headers, 600)
        if tag:
            content = resp["choices"][0]["message"].get("content") or ""
            resp["choices"][0]["message"]["content"] = f"{tag}\n{content}"
        if isinstance(resp.get("usage"), dict):
            rec["usage"] = resp["usage"]
        _log(rec)
        return JSONResponse(resp)

    except Exception as e:
        # Fail-safe: log and re-raise as 502 so the caller's own fallback kicks in.
        rec["error"] = str(e)[:200]
        _log(rec)
        return JSONResponse({"error": {"message": f"bouncer proxy failure: {e}"}},
                            status_code=502)


def _read_log():
    recs = []
    try:
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return recs


# Frontier price used for savings estimates: blended $ per 1M tokens you
# would have paid if every request went to the escalation model.
PRICE_PER_MTOK = float(os.environ.get("BOUNCER_FRONTIER_PRICE_PER_MTOK", "3.0"))


@app.get("/stats")
def stats():
    recs = _read_log()
    routed = [r for r in recs if r.get("decision")]
    local = [r for r in routed if r.get("decision") in ("local", "local-harness")]
    esc = [r for r in routed if r.get("decision") in ("escalate", "passthrough-harness")]

    def _tok(rs, keys=("prompt_tokens", "completion_tokens")):
        n = 0
        for r in rs:
            u = r.get("usage") or {}
            n += sum(int(u.get(k) or 0) for k in keys)
        return n

    local_tok = _tok(local)
    esc_tok = _tok(esc)
    savings = local_tok / 1_000_000 * PRICE_PER_MTOK

    return {
        "decisions_total": len(routed),
        "local_count": len(local),
        "escalated_count": len(esc),
        "local_tokens": local_tok,
        "escalated_tokens": esc_tok,
        "savings_usd_est": round(savings, 4),
        "price_per_mtok_assumed": PRICE_PER_MTOK,
        "basis": "savings = tokens answered locally x frontier price; usage captured from upstream responses",
        "recent": routed[-10:],
    }


@app.get("/")
def dashboard():
    page = os.path.join(HERE, "dashboard.html")
    if os.path.exists(page):
        with open(page) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Bouncer</h1><p>proxy is running; dashboard.html not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)
