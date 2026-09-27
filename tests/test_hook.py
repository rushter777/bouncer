"""Tests for laya_hook — stdlib only, no live sidecar required.

Contract under test (matches laya_sidecar.py as fixed 2026-09-24):
  judge() -> answers dict {"tools_file": {"noul": p, ...}, ...} or None
  apply() -> dispatch config; answers are dicts with a "noul" scalar.
"""

import json
import os
import sys
import unittest
import unittest.mock
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import laya_hook  # noqa: E402


def _ans(p):
    """Build a noul answer dict like the sidecar returns."""
    return {"type": "noul", "noul": p, "confidence": abs(1 - 2 * p)}


class _FakeHandler(BaseHTTPRequestHandler):
    response_body = None
    status = None

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.status is not None:
            self.send_response(self.status)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        if n:
            self.rfile.read(n)
        body = json.dumps({"answers": self.response_body or {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def _env_cleaned():
    """Env snapshot with LAYA_THRESH_* removed."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("LAYA_THRESH")}
    for k in saved:
        del os.environ[k]
    return saved


class TestJudge(unittest.TestCase):
    def setUp(self):
        _FakeHandler.response_body = None
        _FakeHandler.status = None
        self.srv = HTTPServer(("127.0.0.1", 18777), _FakeHandler)
        Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()

    def _url(self, path="/predict"):
        return f"http://127.0.0.1:18777{path}"

    def test_success_parses_answers(self):
        _FakeHandler.response_body = {"tools_file": _ans(0.9)}
        with unittest.mock.patch.object(laya_hook, "_SIDECAR_URL", self._url()):
            r = laya_hook.judge("read a file")
        self.assertIsNotNone(r)
        self.assertIn("tools_file", r)  # judge returns the answers map

    def test_connection_refused_returns_none(self):
        with unittest.mock.patch.object(laya_hook, "_SIDECAR_URL", "http://127.0.0.1:1/predict"):
            self.assertIsNone(laya_hook.judge("x"))

    def test_5xx_returns_none(self):
        _FakeHandler.status = 500
        with unittest.mock.patch.object(laya_hook, "_SIDECAR_URL", self._url()):
            self.assertIsNone(laya_hook.judge("x"))


class TestApply(unittest.TestCase):
    def setUp(self):
        self._saved = _env_cleaned()

    def tearDown(self):
        os.environ.update(self._saved)

    def test_none_returns_failsafe(self):
        r = laya_hook.apply(None)
        self.assertEqual(r["source"], "fallback")
        self.assertTrue(all(r["tools"].values()))
        self.assertTrue(r["needs_verification"])

    def test_dict_answers_thresholded(self):
        d = {"answers": {
            "tools_file": _ans(0.8), "tools_terminal": _ans(0.3),
            "tools_web": _ans(0.5), "needs_verification": _ans(0.6),
        }}
        r = laya_hook.apply(d)
        self.assertEqual(r["source"], "laya")
        self.assertTrue(r["tools"]["file"])
        self.assertFalse(r["tools"]["terminal"])
        self.assertTrue(r["tools"]["web"])  # 0.5 >= 0.5 boundary
        self.assertTrue(r["needs_verification"])

    def test_scalar_answers_tolerated(self):
        d = {"tools_file": 1, "tools_terminal": 0, "needs_verification": True}
        r = laya_hook.apply(d)
        self.assertTrue(r["tools"]["file"])
        self.assertFalse(r["tools"]["terminal"])
        self.assertTrue(r["needs_verification"])

    def test_bare_answers_dict_no_wrapper(self):
        # apply() must accept the answers map directly (no "answers" key)
        d = {"tools_file": _ans(0.9)}
        r = laya_hook.apply(d)
        self.assertTrue(r["tools"]["file"])
        self.assertEqual(r["source"], "laya")

    def test_env_threshold_override(self):
        d = {"tools_file": _ans(0.4)}
        os.environ["LAYA_THRESH_TOOLS"] = "0.3"
        r = laya_hook.apply(d)
        self.assertTrue(r["tools"]["file"])  # 0.4 >= 0.3

    def test_env_threshold_verify_override_blocks(self):
        d = {"needs_verification": _ans(0.4)}
        os.environ["LAYA_THRESH_VERIFY"] = "0.5"
        r = laya_hook.apply(d)
        self.assertFalse(r["needs_verification"])

    def test_missing_gate_defaults_false(self):
        d = {"answers": {}}
        r = laya_hook.apply(d)
        self.assertFalse(r["tools"]["file"])

    def test_fallback_dict_is_fresh_copy(self):
        r1 = laya_hook.apply(None)
        r1["tools"]["file"] = False
        r2 = laya_hook.apply(None)
        self.assertTrue(r2["tools"]["file"])


class TestConstants(unittest.TestCase):
    def test_question_set_exact_criteria(self):
        # The paraphrase bug: criteria must be the corpus text, not "needs this capability"
        for q in laya_hook.QUESTION_SET.values():
            self.assertNotIn("needs this capability", str(q["criteria"]))
        self.assertEqual(
            laya_hook.QUESTION_SET["tools_file"]["criteria"]["true"],
            "needs file access to do",
        )

    def test_question_set_keys(self):
        # Sept 26: difficulty (score-type, advisory) added to the battery.
        self.assertEqual(
            set(laya_hook.QUESTION_SET),
            {"tools_file", "tools_terminal", "tools_web", "needs_verification", "difficulty"},
        )

    def test_difficulty_flag_advisory(self):
        # Flag fires when P(hard) >= threshold; present in output; gates untouched.
        probs = {
            "tools_file": {"noul": 0.9}, "tools_terminal": {"noul": 0.1},
            "tools_web": {"noul": 0.2}, "needs_verification": {"noul": 0.3},
            "difficulty": {"probabilities": {"0-none": 0.05, "1-one demand": 0.4,
                                             "2-two demands": 0.3, "3-three demands": 0.2,
                                             "4-four demands": 0.05}},
        }
        cfg = laya_hook.apply({"answers": probs})
        self.assertTrue(cfg["difficulty_flag"])
        self.assertAlmostEqual(cfg["p_hard"], 0.95)
        self.assertEqual(cfg["tools"], {"file": True, "terminal": False, "web": False})

        # Below threshold -> no flag
        probs["difficulty"] = {"probabilities": {"0-none": 0.5, "1-one demand": 0.5}}
        cfg = laya_hook.apply({"answers": probs})
        self.assertFalse(cfg["difficulty_flag"])
        self.assertAlmostEqual(cfg["p_hard"], 0.5)

        # Missing difficulty answer -> p_hard None, flag False
        del probs["difficulty"]
        cfg = laya_hook.apply({"answers": probs})
        self.assertFalse(cfg["difficulty_flag"])
        self.assertIsNone(cfg["p_hard"])

        # Fail-safe keeps advisory keys with no data
        cfg = laya_hook.apply(None)
        self.assertFalse(cfg["difficulty_flag"])
        self.assertIsNone(cfg["p_hard"])
        self.assertTrue(cfg["tools"]["file"])  # fail-open unchanged


if __name__ == "__main__":
    unittest.main()
