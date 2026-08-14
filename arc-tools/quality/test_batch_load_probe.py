#!/usr/bin/env python3
"""Mock-server smoke test for batch_load_probe.py — no GPU, no model, no pip.

Spins a threaded HTTP mock that speaks the server's verified response shapes
(chat stream: per-token chunks + usage ON the final choice-carrying chunk;
raw completions stream: per-token chunks, NO usage — chunk counting), with
known token counts and known delays, then asserts the probe's math:

  1. chat sweep B=1,4:   exact token counts from usage, TTFT ~= first-token
     delay, per-user decode rate ~= 1/inter-token delay, aggregate ~= B x
     per-user (mock is perfectly parallel), KV-guard estimate exact,
     $/Mtok formula recomputed, zero errors.
  2. raw mode + KV guard: chunk-counted tokens exact, /v1/completions only,
     WARNING[KV] fires when est > --max-ctx (and never blocks).
  3. sustained mode:      closed loop keeps B in flight, completions ~=
     B x duration / per-request wall.

Precedent: wave-1 harness dry-runs against a mock /v1/completions server.
Run:  python3 test_batch_load_probe.py     (exit 0 = PASS)
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TTFT_DELAY = 0.25     # mock: seconds before the first token chunk
INTER_DELAY = 0.02    # mock: seconds between token chunks
PROMPT_TOKENS = 71    # mock: usage.prompt_tokens (chat + non-stream only)


class MockHandler(BaseHTTPRequestHandler):
    hits = []  # POST paths, appended per request

    def log_message(self, *a):  # noqa: D102 — silence
        pass

    def do_GET(self):
        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        MockHandler.hits.append(self.path)
        n_tok = int(req.get("max_tokens", 16))
        chat = "chat" in self.path
        if req.get("stream"):
            self._stream(n_tok, chat)
        else:
            self._json_full(n_tok, chat)

    def _usage(self, n_tok):
        return {"completion_tokens": n_tok, "prompt_tokens": PROMPT_TOKENS,
                "total_tokens": n_tok + PROMPT_TOKENS}

    def _stream(self, n_tok, chat):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(TTFT_DELAY)
        for i in range(n_tok):
            if i:
                time.sleep(INTER_DELAY)
            last = i == n_tok - 1
            if chat:
                obj = {"id": "mock", "object": "chat.completion.chunk",
                       "choices": [{"index": 0,
                                    "delta": {"role": "assistant", "content": "tok "},
                                    "finish_reason": "stop" if last else None}]}
                if last:  # usage rides the final choice-carrying chunk
                    obj["usage"] = self._usage(n_tok)
            else:  # raw completions stream: NO usage field, ever
                obj = {"id": "mock", "object": "text_completion",
                       "choices": [{"index": 0, "text": "tok ",
                                    "finish_reason": "stop" if last else None}]}
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json_full(self, n_tok, chat):
        time.sleep(TTFT_DELAY + (n_tok - 1) * INTER_DELAY)
        choice = ({"index": 0, "message": {"role": "assistant", "content": "tok " * n_tok},
                   "finish_reason": "stop"} if chat else
                  {"index": 0, "text": "tok " * n_tok, "finish_reason": "stop"})
        body = json.dumps({"id": "mock", "choices": [choice],
                           "usage": self._usage(n_tok)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_probe(base_url, out_path, extra):
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "batch_load_probe.py"),
         "--label", "mock", "--out", out_path] + extra,
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "BASE_URL": base_url}, cwd=HERE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise AssertionError(f"probe exited {proc.returncode}")
    with open(out_path) as f:
        return proc, json.load(f)


def approx(v, lo, hi, what):
    assert v is not None and lo <= v <= hi, f"{what}: {v} not in [{lo}, {hi}]"


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    tmp = tempfile.mkdtemp(prefix="batchprobe_")
    n = 24  # decode tokens per measured request
    ideal_rate = 1.0 / INTER_DELAY  # 50 tok/s per stream in the mock

    # ---- 1. chat-endpoint sweep B=1,4 (stream, usage-based counts) ----
    MockHandler.hits = []
    proc, out = run_probe(base, os.path.join(tmp, "chat.json"),
                          ["--batches", "1,4", "--reps", "2",
                           "--max-tokens", str(n), "--warmup-tokens", "8"])
    assert [e["B"] for e in out["per_b"]] == [1, 4]
    b1, b4 = out["per_b"]
    for e in (b1, b4):
        assert e["errors"] == 0, f"errors at B={e['B']}: {e}"
        for rep in e["reps"]:
            for rec in rep["records"]:
                assert rec["completion_tokens"] == n, rec
                assert rec["prompt_tokens"] == PROMPT_TOKENS, rec
    # KV guard estimate is exact once warmup measured prompt_tokens
    assert b4["kv_guard"]["est_kv_tokens"] == 4 * (PROMPT_TOKENS + n), b4["kv_guard"]
    approx(b1["dist"]["ttft_s_p50"], TTFT_DELAY - 0.01, TTFT_DELAY + 0.75, "TTFT p50")
    approx(b1["dist"]["per_req_decode_tok_s_p50"], ideal_rate * 0.3,
           ideal_rate * 1.2, "per-user decode p50")
    agg1 = b1["mean"]["agg_tok_s_headline"]
    agg4 = b4["mean"]["agg_tok_s_headline"]
    approx(agg4 / agg1, 2.5, 5.5, "agg(B=4)/agg(B=1) on a parallel mock")
    # $/Mtok formula: gpu_cost_hr * 1e6 / (agg * 3600)
    s = out["summary"]
    assert s["peak_B"] == 4 and s["peak_agg_decode_tok_s"] == agg4, s
    want_cost = round(4.92e6 / (agg4 * 3600.0), 2)
    assert abs(b4["cost_per_mtok_usd"] - want_cost) < 0.05, (b4["cost_per_mtok_usd"], want_cost)
    assert all(h == "/v1/chat/completions" for h in MockHandler.hits), set(MockHandler.hits)
    assert "BATCHSWEEP[mock]" in proc.stdout
    print(f"PASS chat sweep: tokens exact, agg {agg1} -> {agg4} tok/s, "
          f"TTFT p50 {b1['dist']['ttft_s_p50']}s, $/Mtok {b4['cost_per_mtok_usd']}")

    # ---- 2. raw mode: chunk-counted tokens + KV guard warning ----
    MockHandler.hits = []
    proc, out = run_probe(base, os.path.join(tmp, "raw.json"),
                          ["--raw", "--batches", "4", "--reps", "1",
                           "--max-tokens", str(n), "--warmup-tokens", "8",
                           "--max-ctx", "100"])
    e = out["per_b"][0]
    assert e["errors"] == 0, e
    for rec in e["reps"][0]["records"]:
        assert rec["completion_tokens"] == n, rec          # counted, no usage
        assert rec["prompt_tokens"] is None, rec           # raw stream: none
    assert "WARNING[KV]" in proc.stdout, proc.stdout
    assert e["kv_guard"]["warned"] is True, e["kv_guard"]
    assert all(h == "/v1/completions" for h in MockHandler.hits), set(MockHandler.hits)
    print(f"PASS raw mode: chunk-counted {n} tok/req, KV warning fired (never blocked)")

    # ---- 3. sustained closed loop ----
    duration = 2
    proc, out = run_probe(base, os.path.join(tmp, "sust.json"),
                          ["--batches", "2", "--duration", str(duration),
                           "--max-tokens", str(n), "--warmup-tokens", "8"])
    e = out["per_b"][0]
    rep = e["reps"][0]
    assert e["errors"] == 0, e
    # ~0.71s/request, 2 workers, 2s window -> ~6 completions; >= 4 is a
    # generous floor that still proves the loop refills in-flight slots
    assert rep["requests"] >= 4, rep["requests"]
    assert rep["total_completion_tokens"] == rep["requests"] * n, rep
    assert rep["wall_s"] >= duration * 0.9, rep["wall_s"]
    assert out["meta"]["mode"] == f"sustained_{duration}s", out["meta"]["mode"]
    print(f"PASS sustained: {rep['requests']} completions in {rep['wall_s']}s, "
          f"agg {e['mean']['agg_tok_s_headline']} tok/s")

    server.shutdown()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
