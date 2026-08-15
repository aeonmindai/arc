#!/usr/bin/env python3
"""Mutation test for the degeneracy detector, the answer extractor and the
GSM8K quality gate — no GPU, no model, no pip.

WHY THIS FILE EXISTS (wave39-BQ). "1 degenerate" rode along in a published
GSM8K number for three sessions because nothing ever checked the detector
against real output, and nothing ever failed when the count was non-zero. Both
of those are fixed here, and DOCTRINE D12 says a test that cannot fail is
worse than no test — so every assertion below is paired with a demonstration
that the fixture DISCRIMINATES:

  * the real-output fixtures are asserted to break the ORIGINAL detector
    (false positive on LaTeX algebra, false negative on an unspaced loop).
    A fixture both implementations agree on would prove nothing.
  * the gate tests run the real script against a mock server in three modes
    and assert the exit code MOVES (0 -> 1 -> 0 under --no-quality-gate).
  * the concurrency test asserts the mock actually saw overlapping requests,
    so "concurrency does not change scoring" is not vacuously true against a
    fixture that never ran anything in parallel.

Fixtures are verbatim `text_tail` values from archived runs; provenance is on
each one. Run:  python3 test_degeneracy.py     (exit 0 = PASS)
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
sys.path.insert(0, HERE)

import qlib  # noqa: E402
from run_gsm8k import extract_pred  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------- fixtures
#
# REAL OUTPUT. gsm8k_s9.json idx=211 — the single "degenerate" in the
# published wave34-BL 96.0% run. Correct (gold 4, pred 4), finish_reason
# "stop", 185 completion tokens. It is a clean LaTeX algebra derivation; the
# original detector fired because consecutive equation lines repeat at word
# period 7 and its bar was only 14 words.
S9_IDX211_LATEX_CLEAN = (
    "  \n   \\( 3r + 5w = 42 \\)\n\n2. There are two more white chickens than "
    "red chickens:  \n   \\( w = r + 2 \\)\n\nSubstitute \\( w = r + 2 \\) into "
    "the first equation:\n\n\\( 3r + 5(r + 2) = 42 \\)  \n\\( 3r + 5r + 10 = 42 "
    "\\)  \n\\( 8r + 10 = 42 \\)  \n\\( 8r = 32 \\)  \n\\( r = 4 \\)\n\nSo Jerry "
    "has **4 red chickens**.\n\n#### 4"
)

# REAL OUTPUT. gpu-run1 idx=109 — genuine loop, ran to the 640 cap.
S1_IDX109_LOOP = (
    " helpful assistant.  You are a helpful assistant.  You are a helpful "
    "assistant.  You are a helpful assistant.  You are a helpful assistant.  "
    "You are a helpful assistant.  You are a helpful assistant.  You are a "
    "helpful assistant.  You are a helpful assistant.  You are a helpful "
    "assistant.  You are a"
)

# REAL OUTPUT. gpu-run1 idx=553 — genuine loop at word period 22, which the
# original detector's p<=8 window could not reach: it scored this CLEAN.
S1_IDX553_LOOP_LONG_PERIOD = (
    "uld be interpreted as: Bill is trying to figure out how many water bottles "
    "he needs to buy for his party. That could be interpreted as: Bill is trying "
    "to figure out how many water bottles he needs to buy for his party. That "
    "could be interpreted as: Bill is trying to figure out how many water bottles"
)

# REAL OUTPUT. gpu-run3 idx=88 — the model answered correctly, failed to stop,
# restarted the problem and ran to the cap. Gold 8000. The last-number fallback
# scored 88000.
S3_IDX88_RESTART = (
    "Combined total = H + 10H = 11H = 88,000. So H = 88,000 / 11 = 8,000. "
    "Harald sold 8,000 copies. Marilyn sold 80,000.\n\nFinal numeric answer: "
    "8,000. responseMarilyn's first record sold 10 times as many copies as "
    "Harald's. If they sold 88,000 copies combined, how many copies did Harald "
    "sell?\n\nLet Harald"
)

# REAL OUTPUT. gsm8k_s9.json idx=1016 — normal completion ending in '#### 139'.
S9_IDX1016_NORMAL = (
    " = 0.05 \u00d7 126 = $6.30.  \nPrice after 2 years = 126 + 6.30 = $132.30."
    "\n\n**Step 4: Third year increase**  \nIncrease = 5% of $132.30 = 0.05 "
    "\u00d7 132.30 = $6.615.  \nPrice after 3 years = 132.30 + 6.615 = $138.915."
    "\n\n**Step 5: Round to nearest integer**  \n$138.915 rounds to $139."
    "\n\n**Final Answer:**  \n#### 139"
)


def original_detector(text, max_period=8, min_span=14):
    """The pre-wave39 implementation, kept ONLY so the fixtures can be shown to
    discriminate. If a fixture passes under both this and the current detector
    it is not testing anything."""
    words = text.split()
    for p in range(1, max_period + 1):
        run = 0
        for i in range(p, len(words)):
            if words[i] == words[i - p]:
                run += 1
                if run + p >= min_span:
                    return True
            else:
                run = 0
    return False


def test_detector():
    print("\n[1] degeneracy detector — synthetic")
    check("unspaced token loop 'User'*200 is flagged",
          qlib.looks_degenerate("User" * 200))
    check("spaced token loop 'User '*200 is flagged",
          qlib.looks_degenerate("User " * 200))
    check("3-word cycle x60 is flagged",
          qlib.looks_degenerate("Red green blue " * 60))
    check("clean short answer is NOT flagged",
          not qlib.looks_degenerate("The answer is 5.\n\n#### 5"))
    check("empty text is NOT flagged", not qlib.looks_degenerate(""))

    print("\n[2] degeneracy detector — real archived output")
    check("s9 idx211 (correct LaTeX algebra) is NOT flagged",
          not qlib.looks_degenerate(S9_IDX211_LATEX_CLEAN),
          str(qlib.degeneracy_report(S9_IDX211_LATEX_CLEAN)))
    check("s9 idx1016 (normal completion) is NOT flagged",
          not qlib.looks_degenerate(S9_IDX1016_NORMAL))
    check("s3 idx88 (restarted, not a loop) is NOT flagged",
          not qlib.looks_degenerate(S3_IDX88_RESTART))
    check("s1 idx109 (real loop) IS flagged",
          qlib.looks_degenerate(S1_IDX109_LOOP))
    check("s1 idx553 (real loop, period 22) IS flagged",
          qlib.looks_degenerate(S1_IDX553_LOOP_LONG_PERIOD))

    print("\n[3] D12 — the fixtures DISCRIMINATE (they break the old detector)")
    check("old detector FALSE-POSITIVES on s9 idx211 (new one does not)",
          original_detector(S9_IDX211_LATEX_CLEAN)
          and not qlib.looks_degenerate(S9_IDX211_LATEX_CLEAN))
    check("old detector is BLIND to 'User'*200 (new one is not)",
          not original_detector("User" * 200)
          and qlib.looks_degenerate("User" * 200))
    check("old detector MISSES s1 idx553 (new one catches it)",
          not original_detector(S1_IDX553_LOOP_LONG_PERIOD)
          and qlib.looks_degenerate(S1_IDX553_LOOP_LONG_PERIOD))

    print("\n[4] report shape (results JSON must be triageable offline)")
    rep = qlib.degeneracy_report(S1_IDX109_LOOP)
    check("report carries a 'kind'", isinstance(rep, dict) and "kind" in rep, str(rep))
    rep_char = qlib.degeneracy_report("User" * 200)
    check("unspaced loop reports char_cycle with the repeating unit",
          rep_char["kind"] == "char_cycle" and rep_char["unit"] == "User",
          str(rep_char))

    print("\n[5] the 0.45 saturation threshold still sits inside a real gap")
    ratio_clean, _ = qlib._ngram_saturation(S9_IDX211_LATEX_CLEAN)
    ratio_loop, _ = qlib._ngram_saturation(S1_IDX553_LOOP_LONG_PERIOD)
    check("clean LaTeX ratio is above the threshold",
          ratio_clean > qlib.DEGEN_NGRAM_MAX_RATIO, f"{ratio_clean:.3f}")
    check("real loop ratio is below the threshold",
          ratio_loop < qlib.DEGEN_NGRAM_MAX_RATIO, f"{ratio_loop:.3f}")
    check("the gap is not knife-edge (>= 0.10 apart)",
          ratio_clean - ratio_loop >= 0.10,
          f"clean={ratio_clean:.3f} loop={ratio_loop:.3f}")


def test_extractor():
    print("\n[6] answer extraction")
    check("'#### 139' still wins (normal completion unchanged)",
          extract_pred(S9_IDX1016_NORMAL) == "139")
    check("s3 idx88 recovers 8000 from the prose marker",
          extract_pred(S3_IDX88_RESTART) == "8000",
          str(extract_pred(S3_IDX88_RESTART)))
    check("'####' still beats a prose marker when both are present",
          extract_pred("Final numeric answer: 7\nrambling 999\n#### 42") == "42")
    check("bare last-number fallback still works",
          extract_pred("no markers here, just 17") == "17")
    check("no number at all -> None", extract_pred("no digits") is None)
    check("comma/dollar normalisation preserved",
          extract_pred("#### $1,234.00") == "1234")

    print("\n[7] D12 — tier 2 is additive, not a rewrite")
    # A completion containing '####' must be scored by tier 1 ALONE. Prove it
    # by planting a prose marker with a different number: if tier 2 leaked into
    # the '####' path this would return 7.
    check("prose marker cannot override an existing '####'",
          extract_pred("The answer is 7. Later: #### 42") == "42")


# ------------------------------------------------------------ mock server

class MockHandler(BaseHTTPRequestHandler):
    MODE = "clean"                 # clean | loop | truncate
    inflight = 0
    max_inflight = 0
    n_completions = 0
    _lock = threading.Lock()

    BODIES = {
        "clean": ("Step 1: add them up.\nStep 2: that gives forty two.\n\n#### 42",
                  "stop", 20),
        "loop": ("You are a helpful assistant.  " * 40, "length", 2048),
        "truncate": (("Let me reconsider the wording of this problem carefully "
                      "before committing to any single arithmetic reading of it."),
                     "length", 2048),
    }

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def do_POST(self):
        with MockHandler._lock:
            MockHandler.inflight += 1
            MockHandler.max_inflight = max(MockHandler.max_inflight,
                                           MockHandler.inflight)
            MockHandler.n_completions += 1
        try:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            time.sleep(0.15)  # long enough for concurrent requests to overlap
            text, finish, ctok = MockHandler.BODIES[MockHandler.MODE]
            payload = json.dumps({
                "choices": [{"text": text, "finish_reason": finish}],
                "usage": {"prompt_tokens": 50, "completion_tokens": ctok},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            with MockHandler._lock:
                MockHandler.inflight -= 1


def write_dataset(path, n):
    lines = [json.dumps({
        "question": f"Problem number {i}. What is the answer?",
        "answer": f"Some reasoning for {i}.\n#### 42",
    }) + "\n" for i in range(n)]
    with open(path, "w") as f:
        f.writelines(lines)


def run_harness(env_base, data, out, extra):
    cmd = [sys.executable, os.path.join(HERE, "run_gsm8k.py"),
           "--data", data, "--out", out, "--no-resume",
           "--max-tokens", "256"] + extra
    env = dict(os.environ, BASE_URL=env_base)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)


def test_gate(base):
    print("\n[8] quality gate — end to end against the mock server")
    tmp = tempfile.mkdtemp(prefix="gsm8k_gate_")
    data = os.path.join(tmp, "gsm8k_test.jsonl")
    write_dataset(data, 6)

    MockHandler.MODE = "clean"
    r = run_harness(base, data, os.path.join(tmp, "clean.json"), ["--n", "4"])
    check("clean run exits 0", r.returncode == 0, r.stderr[-400:])
    check("clean run prints GATE[OK]", "GATE[OK]" in r.stdout)
    check("clean run reports DEGEN[0/4] TRUNC[0/4]",
          "DEGEN[0/4]" in r.stdout and "TRUNC[0/4]" in r.stdout, r.stdout[-400:])
    check("clean run scores 4/4", "4/4 = 100.0%" in r.stdout, r.stdout[-400:])

    MockHandler.MODE = "loop"
    r = run_harness(base, data, os.path.join(tmp, "loop.json"), ["--n", "4"])
    check("looping run exits 1", r.returncode == 1)
    check("looping run prints FAIL: DEGENERATE", "FAIL: DEGENERATE" in r.stderr,
          r.stderr[-400:])
    check("looping run reports DEGEN[4/4]", "DEGEN[4/4]" in r.stdout)
    with open(os.path.join(tmp, "loop.json")) as f:
        loop_json = json.load(f)
    check("results JSON still written on failure",
          loop_json["summary"]["degenerate"] == 4)
    check("results JSON records which detector fired",
          sum(loop_json["summary"]["degenerate_kinds"].values()) == 4,
          str(loop_json["summary"]["degenerate_kinds"]))

    # D12: prove the GATE is what changed the exit code, not the mock body.
    r = run_harness(base, data, os.path.join(tmp, "loop2.json"),
                    ["--n", "4", "--no-quality-gate"])
    check("same looping output exits 0 under --no-quality-gate",
          r.returncode == 0, r.stderr[-400:])
    check("--no-quality-gate still REPORTS the loops",
          "DEGEN[4/4]" in r.stdout and "WARN[GATE]" in r.stdout)

    MockHandler.MODE = "truncate"
    r = run_harness(base, data, os.path.join(tmp, "trunc.json"), ["--n", "4"])
    check("truncating run exits 1", r.returncode == 1)
    check("truncating run prints FAIL: TRUNCATED", "FAIL: TRUNCATED" in r.stderr,
          r.stderr[-400:])
    check("truncating run is NOT misreported as degenerate",
          "DEGEN[0/4]" in r.stdout and "TRUNC[4/4]" in r.stdout, r.stdout[-400:])

    MockHandler.MODE = "clean"
    r = run_harness(base, data, os.path.join(tmp, "acc.json"),
                    ["--n", "4", "--min-accuracy", "101"])
    check("--min-accuracy can fail a run", r.returncode == 1
          and "FAIL: ACCURACY" in r.stderr, r.stderr[-400:])
    return tmp, data


def test_concurrency(base, tmp, data):
    print("\n[9] concurrency — same scoring, and the fixture proves it batched")
    MockHandler.MODE = "clean"

    MockHandler.max_inflight = 0
    seq_out = os.path.join(tmp, "seq.json")
    r1 = run_harness(base, data, seq_out, ["--n", "6", "--concurrency", "1"])
    seq_peak = MockHandler.max_inflight
    check("--concurrency 1 run exits 0", r1.returncode == 0, r1.stderr[-400:])

    MockHandler.max_inflight = 0
    par_out = os.path.join(tmp, "par.json")
    r2 = run_harness(base, data, par_out,
                     ["--n", "6", "--concurrency", "4", "--checkpoint-every", "2"])
    par_peak = MockHandler.max_inflight
    check("--concurrency 4 run exits 0", r2.returncode == 0, r2.stderr[-400:])

    # D12: without this the next assertion is vacuous — a "parallel" run that
    # actually executed serially would trivially match the serial one.
    check("mock observed 1 in-flight request at --concurrency 1",
          seq_peak == 1, f"peak={seq_peak}")
    check("mock observed >1 in-flight requests at --concurrency 4",
          par_peak > 1, f"peak={par_peak}")

    with open(seq_out) as f:
        a = json.load(f)
    with open(par_out) as f:
        b = json.load(f)
    def strip(d):
        """Drop the one field that is legitimately timing-dependent."""
        return [{k: v for k, v in it.items() if k != "seconds"} for it in d["items"]]

    check("concurrent run produces identical records (order included)",
          strip(a) == strip(b))
    check("concurrent run produces identical accuracy",
          a["summary"]["correct"] == b["summary"]["correct"] == 6)
    check("--concurrency 4 emits PROGRESS markers", "PROGRESS[" in r2.stdout)

    print("\n[10] resume")
    MockHandler.n_completions = 0
    res_out = os.path.join(tmp, "resume.json")
    r = run_harness(base, data, res_out, ["--n", "3"])
    first = MockHandler.n_completions
    check("first pass answered 3 problems", first == 3, f"{first}")
    # Second pass WITHOUT --no-resume must re-request only the new indices.
    cmd = [sys.executable, os.path.join(HERE, "run_gsm8k.py"),
           "--data", data, "--out", res_out, "--n", "6", "--max-tokens", "256"]
    MockHandler.n_completions = 0
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env=dict(os.environ, BASE_URL=base), timeout=300)
    check("resume run exits 0", r.returncode == 0, r.stderr[-400:])
    check("resume re-requested only the 3 missing problems",
          MockHandler.n_completions == 3, f"{MockHandler.n_completions}")
    check("resume reports the full n=6", "6/6 = 100.0%" in r.stdout,
          r.stdout[-300:])
    check("RUNPLAN marker shows resumed=3 todo=3",
          "resumed=3 todo=3" in r.stdout, r.stdout[:300])


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        test_detector()
        test_extractor()
        tmp, data = test_gate(base)
        test_concurrency(base, tmp, data)
    finally:
        srv.shutdown()

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} assertion(s) failed: {FAILURES}")
        return 1
    print("PASS: all degeneracy / extraction / gate assertions hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
