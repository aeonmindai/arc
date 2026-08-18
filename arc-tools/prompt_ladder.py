#!/usr/bin/env python3
"""Prompt-length latency ladder — the task-#22 curve, measured not asserted.

The last run before teardown reported, serving V4 on an H200:
    ~9 words   -> 100 s for 8 tokens
    ~200 words -> 386 s for 8 tokens
    ~1055 words-> connection refused (server dead)

This reproduces exactly that shape so a fix can be tested against it. It is
deliberately tiny (3 requests x 8 tokens) so it costs seconds of GPU time.

It samples at temperature 1.0 / top_p 0.95 — the "measured-good chat.py"
setting, and the ONLY configuration that reaches the big-vocab GPU radix
top-k sampler path (`sampler.rs`: temperature must be > 1e-7 AND top_p must be
< 1.0, otherwise the argmax / `sample_fast` paths are taken instead). Running
this ladder greedy would exercise different code and prove nothing about the
sampler.

Reports s/token per rung and, crucially, distinguishes a SLOW rung from a DEAD
server: a connection error is recorded as `"error": "..."` rather than being
allowed to abort the whole ladder, because "it died at 1055 words" is itself
the measurement.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors, integrated circuits, and modern accelerators. "
)

# (label, approximate word count) — matches the reported failure ladder.
RUNGS = [("short", 9), ("medium", 200), ("long", 1055)]


def make_prompt(words: int) -> str:
    if words <= 12:
        return "Explain what a GPU does in one sentence."
    body = []
    n = 0
    while n < words:
        body.append(FILLER)
        n += len(FILLER.split())
    return "".join(body) + "\n\nSummarize the passage above in one sentence."


def one(port: int, prompt: str, max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # See module docstring: these two values select the sampler path.
        "temperature": 1.0,
        "top_p": 0.95,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # A dead server is a result, not a crash of this script.
        return {"error": repr(exc), "wall_s": round(time.time() - t0, 2)}
    wall = time.time() - t0
    usage = body.get("usage", {}) or {}
    completion = usage.get("completion_tokens") or max_tokens
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "s_per_token": round(wall / completion, 3) if completion else None,
        "tok_per_s": round(completion / wall, 3) if wall > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 1234)))
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="ladder.json")
    args = ap.parse_args()

    results = {}
    for label, words in RUNGS:
        prompt = make_prompt(words)
        actual_words = len(prompt.split())
        print(f"[ladder] {label}: ~{actual_words} words, {args.max_tokens} tokens ...", flush=True)
        r = one(args.port, prompt, args.max_tokens, args.timeout)
        r["approx_words"] = actual_words
        results[label] = r
        print(f"[ladder] {label}: {json.dumps(r)}", flush=True)
        if "error" in r:
            # Record and keep going — the later rungs will also fail, and that
            # pattern (which rung killed it) is what we want on the record.
            print(f"[ladder] {label} FAILED — server may be dead", flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[ladder] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
