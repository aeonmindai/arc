#!/usr/bin/env python3
"""Prompt-length x batch ladder against ONE already-running server config.

Settles whether the session-8 prefill numbers (9w = 100.15 s, 198w = 386.15 s,
~1,055w = server dead) are a property of V4, or of the configuration and binary
that produced them. A later canary on a provenance-verified build measured
512w in 3.2 s and 1100w in 6.6 s at B=1 — a ~100x disagreement with 198w in
386 s, so at most one of the two is a fact about the model.

Emits one TSV row per cell, with wall time AND peak VRAM, because the two
candidate mechanisms are distinguishable by memory: a paged/TurboQuant KV path
thrashing shows as VRAM climbing with prompt length, while a pure compute cliff
does not.

CELL TIMEOUT IS A RESULT, NOT A FAILURE. A cell that exceeds --cell-timeout is
recorded as TIMEOUT with the elapsed bound, never dropped — session 8's headline
number was 386 s, so a harness that quietly skipped slow cells would erase
exactly the evidence in question. Two consecutive timeouts in a row abort the
remainder of that row to bound rental cost, and the abort is recorded too.

Stdlib only.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

WORD = "fleet capacity mechanism units table cite source file line "
TAIL = " Name three primary colors."


def build_prompt(words):
    """~`words` whitespace-delimited words. Reported length is the real count."""
    unit = WORD.split()
    out = []
    while len(out) < words:
        out.extend(unit)
    return " ".join(out[:words]) + TAIL


def peak_vram_sampler(stop_evt, out, interval=0.4):
    """Max MiB observed while the cell runs. Absent nvidia-smi -> None."""
    peak = 0
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            v = int(r.stdout.strip().splitlines()[0])
            peak = max(peak, v)
        except Exception:  # noqa: BLE001 — a sampling gap must not fail the cell
            pass
        stop_evt.wait(interval)
    out["peak_mib"] = peak or None


def fire(url, prompt, max_tokens, timeout, out, idx):
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = sum(1 for _ in r)
        out[idx] = ("ok", round(time.monotonic() - t0, 2), n)
    except urllib.error.HTTPError as e:
        out[idx] = (f"HTTP{e.code}", round(time.monotonic() - t0, 2), 0)
    except Exception as e:  # noqa: BLE001 — every failure is one datum
        out[idx] = (f"{type(e).__name__}", round(time.monotonic() - t0, 2), 0)


def run_cell(url, words, bsz, args):
    prompt = build_prompt(words)
    stop = threading.Event()
    vram = {}
    sampler = threading.Thread(target=peak_vram_sampler, args=(stop, vram),
                               daemon=True)
    sampler.start()

    out = {}
    threads = [threading.Thread(target=fire,
                                args=(url, prompt, args.max_tokens,
                                      args.cell_timeout, out, i))
               for i in range(bsz)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.cell_timeout + 30)
    wall = round(time.monotonic() - t0, 2)
    stop.set()
    sampler.join(timeout=5)

    results = [out.get(i, ("NORESULT", None, 0)) for i in range(bsz)]
    ok = sum(1 for r in results if r[0] == "ok")
    bad = [r[0] for r in results if r[0] != "ok"]
    status = "OK" if ok == bsz else ("TIMEOUT" if wall >= args.cell_timeout
                                     else "FAIL")
    return {
        "words": len(prompt.split()),
        "B": bsz,
        "status": status,
        "wall_s": wall,
        "ok": ok,
        "failed": bsz - ok,
        "first_error": bad[0] if bad else "",
        "peak_mib": vram.get("peak_mib"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:1234")
    ap.add_argument("--words", default="40,128,512,1100,2048")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--cell-timeout", type=int, default=420,
                    help="a cell exceeding this is RECORDED as TIMEOUT, never "
                         "dropped; session 8's headline was 386 s")
    ap.add_argument("--config", default="unknown",
                    help="label for this server configuration, e.g. pagedon")
    ap.add_argument("--state", default="unknown",
                    help="RESOLVED runtime state read back from the server's own "
                         "startup log (kv cache type, paged on/off, prefix "
                         "caching). Stamped on every row so a slow cell carries "
                         "the mechanism next to it instead of a correlation.")
    ap.add_argument("--revision", default="unknown",
                    help="git revision the SERVER logged. Recorded per row so a "
                         "build difference is attributable rather than assumed.")
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    url = f"{args.base_url}/v1/chat/completions"
    words = [int(w) for w in args.words.split(",") if w.strip()]
    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    rows = []
    hdr = ("config\trevision\twords\tB\tstatus\twall_s\tok\tfailed\tpeak_mib\t"
           "resolved_state\tfirst_error")
    print(hdr)
    for bsz in batches:
        consecutive_timeouts = 0
        for w in words:
            r = run_cell(url, w, bsz, args)
            r["config"] = args.config
            rows.append(r)
            print(f"{args.config}\t{args.revision}\t{r['words']}\t{r['B']}\t"
                  f"{r['status']}\t{r['wall_s']}\t{r['ok']}\t{r['failed']}\t"
                  f"{r['peak_mib']}\t{args.state}\t{r['first_error']}",
                  flush=True)
            if r["status"] == "TIMEOUT":
                consecutive_timeouts += 1
            else:
                consecutive_timeouts = 0
            if consecutive_timeouts >= 2:
                print(f"{args.config}\t{args.revision}\t-\t{bsz}\tABORTED_ROW\t"
                      f"-\t-\t-\t-\t{args.state}\ttwo consecutive timeouts; "
                      f"remaining words skipped to bound rental cost",
                      flush=True)
                break

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write(hdr + "\n")
            for r in rows:
                f.write(f"{r['config']}\t{args.revision}\t{r['words']}\t{r['B']}\t"
                        f"{r['status']}\t{r['wall_s']}\t{r['ok']}\t{r['failed']}\t"
                        f"{r['peak_mib']}\t{args.state}\t{r['first_error']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
