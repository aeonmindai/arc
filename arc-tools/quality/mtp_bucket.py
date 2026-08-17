#!/usr/bin/env python3
"""Bucket per-request MTP acceptance lines by batch size, using wall-clock fences.

WHY THIS EXISTS
---------------
`MTP[b=<B>]` markers are emitted by AcceptanceTelemetry::log(), which only fires
when a 64-proposed-token boundary is crossed AND the env gate is on. When they
do not appear, the only MTP data a serve log carries is the per-request line:

    MTP[req=508] accept_rate=0.2143 accepted=3 proposed=14 steps=60 \
                 drafted_steps=6 committed=63 tok_per_step=1.0500 ...

which has no batch label. A sweep that runs several batch sizes in one process
therefore leaves those lines unattributable, and the run produces nothing.

If the harness fences each batch size in wall-clock time, the lines can be
attributed by timestamp. This reads those fences and sums the RAW COUNTS per
bucket — never the per-request ratios.

READ tok_per_step, NOT accept_rate
----------------------------------
A sequence that has saturated its draft window proposes 0 tokens on that step.
It contributes nothing to the accept_rate denominator, so accept_rate stays
flattering while the actual token yield collapses. tok_per_step = committed /
steps counts every step including the ones that drafted nothing, which is why it
is the number that answers "does MTP still multiply at B=128".

Averaging per-request tok_per_step values would be wrong for the same reason —
it weights a 3-step request equally with a 60-step one. This sums numerators and
denominators separately, which is the same arithmetic the engine's own marker
does, just aggregated at a different point.

Stdlib only.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

# Any leading ISO-8601-ish stamp: 2026-08-17T10:23:45(.123456)(Z)
TS = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})")
REQ = re.compile(r"MTP\[req=(\d+)\]\s+(.*)")
FENCE = re.compile(
    r"FENCE_(START|END)\s+B=(\d+)\s+(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")

# Raw integer counters on the marker. Ratios (accept_rate, tok_per_step,
# mean_batch, tok_per_batch_step) are deliberately NOT summed.
COUNTS = ("accepted", "proposed", "steps", "drafted_steps", "committed")


def _norm(ts):
    return ts.replace(" ", "T")[:19]


def parse_fences(path):
    """[(B, start, end)] in file order."""
    open_start = {}
    out = []
    with open(path) as f:
        for line in f:
            m = FENCE.search(line)
            if not m:
                continue
            kind, b, ts = m.group(1), int(m.group(2)), _norm(m.group(3))
            if kind == "START":
                open_start[b] = ts
            elif b in open_start:
                out.append((b, open_start.pop(b), ts))
    return out


def parse_requests(path):
    """[(timestamp, {field: int})] for every MTP[req=] line carrying a stamp."""
    out = []
    undated = 0
    with open(path, errors="replace") as f:
        for line in f:
            m = REQ.search(line)
            if not m:
                continue
            t = TS.search(line)
            if not t:
                undated += 1
                continue
            fields = {}
            for kv in m.group(2).split():
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                if k in COUNTS:
                    try:
                        fields[k] = int(v)
                    except ValueError:
                        pass
            if fields:
                out.append((_norm(f"{t.group(1)}T{t.group(2)}"), fields))
    return out, undated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-log", required=True)
    ap.add_argument("--fences", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        fences = parse_fences(args.fences)
        reqs, undated = parse_requests(args.serve_log)
    except OSError as e:
        print(f"MTPBUCKET: cannot read inputs ({e}) — UNMEASURED, not 0%")
        return 1

    if not reqs:
        print("MTPBUCKET: no MTP[req=] lines with timestamps in the serve log. "
              f"({undated} MTP[req=] lines had no parseable timestamp.) "
              "MTP acceptance is UNMEASURED for this run — do NOT read it as 0%.")
        return 1
    if not fences:
        print("MTPBUCKET: no FENCE_START/FENCE_END pairs — cannot attribute "
              "per-request lines to a batch size. UNMEASURED.")
        return 1

    buckets = defaultdict(lambda: defaultdict(int))
    n_req = defaultdict(int)
    unattributed = 0
    for ts, fields in reqs:
        hit = None
        for b, start, end in fences:
            if start <= ts <= end:
                hit = b
                break
        if hit is None:
            unattributed += 1
            continue
        n_req[hit] += 1
        for k in COUNTS:
            buckets[hit][k] += fields.get(k, 0)

    result = {}
    for b in sorted(buckets):
        c = buckets[b]
        tps = (c["committed"] / c["steps"]) if c["steps"] else None
        rate = (c["accepted"] / c["proposed"]) if c["proposed"] else None
        result[b] = {
            "requests": n_req[b],
            "accept_rate": round(rate, 4) if rate is not None else None,
            "tok_per_step": round(tps, 4) if tps is not None else None,
            **{k: c[k] for k in COUNTS},
        }
        print(
            f"MTPBUCKET[b={b}] tok_per_step={result[b]['tok_per_step']} "
            f"accept_rate={result[b]['accept_rate']} "
            f"accepted={c['accepted']} proposed={c['proposed']} steps={c['steps']} "
            f"drafted_steps={c['drafted_steps']} committed={c['committed']} "
            f"requests={n_req[b]}"
        )
        # A step that drafted nothing still counts. If NO step ever drafted, the
        # draft KV never primed and this is a defect, not a rate.
        if c["drafted_steps"] == 0 and c["steps"] > 0:
            print(f"MTPBUCKET[b={b}] WARNING: drafted_steps=0 with steps>0 — the "
                  f"MTP block loaded but never drafted. That is a defect, not an "
                  f"acceptance rate.")

    if unattributed:
        print(f"MTPBUCKET: {unattributed} request lines fell outside every fence "
              f"(warmup or between-batch traffic) and were dropped, not "
              f"reassigned.")
    if undated:
        print(f"MTPBUCKET: {undated} MTP[req=] lines had no timestamp and were "
              f"dropped.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"by_batch": result,
                       "unattributed_lines": unattributed,
                       "undated_lines": undated}, f, indent=2)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
