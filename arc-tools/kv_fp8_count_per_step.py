#!/usr/bin/env python3
"""Per-decode-step D2H and kernel-launch counts from an nsys sqlite.

Parent system: ArcLab (measuring ArcInfer / ArcKV / Fp8).

WHY THIS EXISTS
    The 43 blocking `cuMemcpyDtoHAsync_v2` per V4 decode step are invisible to
    the obvious instrument: `*Synchronize*` is 0.0 calls/step in this workload,
    so anyone counting `cudaStreamSynchronize` concludes "no syncs" and is
    wrong. This counts the copies themselves.

THE STEP COUNT IS PINNED FROM THE TRACE, NOT ASSUMED
    A per-step number divided by a guessed step count is a fabricated number.
    Two independent anchors must agree:
      A. the once-per-step logits readback -- the largest DtoH size, whose
         occurrence count IS the number of steps in the window;
      B. structural agreement -- kernels whose counts are integer multiples of
         that same step count (the 43-layer loop makes many). Fewer than three
         such kernels means the window is not steady-state decode and the
         script exits 2.

Exits 2 on environment/analysis failure, never 1.

Usage: kv_fp8_count_per_step.py <trace.sqlite> [tail_seconds]
"""
import collections
import sqlite3
import sys


def die(msg):
    print(f"FATAL {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) < 2:
        die("usage: kv_fp8_count_per_step.py <trace.sqlite> [tail_seconds]")
    tail_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    db = sqlite3.connect(sys.argv[1])
    cur = db.cursor()
    strings = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}
    KT = "CUPTI_ACTIVITY_KIND_KERNEL"
    MT = "CUPTI_ACTIVITY_KIND_MEMCPY"

    end = cur.execute(f"SELECT MAX(end) FROM {KT}").fetchone()[0]
    if end is None:
        die("no kernels in the trace")
    win0 = end - int(tail_s * 1e9)

    # copyKind 2 == device-to-host.
    sizes = list(cur.execute(
        f"SELECT bytes, COUNT(*) FROM {MT} WHERE start>={win0} AND copyKind=2 "
        f"GROUP BY bytes ORDER BY COUNT(*) DESC"))
    if not sizes:
        die("no DtoH copies in the window")
    total_d2h = sum(c for _, c in sizes)

    # ---- Anchor A: the logits readback, once per decode step.
    big = [(b, c) for b, c in sizes if b > 100_000]
    if not big:
        die("no large (>100 kB) DtoH found; cannot pin the step count")
    steps = max(c for _, c in big)
    if steps < 20:
        die(f"only {steps} steps in the window; widen tail_seconds")

    kc = collections.Counter()
    for (n,) in cur.execute(f"SELECT shortName FROM {KT} WHERE start>={win0}"):
        kc[strings.get(n, str(n))] += 1
    if not kc:
        die("no kernels in the window")

    # ---- Anchor B: kernels whose counts are integer multiples of `steps`.
    agreeing = []
    for name, c in kc.items():
        k = c / steps
        if k >= 0.98 and abs(k - round(k)) <= 0.02 * max(round(k), 1):
            agreeing.append((name, c, round(k)))
    if len(agreeing) < 3:
        die(f"step count {steps} unconfirmed: only {len(agreeing)} kernels are "
            f"integer multiples of it; the window is probably not steady-state "
            f"decode")

    total_kern = sum(kc.values())
    print(f"window                 last {tail_s:.0f} s")
    print(f"steps (logits anchor)  {steps}")
    print(f"steps confirmed by     {len(agreeing)} kernels at integer multiples")
    print(f"D2H total              {total_d2h}")
    print(f"D2H PER STEP           {total_d2h / steps:.2f}")
    print(f"kernels total          {total_kern}")
    print(f"LAUNCHES PER STEP      {total_kern / steps:.1f}")
    print("DtoH by size:")
    for b, c in sizes[:8]:
        print(f"   {b:>12,d} B x {c:>9,d}   {c / steps:8.2f}/step")
    print("anchor-B kernels (count = k x steps):")
    for name, c, k in sorted(agreeing, key=lambda t: -t[1])[:6]:
        print(f"   {c:>9,d} = {k:>4d} x steps   {name[:52]}")

    # ---- CUDA API calls per step. `cuMemcpyDtoHAsync_v2` is the one that
    # matters here and it is NOT visible as a "sync": `*Synchronize*` is 0.0
    # calls/step in this workload. An "async" copy costing ~109 us of host time
    # is a pageable-memory staged copy, i.e. blocking, and a CUDA graph cannot
    # record one.
    api = collections.Counter()
    try:
        for (n,) in cur.execute(
                "SELECT nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=?",
                (win0,)):
            api[strings.get(n, str(n))] += 1
    except sqlite3.Error:
        print("(no CUPTI_ACTIVITY_KIND_RUNTIME table; API counts unavailable)")
        return
    watch = ("cuMemcpyDtoHAsync_v2", "cuLaunchKernel", "cuMemAllocAsync",
             "cuMemFreeAsync", "cuMemcpyHtoDAsync_v2", "cuMemsetD8Async")
    print("CUDA API per step:")
    for name in watch:
        print(f"   {api.get(name, 0) / steps:10.2f}/step  {name}")
    syncs = sum(c for n, c in api.items() if "Synchronize" in n)
    print(f"   {syncs / steps:10.2f}/step  ALL *Synchronize* "
          f"(this is why counting syncs finds nothing)")


if __name__ == "__main__":
    main()
