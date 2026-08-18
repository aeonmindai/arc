#!/usr/bin/env python3
"""Per-call cost of the FP8 KV round trip, from an nsys trace.

Parent system: ArcLab (measuring ArcInfer / ArcKV / Fp8).

WHY PER-CALL AND NOT tok/s
    End-to-end tok/s on this shared box is not trustworthy: four separate
    end-to-end numbers here turned out to be pure environment in one night, and
    a naive A-B-A-B delta is *biased*, not merely noisy, because at equal
    spacing the arms differ by exactly one slot of drift. A per-call cost taken
    from the trace is immune to both — it does not depend on how long the step
    took, only on what the step did. The `GpuApprox` conclusion survived a bad
    night for exactly this reason: its evidence was `kv_fp8_quant` 73.49 ->
    134.18 us/call, not a throughput delta.

WHAT IT REPORTS
    BEFORE arm: the host time spent inside `cuMemcpyDtoHAsync_v2` per decode
    step. That is the real cost of the CPU round trip -- an "async" copy taking
    ~109 us of HOST time is a pageable-memory staged copy, i.e. blocking, which
    is also why a CUDA graph cannot record it.
    AFTER arm: the device time of the two fused kernels per decode step, and
    their calls/step (which must be 43 -- one per layer -- or the fused path did
    not engage on every layer and the number is not what it claims).

Exits 2 on analysis failure, never 1.

Usage: kv_fp8_percall.py <trace.sqlite> [tail_seconds]
"""
import collections
import sqlite3
import sys


def die(msg):
    print(f"FATAL {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) < 2:
        die("usage: kv_fp8_percall.py <trace.sqlite> [tail_seconds]")
    tail_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    db = sqlite3.connect(sys.argv[1])
    cur = db.cursor()
    strings = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}
    KT = "CUPTI_ACTIVITY_KIND_KERNEL"
    MT = "CUPTI_ACTIVITY_KIND_MEMCPY"
    RT = "CUPTI_ACTIVITY_KIND_RUNTIME"

    end = cur.execute(f"SELECT MAX(end) FROM {KT}").fetchone()[0]
    if end is None:
        die("no kernels in the trace")
    win0 = end - int(tail_s * 1e9)

    # Steps = the once-per-step logits readback.
    big = list(cur.execute(
        f"SELECT COUNT(*) FROM {MT} WHERE start>=? AND copyKind=2 AND bytes>100000",
        (win0,)))
    steps = big[0][0] if big else 0
    if steps < 20:
        die(f"only {steps} steps in the window; widen tail_seconds")
    print(f"steps in window        {steps}")

    # ---- BEFORE-arm evidence: host time inside the blocking D2H.
    tot_ns = 0
    n = 0
    for st, en in cur.execute(
            f"SELECT start, end FROM {RT} WHERE start>=? AND nameId IN "
            f"(SELECT id FROM StringIds WHERE value='cuMemcpyDtoHAsync_v2')",
            (win0,)):
        tot_ns += en - st
        n += 1
    if n:
        print(f"cuMemcpyDtoHAsync_v2   {n / steps:8.2f} calls/step  "
              f"{tot_ns / n / 1000.0:8.2f} us/call (HOST time)  "
              f"{tot_ns / steps / 1e6:8.3f} ms/step")
    else:
        print("cuMemcpyDtoHAsync_v2      0.00 calls/step")

    # ---- AFTER-arm evidence: the fused kernels' device time.
    dev = collections.defaultdict(lambda: [0, 0])
    for name_id, st, en in cur.execute(
            f"SELECT shortName, start, end FROM {KT} WHERE start>=?", (win0,)):
        nm = strings.get(name_id, str(name_id))
        if "arc_kv_fp8" in nm:
            d = dev[nm]
            d[0] += 1
            d[1] += en - st
    if dev:
        print("fused kernels (device time):")
        tot = 0.0
        for nm, (c, ns) in sorted(dev.items()):
            per_step = c / steps
            print(f"   {nm[:44]:44s} {per_step:7.2f} calls/step "
                  f"{ns / c / 1000.0:7.2f} us/call "
                  f"{ns / steps / 1e6:7.4f} ms/step")
            tot += ns / steps / 1e6
            if abs(per_step - round(per_step)) > 0.05:
                print(f"      WARN {nm} is not an integer per step")
        print(f"   fused total{'':38s} {tot:7.4f} ms/step")
    else:
        print("fused kernels          ABSENT from this trace "
              "(this is the before arm, or the fused path never engaged)")


if __name__ == "__main__":
    main()
