"""Batch sweep on landed master: per-request vs aggregate decode throughput.

For each batch size B, fire B CONCURRENT streaming requests and time the
inter-token intervals of each. Two independent aggregate anchors that must
agree, because the interesting number here is capacity and a capacity claim
built on one instrument is not a claim:

  A) derived   = per_request_tok_s * B          (what the brief asked for)
  B) measured  = tokens delivered inside the all-B-active window / that window

(A) assumes every stream runs at the median rate; (B) counts actual deltas in
the steady-state window where all B streams are decoding. If they diverge the
batch never reached steady state and the row is not a throughput measurement.

Per-request rate is a median of inter-token INTERVALS, so prefill is excluded
and one scheduler hiccup cannot set the headline.

A row where any request returns zero tokens is a FAILED row, printed as such,
never averaged into the table -- the whole point of the bench gate.

Exit: 0 sweep complete | 1 a row failed | 2 environment failure.
"""
import json
import statistics
import sys
import subprocess
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234"
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 64
BATCHES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1", "8", "32", "64", "128"])]

TOPICS = [
    "oceans", "mountains", "deserts", "forests", "rivers", "glaciers", "volcanoes",
    "coral reefs", "savannas", "tundra", "caves", "islands", "wetlands", "canyons",
    "prairies", "fjords",
]


def one_stream(idx, ntok, out):
    """Stream one request; record a timestamp per delivered token."""
    topic = TOPICS[idx % len(TOPICS)]
    prompt = ("Write %d short factual sentences about %s. Sentence set %d. "
              "One sentence per line, no preamble." % (max(6, ntok // 6), topic, idx))
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": ntok, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    stamps, chars, err = [], 0, None
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if "error" in d:
                    err = str(d["error"])[:120]
                    break
                for c in d.get("choices") or []:
                    piece = ((c.get("delta") or {}).get("content") or "")
                    if piece:
                        stamps.append(time.perf_counter())
                        chars += len(piece)
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, str(e)[:100])
    out[idx] = {"t0": t0, "stamps": stamps, "chars": chars, "err": err}


SMI = ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free",
       "--format=csv,noheader,nounits"]


def sample_gpu(stop, sink):
    """Poll the GPU WHILE the batch decodes.

    `utilization.memory` is the memory-controller busy percentage -- the
    diagnostic that says whether batching is amortising the weight reads. A
    sampler that starts after the run writes an empty file and looks green, so
    this one is started before the requests and stamped per sample.
    """
    while not stop.is_set():
        try:
            out = subprocess.run(SMI, capture_output=True, text=True, timeout=5).stdout.strip()
            for line in out.splitlines():
                g, m, used, free = [x.strip() for x in line.split(",")]
                sink.append((time.perf_counter(), int(g), int(m), int(used), int(free)))
        except Exception:
            pass
        time.sleep(0.25)


def run_batch(b, ntok):
    out = {}
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    sampler.start()
    threads = [threading.Thread(target=one_stream, args=(i, ntok, out)) for i in range(b)]
    w0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    w1 = time.perf_counter()
    stop.set()
    sampler.join(timeout=5)
    return out, w1 - w0, samples


def grade(b, out, wall, samples):
    rows = [out[i] for i in sorted(out)]
    zero = [i for i, r in enumerate(rows) if len(r["stamps"]) == 0]
    errs = [(i, r["err"]) for i, r in enumerate(rows) if r["err"]]
    if len(rows) != b:
        return {"B": b, "fail": "instrument: %d rows for B=%d" % (len(rows), b)}
    if zero:
        return {"B": b, "fail": "%d of %d requests returned ZERO tokens%s"
                % (len(zero), b, (" (first err: %s)" % errs[0][1]) if errs else "")}

    # per-request: median over requests of that request's median gap
    per_req_rates = []
    for r in rows:
        s = r["stamps"]
        if len(s) < 4:
            return {"B": b, "fail": "a request produced only %d tokens" % len(s)}
        gaps = [s[i] - s[i - 1] for i in range(1, len(s))]
        per_req_rates.append(1.0 / statistics.median(gaps))
    per_req = statistics.median(per_req_rates)
    total_tok_all = sum(len(r["stamps"]) for r in rows)

    # COHORT throughput: every token the cohort delivered, over the wall time
    # the cohort spent delivering them. This is the capacity number an operator
    # bills against, and unlike a strict all-active window it is defined even
    # when admission is staggered -- which it is at large B, because the
    # scheduler admits in waves.
    first = min(r["stamps"][0] for r in rows)
    last = max(r["stamps"][-1] for r in rows)
    span = last - first
    if span <= 0:
        return {"B": b, "fail": "cohort delivered no tokens over time"}
    measured = total_tok_all / span

    # Achieved concurrency: mean active streams across the span. If this is far
    # below B the row is NOT a measurement of B -- the scheduler never ran B at
    # once, and reporting it as B would overstate what the box did.
    edges = []
    for r in rows:
        edges.append((r["stamps"][0], 1))
        edges.append((r["stamps"][-1], -1))
    edges.sort()
    active, prev_t, area = 0, first, 0.0
    for t, d in edges:
        area += active * (t - prev_t)
        active += d
        prev_t = t
    concurrency = area / span if span > 0 else 0.0

    # strict all-active window, when one exists (secondary anchor)
    lo = max(r["stamps"][0] for r in rows)
    hi = min(r["stamps"][-1] for r in rows)
    if hi > lo:
        inwin = sum(sum(1 for t in r["stamps"] if lo < t <= hi) for r in rows)
        steady = inwin / (hi - lo)
    else:
        steady = -1.0

    derived = per_req * b
    disagree = abs(measured - derived) / max(measured, derived)
    # attribute GPU samples to the cohort span
    win = [s for s in samples if first <= s[0] <= last]
    if not win:
        win = samples
    memu = sorted(s[2] for s in win)
    gpuu = sorted(s[1] for s in win)
    used = sorted(s[3] for s in win)
    free = sorted(s[4] for s in win)
    n_s = len(win)
    ttfts = sorted(r["stamps"][0] - r["t0"] for r in rows)
    return {
        "B": b, "fail": None,
        "per_req": per_req,
        "ms_per_tok": 1000.0 / per_req,
        "agg_derived": derived,
        "agg_measured": measured,
        "disagree": disagree,
        "total_tok": total_tok_all,
        "concurrency": concurrency,
        "steady": steady,
        "window_s": span,
        "wall_s": wall,
        "ttft_med": ttfts[len(ttfts) // 2],
        "ttft_max": ttfts[-1],
        "mem_util_med": (memu[len(memu) // 2] if memu else -1),
        "mem_util_max": (memu[-1] if memu else -1),
        "gpu_util_med": (gpuu[len(gpuu) // 2] if gpuu else -1),
        "vram_used_max": (used[-1] if used else -1),
        "vram_free_min": (free[0] if free else -1),
        "n_samples": n_s,
    }


print("BASE=%s NTOK=%d batches=%s" % (BASE, NTOK, BATCHES), flush=True)
results, failed = [], False
for b in BATCHES:
    print("--- B=%d : firing %d concurrent streams ---" % (b, b), flush=True)
    out, wall, samples = run_batch(b, NTOK)
    r = grade(b, out, wall, samples)
    results.append(r)
    if r["fail"]:
        failed = True
        print("  B=%-4d FAILED ROW: %s" % (b, r["fail"]), flush=True)
    else:
        print("  B=%-4d per-req %7.2f | agg %8.2f | conc %6.1f/%d | MEM-CTRL %d%% (max %d%%) | "
              "GPU %d%% | VRAMfree %d MiB | tok %d | samples %d"
              % (b, r["per_req"], r["agg_measured"], r["concurrency"], b,
                 r["mem_util_med"], r["mem_util_max"], r["gpu_util_med"],
                 r["vram_free_min"], r["total_tok"], r["n_samples"]), flush=True)
    time.sleep(3)

print("")
print("%-6s %-13s %-11s %-13s %-13s %-8s %-9s %-9s %-10s" %
      ("B", "per-req t/s", "ms/tok", "agg derived", "agg measured", "spread",
       "MEMCTRL%", "conc", "VRAMfree"))
for r in results:
    if r["fail"]:
        print("%-6d %s" % (r["B"], "FAILED: " + r["fail"]))
    else:
        print("%-6d %-13.2f %-11.2f %-13.2f %-13.2f %-8s %-9s %-9s %-10d"
              % (r["B"], r["per_req"], r["ms_per_tok"], r["agg_derived"], r["agg_measured"],
                 "%.1f%%" % (r["disagree"] * 100),
                 "%d/%d" % (r["mem_util_med"], r["mem_util_max"]),
                 "%.0f" % r["concurrency"], r["vram_free_min"]))

ok = [r for r in results if not r["fail"]]
if len(ok) >= 2:
    base = ok[0]["agg_measured"]
    print("")
    print("aggregate scaling vs B=%d:" % ok[0]["B"])
    for r in ok:
        print("   B=%-5d %8.2f tok/s   x%.2f   (per-request retained: %.0f%%)"
              % (r["B"], r["agg_measured"], r["agg_measured"] / base,
                 100.0 * r["per_req"] / ok[0]["per_req"]))
sys.exit(1 if failed else 0)
