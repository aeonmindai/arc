#!/bin/bash
# Throughput-vs-batch-size with MIXED prompt lengths — the measurement that
# proves or kills the bucket-shattering hypothesis.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS BEING TESTED
#
# `scheduler/default_scheduler.rs` partitioned the running set by
# `Sequence::cache_bucket_len` and `select_running_bucket` runs exactly ONE
# bucket per step. A CPU simulation of that rule (`bucket_shattering_law`,
# `default_scheduler.rs`) measures a law: with B sequences over D distinct cache
# lengths, far enough apart that the coalescence override is refused, exactly
# B/D of them run per step. At B=128 over 128 lengths 64 tokens apart, that is
# ONE. The other 127 are admitted, resident in KV, counted as `running` by the
# logger — and idle.
#
# That is a property of the scheduler's own logic. Whether it is what limits the
# H200's aggregate curve (session 8: B=32 -> 47.45 tok/s, B=128 -> 43.73, i.e.
# falling where it should rise) is a HARDWARE question, and this is it.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE DESIGN
#
# Four arms, each a full B sweep. Everything is held fixed across A/B and across
# C/D except the prompt-length distribution:
#
#   A  refused (no flags)                      UNIFORM prompt lengths
#   B  refused (no flags)                      MIXED   prompt lengths
#   C  granted (ARC_V4_XS_PER_SEQ=1 +          UNIFORM prompt lengths
#              ARC_MTP_PER_SEQ_KV=1)
#   D  granted (same)                          MIXED   prompt lengths
#
# 🔑 **A vs B is the whole test, and it needs none of this PR's code.** Same
# binary, same flags, same model, same token budget — only the prompt lengths
# differ. If shattering is real, B falls away from A as the batch grows. If
# A ≈ B at every B, length diversity is not what limits the fleet number and the
# hypothesis is dead. C vs D is what the fix is worth.
#
# MTP depth is a CONSTANT across all four arms, so the MTP chain is not a
# confound in the A-vs-B contrast. `--max-seqs 256` so the scheduler cap never
# pins the B=128 row (it did in session 8, at `--max-seqs 128`).
#
# ─────────────────────────────────────────────────────────────────────────────
# THE NUMBERS TO EXTRACT (in priority order)
#
# 1. 🔑 **`tok_s(B) / tok_s(A)` as a function of B** — the same binary and the
#    same flags, differing only in prompt-length spread. Shattering predicts
#    this ratio FALLS as B grows (the refused arm's effective batch is B/D while
#    A's is B). A ratio flat at ~1.0 across the sweep kills the hypothesis for
#    this workload, and that is a real, publishable answer either way.
#
# 2. `tok_s(B, B=128) / tok_s(A, B=128)` — the same thing, as one number.
#
# 3. `tok_s(D, B=128) / tok_s(B, B=128)` — what this PR is worth on the mixed
#    workload. Predicted > 1; the CPU law's ceiling for it is D_distinct.
#
# 4. Non-regression: `tok_s(*, B=1)` must agree across all four arms within
#    noise. B=1 is one sequence and one bucket under either rule; a difference
#    there means something other than admission moved.
#
# 5. `grep "cannot honour it" $LOG` in arms C/D must be EMPTY (the mode was
#    granted). In arms A/B it must cite `ARC_MTP_PER_SEQ_KV` — the flag, not a
#    blocker, because the flag is off.
#
# ⚠️ **The most direct fingerprint is NOT available today, and that is stated
# rather than faked.** The engine's `IntervalLogger` prints
# "<N> running, <M> waiting" every interval, and N is literally how many
# sequences the scheduler put in the forward — no throughput reasoning needed.
# But `throughput_logging_enabled` is only reachable from the Rust SDK builder
# (`with_throughput_logging`); `mistralrs serve` exposes no flag for it, so the
# line is never emitted by a server run. `scrape.py` below looks for it anyway
# and reports `NA` with a reason when it finds nothing — it must never report
# `0.00`, because a zero here would read as "no sequences ran" when it means
# "nobody was logging" (D18). Adding a `--throughput-log` flag to `serve` would
# make observable (1) available and is surfaced separately, not smuggled in
# here.
#
# ─────────────────────────────────────────────────────────────────────────────
# USAGE (self-contained, detached, D15 — the agent never calls `runcrate`)
#
#   MODEL=<v4-model-id> setsid nohup bash arc-tools/ragged_admission_curve.sh \
#     < /dev/null > /root/logs/ragged_curve.log 2>&1 &
#
#   watch -n30 cat /root/logs/ragged_curve.status
#
# Results land in $OUT/results.tsv (one row per arm × B) and $OUT/summary.txt.
set -u

MODEL="${MODEL:?set MODEL to the V4 model id or path}"
PORT="${PORT:-1234}"
OUT="${OUT:-/root/logs/ragged_curve}"
STATUS="${STATUS:-/root/logs/ragged_admission_curve.status}"
PREFLIGHT="${PREFLIGHT:-/root/arc-tools/gpu_box_preflight.sh}"
MTP_DEPTH="${MTP_DEPTH:-3}"
MAX_SEQS="${MAX_SEQS:-256}"
BATCHES="${BATCHES:-1 8 32 64 128}"
# Seconds of steady state per (arm, B) cell after warmup.
STEADY="${STEADY:-45}"
WARMUP="${WARMUP:-20}"
MISTRALRS="${MISTRALRS:-mistralrs}"

mkdir -p "$OUT" "$(dirname "$STATUS")"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }
: > "$STATUS"

# ── Gate on the staged preflight. A box that fails it produces numbers nobody
#    can attribute, which is worse than no numbers.
if [ -x "$PREFLIGHT" ]; then
  say "preflight: running $PREFLIGHT"
  if ! "$PREFLIGHT" >> "$OUT/preflight.log" 2>&1; then
    say "preflight: FAILED — see $OUT/preflight.log. Refusing to measure."
    exit 2
  fi
  say "preflight: OK"
else
  say "preflight: $PREFLIGHT not found or not executable. Refusing to measure."
  exit 2
fi

# ── The load driver. Mixed vs uniform prompt lengths, K concurrent workers,
#    each looping until told to stop. Emits one JSON line of aggregates.
cat > "$OUT/drive.py" <<'PYEOF'
import argparse, json, random, threading, time, urllib.request

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
          "kilo lima mike november oscar papa quebec romeo sierra tango ")

def prompt_of(words: int) -> str:
    return ("Summarise the following log excerpt in detail. " +
            (FILLER * ((words // 20) + 1))[: words * 6])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--steady", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--mixed", action="store_true")
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}/v1/completions"
    stop = threading.Event()
    lock = threading.Lock()
    counted = {"tokens": 0, "requests": 0, "errors": 0}
    t_steady_start = [None]

    # UNIFORM: every worker sends the same 256-word prompt, so every sequence
    # enters at the same length and (with equal max_tokens) stays in lockstep.
    # MIXED: prompt lengths spread over 32..2048 words, deterministically per
    # worker, which is what asynchronous arrival with real users produces. This
    # is the ONLY thing that differs between arms A and B.
    def worker(wid: int):
        rng = random.Random(1000 + wid)
        while not stop.is_set():
            words = rng.choice([32, 64, 128, 256, 512, 1024, 2048]) if a.mixed else 256
            body = json.dumps({
                "model": "default",
                "prompt": prompt_of(words),
                # A spread of output lengths too: a fleet's sequences do not all
                # finish on the same step, which is the other way lengths
                # diverge. Uniform arm keeps it fixed.
                "max_tokens": (rng.randint(a.max_tokens // 2, a.max_tokens)
                               if a.mixed else a.max_tokens),
                "temperature": 0,
            }).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    resp = json.load(r)
                n = resp.get("usage", {}).get("completion_tokens", 0)
                with lock:
                    if t_steady_start[0] is not None:
                        counted["tokens"] += n
                        counted["requests"] += 1
            except Exception:
                with lock:
                    counted["errors"] += 1

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(a.k)]
    for t in threads:
        t.start()
    time.sleep(a.warmup)
    with lock:
        t_steady_start[0] = time.time()
        counted["tokens"] = 0
        counted["requests"] = 0
    time.sleep(a.steady)
    with lock:
        elapsed = time.time() - t_steady_start[0]
        snap = dict(counted)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    snap["elapsed_s"] = elapsed
    snap["tok_s"] = snap["tokens"] / elapsed if elapsed > 0 else 0.0
    print(json.dumps(snap))

main()
PYEOF

# ── Mean `running` / `waiting` over the steady-state window, straight out of
#    the engine's own throughput log — IF anything emitted it. See observable
#    (1): today nothing does from a `serve` run, so this prints `NA` rather than
#    `0.00`. A zero would read as "no sequences ran"; NA reads as "nobody was
#    logging", and those are different claims (D18).
cat > "$OUT/scrape.py" <<'PYEOF'
import re, sys
pat = re.compile(r"Throughput \(T/s\) ([\d.]+).*?(\d+) running, (\d+) waiting")
tps, run, wait = [], [], []
for line in open(sys.argv[1], errors="ignore"):
    m = pat.search(line)
    if m:
        tps.append(float(m.group(1))); run.append(int(m.group(2))); wait.append(int(m.group(3)))
if not run:
    print("NA\tNA\tNA\t0")
    raise SystemExit(0)
# Drop the warmup half: the ramp is not the steady state.
half = len(run) // 2
run, wait, tps = run[half:], wait[half:], tps[half:]
mean = lambda v: (sum(v) / len(v)) if v else 0.0
print(f"{mean(run):.2f}\t{mean(wait):.2f}\t{mean(tps):.2f}\t{len(run)}")
PYEOF

printf 'arm\tflags\tprompts\tB\ttok_s\trequests\terrors\tmean_running\tmean_waiting\tlog_tps\tsamples\n' \
  > "$OUT/results.tsv"

run_arm() {
  local arm="$1" flagdesc="$2" prompts="$3"; shift 3
  local envprefix="$*"
  for b in $BATCHES; do
    local slog="$OUT/server.${arm}.B${b}.log"
    say "arm $arm ($flagdesc, $prompts prompts) B=$b — starting server"
    # shellcheck disable=SC2086
    env RUST_LOG="${RUST_LOG:-info}" $envprefix setsid nohup \
      "$MISTRALRS" serve -p "$PORT" -m "$MODEL" \
      --max-seqs "$MAX_SEQS" --mtp-depth "$MTP_DEPTH" \
      --prefix-cache-n 0 \
      < /dev/null > "$slog" 2>&1 &
    local spid=$!
    for _ in $(seq 1 900); do
      curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
      sleep 2
    done
    if ! curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1; then
      say "arm $arm B=$b — server never came up; see $slog"
      kill -TERM -"$spid" 2>/dev/null || true
      continue
    fi

    local mixflag=""
    [ "$prompts" = "mixed" ] && mixflag="--mixed"
    local res
    res=$(python3 "$OUT/drive.py" --port "$PORT" --k "$b" \
            --warmup "$WARMUP" --steady "$STEADY" $mixflag)
    kill -TERM -"$spid" 2>/dev/null || true
    sleep 8

    local tok_s reqs errs sched
    tok_s=$(python3 -c "import json,sys;print(f\"{json.loads(sys.argv[1])['tok_s']:.2f}\")" "$res")
    reqs=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['requests'])" "$res")
    errs=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['errors'])" "$res")
    sched=$(python3 "$OUT/scrape.py" "$slog")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$arm" "$flagdesc" "$prompts" "$b" "$tok_s" "$reqs" "$errs" "$sched" \
      >> "$OUT/results.tsv"
    say "arm $arm B=$b -> tok/s $tok_s, errors $errs, scheduler(running/waiting/tps/n) $sched"
  done
}

GRANT="ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1"

run_arm A "refused" uniform ""
run_arm B "refused" mixed   ""
run_arm C "granted" uniform "$GRANT"
run_arm D "granted" mixed   "$GRANT"

{
  echo "=== results.tsv ==="
  cat "$OUT/results.tsv"
  echo
  echo "=== the numbers ==="
  python3 - "$OUT/results.tsv" <<'PYEOF'
import sys, collections
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
by = {(r[0], int(r[3])): r for r in rows}
def f(arm, b, col):
    r = by.get((arm, b));  return float(r[col]) if r else float("nan")
maxb = max(int(r[3]) for r in rows)
print("(1) tok_s(B)/tok_s(A) across the sweep  <- FALLS with B iff shattering is real:")
for b in sorted({int(r[3]) for r in rows}):
    a_, b_ = f('A', b, 4), f('B', b, 4)
    print(f"      B={b:<4} mixed/uniform = {(b_ / a_ if a_ else float('nan')):.3f}"
          f"   (A={a_:.2f}, B={b_:.2f})")
run_b = by.get(('B', maxb), [""] * 8)[7]
print(f"    scheduler `running` at B={maxb}, arm B: {run_b}"
      "   (NA = `serve` cannot emit the throughput log; see the header)")
print(f"(2) tok_s(B)/tok_s(A) at B={maxb}: {f('B', maxb, 4) / f('A', maxb, 4):.3f} "
      f"  <- cost of length diversity, fix OFF")
print(f"(3) tok_s(D)/tok_s(B) at B={maxb}: {f('D', maxb, 4) / f('B', maxb, 4):.3f} "
      f"  <- what this PR is worth on the mixed workload")
print("(4) B=1 across arms (must agree within noise): "
      + ", ".join(f"{a}={f(a, 1, 4):.2f}" for a in "ABCD"))
print("(5) errors: "
      + ", ".join(f"{a}={sum(int(r[6]) for r in rows if r[0] == a)}" for a in "ABCD"))
PYEOF
  echo
  echo "=== refusal reasons (arms A/B must cite ARC_MTP_PER_SEQ_KV; C/D must be empty) ==="
  grep -h "cannot honour it" "$OUT"/server.*.log | sort -u || echo "(none)"
} | tee "$OUT/summary.txt"

say "DONE — $OUT/summary.txt"
