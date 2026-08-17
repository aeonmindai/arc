#!/usr/bin/env bash
# ArcSched — decode admission A/B.
#
# Measures what `DecodeAdmission` (mistralrs-core/src/paged_attention/scheduler.rs)
# is worth on hardware. The two arms run the SAME binary, the SAME model and the
# SAME workload; the only difference is one environment variable, so nothing but
# the scheduling rule can move the number.
#
#   arm BUCKETED : ARC_SCHED_BUCKETED_DECODE=1  — partition the decode batch by KV
#                  length, run the shortest bucket, preempt (and free the KV of)
#                  the rest. The pre-change behaviour.
#   arm RAGGED   : unset                        — every running sequence decodes.
#
# ── WHAT THIS IS AND IS NOT SCOPED TO ────────────────────────────────────────
# This exercises the PagedAttention scheduler. **DeepSeek V4 does not use it** —
# `DeepSeekV4Loader::supports_paged_attention` returns false
# (normal_loaders.rs:3265), so V4 runs on `DefaultScheduler` and is unaffected by
# the change under test. Point MODEL at a paged-attention model. A small one is
# both cheaper and a *sharper* instrument: less compute per forward means batch
# occupancy dominates the throughput, so the effect is larger, not smaller.
#
# ── THE PREDICTION, STATED BEFORE THE RUN SO IT CAN BE REFUTED ───────────────
#   BUCKETED: `running` collapses toward 1 once lengths diverge, and aggregate
#             tok/s is roughly FLAT in B.
#   RAGGED:   `running` tracks B, and aggregate tok/s RISES with B.
#   If BUCKETED's `running` does not collapse, the defect did not fire for this
#   workload and the result should be recorded as a refutation, not retried.
#
# ── USAGE (main runs this; agents never call runcrate — D15) ─────────────────
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct \
#     setsid nohup bash arc-tools/arcsched_decode_admission_ab.sh \
#     < /dev/null > /root/logs/arcsched_ab.log 2>&1 &
#
#   watch -n30 cat /root/logs/arcsched_decode_admission_ab.status
#
# Results: $OUT/results.tsv (one row per arm × B) and $OUT/summary.txt.
set -u

MODEL="${MODEL:?set MODEL to a paged-attention model id or path (NOT DeepSeek V4)}"
PORT="${PORT:-1234}"
OUT="${OUT:-/root/logs/arcsched_ab}"
STATUS="${STATUS:-/root/logs/arcsched_decode_admission_ab.status}"
PREFLIGHT="${PREFLIGHT:-/root/arc-tools/gpu_box_preflight.sh}"
MAX_SEQS="${MAX_SEQS:-256}"
BATCHES="${BATCHES:-1 8 32 64 128}"
STEADY="${STEADY:-40}"
WARMUP="${WARMUP:-15}"
MISTRALRS="${MISTRALRS:-mistralrs}"
# Hard wall-clock cap so a hung arm cannot burn the budget unattended.
BUDGET_SECS="${BUDGET_SECS:-2700}"

mkdir -p "$OUT" "$(dirname "$STATUS")"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }
: > "$STATUS"
START_EPOCH=$(date +%s)

budget_left() {
  local now; now=$(date +%s)
  if [ $(( now - START_EPOCH )) -ge "$BUDGET_SECS" ]; then
    say "BUDGET: ${BUDGET_SECS}s wall clock exhausted — stopping with partial results."
    return 1
  fi
  return 0
}

# ── Gate. A box that fails preflight produces numbers nobody can attribute.
if [ -x "$PREFLIGHT" ]; then
  say "preflight: running $PREFLIGHT"
  if ! "$PREFLIGHT" >> "$OUT/preflight.log" 2>&1; then
    say "preflight: FAILED — see $OUT/preflight.log. Refusing to measure."; exit 2
  fi
  say "preflight: OK"
else
  say "preflight: $PREFLIGHT missing or not executable. Refusing to measure."; exit 2
fi

# ── Load driver. K concurrent workers, deterministically MIXED prompt lengths so
#    KV lengths diverge — which is the condition the defect needs and the
#    condition real serving always supplies.
cat > "$OUT/drive.py" <<'PYEOF'
import argparse, json, threading, time, urllib.request, urllib.error

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
          "kilo lima mike november oscar papa quebec romeo sierra tango ")

def prompt_of(words):
    return "Summarise the following log excerpt. " + (FILLER * (words // 20 + 1))[: words * 6]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--steady", type=float, required=True)
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}/v1/completions"
    stop = threading.Event()
    lock = threading.Lock()
    agg = {"tokens": 0, "requests": 0, "errors": 0}
    counting = threading.Event()

    # Worker i gets a distinct prompt length AND a distinct max_tokens, so the
    # cohort's KV lengths spread immediately and keep spreading.
    def worker(i):
        words = 32 + (i * 137) % 2000
        max_tok = 64 + (i * 37) % 192
        body = prompt_of(words)
        while not stop.is_set():
            payload = json.dumps({
                "model": "default", "prompt": body,
                "max_tokens": max_tok, "temperature": 0.7, "seed": 1234 + i,
            }).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    out = json.loads(r.read())
                n = out.get("usage", {}).get("completion_tokens", 0)
                if counting.is_set():
                    with lock:
                        agg["tokens"] += n; agg["requests"] += 1
            except Exception:
                if counting.is_set():
                    with lock:
                        agg["errors"] += 1

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.k)]
    for t in threads: t.start()
    time.sleep(a.warmup)
    with lock:
        agg["tokens"] = 0; agg["requests"] = 0; agg["errors"] = 0
    counting.set()
    t0 = time.time()
    time.sleep(a.steady)
    elapsed = time.time() - t0
    counting.clear()
    stop.set()
    with lock:
        print(json.dumps({"tok_s": agg["tokens"] / elapsed, "elapsed": elapsed, **agg}))

main()
PYEOF

printf 'arm\tB\ttok_s\trequests\terrors\tmean_running\tmax_running\tmean_waiting\tsamples\n' > "$OUT/results.tsv"

run_cell() {                       # $1=arm  $2=B  $3=bucketed(0|1)
  local arm="$1" b="$2" bucketed="$3"
  local slog="$OUT/server_${arm}_B${b}.log"
  say "arm $arm B=$b — starting server"

  if [ "$bucketed" = "1" ]; then export ARC_SCHED_BUCKETED_DECODE=1
  else unset ARC_SCHED_BUCKETED_DECODE || true; fi

  "$MISTRALRS" serve -p "$PORT" -m "$MODEL" --max-seqs "$MAX_SEQS" \
    > "$slog" 2>&1 &
  local pid=$!

  # Wait for readiness rather than sleeping a guess.
  local ready=0
  for _ in $(seq 1 240); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then ready=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    say "arm $arm B=$b — server never became ready; see $slog. Recording NA."
    printf '%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\t0\n' "$arm" "$b" >> "$OUT/results.tsv"
    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    return
  fi

  local res
  res=$(python3 "$OUT/drive.py" --port "$PORT" --k "$b" \
          --warmup "$WARMUP" --steady "$STEADY" 2>>"$OUT/drive.err") || res=""
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true

  local tok_s reqs errs
  tok_s=$(printf '%s' "$res" | python3 -c 'import sys,json;print(f"{json.load(sys.stdin)[\"tok_s\"]:.2f}")' 2>/dev/null || echo NA)
  reqs=$(printf '%s' "$res"  | python3 -c 'import sys,json;print(json.load(sys.stdin)["requests"])' 2>/dev/null || echo NA)
  errs=$(printf '%s' "$res"  | python3 -c 'import sys,json;print(json.load(sys.stdin)["errors"])' 2>/dev/null || echo NA)

  # ── THE DIRECT FINGERPRINT. `N running, M waiting` is emitted every 5s by
  #    IntervalLogger and needs no throughput inference: N is literally how many
  #    sequences the scheduler put in the forward. Reported NA-with-a-reason if
  #    absent — a 0 here would read as "no sequences ran" when it means "nobody
  #    logged" (D18).
  local stats
  stats=$(grep -oE '[0-9]+ running, [0-9]+ waiting' "$slog" \
          | awk '{r+=$1; if($1>mx)mx=$1; w+=$3; n++}
                 END{ if(n==0) print "NA\tNA\tNA\t0";
                      else printf "%.2f\t%d\t%.2f\t%d", r/n, mx, w/n, n }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$b" "$tok_s" "$reqs" "$errs" "$stats" >> "$OUT/results.tsv"
  say "arm $arm B=$b — tok/s=$tok_s  running/waiting=$stats"
}

for b in $BATCHES; do
  budget_left || break
  run_cell BUCKETED "$b" 1
  budget_left || break
  run_cell RAGGED   "$b" 0
done

{
  echo "ArcSched decode admission A/B — $(date -u)"
  echo "model=$MODEL max_seqs=$MAX_SEQS steady=${STEADY}s warmup=${WARMUP}s"
  echo
  cat "$OUT/results.tsv"
  echo
  echo "THE NUMBERS THAT SETTLE IT:"
  echo "  1. mean_running per arm. BUCKETED should collapse toward 1 as B grows;"
  echo "     RAGGED should track B. This needs no throughput reasoning."
  echo "  2. tok_s(RAGGED)/tok_s(BUCKETED) per B. Should rise with B."
  echo "  3. B=1 must AGREE across arms — that is the no-regression control."
  echo "     A disagreement at B=1 invalidates the whole run: one sequence is one"
  echo "     bucket, so the two rules are the same rule there."
  echo
  python3 - "$OUT/results.tsv" <<'PYEOF'
import sys, collections
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
d = collections.defaultdict(dict)
for r in rows:
    if len(r) >= 6: d[r[1]][r[0]] = r
print("  B    tok/s BUCKETED   tok/s RAGGED   ratio    mean_running B→R")
for b in sorted(d, key=lambda x: int(x) if x.isdigit() else 0):
    bu, ra = d[b].get("BUCKETED"), d[b].get("RAGGED")
    if not bu or not ra: continue
    try:
        rt = f"{float(ra[2])/float(bu[2]):.2f}x"
    except Exception:
        rt = "NA"
    print(f"  {b:<4} {bu[2]:>14} {ra[2]:>14} {rt:>8}    {bu[5]} → {ra[5]}")
PYEOF
} > "$OUT/summary.txt" 2>&1

say "DONE — $OUT/summary.txt"
cat "$OUT/summary.txt" | tee -a "$STATUS"
