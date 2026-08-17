#!/bin/bash
# ArcSpec — per-sequence KV advance: THROUGHPUT on the ladder B = 1, 8, 32, 128.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE QUESTION
#
# #116 proved per-sequence advance *works* at B=8 (it returned 1 token with no
# `finish_reason` before; it returns the full 48 after). It did not prove the
# mechanism is worth anything. MTP was measured at 1.93 tok/step at one user
# collapsing to 1.06 at 128, and this stack targets exactly that collapse.
#
# ⚠️ REPORT BOTH NUMBERS OR THE RESULT IS UNREADABLE. tok/step rising while
# aggregate throughput falls has already happened on this chain (+9%/+13%
# tok/step, −5×/−20% aggregate). A tok/step number alone is not a throughput
# result, and an aggregate number alone cannot say what moved. So every cell
# reports:
#
#     aggregate tok/s        (wall clock, the number that matters)
#     tok_per_step           (per-sequence multiplier)
#     tok_per_batch_step     (tokens out per target forward, whole batch)
#     batch_steps/s          (the decomposition: aggregate ≈ tok_per_batch_step
#                             × batch_steps/s — when aggregate does not move,
#                             this says which factor ate it, in ms)
#
# ─────────────────────────────────────────────────────────────────────────────
# ⚠️ TWO PROMPT REGIMES, MEAN-MATCHED. Both are needed, for opposite reasons.
#
#   spread    144 24 320 64 260 40 200 96 words, cycled across workers
#   uniform   144 words for every worker  (= the spread's mean, and its first
#             element, so B=1 is byte-identical in both regimes)
#
# **spread** is the regime this stack exists for. Uniform prompts hide the
# failure mode: the `xs` window defect #116 fixed could not even be reached with
# equal-length prompts, because the retained window is sized for the greediest
# row and a short row only becomes shorter-than-the-window when arrivals differ.
# A benchmark that sends one prompt to every worker measures the case that
# already worked.
#
# **uniform** is the only regime where the batch is WHOLE, and that is why it is
# not optional. The scheduler A/B has now measured the bucketing law directly:
#
#     running bucket size  =  B / (distinct cache lengths)
#
# holding at 8/8=1 and 32/8=4, with `1 running, 7 waiting` sustained. Under a
# spread of 8 distinct lengths, B=8 therefore runs ONE sequence at a time — that
# chain measured B=8 spread at 7.91 tok/s against B=1's 15.36, i.e. batching is
# *negative* on realistic traffic. So a flat aggregate in the spread arm is
# unreadable on its own: it could be this fix not paying, or it could be that
# the batch never existed. Uniform is where a whole batch actually forms, so it
# is the only place this fix's effect on aggregate is visible without the
# serialisation swamping it.
#
# Mean-matched on purpose: same total prompt tokens in both regimes, so the only
# thing that differs is the *spread*, and a difference is attributable to
# raggedness rather than to prompt length.
#
# Session 8 measured V4 serving dying outright at ~1,055-word prompts and this
# branch does NOT carry #97 (`pin V4 against length`), so 320 keeps a >3x margin
# to that cliff while still being a real spread.
#
# ─────────────────────────────────────────────────────────────────────────────
# ⚠️ THE SCHEDULER CONFOUND, MEASURED RATHER THAN ASSUMED.
#
# Both schedulers bucket the running set by EXACT cache length and run ONE
# bucket per step, preempting the rest (`default_scheduler.rs`,
# `paged_attention/scheduler.rs`). So "B=128" in this harness can be a 3-wide
# step in the engine, and an aggregate number measured inside that is a number
# about the scheduler, not about KV advance.
#
# `SCHED[agg]` (added with this harness) is emitted on the SAME log fence as
# `MTP[agg]`, so each cell reports, over exactly its own window:
#
#     buckets_per_step       1.0 = the batch was never split
#     running_bucket_size    the width the engine ACTUALLY ran
#     offered_per_step       the width this harness thinks it asked for
#
# If `running_bucket_size` << `offered_per_step`, the throughput cell is a
# scheduler measurement and says so.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE DESIGN
#
# Two server processes, one per arm; four concurrency cells inside each.
#   OFF  no flags                                  (control)
#   ON   ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1  (treatment)
# Same binary, same model, same prompts, same MTP depth, same --max-seqs. One
# server per arm rather than one per cell because the engine counters are
# cumulative and monotone: differencing them across a wall-clock fence gives a
# cell's own numbers without paying the ~2m30s V4 load eight times.
#
# ⚠️ ENGAGEMENT IS ASSERTED, NOT ASSUMED (D18). "per-sequence KV advance is ON"
# and "Ragged batch admission is ON" must be PRESENT in the ON log and ABSENT
# in OFF, "cannot honour it" must be absent from ON, and `per_seq_steps` must be
# non-zero — granted-but-inert has to show as a visible zero, not as silence.
#
# Environment failure exits 2. A gate that ran and failed exits 1.
set -u

ARCENV="${ARCENV:-/root/arcenv.sh}"
WT="${WT:-/root/wt-ladder}"
BIN="${BIN:-$WT/target/release/mistralrs}"
MODEL="${MODEL:-/root/models/v4-src}"
UQFF="${UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}"
ARCH="${ARCH:-deepseekv4}"
PREFLIGHT="${PREFLIGHT:-/root/arc-tools/gpu_box_preflight.sh}"
OUT="${OUT:-/root/logs/ladder}"
STATUS="${STATUS:-/root/logs/ladder.status}"
PORT="${PORT:-1243}"
MTP_DEPTH="${MTP_DEPTH:-3}"
MAX_SEQS="${MAX_SEQS:-256}"
SEED="${SEED:-1234}"
BATCHES="${BATCHES:-1 8 32 128}"
WARMUP="${WARMUP:-20}"
STEADY="${STEADY:-45}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
# 144 first so B=1 is byte-identical between the two regimes; mean is 144.
WORD_SPREAD="${WORD_SPREAD:-144 24 320 64 260 40 200 96}"
UNIFORM_WORDS="${UNIFORM_WORDS:-144}"
REGIMES="${REGIMES:-uniform spread}"

mkdir -p "$OUT" "$(dirname "$STATUS")"
: > "$STATUS"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$STATUS"; }
die() { say "ABORT: $*"; echo "ABORT: $*" >> "$OUT/summary.txt"; exit 2; }

say "ArcSpec per-seq throughput ladder — start"
[ -f "$ARCENV" ] || die "$ARCENV not found"
# shellcheck disable=SC1090
. "$ARCENV"
say "env: CUDA_HOME=${CUDA_HOME:-unset} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
[ -x "$PREFLIGHT" ] || die "$PREFLIGHT missing — refusing to measure"
"$PREFLIGHT" > "$OUT/preflight.log" 2>&1 || die "preflight refused the box"
say "preflight OK"
[ -x "$BIN" ] || die "$BIN not found"
SHA=$(git -C "$WT" rev-parse HEAD 2>/dev/null) || die "no git ref at $WT"
say "binary ref: $SHA"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > "$OUT/gpu.txt" 2>&1

# ── The load driver. Streaming, so a cell yields tokens even when no request
#    finishes inside it (at B=128 V4 needs minutes per request).
cat > "$OUT/drive.py" <<'PYEOF'
import argparse, json, threading, time, urllib.request

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
          "kilo lima mike november oscar papa quebec romeo sierra tango").split()

def prompt_of(words):
    return ("Continue this list, one item per line. "
            + " ".join(FILLER[i % len(FILLER)] for i in range(words)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--steady", type=float, required=True)
    ap.add_argument("--spread", required=True, help="comma-separated word counts")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    # 🔑 RAGGED BY CONSTRUCTION: worker i takes spread[i % len(spread)], so the
    # arrivals differ in length the way real ones do. Deterministic (no RNG), so
    # the two arms see byte-identical prompt sets.
    spread = [int(x) for x in a.spread.split(",") if x]
    prompts = [prompt_of(spread[i % len(spread)]) for i in range(a.k)]
    url = f"http://127.0.0.1:{a.port}/v1/completions"
    stop = threading.Event()
    lock = threading.Lock()
    n = {"chunks": 0, "errors": 0, "started": 0, "finished": 0}
    counting = [False]

    def worker(wid):
        body_prompt = prompts[wid]
        while not stop.is_set():
            body = json.dumps({
                "model": "default", "prompt": body_prompt,
                "max_tokens": a.max_tokens, "temperature": 0,
                "seed": a.seed, "stream": True,
            }).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            try:
                with lock:
                    n["started"] += 1
                with urllib.request.urlopen(req, timeout=1800) as r:
                    for raw in r:
                        if stop.is_set():
                            break
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            with lock:
                                n["finished"] += 1
                            break
                        try:
                            ch = json.loads(payload)
                        except Exception:
                            continue
                        txt = (ch.get("choices") or [{}])[0].get("text", "")
                        if txt:
                            with lock:
                                if counting[0]:
                                    n["chunks"] += 1
            except Exception:
                with lock:
                    n["errors"] += 1
                time.sleep(0.5)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(a.k)]
    for t in threads:
        t.start()
    time.sleep(a.warmup)
    with lock:
        n["chunks"] = 0
        counting[0] = True
    t0 = time.time()
    fence = {"t_start": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}
    time.sleep(a.steady)
    with lock:
        counting[0] = False
        snap = dict(n)
    fence["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    stop.set()
    elapsed = time.time() - t0
    snap.update(fence)
    snap["k"] = a.k
    snap["prompt_words"] = [spread[i % len(spread)] for i in range(a.k)]
    snap["elapsed_s"] = round(elapsed, 3)
    snap["driver_tok_s"] = round(snap["chunks"] / elapsed, 3) if elapsed else 0.0
    print(json.dumps(snap))
    for t in threads:
        t.join(timeout=3)

main()
PYEOF

# ── The extractor. Differences cumulative MTP[agg] AND SCHED[agg] across the
#    cell's wall-clock fence. Everything reported is a delta of raw counts.
cat > "$OUT/fence.py" <<'PYEOF'
"""Delta of the engine's cumulative counters across one cell's fence.

`MTP[agg] ... steps=S committed=C batch_steps=BS` and `SCHED[agg] ... calls=N
buckets=B offered=O chosen=C` are cumulative for the process and emitted on the
same log fence, so a cell's own numbers are the difference between the last
line at or before its end and the last line at or before its start.

Reports nothing it did not measure: a cell with <2 usable lines prints NA with
a reason rather than 0.0, because a zero would read as "the engine produced
nothing" when it means "the reporter never fired twice" (D18).
"""
import json, re, sys

TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
FIELDS = {
    "MTP[agg]": ("accepted", "proposed", "steps", "drafted_steps", "committed",
                 "batch_steps"),
    "SCHED[agg]": ("calls", "buckets", "offered", "chosen", "shattered"),
}

def parse(path, marker):
    keys = FIELDS[marker]
    out = []
    for line in open(path, errors="ignore"):
        if marker not in line:
            continue
        mt = TS.search(line)
        if not mt:
            continue
        rec, ok = {"ts": mt.group(1)}, True
        for k in keys:
            m = re.search(rf"\b{k}=(\d+)\b", line)
            if not m:
                ok = False
                break
            rec[k] = int(m.group(1))
        if ok:
            out.append(rec)
    return out

def bracket(recs, t0, t1):
    """(last record at or before t0, last record at or before t1)."""
    before = [r for r in recs if r["ts"] <= t0]
    within = [r for r in recs if r["ts"] <= t1]
    if not before or not within:
        return None, None
    return before[-1], within[-1]

def delta(path, marker, t0, t1):
    recs = parse(path, marker)
    if len(recs) < 2:
        return {"NA": f"{marker}: only {len(recs)} line(s) in the whole log"}
    a, b = bracket(recs, t0, t1)
    if a is None or b is None:
        return {"NA": f"{marker}: no line brackets [{t0}, {t1}]"}
    d = {k: b[k] - a[k] for k in FIELDS[marker]}
    if all(v == 0 for v in d.values()):
        return {"NA": f"{marker}: counters did not advance inside the fence"}
    return d

def main():
    log, t0, t1, elapsed, driver_tok_s = (
        sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
    out = {"t_start": t0, "t_end": t1, "elapsed_s": elapsed,
           "aggregate_tok_s": driver_tok_s}
    m = delta(log, "MTP[agg]", t0, t1)
    s = delta(log, "SCHED[agg]", t0, t1)
    out["mtp_raw"], out["sched_raw"] = m, s
    r = lambda x: round(x, 4)
    if "NA" not in m:
        out["tok_per_step"] = r(m["committed"] / m["steps"]) if m["steps"] else None
        out["tok_per_batch_step"] = (
            r(m["committed"] / m["batch_steps"]) if m["batch_steps"] else None)
        out["mean_batch"] = r(m["steps"] / m["batch_steps"]) if m["batch_steps"] else None
        out["batch_steps_per_s"] = r(m["batch_steps"] / elapsed) if elapsed else None
        out["engine_tok_s"] = r(m["committed"] / elapsed) if elapsed else None
        if m["batch_steps"] and elapsed:
            out["ms_per_batch_step"] = r(1000.0 * elapsed / m["batch_steps"])
    if "NA" not in s:
        out["buckets_per_step"] = r(s["buckets"] / s["calls"]) if s["calls"] else None
        out["running_bucket_size"] = r(s["chosen"] / s["calls"]) if s["calls"] else None
        out["offered_per_step"] = r(s["offered"] / s["calls"]) if s["calls"] else None
        out["shattered_frac"] = r(s["shattered"] / s["calls"]) if s["calls"] else None
    print(json.dumps(out))

main()
PYEOF

run_arm() {
  local arm="$1"; shift
  local slog="$OUT/server.$arm.log"
  say "arm $arm — starting server (${*:-no flags})"
  # shellcheck disable=SC2086
  env RUST_LOG=info ARC_MTP_LOG_ACCEPTANCE=1 "$@" \
    "$BIN" serve -p "$PORT" -m "$MODEL" -a "$ARCH" --from-uqff "$UQFF" \
    --max-seqs "$MAX_SEQS" --mtp-depth "$MTP_DEPTH" --prefix-cache-n 0 \
    --paged-attn off --seed "$SEED" \
    < /dev/null > "$slog" 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 1 450); do
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$spid" 2>/dev/null || break
    sleep 2
  done
  [ "$up" = "1" ] || { say "arm $arm — server never came up"; tail -25 "$slog" | tee -a "$STATUS"; return 1; }

  local logged
  logged=$(grep -m1 -o 'git revision: [0-9a-f]*' "$slog" | awk '{print $3}')
  if [ -n "$SHA" ] && [ "$logged" != "$SHA" ]; then
    say "arm $arm — PROVENANCE FAILED: server $logged != worktree $SHA"
    kill -TERM "$spid" 2>/dev/null || true
    return 1
  fi
  say "arm $arm — server up, revision $logged"

  local regime k spread_csv tag
  for regime in $REGIMES; do
    if [ "$regime" = "uniform" ]; then
      spread_csv="$UNIFORM_WORDS"
    else
      spread_csv=$(echo "$WORD_SPREAD" | tr ' ' ',')
    fi
    for k in $BATCHES; do
      tag="$arm.$regime.k$k"
      say "arm $arm — cell $regime B=$k (warmup ${WARMUP}s, steady ${STEADY}s, words $spread_csv)"
      python3 "$OUT/drive.py" --port "$PORT" --k "$k" --warmup "$WARMUP" \
        --steady "$STEADY" --spread "$spread_csv" --max-tokens "$MAX_TOKENS" \
        --seed "$SEED" > "$OUT/$tag.drive.json" 2>"$OUT/$tag.drive.err" \
        || { say "cell $tag — driver failed"; return 1; }
      local t0 t1 el tok
      t0=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['t_start'])" "$OUT/$tag.drive.json")
      t1=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['t_end'])" "$OUT/$tag.drive.json")
      el=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['elapsed_s'])" "$OUT/$tag.drive.json")
      tok=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['driver_tok_s'])" "$OUT/$tag.drive.json")
      python3 "$OUT/fence.py" "$slog" "$t0" "$t1" "$el" "$tok" > "$OUT/$tag.cell.json" \
        || { say "cell $tag — fence extraction failed"; return 1; }
      say "$tag -> $(cat "$OUT/$tag.cell.json")"
      sleep 5
    done
  done

  kill -TERM "$spid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$spid" 2>/dev/null || break; sleep 2; done
  kill -KILL "$spid" 2>/dev/null || true
  sleep 5
  say "arm $arm — done"
  return 0
}

run_arm OFF || die "OFF arm failed"
run_arm ON ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1 || die "ON arm failed"
# The third arm prices the window pin, which is ON by default in every arm
# above. Same flags as ON, pin disabled — so ON vs ON_UNPINNED isolates one
# change: whether the compressor's raw window is reallocated every decode step
# on all 41 compressed layers, or held at its documented bound.
run_arm ON_UNPINNED ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1 ARC_V4_XS_PIN_WINDOW=0 \
  || die "ON_UNPINNED arm failed"

cat > "$OUT/report.py" <<'PYEOF'
import json, os, sys
out, batches, regimes = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
ARMS = ("OFF", "ON", "ON_UNPINNED")
def cell(arm, regime, k):
    p = os.path.join(out, f"{arm}.{regime}.k{k}.cell.json")
    try:
        return json.load(open(p))
    except Exception as e:
        return {"NA": str(e)}
def g(c, k):
    v = c.get(k)
    return "NA" if v is None else v
print("\n=== LADDER: aggregate throughput and its decomposition ===")
print("aggregate_tok/s is wall-clock and is THE number. tok/batch_step x")
print("batch_steps/s reconstructs it: when aggregate does not move, those two")
print("say which factor ate it. run_bucket is the width the engine ACTUALLY ran")
print("— compare it to offered before reading any cell as a batch result.")
hdr = (f"{'regime':<8} {'B':>4} {'arm':<12} {'aggregate_tok/s':>15} {'tok/step':>9} "
       f"{'tok/bstep':>10} {'bsteps/s':>9} {'ms/bstep':>9} "
       f"{'buckets/step':>13} {'run_bucket':>11} {'offered':>8}")
print(hdr); print("-" * len(hdr))
for regime in regimes:
    for k in batches:
        for arm in ARMS:
            c = cell(arm, regime, k)
            print(f"{regime:<8} {k:>4} {arm:<12} {str(g(c,'aggregate_tok_s')):>15} "
                  f"{str(g(c,'tok_per_step')):>9} {str(g(c,'tok_per_batch_step')):>10} "
                  f"{str(g(c,'batch_steps_per_s')):>9} {str(g(c,'ms_per_batch_step')):>9} "
                  f"{str(g(c,'buckets_per_step')):>13} {str(g(c,'running_bucket_size')):>11} "
                  f"{str(g(c,'offered_per_step')):>8}")
print("\n=== ON vs OFF, per regime ===")
for regime in regimes:
    print(f"\n-- {regime} --")
    for k in batches:
        o, n = cell("OFF", regime, k), cell("ON", regime, k)
        def ratio(key):
            a, b = o.get(key), n.get(key)
            if not a or not b:
                return "NA"
            return f"{b/a:.3f}x ({a} -> {b})"
        print(f"  B={k:<4} aggregate  {ratio('aggregate_tok_s')}")
        print(f"        tok/step   {ratio('tok_per_step')}")
        print(f"        tok/bstep  {ratio('tok_per_batch_step')}")
        print(f"        bsteps/s   {ratio('batch_steps_per_s')}")
        print(f"        run_bucket {ratio('running_bucket_size')}")
print("\n=== THE WINDOW PIN, isolated (ON vs ON_UNPINNED) ===")
print("Same per-seq flags in both; the only difference is whether the")
print("compressor's raw window is reallocated every decode step on 41 layers.")
for regime in regimes:
    for k in batches:
        u, p_ = cell("ON_UNPINNED", regime, k), cell("ON", regime, k)
        def r2(key):
            a, b = u.get(key), p_.get(key)
            if not a or not b:
                return "NA"
            return f"{b/a:.3f}x ({a} -> {b})"
        print(f"  {regime:<8} B={k:<4} aggregate {r2('aggregate_tok_s')}  "
              f"ms/bstep {r2('ms_per_batch_step')}")

print("\n=== THE BUCKETING LAW, checked against this run ===")
print("The scheduler A/B measured `running bucket = B / distinct lengths`.")
print("uniform has 1 distinct length, spread has 8, so the prediction is:")
for regime in regimes:
    dist = 1 if regime == "uniform" else 8
    for k in batches:
        pred = max(1.0, int(k) / dist)
        for arm in ARMS:
            c = cell(arm, regime, k)
            got = c.get("running_bucket_size")
            if got is None:
                continue
            print(f"  {regime:<8} B={k:<4} {arm:<12} predicted {pred:>6.2f}  measured {got:>7.3f}")
print("\n⚠️ Read every spread cell against its run_bucket. If run_bucket is ~1,")
print("that cell is a measurement of the scheduler serialising the batch, not")
print("of per-sequence KV advance, and its aggregate cannot be attributed here.")
PYEOF

{
  echo "=== ref ==="; echo "$SHA"
  echo "=== prompts ==="
  echo "  spread : $WORD_SPREAD (8 distinct lengths)"
  echo "  uniform: $UNIFORM_WORDS (the spread's mean AND its first element, so B=1 matches)"
  echo
  echo "=== ENGAGEMENT ==="
  for m in "per-sequence KV advance is ON" "Ragged batch admission is ON"; do
    on=$(grep -c "$m" "$OUT/server.ON.log" 2>/dev/null || echo 0)
    off=$(grep -c "$m" "$OUT/server.OFF.log" 2>/dev/null || echo 0)
    echo "  '$m': ON=$on (want >=1)  OFF=$off (want 0)"
  done
  echo "  'cannot honour it' in ON: $(grep -c 'cannot honour it' "$OUT/server.ON.log" 2>/dev/null || echo 0) (want 0)"
  echo "  per_seq_steps final, ON:  $(grep -o 'per_seq_steps=[0-9]*' "$OUT/server.ON.log" 2>/dev/null | tail -1) (want >0)"
  echo "  per_seq_steps final, OFF: $(grep -o 'per_seq_steps=[0-9]*' "$OUT/server.OFF.log" 2>/dev/null | tail -1) (want =0)"
  echo "  'xs rolling cache' errors ON=$(grep -c 'xs rolling cache' "$OUT/server.ON.log" 2>/dev/null || echo 0) OFF=$(grep -c 'xs rolling cache' "$OUT/server.OFF.log" 2>/dev/null || echo 0) (want 0/0)"
  python3 "$OUT/report.py" "$OUT" "$BATCHES" "$REGIMES"
} | tee "$OUT/summary.txt"

say "DONE — $OUT/summary.txt"
