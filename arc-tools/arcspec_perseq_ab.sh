#!/bin/bash
# ArcSpec — per-sequence KV advance, A/B on hardware.  D15: the agent never
# calls `runcrate`; main executes this and returns raw output.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE QUESTION
#
# PR stack #92→#95→#100→#102→#103→#104 predicts that removing the cohort
# min-rollback lifts `tok_per_step` at B=128 from the measured 1.0558 to
# ~1.48 (+31%).  That has never run on a GPU.  This is the run.
#
# THE METRIC IS `tok_per_step`, NEVER `accept_rate`.  A saturated sequence
# drafts 0, contributing `proposed=0`, so `accept_rate` stays ~0.43 while the
# yield dies.  Every number below is derived from the engine's own cumulative
# counters (`MTP[agg]`), differenced across a cell fence — not from the driver.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE DESIGN
#
# TWO server processes, one per arm; four concurrency cells inside each.
#
#   OFF  no flags                                    (control)
#   ON   ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1    (treatment)
#
# Same binary, same model, same prompts, same MTP depth, same `--max-seqs`.
# Only the two env vars differ.  One server per arm rather than one per cell
# because the counters are cumulative and monotone: differencing them across a
# wall-clock fence gives a cell's own numbers without paying the ~2m30s V4 load
# twenty times.
#
# ⚠️ ENGAGEMENT IS ASSERTED, NOT ASSUMED (D18).  Two INFO lines must be PRESENT
# in the ON log and ABSENT in the OFF log:
#     "MTP: per-sequence KV advance is ON"      (mtp_pipeline.rs kv_advance)
#     "Ragged batch admission is ON"            (engine/mod.rs)
# and "cannot honour it" must be ABSENT from the ON log.  If the treatment arm
# is not engaged, the A/B is a measurement of nothing and the script says so in
# the summary instead of reporting a flat ratio as a refutation.
#
# ⚠️ PROMPTS ARE SHORT ON PURPOSE.  This stack branches from master@#83 and does
# NOT contain #97 (`pin V4 against length`).  Session 8 measured V4 serving
# dying outright at ~1,055-word prompts.  40-word prompts keep both arms far
# from that cliff; the raggedness under test is produced by MTP's own divergent
# accept counts, not by prompt-length spread.
#
# ⚠️ THE BUILD DOES NOT TOUCH /root/arc's CHECKOUT.  The box is shared with
# three other chains.  This adds a git worktree at $WT and builds it, leaving
# /root/arc's HEAD, index and working tree untouched.  It shares
# /root/arc/target, so cargo serialises against a concurrent build rather than
# racing it (and reuses every git/registry dependency).
#
# ─────────────────────────────────────────────────────────────────────────────
# USAGE
#
#   setsid nohup bash /root/arcspec_perseq_ab.sh < /dev/null \
#     > /root/logs/arcspec_perseq_ab.log 2>&1 &
#
#   tail -f /root/logs/arcspec_perseq_ab.status      # progress, one line per event
#   cat /root/logs/arcspec_ab/summary.txt            # the answer, when DONE
#
# THE NUMBER TO EXTRACT, in one line:
#   grep '^ANSWER' /root/logs/arcspec_ab/summary.txt
#
set -u

# ── Box wiring (all overridable) ─────────────────────────────────────────────
ARCENV="${ARCENV:-/root/arcenv.sh}"
REPO="${REPO:-/root/arc}"
WT="${WT:-/root/arc-arcspec}"
BRANCH="${BRANCH:-agent/arcspec-perseq-measure}"
MODEL="${MODEL:-/root/models/v4-src}"
UQFF="${UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}"
ARCH="${ARCH:-deepseekv4}"
PREFLIGHT="${PREFLIGHT:-/root/arc-tools/gpu_box_preflight.sh}"
OUT="${OUT:-/root/logs/arcspec_ab}"
STATUS="${STATUS:-/root/logs/arcspec_perseq_ab.status}"
PORT="${PORT:-1239}"
FEATURES="${FEATURES:-cuda flash-attn}"   # NO cudnn: −62% decode (session 4)
MTP_DEPTH="${MTP_DEPTH:-3}"
MAX_SEQS="${MAX_SEQS:-256}"
BATCHES="${BATCHES:-1 8 32 128}"
WARMUP="${WARMUP:-25}"                    # ramp, discarded
STEADY="${STEADY:-60}"                    # the fenced window
PROMPT_WORDS="${PROMPT_WORDS:-40}"        # FIXED and UNIFORM — see the canary
CANARY_WORDS="${CANARY_WORDS:-40 128 512 1100}"   # brackets session 8's ~1,055-word death
MAX_TOKENS="${MAX_TOKENS:-4096}"          # long enough that no cell drains
SKIP_BUILD="${SKIP_BUILD:-0}"
LOCK="${LOCK:-/root/locks/gpu.lock}"        # whole-GPU mutex; three chains share this box
LOCK_WAIT="${LOCK_WAIT:-0}"                 # seconds to wait for the lock; 0 = fail fast

mkdir -p "$OUT" "$(dirname "$STATUS")" "$(dirname "$LOCK")"
: > "$STATUS"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$STATUS"; }
die() { say "ABORT: $*"; echo "ABORT: $*" >> "$OUT/summary.txt"; exit 2; }

say "ArcSpec per-sequence KV advance A/B — start"

# ── 0. GPU lock.  V4 wants all 143 GB; nothing else may serve while we hold it.
#       `set -o noclobber` + redirect is the atomic create-if-absent primitive
#       that works on every filesystem here (no flock(1) dependency).  The lock
#       records who holds it and since when, so a stale one is diagnosable
#       rather than mysterious.
OWNER="arcspec"
release_lock() {
  if [ -f "$LOCK" ] && grep -q "^$OWNER pid=$$ " "$LOCK" 2>/dev/null; then
    rm -f "$LOCK"
  fi
}
# Is the current holder demonstrably gone? Owner-matched release is safe but
# CANNOT self-heal: a lock whose owner died — or one created empty, with no
# owner tag at all — is unremovable by every well-behaved participant and
# deadlocks the box forever. That happened tonight: a size-0 lock appeared,
# its presumed owner was dead, the H200 sat at 0% for six minutes, and the run
# waiting on it would have aborted after 40 minutes without ever measuring.
#
# So reclaim, but only on EVIDENCE, and only after a grace period so a lock
# caught mid-write is never stolen:
#   * older than STALE_AFTER, AND
#   * empty (no owner claim at all) or its named pid is dead, AND
#   * the GPU is genuinely idle — no memory in use and no compute processes.
# All three, because any one alone has a false positive.
STALE_AFTER="${STALE_AFTER:-120}"
lock_is_stale() {
  [ -f "$LOCK" ] || return 1
  local age holder pid used procs
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
  [ "$age" -ge "$STALE_AFTER" ] || return 1
  holder=$(cat "$LOCK" 2>/dev/null)
  pid=$(printf '%s' "$holder" | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
  if [ -n "$holder" ] && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    return 1                       # a live owner: not stale, full stop
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "${used:-1}" -lt 1024 ] && [ "${procs:-1}" -eq 0 ]
}
acquire_lock() {
  local waited=0
  while :; do
    if ( set -o noclobber; echo "$OWNER pid=$$ started=$(date -u +%FT%TZ)" > "$LOCK" ) 2>/dev/null; then
      trap 'release_lock' EXIT INT TERM
      say "lock: acquired $LOCK"
      return 0
    fi
    if lock_is_stale; then
      say "lock: RECLAIMING stale lock (age>=${STALE_AFTER}s, owner absent or dead, GPU idle): '$(cat "$LOCK" 2>/dev/null)'"
      rm -f "$LOCK"
      continue
    fi
    if [ "$waited" -ge "$LOCK_WAIT" ]; then
      say "lock: HELD BY -> $(cat "$LOCK" 2>/dev/null)"
      return 1
    fi
    [ "$waited" = 0 ] && say "lock: waiting for -> $(cat "$LOCK" 2>/dev/null)"
    sleep 15; waited=$((waited + 15))
  done
}
acquire_lock || die "GPU lock $LOCK is held by another chain — refusing to overlap (set LOCK_WAIT=<secs> to queue, or clear a stale lock)"

# ── 1. Environment.  LD_LIBRARY_PATH=/usr/local/cuda/compat is mandatory. ─────
[ -f "$ARCENV" ] || die "$ARCENV not found"
# shellcheck disable=SC1090
. "$ARCENV"
say "env: CUDA_HOME=${CUDA_HOME:-unset} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

# ── 2. Preflight gate.  A box that fails it produces unattributable numbers. ──
if [ -x "$PREFLIGHT" ]; then
  say "preflight: running $PREFLIGHT"
  if ! "$PREFLIGHT" > "$OUT/preflight.log" 2>&1; then
    say "preflight FAILED — see $OUT/preflight.log"
    die "preflight refused the box"
  fi
  say "preflight OK"
else
  die "$PREFLIGHT missing or not executable — refusing to measure"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv > "$OUT/gpu.txt" 2>&1
say "gpu: $(sed -n 2p "$OUT/gpu.txt")"

# ── 3. Build the stack tip in an ISOLATED worktree. ───────────────────────────
BIN="$WT/target/release/mistralrs"
if [ "$SKIP_BUILD" != "1" ]; then
  git -C "$REPO" fetch origin "$BRANCH" >> "$OUT/build.log" 2>&1 \
    || die "git fetch $BRANCH failed"
  SHA=$(git -C "$REPO" rev-parse FETCH_HEAD)
  if [ -d "$WT" ]; then
    git -C "$WT" checkout -f "$SHA" >> "$OUT/build.log" 2>&1 || die "worktree checkout failed"
  else
    git -C "$REPO" worktree add --detach "$WT" "$SHA" >> "$OUT/build.log" 2>&1 \
      || die "git worktree add failed"
  fi
  say "built ref: $SHA ($BRANCH)"
  echo "$SHA" > "$OUT/sha.txt"
  say "build: cargo build --release --features \"$FEATURES\" -p mistralrs-cli (this is the long part)"
  ( cd "$WT" && CARGO_TARGET_DIR="$WT/target" \
      cargo build --release --features "$FEATURES" -p mistralrs-cli ) \
      >> "$OUT/build.log" 2>&1 || die "build FAILED — tail: $(tail -25 "$OUT/build.log")"
  say "build OK"
fi
[ -x "$BIN" ] || die "$BIN not found — build did not produce the binary"

# ── 4. The load driver.  Streaming, so a cell yields tokens even when no
#       request finishes inside it (at B=128 V4 needs minutes per request).
#       Its token count is only a CROSS-CHECK: the reported numbers come from
#       the engine's own counters. ─────────────────────────────────────────────
cat > "$OUT/drive.py" <<'PYEOF'
import argparse, json, threading, time, urllib.request

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
          "kilo lima mike november oscar papa quebec romeo sierra tango ").split()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--steady", type=float, required=True)
    ap.add_argument("--words", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()

    # Identical prompt for every worker: the raggedness under test is the one
    # MTP itself produces (divergent accept counts), not prompt-length spread.
    prompt = "Continue this list, one item per line. " + " ".join(
        FILLER[i % len(FILLER)] for i in range(a.words))
    url = f"http://127.0.0.1:{a.port}/v1/completions"
    stop = threading.Event()
    lock = threading.Lock()
    n = {"chunks": 0, "errors": 0, "started": 0}
    counting = [False]

    def worker(_wid):
        while not stop.is_set():
            body = json.dumps({
                "model": "default", "prompt": prompt,
                "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
            }).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            try:
                with lock:
                    n["started"] += 1
                with urllib.request.urlopen(req, timeout=1200) as r:
                    for raw in r:
                        if stop.is_set():
                            break
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
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
    # The fence the engine counters are differenced across. Printed in UTC so
    # it lines up with the server log's RFC3339 timestamps.
    fence = {"t_start": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}
    time.sleep(a.steady)
    with lock:
        counting[0] = False
        snap = dict(n)
    fence["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    stop.set()
    elapsed = time.time() - t0
    snap.update(fence)
    snap["elapsed_s"] = round(elapsed, 3)
    snap["driver_tok_s"] = round(snap["chunks"] / elapsed, 3) if elapsed else 0.0
    print(json.dumps(snap))
    for t in threads:
        t.join(timeout=3)

main()
PYEOF

# ── 5. The extractor.  Differences the cumulative `MTP[agg]` counters across
#       the cell fence.  Every reported quantity is a delta of raw counts. ─────
cat > "$OUT/fence.py" <<'PYEOF'
"""Δ of the engine's cumulative MTP counters across one cell's wall-clock fence.

`MTP[agg] ... steps=S ... committed=C batch_steps=BS` is emitted once per 64
proposed tokens and every count in it is cumulative for the process, so a
cell's own numbers are the difference between the last line at or before its
end and the last line at or before its start.

Reports nothing it did not measure: a cell with <2 usable lines prints NA with
a reason rather than 0.0, because a zero here would read as "MTP produced
nothing" when it means "the reporter never fired twice" (D18).
"""
import json, re, sys

TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
AGG = re.compile(r"MTP\[agg\][^\n]*")
KV = {k: re.compile(rf"\b{k}=(\d+)\b") for k in
      ("accepted", "proposed", "steps", "drafted_steps", "committed", "batch_steps")}

def parse(path, marker):
    out = []
    for line in open(path, errors="ignore"):
        if marker not in line:
            continue
        mt = TS.search(line)
        if not mt:
            continue
        rec = {"ts": mt.group(1)}
        ok = True
        for k, kp in KV.items():
            m = kp.search(line)
            if not m:
                ok = False
                break
            rec[k] = int(m.group(1))
        if ok:
            out.append(rec)
    return out

def at_or_before(rows, ts):
    hit = None
    for r in rows:
        if r["ts"] <= ts:
            hit = r
        else:
            break
    return hit

def delta(rows, t0, t1):
    a, b = at_or_before(rows, t0), at_or_before(rows, t1)
    if b is None:
        return None, "no MTP line inside or before this cell"
    if a is None:
        a = {k: 0 for k in KV}          # process started inside the cell
        a["ts"] = t0
    d = {k: b[k] - a[k] for k in KV}
    # The counters are cumulative and monotone. A negative delta is impossible
    # from one process, so it means the parse crossed something it should not
    # have (a second server writing the same file, a restart). Refuse rather
    # than report the number it would produce.
    neg = [k for k, v in d.items() if v < 0]
    if neg:
        return None, f"counters went BACKWARDS ({', '.join(neg)}) — log is not one process"
    if d["steps"] <= 0:
        return None, "no MTP step was reported between the fences"
    return d, None

def main():
    log, t0, t1, k, elapsed, marker = (
        sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
        float(sys.argv[5]), sys.argv[6])
    rows = parse(log, marker)
    d, why = delta(rows, t0, t1)
    res = {"k": k, "marker": marker, "lines_seen": len(rows), "elapsed_s": elapsed}
    if d is None:
        res.update({"status": "NA", "reason": why})
        print(json.dumps(res))
        return
    res.update({"status": "OK", **d})
    res["tok_per_step"] = round(d["committed"] / d["steps"], 4)
    res["accept_rate"] = (round(d["accepted"] / d["proposed"], 4)
                          if d["proposed"] else None)   # reported, never quoted
    res["mean_batch"] = (round(d["steps"] / d["batch_steps"], 3)
                         if d["batch_steps"] else None)
    res["agg_tok_s"] = round(d["committed"] / elapsed, 3) if elapsed else None
    # Per-user tok/s: aggregate over the batch that was actually resident.
    # Dividing by the requested K instead would understate it whenever the
    # scheduler did not run all K.
    res["per_user_tok_s"] = (round(res["agg_tok_s"] / res["mean_batch"], 3)
                             if res["mean_batch"] else None)
    print(json.dumps(res))

main()
PYEOF

# ── 5b. Crash detection.  A V4 prompt step that faults poisons the CUDA
#       context: the first failure is `CUBLAS_STATUS_EXECUTION_FAILED` or
#       `CUDA_ERROR_ILLEGAL_ADDRESS`, and every request after it fails
#       instantly.  A cell measured across that boundary is not a slow cell,
#       it is no cell — so it is reported VOID, never as a number.
#
#       Known live example: the #104 curve run on this same stack tip took
#       91,933 "errors" at B=8 in its MIXED-prompt arm, which was ONE crash
#       amplified by the driver's retry loop.
CRASH_RE='CUDA_ERROR_ILLEGAL_ADDRESS|CUBLAS_STATUS_EXECUTION_FAILED|panicked at|CUDA_ERROR_LAUNCH_FAILED|out of memory'
#       NOTE the shape of this function. `grep -c` prints `0` AND exits 1 when
#       it matches nothing, so the obvious `grep -cE ... || echo 0` emits TWO
#       lines ("0\n0") on every healthy poll; `[ "0\n0" -gt 0 ]` is not an
#       integer comparison, it errors, the crash branch never fires, and the
#       watchdog is silently dead while looking installed. Verified, then
#       written this way instead.
crash_count() {
  local n
  n=$(grep -cE "$CRASH_RE" "$1" 2>/dev/null)
  echo "${n:-0}"
}

# ── 5c. Length canary.  ONE streaming request per prompt length at B=1,
#       before any cell runs.  Session 8 measured V4 dying outright at
#       ~1,055-word prompts; the #104 curve crashed in the arm that drove
#       2,048-word prompts.  This locates the cliff on THIS build for ~2
#       minutes of box time, instead of discovering it at cell three.
#
#       If the canary cannot stream tokens at the sweep's own prompt length,
#       the sweep is not worth running and the script says so and stops.
cat > "$OUT/canary.py" <<'PYEOF'
import json, sys, time, urllib.request

FILLER = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
          "kilo lima mike november oscar papa quebec romeo sierra tango ").split()

def probe(port, words, want=8, timeout=180):
    prompt = "Continue this list, one item per line. " + " ".join(
        FILLER[i % len(FILLER)] for i in range(words))
    body = json.dumps({"model": "default", "prompt": prompt, "max_tokens": want,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    ch = json.loads(p)
                except Exception:
                    continue
                if (ch.get("choices") or [{}])[0].get("text", ""):
                    n += 1
    except Exception as e:
        return {"words": words, "ok": False, "tokens": n,
                "s": round(time.time() - t0, 1), "err": type(e).__name__}
    return {"words": words, "ok": n > 0, "tokens": n, "s": round(time.time() - t0, 1)}

port = int(sys.argv[1])
for w in [int(x) for x in sys.argv[2:]]:
    print(json.dumps(probe(port, w)), flush=True)
PYEOF

printf 'arm\tK\tstatus\ttok_per_step\tper_user_tok_s\tagg_tok_s\tmean_batch\tcommitted\tsteps\tbatch_steps\taccepted\tproposed\tdriver_tok_s\terrors\n' \
  > "$OUT/results.tsv"

# ── 6. One arm: start a server, sweep the cells, stop it. ─────────────────────
run_arm() {
  local arm="$1"; shift
  local slog="$OUT/server.$arm.log"
  say "arm $arm — starting server (env: ${*:-none})"
  # `env` EXECs rather than forks, so `$!` is the server's own pid and can be
  # killed directly. No `setsid` here on purpose: this script is already
  # detached by its launch command, and an extra session would make `$!` a pid
  # that is not the process group leader — after which `kill -- -$!` silently
  # kills nothing and the next arm starts against a server that is still up
  # (and still writing the log both arms are read from).
  # shellcheck disable=SC2086
  env RUST_LOG=info ARC_MTP_LOG_ACCEPTANCE=1 "$@" \
    "$BIN" serve -p "$PORT" -m "$MODEL" -a "$ARCH" --from-uqff "$UQFF" \
    --max-seqs "$MAX_SEQS" --mtp-depth "$MTP_DEPTH" --prefix-cache-n 0 \
    --paged-attn off \
    < /dev/null > "$slog" 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 1 450); do
    if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then up=1; break; fi
    kill -0 "$spid" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != "1" ]; then
    say "arm $arm — server never came up; tail:"
    tail -25 "$slog" | tee -a "$STATUS"
    kill -TERM -"$spid" 2>/dev/null || true
    return 1
  fi
  say "arm $arm — server up"

  # ── 🔑 PROVENANCE ASSERTION: the server running is the code we built.
  #
  #    `mistralrs-core` bakes `git rev-parse HEAD` at compile time
  #    (`build.rs::set_git_revision`) and logs it at startup as
  #    `git revision: <sha>` (`lib.rs:657`). Comparing that to the ref this
  #    script checked out is the only check that actually closes the loop.
  #
  #    WHY IT IS NOT OPTIONAL. A sibling script
  #    (`arc-tools/measure_v4_prefill_curve.sh:241,244`) built `-p arc-cli` —
  #    whose binary is named `arc` — and then launched
  #    `target/release/mistralrs`, a binary left there by an *earlier, unrelated*
  #    build. Its only guard was `[ -x "$BIN" ]`, an existence test that a stale
  #    binary passes happily. It measured code that never contained its
  #    instrumentation, and the result was clean, flat and plausible.
  #
  #    For an A/B that is the worst available failure: both arms run the same
  #    stale binary, the ratio comes out ~1.0, and it reads exactly like an
  #    honest refutation of the effect under test. Engagement markers catch the
  #    case where the stale binary predates them — they do NOT catch a stale
  #    binary that happens to contain them. Only the revision does.
  local want_sha logged_sha
  want_sha=$(cat "$OUT/sha.txt" 2>/dev/null)
  logged_sha=$(grep -m1 -o 'git revision: [0-9a-f]*' "$slog" 2>/dev/null | awk '{print $3}')
  if [ -z "$want_sha" ]; then
    say "arm $arm — provenance: SKIP_BUILD, no ref to compare (logged '${logged_sha:-none}')"
  elif [ -z "$logged_sha" ]; then
    say "arm $arm — PROVENANCE FAILED: server logged no 'git revision:' line; cannot prove \
what it is running. Refusing to measure."
    kill -TERM "$spid" 2>/dev/null || true
    return 1
  elif [ "$logged_sha" != "$want_sha" ]; then
    say "arm $arm — PROVENANCE FAILED: server is running $logged_sha but this run built \
$want_sha. That is a STALE BINARY; both arms would measure the same wrong code and return a \
flat ratio that reads as a refutation. Refusing to measure."
    kill -TERM "$spid" 2>/dev/null || true
    return 1
  else
    say "arm $arm — provenance OK: server is running $logged_sha (the ref this run built)"
  fi

  # ── Length canary, once, on the first arm. Locates the long-prompt cliff on
  #    THIS build before any cell is measured.
  if [ "$arm" = "OFF" ]; then
    say "arm $arm — length canary at B=1 ($CANARY_WORDS words)"
    # shellcheck disable=SC2086
    python3 "$OUT/canary.py" "$PORT" $CANARY_WORDS 2>&1 | tee "$OUT/canary.txt" | tee -a "$STATUS"
    if ! grep -q "\"words\": $PROMPT_WORDS, \"ok\": true" "$OUT/canary.txt"; then
      say "canary: the sweep's own prompt length ($PROMPT_WORDS words) could not stream a token"
      kill -TERM "$spid" 2>/dev/null || true
      return 1
    fi
    say "canary: $PROMPT_WORDS-word prompts stream — the sweep is in the safe regime"
  fi

  for k in $BATCHES; do
    local crashes_before
    crashes_before=$(crash_count "$slog")
    if ! kill -0 "$spid" 2>/dev/null; then
      say "arm $arm K=$k — server is already dead; every remaining cell is VOID"
      printf '%s\t%s\tVOID (server dead before cell)\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n' \
        "$arm" "$k" >> "$OUT/results.tsv"
      continue
    fi

    say "arm $arm K=$k — driving ${WARMUP}s warmup + ${STEADY}s steady"
    # Watchdog: the driver runs detached so a crash can end the cell in ~5s
    # instead of retrying into a poisoned context for the full window.
    local res
    ( python3 "$OUT/drive.py" --port "$PORT" --k "$k" --warmup "$WARMUP" \
        --steady "$STEADY" --words "$PROMPT_WORDS" --max-tokens "$MAX_TOKENS" \
        > "$OUT/drive.$arm.K$k.json" 2>"$OUT/drive.$arm.K$k.err" ) &
    local dpid=$!
    local crashed=0
    while kill -0 "$dpid" 2>/dev/null; do
      sleep 5
      if ! kill -0 "$spid" 2>/dev/null; then crashed=1; break; fi
      if [ "$(crash_count "$slog")" -gt "$crashes_before" ]; then crashed=1; break; fi
    done
    if [ "$crashed" = "1" ]; then
      kill -TERM "$dpid" 2>/dev/null || true
      sleep 2; kill -KILL "$dpid" 2>/dev/null || true
      say "arm $arm K=$k — VOID: server crashed mid-cell (first marker: $(grep -oE "$CRASH_RE" "$slog" | tail -1))"
      printf '%s\t%s\tVOID (server crashed mid-cell)\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n' \
        "$arm" "$k" >> "$OUT/results.tsv"
      grep -E "$CRASH_RE" "$slog" | tail -5 >> "$OUT/crash.$arm.K$k.txt" 2>/dev/null || true
      kill -TERM "$spid" 2>/dev/null || true
      return 1
    fi
    wait "$dpid" 2>/dev/null || true
    res=$(cat "$OUT/drive.$arm.K$k.json")
    if [ -z "$res" ]; then
      say "arm $arm K=$k — VOID: driver produced no result"
      printf '%s\t%s\tVOID (driver produced nothing)\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n' \
        "$arm" "$k" >> "$OUT/results.tsv"
      continue
    fi
    local t0 t1 el dtps errs
    t0=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['t_start'])" "$res")
    t1=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['t_end'])" "$res")
    el=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['elapsed_s'])" "$res")
    dtps=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['driver_tok_s'])" "$res")
    errs=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['errors'])" "$res")
    local fenced
    fenced=$(python3 "$OUT/fence.py" "$slog" "$t0" "$t1" "$k" "$el" "MTP[agg]")
    echo "$fenced" > "$OUT/fenced.$arm.K$k.json"
    python3 - "$arm" "$k" "$fenced" "$dtps" "$errs" >> "$OUT/results.tsv" <<'PYEOF'
import json, sys
arm, k, fenced, dtps, errs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4], sys.argv[5]
g = lambda key: fenced.get(key, "NA")
print("\t".join(str(x) for x in [
    arm, k, g("status"), g("tok_per_step"), g("per_user_tok_s"), g("agg_tok_s"),
    g("mean_batch"), g("committed"), g("steps"), g("batch_steps"),
    g("accepted"), g("proposed"), dtps, errs]))
PYEOF
    say "arm $arm K=$k -> $fenced"

    # ── Drain before the next cell.  The driver disconnects, but sequences the
    #    server is still generating would bleed their steps into the next
    #    cell's fence and quietly mix two batch sizes into one row.  Wait until
    #    the cumulative `committed` count stops moving.
    local last="" cur="" stable=0 waited=0
    while [ "$waited" -lt 150 ]; do
      sleep 5; waited=$((waited + 5))
      cur=$(grep -o 'MTP\[agg\][^\n]*committed=[0-9]*' "$slog" | tail -1 | grep -o 'committed=[0-9]*' || true)
      if [ -n "$cur" ] && [ "$cur" = "$last" ]; then
        stable=$((stable + 1))
        [ "$stable" -ge 2 ] && break
      else
        stable=0
      fi
      last="$cur"
    done
    say "arm $arm K=$k — drained after ${waited}s (counter settled at ${cur:-n/a})"
  done

  kill -TERM "$spid" 2>/dev/null || true
  local gone=0
  for _ in $(seq 1 30); do
    kill -0 "$spid" 2>/dev/null || { gone=1; break; }
    sleep 2
  done
  [ "$gone" = "1" ] || { kill -KILL "$spid" 2>/dev/null || true; sleep 5; }
  # The next arm must not inherit this arm's GPU memory or its port. If the
  # port is still bound, the next `serve` would fail to bind and the arm would
  # be measured against THIS server — same binary, wrong env, silently.
  for _ in $(seq 1 30); do
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break
    sleep 2
  done
  if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    say "arm $arm — WARNING: port $PORT still answering after kill; refusing to start another arm"
    return 1
  fi
  say "arm $arm — server down, port $PORT free"
  return 0
}

run_arm OFF || say "arm OFF did not complete"
run_arm ON ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1 || say "arm ON did not complete"

# ── Drift control.  The arms are necessarily sequential (a 2m30s V4 load makes
#    interleaving cost 8 server starts instead of 2), so ON is always measured
#    later than OFF and anything that degrades with time — thermals, memory
#    fragmentation, another chain landing on the box — biases the comparison in
#    one direction.  Re-running ONE cell of the control at the end turns "it
#    might have drifted" into a measured bound for ~4 minutes.
BATCHES=128 run_arm OFF2 || say "drift control did not complete"

# ── 7. Engagement assertions, then the answer. ────────────────────────────────
{
  echo "=== ref ==="
  cat "$OUT/sha.txt" 2>/dev/null || echo "(SKIP_BUILD)"
  echo
  echo "=== results.tsv ==="
  cat "$OUT/results.tsv"
  echo
  echo "=== ENGAGEMENT (D18: a log line present in ON and absent in OFF) ==="
  on="$OUT/server.ON.log"; off="$OUT/server.OFF.log"
  p() { if grep -qF "$2" "$1" 2>/dev/null; then echo PRESENT; else echo ABSENT; fi; }
  echo "ON  'per-sequence KV advance is ON'   : $(p "$on"  'per-sequence KV advance is ON')  [want PRESENT]"
  echo "ON  'Ragged batch admission is ON'    : $(p "$on"  'Ragged batch admission is ON')   [want PRESENT]"
  echo "ON  'cannot honour it'                : $(p "$on"  'cannot honour it')               [want ABSENT]"
  echo "OFF 'per-sequence KV advance is ON'   : $(p "$off" 'per-sequence KV advance is ON')  [want ABSENT]"
  echo "OFF 'Ragged batch admission is ON'    : $(p "$off" 'Ragged batch admission is ON')   [want ABSENT]"
  echo
  echo "refusal reasons seen (ON arm must show none):"
  grep -h "cannot honour it" "$on" 2>/dev/null | sed 's/^.*cannot honour/cannot honour/' | sort -u | head -5 || true
  echo
  echo "=== ANSWER ==="
  python3 - "$OUT/results.tsv" "$on" "$off" <<'PYEOF'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
by = {(r[0], int(r[1])): r for r in rows}

def has(path, needle):
    try:
        return any(needle in l for l in open(path, errors="ignore"))
    except OSError:
        return False

engaged = (has(sys.argv[2], "per-sequence KV advance is ON")
           and has(sys.argv[2], "Ragged batch admission is ON")
           and not has(sys.argv[2], "cannot honour it")
           and not has(sys.argv[3], "per-sequence KV advance is ON"))

def num(arm, k, col):
    r = by.get((arm, k))
    if not r or r[2] != "OK":
        return None
    try:
        return float(r[col])
    except (ValueError, TypeError):
        return None

print("ANSWER[engaged] " + ("YES — the treatment arm ran the mode under test"
                            if engaged else
                            "NO — the ON arm did not engage; every ratio below "
                            "is a measurement of the SAME code path and refutes "
                            "nothing (D18)"))
for k in sorted({int(r[1]) for r in rows}):
    o, n = num("OFF", k, 3), num("ON", k, 3)
    ou, nu = num("OFF", k, 4), num("ON", k, 4)
    if o and n:
        print(f"ANSWER[tok_per_step] K={k:<4} OFF={o:.4f} ON={n:.4f} "
              f"delta={(n / o - 1) * 100:+.1f}%")
    else:
        print(f"ANSWER[tok_per_step] K={k:<4} OFF={o} ON={n}  (NA — see results.tsv)")
    if ou and nu:
        print(f"ANSWER[per_user_tok_s] K={k:<4} OFF={ou:.3f} ON={nu:.3f} "
              f"delta={(nu / ou - 1) * 100:+.1f}%")
o1, o2 = num("OFF", 128, 3), num("OFF2", 128, 3)
if o1 and o2:
    print(f"ANSWER[drift] control K=128 re-run after the ON arm: first={o1:.4f} "
          f"last={o2:.4f} drift={(o2 / o1 - 1) * 100:+.1f}%  <- must be small "
          f"next to the ON-vs-OFF delta, or the arms are not comparable")
else:
    print(f"ANSWER[drift] control re-run unavailable (first={o1}, last={o2}) — "
          "the ON-vs-OFF delta is UNBOUNDED for time drift")
print("ANSWER[claim] PR #92's arithmetic predicted cohort 1.1297 -> per-sequence "
      "1.4783 at B=128 (+31%). Session 8 measured cohort 1.0558 on an H200. "
      "The K=128 row above is the hardware verdict on the +31%.")
PYEOF
} | tee "$OUT/summary.txt"

say "DONE — $OUT/summary.txt"
