#!/bin/bash
# Parent system: ArcKernels + ArcQuant/Legacy (blockwise FP8).
#
# THREE-ARM FP8 DENSE-GEMM THRESHOLD SWEEP — the measurement that is allowed
# to move `ARC_FP8_CUBLAS_MIN_M` off 512.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY THREE ARMS, AND WHY A TWO-ARM RESULT MUST NOT BE MERGED
#
# `arc_fp8_cublas_min_m`'s own doc (blockwise_fp8/mod.rs) and the WMMA
# kernel's doc (blockwise_fp8/ops.rs, `fp8_wmma_enabled`) both record the
# standing instruction: the only sweep ever taken compared (a) dequantize +
# cuBLASLt against (c) the scalar tiled kernel, BEFORE #200 landed the
# (b) tensor-core WMMA GEMM — and before the rank-3 flatten, so arm (a) was a
# B-way batched GEMV wearing a GEMM's name. Freezing a threshold from that
# sweep would leave (b) dead code at every M that matters, silently. This
# script is the three-arm re-run those docs demand:
#
#   (a) dequantize_w() + cuBLASLt        ARC_FP8_CUBLAS_MIN_M=5
#   (b) native FP8 tensor-core WMMA      ARC_FP8_CUBLAS_MIN_M=1000000, ARC_NO_FP8_WMMA=0
#   (c) native scalar fp8_matmul_tiled   ARC_FP8_CUBLAS_MIN_M=1000000, ARC_NO_FP8_WMMA=1
#
# ONE binary. Every leg differs only in environment; a leg that needed a
# rebuild would be measuring a different artifact. The env gates are latched
# per process (OnceLock/LazyLock), so every leg gets a FRESH server process —
# re-exporting into a running server changes nothing and would silently
# measure the previous leg's setting.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE LADDER: M VARIES BY OFFERED CONCURRENCY, NOT ONLY BY THRESHOLD
#
# The deliverable is "the first M at which (a) beats min((b), (c))". A grid
# that sweeps only ARC_FP8_CUBLAS_MIN_M at one fixed batch cannot name that M:
# every leg with MIN_M <= batch routes the same decode shape to the same
# kernel and the legs replicate each other. So the threshold candidates
# {5, 8, 64, 256, 512} are swept as the OFFERED DECODE BATCH — M concurrent
# greedy completions, T=1 per step, so the dense projections see m_rows = M —
# and each ladder point runs all three arms. 15 legs.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT MAKES A LEG COUNT (all three, or the leg is VOID)
#
# 1. ENGAGEMENT, asserted per leg. Every leg runs with ARC_LOG_FP8_DISPATCH=1;
#    the dispatch sites print `[arc-fp8-dispatch] path=<kernel>` once per path
#    per process. The leg's expected kernel must be PRESENT in the server log
#    and both rival kernels ABSENT. A leg that cannot prove which kernel
#    served it is measuring an assumption (`ARC_NO_DEDICATED_DECODE`'s two
#    void A/Bs are the record of what that costs). In particular: if the WMMA
#    arm's log shows `path=tiled`, WMMA eligibility silently failed and the
#    arm is the control wearing the treatment's name — VOID, loudly.
# 2. A MINIMUM-TOKEN FLOOR. Rates on fewer than MIN_TOKENS_FLOOR generated
#    tokens are warm-up noise. The rate divides by SUMMED
#    `usage.completion_tokens` — the tokens the server says it generated —
#    NEVER by requested max_tokens (a leg that stalls and returns short would
#    otherwise report a healthy rate for work it did not do).
# 3. A TOKEN CANARY vs the baseline leg (arm (c) at M=5 — the shipped scalar
#    kernel, nearest to b=1). Greedy, fixed seed, fixed prompt. The arms are
#    DIFFERENT NUMERICS by design (f32 scalar vs tensor-core promotion vs
#    dequant-BF16 GEMM), so divergence at some index is expected and REPORTED,
#    not failed; what hard-fails a leg is a degenerate stream (< 5 tokens or
#    no finish_reason) — a failure to generate, which no rate can offset.
#
# The final table names, per ladder M, the three clean rates and min((b),(c)),
# and then the FIRST M at which (a) beats that min — the value
# ARC_FP8_CUBLAS_MIN_M should default to. If any arm at that M is void, the
# crossover is NOT ESTABLISHED and the default stays 512.
#
# USAGE (on the GPU box; do not run without a GPU — D14)
#   setsid nohup bash arc-tools/fp8_threshold_sweep.sh </dev/null \
#     > /root/logs/fp8_sweep.log 2>&1 &
#   cat /root/logs/fp8sweep/summary.txt
#
set -u

ARCENV="${ARCENV:-/root/arcenv.sh}"
WT="${WT:-/root/arc}"
BIN="${BIN:-$WT/target/release/mistralrs}"
MODEL="${MODEL:-/root/models/v4-src}"
UQFF="${UQFF:-}"                        # optional: --from-uqff when set
ARCH="${ARCH:-deepseekv4}"
PREFLIGHT="${PREFLIGHT:-/root/arc-tools/gpu_box_preflight.sh}"
OUT="${OUT:-/root/logs/fp8sweep}"
STATUS="${STATUS:-/root/logs/fp8_sweep.status}"
PORT="${PORT:-1241}"
SEED="${SEED:-1234}"
MAX_TOKENS="${MAX_TOKENS:-48}"          # per request; the floor is on the SUM
MIN_TOKENS_FLOOR="${MIN_TOKENS_FLOOR:-1000}"
MAX_SEQS="${MAX_SEQS:-512}"             # must be >= the largest ladder point
LADDER="${LADDER:-5 8 64 256 512}"      # offered decode batch = candidate M
LOCK="${LOCK:-/root/locks/gpu.lock}"
LOCK_WAIT="${LOCK_WAIT:-2400}"
OWNER="fp8sweep"

mkdir -p "$OUT" "$(dirname "$STATUS")" "$(dirname "$LOCK")"
: > "$STATUS"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$STATUS"; }
die() { say "ABORT: $*"; echo "ABORT: $*" >> "$OUT/summary.txt"; exit 2; }

say "fp8 three-arm threshold sweep — start"

# ── GPU lock with OWNER-MATCHED RELEASE and evidence-based stale reclaim
#    (same discipline as arcspec_token_identity_b8.sh; see the incident notes
#    there for why release refuses locks this process does not own).
release_lock() {
  if [ -f "$LOCK" ] && grep -q "^$OWNER pid=$$ " "$LOCK" 2>/dev/null; then
    rm -f "$LOCK"
  fi
}
STALE_AFTER="${STALE_AFTER:-120}"
lock_is_stale() {
  [ -f "$LOCK" ] || return 1
  local age holder pid used procs
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
  [ "$age" -ge "$STALE_AFTER" ] || return 1
  holder=$(cat "$LOCK" 2>/dev/null)
  pid=$(printf '%s' "$holder" | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
  if [ -n "$holder" ] && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    return 1
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
      say "lock: RECLAIMING stale lock (age>=${STALE_AFTER}s, owner absent or dead, GPU idle)"
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
acquire_lock || die "GPU lock held by another chain for >${LOCK_WAIT}s"

[ -f "$ARCENV" ] && . "$ARCENV"
if [ -x "$PREFLIGHT" ]; then
  "$PREFLIGHT" > "$OUT/preflight.log" 2>&1 || die "preflight refused the box"
  say "preflight OK"
else
  say "WARN: no preflight at $PREFLIGHT — continuing without it"
fi
[ -x "$BIN" ] || die "$BIN not found — build once, then sweep (one binary, no rebuild)"
SHA=$(git -C "$WT" rev-parse HEAD 2>/dev/null)
say "binary ref: $SHA"

# ── The prober: one canary + (M-1) filler prompts, all concurrent, all
#    greedy. Non-streaming so `usage.completion_tokens` is authoritative.
cat > "$OUT/probe.py" <<'PYEOF'
"""Drive M concurrent greedy completions; report summed completion_tokens
(from usage — NEVER from requested max_tokens), wall seconds over the
concurrent burst, and the canary text (request 0)."""
import argparse, json, sys, threading, time, urllib.request

CANARY = "List the first twelve prime numbers, separated by commas, then stop."
FILLERS = [
    "Name the seven continents, one per line.",
    "Count from one to twenty in words, separated by spaces.",
    "Write the first eight letters of the alphabet, separated by dashes.",
    "Name eight colours of the visible spectrum, comma separated.",
    "List eight consecutive even numbers starting at two.",
    "Name the days of the week in order, one per line.",
    "List the first eight powers of two, comma separated.",
]

def one(port, prompt, max_tokens, seed, timeout):
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "seed": seed, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = (d.get("choices") or [{}])[0]
    usage = d.get("usage") or {}
    return {
        "text": ch.get("text", ""),
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens", 0),
    }

def burst(port, m, max_tokens, seed, timeout):
    prompts = [CANARY] + [FILLERS[i % len(FILLERS)] for i in range(m - 1)]
    out = [None] * m
    errs = []
    def w(i):
        try:
            out[i] = one(port, prompts[i], max_tokens, seed, timeout)
        except Exception as e:  # noqa: BLE001 — a leg must record, not crash
            errs.append(f"req[{i}]: {e}")
            out[i] = {"text": "", "finish_reason": None, "completion_tokens": 0}
    t0 = time.monotonic()
    ts = [threading.Thread(target=w, args=(i,)) for i in range(m)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.monotonic() - t0
    return out, wall, errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--warmup", action="store_true")
    a = ap.parse_args()

    if a.warmup:
        # Small unmeasured round: load caches, emit the first-dispatch log
        # lines, absorb ramp. Its tokens are never counted.
        burst(a.port, min(a.m, 8), 16, a.seed, a.timeout)
        json.dump({"warmup": True}, sys.stdout)
        return

    out, wall, errs = burst(a.port, a.m, a.max_tokens, a.seed, a.timeout)
    total = sum(r["completion_tokens"] for r in out)
    json.dump({
        "m": a.m,
        "wall_s": wall,
        "completion_tokens_sum": total,      # from usage — the only numerator
        "tok_per_s": (total / wall) if wall > 0 else 0.0,
        "canary_text": out[0]["text"],
        "canary_finish": out[0]["finish_reason"],
        "canary_tokens": out[0]["completion_tokens"],
        "errors": errs,
        "short_streams": sum(1 for r in out if r["completion_tokens"] < 5),
    }, sys.stdout)

main()
PYEOF

# ── One leg = one fresh server process + warm-up + measured burst.
#    $1 arm name, $2 ladder M, then env pairs for the arm.
run_leg() {
  local arm="$1" m="$2"; shift 2
  local leg="${arm}.M${m}"
  local slog="$OUT/server.$leg.log"
  say "leg $leg — starting server ($*)"

  local uqff_args=()
  [ -n "$UQFF" ] && uqff_args=(--from-uqff "$UQFF")
  # shellcheck disable=SC2086
  env RUST_LOG=info ARC_LOG_FP8_DISPATCH=1 "$@" \
    "$BIN" serve -p "$PORT" -m "$MODEL" -a "$ARCH" "${uqff_args[@]}" \
    --max-seqs "$MAX_SEQS" --prefix-cache-n 0 --seed "$SEED" \
    < /dev/null > "$slog" 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 1 450); do
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$spid" 2>/dev/null || break
    sleep 2
  done
  [ "$up" = "1" ] || { say "leg $leg — server never came up"; tail -20 "$slog" | tee -a "$STATUS"; return 1; }

  # Provenance: the running server must be the code we think it is.
  local logged
  logged=$(grep -m1 -o 'git revision: [0-9a-f]*' "$slog" | awk '{print $3}')
  if [ -n "$SHA" ] && [ -n "$logged" ] && [ "$logged" != "$SHA" ]; then
    say "leg $leg — PROVENANCE FAILED: server $logged != worktree $SHA"
    kill -TERM "$spid" 2>/dev/null || true
    return 1
  fi

  say "leg $leg — warm-up (unmeasured)"
  python3 "$OUT/probe.py" --port "$PORT" --m "$m" --max-tokens "$MAX_TOKENS" \
    --seed "$SEED" --warmup > /dev/null || { say "leg $leg — warm-up failed"; kill -TERM "$spid" 2>/dev/null; return 1; }

  say "leg $leg — measured burst: M=$m x max_tokens=$MAX_TOKENS"
  python3 "$OUT/probe.py" --port "$PORT" --m "$m" --max-tokens "$MAX_TOKENS" \
    --seed "$SEED" > "$OUT/$leg.json" || { say "leg $leg — probe failed"; kill -TERM "$spid" 2>/dev/null; return 1; }

  kill -TERM "$spid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$spid" 2>/dev/null || break; sleep 2; done
  kill -KILL "$spid" 2>/dev/null || true
  sleep 5

  # ENGAGEMENT: expected kernel present, both rivals absent, in the server log.
  local want forbid1 forbid2
  case "$arm" in
    cublaslt) want="dequant_cublaslt"; forbid1="wmma";             forbid2="tiled" ;;
    wmma)     want="wmma";             forbid1="dequant_cublaslt"; forbid2="tiled" ;;
    tiled)    want="tiled";            forbid1="wmma";             forbid2="dequant_cublaslt" ;;
    *) say "leg $leg — unknown arm '$arm'"; return 1 ;;
  esac
  local ok=1
  grep -q "\[arc-fp8-dispatch\] path=$want " "$slog" || ok=0
  grep -q "\[arc-fp8-dispatch\] path=$forbid1 " "$slog" && ok=0
  grep -q "\[arc-fp8-dispatch\] path=$forbid2 " "$slog" && ok=0
  {
    echo "leg=$leg"
    grep '\[arc-fp8-dispatch\]' "$slog" | sed 's/^/  /'
    echo "  engagement=$([ "$ok" = 1 ] && echo PASS || echo FAIL)"
  } >> "$OUT/engagement.txt"
  if [ "$ok" != 1 ]; then
    say "leg $leg — ENGAGEMENT FAILED (want $want, forbid $forbid1/$forbid2) — leg VOID"
    echo '{"void": "engagement"}' > "$OUT/$leg.void"
  fi
  say "leg $leg — done"
  return 0
}

# ── The 15 legs. Baseline (canary reference) FIRST: tiled @ smallest M.
MIN_M_NATIVE=1000000   # keeps every real m on the native path in arms (b),(c)
for m in $LADDER; do
  run_leg tiled    "$m" ARC_FP8_CUBLAS_MIN_M=$MIN_M_NATIVE ARC_NO_FP8_WMMA=1 \
    || die "tiled M=$m leg failed"
done
for m in $LADDER; do
  run_leg wmma     "$m" ARC_FP8_CUBLAS_MIN_M=$MIN_M_NATIVE ARC_NO_FP8_WMMA=0 \
    || die "wmma M=$m leg failed"
done
for m in $LADDER; do
  run_leg cublaslt "$m" ARC_FP8_CUBLAS_MIN_M=5 \
    || die "cublaslt M=$m leg failed"
done

# ── Summary: floors, canaries, the crossover table, the verdict.
python3 - "$OUT" "$MIN_TOKENS_FLOOR" $LADDER <<'PYEOF' | tee "$OUT/summary.txt"
import json, os, sys

out, floor = sys.argv[1], int(sys.argv[2])
ladder = [int(x) for x in sys.argv[3:]]
ARMS = ["tiled", "wmma", "cublaslt"]

def load(arm, m):
    leg = f"{arm}.M{m}"
    if os.path.exists(os.path.join(out, leg + ".void")):
        return None, "VOID(engagement)"
    p = os.path.join(out, leg + ".json")
    if not os.path.exists(p):
        return None, "MISSING"
    with open(p) as f:
        d = json.load(f)
    if d.get("errors"):
        return d, f"ERRORS({len(d['errors'])})"
    if d["completion_tokens_sum"] < floor:
        return d, f"VOID(floor {d['completion_tokens_sum']}<{floor})"
    if d["canary_tokens"] < 5 or not d.get("canary_finish"):
        return d, "VOID(degenerate canary)"
    return d, "CLEAN"

base, base_status = load("tiled", ladder[0])
base_text = base["canary_text"] if base else ""

def canary(d):
    if d is None or base is None:
        return "-"
    a, b = base_text, d["canary_text"]
    if a == b:
        return "identical"
    n = min(len(a), len(b))
    i = next((j for j in range(n) if a[j] != b[j]), n)
    return f"diverges@char{i}"

print("=== FP8 three-arm threshold sweep ===")
print(f"floor: rate counted only when summed usage.completion_tokens >= {floor}")
print(f"canary baseline: tiled @ M={ladder[0]} [{base_status}]")
print()
print(f"{'M':>5} {'arm':>9} {'tokens':>7} {'wall_s':>8} {'tok/s':>9} {'status':>26} canary_vs_baseline")
rates = {}
for m in ladder:
    for arm in ARMS:
        d, status = load(arm, m)
        rates[(arm, m)] = d["tok_per_s"] if (d and status == "CLEAN") else None
        if d:
            print(f"{m:>5} {arm:>9} {d['completion_tokens_sum']:>7} {d['wall_s']:>8.1f} "
                  f"{d['tok_per_s']:>9.1f} {status:>26} {canary(d)}")
        else:
            print(f"{m:>5} {arm:>9} {'-':>7} {'-':>8} {'-':>9} {status:>26} -")
print()
print("=== CROSSOVER (the number ARC_FP8_CUBLAS_MIN_M may move to) ===")
crossover = None
for m in ladder:
    a = rates[("cublaslt", m)]
    b = rates[("wmma", m)]
    c = rates[("tiled", m)]
    if a is None or b is None or c is None:
        print(f"M={m:>4}: NOT ESTABLISHED — a leg is void; no claim about this M")
        continue
    native = min(b, c)
    beat = a > native
    print(f"M={m:>4}: cublaslt {a:.1f} vs min(wmma {b:.1f}, tiled {c:.1f}) = {native:.1f}"
          f" -> {'cuBLASLt WINS' if beat else 'native wins'}")
    if beat and crossover is None:
        crossover = m
print()
if crossover is not None:
    print(f"VERDICT: first clean M where cuBLASLt beats min(WMMA, tiled): {crossover}.")
    print(f"  Recommended default: ARC_FP8_CUBLAS_MIN_M={crossover} — merge WITH these")
    print("  rows quoted in the PR (a threshold is a claim about every value it")
    print("  excludes; these are the rows that back the claim).")
else:
    print("VERDICT: no clean crossover on this ladder — the default STAYS 512.")
    print("  Do not move the threshold from a partial table; re-run the void legs.")
PYEOF

say "DONE — $OUT/summary.txt  (engagement detail: $OUT/engagement.txt)"
