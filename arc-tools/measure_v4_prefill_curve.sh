#!/usr/bin/env bash
#
# measure_v4_prefill_curve.sh — test, do not assume, the claim that V4's
# long-prompt serving death is quadratic prefill from the missing fused
# head_dim-512 attention kernel.
#
# ---------------------------------------------------------------------------
# WHAT IS ACTUALLY BEING TESTED
# ---------------------------------------------------------------------------
# The claim (PR #97): V4 has no fused attention kernel at head_dim = 512
# (`flash_sinks_ok` = {64,80,96,112,128,192,256}), so every prefill layer runs
# unfused through `sinks_attn_cpu`, which `repeat_kv`-expands K AND V to all 64
# heads with a real `Tensor::cat` (attention/mod.rs:98) and materialises the
# full score matrix. Per layer, bf16, CSA (`n_keys = 1.25*T`):
#
#     repeat_kv K+V : 2 * 64 * 1.25T * 512 * 2 B  =  163,840 * T   bytes
#     scores, twice : 2 * 64 * 1.25T^2     * 2 B  =        320 * T^2 bytes
#
# That was fitted to TWO measured points (9-word and 198-word prompts, FACTS
# session 8). Two points determine a two-parameter curve exactly, so the fit
# is CONSISTENT WITH the mechanism and is not evidence for it. This script
# exists to make it evidence, or to kill it.
#
#   ⚠️  THE 43 LAYERS DO NOT ACCUMULATE. They run sequentially and each layer's
#   score matrices are freed before the next layer allocates, so the predicted
#   PEAK is ONE layer's working set — ~0.5 GB at T=1024, NOT 43 * 0.5 GB. Any
#   refutation criterion built on "43x" would reject a correct mechanism.
#   What is falsifiable is the SHAPE of the curve, not its height.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED PREDICTIONS AND REFUTATIONS  (read before the numbers exist)
# ---------------------------------------------------------------------------
# ARM B — memory law (cheap, wide span, runs FIRST because it can kill the
# mechanism in ~6 minutes before Arm A spends an hour):
#
#   predicted peak transient = (320*T^2 + 163840*T) bytes
#     T=512  -> 160 MiB      T=1024 ->  480 MiB
#     T=2048 -> 1600 MiB      T=4096 -> 5760 MiB
#   predicted adjacent ratios: 3.0, 3.3, 3.6   (a linear mechanism gives ~2.0)
#
#   REFUTED IF: any adjacent peak ratio < 2.2, OR peak(4096) < 2048 MiB (vs 5760 predicted).
#   Either means the score matrix is not being materialised the way claimed,
#   and the mechanism is wrong EVEN IF the timings fit.
#
# ARM A — timing law:
#   predicted prefill(1024)/prefill(512) = 4.0   (linear mechanism gives 2.0)
#   REFUTED IF: that ratio < 2.8 (the geometric midpoint of 2 and 4), OR the
#   fitted quadratic term contributes < 50% of prefill(1024).
#
# ARM C — mechanism discriminator (optional, one extra request):
#   `ARC_V4_WINDOW_ONLY=1` removes the compressed branch, taking n_keys from
#   1.25T to T. If prefill is score-matrix-bound, prefill falls by ~1/1.25^2
#   = 0.64 at fixed T. If prefill is MoE/dispatch-bound it barely moves.
#   REFUTED IF: the window-only ratio is > 0.90.
#
# WHAT EACH OUTCOME MEANS FOR THE PROJECT — say this out loud, because six
# workstreams were re-ranked on the un-tested fit:
#   * quadratic confirmed  -> task #21 (fused hd-512 kernel) is the fix, and
#     the rerank was right.
#   * LINEAR               -> the cost is per-token (MoE / dispatch / host
#     overhead), the fused attention kernel does NOT fix long prompts, and the
#     rerank of six workstreams was made on a bad curve. This is the outcome
#     that costs the most to learn late, which is why it is worth $9 now.
#   * timings fit, memory does not -> some other quadratic is at work; #21 is
#     still not established as the lever.
#
# ---------------------------------------------------------------------------
# EXIT CODES  (never conflate "could not answer" with "refuted")
# ---------------------------------------------------------------------------
#   0  the box answered. VERDICT is in the JSON: supported | refuted | mixed
#   2  the environment could not answer (gate failed, model missing, server
#      never healthy, canary failed, GPU not exclusively ours). NOT a
#      refutation. Nothing about the mechanism is learned.
#   1  script fault.
#
# ---------------------------------------------------------------------------
# USAGE (D15: the coordinator executes this; agents never call `runcrate`)
# ---------------------------------------------------------------------------
#   cd /root/arc
#   setsid nohup bash arc-tools/measure_v4_prefill_curve.sh \
#       < /dev/null > /root/logs/prefill_curve.log 2>&1 &
#   tail -f /root/logs/prefill_curve.status     # one line, overwritten
#   cat /root/results/v4_prefill_curve.json     # machine-readable, at the end
#
# Knobs (env):
#   ARC_PC_SRC        source checkpoint dir          (default /workspace/src)
#   ARC_PC_UQFF       first UQFF shard               (default /workspace/uqff/qtip2b-0.uqff)
#   ARC_PC_PORT       server port                    (default 1234)
#   ARC_PC_BUDGET_S   wall-clock budget for Arm A    (default 4200 = 70 min)
#   ARC_PC_REQ_TMO    per-request timeout, seconds   (default 2400)
#   ARC_PC_MEM_WIN    Arm B sampling window, seconds (default 90)
#   ARC_PC_TIMING_TS  Arm A ladder    (default "128 256 512 1024")
#   ARC_PC_SUBTRACT_TS  T values that also get the max_tokens=5 probe, i.e.
#                     where the decode step is MEASURED rather than modelled
#                     (default "128 256"). Decode is linear in context while
#                     prefill (the quantity under test) is not, so decode's
#                     share of t1 falls as 1/T: measuring it at the cheap end
#                     and extrapolating costs almost nothing in accuracy and
#                     nearly halves the arm. Every point records which it got,
#                     and what share of t1 the subtraction was, so a reader can
#                     check that the modelled part stayed negligible.
#   ARC_PC_MEM_TS     Arm B ladder    (default "512 1024 2048 4096")
#   ARC_PC_EDGE       1 = also probe T=8192 LAST, to test the OOM prediction
#                     for real. Off by default: it may kill the server, so it
#                     runs only after every other number is banked.
#   ARC_PC_DISCRIM    1 = run Arm C (ARC_V4_WINDOW_ONLY discriminator)
#   ARC_PC_SKIP_BUILD 1 = a usable ./target/release/mistralrs already exists
#
# COST, IF THE MECHANISM IS REAL (that is its own prediction, so read it as one):
#   Arm B ~6 min. Arm A ~1.8 h for the full 128/256/512/1024 ladder — T=1024 is
#   ~75 min of that by itself. Under the default 70-min budget the ladder will
#   therefore stop after T=512 and report {128, 256, 512}, which still spans 4x
#   and separates 16x (quadratic) from 4x (linear) decisively. Raise
#   ARC_PC_BUDGET_S to ~7000 only if the 1024 point is worth ~$3 more and the
#   queue can afford it. If the mechanism is FALSE the whole run is ~15 min,
#   because a linear prefill is ~30x cheaper at T=1024 — the cheap outcome is
#   the one that refutes.
#
# The box is released (server killed) on every exit path — two other jobs are
# queued behind this one (measure_radix_topk_ab.sh, wave63_turboquant_hd512_gate.sh).
#
set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Paths, knobs, status plumbing
# ---------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO" || { echo "FAIL[REPO] cannot cd to $REPO"; exit 2; }

ARC_PC_SRC="${ARC_PC_SRC:-/workspace/src}"
ARC_PC_UQFF="${ARC_PC_UQFF:-/workspace/uqff/qtip2b-0.uqff}"
ARC_PC_PORT="${ARC_PC_PORT:-1234}"
ARC_PC_BUDGET_S="${ARC_PC_BUDGET_S:-4200}"
ARC_PC_REQ_TMO="${ARC_PC_REQ_TMO:-2400}"
ARC_PC_MEM_WIN="${ARC_PC_MEM_WIN:-90}"
ARC_PC_TIMING_TS="${ARC_PC_TIMING_TS:-128 256 512 1024}"
ARC_PC_SUBTRACT_TS="${ARC_PC_SUBTRACT_TS:-128 256}"
ARC_PC_MEM_TS="${ARC_PC_MEM_TS:-512 1024 2048 4096}"
ARC_PC_EDGE="${ARC_PC_EDGE:-0}"
ARC_PC_DISCRIM="${ARC_PC_DISCRIM:-0}"
ARC_PC_SKIP_BUILD="${ARC_PC_SKIP_BUILD:-0}"

LOGDIR="${LOGDIR:-/root/logs}"
RESDIR="${RESDIR:-/root/results}"
mkdir -p "$LOGDIR" "$RESDIR" 2>/dev/null || true
STATUS="$LOGDIR/prefill_curve.status"
SERVERLOG="$LOGDIR/prefill_curve.server.log"
OUT="$RESDIR/v4_prefill_curve.json"
RAW="$RESDIR/v4_prefill_curve.raw.jsonl"
BASE_URL="http://127.0.0.1:${ARC_PC_PORT}"
: > "$RAW"

T0=$(date +%s)
say()  { printf '[%6ss] %s\n' "$(( $(date +%s) - T0 ))" "$*"; }
stat_() { printf '[%6ss] %s\n' "$(( $(date +%s) - T0 ))" "$*" > "$STATUS"; }
step() { say "$*"; stat_ "$*"; }

SERVER_PID=""
cleanup() {
    local rc=$?
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        say "releasing the box: killing server pid $SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -f "mistralrs serve -p ${ARC_PC_PORT}" 2>/dev/null || true
    say "exit rc=$rc"
    stat_ "DONE rc=$rc — see $OUT"
}
trap cleanup EXIT

write_json_min() {  # $1 = verdict, $2 = note — used on the early-exit paths
    python3 - "$1" "$2" "$OUT" "$RAW" <<'PY' 2>/dev/null || true
import json, os, sys
verdict, note, out, raw = sys.argv[1:5]
rows = []
if os.path.exists(raw):
    for ln in open(raw):
        ln = ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except Exception: pass
json.dump({"verdict": verdict, "note": note, "rows": rows,
           "what_this_means": "the environment could not answer. This is NOT a "
                              "refutation — nothing was learned about the mechanism."},
          open(out, "w"), indent=2)
PY
}

# `env_fail` = exit 2. Reserved for "the box could not answer the question".
env_fail() { say "FAIL[ENV] $*"; stat_ "ENV-FAIL: $*"; write_json_min "insufficient" "$*"; exit 2; }

# ---------------------------------------------------------------------------
# 1. HARD GATE — box preflight (the silent-PTX trap)
# ---------------------------------------------------------------------------
# Driver 580.173.02 caps at CUDA 13.0 while the only toolkit is 13.1.
# candle-kernels ships PTX (build_ptx), the driver refuses the newer ISA, and
# under the runtime API that failure lands in cudaGetLastError() rather than
# cudaDeviceSynchronize() — a clean exit code over an unwritten buffer. A box
# without `cuda-compat-13-1` + LD_LIBRARY_PATH=/usr/local/cuda/compat does not
# necessarily crash; it produces plausible wrong numbers. Refusing to proceed
# is the whole point of the gate.
PREFLIGHT="arc-tools/gpu_box_preflight.sh"
step "gate 1/3: $PREFLIGHT"
if [ ! -f "$PREFLIGHT" ]; then
    env_fail "$PREFLIGHT is absent on this checkout. It lands in commit 0f1805a7e; \
check out a ref that contains it. Refusing to measure without the PTX gate — an \
ungated box reports a BETTER number, not a worse one, so a missing gate is \
indistinguishable from a passing one in the results."
fi
# ⚠️ `source preflight || handler` DOES NOT WORK, and silently. When sourced,
# the script ends with `[ "$_arc_pf_sourced" = "1" ] || exit 1` — on a failing
# box that `[` succeeds, short-circuits the `||`, and becomes the last command,
# so `source` returns 0. A caller that chains on `||` therefore treats a FAILED
# preflight as a PASS. That is the exact silent-gate failure mode the preflight
# was written to prevent, reintroduced by its own calling convention. Check the
# flag it sets instead, and require that the flag exists at all — an unset flag
# means the checks never ran.
# shellcheck disable=SC1090
source "$PREFLIGHT" --flags "serve --from-uqff --max-seq-len --prefix-cache-n --paged-attn" || true
if [ -z "${_ARC_PF_FAILED+x}" ]; then
    env_fail "sourcing $PREFLIGHT did not set _ARC_PF_FAILED — the checks did not run. \
Refusing to measure behind a gate that never executed."
fi
if [ "$_ARC_PF_FAILED" != "0" ]; then
    env_fail "gpu_box_preflight.sh FAILED; its FAIL[<CHECK>] line above names the reason. \
Most likely the CUDA-compat trap: apt-get install cuda-compat-13-1 and put \
/usr/local/cuda/compat first on LD_LIBRARY_PATH."
fi
say "preflight passed; CUDA_HOME=${CUDA_HOME:-unset} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

# ---------------------------------------------------------------------------
# 2. Build — cuda + flash-attn, NEVER cudnn
# ---------------------------------------------------------------------------
# cudnn costs -62% decode on V4 (5.45 vs 14.58 tok/s, session 4). It is banned
# in CLAUDE.md. Asserted rather than merely omitted, because a stray
# CARGO_BUILD_* or a .cargo/config could reintroduce it.
FEATURES="cuda flash-attn"
case "$FEATURES" in *cudnn*) env_fail "cudnn is in the feature set; banned (-62% decode on V4)";; esac
BIN="$REPO/target/release/mistralrs"
if [ "$ARC_PC_SKIP_BUILD" != "1" ]; then
    step "build: cargo build --release --features \"$FEATURES\" (never cudnn)"
    cargo build --release --features "$FEATURES" -p arc-cli >>"$LOGDIR/prefill_curve.build.log" 2>&1 \
        || env_fail "build failed; see $LOGDIR/prefill_curve.build.log"
fi
[ -x "$BIN" ] || env_fail "$BIN missing or not executable"

# ---------------------------------------------------------------------------
# 3. Model presence + GPU exclusivity
# ---------------------------------------------------------------------------
step "gate 2/3: model + exclusive GPU"
[ -d "$ARC_PC_SRC" ]  || env_fail "source checkpoint $ARC_PC_SRC absent (deepseek-ai/DeepSeek-V4-Flash, 149 GB)"
[ -f "$ARC_PC_UQFF" ] || env_fail "UQFF shard $ARC_PC_UQFF absent (aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b)"

command -v nvidia-smi >/dev/null 2>&1 || env_fail "nvidia-smi absent; Arm B cannot be measured"
# Arm B measures peak DEVICE memory. Another job on the same GPU makes that
# number meaningless — so exclusivity is a gate, not a nicety. Two jobs are
# queued behind this one; they must not overlap it.
OTHER_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [ "${OTHER_PROCS:-0}" != "0" ]; then
    env_fail "$OTHER_PROCS compute process(es) already on the GPU. Arm B needs an idle \
device — peak-memory deltas are not attributable otherwise. Let the queue drain and re-run."
fi
GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
say "GPU total ${GPU_TOTAL_MIB} MiB, idle"

# ---------------------------------------------------------------------------
# 4. Serve
# ---------------------------------------------------------------------------
# Flag choices that are each load-bearing for THIS measurement:
#   --prefix-cache-n 0  a cache hit deletes the prefill we are timing. Every
#                       prompt also carries a unique nonce as a second guard.
#   --paged-attn off    V4's supports_paged_attention() is false, and the paged
#                       pool would eat the VRAM headroom Arm B reads.
#   --max-seqs 1        one request at a time; minimises KV prealloc so the
#                       transient signal is not buried, and leaves the most
#                       headroom for the edge probe.
#   --max-seq-len 16384 must exceed the longest ladder point (8192 + reply).
#   no --mtp-depth      MTP off, so a decode step is a plain decode step and
#                       the max_tokens 5-vs-1 subtraction stays clean.
step "starting server (max-seqs 1, prefix cache off, paged-attn off, MTP off)"
"$BIN" serve -p "$ARC_PC_PORT" \
    -m "$ARC_PC_SRC" -a deepseekv4 \
    --from-uqff "$ARC_PC_UQFF" \
    --chat-template chat_templates/deepseek_v4.json \
    --prefix-cache-n 0 --paged-attn off \
    --max-seqs 1 --max-seq-len 16384 \
    >"$SERVERLOG" 2>&1 &
SERVER_PID=$!

step "waiting for /health (model load is ~10 min)"
HEALTHY=0
for _ in $(seq 1 180); do
    if curl -fsS --max-time 5 "$BASE_URL/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 10
done
[ "$HEALTHY" = "1" ] || env_fail "server never became healthy; tail of $SERVERLOG: $(tail -5 "$SERVERLOG" 2>/dev/null | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# 5. GATE 3 — coherence canary (D18)
# ---------------------------------------------------------------------------
# A degraded box does not report a worse number, it reports a BETTER one: a
# model emitting garbage still emits it fast, and a prefill that silently
# skipped work is a prefill that looks cheap. Timing a box that cannot form a
# sentence measures nothing. 5/6 is the runbook's threshold.
step "gate 3/3: coherence canary (D18 — a broken box reports a BETTER number)"
COH_JSON="$RESDIR/prefill_curve_coherence.json"
if ! BASE_URL="$BASE_URL" python3 arc-tools/quality/run_coherence.py --skip-facts --out "$COH_JSON" \
        >>"$LOGDIR/prefill_curve.coherence.log" 2>&1; then
    env_fail "coherence canary did not complete; see $LOGDIR/prefill_curve.coherence.log"
fi
COH_PASS=$(python3 -c "
import json,sys
d=json.load(open('$COH_JSON'))
def dig(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in ('coherence6_pass','passed','n_pass') and isinstance(v,int): return v
            r=dig(v)
            if r is not None: return r
    if isinstance(o,list):
        n=sum(1 for x in o if isinstance(x,dict) and (x.get('pass') is True or x.get('ok') is True))
        return n if n else None
    return None
v=dig(d); print(v if v is not None else -1)
" 2>/dev/null || echo -1)
say "coherence6 = ${COH_PASS}/6"
if [ "${COH_PASS:--1}" -lt 5 ] 2>/dev/null; then
    env_fail "coherence6 = ${COH_PASS}/6 (< 5). The box is not producing a trustworthy \
model; any timing taken now would be fast for the wrong reason (D18)."
fi

# ---------------------------------------------------------------------------
# 6. Prompt calibration — measure T, never assume it
# ---------------------------------------------------------------------------
# The fit uses the ACTUAL prompt_tokens the server reports, not the requested
# T, so a tokenizer that does not split "the " one-to-one cannot bias the
# exponent. Each prompt carries a unique nonce so no cache can shortcut it.
NONCE_SEQ=0
build_prompt() {  # $1 = target tokens -> stdout
    local want="$1"
    NONCE_SEQ=$((NONCE_SEQ+1))
    python3 - "$want" "$NONCE_SEQ$$" <<'PY'
import sys
want=int(sys.argv[1]); nonce=sys.argv[2]
# ~1 token per word for this filler; verified per-request from usage.prompt_tokens.
words=["the","quick","brown","fox","jumps","over","lazy","dog","and","then"]
out=[f"session{nonce}"]
i=0
while len(out) < want:
    out.append(words[i % len(words)]); i+=1
sys.stdout.write(" ".join(out))
PY
}

# fire_completion <T> <max_tokens> <timeout_s> [extra_curl...]
#   stdout: "<elapsed_s> <prompt_tokens> <completion_tokens> <http_ok>"
fire_completion() {
    local want="$1" maxtok="$2" tmo="$3"
    local prompt body t_start t_end resp
    prompt="$(build_prompt "$want")"
    body=$(python3 -c "
import json,sys
print(json.dumps({'model':'default','prompt':sys.stdin.read(),
                  'max_tokens':int('$maxtok'),'temperature':0.0,'stream':False}))
" <<<"$prompt")
    t_start=$(python3 -c 'import time;print(time.time())')
    resp=$(curl -fsS --max-time "$tmo" -H 'Content-Type: application/json' \
            -d "$body" "$BASE_URL/v1/completions" 2>/dev/null)
    local rc=$?
    t_end=$(python3 -c 'import time;print(time.time())')
    if [ $rc -ne 0 ] || [ -z "$resp" ]; then echo "$(python3 -c "print(f'{$t_end-$t_start:.3f}')") 0 0 0"; return; fi
    python3 -c "
import json,sys
r=json.loads(sys.stdin.read()); u=r.get('usage',{}) or {}
print(f'{$t_end-$t_start:.3f}', u.get('prompt_tokens',0), u.get('completion_tokens',0), 1)
" <<<"$resp"
}

record() { printf '%s\n' "$1" >> "$RAW"; }

in_list() { case " $2 " in *" $1 "*) return 0;; *) return 1;; esac; }

# Decode cost is linear in context length, so fit `d = d0 + d1*T` over the T
# values where it was MEASURED and evaluate there. One point: carry it forward.
extrapolate_decode() {  # $1 = T, $2 = "T:d T:d ..."
    python3 - "$1" "$2" <<'PY'
import sys
T = float(sys.argv[1]); pts = []
for tok in sys.argv[2].split():
    if ':' in tok:
        a, b = tok.split(':', 1)
        try: pts.append((float(a), float(b)))
        except ValueError: pass
if len(pts) >= 2:
    n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if abs(den) > 1e-12:
        d1 = (n * sxy - sx * sy) / den; d0 = (sy - d1 * sx) / n
        print(f'{max(0.0, d0 + d1 * T):.4f}'); sys.exit()
print(f'{pts[-1][1]:.4f}' if pts else '0.0')
PY
}

# ---------------------------------------------------------------------------
# 7. ARM B — the memory law. Runs FIRST: it is cheap, spans 8x, and can kill
#    the mechanism before Arm A spends an hour on it.
# ---------------------------------------------------------------------------
# Trick that makes this cheap: the score matrix is allocated in layer 0, within
# seconds of the request starting. So the PEAK is observable long before the
# request finishes — fire it, sample nvidia-smi at 100 ms for a fixed window,
# then abort. A T=4096 point costs 90 s instead of hours.
step "ARM B: peak transient VRAM vs T (ladder: $ARC_PC_MEM_TS)"
sleep 5
MEM_BASE_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
say "idle baseline after load: ${MEM_BASE_MIB} MiB"

probe_mem() {  # $1 = T ; echoes peak-minus-baseline MiB
    local want="$1" peak=0 cur
    local prompt body
    prompt="$(build_prompt "$want")"
    body=$(python3 -c "
import json,sys
print(json.dumps({'model':'default','prompt':sys.stdin.read(),
                  'max_tokens':1,'temperature':0.0,'stream':False}))
" <<<"$prompt")
    curl -fsS --max-time "$ARC_PC_MEM_WIN" -H 'Content-Type: application/json' \
        -d "$body" "$BASE_URL/v1/completions" >/dev/null 2>&1 &
    local cpid=$!
    local deadline=$(( $(date +%s) + ARC_PC_MEM_WIN ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        kill -0 "$cpid" 2>/dev/null || break
        cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -n "$cur" ] && [ "$cur" -gt "$peak" ] 2>/dev/null && peak="$cur"
        sleep 0.1
    done
    kill "$cpid" 2>/dev/null || true; wait "$cpid" 2>/dev/null || true
    sleep 3
    echo $(( peak - MEM_BASE_MIB ))
}

MEM_RESULTS=""
for T in $ARC_PC_MEM_TS; do
    kill -0 "$SERVER_PID" 2>/dev/null || { say "server died before Arm B T=$T"; break; }
    step "ARM B: T=$T (${ARC_PC_MEM_WIN}s window)"
    d=$(probe_mem "$T")
    pred=$(python3 -c "print(f'{(320*$T*$T + 163840*$T)/1048576:.1f}')")
    say "ARM B T=$T: peak transient ${d} MiB (predicted ${pred} MiB)"
    record "{\"arm\":\"B\",\"T\":$T,\"peak_transient_mib\":$d,\"predicted_mib\":$pred}"
    MEM_RESULTS="$MEM_RESULTS $T:$d"
done

# ---------------------------------------------------------------------------
# 8. ARM A — the timing law. Prefill is ISOLATED, not assumed:
#       t(max_tokens=1) = prefill + 1 decode step
#       t(max_tokens=5) = prefill + 5 decode steps
#    => decode_step = (t5 - t1)/4          [MEASURED at this T]
#       prefill     = t1 - decode_step
#    This is why the constant term `a` and the quadratic term `b` come out
#    separately instead of being confounded, which is exactly the weakness of
#    the two-point fit this run exists to repair.
# ---------------------------------------------------------------------------
step "ARM A: prefill timing vs T (ladder: $ARC_PC_TIMING_TS, budget ${ARC_PC_BUDGET_S}s)"
ARM_A_START=$(date +%s)
TIMING_RESULTS=""
DECODE_POINTS=""
TRUNCATED_AT=""
for T in $ARC_PC_TIMING_TS; do
    kill -0 "$SERVER_PID" 2>/dev/null || { say "server died before Arm A T=$T"; TRUNCATED_AT="$T"; break; }
    spent=$(( $(date +%s) - ARM_A_START ))
    left=$(( ARC_PC_BUDGET_S - spent ))
    if [ "$left" -lt 120 ]; then
        say "ARM A: budget exhausted before T=$T (${spent}s spent). Reporting completed points."
        TRUNCATED_AT="$T"; break
    fi
    step "ARM A: T=$T (max_tokens=1), ${left}s of budget left"
    read -r t1 pt1 _ ok1 <<<"$(fire_completion "$T" 1 "$(( left < ARC_PC_REQ_TMO ? left : ARC_PC_REQ_TMO ))")"
    if [ "$ok1" != "1" ]; then
        say "ARM A: T=$T max_tokens=1 did not return within budget/timeout (t=${t1}s)."
        record "{\"arm\":\"A\",\"T\":$T,\"status\":\"timeout\",\"lower_bound_s\":$t1}"
        TRUNCATED_AT="$T"; break
    fi
    if in_list "$T" "$ARC_PC_SUBTRACT_TS"; then
        spent=$(( $(date +%s) - ARM_A_START )); left=$(( ARC_PC_BUDGET_S - spent ))
        step "ARM A: T=$T (max_tokens=5, measuring the decode step), ${left}s of budget left"
        read -r t5 pt5 _ ok5 <<<"$(fire_completion "$T" 5 "$(( left < ARC_PC_REQ_TMO ? left : ARC_PC_REQ_TMO ))")"
        if [ "$ok5" != "1" ]; then
            say "ARM A: T=$T max_tokens=5 timed out; decode cannot be measured at this T."
            record "{\"arm\":\"A\",\"T\":$T,\"actual_tokens\":$pt1,\"t1_s\":$t1,\"status\":\"no_decode_subtraction\"}"
            TRUNCATED_AT="$T"; break
        fi
        dstep=$(python3 -c "print(f'{($t5-$t1)/4.0:.4f}')")
        DECODE_POINTS="$DECODE_POINTS $pt1:$dstep"
        dsrc="measured"; t5_field="$t5"
    else
        dstep=$(extrapolate_decode "$T" "$DECODE_POINTS")
        dsrc="extrapolated"; t5_field="null"
    fi
    read -r prefill dshare <<<"$(python3 -c "
p=$t1-$dstep
print(f'{p:.4f}', f'{($dstep/$t1 if $t1>0 else 0):.5f}')")"
    say "ARM A T=$T (actual ${pt1} tok): t1=${t1}s decode_step=${dstep}s ($dsrc, ${dshare} of t1) -> prefill=${prefill}s"
    record "{\"arm\":\"A\",\"T\":$T,\"actual_tokens\":$pt1,\"t1_s\":$t1,\"t5_s\":$t5_field,\"decode_step_s\":$dstep,\"decode_source\":\"$dsrc\",\"decode_share_of_t1\":$dshare,\"prefill_s\":$prefill}"
    TIMING_RESULTS="$TIMING_RESULTS $pt1:$prefill"
done

# ---------------------------------------------------------------------------
# 9. ARM C — mechanism discriminator (optional)
# ---------------------------------------------------------------------------
# ARC_V4_WINDOW_ONLY=1 drops the compressed branch, taking n_keys from 1.25T to
# T. Score-matrix-bound prefill falls to ~1/1.25^2 = 0.64 of baseline at fixed
# T; MoE/dispatch-bound prefill barely moves. Needs a server restart, so it is
# opt-in and runs last among the timed arms.
DISCRIM_RATIO="null"
if [ "$ARC_PC_DISCRIM" = "1" ] && [ -n "$TIMING_RESULTS" ]; then
    step "ARM C: ARC_V4_WINDOW_ONLY discriminator at T=256"
    read -r base_t256 _ _ ok <<<"$(fire_completion 256 1 "$ARC_PC_REQ_TMO")"
    kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; sleep 15
    ARC_V4_WINDOW_ONLY=1 "$BIN" serve -p "$ARC_PC_PORT" -m "$ARC_PC_SRC" -a deepseekv4 \
        --from-uqff "$ARC_PC_UQFF" --chat-template chat_templates/deepseek_v4.json \
        --prefix-cache-n 0 --paged-attn off --max-seqs 1 --max-seq-len 16384 \
        >>"$SERVERLOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 180); do curl -fsS --max-time 5 "$BASE_URL/health" >/dev/null 2>&1 && break; sleep 10; done
    read -r wo_t256 _ _ ok2 <<<"$(fire_completion 256 1 "$ARC_PC_REQ_TMO")"
    if [ "$ok" = "1" ] && [ "$ok2" = "1" ]; then
        DISCRIM_RATIO=$(python3 -c "print(f'{$wo_t256/$base_t256:.4f}')")
        say "ARM C: window-only/full = $DISCRIM_RATIO (score-matrix-bound predicts ~0.64; MoE-bound ~1.0)"
        record "{\"arm\":\"C\",\"T\":256,\"full_s\":$base_t256,\"window_only_s\":$wo_t256,\"ratio\":$DISCRIM_RATIO}"
    fi
fi

# ---------------------------------------------------------------------------
# 10. EDGE probe — test the OOM prediction for real, LAST, after every other
#     number is banked, because it may take the server down.
# ---------------------------------------------------------------------------
EDGE_RESULT="null"
if [ "$ARC_PC_EDGE" = "1" ]; then
    step "EDGE: T=8192 (predicted ~21760 MiB transient — may OOM; all other numbers are already banked)"
    d=$(probe_mem 8192 2>/dev/null || echo -1)
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        EDGE_RESULT="{\"T\":8192,\"peak_transient_mib\":$d,\"server_survived\":true}"
        say "EDGE: survived, peak transient ${d} MiB (predicted 21760 MiB)"
    else
        EDGE_RESULT="{\"T\":8192,\"peak_transient_mib\":$d,\"server_survived\":false}"
        say "EDGE: server DIED at T=8192 — this is the session-8 death reproduced under control"
    fi
    record "{\"arm\":\"EDGE\",$(echo "$EDGE_RESULT" | sed 's/^{//')"
fi

# ---------------------------------------------------------------------------
# 11. Fit, verdict against the PRE-REGISTERED criteria, and emit
# ---------------------------------------------------------------------------
write_json() {  # $1 = verdict, $2 = note
    python3 - "$1" "$2" "$OUT" "$RAW" <<'PY' 2>/dev/null || true
import json,sys,os
verdict,note,out,raw=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
rows=[]
if os.path.exists(raw):
    for ln in open(raw):
        ln=ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except Exception: pass
json.dump({"verdict":verdict,"note":note,"rows":rows},open(out,"w"),indent=2)
PY
}

step "fitting and scoring against the pre-registered criteria"
python3 - "$TIMING_RESULTS" "$MEM_RESULTS" "$DISCRIM_RATIO" "$OUT" "$RAW" "${TRUNCATED_AT:-}" <<'PY'
import json, sys, os

timing_s, mem_s, discrim, out, raw, truncated = sys.argv[1:7]

def parse(s):
    pts=[]
    for tok in s.split():
        if ':' in tok:
            a,b=tok.split(':',1)
            try: pts.append((float(a), float(b)))
            except ValueError: pass
    return sorted(pts)

timing, mem = parse(timing_s), parse(mem_s)
rows=[]
if os.path.exists(raw):
    for ln in open(raw):
        ln=ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except Exception: pass

res = {
    "question": "Is V4 prefill quadratic in prompt length, as PR #97's two-point fit claimed?",
    "pre_registered": {
        "timing_refuted_if": "prefill(1024)/prefill(512) < 2.8, or quadratic share at T=1024 < 50%",
        "memory_refuted_if": "any adjacent peak ratio < 2.2, or peak(4096) < 2048 MiB",
        "discriminator_refuted_if": "ARC_V4_WINDOW_ONLY ratio > 0.90",
        "note": "peak is ONE layer's working set; layers are sequential and free before the next allocates. A 43x criterion would reject a correct mechanism.",
    },
    "timing_points": [{"tokens": t, "prefill_s": v} for t, v in timing],
    "memory_points": [{"tokens": t, "peak_transient_mib": v} for t, v in mem],
    "rows": rows,
}
if truncated:
    res["truncated_at_T"] = int(truncated)
    res["truncation_note"] = ("the ladder did not complete; points below are still valid, "
                              "and a timeout is a LOWER BOUND on prefill at that T, not a refutation")

# ---- timing: least-squares a2*T^2 + a1*T (no intercept: prefill(0)=0) -------
verdict_t = "insufficient"
if len(timing) >= 3:
    s4=s3=s2=sy2=sy1=0.0
    for t,y in timing:
        s4+=t**4; s3+=t**3; s2+=t**2; sy2+=y*t*t; sy1+=y*t
    det = s4*s2 - s3*s3
    if abs(det) > 1e-9:
        a2 = ( sy2*s2 - sy1*s3) / det
        a1 = ( sy1*s4 - sy2*s3) / det
        q1024 = a2*1024**2; l1024 = a1*1024
        share = q1024/(q1024+l1024) if (q1024+l1024) > 0 else 0.0
        res["fit"] = {"a2_s_per_tok2": a2, "a1_s_per_tok": a1,
                      "quadratic_share_at_1024": share,
                      "predicted_prefill_s_at_1400": a2*1400**2 + a1*1400}
        near = lambda x: min(timing, key=lambda p: abs(p[0]-x))
        lo, hi = near(512), near(1024)
        ratio = hi[1]/lo[1] if lo[1] > 0 else 0.0
        res["timing_ratio_hi_lo"] = {"lo_tokens": lo[0], "hi_tokens": hi[0], "ratio": ratio,
                                     "quadratic_predicts": (hi[0]/lo[0])**2,
                                     "linear_predicts": hi[0]/lo[0]}
        verdict_t = "supported" if (ratio >= 2.8 and share >= 0.5) else "refuted"
elif len(timing) == 2:
    verdict_t = "insufficient"   # exactly the weakness this run exists to repair
res["timing_verdict"] = verdict_t

# ---- memory ---------------------------------------------------------------
verdict_m = "insufficient"
if len(mem) >= 2:
    ratios=[]
    for (t0,v0),(t1,v1) in zip(mem, mem[1:]):
        ratios.append({"from": t0, "to": t1, "ratio": (v1/v0) if v0 > 0 else None,
                       "quadratic_predicts": (t1/t0)**2, "linear_predicts": t1/t0})
    res["memory_ratios"] = ratios
    vals=[r["ratio"] for r in ratios if r["ratio"] is not None]
    big = [v for t,v in mem if t >= 4000]
    ok = bool(vals) and min(vals) >= 2.2 and (not big or max(big) >= 2048)
    verdict_m = "supported" if ok else "refuted"
res["memory_verdict"] = verdict_m

# ---- discriminator --------------------------------------------------------
if discrim not in ("null", ""):
    d=float(discrim)
    res["discriminator"] = {"window_only_over_full": d,
                            "verdict": "supported" if d <= 0.90 else "refuted"}

# ---- overall --------------------------------------------------------------
vs = [v for v in (verdict_t, verdict_m) if v != "insufficient"]
if not vs:
    overall = "insufficient"
elif all(v == "supported" for v in vs):
    overall = "supported"
elif all(v == "refuted" for v in vs):
    overall = "refuted"
else:
    overall = "mixed"
res["verdict"] = overall
res["what_this_means"] = {
    "supported": "task #21 (fused head_dim-512 kernel) is the lever for long prompts; the rerank stands.",
    "refuted":   "prefill is NOT quadratic. The fused attention kernel does not fix long prompts, and six workstreams were re-ranked on a bad curve. Re-plan before spending more.",
    "mixed":     "the timing and memory laws disagree. Some other quadratic may be at work; #21 is NOT established as the lever. Do not re-rank further on this.",
    "insufficient": "the ladder did not produce enough points. This is NOT a refutation — nothing was learned about the mechanism.",
}[overall]

json.dump(res, open(out, "w"), indent=2)

lin = lambda k: res.get(k, {})
print("SUMMARY " + json.dumps({
    "verdict": overall,
    "timing": verdict_t, "memory": verdict_m,
    "timing_points": [[t, round(v,1)] for t,v in timing],
    "memory_points_mib": [[t, v] for t,v in mem],
    "ratio_1024_over_512": round(lin("timing_ratio_hi_lo").get("ratio", 0.0), 3) if "timing_ratio_hi_lo" in res else None,
    "predicted_prefill_s_at_1400": round(res["fit"]["predicted_prefill_s_at_1400"], 1) if "fit" in res else None,
}))
PY

SUMMARY_LINE=$(python3 -c "
import json
d=json.load(open('$OUT'))
t=[f\"{int(p['tokens'])}:{p['prefill_s']:.0f}s\" for p in d.get('timing_points',[])]
m=[f\"{int(p['tokens'])}:{p['peak_transient_mib']}MiB\" for p in d.get('memory_points',[])]
r=d.get('timing_ratio_hi_lo',{}).get('ratio')
print(f\"V4 PREFILL CURVE: verdict={d['verdict']} (timing={d.get('timing_verdict')}, memory={d.get('memory_verdict')}) \"
      f\"prefill[{' '.join(t)}] peak[{' '.join(m)}] ratio_1024/512={r if r is None else round(r,2)} \"
      f\"(quadratic predicts 4.0, linear 2.0) -> {d.get('what_this_means','')}\")
" 2>/dev/null || echo "V4 PREFILL CURVE: could not summarise; see $OUT")

say "$SUMMARY_LINE"
stat_ "$SUMMARY_LINE"
echo "$SUMMARY_LINE" > "$RESDIR/v4_prefill_curve.summary.txt"
exit 0
