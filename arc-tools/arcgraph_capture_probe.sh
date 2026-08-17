#!/bin/bash
# ArcGraph capture probe — is the V4 decode forward dispatch-bound or kernel-bound?
#
# WHY THIS EXISTS
# ---------------
# FACTS.md has no CUDA-graph entry at all. Nobody has ever measured whether
# Arc's decode forward is dominated by kernel launch dispatch (which CUDA graphs
# remove) or by the kernels themselves (which they cannot touch). That single
# ratio decides whether the entire 3-tier plan in project_cuda_graph_plan.md
# (capture-once -> fused sampling -> GPU-side WHILE loop at 0 us/step) is worth
# building.
#
# WHAT IT DOES *NOT* MEASURE, AND WHY
# -----------------------------------
# It does NOT produce a "tok/s with graphs vs without". That number cannot
# honestly exist today: normal.rs discards the replayed logits and the eager
# forward supplies the real ones (replay correctness needs static input buffers
# + a device-indexed KV write/read, neither of which is implemented). No token
# Arc has ever emitted came from a graph replay. Any tok/s delta attributed to
# graphs would be fabricated. See PROFILING.md and wave52-CC.
#
# THE NUMBERS TO EXTRACT (in priority order)
# ------------------------------------------
#   N1  ARC capture: EAGER forward (sync'd) = <T_eager>
#   N2  ARC capture: CLEAN graph-launch best-of-10 = <T_graph>
#       => VERDICT. T_graph / T_eager near 1.0 = kernel-bound, graphs cannot
#          help, tier 2/3 of the plan is dead. Much less than 1.0 = dispatch-
#          bound, graphs attack the dominant cost, plan is live.
#   N3  ARCGRAPH STATUS: capture_possible=? captured=? replayed=?
#       => PROOF capture happened. captured>=1 is the only acceptable evidence.
#          "runner initialized" is NOT evidence (D18).
#   N4  leg A baseline decode tok/s at b=1 (steady-state, includes sampling)
#   N5  THE FWD/HOST SPLIT, which V4 has never had. FACTS.md:588 records that
#       the fwd/host split log exists only on the PagedAttention branch, so
#       "V4's actual path logs no fwd/host split at all." N1 (the sync'd eager
#       forward) against 1000/N4 ms (the full step) gives it directly:
#           forward_fraction = T_eager / (1000 / tok_s)
#       This matters MORE than the ratio. FACTS records eight host costs
#       (engine/mod.rs, sampling.rs, utils/mod.rs, sequence.rs) that sit
#       OUTSIDE the forward and that CUDA graphs cannot touch. If the forward
#       is a small fraction of the step, then even a total win on N2/N1 moves
#       end-to-end throughput very little, and graphs are the wrong lever
#       regardless of how good the ratio looks.
#
# METHOD NOTE, STATED BEFORE THE RESULT
# -------------------------------------
# N1 is a SINGLE sync'd sample; N2 is a best-of-10. That asymmetry biases the
# comparison IN FAVOUR of graphs. So a "kernel-bound" verdict (N2 ~ N1) is
# robust a fortiori. A "dispatch-bound" verdict (N2 << N1) must be treated as an
# upper bound on the win and re-measured before anything is built on it.
#
# EXIT CODES
#   0  ran, numbers in the status file
#   2  the box could not answer (gate failed, model missing, build failed)
set -u

LOCKFILE=""
LOGDIR="${LOGDIR:-/root/logs/arcgraph}"
REPO="${REPO:-/root/arc}"
BRANCH="${BRANCH:-arcgraph/capture-truth}"
# Private worktree so we never disturb the shared checkout at $REPO.
WORKTREE="${WORKTREE:-/root/arc-arcgraph}"
SRC="${SRC:-/root/models/v4-src}"
UQFF="${UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}"
PORT="${PORT:-1234}"
BASE_URL="http://localhost:${PORT}"
# 24 tokens is enough: 4 eager warmups + 1 deferred-free pass + capture on step 6,
# leaving replays to observe. More just costs box time.
GEN_TOKENS="${GEN_TOKENS:-24}"

mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.txt"
: > "$STATUS"

say()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }
step() { say "=== $* ==="; }
env_fail() { say "ENV_FAIL: $*"; say "RESULT: UNANSWERED"; exit 2; }

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        say "releasing the box: killing server pid $SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SERVER_PID" 2>/dev/null
    fi
    pkill -f "mistralrs serve -p ${PORT}" 2>/dev/null
    # Release the GPU lock only if it is ours — never clear another chain's.
    if [ -s "${LOCKFILE:-/nonexistent}" ] && grep -q "pid=$$ " "$LOCKFILE" 2>/dev/null; then
        rm -f "$LOCKFILE"
        say "GPU lock released"
    fi
    return 0
}
trap cleanup EXIT INT TERM

[ -d "$REPO" ] || env_fail "repo $REPO absent"
cd "$REPO" || env_fail "cannot enter $REPO"

# --------------------------------------------------------------------------
# 0. Environment — MANDATORY. Driver caps at CUDA 13.0, only toolkit is 13.1.
#    Without the compat libs PTX-JIT fails in a way that yields plausible wrong
#    numbers rather than a crash.
# --------------------------------------------------------------------------
# shellcheck disable=SC1091
[ -f /root/arcenv.sh ] && . /root/arcenv.sh
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
say "CUDA_HOME=$CUDA_HOME LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# --------------------------------------------------------------------------
# 1. HARD GATE — box preflight.
#    `source preflight || handler` DOES NOT WORK (it returns 0 on a failing
#    box; see measure_v4_prefill_curve.sh:212). Check the flag it sets, and
#    require the flag to exist at all — unset means the checks never ran.
# --------------------------------------------------------------------------
step "gate 1/3: preflight"
PREFLIGHT="/root/arc-tools/gpu_box_preflight.sh"
[ -f "$PREFLIGHT" ] || PREFLIGHT="$REPO/arc-tools/gpu_box_preflight.sh"
[ -f "$PREFLIGHT" ] || env_fail "gpu_box_preflight.sh absent; refusing to measure without the PTX gate"
# shellcheck disable=SC1090
source "$PREFLIGHT" --flags "serve --from-uqff --max-seq-len --prefix-cache-n --paged-attn" || true
if [ -z "${_ARC_PF_FAILED+x}" ]; then
    env_fail "preflight did not set _ARC_PF_FAILED — the checks did not run"
fi
[ "$_ARC_PF_FAILED" = "0" ] || env_fail "preflight FAILED; see its FAIL[<CHECK>] line above"
say "preflight passed"

# --------------------------------------------------------------------------
# 2. Build the instrumented branch.
#    NOTE FOR THE OPERATOR: a compile error here is a LEGITIMATE result to
#    report back, not a script bug. The changed code is CUDA-gated and macOS
#    `cargo check` does not type-check CUDA-gated Rust — this is its first
#    real type-check. If it fails, send back the first error block verbatim.
#    NEVER cudnn (-62% decode on V4, banned in CLAUDE.md).
# --------------------------------------------------------------------------
step "gate 2/3: build $BRANCH"
# THE BOX IS SHARED WITH THREE OTHER CHAINS. Do NOT check out into /root/arc —
# it currently sits on another chain's detached HEAD with uncommitted edits, and
# a checkout there would silently destroy their working state. Use a private
# worktree instead. CARGO_TARGET_DIR is shared with /root/arc on purpose: cargo
# locks it (so concurrent builds serialise rather than corrupt) and dependency
# artifacts — which are almost all of the build time — are reused, turning a
# ~11 min cold build into an incremental one.
git -C "$REPO" fetch -q origin "$BRANCH" >>"$LOGDIR/build.log" 2>&1 \
    || env_fail "git fetch $BRANCH failed"
if [ ! -d "$WORKTREE/.git" ] && [ ! -f "$WORKTREE/.git" ]; then
    git -C "$REPO" worktree add --detach "$WORKTREE" FETCH_HEAD >>"$LOGDIR/build.log" 2>&1 \
        || env_fail "could not create worktree at $WORKTREE"
else
    git -C "$WORKTREE" checkout -q --detach FETCH_HEAD >>"$LOGDIR/build.log" 2>&1 \
        || env_fail "could not update worktree at $WORKTREE"
fi
cd "$WORKTREE" || env_fail "worktree $WORKTREE unusable"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target}"
say "worktree = $WORKTREE (host repo $REPO left untouched)"
say "HEAD = $(git log --oneline -1)"
say "CARGO_TARGET_DIR = $CARGO_TARGET_DIR"
FEATURES="cuda flash-attn"
case "$FEATURES" in *cudnn*) env_fail "cudnn present; banned";; esac

# 🔴 THE BUG THIS BLOCK EXISTS TO PREVENT, found the hard way on 2026-08-17.
# The idiom copied from measure_v4_prefill_curve.sh:241-244 builds `-p arc-cli`
# and then runs `$TARGET/release/mistralrs`. Those are DIFFERENT BINARIES:
# arc-cli's bin is named `arc` (arc-cli/Cargo.toml), and `mistralrs` comes from
# mistralrs-cli, which arc-cli does not depend on. So the build succeeded, the
# `[ -x "$BIN" ]` existence check passed against an hour-old binary from another
# chain's commit, and the run measured code that was never built. The server
# even logged `git revision: ab42c4508` while the checkout was 7fbdfcfdb — the
# evidence was right there and the script did not look.
#
# So: build the package that actually produces the binary, take the path from
# CARGO'S OWN artifact message rather than guessing it, and refuse unless the
# binary contains a string that only the code under test has. Existence is not
# freshness.
PKG="${PKG:-mistralrs-cli}"
BIN_NAME="${BIN_NAME:-mistralrs}"
# A string present only in the instrumented code. If the binary lacks it, we
# are about to measure something else.
FRESHNESS_MARKER="${FRESHNESS_MARKER:-ARCGRAPH STATUS}"
# shellcheck disable=SC1090,SC1091
source "$WORKTREE/arc-tools/lib/build_and_verify.sh" \
    || env_fail "arc-tools/lib/build_and_verify.sh missing from the checkout"
arc_build_and_verify \
    --package "$PKG" --bin "$BIN_NAME" --features "$FEATURES" \
    --marker "$FRESHNESS_MARKER" --log "$LOGDIR/build.log" \
    || env_fail "build/verify failed; see $LOGDIR/build.log (diagnosis above)"
BIN="$ARC_VERIFIED_BIN"
say "build ok — BIN=$BIN (cargo-reported), freshness marker present"

# --------------------------------------------------------------------------
# 3. Model + GPU
# --------------------------------------------------------------------------
step "gate 3/3: model + GPU"
[ -d "$SRC" ]  || env_fail "source checkpoint $SRC absent"
[ -f "$UQFF" ] || env_fail "UQFF shard $UQFF absent"

# The box is shared by four chains. `nvidia-smi` alone is NOT enough: a chain
# between two runs holds no compute process but is still mid-measurement, and
# starting under it corrupts both its numbers and ours. Honour the advisory
# lock first, then check the device, then take the lock so the next chain sees
# us. A held lock is a refusal, not a wait — the caller decides when to retry.
LOCKDIR="${LOCKDIR:-/root/locks}"
LOCKFILE="$LOCKDIR/gpu.lock"
mkdir -p "$LOCKDIR"
if [ -s "$LOCKFILE" ]; then
    HOLDER=$(head -1 "$LOCKFILE")
    HOLDER_PID=$(sed -n 's/.*pid=\([0-9]*\).*/\1/p' <<<"$HOLDER")
    if [ -n "$HOLDER_PID" ] && kill -0 "$HOLDER_PID" 2>/dev/null; then
        env_fail "GPU lock held: $HOLDER (pid alive). Not starting — overlapping runs give both \
chains garbage. Retry when $LOCKFILE clears."
    fi
    say "stale lock from a dead holder ($HOLDER); reclaiming"
fi
OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
[ "${OTHER:-0}" = "0" ] || env_fail "$OTHER compute process(es) already on the GPU; latency numbers are not attributable. Let the queue drain."
echo "arcgraph pid=$$ since=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCKFILE"
say "GPU idle; lock taken ($(cat "$LOCKFILE"))"

# --------------------------------------------------------------------------
# 4. The two legs.
# --------------------------------------------------------------------------
# Env vars are read at process start, so each leg needs its own server.
#   --paged-attn off   V4's supports_paged_attention() is false, AND
#                      ARC_V4_PAGED_ATTN=1 would SHADOW the only capturable KV
#                      write arm V4 has (wave53-CD §5a). Capture requires it off.
#   --prefix-cache-n 0 a cache hit deletes the decode steps we are timing.
#   --max-seqs 1       b=1. The probe path is `seq_len == 1` only.
#   no --mtp-depth     a decode step must be a plain decode step.
run_leg() {
    local name="$1"; shift
    local slog="$LOGDIR/${name}.server.log"
    step "leg $name: starting server"
    say "leg $name env: ${LEG_ENV:-<none>}"
    # shellcheck disable=SC2086
    env $LEG_ENV "$BIN" serve -p "$PORT" \
        -m "$SRC" -a deepseekv4 \
        --from-uqff "$UQFF" \
        --chat-template chat_templates/deepseek_v4.json \
        --prefix-cache-n 0 --paged-attn off \
        --max-seqs 1 --max-seq-len 4096 \
        >"$slog" 2>&1 &
    SERVER_PID=$!

    local healthy=0
    for _ in $(seq 1 60); do
        curl -fsS --max-time 5 "$BASE_URL/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 10
    done
    if [ "$healthy" != "1" ]; then
        say "leg $name: server never became healthy. tail:"
        tail -20 "$slog" | tee -a "$STATUS"
        cleanup; SERVER_PID=""
        return 1
    fi
    # PROVENANCE (step 4). Engagement proved the feature is compiled in; this
    # proves the RUNNING process is the commit we built. Different guarantees.
    if ! arc_assert_running_revision --log "$slog" --timeout-s 60; then
        say "leg $name: PROVENANCE ASSERTION FAILED — voiding this leg rather than \
banking a number of unknown origin (see stderr in $LOGDIR/run.log)"
        cleanup; SERVER_PID=""
        return 1
    fi
    say "leg $name: provenance ok (running rev $ARC_RUNNING_REV)"
    say "leg $name: healthy, generating $GEN_TOKENS tokens at b=1"

    local t0 t1 resp ntok
    t0=$(date +%s.%N)
    resp=$(curl -s --max-time 900 "$BASE_URL/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"default\",\"prompt\":\"Count from one to twenty in words, then stop.\",\"max_tokens\":$GEN_TOKENS,\"temperature\":0}")
    t1=$(date +%s.%N)
    echo "$resp" > "$LOGDIR/${name}.response.json"
    ntok=$(python3 -c "import json,sys;d=json.load(open('$LOGDIR/${name}.response.json'));print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
    say "leg $name: completion_tokens=$ntok wall=$(python3 -c "print(f'{$t1-$t0:.2f}')")s tok_s=$(python3 -c "
w=$t1-$t0
n=$ntok
print(f'{n/w:.2f}' if w>0 and n>0 else 'n/a')")"

    cleanup; SERVER_PID=""
    sleep 5
    return 0
}

# LEG A — production default. Establishes the inert baseline and the tok/s
# denominator. Expect: ArcGraph INERT + capture_possible=false captured=0.
LEG_ENV="ARC_NO_DEDICATED_DECODE=0"
run_leg baseline || say "leg baseline did not complete"

# LEG B — all three capture gates on, plus the profiler's three time channels.
# ARC_NO_DEDICATED_DECODE is left OFF deliberately so the architecture guard's
# refusal line is exercised on the real V4 model.
LEG_ENV="ARC_CAPTURE_STREAM=1 ARC_V4_CAPTURE_PROBE=1 ARC_CANDLE_ALLOC_CACHE=1 ARC_PROFILE=1 ARC_PROFILE_STEPS=12 ARC_PROFILE_OUT=$LOGDIR/profile ARC_PROFILE_LABEL=arcgraph_capture"
run_leg capture || say "leg capture did not complete"

# --------------------------------------------------------------------------
# 5. Extract. Every line below is the answer to a specific question.
# --------------------------------------------------------------------------
step "EXTRACTED NUMBERS"
{
  echo "---------------- N3: did capture actually happen? (D18 proof) ----------------"
  grep -h "ARCGRAPH STATUS" "$LOGDIR"/*.server.log 2>/dev/null | sort -u || echo "  (no ARCGRAPH STATUS line — binary predates the instrumentation?)"
  echo
  echo "---------------- stream / gate state ----------------"
  grep -hE "ArcGraph INERT|non-null stream|runner initialized|capture enabled" "$LOGDIR"/*.server.log 2>/dev/null | sort -u
  echo
  echo "---------------- N1/N2: THE VERDICT NUMBERS ----------------"
  grep -hE "EAGER forward \(sync'd\)|CLEAN graph-launch best-of-10" "$LOGDIR"/*.server.log 2>/dev/null
  echo
  echo "---------------- capture chain, step by step ----------------"
  grep -hE "warmup done|deferred-free warmup pass done|forward RECORDED|graph CAPTURED|captured \+ launched|begin_capture failed|forward errored DURING capture|instantiate/launch failed|replay failed|REPLAY latency" "$LOGDIR"/*.server.log 2>/dev/null | head -40
  echo
  echo "---------------- architecture guard (the other half of the PR) ----------------"
  grep -hE "Dedicated decode path declined|Decode path extraction FAILED|Decode path: .* layers extracted" "$LOGDIR"/*.server.log 2>/dev/null | sort -u
  echo
  echo "---------------- profiler artifacts ----------------"
  ls -la "$LOGDIR/profile" 2>/dev/null || echo "  (no profile written — ARC_PROFILE_STEPS may not have been reached)"
} | tee -a "$STATUS"

say "RESULT: COMPLETE — full status in $STATUS, server logs in $LOGDIR/*.server.log"
