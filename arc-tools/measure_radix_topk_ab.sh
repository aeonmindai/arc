#!/usr/bin/env bash
#
# Same-box A/B for the GPU radix top-k sampler dtype fix (PR: fix/radix-topk-i32-dtype).
#
# WHAT THIS MEASURES, AND WHY IT IS NOT THE OBVIOUS SWEEP
# -------------------------------------------------------
# The bug: `arc-cuda-graph/src/weights.rs::tensor_device_ptr` had no `DType::I32`
# arm, so `radix_topk_rows_f32`'s I32 `seq_lens` buffer could never be turned
# into a device pointer. Every call bailed and the sampler fell back to the CPU.
#
# BUT the big-vocab GPU radix path is only reached when
# (mistralrs-core/src/sampler.rs, `sample`):
#     temperature > 1e-7            (else -> argmax, a different path entirely)
#   AND (top_k > 0 OR 0 < top_p < 1) (else -> `sample_fast`, no sort needed)
#   AND vocab too large for candle's shared-memory arg_sort (V4: 129,280 -> yes)
#
# The published 111.69 tok/s @ B=256 baseline was measured with
# `batch_load_probe.py` at its defaults: temperature 0.0 and a HARDCODED
# top_p 1.0. Both of those bypass the broken code. So that number CANNOT move,
# and this script does not pretend otherwise — it measures the greedy config as
# an explicit CONTROL (expected: unchanged) and the sampled config
# (temperature 1.0 / top_p 0.95, the "measured-good chat.py setting" and the
# one the live box was running when it logged the warning ~10x/s) as the
# TREATMENT.
#
# Three measurements per phase:
#   1. sweep_sampled  — B sweep at temp 1.0 / top_p 0.95   (treatment)
#   2. sweep_greedy   — B sweep at temp 0.0 / top_p 1.0    (control)
#   3. promptladder   — 8 tokens at ~9 / ~200 / ~1055 words (task #22: the
#                       "100 s for 8 tokens, dead at 1055 words" curve)
# plus the server-log fallback counter, which is the direct proof:
#   BEFORE must print "SAMPLER ON CPU SLOW PATH"; AFTER must not print it once.
#
# Phases: AFTER (this branch as-is) then BEFORE (only
# `arc-cuda-graph/src/weights.rs` reverted to origin/master, incremental
# rebuild). Reverting that one file restores exactly the pre-fix helper while
# leaving the new counter/logging in place, so BOTH phases report the counter.
#
# USAGE (background, survives disconnect):
#   setsid nohup /root/arc/arc-tools/measure_radix_topk_ab.sh \
#     < /dev/null > /root/radix_ab.log 2>&1 &
#   # progress:  cat /root/radix_ab.status
#   # results:   cat /root/radix_ab/RESULTS.md
#
# ENV OVERRIDES (defaults match the standard V4 box layout):
#   ARC_DIR   REPO checkout                 (/root/arc)
#   V4_SRC    source checkpoint for -m      (/root/models/v4-src)
#   V4_UQFF   FIRST uqff shard file         (/root/models/v4-uqff/qtip2b-0.uqff)
#   OUT_DIR   results directory             (/root/radix_ab)
#   PORT      server port                   (1234)

set -uo pipefail

ARC_DIR="${ARC_DIR:-/root/arc}"
V4_SRC="${V4_SRC:-/root/models/v4-src}"
V4_UQFF="${V4_UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}"
OUT_DIR="${OUT_DIR:-/root/radix_ab}"
PORT="${PORT:-1234}"
STATUS="/root/radix_ab.status"

# Mandatory box environment (driver caps at CUDA 13.0, only toolkit is 13.1 —
# PTX-JIT fails without the compat libs on LD_LIBRARY_PATH).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT_DIR"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }
die() { say "FAILED: $*"; echo "STATUS=FAILED" >> "$STATUS"; exit 1; }

echo "STATUS=RUNNING pid=$$" > "$STATUS"
say "A/B start. ARC_DIR=$ARC_DIR OUT_DIR=$OUT_DIR"

# ---------------------------------------------------------------- preflight --
# Refuse to proceed on an unverified box rather than re-deriving the checks.
if [[ -x "$ARC_DIR/arc-tools/gpu_box_preflight.sh" ]]; then
    say "running gpu_box_preflight.sh"
    "$ARC_DIR/arc-tools/gpu_box_preflight.sh" 2>&1 | tee "$OUT_DIR/preflight.log" \
        || die "gpu_box_preflight.sh did not pass — refusing to measure"
else
    die "no $ARC_DIR/arc-tools/gpu_box_preflight.sh — refusing to proceed (D14/D15: \
the preflight is the gate; do not re-derive it)"
fi

cd "$ARC_DIR" || die "cannot cd $ARC_DIR"
[[ -f "$V4_UQFF" ]] || die "UQFF shard not found: $V4_UQFF (must be the FIRST shard FILE, not a dir)"
[[ -d "$V4_SRC"  ]] || die "source checkpoint not found: $V4_SRC"

# ------------------------------------------------------------- server helpers --
SERVER_PID=""
start_server() {  # $1 = phase name
    local phase="$1" log="$OUT_DIR/server_$1.log"
    say "[$phase] starting server"
    setsid nohup ./target/release/mistralrs serve -p "$PORT" \
        -m "$V4_SRC" \
        -a deepseekv4 \
        --from-uqff "$V4_UQFF" \
        --chat-template chat_templates/deepseek_v4.json \
        --max-seqs 256 \
        --prefix-cache-n 0 \
        --max-seq-len 4096 --max-batch-size 128 \
        < /dev/null > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 180); do
        sleep 10
        if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
            say "[$phase] server up (pid $SERVER_PID)"
            # `--from-uqff` must NOT re-quantize; if it does the numbers are
            # not measuring what we think they are.
            if grep -q "Applying ISQ" "$log"; then
                die "[$phase] 'Applying ISQ' in server log — the UQFF overlay did not take"
            fi
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -40 "$log"; die "[$phase] server died during load"; }
    done
    die "[$phase] server did not come up within 30 min"
}
stop_server() {
    [[ -n "$SERVER_PID" ]] || return 0
    kill "$SERVER_PID" 2>/dev/null
    sleep 15
    kill -9 "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}
trap stop_server EXIT

# ------------------------------------------------------------ one full phase --
run_phase() {  # $1 = phase name (after|before)
    local phase="$1" q="$ARC_DIR/arc-tools/quality"
    start_server "$phase"

    # COHERENCE CANARY FIRST. A broken kernel produces garbage FAST, so a
    # broken run reports a BETTER tok/s, not a worse one. Gate before timing.
    say "[$phase] coherence canary"
    python3 "$q/run_coherence.py" 2>&1 | tee "$OUT_DIR/coherence_$phase.log"
    if ! grep -qiE "GATE\[OK\]|PASS" "$OUT_DIR/coherence_$phase.log"; then
        die "[$phase] coherence canary did not pass — any tok/s from this build is meaningless"
    fi

    # TREATMENT: the sampled config, which is the only one that reaches the
    # GPU radix top-k path at all.
    say "[$phase] sweep_sampled (temp 1.0 / top_p 0.95) — TREATMENT"
    python3 "$q/batch_load_probe.py" \
        --batches 1,8,32,128,256 --reps 2 --max-tokens 128 \
        --temperature 1.0 --top-p 0.95 \
        --label "radix_${phase}_sampled" \
        2>&1 | tee "$OUT_DIR/sweep_sampled_$phase.log"

    # CONTROL: the exact config the 111.69 baseline used. Expected: unchanged,
    # because temperature 0 takes the argmax path and never calls the fix.
    say "[$phase] sweep_greedy (temp 0.0 / top_p 1.0) — CONTROL"
    python3 "$q/batch_load_probe.py" \
        --batches 1,8,32,128,256 --reps 2 --max-tokens 128 \
        --temperature 0.0 --top-p 1.0 \
        --label "radix_${phase}_greedy" \
        2>&1 | tee "$OUT_DIR/sweep_greedy_$phase.log"

    # TASK #22: the long-prompt latency curve, at the sampled config.
    say "[$phase] prompt-length ladder"
    PORT="$PORT" python3 "$ARC_DIR/arc-tools/prompt_ladder.py" \
        --out "$OUT_DIR/ladder_$phase.json" \
        2>&1 | tee "$OUT_DIR/ladder_$phase.log"

    # The direct proof, straight from the server's own log.
    local slow warn
    slow=$(grep -c "SAMPLER ON CPU SLOW PATH" "$OUT_DIR/server_$phase.log" || true)
    warn=$(grep -cE "GPU radix top-k sampling (failed|FAILED|still failing)" \
        "$OUT_DIR/server_$phase.log" || true)
    say "[$phase] interval-logger slow-path lines: $slow ; sampler fallback log lines: $warn"
    echo "$phase slow_path_log_lines=$slow fallback_log_lines=$warn" >> "$OUT_DIR/counters.txt"

    stop_server
}

# ------------------------------------------------------------------ phase AFTER --
say "=== PHASE AFTER (fix applied) ==="
say "building --features 'cuda flash-attn' (NEVER cudnn: -62% decode on V4)"
cargo build --release -p mistralrs-cli --features "cuda flash-attn" \
    2>&1 | tail -20 | tee "$OUT_DIR/build_after.log" || die "build (after) failed"
run_phase after

# ----------------------------------------------------------------- phase BEFORE --
# Revert ONLY the helper. Everything else — including the new counter and the
# interval-logger line — stays, so the BEFORE run reports its own breakage.
say "=== PHASE BEFORE (helper reverted to origin/master) ==="
git -C "$ARC_DIR" fetch origin master --quiet 2>/dev/null
git -C "$ARC_DIR" checkout origin/master -- arc-cuda-graph/src/weights.rs \
    || die "could not revert weights.rs"
say "reverted arc-cuda-graph/src/weights.rs; rebuilding"
cargo build --release -p mistralrs-cli --features "cuda flash-attn" \
    2>&1 | tail -20 | tee "$OUT_DIR/build_before.log" || die "build (before) failed"
run_phase before

git -C "$ARC_DIR" checkout HEAD -- arc-cuda-graph/src/weights.rs || true
say "restored weights.rs"

# --------------------------------------------------------------------- report --
{
    echo "# Radix top-k dtype fix — same-box A/B"
    echo
    echo "Box: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "Commit: $(git -C "$ARC_DIR" rev-parse --short HEAD)"
    echo
    echo "## Server-log counters (the direct proof)"
    echo '```'
    cat "$OUT_DIR/counters.txt"
    echo '```'
    echo "Expected: before slow_path_log_lines > 0, after == 0."
    echo
    echo "## Numbers"
    echo "Extract from the sweep logs in $OUT_DIR:"
    echo "  - sweep_sampled_{before,after}.log  decode agg tok/s per B  <- TREATMENT"
    echo "  - sweep_greedy_{before,after}.log   decode agg tok/s per B  <- CONTROL (expect no change)"
    echo "  - ladder_{before,after}.json        s/token at 9 / 200 / 1055 words"
} > "$OUT_DIR/RESULTS.md"

grep -hE "decode_agg|agg tok|effective_B|FAIL" "$OUT_DIR"/sweep_*.log >> "$OUT_DIR/RESULTS.md" 2>/dev/null

say "DONE — see $OUT_DIR/RESULTS.md"
echo "STATUS=DONE" >> "$STATUS"
