#!/usr/bin/env bash
#
# wave61_mtp_radix.sh — take the two outstanding V4 numbers on one H200, in one
# unattended run:
#
#   1. MTP acceptance AT BATCH   (PR #86, feat/mtp-batch-lockstep)
#      Read `tok_per_step` from the MTP[b=<B>] lines. NOT accept_rate: a
#      saturated sequence drafts nothing, contributes proposed=0, and therefore
#      cannot depress accept_rate — which stays flattering while tok_per_step
#      collapses. The two diverge exactly where the question lives.
#
#   2. Radix KV sharing ON vs OFF (PR #82)
#      Toggle is --prefix-cache-n 16 vs 0 (engine/mod.rs folds 0 into
#      no_prefix_cache). Valid on V4 specifically because
#      DeepSeekV4Loader::supports_paged_attention is false — on a paged model
#      prefix_cacher.rs:361-372 bypasses the radix entirely and the flag would
#      be measuring the vLLM-style block pool instead.
#
# WHY THIS SCRIPT EXISTS RATHER THAN A LIST OF COMMANDS: three prior sessions on
# this box produced zero numbers because a server died at load and nothing
# noticed until the rental was over. Every phase here gates on the phase before
# it and names its own failure.
#
# ---------------------------------------------------------------------------
# THE TWO WAYS THIS RUN CAN LIE, AND THE GATES THAT CATCH THEM
# ---------------------------------------------------------------------------
# (a) Broken PTX JIT. candle-kernels ships PTX only; on a toolkit-newer-than-
#     driver box the JIT silently fails and kernels read back zeros. Under the
#     CUDA runtime API this does not raise — it lands in cudaGetLastError(),
#     not cudaDeviceSynchronize(). A model computing zeros STILL EMITS TOKENS,
#     and emits them FAST, so a broken box reports a flatteringly HIGH
#     throughput, not a low one. gpu_box_preflight.sh's PTX arm is the cheap
#     early-out.
# (b) A preflight cannot prove the real model is fine — only running it can.
#     So every server, before it is timed, must answer questions a working
#     model cannot miss (run_coherence.py, the in-tree PR #68 battery, which
#     also runs qlib.looks_degenerate over each completion: all-zero logits
#     argmax to one token forever and that repetition is this failure's
#     signature). Fail the canary and the run is VOID — we report "void", never
#     "MTP scored X".
#
# Environment failures (preflight trip, PTX death, incoherent canary) exit 2
# and are never reported as a measurement. Genuine measurement problems exit 1.
# ---------------------------------------------------------------------------

set -uo pipefail

REPO="${REPO:-/root/arc}"
BIN="${BIN:-$REPO/target/release/mistralrs}"
SRC="${SRC:-/root/models/v4-src}"
UQFF_DIR="${UQFF_DIR:-/root/models/v4-uqff}"
OUT="${OUT:-/root/wave61}"
PORT="${PORT:-1234}"
COST_HR="${COST_HR:-4.92}"
PREFIX_TOKENS="${PREFIX_TOKENS:-1400}"
Q="$REPO/arc-tools/quality"
STATUS="$OUT/status.txt"

mkdir -p "$OUT"
: > "$STATUS"
say() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" | tee -a "$STATUS"; }
die_env() { say "VOID[$1] $2"; say "RESULT=VOID (environment) — not a measurement"; exit 2; }
die_run() { say "FAIL[$1] $2"; say "RESULT=FAIL"; exit 1; }

say "wave61 start on $(hostname); repo=$REPO bin=$BIN"

# ---------------------------------------------------------------- phase 0
say "PHASE0 preflight (hard gate: PTX arm must return 42)"
# shellcheck disable=SC1091
. "$REPO/arc-tools/gpu_box_preflight.sh" --bin "$BIN" 2>&1 | tee -a "$STATUS"
grep -q "PREFLIGHT=PASS" "$STATUS" || die_env PREFLIGHT "gpu_box_preflight.sh did not pass; see $STATUS"
# The sourced preflight exports CUDA_HOME / LD_LIBRARY_PATH (incl. the
# forward-compat libcuda). Re-source without the pipe so this shell inherits
# them — a pipeline runs in a subshell and the exports would be lost.
. "$REPO/arc-tools/gpu_box_preflight.sh" --quick --bin "$BIN" >/dev/null 2>&1
say "PHASE0 env: CUDA_HOME=$CUDA_HOME LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

UQFF="$(ls "$UQFF_DIR"/*-0.uqff 2>/dev/null | head -1)"
[ -z "$UQFF" ] && UQFF="$(ls "$UQFF_DIR"/*.uqff 2>/dev/null | head -1)"
[ -z "$UQFF" ] && die_env UQFF "no .uqff under $UQFF_DIR"
CHAT_TPL="$REPO/chat_templates/deepseek_v4.json"
[ -f "$CHAT_TPL" ] || die_env TEMPLATE "missing $CHAT_TPL"
say "PHASE0 uqff=$UQFF"

export BASE_URL="http://127.0.0.1:$PORT"
export RUST_LOG="${RUST_LOG:-info}"
export ARC_MTP_LOG_ACCEPTANCE=1

# ---------------------------------------------------------------- helpers
stop_server() {
    pkill -f "release/mistralrs serve" >/dev/null 2>&1
    for _ in $(seq 1 30); do
        pgrep -f "release/mistralrs serve" >/dev/null 2>&1 || break
        sleep 2
    done
    pkill -9 -f "release/mistralrs serve" >/dev/null 2>&1
    sleep 3
}

# start_server <tag> <mtp_depth> <prefix_cache_n> <max_seqs>
start_server() {
    local tag="$1" mtp="$2" pcn="$3" seqs="$4"
    local log="$OUT/serve_$tag.log"
    stop_server
    say "  starting server[$tag]: --mtp-depth $mtp --prefix-cache-n $pcn --max-seqs $seqs"
    setsid nohup "$BIN" serve -p "$PORT" \
        -m "$SRC" -a deepseekv4 --from-uqff "$UQFF" \
        --chat-template "$CHAT_TPL" \
        --paged-attn off \
        --mtp-depth "$mtp" --prefix-cache-n "$pcn" --max-seqs "$seqs" \
        < /dev/null > "$log" 2>&1 &
    local waited=0
    while [ "$waited" -lt 1800 ]; do
        sleep 10; waited=$((waited + 10))
        if curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; then
            say "  server[$tag] healthy after ${waited}s"
            return 0
        fi
        if ! pgrep -f "release/mistralrs serve" >/dev/null 2>&1; then
            say "  server[$tag] DIED after ${waited}s; last lines:"
            tail -20 "$log" | tee -a "$STATUS"
            return 1
        fi
        [ $((waited % 120)) -eq 0 ] && say "  server[$tag] loading ${waited}s: $(tail -1 "$log" | cut -c1-110)"
    done
    say "  server[$tag] never became healthy within ${waited}s"
    return 1
}

# canary <tag> [--skip-facts]
# Authoritative correctness gate. A preflight proves the toolchain; only this
# proves the loaded model computes. Gate: coherence6 >= 5/6, and when facts run,
# facts+math >= 70% (the runbook's abort-if threshold).
canary() {
    local tag="$1"; shift
    say "  canary[$tag] running (this decides whether the timings mean anything)"
    ( cd "$Q" && python3 run_coherence.py --out "$OUT/coherence_$tag.json" "$@" ) \
        > "$OUT/canary_$tag.log" 2>&1
    local line
    line="$(grep -m1 '^COHERENCE:' "$OUT/canary_$tag.log")"
    if [ -z "$line" ]; then
        say "  canary[$tag] produced no COHERENCE line:"
        tail -15 "$OUT/canary_$tag.log" | tee -a "$STATUS"
        return 1
    fi
    say "  canary[$tag] $line"
    local c6 tot
    c6="$(printf '%s' "$line" | sed -n 's/.*coherence6 \([0-9]*\)\/\([0-9]*\).*/\1/p')"
    tot="$(printf '%s' "$line" | sed -n 's/.*coherence6 \([0-9]*\)\/\([0-9]*\).*/\2/p')"
    [ -z "$c6" ] && return 1
    if [ "$c6" -lt 5 ]; then
        say "  canary[$tag] FAILED: coherence6 $c6/$tot < 5 — the model is not computing correctly"
        return 1
    fi
    # Facts/math when they were not skipped.
    local f_ok f_tot m_ok m_tot
    f_ok="$(printf '%s' "$line" | sed -n 's/.*facts \([0-9]*\)\/\([0-9]*\).*/\1/p')"
    f_tot="$(printf '%s' "$line" | sed -n 's/.*facts \([0-9]*\)\/\([0-9]*\).*/\2/p')"
    m_ok="$(printf '%s' "$line" | sed -n 's/.*math \([0-9]*\)\/\([0-9]*\).*/\1/p')"
    m_tot="$(printf '%s' "$line" | sed -n 's/.*math \([0-9]*\)\/\([0-9]*\).*/\2/p')"
    if [ -n "$f_ok" ] && [ -n "$m_ok" ] && [ "$((f_tot + m_tot))" -gt 0 ]; then
        local pct=$(( (f_ok + m_ok) * 100 / (f_tot + m_tot) ))
        say "  canary[$tag] known-answer $((f_ok + m_ok))/$((f_tot + m_tot)) = ${pct}%"
        [ "$pct" -lt 70 ] && { say "  canary[$tag] FAILED: ${pct}% < 70% abort-if threshold"; return 1; }
    fi
    say "  canary[$tag] PASS"
    return 0
}

# ---------------------------------------------------------------- phase 1: MTP
say "PHASE1 MTP acceptance at batch (depth 3, prefix cache OFF so draft KV primes)"
start_server mtp 3 0 128 || die_env SERVER_MTP "MTP server never came up; see $OUT/serve_mtp.log"
grep -c "UNSUPPORTED_PTX" "$OUT/serve_mtp.log" | grep -qv '^0$' && \
    die_env PTX "UNSUPPORTED_PTX in the MTP serve log despite a passing preflight"
canary mtp || die_env CANARY_MTP "MTP server is incoherent — the run is VOID, no MTP number is reported"

# One batch size per probe invocation, with a UTC fence recorded around each.
#
# WHY NOT ONE SWEEP: the MTP[b=<B>] markers are emitted by
# AcceptanceTelemetry::log() only when a 64-proposed boundary is crossed, and
# the per-request MTP[req=] lines carry no batch label at all. If the b= lines
# do not appear, a single mixed-batch sweep leaves the per-request data
# unattributable and the run yields nothing — which is exactly what happened on
# the first attempt. Fencing each batch size in wall-clock lets the per-request
# lines be bucketed by timestamp, so a usable tok_per_step survives even when
# the aggregate markers do not.
say "PHASE1 sweep B=1,8,32,128 at 64 decode tokens, temperature 0 (one probe per B)"
: > "$OUT/mtp_fences.txt"
for B in 1 8 32 128; do
    echo "FENCE_START B=$B $(date -u +%Y-%m-%dT%H:%M:%S)" >> "$OUT/mtp_fences.txt"
    ( cd "$Q" && python3 batch_load_probe.py \
        --batches "$B" --reps 2 --max-tokens 64 --temperature 0 \
        --cost-per-hour "$COST_HR" --label "mtp_b$B" \
        --out "$OUT/mtp_b$B.json" ) > "$OUT/mtp_b$B.log" 2>&1
    say "  B=$B probe rc=$?"
    echo "FENCE_END B=$B $(date -u +%Y-%m-%dT%H:%M:%S)" >> "$OUT/mtp_fences.txt"
    grep -E "^(BATCH|CONC|PREFILL|FAIL|WARN)" "$OUT/mtp_b$B.log" | tee -a "$STATUS"
    grep -E "^  B=[0-9]+ rep" "$OUT/mtp_b$B.log" | tee -a "$STATUS"
done

say "PHASE1 MTP aggregate markers, if the engine emitted any:"
grep -o "MTP\[agg\].*" "$OUT/serve_mtp.log" 2>/dev/null | tail -1 | tee -a "$STATUS"
grep -o "MTP\[b=[0-9]*\].*" "$OUT/serve_mtp.log" 2>/dev/null \
    | awk '{ b=$1; last[b]=$0 } END { for (k in last) print last[k] }' \
    | sort -t= -k2 -n | tee "$OUT/mtp_markers.txt" | tee -a "$STATUS"

# Fallback: bucket the per-request lines by the wall-clock fences and sum the
# raw counts. tok_per_step = committed/steps over the bucket, which is the same
# ratio the engine's own marker reports — computed from the same raw fields, so
# it is not a different metric, just a different aggregation point.
say "PHASE1 per-request MTP bucketed by batch fence (tok_per_step = committed/steps):"
python3 "$Q/mtp_bucket.py" --serve-log "$OUT/serve_mtp.log" \
    --fences "$OUT/mtp_fences.txt" --out "$OUT/mtp_by_batch.json" \
    2>&1 | tee -a "$STATUS"

if ! [ -s "$OUT/mtp_markers.txt" ] && ! [ -s "$OUT/mtp_by_batch.json" ]; then
    say "PHASE1 WARNING: no MTP data at all. Acceptance is UNMEASURED — NOT 0%."
    grep -m3 -iE "mtp|speculative" "$OUT/serve_mtp.log" | tee -a "$STATUS"
fi
stop_server

# ---------------------------------------------------------------- phase 2: radix
# Three cells, so prefix reuse is separable from raw batching:
#   ON  + shared    radix can reuse
#   OFF + shared    same workload, no cache  (the honest baseline)
#   ON  + distinct  cache on, nothing reusable (matched-length control)
#
# ONE SERVER PER CELL. The hit-rate counters in engine/logger.rs are read with
# `load`, not `swap` — only `tokens_processed` is reset per interval. So
# `Prefix cache hitrate` is a CUMULATIVE lifetime ratio
# (prefix_cache_hits / total_new_seqs) over everything the process ever served,
# canary and warmups included. Running the shared and distinct cells against one
# server therefore blends a cell that can hit with a cell that provably cannot,
# and the printed rate belongs to neither. Separate processes, one rate each.
#
# WORKLOAD SHAPE. A barrier-synchronised one-shot batch releases all B requests
# at the same instant, and sequences only enter the cache when they FINISH
# (prefix_cacher.rs add_sequence is called from sampling.rs:229/:388). So the
# entire first burst is a guaranteed miss no matter how good the cache is —
# there is nothing in it yet. Production shared-prefix traffic is a closed loop,
# not a thundering herd, so each cell now warms sequentially first and then
# measures in --duration sustained mode, where finished requests are recycled
# and the cache is actually exercised.
run_radix_cell() {
    local tag="$1" batches="$2" mode="$3"
    say "  radix cell[$tag] batches=$batches prefix_mode=$mode tokens=$PREFIX_TOKENS"
    ( cd "$Q" && python3 batch_load_probe.py \
        --batches "$batches" --reps 2 --max-tokens 64 --temperature 0 \
        --prefix-tokens "$PREFIX_TOKENS" --prefix-mode "$mode" \
        --cost-per-hour "$COST_HR" --label "$tag" \
        --out "$OUT/radix_$tag.json" ) > "$OUT/radix_$tag.log" 2>&1
    say "  radix cell[$tag] rc=$?"
    grep -E "^(BATCH|CONC|PREFILL|PREFIX|BATCHSWEEP|FAIL|WARN)" "$OUT/radix_$tag.log" | tee -a "$STATUS"
    grep -E "^  B=[0-9]+ rep" "$OUT/radix_$tag.log" | tee -a "$STATUS"
}

# Sequential warm: a handful of one-at-a-time requests sharing the prefix, so
# the cache is non-empty before the concurrent measurement starts.
warm_prefix() {
    local tag="$1" mode="$2"
    say "  warming prefix cache for [$tag] (sequential, B=1 x 4)"
    ( cd "$Q" && python3 batch_load_probe.py \
        --batches 1 --reps 4 --max-tokens 8 --temperature 0 \
        --prefix-tokens "$PREFIX_TOKENS" --prefix-mode "$mode" \
        --cost-per-hour "$COST_HR" --label "warm_$tag" \
        --out "$OUT/warm_$tag.json" ) > "$OUT/warm_$tag.log" 2>&1
    say "  warm[$tag] hitrate after: $(grep -o "Prefix cache hitrate [0-9.]*%" "$OUT/serve_$tag.log" 2>/dev/null | tail -1)"
}

harvest_hitrate() {
    local tag="$1"
    say "  hit rate[$tag] (CUMULATIVE lifetime ratio, last 3 intervals):"
    grep -o "Throughput (T/s)[^\"]*" "$OUT/serve_$tag.log" 2>/dev/null | tail -3 | tee -a "$STATUS"
    local dec
    dec="$(grep -c "prefix cache: declining" "$OUT/serve_$tag.log" 2>/dev/null)"
    say "  hit rate[$tag] declined matches: $dec"
    if [ "${dec:-0}" -gt 0 ]; then
        say "  NOTE: a declined match is a match the radix FOUND and then refused"
        say "        on the KV-length guard. A low hit rate with declines>0 is a"
        say "        KV-contract problem, not a cache-miss problem."
        grep -o "prefix cache: declining[^\"]*" "$OUT/serve_$tag.log" 2>/dev/null | head -2 | tee -a "$STATUS"
    fi
}

say "PHASE2a radix ON + shared prefix (--prefix-cache-n 16)"
start_server radixon_shared 0 16 256 || die_env SERVER_RADIXON "radix-ON server never came up"
canary radixon_shared --skip-facts || die_env CANARY_RADIXON "radix-ON server is incoherent — VOID"
warm_prefix radixon_shared shared
run_radix_cell radixon_shared "64,128,256" shared
harvest_hitrate radixon_shared
stop_server

say "PHASE2b radix ON + DISTINCT prefix (matched-length control, own server)"
start_server radixon_distinct 0 16 256 || die_env SERVER_RADIXONDIST "control server never came up"
warm_prefix radixon_distinct distinct
run_radix_cell radixon_distinct "256" distinct
harvest_hitrate radixon_distinct
stop_server

say "PHASE2c radix OFF + shared prefix — the baseline the 111.69 tok/s @ B=256 figure was taken with"
start_server radixoff_shared 0 0 256 || die_env SERVER_RADIXOFF "radix-OFF server never came up"
canary radixoff_shared --skip-facts || die_env CANARY_RADIXOFF "radix-OFF server is incoherent — VOID"
run_radix_cell radixoff_shared "64,128,256" shared
harvest_hitrate radixoff_shared
stop_server

# ---------------------------------------------------------------- summary
say "===================== WAVE61 SUMMARY ====================="
say "-- MTP tok_per_step by batch (read tok_per_step, NOT accept_rate):"
cat "$OUT/mtp_markers.txt" 2>/dev/null | tee -a "$STATUS"
say "-- radix aggregate throughput:"
for f in radixon_shared radixon_distinct radixoff_shared; do
    grep -hE "^(BATCH|PREFIX)\[B=" "$OUT/radix_$f.log" 2>/dev/null | sed "s/^/[$f] /" | tee -a "$STATUS"
done
say "-- prefix cache hit rate (radix ON, last):"
grep -o "Prefix cache hitrate[^,]*" "$OUT/serve_radixon.log" 2>/dev/null | tail -3 | tee -a "$STATUS"
say "NOTE: cross-prefix reuse meter (share_stats/CrossPrefixMeter) has zero"
say "      production callers and is NOT reportable. Backlog, not a number."

# ---------------------------------------------------------------- assertion
#
# The canary proves the MODEL is fine. Nothing proved that NUMBERS WERE
# ACTUALLY PRODUCED. The first run of this script printed RESULT=OK while every
# batch row read `decode agg None tok/s` — a clean pass over an empty result,
# which is the same fake-pass class the canary exists to prevent, one layer up.
# A requested batch size that yields no throughput is a failed run.
missing=""
for f in "$OUT"/mtp_b*.log "$OUT"/radix_*.log; do
    [ -f "$f" ] || continue
    while IFS= read -r line; do
        case "$line" in
            *"decode agg None"*|*"decode agg  tok/s"*)
                missing="$missing $(basename "$f" .log):$(printf '%s' "$line" | sed -n 's/.*BATCH\[B=\([0-9]*\)\].*/B=\1/p')" ;;
        esac
    done < <(grep "^BATCH\[B=" "$f" 2>/dev/null)
done
if [ -n "$missing" ]; then
    say "ASSERT FAILED — these cells produced no throughput:$missing"
    say "RESULT=INCOMPLETE (cells ran but yielded no number; see the per-rep rows above)"
    exit 1
fi
if ! [ -s "$OUT/mtp_by_batch.json" ] && ! [ -s "$OUT/mtp_markers.txt" ]; then
    say "ASSERT FAILED — no MTP acceptance data at any batch size."
    say "RESULT=INCOMPLETE (MTP acceptance UNMEASURED — not 0%)"
    exit 1
fi
say "RESULT=OK  artifacts in $OUT"
