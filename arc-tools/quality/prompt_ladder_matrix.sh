#!/usr/bin/env bash
#
# prompt_ladder_matrix.sh — re-measure the V4 prefill cost curve under a matrix
# of configurations, on a binary whose identity is PROVEN rather than assumed.
#
# WHY: session 8 recorded 9w = 100.15 s, 198w = 386.15 s and ~1,055w = server
# dead. A later canary on a provenance-verified build measured 512w in 3.2 s and
# 1100w in 6.6 s at B=1. Those cannot both describe the same code on the same
# hardware. Two task reprioritisations and a public claim about what V4 can
# serve currently rest on the first set.
#
# The four candidate explanations are distinguishable by this matrix:
#   1. stale/wrong binary  -> the provenance gate below makes it unmeasurable
#   2. configuration       -> paged-attn auto(on) vs off is the axis
#   3. load, not length    -> B=1 vs B=8 is the axis
#   4. genuine fix         -> survives 1-3 and reproduces nowhere
#
# THE PROVENANCE GATE IS NOT OPTIONAL. Session 8's harness family was later
# shown to build one package and launch another, so a run there may have
# measured a leftover binary describing unknown code. This script builds into a
# private CARGO_TARGET_DIR inside a fresh worktree, then reads the `git
# revision` the SERVER ITSELF logs at startup and refuses to take a single
# measurement unless it equals the ref that was built. A number whose binary
# cannot be identified is not a measurement, and producing one is worse than
# producing none because it outlives the session that made it.
#
# Usage:
#   REF=origin/master bash arc-tools/quality/prompt_ladder_matrix.sh
#   REF=ab42c4508 OUT=/root/ladder_stack bash .../prompt_ladder_matrix.sh
#
# Exit: 0 all cells attempted · 2 environment/provenance (never a measurement)

set -uo pipefail

REF="${REF:-origin/master}"
REPO="${REPO:-/root/arc}"
OUT="${OUT:-/root/ladder}"
PORT="${PORT:-1234}"
MODEL="${MODEL:-/root/models/v4-src}"
UQFF="${UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}"
ARCH="${ARCH:-deepseekv4}"
WORDS="${WORDS:-40,128,512,1100,2048}"
BATCHES="${BATCHES:-1,8}"
CELL_TIMEOUT="${CELL_TIMEOUT:-420}"
LOCK="${LOCK:-/root/locks/gpu.lock}"
LOCK_WAIT="${LOCK_WAIT:-5400}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="${PROBE:-$SELF_DIR/ladder_probe.py}"

mkdir -p "$OUT" "$(dirname "$LOCK")"
STATUS="$OUT/status.txt"; : > "$STATUS"
say(){ printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" | tee -a "$STATUS"; }
die_env(){ say "VOID[$1] $2"; say "RESULT=VOID (environment) — not a measurement"; exit 2; }

say "ladder start ref=$REF out=$OUT"
# Assert the tools exist BEFORE taking the lock or building. A launcher that
# prints a PID for a missing script is not a launch, and the only trace is one
# line in a log nobody tails — the same shape as a server that never started.
[ -r "$PROBE" ] || die_env PROBE_MISSING "no ladder_probe.py at $PROBE (set PROBE=/path)"
command -v cargo >/dev/null 2>&1 || die_env CARGO "cargo not on PATH"
[ -d "$REPO/.git" ] || die_env REPO "no git repo at $REPO"
[ -f "$MODEL/config.json" ] || die_env MODEL "no model at $MODEL"
[ -f "$UQFF" ] || die_env UQFF "no uqff artifact at $UQFF"

# ------------------------------------------------------------------ the lock
# Queue behind whoever holds the GPU. `set -o noclobber` makes the create
# atomic, so two agents cannot both believe they hold it.
waited=0
while [ -e "$LOCK" ]; do
    [ "$waited" -eq 0 ] && say "waiting on $LOCK (held by: $(cat "$LOCK" 2>/dev/null | head -1))"
    sleep 20; waited=$((waited + 20))
    [ "$waited" -ge "$LOCK_WAIT" ] && die_env LOCK "waited ${waited}s for $LOCK; giving up rather than racing the holder"
done
if ! (set -o noclobber; printf "ladder pid=%s ref=%s started=%s\n" "$$" "$REF" "$(date -u +%FT%TZ)" > "$LOCK") 2>/dev/null; then
    die_env LOCK "lost the race for $LOCK; another agent took it first"
fi
say "lock acquired after ${waited}s"
release(){ rm -f "$LOCK"; pkill -f "release/mistralrs serve" >/dev/null 2>&1; }
trap release EXIT INT TERM

# ------------------------------------------------------- build with provenance
git -C "$REPO" fetch -q origin 2>/dev/null
SHA="$(git -C "$REPO" rev-parse "$REF" 2>/dev/null)"
[ -z "$SHA" ] && die_env REF "cannot resolve ref $REF"
SHORT="${SHA:0:9}"
WT="/root/wt-ladder-${SHA:0:8}"
say "ref=$REF sha=$SHA worktree=$WT"

if [ ! -d "$WT" ]; then
    git -C "$REPO" worktree add -f "$WT" "$SHA" >>"$STATUS" 2>&1 \
        || die_env WORKTREE "could not create $WT"
fi
# A private target dir is half the provenance guarantee: a shared one is how a
# leftover binary from another branch gets launched by a script that thinks it
# just built it.
export CARGO_TARGET_DIR="$WT/target-ladder"
BIN="$CARGO_TARGET_DIR/release/mistralrs"

# shellcheck disable=SC1091
. /root/arcenv.sh 2>/dev/null
if [ -f "$REPO/arc-tools/gpu_box_preflight.sh" ]; then
    . "$REPO/arc-tools/gpu_box_preflight.sh" --quick >>"$STATUS" 2>&1
fi

say "building $SHORT (private target dir)"
t0=$SECONDS
( cd "$WT" && cargo build --release --features "cuda flash-attn" ) >>"$OUT/build.log" 2>&1
rc=$?
say "build rc=$rc in $((SECONDS-t0))s"
[ $rc -ne 0 ] && { tail -25 "$OUT/build.log" | tee -a "$STATUS"; die_env BUILD "cargo build failed"; }
[ -x "$BIN" ] || die_env BUILD "no binary at $BIN after a successful build"
say "binary $BIN mtime $(stat -c %y "$BIN")"

# ---------------------------------------------------------------- one config
# $1 = label, rest = extra serve flags
run_config(){
    local label="$1"; shift
    local slog="$OUT/serve_$label.log"
    pkill -f "release/mistralrs serve" >/dev/null 2>&1; sleep 3
    say "config[$label]: serve $*"
    setsid nohup "$BIN" serve -p "$PORT" -m "$MODEL" -a "$ARCH" \
        --from-uqff "$UQFF" \
        --chat-template "$WT/chat_templates/deepseek_v4.json" \
        --max-seqs 32 "$@" \
        < /dev/null > "$slog" 2>&1 &
    local spid=$!
    local up=0
    for _ in $(seq 1 300); do
        if curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1; then up=1; break; fi
        if ! kill -0 "$spid" 2>/dev/null; then
            say "  config[$label] server DIED during load"
            tail -20 "$slog" | tee -a "$STATUS"
            return 1
        fi
        sleep 2
    done
    [ "$up" -eq 1 ] || { say "  config[$label] never became healthy"; tail -15 "$slog" | tee -a "$STATUS"; return 1; }

    # ---- THE GATE ----
    local logged
    logged="$(grep -m1 -oE "git revision: [0-9a-f]+" "$slog" | awk '{print $3}')"
    if [ -z "$logged" ]; then
        say "  config[$label] server logged NO git revision — cannot identify the binary"
        return 2
    fi
    case "$SHA" in
        "$logged"*) : ;;
        *) case "$logged" in
               "${SHA:0:${#logged}}") : ;;
               *) say "  PROVENANCE MISMATCH: server logged $logged, built $SHA"
                  say "  refusing to measure — this is exactly the failure that put"
                  say "  unattributable numbers into FACTS.md"
                  return 2 ;;
           esac ;;
    esac
    say "  provenance OK: server revision $logged == built $SHORT"

    # The probe comes from THIS script's own directory, never from the
    # worktree. The worktree is checked out at $REF — the code under test —
    # and the measurement tool is not part of what is being measured. Reading
    # it from $WT would also simply fail whenever $REF predates the tool.
    python3 "$PROBE" \
        --base-url "http://127.0.0.1:$PORT" \
        --words "$WORDS" --batches "$BATCHES" \
        --cell-timeout "$CELL_TIMEOUT" --config "$label" \
        --tsv "$OUT/ladder_$label.tsv" 2>&1 | tee -a "$STATUS"

    say "  config[$label] server faults: cublas=$(grep -c CublasError "$slog") illegal=$(grep -c ILLEGAL_ADDRESS "$slog") panics=$(grep -c panicked "$slog")"
    pkill -f "release/mistralrs serve" >/dev/null 2>&1; sleep 3
    return 0
}

# `pagedauto` deliberately passes NO --paged-attn flag, reproducing session 8:
# the default is `auto`, which is ON for CUDA, on a model whose loader reports
# supports_paged_attention = false. `pagedoff` is the same binary and workload
# with the one flag changed, so any difference between them is that flag.
run_config pagedauto
rc_auto=$?
run_config pagedoff --paged-attn off
rc_off=$?

say "===================== LADDER SUMMARY ====================="
say "ref=$REF sha=$SHORT  (session 8 claimed: 9w=100.15s, 198w=386.15s, 1055w=dead)"
for f in "$OUT"/ladder_*.tsv; do
    [ -f "$f" ] || continue
    say "-- $(basename "$f")"
    cat "$f" | tee -a "$STATUS"
done
[ "$rc_auto" = "2" ] || [ "$rc_off" = "2" ] && die_env PROVENANCE "at least one config refused to measure; see above"
if ! ls "$OUT"/ladder_*.tsv >/dev/null 2>&1; then
    say "RESULT=INCOMPLETE — no cells produced"
    exit 1
fi
say "RESULT=OK  artifacts in $OUT"
