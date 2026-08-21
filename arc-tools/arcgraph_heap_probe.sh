#!/bin/bash
# arcgraph_heap_probe.sh — is the heap corruption ours, and does it need capture?
#
# `malloc_consolidate(): invalid chunk size` after a CUDA-graph capture is host
# heap corruption: something wrote outside an allocation, double-freed, or gave
# glibc a pointer it never owned. Two questions, one leg each, plus a control.
#
#   Q1 SCOPE   Does the corruption require capture at all? If the deferred-free
#              bookkeeping (gated on `capturing`) is the culprit, graphs-off is
#              clean. If graphs-off ALSO corrupts, then every number this
#              project has ever measured rode on a corrupted heap.
#   Q2 SITE    Can glibc be made to abort at the bad operation instead of at
#              consolidation, moving the abort earlier and naming a different
#              call?
#
# ⚠️ THE ABORT SITE IS NOT THE CORRUPTION SITE. glibc notices a bad chunk header
# when it next walks that arena, which can be arbitrarily many allocations after
# the write that broke it. Everything below reports where corruption BECAME
# VISIBLE. Nothing here locates the bug.
#
# ⚠️ AND A NO-OP ENV VAR LOOKS EXACTLY LIKE A CLEAN RUN. So the tunables are
# proven to engage with a deliberate positive control BEFORE either leg runs; if
# the control does not abort, the legs are reported UNPROVEN rather than clean.
set -u

LOGDIR=/root/logs/arcgraph-heap
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.txt"
: > "$STATUS"
REPO=/root/arc
SRC=/root/models/v4-src
UQFF=/root/models/v4-uqff/qtip2b-0.uqff
PORT=1234
BASE_URL="http://localhost:$PORT"
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"

say()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }
step() { say "=== $* ==="; }
env_fail() { say "ENV_FAIL: $*"; say "RESULT: UNANSWERED"; exit 2; }

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    sleep 3
    [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null
    pkill -f "mistralrs serve -p $PORT" 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------
# 0. Exclusivity — a neighbour's process would put its own Xids in dmesg and
#    make every attribution below worthless.
# --------------------------------------------------------------------------
OCC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c .)
[ "${OCC:-0}" = "0" ] || env_fail "$OCC foreign compute process(es) on the GPU; dmesg attribution would be unsound"
say "GPU clear"

BIN=$(ls -1 "$REPO/target/release/mistralrs" 2>/dev/null | head -1)
[ -x "$BIN" ] || env_fail "no built binary at $REPO/target/release/mistralrs"
say "BIN=$BIN"

# --------------------------------------------------------------------------
# 1. POSITIVE CONTROL — prove the tunables actually engage on THIS glibc.
#    A deliberate one-byte overflow, freed and re-malloc'd. Under a working
#    MALLOC_CHECK_ this aborts; unset it usually does not. If the control does
#    not abort, the env vars are a no-op here and the legs prove nothing.
# --------------------------------------------------------------------------
step "1/4 positive control: do MALLOC_CHECK_/MALLOC_PERTURB_ engage?"
cat > /tmp/heapctl.c <<'C'
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
int main(void) {
    char *a = malloc(32);
    memset(a, 'A', 40);          /* deliberate 8-byte overflow past the chunk */
    free(a);                     /* glibc may notice here, or later */
    char *b = malloc(32);        /* ... or here, walking the arena */
    free(b);
    printf("NO_ABORT\n");
    return 0;
}
C
gcc -O0 -o /tmp/heapctl /tmp/heapctl.c 2>>"$LOGDIR/control.log" || env_fail "could not build the heap control"

CTL_OFF=$(/tmp/heapctl 2>&1; echo "rc=$?")
CTL_ON=$(MALLOC_CHECK_=3 MALLOC_PERTURB_=170 GLIBC_TUNABLES=glibc.malloc.check=3 /tmp/heapctl 2>&1; echo "rc=$?")
say "control WITHOUT tunables: $(echo "$CTL_OFF" | tr '\n' ' ')"
say "control WITH    tunables: $(echo "$CTL_ON" | tr '\n' ' ')"

# 🔴 The first version of this check grepped CTL_ON for "abort" — and the
# control's own success sentinel is the literal string "NO_ABORT", so it matched
# its own all-clear and declared the tunables working when both runs had printed
# NO_ABORT rc=0. The guard against a false clean produced a false clean, by
# substring. Detect the SENTINEL exactly, and require the two runs to DIFFER —
# engagement is a behavioural difference, not the presence of a word.
CTL_OFF_SENTINEL=$(printf '%s' "$CTL_OFF" | grep -c '^NO_ABORT$')
CTL_ON_SENTINEL=$(printf '%s' "$CTL_ON" | grep -c '^NO_ABORT$')
TUNABLES_ENGAGED=0
if [ "$CTL_ON_SENTINEL" = "0" ] && [ "$CTL_OFF_SENTINEL" != "0" ]; then
    TUNABLES_ENGAGED=1
    say "✅ tunables ENGAGE: control reached NO_ABORT without them and did NOT with them"
elif [ "$CTL_ON_SENTINEL" = "0" ] && [ "$CTL_OFF_SENTINEL" = "0" ]; then
    say "⚠️ control aborted in BOTH configurations — glibc catches this overflow"
    say "   unconditionally, so it cannot discriminate. Tunable engagement UNPROVEN."
else
    say "⚠️ tunables DO NOT ENGAGE: the control printed NO_ABORT with them set."
    say "   (MALLOC_CHECK_ is a no-op on glibc builds without malloc-check compiled in;"
    say "   GLIBC_TUNABLES=glibc.malloc.check needs the same support.)"
    say "   Every heap result below is therefore UNPROVEN, not clean."
fi

dmesg_count() { dmesg 2>/dev/null | grep -ciE "xid" || echo 0; }

# --------------------------------------------------------------------------
# 2/3. The two legs.
# --------------------------------------------------------------------------
run_leg() { # $1 = name, $2 = extra env
    local name="$1" extra="$2"
    local slog="$LOGDIR/$name.server.log"
    local before after
    before=$(dmesg_count)
    step "leg $name"
    say "  env: $extra"
    # The chat-template path is relative to the repo. cwd here is wherever the
    # script was launched (/root), so BOTH legs died at startup with
    # "Loading chat template failed: NotFound" and the verdict below still
    # printed 0 diagnostics for each — a vacuous zero that read as "clean".
    cd "$REPO" || { say "  cannot enter $REPO"; return 1; }
    # shellcheck disable=SC2086
    env $extra MALLOC_CHECK_=3 MALLOC_PERTURB_=170 GLIBC_TUNABLES=glibc.malloc.check=3 \
        "$BIN" serve -p "$PORT" -m "$SRC" -a deepseekv4 --from-uqff "$UQFF" \
        --chat-template chat_templates/deepseek_v4.json \
        --prefix-cache-n 0 --paged-attn off --max-seqs 1 --max-seq-len 4096 \
        >"$slog" 2>&1 &
    SERVER_PID=$!

    local healthy=0
    for _ in $(seq 1 90); do
        curl -fsS --max-time 5 "$BASE_URL/health" >/dev/null 2>&1 && { healthy=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 5
    done
    if [ "$healthy" != "1" ]; then
        say "  server never became healthy; tail:"
        tail -6 "$slog" | sed 's/^/    /' | tee -a "$STATUS"
        cleanup; SERVER_PID=""
        after=$(dmesg_count); say "  new Xid lines during leg: $((after - before))"
        return 1
    fi

    curl -s --max-time 600 "$BASE_URL/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"default","prompt":"Count from one to twenty in words, then stop.","max_tokens":24,"temperature":0}' \
        >"$LOGDIR/$name.response.json" 2>&1
    local ntok
    ntok=$(python3 -c "import json;print(json.load(open('$LOGDIR/$name.response.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
    say "  completion_tokens=$ntok"
    echo "${ntok:-0}" > "$LOGDIR/$name.tokens"

    # Did the process survive its own request?
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        say "  server ALIVE after the request"
    else
        say "  🔴 server DIED during the request"
    fi

    say "  glibc diagnostics in this leg:"
    grep -oiE "malloc_consolidate\(\)[^\"]{0,60}|free\(\): [a-z ]{0,40}|malloc\(\): [a-z ]{0,40}|corrupted[a-z -]{0,40}|double free[a-z ]{0,20}" "$slog" \
        | sort | uniq -c | sed 's/^/    /' | tee -a "$STATUS"
    [ -z "$(grep -oiE 'malloc_consolidate|free\(\):|malloc\(\):|corrupted|double free' "$slog")" ] && say "    (none)"

    say "  last line of the server log:"
    tail -1 "$slog" | sed 's/^/    /' | tee -a "$STATUS"

    cleanup; SERVER_PID=""
    sleep 4
    after=$(dmesg_count)
    say "  new Xid lines during leg: $((after - before))"
    if [ "$((after - before))" -gt 0 ]; then
        dmesg 2>/dev/null | grep -iE "xid" | tail -$((after - before)) | sed 's/^/    /' | tee -a "$STATUS"
    fi
    return 0
}

# Q1 SCOPE: graphs fully OFF. No capture stream, no probe, no alloc cache.
#
# POLARITY. This leg used to pass `ARC_NO_DEDICATED_DECODE=0`, which reads as
# "off" and was NOT: `normal.rs` tested the variable with `var_os(..).is_some()`,
# so ANY value — `0` included — disabled the dedicated decode path. Both legs of
# this A/B therefore ran with the path off and the comparison cancelled out. The
# reader is fixed (`normal.rs:env_flag_is_set`), and this leg now says `=1`,
# which is what "graphs fully OFF" always meant.
run_leg graphs_off "ARC_NO_DEDICATED_DECODE=1" || say "graphs_off leg did not complete"

# Q2 SITE: graphs ON, same tunables. Does the abort move earlier / elsewhere?
run_leg graphs_on "ARC_CAPTURE_STREAM=1 ARC_V4_CAPTURE_PROBE=1 ARC_CANDLE_ALLOC_CACHE=1 ARC_GRAPH_WARMUP=8 ARC_GRAPH_DEFERRED_PASSES=3" \
    || say "graphs_on leg did not complete"

# --------------------------------------------------------------------------
# 4. Verdict
# --------------------------------------------------------------------------
step "4/4 verdict"
if [ "$TUNABLES_ENGAGED" != "1" ]; then
    say "⚠️ MALLOC tunables did not engage; treat the heap findings as UNPROVEN."
fi
say "Q1 SCOPE — does the corruption need capture?"
# A leg that never served a token has zero diagnostics for reasons that have
# nothing to do with the heap. Report UNPROVEN, never 0.
verdict_for() {
    local name="$1" f="$LOGDIR/$1.server.log"
    if [ ! -s "$f" ]; then echo "UNPROVEN (no log)"; return; fi
    if ! grep -q "Loading model" "$f" 2>/dev/null && ! grep -q "Model loaded" "$f" 2>/dev/null; then
        echo "UNPROVEN (server never loaded)"; return
    fi
    local n tok
    n=$(grep -ciE 'malloc_consolidate|free\(\):|malloc\(\):|corrupted|double free' "$f" 2>/dev/null)
    n=${n:-0}
    tok=$(cat "$LOGDIR/$name.tokens" 2>/dev/null || echo 0)
    # A leg that loaded, ran, and then DIED WITH A GLIBC DIAGNOSTIC is a result,
    # not a vacuum — the zero tokens are the CONSEQUENCE of the finding, not
    # evidence the leg never happened. The first version of this check reported
    # exactly that case as "UNPROVEN (0 tokens)" and buried a positive result
    # under a guard written for the opposite failure. Distinguish "never ran"
    # from "ran and was killed by the thing we are measuring".
    if [ "$tok" = "0" ] && [ "$n" -eq 0 ] 2>/dev/null; then
        echo "UNPROVEN (server loaded but served 0 tokens, and no diagnostic to explain why)"
        return
    fi
    if [ "$tok" = "0" ]; then
        echo "$n glibc diagnostic(s); served 0 tokens BECAUSE it died — a result, not a vacuum"
        return
    fi
    echo "$n glibc diagnostic(s) over $tok tokens"
}
say "  graphs_off: $(verdict_for graphs_off)"
say "  graphs_on : $(verdict_for graphs_on)"
say "REMINDER: these are where corruption BECAME VISIBLE, never where it happened."
say "RESULT: COMPLETE"
