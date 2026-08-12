#!/usr/bin/env bash
# Perplexity self-ladder for V4-Flash on one H200 (RUN-161 quality battery).
#
# IMPORTANT: the perplexity example loads its OWN model instance — the serve
# process must be STOPPED before running this (else CUDA OOM).
#
# The ladder (qtip3/qtip4 do NOT exist — IsqType has only QtipBitshift2; the
# rungs below are the real available ladder):
#   qtip2_c1024   qtip2 UQFF (experts 2-bit trellis, attention native FP8,
#                 lm_head native — the headline Arc config), chunk 1024
#   qtip2_c64     same weights, chunk 64 < sliding_window 128 — isolates
#                 short-context behavior from the compressed long-ctx path
#   q2k_c1024     GGUF Q2K ISQ (2.56 bpw) — nearest-bpw round-number rung
#   q3k_c1024     GGUF Q3K ISQ (3.44 bpw) — higher-quality rung; ~120GB weights,
#                 TIGHT on 141GB: opt-in via RUN_Q3K=1, expect possible OOM
#   (a no-ISQ / native run does NOT fit: ~148GB checkpoint > 141GB HBM, and
#   experts dequant to BF16 ~560GB. Absolute KLD vs FP8 needs a bigger box.)
#
# --sinkhorn-ab : instead of the ladder, run the qtip2 rung twice on the mini
#   corpus with ARC_FUSED_SINKHORN unset/1 and diff per-chunk ppl strings —
#   identical strings == bit-identical f32 forward (see parse_ppl.py).
#
# Env:
#   MODEL_DIR (required) — local V4-Flash dir
#   UQFF0     (default: auto-glob $MODEL_DIR/uqff/qtip2*.uqff first shard)
#   PPL_FILE  (default: arc-tools/quality/data/wiki.test_small.raw, ~70K tok)
#   RUN_Q3K=1 to enable the tight q3k rung
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RES="$HERE/results"
mkdir -p "$RES"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local DeepSeek-V4-Flash directory}"
PPL_FILE="${PPL_FILE:-$HERE/data/wiki.test_small.raw}"
MINI_FILE="${MINI_FILE:-$HERE/data/wiki.test_mini.raw}"
BIN="$REPO/target/release/examples/perplexity"

fail() { echo "FAIL: $*" >&2; exit 1; }
[ -f "$PPL_FILE" ] || fail "$PPL_FILE missing — run fetch_data.sh"
if [ ! -x "$BIN" ]; then
  echo "perplexity example binary missing — building (one-time, ~minutes)..."
  (cd "$REPO" && cargo build --release -p mistralrs --example perplexity --features "cuda flash-attn") || fail "example build"
fi

if [ -z "${UQFF0:-}" ]; then
  UQFF0=$(ls "$MODEL_DIR"/uqff/qtip2*.uqff 2>/dev/null | sort | head -1 || true)
fi
[ -n "$UQFF0" ] || fail "no UQFF shard found (set UQFF0 or bake first: see runbook step 2)"

# refuse to start while a server is holding VRAM
if pgrep -f "mistralrs serve" >/dev/null 2>&1; then
  fail "a 'mistralrs serve' process is running — stop it first (CUDA OOM otherwise)"
fi

run_rung() { # $1 rung-name, $2 corpus file, remaining args appended
  local rung=$1; shift
  local corpus=$1; shift
  local log="$RES/ppl_${rung}.log"
  echo; echo ":::::: ppl rung: $rung ($corpus) ::::::"; date -u +%H:%M:%S
  "$BIN" -m "$MODEL_DIR" -a deepseekv4 -f "$corpus" "$@" 2>&1 | tee "$log"
  python3 "$HERE/parse_ppl.py" "$log" "$rung" "$RES/ppl_${rung}.json" \
      | tee -a "$RES/ppl_summary.txt" \
    || echo "WARN: rung $rung produced no final ppl (see $log)"
}

if [ "${1:-}" = "--sinkhorn-ab" ]; then
  [ -f "$MINI_FILE" ] || fail "$MINI_FILE missing — run fetch_data.sh"
  echo ":::::: sinkhorn bit-identity: qtip2 on mini corpus, gate OFF then ON ::::::"
  env -u ARC_FUSED_SINKHORN "$BIN" -m "$MODEL_DIR" -a deepseekv4 -f "$MINI_FILE" -u "$UQFF0" --chunk-size 1024 2>&1 | tee "$RES/ppl_sink_off.log"
  ARC_FUSED_SINKHORN=1     "$BIN" -m "$MODEL_DIR" -a deepseekv4 -f "$MINI_FILE" -u "$UQFF0" --chunk-size 1024 2>&1 | tee "$RES/ppl_sink_on.log"
  python3 "$HERE/parse_ppl.py" "$RES/ppl_sink_off.log" sink_off "$RES/ppl_sink_off.json" || fail "gate-off run failed"
  python3 "$HERE/parse_ppl.py" "$RES/ppl_sink_on.log"  sink_on  "$RES/ppl_sink_on.json"  || fail "gate-on run failed"
  python3 - "$RES/ppl_sink_off.json" "$RES/ppl_sink_on.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
pa = [c["ppl"] for c in a["chunks"]]; pb = [c["ppl"] for c in b["chunks"]]
ident = pa == pb and a["ppl_mean"] == b["ppl_mean"]
print(f"SINKHORN-PPL-IDENTITY: {'PASS (bit-identical per-chunk ppl)' if ident else 'FAIL'}")
if not ident:
    for i, (x, y) in enumerate(zip(pa, pb)):
        if x != y:
            print(f"  chunk {i}: off={x} on={y}")
json.dump({"bit_identical": ident, "off": pa, "on": pb},
          open(sys.argv[1].replace("sink_off", "sink_verdict"), "w"), indent=2)
sys.exit(0 if ident else 1)
EOF
  exit $?
fi

# ---------------- the ladder ----------------
C64_FILE="${C64_FILE:-$HERE/data/wiki.test_c64slice.raw}"
run_rung qtip2_c1024 "$PPL_FILE" -u "$UQFF0" --chunk-size 1024
run_rung qtip2_c64   "$C64_FILE" -u "$UQFF0" --chunk-size 64
run_rung q2k_c1024   "$PPL_FILE" -i q2k --chunk-size 1024
if [ "${RUN_Q3K:-0}" = "1" ]; then
  run_rung q3k_c1024 "$PPL_FILE" -i q3k --chunk-size 1024 || echo "WARN: q3k rung failed (expected fit risk on 141GB)"
else
  echo "SKIP: q3k rung (enable with RUN_Q3K=1; ~120GB weights, OOM risk)"
fi

echo; echo "== PPL LADDER SUMMARY =="; cat "$RES/ppl_summary.txt" 2>/dev/null || true
