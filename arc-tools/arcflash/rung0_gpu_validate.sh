#!/usr/bin/env bash
# ArcFlash rung 0 — GPU validation. Run by main on a live box (D15: agents do
# not call `runcrate`).
#
# WHAT THIS PROVES, and what it deliberately does not:
#   * That the CUDA-gated Rust in this branch COMPILES. A macOS `cargo check`
#     does not type-check `#[cfg(feature = "cuda")]` bodies; one PR shipped 15
#     such errors. Section 1 is that gate and it is not skippable.
#   * That ArcFlash/Tile ENGAGES on V4 and does NOT engage on a GQA model
#     (D18: a log line present in the treatment arm and absent in the control).
#     Both directions are checked. One direction alone proves nothing.
#   * The long-prompt numbers: wall time and peak VRAM at ~200 and ~1055 words,
#     the two prompt lengths that measured 386 s and server-death before.
#
#   It does NOT prove anything about a fused CuTeDSL d=512 kernel. That kernel
#   does not exist yet; rung 0 is the Rust path that removes the GQA expansion
#   and bounds the score matrix.
#
# USAGE (main, on the box):
#   export ARC_V4_MODEL=<path-or-repo-id for DeepSeek-V4 Flash>
#   export ARC_GQA_MODEL=<path-or-repo-id for any Llama/Qwen-class GQA model>
#   setsid nohup bash arc-tools/arcflash/rung0_gpu_validate.sh \
#     < /dev/null > /root/arcflash_rung0.log 2>&1 &
#
#   Status file: /root/arcflash_rung0.status  (one line: RUNNING|PASS|FAIL:<why>)
#   Results:     /root/arcflash_rung0.json

set -uo pipefail

STATUS=/root/arcflash_rung0.status
RESULTS=/root/arcflash_rung0.json
WORK="${ARC_WORK:-$(pwd)}"

say()  { echo "[arcflash] $*"; }
fail() { echo "FAIL:$*" > "$STATUS"; say "FAILED: $*"; exit 1; }

echo "RUNNING" > "$STATUS"
say "worktree: $WORK"

# --- 0. Preflight ------------------------------------------------------------
# Gated on the staged preflight rather than re-deriving box state here.
PREFLIGHT=/root/arc-tools/gpu_box_preflight.sh
if [[ -x "$PREFLIGHT" ]]; then
  say "running preflight"
  bash "$PREFLIGHT" || fail "preflight rejected the box"
else
  say "WARNING: $PREFLIGHT absent; continuing without it (record this in the log)"
fi

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader \
  || fail "no GPU visible"

: "${ARC_V4_MODEL:?set ARC_V4_MODEL to the DeepSeek-V4 Flash model}"
: "${ARC_GQA_MODEL:?set ARC_GQA_MODEL to a Llama/Qwen-class GQA model}"

cd "$WORK" || fail "cannot enter $WORK"

# --- 1. THE COMPILE GATE -----------------------------------------------------
# Non-negotiable: this is the check a macOS host cannot perform.
say "section 1: compiling CUDA-gated Rust"
cargo check -p mistralrs-core --tests --features cuda 2>&1 | tail -40
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "cargo check --features cuda did not compile"

say "section 1b: unit tests under the cuda feature"
cargo test -p mistralrs-core --lib --features cuda attention:: 2>&1 | tail -25
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "attention unit tests failed under --features cuda"

say "section 2: release build"
cargo build --release --features "cuda flash-attn" 2>&1 | tail -20
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "release build failed"
BIN=./target/release/mistralrs
[[ -x "$BIN" ]] || BIN=./target/release/arc
[[ -x "$BIN" ]] || fail "no CLI binary at target/release/{mistralrs,arc}"

# --- 3. Prompts --------------------------------------------------------------
mkdir -p /tmp/arcflash
python3 - <<'PY'
# ~200 words and ~1055 words: the two lengths that measured 386 s and OOM.
base = ("The engine schedules requests across a fleet of accelerators, admitting "
        "work when memory allows and coalescing short sequences into shared "
        "batches so the decode loop stays saturated. ")
w = base.split()
def take(n):
    out = []
    while len(out) < n:
        out.extend(w)
    return " ".join(out[:n])
open("/tmp/arcflash/p200.txt", "w").write(take(200) + "\n\nSummarize the above in one sentence.")
open("/tmp/arcflash/p1055.txt", "w").write(take(1055) + "\n\nSummarize the above in one sentence.")
print("prompts written")
PY

# --- 4. One measured run -----------------------------------------------------
# Samples VRAM in the background so peak is observed, not inferred from a
# single end-of-run reading.
run_one() {
  local tag="$1" model="$2" prompt_file="$3" logf="/tmp/arcflash/${tag}.log"
  local vramf="/tmp/arcflash/${tag}.vram"

  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vramf"
      sleep 0.5
    done ) & local sampler=$!

  local t0 t1
  t0=$(date +%s.%N)
  RUST_LOG=info "$BIN" run -m "$model" --prompt "$(cat "$prompt_file")" -n 8 \
    > "$logf" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  kill "$sampler" 2>/dev/null

  local peak
  peak=$(sort -n "$vramf" 2>/dev/null | tail -1)
  echo "${tag} rc=${rc} secs=$(echo "$t1 - $t0" | bc) peak_vram_mib=${peak:-unknown}"
}

say "section 4: V4 treatment arm"
V4_200=$(run_one v4_200 "$ARC_V4_MODEL" /tmp/arcflash/p200.txt)
say "$V4_200"
V4_1055=$(run_one v4_1055 "$ARC_V4_MODEL" /tmp/arcflash/p1055.txt)
say "$V4_1055"

say "section 4b: GQA control arm"
GQA_200=$(run_one gqa_200 "$ARC_GQA_MODEL" /tmp/arcflash/p200.txt)
say "$GQA_200"

# --- 5. D18: engagement, both directions ------------------------------------
say "section 5: D18 engagement"
v4_tile=$(grep -c "first dispatch on arcflash-tile" /tmp/arcflash/v4_1055.log || true)
v4_vend=$(grep -c "first dispatch on vendor-flash"  /tmp/arcflash/v4_1055.log || true)
gqa_tile=$(grep -c "first dispatch on arcflash-tile" /tmp/arcflash/gqa_200.log || true)
gqa_vend=$(grep -c "first dispatch on vendor-flash"  /tmp/arcflash/gqa_200.log || true)

say "V4:  arcflash-tile=$v4_tile vendor-flash=$v4_vend"
say "GQA: arcflash-tile=$gqa_tile vendor-flash=$gqa_vend"

D18_OK=1
[[ "$v4_tile"  -ge 1 ]] || { say "D18 FAIL: V4 did not engage ArcFlash/Tile"; D18_OK=0; }
[[ "$gqa_tile" -eq 0 ]] || { say "D18 FAIL: the GQA control was absorbed by ArcFlash/Tile"; D18_OK=0; }
[[ "$gqa_vend" -ge 1 ]] || { say "D18 NOTE: GQA control did not log vendor-flash (check the build's flash features)"; }

cat > "$RESULTS" <<JSON
{
  "rung": 0,
  "system": "ArcFlash/Tile",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
  "v4_200":  "$V4_200",
  "v4_1055": "$V4_1055",
  "gqa_200": "$GQA_200",
  "d18": {
    "v4_arcflash_tile": $v4_tile,
    "v4_vendor_flash": $v4_vend,
    "gqa_arcflash_tile": $gqa_tile,
    "gqa_vendor_flash": $gqa_vend,
    "both_directions_ok": $D18_OK
  }
}
JSON

[[ "$D18_OK" -eq 1 ]] || fail "D18 engagement check failed (see $RESULTS)"
grep -qi "out of memory\|CUDA error" /tmp/arcflash/v4_1055.log && fail "V4 still OOMs at ~1055 words"

echo "PASS" > "$STATUS"
say "PASS — results in $RESULTS"
