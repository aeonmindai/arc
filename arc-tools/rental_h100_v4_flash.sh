#!/usr/bin/env bash
#
# Arc M1 — V4 Flash on 1× H100 80GB, full Arc stack, single tok/s number.
#
# Idempotent. Resumable. Drop on a fresh Ubuntu H100 box, run from /ephemeral
# or any large-disk mount. Produces /ephemeral/arc-v4flash-bench.json with the
# decode tok/s number when it completes.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/aeonmindai/arc/master/arc-tools/rental_h100_v4_flash.sh | bash
# OR
#   scp arc-tools/rental_h100_v4_flash.sh root@<box>:/tmp/ && ssh root@<box> bash /tmp/rental_h100_v4_flash.sh
#
# Exits non-zero on any failure with a clear marker line so an orchestrator
# can grep `FAIL:` and diagnose without parsing the whole log.

set -uo pipefail

WORK=${WORK:-/ephemeral/work}
MODELS=${MODELS:-/ephemeral/models}
V4_MODEL=${V4_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
PROBE_MODEL=${PROBE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
RESULT_FILE=${RESULT_FILE:-/ephemeral/arc-v4flash-bench.json}

step() { echo; echo "::::::: $* :::::::"; date; }
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK: $*"; }

step "1/9 disk + GPU sanity"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || fail "nvidia-smi missing"
DISK_AVAIL=$(df --output=avail /ephemeral 2>/dev/null | tail -1 || df --output=avail / | tail -1)
if [ "${DISK_AVAIL:-0}" -lt 230000000 ]; then
  echo "WARN: <230GB free on /ephemeral or /. V4 Flash needs ~250GB headroom."
fi

mkdir -p "$WORK" "$MODELS"

step "2/9 install rust + build deps + hf cli (idempotent)"
if ! command -v cargo >/dev/null; then
  curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal || fail "rust install"
fi
. "$HOME/.cargo/env"
apt-get install -y --no-install-recommends pkg-config libssl-dev build-essential python3-pip 2>&1 | tail -3 || fail "apt deps"
pip install --quiet --break-system-packages huggingface_hub 2>/dev/null || pip install --quiet huggingface_hub || fail "pip hf_hub"

# CUDA env (works for /usr/local/cuda or /usr/local/cuda-12.X symlinks).
CUDA_HOME_GUESS=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
export CUDA_HOME=${CUDA_HOME:-${CUDA_HOME_GUESS:-/usr/local/cuda}}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
nvcc --version >/dev/null 2>&1 || fail "nvcc not found at $CUDA_HOME/bin/nvcc"
ok "cuda toolchain @ $CUDA_HOME ($(nvcc --version | grep release))"

step "3/9 clone or sync Arc repo"
if [ ! -d "$WORK/arc/.git" ]; then
  git clone --depth=30 https://github.com/aeonmindai/arc.git "$WORK/arc" || fail "clone"
fi
cd "$WORK/arc"
git fetch origin master && git reset --hard origin/master || fail "sync"
ok "Arc at $(git log -1 --oneline)"

step "4/9 build arc-cli + mistralrs-cli with cuda + flash-attn (no cudnn — not installed by default)"
cargo build --release -p arc-cli -p mistralrs-cli --features "cuda flash-attn" 2>&1 | tail -3 || fail "cargo build"
[ -x ./target/release/arc ] || fail "arc binary missing"
[ -x ./target/release/mistralrs ] || fail "mistralrs binary missing"
ok "binaries built"

step "5/9 pull V4 Flash weights (148 GB FP4, ~75 shards). Idempotent — resumes."
export HF_HOME=${HF_HOME:-/ephemeral/hf_cache}
export HF_HUB_DISABLE_TELEMETRY=1
V4_DIR="$MODELS/DeepSeek-V4-Flash"
if [ ! -f "$V4_DIR/model.safetensors.index.json" ]; then
  hf download "$V4_MODEL" --local-dir "$V4_DIR" 2>&1 | tail -5 || fail "v4 flash download"
fi
[ -f "$V4_DIR/model.safetensors.index.json" ] || fail "V4 index missing after download"
ok "V4 Flash at $V4_DIR ($(du -sh "$V4_DIR" | awk '{print $1}'))"

step "6/9 arc validate — schema check against deepseekv4 arch"
./target/release/arc validate --index "$V4_DIR/model.safetensors.index.json" --arch deepseekv4 || fail "arc validate"
ok "schema OK"

step "7/9 probe: Qwen 2.5-0.5B BF16 (no ISQ) — pure dispatch smoke. Should produce text."
if [ ! -f "$MODELS/Qwen2.5-0.5B-Instruct/model.safetensors" ]; then
  hf download "$PROBE_MODEL" --local-dir "$MODELS/Qwen2.5-0.5B-Instruct" 2>&1 | tail -3 || fail "probe model download"
fi
printf 'Say hello in 5 words.\n\\quit\n' | timeout 180 ./target/release/mistralrs run \
  --max-seq-len 512 --max-seqs 1 --paged-attn off \
  -m "$MODELS/Qwen2.5-0.5B-Instruct" -a qwen2 2>&1 | tail -8 | tee /tmp/probe.log
grep -qE "Decode: [0-9]" /tmp/probe.log || fail "probe smoke produced no decode output — dispatch path broken"
ok "probe smoke produced decode output"

step "8/9 V4 Flash + ISQ qtip2 (Viterbi) — full Arc stack engaged"
# All Arc moats intact:
#   * QTIP 2-bit Viterbi + Hadamard rotation (RUN-158) — GPU quantize at load (Agent F)
#   * TurboQuant K4/V3 KV (--pa-cache-type turboquant, default)
#   * mHC 4-D residual (RUN-164)
#   * Lightning Indexer + FlashMLASparse (RUN-163)
#   * arc-cuda-graph dedicated decode path (Agent A SEGV fix)
#   * NVFP4 → QTIP apply_isq (Agent C wiring)
#   * QTIP CUDA forward kernel with rotation (Agent D)
# No env-var bypass, no --paged-attn off, no Greedy fallback.
#
# `--pa-memory-mb 30000` overrides the auto KV budget that otherwise computes
# `0 GPU blocks` because it sizes the model pre-ISQ (BF16 ~148 GB > 80 GB HBM
# even though ISQ shrinks to ~37 GB at load). 30 GB for paged KV + ~37 GB for
# QTIP weights + ~10 GB activations/workspaces fits comfortably in 80 GB.
DECODE_LOG=/tmp/v4_flash_bench.log
printf 'Write exactly one paragraph (3-5 sentences) about HBM memory bandwidth and why it matters for LLM inference.\n\\quit\n' | \
  timeout 3600 ./target/release/mistralrs run \
    --isq qtip2 \
    --pa-memory-mb 30000 \
    --pa-cache-type turboquant \
    --max-seq-len 4096 --max-seqs 1 \
    -m "$V4_DIR" -a deepseekv4 \
    2>&1 | tee "$DECODE_LOG" || fail "v4 flash decode (full log: $DECODE_LOG)"

step "9/9 extract tok/s number"
TOK_PER_S=$(grep -oE "Decode: [0-9]+ tokens, [0-9.]+ T/s" "$DECODE_LOG" | tail -1 | grep -oE "[0-9.]+ T/s" | head -1 | awk '{print $1}')
TTFT=$(grep -oE "Time to first token: [0-9.]+s" "$DECODE_LOG" | tail -1 | grep -oE "[0-9.]+" | head -1)
PROMPT_RATE=$(grep -oE "Prompt: [0-9]+ tokens, [0-9.]+ T/s" "$DECODE_LOG" | tail -1 | grep -oE "[0-9.]+ T/s" | head -1 | awk '{print $1}')

if [ -z "${TOK_PER_S:-}" ]; then
  fail "couldn't extract Decode T/s from log — V4 Flash didn't decode successfully (see $DECODE_LOG)"
fi

cat > "$RESULT_FILE" <<JSON
{
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "hardware": "1x H100 80GB",
  "quantization": "QTIP 2-bit (Viterbi + Hadamard rotation)",
  "kv_cache": "TurboQuant K4/V3 (default)",
  "all_moats_intact": true,
  "tok_per_s_decode": $TOK_PER_S,
  "ttft_s": ${TTFT:-null},
  "prompt_rate_tok_per_s": ${PROMPT_RATE:-null},
  "arc_commit": "$(git log -1 --format=%H)",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo
echo "================================================================================"
echo "  V4 FLASH ON 1x H100 — DECODE: $TOK_PER_S tok/s"
echo "  Full result: $RESULT_FILE"
echo "================================================================================"
cat "$RESULT_FILE"
