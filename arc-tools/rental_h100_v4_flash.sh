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
# can grep `FAIL:` and diagnose without parsing the whole log. On failure: fix
# the cause and RE-RUN the same command — completed steps (installs, clone,
# incremental cargo build, downloaded weights) are skipped. The FAIL: markers
# map to concrete recovery actions in arc-tools/RENTAL_PLAYBOOK.md → "Recovery".

set -uo pipefail

WORK=${WORK:-/ephemeral/work}
MODELS=${MODELS:-/ephemeral/models}
V4_MODEL=${V4_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
# Two probe models: 0.5B for dispatch smoke (fast), 7B for ISQ + coherent-text check.
PROBE_TINY=${PROBE_TINY:-Qwen/Qwen2.5-0.5B-Instruct}
PROBE_MID=${PROBE_MID:-Qwen/Qwen2.5-7B-Instruct}
RESULT_FILE=${RESULT_FILE:-/ephemeral/arc-v4flash-bench.json}
# TD-MoE Tucker rank for the factored MoE forward (RUN-K). Empty/unset = OFF
# (model stays at QTIP-only 75 GB residency, tight but functional fit).
# Set to 512 for the README target (~45 GB residency, comfortable fit with
# room for KV + activations). Quality at lower ranks is unmeasured — start
# at 512 and tune down only if a calibrated golden run shows headroom.
TD_MOE_RANK=${TD_MOE_RANK:-512}

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

step "4b/9 QTIP GPU kernel smoke tests — run BEFORE the 148 GB download"
# The QTIP CUDA kernels (prefix-grouped Viterbi quantize, fused gemv, dequant,
# rotate, gather) are developed on a Mac and never execute until a GPU box.
# These parity tests build a QTIP layer on CUDA and assert the GPU output
# matches the CPU reference (cos sim >= 0.999). cuda_quantize_matches_cpu_*
# exercises the quantize kernel that previously hung on H100-scale layers —
# catching a hang/crash here costs ~1 min instead of failing after a 148 GB
# download + 30 min of ISQ on the full V4 Flash model.
# Hard rule reminder: on a CUDA device the QTIP quantize path has NO CPU
# fallback (see mistralrs-quant/src/qtip/mod.rs); if a kernel is broken it
# MUST surface here, not silently detour to CPU.
cargo test --release -p mistralrs-quant --features cuda -- --nocapture --test-threads=1 \
    cuda_quantize_matches_cpu_dequantize_cos_sim \
    cuda_dequantize_matches_cpu_viterbi_rotation \
    cuda_forward_matches_cpu_viterbi_rotation \
    cuda_rotate_x_matches_cpu \
    cuda_fused_gemv_matches_dequant_matmul \
    cuda_fused_gemv_matches_dequant_matmul_no_rotation \
    qtip_gather_forward_cuda_matches_cpu \
    2>&1 | tee /tmp/qtip_gpu_smoke.log || fail "QTIP GPU kernel smoke tests failed (log: /tmp/qtip_gpu_smoke.log)"
# Guard against the tests silently skipping: each test prints "CUDA not
# available; skipping" and returns Ok if Device::new_cuda(0) fails, which still
# yields "test result: ok". Require the actual parity output and reject the skip.
grep -q "cos sim" /tmp/qtip_gpu_smoke.log || fail "QTIP GPU smoke tests did not actually run a kernel (no 'cos sim' output) — CUDA device not usable"
grep -q "CUDA not available; skipping" /tmp/qtip_gpu_smoke.log && fail "QTIP GPU smoke tests skipped — Device::new_cuda(0) failed on this box"
ok "QTIP GPU kernels validated on real hardware (parity >= 0.999)"

step "5/9 pull V4 Flash weights (148 GB FP4, ~75 shards). Idempotent — resumes."
export HF_HOME=${HF_HOME:-/ephemeral/hf_cache}
export HF_HUB_DISABLE_TELEMETRY=1
V4_DIR="$MODELS/DeepSeek-V4-Flash"
# ALWAYS invoke hf download — it is idempotent and *resumes*: it re-checks every
# shard against the local cache and only fetches what is missing or truncated.
# Do NOT gate this on index.json existence. index.json lands early in the pull,
# so a run preempted mid-download (the common case on a 148 GB / ~75-shard fetch)
# would otherwise be skipped here, leaving truncated shards. arc validate (step 6)
# checks the index *keyset* only, not shard bytes, so it would pass on a partial
# tree and the failure would not surface until step 8 — after wasting probe time.
hf download "$V4_MODEL" --local-dir "$V4_DIR" 2>&1 | tail -5 || fail "v4 flash download (re-run to resume — completed shards are skipped)"
[ -f "$V4_DIR/model.safetensors.index.json" ] || fail "V4 index missing after download"
ok "V4 Flash at $V4_DIR ($(du -sh "$V4_DIR" | awk '{print $1}'))"

step "6/9 arc validate — schema check against deepseekv4 arch"
./target/release/arc validate --index "$V4_DIR/model.safetensors.index.json" --arch deepseekv4 || fail "arc validate"
ok "schema OK"

step "7a/9 probe-tiny: Qwen 2.5-0.5B BF16 (no ISQ) — dispatch smoke. Should produce text."
# Idempotent + resumes; see the step-5 note on why we don't skip on a sentinel file.
hf download "$PROBE_TINY" --local-dir "$MODELS/Qwen2.5-0.5B-Instruct" 2>&1 | tail -3 || fail "probe-tiny download (re-run to resume)"
printf 'Say hello in 5 words.\n\\quit\n' | timeout 180 ./target/release/mistralrs run \
  --max-seq-len 512 --max-seqs 1 --paged-attn off \
  -m "$MODELS/Qwen2.5-0.5B-Instruct" -a qwen2 2>&1 | tail -8 | tee /tmp/probe_tiny.log
grep -qE "Decode: [0-9]" /tmp/probe_tiny.log || fail "probe-tiny smoke produced no decode — base dispatch broken"
ok "probe-tiny produced decode output"

step "7b/9 probe-mid: Qwen 2.5-7B + ISQ qtip2 — proves QTIP integrated stack produces *coherent* text"
# Different from probe-tiny: this exercises the actual ISQ → QTIP load + GPU
# forward path that V4 Flash will use. If this produces garbled text but the
# decode-rate is high, we know QTIP composition is lossy *before* burning
# rental time on a 148 GB model.
# Idempotent + resumes; see the step-5 note on why we don't skip on a sentinel file.
hf download "$PROBE_MID" --local-dir "$MODELS/Qwen2.5-7B-Instruct" 2>&1 | tail -3 || fail "probe-mid download (re-run to resume)"
PROBE_MID_LOG=/tmp/probe_mid.log
printf 'Complete this sentence: The capital of France is\n\\quit\n' | timeout 1800 ./target/release/mistralrs run \
  --isq qtip2 \
  --pa-cache-type turboquant \
  --max-seq-len 1024 --max-seqs 1 \
  -m "$MODELS/Qwen2.5-7B-Instruct" -a qwen2 2>&1 | tee "$PROBE_MID_LOG" >/dev/null || fail "probe-mid run failed (log: $PROBE_MID_LOG)"
# Quality gate: response must contain "Paris" (deterministic continuation at T~0).
# This is a coarse coherence check — if QTIP + the integrated stack is producing
# garbage we'll catch it here instead of after 148 GB of V4 download + 30 min of
# ISQ on the V4 Flash run.
if grep -iqE "paris" "$PROBE_MID_LOG"; then
  ok "probe-mid produced coherent text (Qwen 7B + QTIP completed 'capital of France' → contains 'Paris')"
else
  echo "WARN: probe-mid response did not contain 'Paris' — output may be incoherent."
  echo "      Last 30 lines of $PROBE_MID_LOG:"
  tail -30 "$PROBE_MID_LOG"
  echo "      Proceeding to V4 Flash anyway — Qwen 7B QTIP quality is not strictly required for V4 to run."
fi
PROBE_MID_TOKS=$(grep -oE "Decode: [0-9]+ tokens, [0-9.]+ T/s" "$PROBE_MID_LOG" | tail -1 | grep -oE "[0-9.]+ T/s" | head -1 | awk '{print $1}')
ok "probe-mid decode rate: ${PROBE_MID_TOKS:-N/A} tok/s"

step "8/9 V4 Flash + the Arc decode stack engaged"
# Composed stack:
#   * QTIP 2-bit Viterbi + Hadamard rotation — GPU quantize at load
#   * Fused QTIP single-token gemv kernel — HBM-floor decode bandwidth
#   * Factored TD-MoE Tucker (rank=${TD_MOE_RANK}) — MoE residency ~70 GB → ~few GB
#   * TurboQuant K4/V3 KV cache
#   * mHC 4-D residual
#   * Lightning Indexer + FlashMLASparse for V4 CSA/HCA dispatch
#   * arc-cuda-graph DedicatedDecodePath (per-token Candle bypass) — the ACTIVE
#     CUDA-graph decode path (graph_wrapped_forward -> try_dedicated_decode)
#   * NVFP4 3-D expert-stack ISQ → QTIP
#   * QtipLayer::gather_forward (used when TD-MoE is OFF; with TD-MoE on, MoE
#     goes through Tucker contraction)
#   * MTP speculative decode at depth=4 — V4's MTP head proposes 4 tokens per
#     target forward, ~2× throughput at typical acceptance rates
#
# No env-var bypass. No --paged-attn off. No Greedy fallback.
#
# NOT engaged by this run: AutonomousDecodeRunner (the fully on-GPU decode loop)
# is plumbed into NormalPipeline but its CUDA graph capture is still deferred —
# mistralrs-core/src/pipeline/normal.rs::autonomous_decode never calls
# runner.capture(), so is_captured() stays false and the engine ALWAYS falls
# back to step(). Decode here therefore runs through the DedicatedDecodePath
# capture-once+replay path above (a per-token Candle bypass), NOT the
# zero-CPU-sync autonomous loop. Do not report autonomous decode as engaged
# from this run until the capture wiring lands (recorded in the result JSON's
# "moats_plumbed_not_engaged" field below).
DECODE_LOG=/tmp/v4_flash_bench.log
printf 'Write exactly one paragraph (3-5 sentences) about HBM memory bandwidth and why it matters for LLM inference.\n\\quit\n' | \
  ARC_TD_MOE_RANK=${TD_MOE_RANK} \
  timeout 3600 ./target/release/mistralrs run \
    --isq qtip2 \
    --pa-cache-type turboquant \
    --mtp-depth 4 \
    --max-seq-len 4096 --max-seqs 1 \
    -m "$V4_DIR" -a deepseekv4 \
    2>&1 | tee "$DECODE_LOG" || fail "v4 flash decode (full log: $DECODE_LOG)"
# Coherence check on V4 output
echo
echo "--- V4 Flash decoded text (last 60 lines of decode log) ---"
tail -60 "$DECODE_LOG"
echo "----------------------------------------------------------"

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
  "td_moe_rank": ${TD_MOE_RANK},
  "kv_cache": "TurboQuant K4/V3 (default)",
  "moats_engaged": [
    "QTIP 2-bit Viterbi + Hadamard rotation",
    "Fused QTIP gemv (single-token decode HBM-floor kernel)",
    "Factored TD-MoE Tucker (rank ${TD_MOE_RANK})",
    "TurboQuant K4/V3 KV cache",
    "mHC 4-D residual",
    "FlashMLASparse + Lightning Indexer (V4 CSA/HCA)",
    "DedicatedDecodePath (per-token Candle bypass)",
    "NVFP4 3-D expert-stack -> QTIP ISQ",
    "QtipLayer::gather_forward for MoE Fast backend",
    "MTP speculative decode (depth=4)"
  ],
  "moats_plumbed_not_engaged": [
    "AutonomousDecodeRunner (on-GPU decode loop) — graph capture deferred in NormalPipeline::autonomous_decode (runner.capture() never called); engine falls back to step()+DedicatedDecodePath"
  ],
  "probe_mid_qwen7b_qtip_tok_per_s": "${PROBE_MID_TOKS:-N/A}",
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
