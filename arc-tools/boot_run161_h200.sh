#!/usr/bin/env bash
# RUN-161 V4-Flash bootstrap for a fresh H200 box (chat-quality validation).
#
# Differs from rental_h100_v4_flash.sh (stale): clones the asmit383/candle
# `run-161-cuda-graph-capture` fork as a SIBLING ../candle (arc uses path deps,
# Cargo.toml: candle-core = { path = "../candle/candle-core" }) and checks out
# arc branch RUN-161 (faithful compressor + 2-bit-attn gate + decode-path fix).
#
# Idempotent + resumable. Launch detached and poll the log:
#   nohup bash boot_run161_h200.sh > /root/boot.log 2>&1 & disown
#   then poll /root/boot.log for STEP/OK/FAIL markers.
set -uo pipefail

# Pick the largest-free mount for WORK/MODELS (provider VM may not have /ephemeral).
BIGMOUNT=/root
for m in /ephemeral /workspace /mnt /root; do
  [ -d "$m" ] && BIGMOUNT=$m && break
done
WORK=${WORK:-$BIGMOUNT/work}
MODELS=${MODELS:-$BIGMOUNT/models}
V4_MODEL=${V4_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
CANDLE_REPO=https://github.com/asmit383/candle.git
CANDLE_BRANCH=run-161-cuda-graph-capture
ARC_REPO=https://github.com/aeonmindai/arc.git
ARC_BRANCH=RUN-161

step() { echo; echo ":::::: $* ::::::"; date -u +%H:%M:%S; }
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK: $*"; }

step "0 env: WORK=$WORK MODELS=$MODELS"
mkdir -p "$WORK" "$MODELS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || fail "nvidia-smi missing"
df -h "$BIGMOUNT" | tail -1

step "1 rust + build deps + hf cli (idempotent)"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal || fail "rust install"
fi
. "$HOME/.cargo/env"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y 2>&1 | tail -1
apt-get install -y --no-install-recommends pkg-config libssl-dev build-essential git python3-pip 2>&1 | tail -2 || fail "apt deps"
pip install --quiet --break-system-packages "huggingface_hub[cli]" 2>/dev/null || pip install --quiet "huggingface_hub[cli]" || fail "pip hf"

CUDA_HOME_GUESS=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
export CUDA_HOME=${CUDA_HOME:-${CUDA_HOME_GUESS:-/usr/local/cuda}}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
nvcc --version >/dev/null 2>&1 || fail "nvcc not found at $CUDA_HOME"
ok "cuda @ $CUDA_HOME ($(nvcc --version | grep release))"

step "2 clone candle fork SIBLING ($CANDLE_BRANCH)"
if [ ! -d "$WORK/candle/.git" ]; then
  git clone --depth=1 -b "$CANDLE_BRANCH" "$CANDLE_REPO" "$WORK/candle" || fail "candle clone"
fi
ok "candle at $(cd "$WORK/candle" && git log -1 --oneline)"

step "3 clone arc ($ARC_BRANCH) as sibling of candle"
if [ ! -d "$WORK/arc/.git" ]; then
  git clone -b "$ARC_BRANCH" "$ARC_REPO" "$WORK/arc" || fail "arc clone"
fi
cd "$WORK/arc"
git fetch origin "$ARC_BRANCH" && git checkout "$ARC_BRANCH" && git reset --hard "origin/$ARC_BRANCH" || fail "arc sync"
ok "arc at $(git log -1 --oneline)"

step "4 V4-Flash download (~148GB, public, resumes) — START EARLY in background"
export HF_HOME=${HF_HOME:-$MODELS/hf_cache}
export HF_HUB_DISABLE_TELEMETRY=1
V4_DIR="$MODELS/DeepSeek-V4-Flash"
mkdir -p "$V4_DIR"
# Background the download so the cargo build runs concurrently. Marker file
# signals completion; poll for it before serving.
if [ ! -f "$V4_DIR/.download_done" ]; then
  ( hf download "$V4_MODEL" --local-dir "$V4_DIR" \
      && [ -f "$V4_DIR/model.safetensors.index.json" ] \
      && touch "$V4_DIR/.download_done" \
      && echo "OK: V4 download done ($(du -sh "$V4_DIR" | awk '{print $1}'))" \
      || echo "FAIL: V4 download" ) > "$WORK/dl.log" 2>&1 &
  echo "download PID $! → $WORK/dl.log"
else
  ok "V4 already downloaded"
fi

step "5 build arc-cli + mistralrs-cli (cuda flash-attn) — concurrent with download"
cargo build --release -p arc-cli -p mistralrs-cli --features "cuda flash-attn" 2>&1 | tail -4 || fail "cargo build"
[ -x ./target/release/mistralrs ] || fail "mistralrs binary missing"
ok "binaries built: $(ls -1 target/release/{arc,mistralrs} 2>/dev/null | tr '\n' ' ')"

step "6 wait for V4 download to finish"
for i in $(seq 1 240); do
  [ -f "$V4_DIR/.download_done" ] && break
  sleep 30
done
[ -f "$V4_DIR/.download_done" ] || fail "V4 download did not finish in time (see $WORK/dl.log)"
ok "V4 ready at $V4_DIR"

step "7 schema validate"
./target/release/arc validate --index "$V4_DIR/model.safetensors.index.json" --arch deepseekv4 2>&1 | tail -3 || echo "WARN: arc validate non-zero (continuing)"

echo
echo "BOOTSTRAP_COMPLETE — ready to serve. V4_DIR=$V4_DIR"
echo "Serve (run detached separately):"
echo "  ./target/release/mistralrs serve -p 1234 --isq qtip2 --prefix-cache-n 0 -m $V4_DIR -a deepseekv4"
