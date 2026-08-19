#!/bin/bash
# arc-graph-probe — bootstrap a bare H200 box and run the capture sweep.
#
# One-shot and idempotent-ish: safe to re-run, skips what is already done.
# Every environment trap below was MEASURED on the sibling box, not guessed.
#
#   setsid nohup /root/arcgraph_box_bootstrap.sh < /dev/null \
#       > /root/logs/arcgraph/boot.log 2>&1 &
#
# Progress: tail /root/logs/arcgraph/STATUS.txt  (append-only, timestamped)
# Ends with exactly one of: RESULT: COMPLETE | RESULT: UNANSWERED
set -u

LOGDIR=/root/logs/arcgraph
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.txt"
REPO=/root/arc
BRANCH="${BRANCH:-arcgraph/capture-truth}"
SRC=/root/models/v4-src
UQFF_DIR=/root/models/v4-uqff
UQFF="$UQFF_DIR/qtip2b-0.uqff"
CUDA_HOME=/usr/local/cuda-13.1

say()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }
step() { say "=== $* ==="; }
env_fail() { say "ENV_FAIL: $*"; say "RESULT: UNANSWERED"; exit 2; }

say "bootstrap start on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
say "disk: $(df -h / | awk 'NR==2{print $4" free of "$2}')"

# ---------------------------------------------------------------------------
# 1. System deps
# ---------------------------------------------------------------------------
step "1/7 apt deps"
export DEBIAN_FRONTEND=noninteractive
if [ ! -f /root/.arc_apt_done ]; then
    apt-get update -qq >>"$LOGDIR/boot.apt.log" 2>&1
    apt-get install -y -qq build-essential git curl pkg-config libssl-dev python3-pip \
        >>"$LOGDIR/boot.apt.log" 2>&1 || env_fail "apt install failed; see $LOGDIR/boot.apt.log"
    touch /root/.arc_apt_done
fi
say "base deps ok"

# ---------------------------------------------------------------------------
# 2. THE CUDA-COMPAT TRAP — the one that produces WRONG NUMBERS WITH A CLEAN EXIT
# ---------------------------------------------------------------------------
# Driver 580.173.02 caps at CUDA 13.0; the only toolkit here is 13.1. And
# candle-kernels/build.rs calls build_ptx() unconditionally — CUDA_COMPUTE_CAP
# cannot stop it — so the driver is always asked to JIT PTX from a newer ISA.
# Without the compat libs that JIT fails into cudaGetLastError(), which nothing
# checks, and the run completes normally over unwritten buffers. An ungated box
# reports a BETTER number, not a worse one, which is why this is asserted rather
# than assumed.
step "2/7 cuda-compat + PTX-JIT assertion"
if ! dpkg -l | grep -q cuda-compat-13-1; then
    apt-get install -y -qq cuda-compat-13-1 >>"$LOGDIR/boot.apt.log" 2>&1 \
        || env_fail "cuda-compat-13-1 install failed — REFUSING to measure. \
Without it PTX-JIT fails silently and the numbers are plausible garbage."
fi
[ -d /usr/local/cuda/compat ] || env_fail "/usr/local/cuda/compat absent after install"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
[ -x "$CUDA_HOME/bin/nvcc" ] || env_fail "nvcc not at $CUDA_HOME/bin/nvcc"
say "nvcc: $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
say "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# The assertion itself. Compiled to PTX ONLY (-arch=compute_90 -code=compute_90)
# so the driver MUST JIT it — which is precisely the path that breaks without
# the compat libs. A cubin would sail through and prove nothing.
cat > /tmp/ptxgate.cu <<'CUDA'
#include <cstdio>
__global__ void k(int* out) { *out = 42; }
int main() {
    int* d; int h = 0;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) { printf("MALLOC_FAIL\n"); return 1; }
    k<<<1,1>>>(d);
    cudaError_t le = cudaGetLastError();          // the error the silent path drops
    if (le != cudaSuccess) { printf("LAUNCH_FAIL %s\n", cudaGetErrorString(le)); return 1; }
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("SYNC_FAIL\n"); return 1; }
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h);
    return 0;
}
CUDA
"$CUDA_HOME/bin/nvcc" -arch=compute_90 -code=compute_90 -o /tmp/ptxgate /tmp/ptxgate.cu \
    >>"$LOGDIR/boot.ptx.log" 2>&1 || env_fail "PTX gate failed to COMPILE; see $LOGDIR/boot.ptx.log"
PTXOUT=$(/tmp/ptxgate 2>&1)
if [ "$PTXOUT" != "42" ]; then
    env_fail "🔴 PTX-JIT GATE FAILED — kernel returned '$PTXOUT', expected 42. \
This is the trap that yields wrong numbers with a clean exit. Do NOT measure on this box."
fi
say "✅ PTX-JIT gate PASSED (kernel returned 42 through a JIT-only path)"
# Receipt for the probe: gpu_box_preflight.sh is not on master, so this is the
# gate of record on this box. The probe refuses to run without one or the other.
echo "PTX-JIT JIT-only kernel returned 42 at $(date -u +%Y-%m-%dT%H:%M:%SZ); \
nvcc -arch=compute_90 -code=compute_90; cudaGetLastError checked; \
driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); \
cuda-compat-13-1 installed; LD_LIBRARY_PATH=$LD_LIBRARY_PATH" > "$LOGDIR/PTX_GATE_PASSED"

# ---------------------------------------------------------------------------
# 3. Python (pip 22.0.2 ships with no numpy and cannot resolve modern wheels)
# ---------------------------------------------------------------------------
step "3/7 python"
if [ ! -f /root/.arc_pip_done ]; then
    python3 -m pip install -q --upgrade pip setuptools wheel >>"$LOGDIR/boot.pip.log" 2>&1 \
        || env_fail "pip upgrade failed"
    python3 -m pip install -q numpy >>"$LOGDIR/boot.pip.log" 2>&1 || env_fail "numpy install failed"
    python3 -m pip install -q "huggingface_hub[cli,hf_transfer]" >>"$LOGDIR/boot.pip.log" 2>&1 \
        || env_fail "huggingface_hub install failed"
    touch /root/.arc_pip_done
fi
say "python ok: $(python3 -c 'import numpy; print("numpy", numpy.__version__)')"

# ---------------------------------------------------------------------------
# 4. Rust
# ---------------------------------------------------------------------------
step "4/7 rust"
if [ ! -x /root/.cargo/bin/cargo ]; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal \
        >>"$LOGDIR/boot.rust.log" 2>&1 || env_fail "rustup failed"
fi
export PATH="/root/.cargo/bin:$PATH"
command -v cargo >/dev/null || env_fail "cargo still not on PATH"
say "cargo: $(cargo --version)"

# ---------------------------------------------------------------------------
# 5. Repo
# ---------------------------------------------------------------------------
step "5/7 repo @ $BRANCH"
if [ ! -d "$REPO/.git" ]; then
    git clone -q https://github.com/aeonmindai/arc.git "$REPO" >>"$LOGDIR/boot.git.log" 2>&1 \
        || env_fail "git clone failed"
fi
git -C "$REPO" fetch -q origin "$BRANCH" >>"$LOGDIR/boot.git.log" 2>&1 || env_fail "git fetch $BRANCH failed"
git -C "$REPO" checkout -q --detach FETCH_HEAD >>"$LOGDIR/boot.git.log" 2>&1 || env_fail "git checkout failed"
say "HEAD = $(git -C "$REPO" log --oneline -1)"

# ---------------------------------------------------------------------------
# 6. Model download, IN PARALLEL with the build (the two never contend)
# ---------------------------------------------------------------------------
step "6/7 model download (background) + build (foreground)"
[ -f /root/.hf_token ] || env_fail "/root/.hf_token absent — cannot fetch the gated artifacts"
chmod 600 /root/.hf_token
# Read it into the env WITHOUT ever echoing it, and keep it out of `ps`.
HF_TOKEN=$(cat /root/.hf_token); export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$SRC" "$UQFF_DIR"

dl() { # $1 repo, $2 dest, $3 tag, $4.. extra args
    local repo="$1" dest="$2" tag="$3"; shift 3
    if hf download "$repo" --local-dir "$dest" "$@" >>"$LOGDIR/dl.$tag.log" 2>&1; then
        echo OK > "$LOGDIR/dl.$tag.done"
    else
        echo FAIL > "$LOGDIR/dl.$tag.done"
    fi
}

# MEASURED 2026-08-17: excluding *.safetensors does NOT work, even with
# `--from-uqff`. The loader enumerates model files before it ever consults the
# UQFF and dies with:
#   Error: Expected file with extension one of .safetensors, .pth, .pt, .bin.
# The 149 GB is not optional. (It failed loudly at model load rather than
# silently mis-loading, which is the one good thing about the attempt.)
if [ ! -f "$SRC/config.json" ] || ! ls "$SRC"/*.safetensors >/dev/null 2>&1; then
    dl deepseek-ai/DeepSeek-V4-Flash "$SRC" src &
    DL_SRC=$!
else
    echo OK > "$LOGDIR/dl.src.done"; DL_SRC=""
fi
if [ ! -f "$UQFF" ]; then
    dl aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b "$UQFF_DIR" uqff &
    DL_UQFF=$!
else
    echo OK > "$LOGDIR/dl.uqff.done"; DL_UQFF=""
fi
say "downloads started in background"

# Build while they stream. NEVER cudnn: -62% decode on V4 (CLAUDE.md).
FEATURES="cuda flash-attn"
case "$FEATURES" in *cudnn*) env_fail "cudnn present; banned";; esac
cd "$REPO" || env_fail "cannot enter $REPO"
# shellcheck disable=SC1090,SC1091
source "$REPO/arc-tools/lib/build_and_verify.sh" || env_fail "build_and_verify.sh missing"
say "building -p mistralrs-cli --features \"$FEATURES\" (cold build, ~11 min)"
if ! arc_build_and_verify --package mistralrs-cli --bin mistralrs --features "$FEATURES" \
        --marker "ARCGRAPH STATUS" --repo "$REPO" --log "$LOGDIR/boot.build.log"; then
    say "BUILD/VERIFY FAILED — first errors:"
    grep -m 30 -A 5 -E "^error" "$LOGDIR/boot.build.log" 2>/dev/null | tee -a "$STATUS"
    env_fail "build failed; full log $LOGDIR/boot.build.log"
fi
say "✅ build ok, freshness marker present: $ARC_VERIFIED_BIN"
say "   built rev = $ARC_VERIFIED_REV"

[ -n "$DL_SRC" ]  && { say "waiting on source download";  wait "$DL_SRC"  || true; }
[ -n "$DL_UQFF" ] && { say "waiting on UQFF download";    wait "$DL_UQFF" || true; }
for t in src uqff; do
    r=$(cat "$LOGDIR/dl.$t.done" 2>/dev/null || echo MISSING)
    [ "$r" = "OK" ] || env_fail "download '$t' reported $r; see $LOGDIR/dl.$t.log"
done
[ -f "$SRC/config.json" ] || env_fail "$SRC/config.json absent after download"
ls "$UQFF_DIR"/*.uqff >/dev/null 2>&1 || env_fail "no .uqff shards in $UQFF_DIR"
UQFF=$(ls "$UQFF_DIR"/*-0.uqff 2>/dev/null | head -1 || ls "$UQFF_DIR"/*.uqff | head -1)
say "models ready: src=$(du -sh "$SRC" | cut -f1) uqff=$(du -sh "$UQFF_DIR" | cut -f1), first shard $(basename "$UQFF")"

# ---------------------------------------------------------------------------
# 7. The sweep
# ---------------------------------------------------------------------------
step "7/7 capture sweep"
export SRC UQFF REPO
# Build IN PLACE. A separate worktree would be a different workspace root and
# would trigger a second ~11 min cold build for no benefit — this box is ours
# alone, so there is no other chain's checkout to protect.
export WORKTREE="$REPO"
export CARGO_TARGET_DIR="$REPO/target"
export LOGDIR="$LOGDIR"
# This box is exclusively ours, so the lock is a formality — but take it anyway
# so the invariant is uniform across boxes and a future second chain sees us.
mkdir -p /root/locks
exec /root/arc/arc-tools/arcgraph_capture_probe.sh
