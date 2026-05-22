#!/usr/bin/env bash
#
# Arc rental pre-flight checklist.
#
# Run this on a fresh B200/H100 rental host to verify:
#   1. Rust toolchain present + correct version
#   2. CUDA toolchain present (12.4+) when --features cuda requested
#   3. Arc builds cleanly with intended feature flags
#   4. Tests pass (offline subset)
#   5. Architecture-specific weight schema validation (if a model is named)
#
# Usage:
#   ./preflight.sh                    # offline checks only
#   ./preflight.sh --cuda             # also build with --features cuda
#   ./preflight.sh --model <hf-repo>  # download config.json + index, run arc validate
#
# Exit codes:
#   0 — all green, safe to proceed to mistralrs run
#   1 — one or more checks failed
#   2 — script invocation error

set -o pipefail

# Source Rust env if available (rental hosts may not have it in default shell rc)
if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WITH_CUDA=0
MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            WITH_CUDA=1
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "$0" | tail -n +2 | sed 's/^# //;s/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}" >&2
            exit 2
            ;;
    esac
done

# Find repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${BLUE}=== Arc rental pre-flight checklist ===${NC}"
echo "Repo: $REPO_ROOT"
echo "Time: $(date)"
echo "Host: $(hostname)"
[[ $WITH_CUDA -eq 1 ]] && echo "Mode: CUDA build" || echo "Mode: CPU-only"
echo

OVERALL_OK=1

step() {
    local name="$1"
    echo -e "${BLUE}→ $name${NC}"
}

ok() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

fail() {
    echo -e "${RED}  ✗ $1${NC}"
    OVERALL_OK=0
}

# === Check 1: Rust toolchain ===
step "Rust toolchain"
if command -v cargo >/dev/null 2>&1; then
    RUST_VERSION=$(cargo --version | awk '{print $2}')
    ok "cargo $RUST_VERSION"
else
    fail "cargo not found — install via https://rustup.rs/"
fi

if command -v rustc >/dev/null 2>&1; then
    RUSTC_VERSION=$(rustc --version | awk '{print $2}')
    ok "rustc $RUSTC_VERSION"
else
    fail "rustc not found"
fi
echo

# === Check 2: CUDA toolchain (if requested) ===
if [[ $WITH_CUDA -eq 1 ]]; then
    step "CUDA toolchain"
    if command -v nvcc >/dev/null 2>&1; then
        NVCC_VERSION=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
        ok "nvcc $NVCC_VERSION"
        MAJOR=$(echo "$NVCC_VERSION" | cut -d. -f1)
        if [[ "$MAJOR" -lt 12 ]]; then
            fail "CUDA $NVCC_VERSION is too old — Arc requires CUDA 12.4+"
        fi
    else
        fail "nvcc not found — CUDA toolkit not installed"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        ok "nvidia driver $DRIVER"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        ok "GPU: $GPU_NAME"
    else
        fail "nvidia-smi not found"
    fi
    echo
fi

# === Check 3: Workspace build ===
step "Build arc-engine, arc-cli, mistralrs-quant (offline subset)"
if cargo build -p arc-engine -p arc-cli -p mistralrs-quant --no-default-features --release 2>&1 | tail -3; then
    ok "core crates build clean"
else
    fail "core crates failed to build"
fi
echo

# === Check 4: Test sweep (offline) ===
step "Test sweep — arc-engine + mistralrs-quant + arc-cuda-graph"
TEST_OK=1
for crate in arc-engine mistralrs-quant arc-cuda-graph; do
    RESULT=$(cargo test -p $crate --no-default-features --lib 2>&1 | grep "test result" | head -1)
    if [[ "$RESULT" == *"0 failed"* ]]; then
        COUNT=$(echo "$RESULT" | awk '{print $4}')
        ok "$crate: $COUNT tests"
    else
        fail "$crate failed: $RESULT"
        TEST_OK=0
    fi
done
echo

# === Check 5: CUDA build (if requested) ===
if [[ $WITH_CUDA -eq 1 ]]; then
    step "CUDA build of mistralrs-core"
    if cargo build -p mistralrs-core --features "cuda flash-attn cudnn" --release 2>&1 | tail -3; then
        ok "mistralrs-core builds with --features cuda"
    else
        fail "CUDA build failed — see cargo output above"
    fi
    echo
fi

# === Check 6: Model-specific preflight ===
if [[ -n "$MODEL" ]]; then
    step "Model preflight: $MODEL"

    # Determine architecture from HF repo name conventions
    ARCH=""
    if [[ "$MODEL" == *"DeepSeek-V4"* ]] || [[ "$MODEL" == *"deepseek-v4"* ]]; then
        ARCH="deepseekv4"
    elif [[ "$MODEL" == *"Kimi-K2"* ]] || [[ "$MODEL" == *"kimi-k2"* ]] || [[ "$MODEL" == *"K2.5"* ]] || [[ "$MODEL" == *"K2.6"* ]]; then
        ARCH="kimi_k2"
    elif [[ "$MODEL" == *"GLM-5"* ]] || [[ "$MODEL" == *"glm-5"* ]] || [[ "$MODEL" == *"glm5"* ]]; then
        ARCH="glm5moedsa"
    else
        warn "Cannot auto-detect architecture from `$MODEL` — skip schema check"
    fi

    if [[ -n "$ARCH" ]]; then
        # Try to download the safetensors index
        CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
        # Pattern: HF caches under hub/models--<org>--<name>/snapshots/<sha>/
        # Use huggingface_hub if available
        if command -v huggingface-cli >/dev/null 2>&1; then
            INDEX_PATH=$(huggingface-cli download "$MODEL" model.safetensors.index.json 2>/dev/null || true)
            if [[ -n "$INDEX_PATH" && -f "$INDEX_PATH" ]]; then
                ok "downloaded $INDEX_PATH"

                # Run arc validate
                if ./target/release/arc validate --index "$INDEX_PATH" --arch "$ARCH"; then
                    ok "arc validate passed for $MODEL ($ARCH)"
                else
                    fail "arc validate FAILED — see report above for missing tensors"
                fi
            else
                warn "could not download index for $MODEL — check network / HF_TOKEN"
            fi
        else
            warn "huggingface-cli not installed — skipping model index download"
            warn "  install: pip install huggingface_hub"
        fi
    fi
    echo
fi

# === Summary ===
echo -e "${BLUE}=== Pre-flight summary ===${NC}"
if [[ $OVERALL_OK -eq 1 ]]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo "  Safe to proceed:"
    if [[ -n "$MODEL" && -n "$ARCH" ]]; then
        echo "    arc run -m $MODEL"
        echo "    arc serve -m $MODEL"
        echo "    arc bench -m $MODEL"
    else
        echo "    arc run -m <hf-repo-id>"
    fi
    exit 0
else
    echo -e "${RED}✗ ONE OR MORE CHECKS FAILED${NC}"
    echo "  Address the FAIL lines above before proceeding."
    exit 1
fi
