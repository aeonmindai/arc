#!/bin/sh
set -e

# Arc Installation Script
# Downloads prebuilt binaries or builds from source with hardware auto-detection.

REPO="aeonmindai/arc"
BINARY="arc"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info() { printf "${BLUE}info:${NC} %s\n" "$1" >&2; }
success() { printf "${GREEN}success:${NC} %s\n" "$1" >&2; }
warn() { printf "${YELLOW}warning:${NC} %s\n" "$1" >&2; }
error() { printf "${RED}error:${NC} %s\n" "$1" >&2; exit 1; }

can_prompt() { [ -t 0 ] || [ -e /dev/tty ]; }
read_input() {
    if [ -t 0 ]; then read -r REPLY; else read -r REPLY </dev/tty; fi
}

print_banner() {
    printf "${BOLD}"
    cat <<'BANNER'

     _
    / \   _ __ ___
   / _ \ | '__/ __|
  / ___ \| | | (__
 /_/   \_\_|  \___|

BANNER
    printf "${NC}${BLUE}Inference at the speed of physics.${NC}\n"
    printf "${NC}Aeonmind, LLC | https://runcrate.ai/arc${NC}\n\n"
}

detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *) error "Unsupported OS: $(uname -s)" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)  echo "x86_64" ;;
        arm64|aarch64) echo "aarch64" ;;
        *) error "Unsupported architecture: $(uname -m)" ;;
    esac
}

detect_cuda() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'
    fi
}

# Try to download a prebuilt binary from GitHub Releases
try_prebuilt() {
    os="$1"
    arch="$2"
    cuda_cc="$3"

    # Determine variant — match GPU architecture to binary
    variant=""
    if [ "$os" = "macos" ]; then
        variant="${os}-${arch}-metal"
    elif [ -n "$cuda_cc" ]; then
        # Pick the right CUDA binary for this GPU
        if [ "$cuda_cc" -ge 100 ] 2>/dev/null; then
            variant="${os}-${arch}-cuda-blackwell"
            info "Blackwell GPU detected (SM ${cuda_cc})"
        elif [ "$cuda_cc" -ge 90 ] 2>/dev/null; then
            variant="${os}-${arch}-cuda-hopper"
            info "Hopper GPU detected (SM ${cuda_cc})"
        elif [ "$cuda_cc" -ge 89 ] 2>/dev/null; then
            variant="${os}-${arch}-cuda-ada"
            info "Ada Lovelace GPU detected (SM ${cuda_cc})"
        else
            variant="${os}-${arch}-cuda-ampere"
            info "Ampere GPU detected (SM ${cuda_cc})"
        fi
    else
        variant="${os}-${arch}-cpu"
    fi

    # Check for latest release
    api_url="https://api.github.com/repos/${REPO}/releases/latest"
    info "Checking for prebuilt binary (${variant})..."

    release_json=$(curl -sf "$api_url" 2>/dev/null) || {
        warn "No prebuilt releases found — will build from source"
        return 1
    }

    # Look for matching asset
    download_url=$(echo "$release_json" | grep -o "\"browser_download_url\": *\"[^\"]*${variant}[^\"]*\"" | head -1 | cut -d'"' -f4)

    if [ -z "$download_url" ]; then
        warn "No prebuilt binary for ${variant} — will build from source"
        return 1
    fi

    # Download
    tmpdir=$(mktemp -d)
    info "Downloading ${download_url}..."
    curl -fSL "$download_url" -o "${tmpdir}/arc.tar.gz" || {
        warn "Download failed — will build from source"
        rm -rf "$tmpdir"
        return 1
    }

    # Extract
    tar -xzf "${tmpdir}/arc.tar.gz" -C "$tmpdir" || {
        warn "Extraction failed — will build from source"
        rm -rf "$tmpdir"
        return 1
    }

    # Install
    install_dir="${HOME}/.local/bin"
    mkdir -p "$install_dir"

    # Install both binaries
    found=0
    for bin in arc mistralrs; do
        if [ -f "${tmpdir}/${bin}" ]; then
            mv "${tmpdir}/${bin}" "${install_dir}/${bin}"
            chmod +x "${install_dir}/${bin}"
            found=1
        fi
    done

    if [ "$found" = "0" ]; then
        warn "Binary not found in archive — will build from source"
        rm -rf "$tmpdir"
        return 1
    fi

    rm -rf "$tmpdir"
    success "Installed arc to ${install_dir}/arc"

    # Check PATH
    case ":$PATH:" in
        *":${install_dir}:"*) ;;
        *)
            warn "${install_dir} is not in your PATH"
            echo ""
            echo "Add it to your shell profile:"
            echo "  export PATH=\"${install_dir}:\$PATH\""
            echo ""
            ;;
    esac

    return 0
}

# Build from source (fallback)
build_from_source() {
    os="$1"

    # Check Rust
    if ! command -v cargo >/dev/null 2>&1; then
        info "Rust not found — installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        . "$HOME/.cargo/env"
        success "Rust installed"
    fi

    # Detect features
    features=""
    if [ "$os" = "macos" ]; then
        # Check Xcode CLI tools
        if ! xcrun --version >/dev/null 2>&1; then
            info "Installing Xcode Command Line Tools..."
            xcode-select --install
            echo "Complete the installation dialog, then press Enter..."
            read_input
        fi
        # Check Metal toolchain
        if ! xcrun metal --version >/dev/null 2>&1; then
            info "Installing Metal Toolchain..."
            xcodebuild -downloadComponent MetalToolchain
        fi
        features="metal"
        info "macOS detected — enabling Metal"
    else
        cuda_cc=$(detect_cuda)
        if [ -n "$cuda_cc" ]; then
            features="cuda"
            cc_major=$(echo "$cuda_cc" | cut -c1)
            info "CUDA detected (compute ${cc_major}.x)"

            # FlashAttention
            if [ "$cuda_cc" = "90" ]; then
                features="$features flash-attn-v3"
                info "Hopper GPU — enabling flash-attn-v3"
            elif [ "$cuda_cc" -ge 80 ] 2>/dev/null; then
                features="$features flash-attn"
                info "Ampere+ GPU — enabling flash-attn"
            fi
        fi
    fi

    # Clone and build
    info "Building Arc from source..."
    tmpdir=$(mktemp -d)
    git clone --depth 1 "https://github.com/${REPO}.git" "$tmpdir/arc"

    if [ -n "$features" ]; then
        info "Features: $features"
        cargo install --path "$tmpdir/arc/arc-cli" --features "$features"
    else
        cargo install --path "$tmpdir/arc/arc-cli"
    fi

    rm -rf "$tmpdir"
    success "Arc built and installed"
}

# Main
main() {
    print_banner

    os=$(detect_os)
    arch=$(detect_arch)
    cuda_cc=$(detect_cuda)

    info "Detected: ${os}/${arch}$([ -n "$cuda_cc" ] && echo " (CUDA ${cuda_cc})" || echo "")"
    echo ""

    # Try prebuilt first, fall back to source
    if try_prebuilt "$os" "$arch" "$cuda_cc"; then
        echo ""
        success "Arc is ready!"
    else
        echo ""
        build_from_source "$os"
        echo ""
        success "Arc is ready!"
    fi

    echo ""
    printf "${BOLD}Get started:${NC}\n"
    echo "  arc run -m Qwen/Qwen3-4B          # Interactive chat"
    echo "  arc serve --ui -m Qwen/Qwen3-4B   # Server with web UI"
    echo "  arc bench -m Qwen/Qwen3-4B        # Benchmark"
    echo ""
    printf "${BLUE}TurboQuant 3.5-bit KV cache compression is enabled by default.${NC}\n"
    echo ""
}

main "$@"
