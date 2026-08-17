#!/usr/bin/env bash
# RUNG 1 — the FA4->MLA gate, one command.
#
#   Does export_to_c() emit a linkable .o exposing __tvm_ffi_<name> that Rust
#   can call, with NO Python at runtime?
#
# Everything above rung 1 is worthless if this fails, so this runs first and
# alone. Total cost is a few minutes on any SM90 box; it does not need a free
# GPU for long and does not touch model weights.
#
#   bash arc-tools/fa4/rung1_gate.sh 2>&1 | tee /tmp/arc_fa4_rung1/gate.log
#
# Then send back:
#   /tmp/arc_fa4_rung1/manifest.json
#   /tmp/arc_fa4_rung1/link_verdict.json
#   /tmp/arc_fa4_rung1/rustc.log      (only if the link failed)

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /tmp/arc_fa4_rung1

echo "############ 0. bootstrap ############"
# The runcrate image ships pip 22.0.2 with no numpy; flash-attn's metadata
# generation then dies with:
#   TypeError: canonicalize_version() got an unexpected keyword argument
#              'strip_trailing_zero'
python3 -m pip install -q --upgrade pip setuptools wheel || true
python3 -m pip install -q numpy || true
# nvidia-cutlass-dsl is what actually matters; flash-attn-4 pulls it in, but we
# install it directly too so the gate can run even if the FA4 wheel is
# unavailable for this platform. The gate tests the DSL toolchain, not FA4.
python3 -m pip install -q "nvidia-cutlass-dsl>=4.6.2" || true
python3 -m pip install -q flash-attn-4 || echo "  (flash-attn-4 unavailable — gate does not require it)"

python3 - <<'PY'
for m in ("cutlass", "flash_attn", "torch"):
    try:
        mod = __import__(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:12s} MISSING ({type(e).__name__})")
PY
command -v rustc >/dev/null 2>&1 && echo "  rustc        $(rustc --version)" || echo "  rustc        MISSING — stage C cannot run"
command -v nvcc  >/dev/null 2>&1 && echo "  nvcc         $(nvcc --version | tail -1)" || echo "  nvcc         missing"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/  gpu          /'

# ---- driver vs toolkit: the PTX-JIT trap ----------------------------------
# `nvidia-smi` reports the highest CUDA version the DRIVER can JIT. If the
# toolkit is newer than that, anything shipping PTX and relying on JIT dies with
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION, while a NATIVE CUBIN for the exact arch
# loads fine. This is a live mismatch on our H200 (driver 580.173.02 -> CUDA
# 13.0, toolkit 13.1), so it is checked here and it constrains rung 2: load
# native cubin/SASS, never PTX.
DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
TK_CUDA=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
echo "  driver max CUDA: ${DRV_CUDA:-unknown}   toolkit: ${TK_CUDA:-unknown}"
if [ -n "${DRV_CUDA:-}" ] && [ -n "${TK_CUDA:-}" ]; then
    if [ "$(printf '%s\n%s\n' "$DRV_CUDA" "$TK_CUDA" | sort -V | head -1)" = "$DRV_CUDA" ] \
       && [ "$DRV_CUDA" != "$TK_CUDA" ]; then
        echo "  ** PTX-JIT TRAP: toolkit $TK_CUDA > driver-supported $DRV_CUDA."
        echo "     PTX will fail with CUDA_ERROR_UNSUPPORTED_PTX_VERSION."
        echo "     Rung 2 MUST load native cubin (SASS) for sm_90, not PTX."
    else
        echo "  driver/toolkit CUDA versions compatible for PTX JIT"
    fi
fi

echo
echo "############ 1. stage A/B — export + ABI ############"
python3 "$HERE/rung1_export.py"
A=$?

echo
echo "############ 2. stage C — Rust link, no Python ############"
bash "$HERE/rung1_link_test.sh"
C=$?

echo
echo "############ RUNG 1 RESULT ############"
python3 - <<'PY'
import json, os
for f in ("/tmp/arc_fa4_rung1/manifest.json",
          "/tmp/arc_fa4_rung1/link_verdict.json"):
    if not os.path.exists(f):
        print(f"  {os.path.basename(f)}: MISSING")
        continue
    d = json.load(open(f))
    v = d.get("verdict")
    print(f"  {os.path.basename(f)}: "
          f"{v.get('code') if isinstance(v, dict) else v}")
PY
echo "  stage A/B exit=$A   stage C exit=$C"
exit $C
