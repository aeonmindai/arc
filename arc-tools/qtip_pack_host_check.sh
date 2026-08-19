#!/usr/bin/env bash
# Compile the REAL `kernels/qtip/qtip_geom.cuh` with a host C++ compiler and
# assert the packed bitstream it emits, at every baked geometry.
#
# WHY THIS EXISTS
# ---------------
# `qtip_pack_symbol` stopped being "one byte per 8/K symbols" and became a
# bitstream, because K=9/V=4/L=12 — the rung the quality sweep selected, at
# 2.25 bpw — puts a symbol across a byte boundary. Two things then need
# proving, and the obvious instruments cannot prove either:
#
#   * K=4/V=2/L=16 must stay BYTE-IDENTICAL. It is the rung that ships; a
#     regression here breaks the published artifact, not a future one.
#   * K=9 must land its bits exactly where the DECODER looks for them
#     (`qtip/trellis_v4l12.rs::Rung::extract`), and must not write past
#     `ceil(num_symbols*K/8)`.
#
# The CUDA parity tests (`cuda_beam_*_matches_cpu_*`) are the hardware gate for
# both, and they SKIP without a device — so on a Mac dev box, and on every
# GPU-less CI runner, nothing checked this at all. The packer is pure integer
# arithmetic on a byte array, though: stub `__device__`/`__forceinline__` and
# the exact source text nvcc compiles also compiles with clang++/g++. That is
# what this does. No CUDA toolkit, no GPU, ~1 second.
#
# It checks the SOURCE, not a transcription of it: the header is #included, so
# a change to `qtip_pack_symbol` or `MAX_BYTES_PER_SYMBOL` is seen here even
# though the reference packer below is written independently.
#
# THIS GATE IS PROVEN RED
# -----------------------
# Mutations run against it while it was written, each red and only in the row
# it should be:
#   * `MAX_BYTES_PER_SYMBOL` -> 1                   : K=9 and K=6 rows truncate
#   * byte-aligned shift `4*(t&1)` -> `4*((t+1)&1)` : K=4 nibble identity fails
#   * `packed_bytes_per_row` -> `num_symbols`       : K=9 length assertion fails
#   * drop `if (8u*b < used)` from the packer       : ASan heap-buffer-overflow
#
# AND ONE MUTATION THAT DOES **NOT** GO RED, STATED SO NOBODY BUILDS ON IT.
# Enlarging `MAX_BYTES_PER_SYMBOL` (e.g. to the naive `ceil((7+K)/8)`, which is
# 3 at K=9) is NOT caught here, and should not be: the packer's per-symbol
# `used = off + K` guard already stops the extra unrolled iteration from
# storing, so an over-large bound costs a dead instruction and nothing else.
# The bound's EXACTNESS matters on the serving side, where the same formula is
# a read width — that is pinned by `qtip/mod.rs`'s
# `the_straddle_bound_is_exact_at_every_baked_k`, not by this script.
#
# `QtipGeom<6,3,12>` is instantiated below purely to make that `used` guard
# non-vacuous. It is NOT a rung anyone bakes: at K=9 and K=10 every reachable
# offset gives `ceil((off+K)/8) == 2 == MAX_BYTES_PER_SYMBOL`, so the guard
# never fires and would be untested code. K=6 has `off == 0 => used == 6`, one
# byte, against a bound of 2 — the case that runs off the end of the row when
# the guard is removed.
#
# ⚠ Removing that guard writes `|= 0`, so the byte VALUE is unchanged and a
# guard band of zeros — or of any sentinel — cannot detect it. Only ASan can,
# which is why the row is an exact-size heap allocation and the build asks for
# `-fsanitize=address`. The overrun still matters in the kernel: rows are
# contiguous in one buffer and different blocks own different rows at the same
# time, so a non-atomic read-modify-write on the next row's first byte can
# clobber that row's own store.
#
# Usage:  arc-tools/qtip_pack_host_check.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KDIR="${REPO_ROOT}/mistralrs-quant/kernels/qtip"
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

CXX="${CXX:-}"
if [ -z "${CXX}" ]; then
  for c in clang++ g++ c++; do
    if command -v "${c}" >/dev/null 2>&1; then CXX="${c}"; break; fi
  done
fi
if [ -z "${CXX}" ]; then
  echo "FAIL: no host C++ compiler found (tried clang++, g++, c++)."
  echo "      This gate needs no CUDA toolkit — only a C++17 compiler."
  exit 1
fi

cat >"${OUT}/qtip_pack_host_check.cpp" <<'EOF'
// The CUDA attributes are the only thing that stops a host compiler from
// reading this header. Stub them and the source text under test is byte for
// byte what nvcc sees.
#define __device__
#define __forceinline__ inline
#define __restrict__
#include "qtip_geom.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// INDEPENDENT reference, written from the wire format rather than from the
// header: symbol `t` at bits [t*K, t*K + K), LSB-first. This is the rule
// `qtip/trellis_v4l12.rs::Rung::pack` implements and `Rung::extract` inverts.
static std::vector<uint8_t> ref_pack(const std::vector<uint32_t>& syms, uint32_t k) {
    const size_t bytes = (syms.size() * (size_t)k + 7) / 8;
    std::vector<uint8_t> out(bytes, 0);
    const uint64_t mask = (1ull << k) - 1;
    for (size_t t = 0; t < syms.size(); ++t) {
        const size_t   bit  = t * (size_t)k;
        const uint64_t v    = ((uint64_t)syms[t] & mask) << (bit % 8);
        const size_t   need = ((bit % 8) + k + 7) / 8;
        for (size_t i = 0; i < need; ++i) {
            out[bit / 8 + i] |= (uint8_t)((v >> (8 * i)) & 0xFF);
        }
    }
    return out;
}

// The SHIPPED K=4 nibble packer, transcribed from `cpu_reference_packed` in
// `qtip/mod.rs`. "Byte-identical" is a claim about this, not about ref_pack.
static std::vector<uint8_t> nibble_pack_k4(const std::vector<uint32_t>& syms) {
    std::vector<uint8_t> p(syms.size() / 2, 0);
    for (size_t i = 0; i < syms.size(); ++i) {
        if (i % 2 == 0) p[i / 2]  = (uint8_t)(syms[i] & 0xF);
        else            p[i / 2] |= (uint8_t)((syms[i] & 0xF) << 4);
    }
    return p;
}

static uint32_t rng_state = 0x9E3779B9u;
static uint32_t nxt() { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }

template <class G>
static int check(const char* name, int n) {
    std::vector<uint32_t> syms((size_t)n);
    for (int i = 0; i < n; ++i) syms[(size_t)i] = nxt() & G::SYM_MASK;

    const int  bytes = qtip_packed_bytes_per_row<G>(n);
    const auto want  = ref_pack(syms, G::K);
    if ((size_t)bytes != want.size()) {
        printf("FAIL %s: packed_bytes_per_row = %d, wire format needs %zu\n",
               name, bytes, want.size());
        return 1;
    }

    // EXACT-SIZE heap row, on purpose. A packer that runs one byte past the
    // end writes `|= 0` there — the value is unchanged, so a guard band of
    // zeros (or of any sentinel) cannot see it. ASan's redzone can, which is
    // why this allocates exactly `bytes` and the script builds with
    // `-fsanitize=address` whenever the compiler has it.
    //
    // That overrun is NOT harmless in the kernel even though the byte value
    // does not change: rows are contiguous in one buffer and different blocks
    // own different rows concurrently, so a non-atomic read-modify-write on
    // the next row's first byte can clobber that row's own store.
    uint8_t* got = new uint8_t[(size_t)bytes]();
    for (int t = 0; t < n; ++t) qtip_pack_symbol<G>(got, t, syms[(size_t)t]);

    int rc = 0;
    if (memcmp(got, want.data(), (size_t)bytes) != 0) {
        for (int i = 0; i < bytes; ++i) {
            if (got[(size_t)i] != want[(size_t)i]) {
                printf("FAIL %s: byte %d is 0x%02X, wire format says 0x%02X\n",
                       name, i, got[(size_t)i], want[(size_t)i]);
                break;
            }
        }
        rc = 1;
    } else {
        printf("ok   %-16s K=%u V=%u L=%2u  bytes=%-5d  byte_aligned=%d  max_bytes/sym=%u\n",
               name, G::K, G::V, G::L, bytes, (int)G::BYTE_ALIGNED, G::MAX_BYTES_PER_SYMBOL);
    }
    delete[] got;
    return rc;
}

int main() {
    int bad = 0;
    // The real V4-Flash expert shapes: in_features 7168 (gate/up) and 2048
    // (down), so num_symbols is 3584/1024 at V=2 and 1792/512 at V=4.
    bad |= check<QtipGeomK4V2L16>("K4V2L16 n=3584", 3584);
    bad |= check<QtipGeomK4V2L16>("K4V2L16 n=1024", 1024);
    bad |= check<QtipGeomK8V4L12>("K8V4L12 n=1792", 1792);
    bad |= check<QtipGeomK8V4L12>("K8V4L12 n=512",  512);
    bad |= check<QtipGeomK9V4L12>("K9V4L12 n=1792", 1792);
    bad |= check<QtipGeomK9V4L12>("K9V4L12 n=512",  512);

    // NOT a rung. See the header: K=6 is the geometry that makes the packer's
    // `used = off + K` guard fire (off == 0 gives a 6-bit symbol inside ONE
    // byte, against MAX_BYTES_PER_SYMBOL == 2). Without that guard the last
    // symbol of the row writes into the guard band. `n` is a multiple of 4 so
    // the row is whole bytes.
    bad |= check<QtipGeom<6, 3, 12>>("K6V3L12 n=1024", 1024);

    // THE shipping guarantee: K=4 is byte-identical to the packer that baked
    // the published artifact, not merely consistent with the new wire rule.
    std::vector<uint32_t> s(4096);
    for (size_t i = 0; i < s.size(); ++i) s[i] = nxt() & 0xF;
    std::vector<uint8_t> got(s.size() / 2, 0);
    for (size_t t = 0; t < s.size(); ++t) {
        qtip_pack_symbol<QtipGeomK4V2L16>(got.data(), (int)t, s[t]);
    }
    if (got != nibble_pack_k4(s)) {
        printf("FAIL K=4 is NOT byte-identical to the shipped nibble packer\n");
        bad = 1;
    } else {
        printf("ok   K=4 is byte-identical to the shipped nibble packer (4096 symbols)\n");
    }

    printf(bad ? "RESULT: FAILED\n" : "RESULT: PASS\n");
    return bad;
}
EOF

echo "== QTIP packed-bitstream host gate (no CUDA toolkit, no GPU) =="
echo "compiler: $(${CXX} --version | head -1)"

# AddressSanitizer is what detects an overrun whose written VALUE is unchanged
# (`|= 0` past the end of the row). Without it this script still checks every
# byte of every row, but it cannot see that class at all — so say which mode
# it ran in rather than implying a coverage it does not have.
SAN=(-fsanitize=address -fno-omit-frame-pointer)
if ! "${CXX}" -std=c++17 "${SAN[@]}" -x c++ -c -o /dev/null - </dev/null 2>/dev/null; then
  SAN=()
  echo "asan:     UNAVAILABLE — byte contents are checked, overruns are NOT"
else
  echo "asan:     on (row allocated at exactly packed_bytes_per_row)"
fi
echo
# `${SAN[@]+...}` and not `${SAN[@]}`: under `set -u`, expanding an EMPTY array
# is an unbound-variable error in bash <= 4.3 (macOS ships 3.2), so the
# no-sanitizer fallback would abort the gate on exactly the runners that need
# it. Caught by forcing the fallback, not by reading the script.
"${CXX}" -std=c++17 -O1 -g -Wall -Wextra -Werror ${SAN[@]+"${SAN[@]}"} \
         -I "${KDIR}" -o "${OUT}/qtip_pack_host_check" \
         "${OUT}/qtip_pack_host_check.cpp"
"${OUT}/qtip_pack_host_check"
