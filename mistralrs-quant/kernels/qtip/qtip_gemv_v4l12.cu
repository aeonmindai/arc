// Parent system: ArcKernels (rung owner: ArcQuant / QTIP)
//
// Fused QTIP decode + gemv for the **V=4 / L=12** family, with the whole
// reproduction table staged in shared memory as bf16 and **K as a template
// parameter over symbol extraction only**.
//
// WHAT IS FIXED AND WHAT VARIES
// -----------------------------
// L=12 and V=4 fix the table at 2^12 x 4 bf16 = 32,768 B, which is under the
// 48 KiB static __shared__ limit, so it is staged once per block with NO
// `cudaFuncSetAttribute` opt-in and every lookup is an LDS. **The table does
// not read K at all** -- confirmed by a codebook sweep at fixed L=12/V=4 --
// so K moves without the table, its size, its layout, or the activation
// indexing moving with it.
//
// K changes symbol extraction and nothing else:
//
//     K=8   2.00 bpw   a symbol IS a byte      1 byte  read, no shift/mask
//     K=9   2.25 bpw   spans a byte boundary   2 bytes read + shift + mask
//     K=10  2.50 bpw   spans a byte boundary   2-3 bytes read + shift + mask
//
// WHY THIS FAMILY EXISTS AT ALL
// -----------------------------
// The shipped K=4/V=2/L=16 rung's table is 2^16 x 2 **f32** = 524,288 B. That
// does not fit shared memory at any occupancy, so every decoded symbol pays a
// dependent, data-scattered load to L2 -- measured at 388 GB/s, ~8% of H200
// HBM, and it is the decode limiter. That is the *shipped default* path:
// `QtipCodebook::DEFAULT` is `Gaussian`, which gathers from the stored table.
// (`qtip/mod.rs`'s claim that "the GPU decode paths compute instead" holds only
// for the `Mcg` codebook, which is not the default.) Killing that gather is
// what the 32 KiB table is for.
//
// WHICH K SHIPS
// -------------
// K=8 is the **control**: byte-aligned, and the geometry the early probe
// figures were taken on. It is not expected to be the K that ships -- CPU
// sweeps put K=8/V=4/L=12 at Delta w_cos -0.00698 (random codebook) / -0.00307
// (converged trellis-Lloyd) against a +/-0.0008 threshold, and no codebook
// design recovered it. **K=9 at 2.25 bpw measured +0.00402.**
//
// WARNING: the control's own 5.375 / 4.375 inst/weight have NOT been
// reproduced by the census rig, so they are provisional, as is the 15.125
// figure for the shipped rung that they are compared against.
//
// MEASUREMENT STATUS -- READ BEFORE QUOTING A NUMBER
// -------------------------------------------------
// **This file has never been compiled with nvcc and never run on a GPU by its
// author. Its instruction count is UNESTABLISHED at every K.**
//
// The absolute per-route figures circulating for this family (funnel 5.625,
// padded 6.312 at K=9 / 6.250 at K=10, clamped 7.438 / 7.375) are PROVISIONAL:
// the rig that produced them has not yet reproduced the K=8 control's published
// 5.375 / 4.375, so it is not validated end to end. An earlier isolated
// micro-census was withdrawn outright -- its K=8 control measured 100%
// non-linear, because K=8's contiguous-byte symbols let ptxas merge loads while
// K=9's (9t)>>3 indices cannot, making every "vs K=8" per-symbol delta a
// comparison against a moving baseline.
//
// The ONE number here that does not depend on a baseline is the clamp-vs-pad
// delta, because it is a difference between two kernels in the same mode. That
// is why the padding decision was taken on it and nothing else was.
//
// Hand-counting C++ undercounts SASS by ~2.05x on this kernel family, so a
// count that was not compiled is not a count.
//
// ROWS ARE PADDED; THERE IS NO TAIL CLAMP
// ---------------------------------------
// `QtipSymExtract<K>` reads a COMPILE-TIME count of bytes (MAX_BYTES) so the
// loop unrolls with no data-dependent branch and no clamp. That requires the
// row to be at least `row_stride = align_up(data_bytes + MAX_BYTES - 1, 4)`
// bytes, which the format now guarantees and the Rust launcher enforces.
//
// This was a measured decision, not a preference. Full-kernel differencing at
// sm_90 put the clamped variant at +1.126 inst/weight over the padded one at
// K=9 and +1.125 at K=10 -- two independent geometries agreeing to 0.1%. The
// padding costs at most 4 bytes per row (0.35% at in_features=4096).
//
// The round-up to 4 is not for tail safety, which needs only MAX_BYTES-1. It
// makes every row base 4-byte aligned, which is what the cheapest measured
// route -- the funnel route, 5.625 inst/weight at both K=9 and K=10 -- needs.
// The stride is format, and changing it twice is the expensive outcome, so it
// is chosen once to support both.
//
// PARITY
// ------
// `mistralrs-quant/src/qtip/trellis_v4l12.rs::Rung::gemv_row(.., Off)` is the
// reference, and the correspondence is operation for operation:
//
//     w   = __fmul_rn(cb, scale)   <->  let w = cb * scale;
//     acc = fmaf(w, x, acc)        <->  acc = w.mul_add(x, acc);
//
// `__fmul_rn` rather than `*` because build.rs compiles this directory with
// `--use_fast_math`, which implies `-fmad=true`; a bare `cb * scale` feeding
// the FMA's multiplicand is not contractible today, but writing the intent is
// what `qtip_exact_fp.cuh` exists for and costs nothing. bf16 -> f32 widening
// is exact on both sides, and `fmaf` and Rust's `f32::mul_add` are both the
// single-rounding fused op.
//
// The row-scale hoist (`ROW_SCALE_HOIST`) is the ONE thing that breaks this.
// It reassociates the sum -- `sum (cb*s)*x` becomes `s * sum cb*x` -- so it is
// a separate template parameter and the parity gate runs with it off.
//
// SM80+ (uses __nv_bfloat16). Gated by `has_qtip_kernels` in build.rs.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// ---- fixed geometry: the table --------------------------------------------

constexpr uint32_t QV4_L           = 12;
constexpr uint32_t QV4_V           = 4;
constexpr uint32_t QV4_STATE_MASK  = (1u << QV4_L) - 1u;   // 0xFFF
constexpr uint32_t QV4_LUT_STATES  = 1u << QV4_L;          // 4096
constexpr uint32_t QV4_LUT_ENTRIES = QV4_LUT_STATES * QV4_V; // 16384

static_assert(QV4_LUT_ENTRIES * sizeof(__nv_bfloat16) == 32768,
              "the V=4/L=12 table must be exactly 32 KiB; that is the property that lets it "
              "live in static shared memory without a cudaFuncSetAttribute opt-in");
static_assert(QV4_LUT_ENTRIES * sizeof(__nv_bfloat16) <= 48 * 1024,
              "static __shared__ is capped at 48 KiB per block");

// ---- the K seam ------------------------------------------------------------
//
// Everything K-dependent lives here. The table path below never sees K.

template <int K>
struct QtipSymExtract {
    static_assert(K >= 2 && K <= 16, "K must fit the 3-byte extraction window");

    // Bytes one extraction may touch, worst case over the bit offsets that are
    // actually REACHABLE. The offset of symbol t is (t*K) mod 8, which ranges
    // over the multiples of gcd(K, 8) and nothing else, so the worst reachable
    // offset is 8 - gcd(K,8), not 7.
    //
    // At K=10 that is the difference between 2 bytes and 3: the offsets are
    // only {0, 2, 4, 6}, so off + K <= 16 and two bytes always suffice. The
    // naive ceil((7+K)/8) would have every symbol read, shift and discard a
    // wasted third byte.
    static constexpr int GCD8 = (K % 8 == 0) ? 8 : ((K % 4 == 0) ? 4 : ((K % 2 == 0) ? 2 : 1));
    static constexpr int MAX_BYTES = (8 - GCD8 + K + 7) / 8;

    // Symbol t occupies bits [t*K, t*K + K), LSB-first: bit j lives in byte
    // j/8 at bit position j%8. Same convention the shipped K=4/V=2 rung uses
    // (symbol 2b is the low nibble of byte b), pinned on the Rust side by
    // `the_bit_layout_matches_the_shipped_k4_rungs_nibble_order`.
    // PRECONDITION: the row is at least `Rung::row_stride(num_symbols)` bytes,
    // i.e. the symbol data plus MAX_BYTES-1 bytes of zero padding, rounded up
    // to a multiple of 4. The Rust launcher refuses anything shorter.
    //
    // That precondition is what deletes the tail clamp this loop used to
    // carry, and the clamp was NOT cheap: measured at sm_90 by full-kernel
    // differencing, it cost +1.126 inst/weight at K=9 and +1.125 at K=10
    // (~+4.50 per extraction). Two independent geometries agreeing to 0.1% is
    // why that number is trusted — it is a difference between two kernels in
    // the same mode, so it survives the baseline problem that invalidated the
    // isolated per-symbol route figures.
    //
    // Reading into the padding is harmless as well as in-bounds: byte i
    // contributes bits [8i, 8i+8) of `w`, and the mask keeps only bits
    // [off, off+K), so any byte beyond ceil((off+K)/8) lands entirely above
    // the window and cannot change the symbol.
    __device__ __forceinline__ static uint32_t get(
        const uint8_t* __restrict__ row, int t
    ) {
        const int bit = t * K;
        const int b0  = bit >> 3;
        const int off = bit & 7;
        uint32_t w = 0u;
        #pragma unroll
        for (int i = 0; i < MAX_BYTES; ++i) {
            w |= (uint32_t)__ldg(row + b0 + i) << (8 * i);
        }
        return (w >> off) & ((1u << K) - 1u);
    }
};

// The byte-aligned control. A symbol IS a byte: one LDG, no shift, no mask, no
// clamp. This is the floor every other K is measured against, so it is a real
// specialisation rather than a lucky path through the general code.
template <>
struct QtipSymExtract<8> {
    static constexpr int MAX_BYTES = 1;
    __device__ __forceinline__ static uint32_t get(
        const uint8_t* __restrict__ row, int t
    ) {
        return (uint32_t)__ldg(row + t);
    }
};

// ---- dtype helpers ---------------------------------------------------------

template <typename T>
__device__ __forceinline__ T from_f32(float v);

template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
    return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float to_f32(T v);

template <>
__device__ __forceinline__ float to_f32<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// Warp-level XOR-shuffle sum reduction.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

// Stage the whole 32 KiB table into shared memory, cooperatively.
//
// K-independent, like everything else that touches the table. 16-byte loads
// when the source pointer allows it (a freshly materialized [4096, 4] tensor is
// cudaMalloc-aligned, so this is the normal case), with a scalar fallback so a
// view with a nonzero element offset degrades in speed rather than faulting.
// Caller issues the __syncthreads().
__device__ __forceinline__ void stage_lut(
    __nv_bfloat16* __restrict__ lut_s, const __nv_bfloat16* __restrict__ lut_g, int threads
) {
    const int tid = threadIdx.x;
    if ((reinterpret_cast<uintptr_t>(lut_g) & 0xFu) == 0u) {
        constexpr int N_VEC = (int)(QV4_LUT_ENTRIES * sizeof(__nv_bfloat16) / sizeof(uint4)); // 2048
        const uint4* src = reinterpret_cast<const uint4*>(lut_g);
        uint4*       dst = reinterpret_cast<uint4*>(lut_s);
        for (int i = tid; i < N_VEC; i += threads) {
            dst[i] = src[i];
        }
    } else {
        for (int i = tid; i < (int)QV4_LUT_ENTRIES; i += threads) {
            lut_s[i] = lut_g[i];
        }
    }
}

// Fused decode + gemv. `y[row] = sum_k W[row,k] * x[k]`, with W reconstructed
// on the fly from `packed`, `row_scales` and the staged table. `x` must
// already be in the QTIP-rotated frame (the caller applies
// `launch_qtip_rotate_x_*` first, exactly as for the K=4 rung).
//
// Grid:  (n_blocks, 1, 1)  -- each block walks rows blockIdx.x, +gridDim.x, ...
// Block: (THREADS, 1, 1)
//
// The grid-stride over rows is what makes the staging pay. Staging costs
// 32 KiB of L2->SMEM per BLOCK, not per row, so with a grid sized to fill the
// device the cost is bounded by the grid rather than by n_rows. One block per
// row would stage 32 KiB to do (in_features/4) bytes of packed work -- 32x
// overhead at in_features=4096.
// ONE WARP OWNS A ROW, and a block owns THREADS/32 rows at once.
//
// Not one block per row. Every lane replays WARMUP_SYMS symbols before decoding
// its own slice, so extractions per weight are (S + W) / (S * V) against an
// ideal of 1/V -- a tax of 1 + W/S set entirely by slice length S, and S is
// num_symbols / lanes_per_row. Spreading a row across all 128 threads makes S
// short and the tax large:
//
//     in_features   128 lanes/row      32 lanes/row
//     4096          S=8,  1.25x        S=32, 1.06x
//     1024          S=2,  1.99x        S=8,  1.24x
//      512          S=1,  2.98x        S=4,  1.48x
//
// A warp per row is the same parallelism with four times the slice. It is also
// strictly simpler: a warp reduces with shuffles alone, so the cross-warp
// butterfly, the `warp_sums` shared array and BOTH `__syncthreads` disappear
// from the row loop.
//
// This is K-INDEPENDENT -- it multiplies the extraction count at every K, so it
// scales the alignment penalty by the same factor. It is not a substitute for
// picking a good extraction route; it multiplies whichever one is picked.
template <typename T, int THREADS, int K, bool ROW_SCALE_HOIST>
__global__ void __launch_bounds__(THREADS)
qtip_fused_gemv_v4_l12_kernel(
    const uint8_t*       __restrict__ packed,      // [n_rows, packed_per_row]
    const float*         __restrict__ row_scales,  // [n_rows]
    const __nv_bfloat16* __restrict__ lut,         // [2^L * V] bf16, 32 KiB
    const T*             __restrict__ x,           // [k_in], k_in == num_symbols*V
    T*                   __restrict__ y,           // [n_rows]
    int n_rows,
    int packed_per_row,
    int num_symbols
) {
    // 32,768 B, and that is the WHOLE static shared footprint of this kernel
    // now that the cross-warp reduction is gone.
    __shared__ __align__(16) __nv_bfloat16 lut_s[QV4_LUT_ENTRIES];
    constexpr int N_WARPS = THREADS / 32;
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");
    // No `warp_sums`: a warp owns a whole row, so the reduction never leaves
    // the warp and there is nothing to hand across one.

    // Prior symbols consumed before a thread decodes its own slice: ceil(L/K).
    // 2 at every supported K. One more than strictly required -- the state that
    // decodes symbol t keeps only L-K bits of s_{t-1}, and those are supplied
    // by symbol t-1 -- but it matches `qtip_gemv.cu`'s definition at K=4/L=16,
    // and the Rust side pins both facts.
    constexpr int WARMUP_SYMS = (QV4_L + K - 1) / K;
    static_assert(WARMUP_SYMS * K >= (int)QV4_L,
                  "warmup must shift at least L bits through the state register");

    // With the gcd bound, TWO bytes is exact for every K anyone has: K=8 needs
    // 1, K=9 and K=10 need 2. The general 3-byte path in QtipSymExtract is
    // therefore unreachable for every supported rung, and exists only to price
    // the unnecessary generality. A K that needs three (K=11, K=13, ...) is a
    // real per-symbol cost increase, so it has to be an explicit decision here
    // rather than something a new `case` quietly turns on.
    static_assert(QtipSymExtract<K>::MAX_BYTES <= 2,
                  "this K needs a 3-byte extraction window. That is a per-symbol cost increase "
                  "on every weight; if it is intended, widen this assert deliberately.");

    stage_lut(lut_s, lut, THREADS);
    __syncthreads();

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    // Each LANE takes a contiguous slice of its warp's row. Dividing by 32
    // rather than by THREADS is the warmup-tax fix described above.
    const int sym_per_thread = (num_symbols + 31) / 32;
    const int sym_start_raw  = lane * sym_per_thread;
    const int sym_end        = min(num_symbols, sym_start_raw + sym_per_thread);

    for (int row = blockIdx.x * N_WARPS + warp_id; row < n_rows;
         row += gridDim.x * N_WARPS) {
        const uint8_t* row_packed = packed + (size_t)row * packed_per_row;
        const float    scale      = __ldg(row_scales + row);

        float acc = 0.0f;

        if (sym_start_raw < num_symbols) {
            // ---- Warmup ----
            // Threads i>0 replay WARMUP_SYMS prior symbols to seed the state;
            // their decoded weights belong to the previous thread's slice and
            // are NOT accumulated. Thread 0 starts from the true initial state
            // 0. A thread starting inside the first WARMUP_SYMS symbols walks
            // from 0, which is exact rather than approximate for the same
            // reason.
            uint32_t state = 0u;
            if (sym_start_raw > 0) {
                const int warm_start = max(0, sym_start_raw - WARMUP_SYMS);
                for (int t = warm_start; t < sym_start_raw; ++t) {
                    const uint32_t sym =
                        QtipSymExtract<K>::get(row_packed, t);
                    state = ((state << K) | sym) & QV4_STATE_MASK;
                }
            }

            // ---- Decode ----
            // One symbol -> one 8-byte table entry -> four weights. The only
            // K-dependent step is the extraction; everything below is shared.
            for (int sym_idx = sym_start_raw; sym_idx < sym_end; ++sym_idx) {
                const uint32_t sym =
                    QtipSymExtract<K>::get(row_packed, sym_idx);
                state = ((state << K) | sym) & QV4_STATE_MASK;

                // `lut_s` is 16-byte aligned and the entry is 4 bf16, so this
                // address is 8-byte aligned for every state -- the layout that
                // makes an LDS.64 (and an mma.m16n8k16.bf16 B-operand pair)
                // possible. Whether the compiler actually fuses the four 2-byte
                // reads into one 8-byte load is a codegen outcome to be
                // confirmed with `nvcc -cubin`, NOT something this comment
                // asserts.
                const __nv_bfloat16* c = &lut_s[(size_t)state * QV4_V];
                const int x_off = sym_idx * (int)QV4_V;

                #pragma unroll
                for (int v = 0; v < (int)QV4_V; ++v) {
                    const float cb = __bfloat162float(c[v]);
                    const float xv = to_f32<T>(__ldg(x + x_off + v));
                    if constexpr (ROW_SCALE_HOIST) {
                        acc = fmaf(cb, xv, acc);
                    } else {
                        acc = fmaf(__fmul_rn(cb, scale), xv, acc);
                    }
                }
            }
        }

        // ---- Warp reduction ----
        // Shuffles only. No shared memory and no __syncthreads: warps in this
        // block are working on DIFFERENT rows and never need to meet, which is
        // the second thing a warp-per-row shape buys.
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            y[row] = from_f32<T>(ROW_SCALE_HOIST ? (acc * scale) : acc);
        }
    }
}

// Blocks resident at once. Staging is 32 KiB of L2->SMEM per block, so this
// caps total staging traffic at 64 MiB regardless of how many rows the layer
// has. Sized to oversubscribe any current datacentre part (H100 has 132 SMs)
// without making the cap the thing that limits parallelism.
constexpr int QV4_MAX_BLOCKS = 2048;

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// `k` selects the symbol width; `row_scale_hoist` is 0 for the bit-exact
// per-weight scale (the policy the parity gate pins) and nonzero for the
// hoisted one. Both are ints and not enums so the ABI is unambiguous across
// the FFI boundary; `Rung::k()` and `RowScaleHoist::as_abi` on the Rust side
// are the only things that produce them.
//
// An unsupported `k` returns WITHOUT launching, so `y` keeps whatever it held.
// The Rust launcher validates `k` before calling and a source guard
// (`the_rust_model_and_the_cuda_kernel_agree_on_their_shared_constants`)
// asserts that every K the Rust side advertises has a case here — a missing
// case would otherwise be a silent all-zeros layer.
#define QTIP_V4L12_LAUNCH_K(T, K_VAL, HOIST)                                       \
    qtip_fused_gemv_v4_l12_kernel<T, THREADS, K_VAL, HOIST>                        \
        <<<blocks, THREADS, 0, stream>>>(                                          \
            d_packed, d_row_scales, d_lut, d_x_rotated, d_y,                       \
            n_rows, packed_per_row, num_symbols)

#define QTIP_V4L12_GEMV_LAUNCHER(NAME, T)                                          \
    void NAME(const uint8_t*       d_packed,                                       \
              const float*         d_row_scales,                                   \
              const __nv_bfloat16* d_lut,                                          \
              const T*             d_x_rotated,                                    \
              T*                   d_y,                                            \
              int n_rows,                                                          \
              int packed_per_row,                                                  \
              int num_symbols,                                                     \
              int k,                                                               \
              int row_scale_hoist,                                                 \
              cudaStream_t stream) {                                               \
        constexpr int THREADS = 128;                                               \
        if (n_rows <= 0 || num_symbols <= 0 || packed_per_row <= 0) return;        \
        constexpr int ROWS_PER_BLOCK = THREADS / 32;                               \
        const int want = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;           \
        const int blocks = want < QV4_MAX_BLOCKS ? want : QV4_MAX_BLOCKS;          \
        if (row_scale_hoist != 0) {                                                \
            switch (k) {                                                           \
                case 8:  QTIP_V4L12_LAUNCH_K(T, 8,  true); break;                  \
                case 9:  QTIP_V4L12_LAUNCH_K(T, 9,  true); break;                  \
                case 10: QTIP_V4L12_LAUNCH_K(T, 10, true); break;                  \
                default: return;                                                   \
            }                                                                      \
        } else {                                                                   \
            switch (k) {                                                           \
                case 8:  QTIP_V4L12_LAUNCH_K(T, 8,  false); break;                 \
                case 9:  QTIP_V4L12_LAUNCH_K(T, 9,  false); break;                 \
                case 10: QTIP_V4L12_LAUNCH_K(T, 10, false); break;                 \
                default: return;                                                   \
            }                                                                      \
        }                                                                          \
    }

QTIP_V4L12_GEMV_LAUNCHER(launch_qtip_fused_gemv_v4_l12_bf16, __nv_bfloat16)
QTIP_V4L12_GEMV_LAUNCHER(launch_qtip_fused_gemv_v4_l12_f16,  __half)
QTIP_V4L12_GEMV_LAUNCHER(launch_qtip_fused_gemv_v4_l12_f32,  float)

#undef QTIP_V4L12_GEMV_LAUNCHER
#undef QTIP_V4L12_LAUNCH_K

} // extern "C"
