// QTIP gmin-only exhaustive Viterbi trellis quantize kernel (K=4 / V=2 / L=16).
//
// WHAT THIS IS
// ------------
// The exhaustive DP in `qtip_quantize.cu` materialises the full 2^L = 65,536
// state cost array in HBM and ping-pongs it every timestep (512 KiB of traffic
// per symbol position) plus a 4 KiB per-timestep backtrace. This kernel does
// the SAME search — optimal Viterbi, no pruning — with the entire dynamic
// program resident in 32 KiB of shared memory, because the recursion closes on
// the 4,096 *group* minima alone.
//
// THE RECURSION (why the 65,536-state array is never needed)
// ----------------------------------------------------------
// Successor states of `s` are `s' = ((s << K) | sym) & STATE_MASK`, which
// depends on `s` only through `g = s & 0xFFF`. Define
//
//     gmin_t[g] = min over states s with (s & 0xFFF) == g of cost_t[s]
//
// which is exactly the `group_cost[]` that `viterbi.rs::exhaustive_quantize_row`
// and `qtip_quantize_rows_viterbi_kernel` already compute as an intermediate
// (`s_best_cost[p] = min_j prev[(j << 12) | p]` is the min over the states whose
// low 12 bits are `p`). The states with low 12 bits `g'` are `{(m << 12) | g'}`
// for m in [0,16), and `((m << 12) | g') >> 4 == (m << 8) | (g' >> 4)`, so
//
//     gmin_{t}[g'] = min over m of
//                      ( err_t( (m << 12) | g' ) + gmin_{t-1}[ (m << 8) | (g' >> 4) ] )
//
// — a recursion entirely on `gmin[4096]`. 4096 f32 = 16 KiB; ping-ponged, 32 KiB.
//
// The per-group argmin `m` is the backtrace: 4 bits per group per position,
// 2 KiB per position (vs the exhaustive kernel's 4 KiB, and against a 32 KiB
// per-state backtrace). The walk is closed on groups too, because the emitted
// symbol is the low K bits of the state and `(s & 0xFFF) & 0xF == s & 0xF`:
//
//     sym_t     = g_t & 0xF
//     g_{t-1}   = (arg_t[g_t] << 8) | (g_t >> 4)
//
// BIT-IDENTITY TO THE EXISTING EXHAUSTIVE DP
// ------------------------------------------
// The scan order and the tie-break are preserved exactly, so this is not an
// approximation of the exhaustive DP — it is the same DP with the redundant
// 16x state replication removed:
//
//   * `m` ascends 0..15 with a strict `<` update, which is the CPU's
//     `for j in 0..ALPHABET { if c < best_cost { ... } }` (first index wins
//     ties) and the `s_best_j` rule the exhaustive kernel uses.
//   * The branch metric is `err + gmin`, in that operand order, through
//     `qtip_exact_fp.cuh` — no FMA contraction, no approximate reciprocal.
//   * The final argmin is `min (cost, state)` over the 65,536 states, expressed
//     as a min over the 4,096 group representatives `(arg[g] << 12) | g`: within
//     a group the group-argmin is by construction the lowest-indexed state
//     attaining the group minimum, so the smallest state attaining the global
//     minimum is among the representatives. Same 64-bit ordered key the beam
//     kernel uses (`qtip_total_order_key(cost) << 16 | state`).
//
// `gmin_replay_matches_exhaustive_bit_for_bit` (viterbi.rs) proves the recursion
// on the CPU against `exhaustive_quantize_row`, with mutations asserted to
// diverge; `cuda_gmin_matches_cpu_exhaustive_bit_for_bit` (mod.rs) proves this
// kernel against the same CPU reference on hardware.
//
// THREAD MAPPING
// --------------
// One block per row, 256 threads. Thread `tid` owns the 16 groups
// `g_i = tid + 256*i`, i in [0,16). That mapping is chosen for the LUT:
// at fixed (m, i) the 32 lanes of a warp read 32 *consecutive* states, i.e. one
// contiguous 256-byte run, so the codebook read is fully coalesced. (The
// alternative — one thread per `g >> 4`, which would let the 16 predecessor
// minima be hoisted into 16 registers — makes each lane read a 128-byte run
// 128 bytes away from its neighbour's, i.e. 32 transactions per instruction.
// The shared-memory reads this mapping costs instead are 2-way broadcasts,
// which are free.)
//
// The predecessor read `s_gmin[(m << 8) | ((tid >> 4) + 16*i)]` has only two
// distinct addresses per warp (lanes 0-15 and 16-31), so it multicasts.
// The write `s_gmin[tid + 256*i]` is stride-1 across lanes: conflict-free.
//
// ONE `__syncthreads()` PER SYMBOL POSITION. The read set and the write set
// live in different halves of the ping-pong, so the only ordering requirement
// is "everyone finished writing before anyone reads next step".
//
// LAYOUT / LAUNCH
// ---------------
//   grid  = (rows_in_flight, 1, 1)      one block per row
//   block = (GM_THREADS = 256, 1, 1)
//   trace = [rows_in_flight, num_symbols, 512] u32; word `w` at position `t`
//           holds the 4-bit argmin of groups `{w + 512*u : u in [0,8)}` in
//           nibble `u`. 2 KiB per position, 19.4 MB per 9472-symbol row.
//           Position 0 is never read (the walk terminates there).
//
// SM80+ (gated by `has_qtip_kernels` in build.rs).

#include <cstdint>
#include <cuda_runtime.h>

#include "qtip_exact_fp.cuh"

namespace {

constexpr uint32_t GM_K           = 4;
constexpr uint32_t GM_L           = 16;
constexpr uint32_t GM_ALPHABET    = 1u << GM_K;            // 16
constexpr uint32_t GM_GROUP_BITS  = GM_L - GM_K;           // 12
constexpr uint32_t GM_GROUP_COUNT = 1u << GM_GROUP_BITS;   // 4096

constexpr int GM_THREADS = 256;
// Groups per thread. 4096 / 256 = 16, which is also the alphabet size, so the
// inner (i) loop and the (m) loop are both 16 wide.
constexpr int GM_GPT = (int)GM_GROUP_COUNT / GM_THREADS;

// Nibble-packed backtrace: 8 groups per u32.
constexpr int GM_WORDS = (int)GM_GROUP_COUNT / 8;          // 512
// Group -> (word, nibble): word = g % GM_WORDS, nibble = g / GM_WORDS.
static_assert(GM_WORDS == 2 * GM_THREADS,
              "each thread must contribute to exactly two trace words");

constexpr unsigned long long GM_KEY_MAX = ~0ull;

// Backtrace staging window, in symbol positions: the whole ping-pong is dead
// once the forward pass ends, so 32 KiB stages 16 positions of trace and the
// dependent pointer chase runs in shared memory instead of paying a full
// global-memory latency per position.
constexpr int GM_WALK_WINDOW = (2 * (int)GM_GROUP_COUNT) / GM_WORDS;   // 16

// `WRITE_TRACE == false` is a measurement-only variant: it runs the identical
// forward DP but emits neither the backtrace nor the packed symbols, which
// isolates the forward arithmetic from the trace traffic. It must never be
// reachable from a bake — `launch_qtip_quantize_rows_gmin_f32` only ever
// instantiates the tracing variants, and the no-trace one is behind the
// explicitly-named bench entry point.
template <int MIN_BLOCKS, bool WRITE_TRACE>
__global__ void __launch_bounds__(GM_THREADS, MIN_BLOCKS)
qtip_quantize_rows_gmin_kernel(
    const float*   __restrict__ weight,      // [n_rows, in_features]
    const float*   __restrict__ lut,         // [2^L * V]
    const float*   __restrict__ row_scales,  // [n_rows]
    uint8_t*       __restrict__ packed,      // [n_rows, num_symbols / 2]
    uint32_t*      __restrict__ trace,       // [BATCH, num_symbols, GM_WORDS]
    int in_features,
    int num_symbols,
    int row_offset
) {
    __shared__ float s_gmin[2][GM_GROUP_COUNT];    // 32 KiB ping-pong
    __shared__ unsigned long long s_fin;
    __shared__ unsigned int s_walk_g;

    const int local_row = blockIdx.x;
    const int row       = row_offset + local_row;
    const int tid       = (int)threadIdx.x;

    const float* __restrict__ my_row = weight + (size_t)row * (size_t)in_features;
    uint8_t*  __restrict__ my_pkd    = packed + (size_t)row * (size_t)(num_symbols / 2);
    uint32_t* __restrict__ my_trace  =
        trace + (size_t)local_row * (size_t)num_symbols * (size_t)GM_WORDS;

    const float inv_scale = qtip_inv_scale_exact(row_scales[row]);

    // All GM_GPT groups this thread owns share `g >> 4` only in its low bits;
    // `qbase` is the part that does not move with `i`.
    const unsigned int qbase = (unsigned int)tid >> GM_K;   // 0..15

    float        best[GM_GPT];
    unsigned int bm  [GM_GPT];

    // ---- t = 0 --------------------------------------------------------------
    // From the implicit start state 0 only states 0..15 are reachable, so
    // cost_0[s] = err_0(s) there and +inf elsewhere. Group g collects the states
    // {(m << 12) | g}, of which only m = 0 can be below 16 — hence
    // gmin_0[g] = err_0(g) for g < 16, +inf otherwise, and arg_0[g] = 0 (which
    // the walk never reads: it terminates by emitting `g_0 & 0xF`).
    {
        const float t0 = qtip_scaled_target_exact(my_row[0], inv_scale);
        const float t1 = qtip_scaled_target_exact(my_row[1], inv_scale);
        #pragma unroll
        for (int i = 0; i < GM_GPT; ++i) {
            const unsigned int g = (unsigned int)tid + (unsigned int)(GM_THREADS * i);
            const float c = (g < GM_ALPHABET)
                                ? qtip_decode_err_exact(lut, g, t0, t1)
                                : INFINITY;
            best[i]      = c;
            bm[i]        = 0u;
            s_gmin[0][g] = c;
        }
    }
    __syncthreads();

    int cur = 0;

    // ---- forward pass, t = 1 .. T-1 -----------------------------------------
    for (int t = 1; t < num_symbols; ++t) {
        const float t0 = qtip_scaled_target_exact(my_row[(size_t)t * 2 + 0], inv_scale);
        const float t1 = qtip_scaled_target_exact(my_row[(size_t)t * 2 + 1], inv_scale);

        const float* __restrict__ sp = s_gmin[cur];
        float*       __restrict__ sn = s_gmin[cur ^ 1];

        // m = 0 is peeled: seeding `best` with it is exactly what starting from
        // +inf under a strict `<` would produce, and saves a compare per group.
        {
            const float* __restrict__ lp = lut + (size_t)((unsigned int)tid * 2u);
            #pragma unroll
            for (int i = 0; i < GM_GPT; ++i) {
                const float cp = sp[qbase + 16u * (unsigned int)i];
                const float d0 = __fsub_rn(lp[512 * i + 0], t0);
                const float d1 = __fsub_rn(lp[512 * i + 1], t1);
                const float e  = __fadd_rn(__fmul_rn(d0, d0), __fmul_rn(d1, d1));
                best[i] = __fadd_rn(e, cp);
                bm[i]   = 0u;
            }
        }
        // The m loop is deliberately NOT unrolled: the body is already 16
        // independent 11-instruction chains (ample ILP), and unrolling it would
        // multiply a 176-instruction body by 15 while `best[]`/`bm[]` stay live
        // across all of it — which is how this kernel would end up spilling.
        // `best[]`/`bm[]` are indexed by the *inner* loop, which must stay fully
        // unrolled or they leave registers for local memory.
        #pragma unroll 1
        for (unsigned int m = 1; m < GM_ALPHABET; ++m) {
            const float* __restrict__ lp =
                lut + (size_t)(((m << GM_GROUP_BITS) | (unsigned int)tid) * 2u);
            #pragma unroll
            for (int i = 0; i < GM_GPT; ++i) {
                const float cp = sp[(m << 8) | (qbase + 16u * (unsigned int)i)];
                const float d0 = __fsub_rn(lp[512 * i + 0], t0);
                const float d1 = __fsub_rn(lp[512 * i + 1], t1);
                const float e  = __fadd_rn(__fmul_rn(d0, d0), __fmul_rn(d1, d1));
                const float c  = __fadd_rn(e, cp);
                if (c < best[i]) {
                    best[i] = c;
                    bm[i]   = m;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < GM_GPT; ++i) {
            sn[(unsigned int)tid + (unsigned int)(GM_THREADS * i)] = best[i];
        }

        if (WRITE_TRACE) {
            // g = tid + 256*i  =>  word = tid + 256*(i & 1), nibble = i >> 1.
            unsigned int w0 = 0u;
            unsigned int w1 = 0u;
            #pragma unroll
            for (int i = 0; i < GM_GPT; ++i) {
                if ((i & 1) == 0) w0 |= bm[i] << (4 * (i >> 1));
                else              w1 |= bm[i] << (4 * (i >> 1));
            }
            uint32_t* __restrict__ tw = my_trace + (size_t)t * (size_t)GM_WORDS;
            tw[tid]               = w0;
            tw[tid + GM_THREADS]  = w1;
        }

        __syncthreads();
        cur ^= 1;
    }

    if (!WRITE_TRACE) {
        // Measurement-only variant: keep the forward result alive so the
        // compiler cannot delete the DP, then stop.
        if (tid == 0 && best[0] == 12345.678f) my_pkd[0] = 1u;
        return;
    }

    // ---- best final state ----------------------------------------------------
    // `best[]`/`bm[]` still hold position T-1 (or t=0 when T == 1).
    if (tid == 0) s_fin = GM_KEY_MAX;
    __syncthreads();
    unsigned long long myk = GM_KEY_MAX;
    #pragma unroll
    for (int i = 0; i < GM_GPT; ++i) {
        const unsigned int g  = (unsigned int)tid + (unsigned int)(GM_THREADS * i);
        const unsigned int st = (bm[i] << GM_GROUP_BITS) | g;
        const unsigned long long k =
            ((unsigned long long)qtip_total_order_key(best[i]) << 16) | (unsigned long long)st;
        if (k < myk) myk = k;
    }
    atomicMin(&s_fin, myk);
    __syncthreads();

    // ---- backtrace + pack ----------------------------------------------------
    {
        const int ppr = num_symbols / 2;
        for (int i = tid; i < ppr; i += GM_THREADS) my_pkd[i] = 0u;

        uint32_t* s_stage = reinterpret_cast<uint32_t*>(&s_gmin[0][0]);
        __syncthreads();

        if (tid == 0) {
            const unsigned int gf = (unsigned int)(s_fin & (unsigned long long)(GM_GROUP_COUNT - 1u));
            s_walk_g = gf;
            const uint8_t sym = (uint8_t)(gf & (GM_ALPHABET - 1u));
            const int tl = num_symbols - 1;
            if ((tl & 1) == 0) my_pkd[tl / 2] |= sym;
            else               my_pkd[tl / 2] |= (uint8_t)(sym << 4);
        }
        __syncthreads();

        for (int t_hi = num_symbols - 1; t_hi >= 1; t_hi -= GM_WALK_WINDOW) {
            int t_lo = t_hi - GM_WALK_WINDOW + 1;
            if (t_lo < 1) t_lo = 1;
            const size_t span = (size_t)(t_hi - t_lo + 1) * (size_t)GM_WORDS;
            const uint32_t* __restrict__ src = my_trace + (size_t)t_lo * (size_t)GM_WORDS;
            for (size_t i = tid; i < span; i += GM_THREADS) s_stage[i] = src[i];
            __syncthreads();

            if (tid == 0) {
                unsigned int g = s_walk_g;
                for (int t = t_hi; t >= t_lo; --t) {
                    const uint32_t word =
                        s_stage[(size_t)(t - t_lo) * (size_t)GM_WORDS
                                + (size_t)(g % (unsigned int)GM_WORDS)];
                    const unsigned int m =
                        (word >> (4u * (g / (unsigned int)GM_WORDS))) & (GM_ALPHABET - 1u);
                    g = (m << (GM_GROUP_BITS - GM_K)) | (g >> GM_K);
                    const uint8_t sym = (uint8_t)(g & (GM_ALPHABET - 1u));
                    const int tp = t - 1;
                    if ((tp & 1) == 0) my_pkd[tp / 2] |= sym;
                    else               my_pkd[tp / 2] |= (uint8_t)(sym << 4);
                }
                s_walk_g = g;
            }
            __syncthreads();
        }
    }
}

} // anonymous namespace

// ============================================================================
// extern "C" launchers
// ============================================================================

extern "C" {

// u32 words of backtrace per symbol position per row. The host sizes the trace
// scratch from this rather than duplicating the packing rule.
int qtip_gmin_trace_words_per_position() { return GM_WORDS; }

// Production entry point: always the tracing kernel.
int launch_qtip_quantize_rows_gmin_f32(
    const float*  d_weight,
    const float*  d_lut,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    uint32_t*     d_trace,
    int n_rows,
    int in_features,
    int num_symbols,
    int row_offset,
    cudaStream_t  stream
) {
    if (n_rows <= 0 || num_symbols <= 0) return -1;
    qtip_quantize_rows_gmin_kernel<4, true><<<n_rows, GM_THREADS, 0, stream>>>(
        d_weight, d_lut, d_row_scales, d_packed, d_trace,
        in_features, num_symbols, row_offset);
    return 0;
}

// Bench-only entry point. `variant`:
//   0 = __launch_bounds__(256, 4), traced   (identical to the production path)
//   1 = __launch_bounds__(256, 2), traced   (register-pressure A/B)
//   2 = __launch_bounds__(256, 4), forward DP only, no trace and no pack
//       (isolates the forward arithmetic from the backtrace traffic; produces
//        NO valid output and is never reachable from a bake)
int launch_qtip_quantize_rows_gmin_variant_f32(
    const float*  d_weight,
    const float*  d_lut,
    const float*  d_row_scales,
    uint8_t*      d_packed,
    uint32_t*     d_trace,
    int n_rows,
    int in_features,
    int num_symbols,
    int row_offset,
    int variant,
    cudaStream_t  stream
) {
    if (n_rows <= 0 || num_symbols <= 0) return -1;
    switch (variant) {
        case 0:
            qtip_quantize_rows_gmin_kernel<4, true><<<n_rows, GM_THREADS, 0, stream>>>(
                d_weight, d_lut, d_row_scales, d_packed, d_trace,
                in_features, num_symbols, row_offset);
            return 0;
        case 1:
            qtip_quantize_rows_gmin_kernel<2, true><<<n_rows, GM_THREADS, 0, stream>>>(
                d_weight, d_lut, d_row_scales, d_packed, d_trace,
                in_features, num_symbols, row_offset);
            return 0;
        case 2:
            qtip_quantize_rows_gmin_kernel<4, false><<<n_rows, GM_THREADS, 0, stream>>>(
                d_weight, d_lut, d_row_scales, d_packed, d_trace,
                in_features, num_symbols, row_offset);
            return 0;
        default:
            return -1;
    }
}

} // extern "C"
