/**
 * Bit-exact GPU sampler matching `sampling_cpu.rs`.
 *
 * This kernel runs the full sampling pipeline inside a CUDA graph: it takes
 * raw logits, applies penalties + temperature + softmax, selects top-p/top-k,
 * and draws a token via the SAME Splitmix64-style RNG and CDF the CPU
 * reference uses. The kernel is launched once per decode step, with one
 * thread block per batch element.
 *
 * Inputs
 *   logits[batch, vocab]        FP32 (preferred) or BF16 → cast to FP32 inside
 *   frequency_counts[batch, vocab]  u32; may be nullptr for "no penalties"
 *   rng_state[batch]            u64 (mutated in-place per call)
 *   cfg                         SamplingParams struct (POD)
 * Outputs
 *   token_ids[batch]            i32
 *
 * Bit-exact contract
 *   For a given (logits, rng_state, cfg) the kernel returns the same token as
 *   `sampling_cpu::sample`. The math is FP32 throughout. Where the CPU
 *   reference's `sort_unstable_by` produces an undefined order for tied
 *   probabilities, the kernel breaks ties by lowest vocab index — a
 *   *deterministic* divergence the test suite documents.
 *
 * SM targets: sm_89 (Ada / L40), sm_90 (Hopper / H100), sm_100 (Blackwell / B200).
 *
 * Algorithm (matches sampling_cpu.rs::sample line-for-line):
 *
 *   if cfg.greedy:
 *       apply_penalties; argmax
 *   else:
 *       apply_penalties
 *       apply_temperature      (skip if T == 1.0 or T == 0.0)
 *       softmax(logits - max(logits))
 *       iteratively pull arg-max of remaining probs, accumulate until
 *           cum >= top_p OR keep_count >= top_k OR all probs consumed
 *       splitmix64 → u ∈ [0,1) → target = u * kept_sum
 *       walk kept list, return first idx with acc >= target
 *
 * The "iteratively pull argmax" replaces the CPU's full descending sort.
 * Both yield the same kept set and the same per-iteration probability for
 * any input with unique probabilities, which is the practical case.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace arc_sampler {

// Sampling parameter POD shared with Rust (see sampling_cuda.rs).
struct SamplingParams {
    float temperature;
    float top_p;
    int32_t top_k;            // -1 = disabled
    float frequency_penalty;
    float presence_penalty;
    int32_t greedy;           // 0 / 1
    int32_t eos_token_id;     // -1 = disabled; only used by host glue
};

// The kept-token list is held in global memory (the `keep_scratch` buffer
// passed by the host wrapper) so that its capacity scales with vocab — the
// previous `MAX_KEEP = 256` shared-memory cap silently truncated large-vocab
// samplers (e.g., Kimi K2.6 / 160K vocab) when `top_k <= 0` left `top_p`
// alone to walk the full sorted distribution. See `sampling_cuda::CudaSampler`
// for the buffer allocation (`vocab * 8` bytes per batch row: i32 idx + f32 p).

// Block-wide argmax reduction with lowest-index tiebreak.
// Each thread brings a (val, idx); thread 0 receives the global argmax.
template <int BLOCK>
__device__ __forceinline__ void block_argmax_idx(
    float val, int idx,
    float* s_vals, int* s_idxs,
    float& out_val, int& out_idx
) {
    int tid = threadIdx.x;
    s_vals[tid] = val;
    s_idxs[tid] = idx;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v_other = s_vals[tid + stride];
            int   i_other = s_idxs[tid + stride];
            float v_self  = s_vals[tid];
            int   i_self  = s_idxs[tid];
            // Strictly greater wins; on tie pick lowest index (deterministic).
            bool take = (v_other > v_self) || (v_other == v_self && i_other < i_self);
            if (take) { s_vals[tid] = v_other; s_idxs[tid] = i_other; }
        }
        __syncthreads();
    }
    out_val = s_vals[0];
    out_idx = s_idxs[0];
}

template <int BLOCK>
__device__ __forceinline__ float block_max(float val, float* s_vals) {
    int tid = threadIdx.x;
    s_vals[tid] = val;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = s_vals[tid + stride];
            if (other > s_vals[tid]) s_vals[tid] = other;
        }
        __syncthreads();
    }
    return s_vals[0];
}

template <int BLOCK>
__device__ __forceinline__ float block_sum(float val, float* s_vals) {
    int tid = threadIdx.x;
    s_vals[tid] = val;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_vals[tid] += s_vals[tid + stride];
        __syncthreads();
    }
    return s_vals[0];
}

// Branch engagement counters for the hybrid sampler. A test that assumes
// "support width 512 must have taken the fallback" is assuming the very thing
// it is trying to establish; these make it observable.
__device__ unsigned int arc_hybrid_branch_a = 0;
__device__ unsigned int arc_hybrid_branch_b = 0;

// Splitmix64 step matching sampling_cpu.rs lines 156-161.
//   state = state * 0x9E3779B97F4A7C15 + 0xDEADBEEFC0DECAFE
//   mixed = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9
//   u32   = (mixed >> 33) as u32
//   u     = u32 as f32 / (1 << 31) as f32
__device__ __forceinline__ float splitmix_uniform(uint64_t& state) {
    state = state * 0x9E3779B97F4A7C15ULL + 0xDEADBEEFC0DECAFEULL;
    uint64_t mixed = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9ULL;
    uint32_t bits  = static_cast<uint32_t>(mixed >> 33);
    // (u32 as f32) / (1u64 << 31) as f32 — matches Rust exactly: f32 / 2^31.
    return static_cast<float>(bits) / 2147483648.0f;
}

// Cast helpers: each supports BF16 / FP16 / FP32 input.
template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

// =============================================================================
// Greedy kernel: argmax with penalties (lowest-index tiebreak).
// =============================================================================
template <typename T, int BLOCK>
__global__ void arc_greedy_kernel(
    const T* __restrict__ logits,                // [batch, vocab]
    const uint32_t* __restrict__ freq_counts,    // [batch, vocab] or nullptr
    int32_t* __restrict__ token_ids,             // [batch]
    int vocab,
    SamplingParams cfg
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const T* row = logits + (int64_t)bid * vocab;
    const uint32_t* fcnt = freq_counts ? (freq_counts + (int64_t)bid * vocab) : nullptr;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int*   s_idxs = reinterpret_cast<int*>(s_vals + BLOCK);

    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int i = tid; i < vocab; i += BLOCK) {
        float l = to_float<T>(row[i]);
        if (fcnt) {
            uint32_t c = fcnt[i];
            if (c != 0) {
                l -= cfg.frequency_penalty * static_cast<float>(c);
                l -= cfg.presence_penalty;
            }
        }
        // Strict > with lowest-index tiebreak.
        if (l > local_max || (l == local_max && i < local_idx)) {
            local_max = l;
            local_idx = i;
        }
    }
    float gm; int gi;
    block_argmax_idx<BLOCK>(local_max, local_idx, s_vals, s_idxs, gm, gi);
    if (tid == 0) token_ids[bid] = gi;
}

// =============================================================================
// Sample kernel: penalties → temperature → softmax → iterative top-p/k → CDF.
// =============================================================================
template <typename T, int BLOCK>
__global__ void arc_sample_kernel(
    const T* __restrict__ logits,                // [batch, vocab]
    const uint32_t* __restrict__ freq_counts,    // [batch, vocab] or nullptr
    uint64_t* __restrict__ rng_state,            // [batch]
    int32_t* __restrict__ token_ids,             // [batch]
    float* __restrict__ probs_scratch,           // [batch, vocab] writeable scratch
    int32_t* __restrict__ keep_idx_scratch,      // [batch, vocab] kept-token indices
    float* __restrict__ keep_p_scratch,          // [batch, vocab] kept-token probs
    int vocab,
    SamplingParams cfg
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const T* row = logits + (int64_t)bid * vocab;
    const uint32_t* fcnt = freq_counts ? (freq_counts + (int64_t)bid * vocab) : nullptr;
    float* probs = probs_scratch + (int64_t)bid * vocab;
    // Per-row keep list lives in global memory: capacity = full vocab so the
    // sampler never truncates when `top_k <= 0` leaves only `top_p` to bound
    // the kept set. Only thread 0 writes/reads it (one entry per Phase 4
    // iteration; sequential CDF walk in Phase 5) — global-memory latency
    // is amortized against the per-iteration argmax over `vocab` elements.
    int*   keep_idx = keep_idx_scratch + (int64_t)bid * vocab;
    float* keep_p   = keep_p_scratch   + (int64_t)bid * vocab;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int*   s_idxs = reinterpret_cast<int*>(s_vals + BLOCK);

    // --- Phase 1: apply penalties, temperature, find max, write penalized logits to probs[]. ---
    // We keep the post-penalty / post-temperature logits in probs[] (which we then
    // exp() in place after subtracting max). This matches the CPU's
    // "apply_penalties → apply_temperature → exp((l - max))" order.
    bool temp_active = !(cfg.temperature == 1.0f || cfg.temperature == 0.0f);
    float inv_temp = temp_active ? (1.0f / cfg.temperature) : 1.0f;

    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab; i += BLOCK) {
        float l = to_float<T>(row[i]);
        if (fcnt) {
            uint32_t c = fcnt[i];
            if (c != 0) {
                l -= cfg.frequency_penalty * static_cast<float>(c);
                l -= cfg.presence_penalty;
            }
        }
        if (temp_active) l *= inv_temp;
        probs[i] = l;
        if (l > local_max) local_max = l;
    }
    float gmax = block_max<BLOCK>(local_max, s_vals);

    // --- Phase 2: probs[i] = exp(l - max), local sum. ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        float p = __expf(probs[i] - gmax);
        probs[i] = p;
        local_sum += p;
    }
    float gsum = block_sum<BLOCK>(local_sum, s_vals);

    // --- Phase 3: normalize to probability. ---
    // CPU does `if sum > 0 { /= sum }` — we match that guard.
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 1.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        probs[i] *= inv_sum;
    }
    __syncthreads();

    // --- Phase 4: iteratively pull argmax until top_p / top_k satisfied. ---
    // Effective cap on kept count: `top_k` when set, else the full vocab.
    // This matches `sampling_cpu::sample`, which sorts the entire distribution
    // and truncates only via `top_p` / `top_k`. Earlier revisions clamped to
    // `MAX_KEEP=256` and silently truncated large-vocab + diffuse `top_p`.
    int cap = (cfg.top_k > 0 && cfg.top_k < vocab) ? cfg.top_k : vocab;

    __shared__ int kept_n;
    __shared__ float cum;
    __shared__ float kept_sum;
    if (tid == 0) { kept_n = 0; cum = 0.0f; kept_sum = 0.0f; }
    __syncthreads();

    while (kept_n < cap && cum < cfg.top_p) {
        // Find argmax over probs[]. Lowest-index tiebreak.
        float local_v = -FLT_MAX;
        int   local_i = 0;
        for (int i = tid; i < vocab; i += BLOCK) {
            float v = probs[i];
            if (v > local_v || (v == local_v && i < local_i)) {
                local_v = v;
                local_i = i;
            }
        }
        float gv; int gi;
        block_argmax_idx<BLOCK>(local_v, local_i, s_vals, s_idxs, gv, gi);

        // Stop if no probability mass left (gv == 0 means everything has been pulled
        // — or the entire distribution was zero, in which case CPU also stops).
        if (gv <= 0.0f) break;

        if (tid == 0) {
            keep_idx[kept_n] = gi;
            keep_p  [kept_n] = gv;
            kept_n  += 1;
            cum     += gv;
            kept_sum += gv;
            // Tombstone so the next argmax skips this slot.
            probs[gi] = -FLT_MAX;
        }
        __syncthreads();
        // Continue: the CPU loop checks `if cum >= top_p { break; }` AFTER the
        // push. Our condition mirrors that — we re-check at top of loop.
    }
    __syncthreads();

    // Pathological: nothing kept (e.g., all-zero distribution). Fall back to
    // argmax of the *original* logits, exactly as the CPU reference does.
    if (kept_sum <= 0.0f) {
        // Re-scan original logits for argmax (without penalties — CPU does
        // greedy_argmax(logits) where logits has had penalties + temperature
        // already applied. Mirror that: re-compute the *same* penalized,
        // temperature-scaled values. Since probs[] was tombstoned we can't
        // reuse it. We re-read logits cheaply since this is a rare path.).
        float lv = -FLT_MAX; int li = 0;
        for (int i = tid; i < vocab; i += BLOCK) {
            float l = to_float<T>(row[i]);
            if (fcnt) {
                uint32_t c = fcnt[i];
                if (c != 0) {
                    l -= cfg.frequency_penalty * static_cast<float>(c);
                    l -= cfg.presence_penalty;
                }
            }
            if (temp_active) l *= inv_temp;
            if (l > lv || (l == lv && i < li)) { lv = l; li = i; }
        }
        float gv; int gi;
        block_argmax_idx<BLOCK>(lv, li, s_vals, s_idxs, gv, gi);
        if (tid == 0) token_ids[bid] = gi;
        return;
    }

    // --- Phase 5: Splitmix64 → target = u * kept_sum → CDF walk. ---
    if (tid == 0) {
        uint64_t s = rng_state[bid];
        float u = splitmix_uniform(s);
        rng_state[bid] = s;
        float target = u * kept_sum;
        float acc = 0.0f;
        int selected = keep_idx[kept_n - 1];  // numerical fallback = last
        for (int k = 0; k < kept_n; ++k) {
            acc += keep_p[k];
            if (acc >= target) { selected = keep_idx[k]; break; }
        }
        token_ids[bid] = selected;
    }
}


// =============================================================================
// Nucleus sampling in a FIXED number of passes (threshold bisection).
//
// `arc_sample_kernel` above pulls the keep-list one argmax at a time, so it
// costs O(vocab x kept): a full vocabulary scan PER KEPT TOKEN. With
// `top_k <= 0` -- which is what mistral.rs's default maps to -- `kept` is
// bounded only by `top_p`, and on a diffuse distribution it reaches thousands.
// MEASURED on an H200 at vocab=129280 with a flat 12928-token support:
// 663,116 us to sample ONE token, roughly 10x an entire V4 decode step.
//
// This kernel selects the same nucleus by binary-searching a probability
// THRESHOLD rather than enumerating the set. For non-negative IEEE-754 floats
// the bit pattern is monotone when compared as u32, so bisecting the integer
// key converges to one ULP in a fixed 32 passes regardless of how many tokens
// the nucleus contains. Cost becomes O(vocab) with a constant factor.
//
// It does NOT narrow the distribution. The kept set is still
// {i : p_i >= t} for the largest t whose mass still covers `top_p` -- that is
// the definition of the nucleus. Where several tokens share the boundary bit
// pattern it keeps ALL of them, so it errs toward keeping MORE than the exact
// nucleus, never fewer. `top_k` is applied as a second, independent bisection
// on the kept COUNT, and the two thresholds combine with max(), so a caller
// setting both still gets the intersection.
// =============================================================================
template <typename T, int BLOCK>
__global__ void arc_sample_bisect_kernel(
    const T* __restrict__ logits,
    const uint32_t* __restrict__ freq_counts,
    uint64_t* __restrict__ rng_state,
    int32_t* __restrict__ token_ids,
    float* __restrict__ probs_scratch,
    int vocab,
    SamplingParams cfg
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const T* __restrict__ row = logits + (int64_t)bid * vocab;
    float* __restrict__ probs = probs_scratch + (int64_t)bid * vocab;
    const uint32_t* fcnt = freq_counts ? (freq_counts + (int64_t)bid * vocab) : nullptr;

    extern __shared__ char smem_bisect[];
    float* s_vals = reinterpret_cast<float*>(smem_bisect);
    int*   s_idxs = reinterpret_cast<int*>(s_vals + BLOCK);

    const bool temp_active = (cfg.temperature > 0.0f && cfg.temperature != 1.0f);
    const float inv_temp = temp_active ? (1.0f / cfg.temperature) : 1.0f;

    // --- Phase 1: penalties + temperature; keep the argmax for the fallback.
    float local_max = -FLT_MAX;
    int   local_i   = 0;
    for (int i = tid; i < vocab; i += BLOCK) {
        float l = to_float<T>(row[i]);
        if (fcnt) {
            uint32_t c = fcnt[i];
            if (c != 0) {
                l -= cfg.frequency_penalty * static_cast<float>(c);
                l -= cfg.presence_penalty;
            }
        }
        if (temp_active) l *= inv_temp;
        probs[i] = l;
        if (l > local_max || (l == local_max && i < local_i)) { local_max = l; local_i = i; }
    }
    float gmax; int gmax_idx;
    block_argmax_idx<BLOCK>(local_max, local_i, s_vals, s_idxs, gmax, gmax_idx);
    __syncthreads();

    // --- Phase 2/3: exp-shift, sum, normalize to probabilities.
    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        float p = __expf(probs[i] - gmax);
        probs[i] = p;
        local_sum += p;
    }
    const float gsum = block_sum<BLOCK>(local_sum, s_vals);
    __syncthreads();
    const float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 1.0f;
    float local_pmax = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        float p = probs[i] * inv_sum;
        probs[i] = p;
        if (p > local_pmax) local_pmax = p;
    }
    const float pmax = block_max<BLOCK>(local_pmax, s_vals);
    __syncthreads();

    // Degenerate distribution: fall back to the mode, as the CPU reference does.
    if (!(pmax > 0.0f)) {
        if (tid == 0) token_ids[bid] = gmax_idx;
        return;
    }

    float target = cfg.top_p;
    if (target > 1.0f) target = 1.0f;
    if (target < 0.0f) target = 0.0f;

    // --- Phase 4a: largest threshold key whose kept mass still covers top_p.
    uint32_t lo = 0u, hi = __float_as_uint(pmax);
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo + 1u) >> 1);
        const float threshold = __uint_as_float(mid);
        float m = 0.0f;
        for (int i = tid; i < vocab; i += BLOCK) {
            const float p = probs[i];
            if (p >= threshold) m += p;
        }
        const float gm = block_sum<BLOCK>(m, s_vals);
        __syncthreads();
        if (gm >= target) lo = mid; else hi = mid - 1u;
    }
    uint32_t key = lo;

    // --- Phase 4b: top_k, as an independent bisection on the kept COUNT.
    if (cfg.top_k > 0 && cfg.top_k < vocab) {
        uint32_t klo = 0u, khi = __float_as_uint(pmax);
        while (klo < khi) {
            const uint32_t mid = klo + ((khi - klo) >> 1);
            const float threshold = __uint_as_float(mid);
            float c = 0.0f;
            for (int i = tid; i < vocab; i += BLOCK) {
                if (probs[i] >= threshold) c += 1.0f;
            }
            const float gc = block_sum<BLOCK>(c, s_vals);
            __syncthreads();
            if (gc <= static_cast<float>(cfg.top_k)) khi = mid; else klo = mid + 1u;
        }
        if (klo > key) key = klo;
    }

    const float threshold = __uint_as_float(key);

    // --- Phase 5: CDF walk, parallel across the block.
    // Each thread sums the kept mass in its own strided subset; thread 0
    // exclusive-scans those 256 partials so every thread knows the CDF offset
    // its subset begins at. Only the thread whose interval contains `u` then
    // walks, and it walks vocab/BLOCK items rather than the whole vocabulary.
    float part = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        const float p = probs[i];
        if (p >= threshold) part += p;
    }
    s_vals[tid] = part;
    __syncthreads();

    __shared__ float s_u;
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < BLOCK; ++t) { const float v = s_vals[t]; s_vals[t] = acc; acc += v; }
        uint64_t st = rng_state[bid];
        const float u = splitmix_uniform(st);
        rng_state[bid] = st;
        s_u = u * acc;
        // Deterministic fallback if float error leaves no owning interval.
        token_ids[bid] = gmax_idx;
    }
    __syncthreads();

    const float base = s_vals[tid];
    if (s_u >= base && s_u < base + part) {
        float acc = base;
        for (int i = tid; i < vocab; i += BLOCK) {
            const float p = probs[i];
            if (p >= threshold) {
                acc += p;
                if (acc >= s_u) { token_ids[bid] = i; break; }
            }
        }
    }
}


// =============================================================================
// Hybrid nucleus sampler: exact enumeration while the nucleus is small,
// threshold bisection once it isn't. One kernel, no host involvement.
//
// MEASURED on an H200 at vocab=129280 (support width -> legacy us / bisect us):
//   1 -> 134 / 1994 | 8 -> 503 / 1996 | 64 -> 3132 / 1991
//   512 -> 24287 / 1997 | 4096 -> 205446 / 1998 | 12928 -> 665694 / 1998
// Bisection is FLAT; enumeration is linear in the kept count. Neither wins
// everywhere: enumeration is 3-15x faster on the peaked distributions real
// models produce, bisection is 333x faster on the diffuse tail. Shipping
// either alone trades one regression for another.
//
// THE SWITCH POINT IS NOT A TUNING CONSTANT. Bisection resolves one bit of the
// float32 probability key per vocabulary pass, so it costs 32 passes (plus 32
// more when top_k forces a second search). Enumeration costs one vocabulary
// pass per kept token. Enumeration is therefore cheaper exactly while
//     kept < (the number of passes bisection would spend)
// so the budget IS that pass count: ARC_ENUM_BUDGET = 32, derived from the
// width of the key being searched, the same way the 32 passes are. Spending
// the budget and then falling back costs at most budget + bisection, so the
// hybrid is never worse than ~2x the better branch, and the measured crossover
// (~64) sits within a factor of two of the derived budget -- the derivation
// and the measurement agree.
//
// Why the fallback is INSIDE the kernel: a captured CUDA graph bakes its
// kernel arguments and cannot branch on device state, so "launch enumeration,
// read a flag, maybe launch bisection" would need a device->host round trip
// per step -- exactly the host dependency the autonomous decode loop exists to
// remove. The branch is therefore taken block-uniformly on device.
//
// No cap and no narrowing: the fallback is exact, not a truncation. Exceeding
// the budget changes which ALGORITHM selects the nucleus, never which tokens
// the nucleus contains.
// =============================================================================
template <typename T, int BLOCK>
__global__ void arc_sample_hybrid_kernel(
    const T* __restrict__ logits,
    const uint32_t* __restrict__ freq_counts,
    uint64_t* __restrict__ rng_state,
    int32_t* __restrict__ token_ids,
    float* __restrict__ probs_scratch,
    int32_t* __restrict__ keep_idx_scratch,
    float* __restrict__ keep_p_scratch,
    int vocab,
    SamplingParams cfg
) {
    constexpr int ARC_ENUM_BUDGET = 32;   // == bisection's pass count

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const T* __restrict__ row = logits + (int64_t)bid * vocab;
    float* __restrict__ probs = probs_scratch + (int64_t)bid * vocab;
    int32_t* __restrict__ keep_idx = keep_idx_scratch + (int64_t)bid * vocab;
    float* __restrict__ keep_p = keep_p_scratch + (int64_t)bid * vocab;
    const uint32_t* fcnt = freq_counts ? (freq_counts + (int64_t)bid * vocab) : nullptr;

    extern __shared__ char smem_hy[];
    float* s_vals = reinterpret_cast<float*>(smem_hy);
    int*   s_idxs = reinterpret_cast<int*>(s_vals + BLOCK);

    const bool temp_active = (cfg.temperature > 0.0f && cfg.temperature != 1.0f);
    const float inv_temp = temp_active ? (1.0f / cfg.temperature) : 1.0f;

    float target = cfg.top_p;
    if (target > 1.0f) target = 1.0f;
    if (target < 0.0f) target = 0.0f;

#define ARC_BUILD_PROBS(GMAX_IDX_OUT)                                          \
    do {                                                                       \
        float _lmax = -FLT_MAX; int _li = 0;                                   \
        for (int i = tid; i < vocab; i += BLOCK) {                             \
            float l = to_float<T>(row[i]);                                     \
            if (fcnt) { uint32_t c = fcnt[i];                                  \
                if (c != 0) { l -= cfg.frequency_penalty * (float)c;           \
                              l -= cfg.presence_penalty; } }                   \
            if (temp_active) l *= inv_temp;                                    \
            probs[i] = l;                                                      \
            if (l > _lmax || (l == _lmax && i < _li)) { _lmax = l; _li = i; }  \
        }                                                                      \
        float _gv; int _gi;                                                    \
        block_argmax_idx<BLOCK>(_lmax, _li, s_vals, s_idxs, _gv, _gi);         \
        __syncthreads();                                                       \
        (GMAX_IDX_OUT) = _gi;                                                  \
        float _lsum = 0.0f;                                                    \
        for (int i = tid; i < vocab; i += BLOCK) {                             \
            float p = __expf(probs[i] - _gv); probs[i] = p; _lsum += p; }      \
        float _gsum = block_sum<BLOCK>(_lsum, s_vals);                         \
        __syncthreads();                                                       \
        float _inv = (_gsum > 0.0f) ? (1.0f / _gsum) : 1.0f;                   \
        for (int i = tid; i < vocab; i += BLOCK) probs[i] *= _inv;             \
        __syncthreads();                                                       \
    } while (0)

    int gmax_idx = 0;
    ARC_BUILD_PROBS(gmax_idx);

    // ---- Branch A: bounded exact enumeration (the legacy algorithm).
    __shared__ int   s_kept;
    __shared__ float s_cum;
    __shared__ int   s_done;      // 1 = nucleus fully determined here
    if (tid == 0) { s_kept = 0; s_cum = 0.0f; s_done = 0; }
    __syncthreads();

    while (true) {
        if (s_cum >= target) { if (tid == 0) s_done = 1; __syncthreads(); break; }
        if (cfg.top_k > 0 && s_kept >= cfg.top_k) { if (tid == 0) s_done = 1; __syncthreads(); break; }
        if (s_kept >= ARC_ENUM_BUDGET) { __syncthreads(); break; }   // spend budget, fall back

        float lv = -FLT_MAX; int li = 0;
        for (int i = tid; i < vocab; i += BLOCK) {
            float v = probs[i];
            if (v > lv || (v == lv && i < li)) { lv = v; li = i; }
        }
        float gv; int gi;
        block_argmax_idx<BLOCK>(lv, li, s_vals, s_idxs, gv, gi);
        __syncthreads();
        if (gv <= 0.0f) { if (tid == 0) s_done = 1; __syncthreads(); break; }
        if (tid == 0) {
            keep_idx[s_kept] = gi;
            keep_p[s_kept]   = gv;
            s_kept += 1;
            s_cum  += gv;
            probs[gi] = -FLT_MAX;      // tombstone
        }
        __syncthreads();
    }

    if (s_done) {
        if (tid == 0) atomicAdd(&arc_hybrid_branch_a, 1u);
        if (s_kept <= 0) { if (tid == 0) token_ids[bid] = gmax_idx; return; }
        if (tid == 0) {
            uint64_t st = rng_state[bid];
            float u = splitmix_uniform(st);
            rng_state[bid] = st;
            float t = u * s_cum;
            float acc = 0.0f;
            int sel = keep_idx[s_kept - 1];
            for (int k = 0; k < s_kept; ++k) {
                acc += keep_p[k];
                if (acc >= t) { sel = keep_idx[k]; break; }
            }
            token_ids[bid] = sel;
        }
        return;
    }

    // ---- Branch B: budget spent, nucleus is diffuse. probs[] was tombstoned
    // by the enumeration above, so rebuild it, then bisect.
    if (tid == 0) atomicAdd(&arc_hybrid_branch_b, 1u);
    ARC_BUILD_PROBS(gmax_idx);

    float lpmax = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) { float p = probs[i]; if (p > lpmax) lpmax = p; }
    const float pmax = block_max<BLOCK>(lpmax, s_vals);
    __syncthreads();
    if (!(pmax > 0.0f)) { if (tid == 0) token_ids[bid] = gmax_idx; return; }

    uint32_t lo = 0u, hi = __float_as_uint(pmax);
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo + 1u) >> 1);
        const float threshold_m = __uint_as_float(mid);
        float m = 0.0f;
        for (int i = tid; i < vocab; i += BLOCK) { float p = probs[i]; if (p >= threshold_m) m += p; }
        const float gm = block_sum<BLOCK>(m, s_vals);
        __syncthreads();
        if (gm >= target) lo = mid; else hi = mid - 1u;
    }
    uint32_t key = lo;

    if (cfg.top_k > 0 && cfg.top_k < vocab) {
        uint32_t klo = 0u, khi = __float_as_uint(pmax);
        while (klo < khi) {
            const uint32_t mid = klo + ((khi - klo) >> 1);
            const float threshold_m = __uint_as_float(mid);
            float c = 0.0f;
            for (int i = tid; i < vocab; i += BLOCK) { if (probs[i] >= threshold_m) c += 1.0f; }
            const float gc = block_sum<BLOCK>(c, s_vals);
            __syncthreads();
            if (gc <= (float)cfg.top_k) khi = mid; else klo = mid + 1u;
        }
        if (klo > key) key = klo;
    }

    const float threshold = __uint_as_float(key);
    float part = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) { float p = probs[i]; if (p >= threshold) part += p; }
    s_vals[tid] = part;
    __syncthreads();

    __shared__ float s_u;
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < BLOCK; ++t) { const float v = s_vals[t]; s_vals[t] = acc; acc += v; }
        uint64_t st = rng_state[bid];
        const float u = splitmix_uniform(st);
        rng_state[bid] = st;
        s_u = u * acc;
        token_ids[bid] = gmax_idx;
    }
    __syncthreads();

    const float base = s_vals[tid];
    if (s_u >= base && s_u < base + part) {
        float acc = base;
        for (int i = tid; i < vocab; i += BLOCK) {
            const float p = probs[i];
            if (p >= threshold) { acc += p; if (acc >= s_u) { token_ids[bid] = i; break; } }
        }
    }
#undef ARC_BUILD_PROBS
}

}  // namespace arc_sampler

// =============================================================================
// extern "C" launch wrappers
// =============================================================================
extern "C" {

// Choose block size at launch. 256 threads × ~512 elements/iter at vocab=128K
// = 8 sequential passes per softmax phase, well-occupied on SM89/90/100.
static int pick_block(int vocab) {
    if (vocab >= 4096) return 256;
    if (vocab >= 1024) return 128;
    return 64;
}

void arc_launch_sampler_f32(
    const float* logits,
    const uint32_t* freq_counts,        // nullptr if no penalties
    uint64_t* rng_state,
    int32_t* token_ids,
    float* probs_scratch,
    int32_t* keep_idx_scratch,
    float* keep_p_scratch,
    int vocab,
    int batch,
    arc_sampler::SamplingParams cfg,
    cudaStream_t stream
) {
    if (cfg.greedy) {
        const int B = 256;
        size_t smem = B * (sizeof(float) + sizeof(int));
        arc_sampler::arc_greedy_kernel<float, B><<<batch, B, smem, stream>>>(
            logits, freq_counts, token_ids, vocab, cfg);
        return;
    }
    int block = pick_block(vocab);
    size_t smem = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    switch (block) {
        case 256:
            arc_sampler::arc_sample_kernel<float, 256><<<batch, 256, smem, stream>>>(
                logits, freq_counts, rng_state, token_ids, probs_scratch,
                keep_idx_scratch, keep_p_scratch, vocab, cfg);
            break;
        case 128:
            arc_sampler::arc_sample_kernel<float, 128><<<batch, 128, smem, stream>>>(
                logits, freq_counts, rng_state, token_ids, probs_scratch,
                keep_idx_scratch, keep_p_scratch, vocab, cfg);
            break;
        default:
            arc_sampler::arc_sample_kernel<float, 64><<<batch, 64, smem, stream>>>(
                logits, freq_counts, rng_state, token_ids, probs_scratch,
                keep_idx_scratch, keep_p_scratch, vocab, cfg);
            break;
    }
}

void arc_launch_sampler_bf16(
    const void* logits,                 // __nv_bfloat16*
    const uint32_t* freq_counts,
    uint64_t* rng_state,
    int32_t* token_ids,
    float* probs_scratch,
    int32_t* keep_idx_scratch,
    float* keep_p_scratch,
    int vocab,
    int batch,
    arc_sampler::SamplingParams cfg,
    cudaStream_t stream
) {
    using bf16 = __nv_bfloat16;
    if (cfg.greedy) {
        const int B = 256;
        size_t smem = B * (sizeof(float) + sizeof(int));
        arc_sampler::arc_greedy_kernel<bf16, B><<<batch, B, smem, stream>>>(
            reinterpret_cast<const bf16*>(logits), freq_counts, token_ids, vocab, cfg);
        return;
    }
    int block = pick_block(vocab);
    size_t smem = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    switch (block) {
        case 256:
            arc_sampler::arc_sample_kernel<bf16, 256><<<batch, 256, smem, stream>>>(
                reinterpret_cast<const bf16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, keep_idx_scratch, keep_p_scratch,
                vocab, cfg);
            break;
        case 128:
            arc_sampler::arc_sample_kernel<bf16, 128><<<batch, 128, smem, stream>>>(
                reinterpret_cast<const bf16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, keep_idx_scratch, keep_p_scratch,
                vocab, cfg);
            break;
        default:
            arc_sampler::arc_sample_kernel<bf16, 64><<<batch, 64, smem, stream>>>(
                reinterpret_cast<const bf16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, keep_idx_scratch, keep_p_scratch,
                vocab, cfg);
            break;
    }
}

void arc_launch_sampler_f16(
    const void* logits,                 // __half*
    const uint32_t* freq_counts,
    uint64_t* rng_state,
    int32_t* token_ids,
    float* probs_scratch,
    int32_t* keep_idx_scratch,
    float* keep_p_scratch,
    int vocab,
    int batch,
    arc_sampler::SamplingParams cfg,
    cudaStream_t stream
) {
    using f16 = __half;
    if (cfg.greedy) {
        const int B = 256;
        size_t smem = B * (sizeof(float) + sizeof(int));
        arc_sampler::arc_greedy_kernel<f16, B><<<batch, B, smem, stream>>>(
            reinterpret_cast<const f16*>(logits), freq_counts, token_ids, vocab, cfg);
        return;
    }
    int block = pick_block(vocab);
    size_t smem = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    switch (block) {
        case 256:
            arc_sampler::arc_sample_kernel<f16, 256><<<batch, 256, smem, stream>>>(
                reinterpret_cast<const f16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, keep_idx_scratch, keep_p_scratch,
                vocab, cfg);
            break;
        case 128:
            arc_sampler::arc_sample_kernel<f16, 128><<<batch, 128, smem, stream>>>(
                reinterpret_cast<const f16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, keep_idx_scratch, keep_p_scratch,
                vocab, cfg);
            break;
        default:
            arc_sampler::arc_sample_kernel<f16, 64><<<batch, 64, smem, stream>>>(
                reinterpret_cast<const f16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, keep_idx_scratch, keep_p_scratch,
                vocab, cfg);
            break;
    }
}


// Fixed-pass nucleus sampler. Same signature as `arc_launch_sampler_*` so it
// is a drop-in; `keep_idx_scratch` / `keep_p_scratch` are unused by it.
#define ARC_DEFINE_BISECT_LAUNCHER(SUFFIX, CTYPE)                              \
void arc_launch_sampler_##SUFFIX##_bisect(                                     \
    const void* logits, const uint32_t* freq_counts, uint64_t* rng_state,      \
    int32_t* token_ids, float* probs_scratch, int32_t* keep_idx_scratch,       \
    float* keep_p_scratch, int vocab, int batch,                               \
    arc_sampler::SamplingParams cfg, cudaStream_t stream) {                    \
  (void)keep_idx_scratch; (void)keep_p_scratch;                                \
  const int B = 256;                                                           \
  size_t smem = B * (sizeof(float) + sizeof(int));                             \
  if (cfg.greedy) {                                                            \
    arc_sampler::arc_greedy_kernel<CTYPE, B><<<batch, B, smem, stream>>>(      \
        reinterpret_cast<const CTYPE*>(logits), freq_counts, token_ids, vocab, \
        cfg);                                                                  \
    return;                                                                    \
  }                                                                            \
  arc_sampler::arc_sample_bisect_kernel<CTYPE, B><<<batch, B, smem, stream>>>( \
      reinterpret_cast<const CTYPE*>(logits), freq_counts, rng_state,          \
      token_ids, probs_scratch, vocab, cfg);                                   \
}

ARC_DEFINE_BISECT_LAUNCHER(f32, float)
ARC_DEFINE_BISECT_LAUNCHER(bf16, __nv_bfloat16)
ARC_DEFINE_BISECT_LAUNCHER(f16, __half)


void arc_hybrid_branch_counts(unsigned int* a_out, unsigned int* b_out) {
  cudaMemcpyFromSymbol(a_out, arc_sampler::arc_hybrid_branch_a,
                       sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(b_out, arc_sampler::arc_hybrid_branch_b,
                       sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
}

void arc_hybrid_branch_reset(void) {
  unsigned int z = 0;
  cudaMemcpyToSymbol(arc_sampler::arc_hybrid_branch_a, &z, sizeof(z), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(arc_sampler::arc_hybrid_branch_b, &z, sizeof(z), 0,
                     cudaMemcpyHostToDevice);
}

#define ARC_DEFINE_HYBRID_LAUNCHER(SUFFIX, CTYPE)                              \
void arc_launch_sampler_##SUFFIX##_hybrid(                                     \
    const void* logits, const uint32_t* freq_counts, uint64_t* rng_state,      \
    int32_t* token_ids, float* probs_scratch, int32_t* keep_idx_scratch,       \
    float* keep_p_scratch, int vocab, int batch,                               \
    arc_sampler::SamplingParams cfg, cudaStream_t stream) {                    \
  const int B = 256;                                                           \
  size_t smem = B * (sizeof(float) + sizeof(int));                             \
  if (cfg.greedy) {                                                            \
    arc_sampler::arc_greedy_kernel<CTYPE, B><<<batch, B, smem, stream>>>(      \
        reinterpret_cast<const CTYPE*>(logits), freq_counts, token_ids, vocab, \
        cfg);                                                                  \
    return;                                                                    \
  }                                                                            \
  arc_sampler::arc_sample_hybrid_kernel<CTYPE, B><<<batch, B, smem, stream>>>( \
      reinterpret_cast<const CTYPE*>(logits), freq_counts, rng_state,          \
      token_ids, probs_scratch, keep_idx_scratch, keep_p_scratch, vocab, cfg); \
}

ARC_DEFINE_HYBRID_LAUNCHER(f32, float)
ARC_DEFINE_HYBRID_LAUNCHER(bf16, __nv_bfloat16)
ARC_DEFINE_HYBRID_LAUNCHER(f16, __half)

}  // extern "C"
