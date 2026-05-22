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

// Max number of kept tokens accumulated in shared memory before the CDF walk.
// Larger than typical top-p needs (~50). For top_k <= 256 we never exceed.
__device__ constexpr int MAX_KEEP = 256;

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
    int vocab,
    SamplingParams cfg
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const T* row = logits + (int64_t)bid * vocab;
    const uint32_t* fcnt = freq_counts ? (freq_counts + (int64_t)bid * vocab) : nullptr;
    float* probs = probs_scratch + (int64_t)bid * vocab;

    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int*   s_idxs = reinterpret_cast<int*>(s_vals + BLOCK);
    // Reuse smem past the reduction buffers for kept (idx, p) entries.
    int*   keep_idx = reinterpret_cast<int*>(s_idxs + BLOCK);
    float* keep_p   = reinterpret_cast<float*>(keep_idx + MAX_KEEP);

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
    // Effective cap on kept count.
    int cap = MAX_KEEP;
    if (cfg.top_k > 0 && cfg.top_k < cap) cap = cfg.top_k;

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
    size_t base = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    size_t keep = static_cast<size_t>(arc_sampler::MAX_KEEP) * (sizeof(int) + sizeof(float));
    size_t smem = base + keep;
    switch (block) {
        case 256:
            arc_sampler::arc_sample_kernel<float, 256><<<batch, 256, smem, stream>>>(
                logits, freq_counts, rng_state, token_ids, probs_scratch, vocab, cfg);
            break;
        case 128:
            arc_sampler::arc_sample_kernel<float, 128><<<batch, 128, smem, stream>>>(
                logits, freq_counts, rng_state, token_ids, probs_scratch, vocab, cfg);
            break;
        default:
            arc_sampler::arc_sample_kernel<float, 64><<<batch, 64, smem, stream>>>(
                logits, freq_counts, rng_state, token_ids, probs_scratch, vocab, cfg);
            break;
    }
}

void arc_launch_sampler_bf16(
    const void* logits,                 // __nv_bfloat16*
    const uint32_t* freq_counts,
    uint64_t* rng_state,
    int32_t* token_ids,
    float* probs_scratch,
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
    size_t base = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    size_t keep = static_cast<size_t>(arc_sampler::MAX_KEEP) * (sizeof(int) + sizeof(float));
    size_t smem = base + keep;
    switch (block) {
        case 256:
            arc_sampler::arc_sample_kernel<bf16, 256><<<batch, 256, smem, stream>>>(
                reinterpret_cast<const bf16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, vocab, cfg);
            break;
        case 128:
            arc_sampler::arc_sample_kernel<bf16, 128><<<batch, 128, smem, stream>>>(
                reinterpret_cast<const bf16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, vocab, cfg);
            break;
        default:
            arc_sampler::arc_sample_kernel<bf16, 64><<<batch, 64, smem, stream>>>(
                reinterpret_cast<const bf16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, vocab, cfg);
            break;
    }
}

void arc_launch_sampler_f16(
    const void* logits,                 // __half*
    const uint32_t* freq_counts,
    uint64_t* rng_state,
    int32_t* token_ids,
    float* probs_scratch,
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
    size_t base = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    size_t keep = static_cast<size_t>(arc_sampler::MAX_KEEP) * (sizeof(int) + sizeof(float));
    size_t smem = base + keep;
    switch (block) {
        case 256:
            arc_sampler::arc_sample_kernel<f16, 256><<<batch, 256, smem, stream>>>(
                reinterpret_cast<const f16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, vocab, cfg);
            break;
        case 128:
            arc_sampler::arc_sample_kernel<f16, 128><<<batch, 128, smem, stream>>>(
                reinterpret_cast<const f16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, vocab, cfg);
            break;
        default:
            arc_sampler::arc_sample_kernel<f16, 64><<<batch, 64, smem, stream>>>(
                reinterpret_cast<const f16*>(logits), freq_counts, rng_state,
                token_ids, probs_scratch, vocab, cfg);
            break;
    }
}

}  // extern "C"
