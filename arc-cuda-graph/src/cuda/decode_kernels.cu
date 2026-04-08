/**
 * Basic decode kernels: embedding gather, RMSNorm, residual add, SiLU*mul.
 * Used by the dedicated decode path that bypasses Candle.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Embedding gather
// ============================================================================

__global__ void gather_embedding_bf16_kernel(
    const __nv_bfloat16* __restrict__ table,
    const int32_t* __restrict__ ids,
    __nv_bfloat16* __restrict__ output,
    int hidden_size
) {
    int bid = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < hidden_size) {
        output[bid * hidden_size + tid] = table[ids[bid] * hidden_size + tid];
    }
}

extern "C" void launch_gather_embedding_bf16(
    const void* table, const int32_t* ids, void* output,
    int hidden_size, int batch_size, cudaStream_t stream
) {
    dim3 grid((hidden_size + 255) / 256, batch_size);
    gather_embedding_bf16_kernel<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)table, ids, (__nv_bfloat16*)output, hidden_size
    );
}

// ============================================================================
// Fused RMSNorm + optional residual add
// ============================================================================

// Shared-memory cached version: load input (and add residual) ONCE into smem,
// reuse for both the sum-of-squares pass and the normalize+write pass.
// Bench v80 showed the original kernel spent 60-70% of cycles in normalize+write
// due to re-reading input from L2 a second time.
__global__ void fused_rmsnorm_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ residual_out,
    int hidden_size, float eps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    extern __shared__ float smem[];
    // Layout: [hidden_size floats for cached input] [stride floats for reduction]
    float* cached = smem;                              // [hidden_size]
    float* red = smem + hidden_size;                   // [stride]

    const __nv_bfloat16* x = input + bid * hidden_size;
    const __nv_bfloat16* r = residual;

    // Pass 1: load input (+ residual) into shared, accumulate sum_sq.
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __bfloat162float(x[i]);
        if (r) val += __bfloat162float(r[bid * hidden_size + i]);
        cached[i] = val;
        sum_sq += val * val;
    }
    red[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / (float)hidden_size + eps);

    // Pass 2: read from shared, write residual_out (if requested) and norm output.
    __nv_bfloat16* o = output + bid * hidden_size;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = cached[i];
        if (residual_out) residual_out[bid * hidden_size + i] = __float2bfloat16(val);
        o[i] = __float2bfloat16(val * rms * __bfloat162float(weight[i]));
    }
}

extern "C" void launch_fused_rmsnorm_residual_bf16(
    const void* input, const void* residual, const void* weight,
    void* output, void* residual_out,
    int hidden_size, int batch_size, float eps, cudaStream_t stream
) {
    int threads = 256;
    // smem = hidden_size floats (cached input) + threads floats (reduction).
    int smem = (hidden_size + threads) * sizeof(float);
    fused_rmsnorm_residual_bf16_kernel<<<batch_size, threads, smem, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, (__nv_bfloat16*)residual_out,
        hidden_size, eps
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Clocked variant of fused_rmsnorm_residual: per-block stamps at:
//   phase 0 = entry
//   phase 1 = end of sum-of-squares pass
//   phase 2 = end of cross-thread reduction
//   phase 3 = end of normalize+write pass
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_rmsnorm_residual_bf16_clocked_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ residual_out,
    unsigned long long* __restrict__ clocks,
    int hidden_size, float eps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    extern __shared__ float smem[];
    float* cached = smem;
    float* red = smem + hidden_size;

    if (tid == 0) clocks[bid * 4 + 0] = clock64();

    const __nv_bfloat16* x = input + bid * hidden_size;
    const __nv_bfloat16* r = residual;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __bfloat162float(x[i]);
        if (r) val += __bfloat162float(r[bid * hidden_size + i]);
        cached[i] = val;
        sum_sq += val * val;
    }
    if (tid == 0) clocks[bid * 4 + 1] = clock64();

    red[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / (float)hidden_size + eps);

    if (tid == 0) clocks[bid * 4 + 2] = clock64();

    __nv_bfloat16* o = output + bid * hidden_size;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = cached[i];
        if (residual_out) residual_out[bid * hidden_size + i] = __float2bfloat16(val);
        o[i] = __float2bfloat16(val * rms * __bfloat162float(weight[i]));
    }

    if (tid == 0) clocks[bid * 4 + 3] = clock64();
}

extern "C" void launch_fused_rmsnorm_residual_bf16_clocked(
    const void* input, const void* residual, const void* weight,
    void* output, void* residual_out, void* clocks,
    int hidden_size, int batch_size, float eps, cudaStream_t stream
) {
    int threads = 256;
    int smem = (hidden_size + threads) * sizeof(float);
    fused_rmsnorm_residual_bf16_clocked_kernel<<<batch_size, threads, smem, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, (__nv_bfloat16*)residual_out,
        (unsigned long long*)clocks,
        hidden_size, eps
    );
}

// ============================================================================
// RMSNorm per head (for q_norm / k_norm)
// ============================================================================

__global__ void rmsnorm_head_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int head_dim, float eps
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    extern __shared__ float smem[];

    const __nv_bfloat16* x = input + head_idx * head_dim;
    __nv_bfloat16* o = output + head_idx * head_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += stride) {
        float v = __bfloat162float(x[i]);
        sum_sq += v * v;
    }
    smem[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += stride) {
        o[i] = __float2bfloat16(__bfloat162float(x[i]) * rms * __bfloat162float(weight[i]));
    }
}

extern "C" void launch_rmsnorm_head_bf16(
    const void* input, const void* weight, void* output,
    int head_dim, int total_heads, float eps, cudaStream_t stream
) {
    int threads = (head_dim <= 128) ? 128 : 256;
    int smem = threads * sizeof(float);
    rmsnorm_head_bf16_kernel<<<total_heads, threads, smem, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, head_dim, eps
    );
}

// Fused Q-norm + K-norm: one launch instead of two. Each block (1 warp = 32
// threads) handles one head. Bench v80 showed the previous 128-thread version
// spent 30% of cycles in __syncthreads reduction and 60% in nearly-empty
// pass loops (1 elem/thread). The 32-thread variant uses warp shuffle (no
// __syncthreads), each lane handles head_dim/32 = 4 elements for head_dim=128.
__global__ __launch_bounds__(32, 16)
void rmsnorm_qk_pair_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ q_w,
    __nv_bfloat16* __restrict__ q_out,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_w,
    __nv_bfloat16* __restrict__ k_out,
    int head_dim, int n_q_heads, float eps
) {
    int head_idx = blockIdx.x;
    bool is_q = head_idx < n_q_heads;
    int local = is_q ? head_idx : (head_idx - n_q_heads);
    const __nv_bfloat16* in_buf  = is_q ? q_in  : k_in;
    const __nv_bfloat16* w_buf   = is_q ? q_w   : k_w;
    __nv_bfloat16* out_buf       = is_q ? q_out : k_out;

    int lane = threadIdx.x;  // 0..31
    const __nv_bfloat16* x = in_buf + local * head_dim;
    __nv_bfloat16* o = out_buf + local * head_dim;

    // Sum of squares: each lane handles every 32nd element.
    float sum_sq = 0.0f;
    for (int i = lane; i < head_dim; i += 32) {
        float v = __bfloat162float(x[i]);
        sum_sq += v * v;
    }
    // Warp reduction (no __syncthreads required, all lanes in same warp).
    #pragma unroll
    for (int o_ = 16; o_ > 0; o_ >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, o_);
    }
    float rms = rsqrtf(sum_sq / (float)head_dim + eps);

    // Apply norm + write.
    for (int i = lane; i < head_dim; i += 32) {
        o[i] = __float2bfloat16(__bfloat162float(x[i]) * rms * __bfloat162float(w_buf[i]));
    }
}

extern "C" void launch_rmsnorm_qk_pair_bf16(
    const void* q_in, const void* q_w, void* q_out,
    const void* k_in, const void* k_w, void* k_out,
    int head_dim, int n_q_heads, int n_k_heads, float eps,
    cudaStream_t stream
) {
    int total_heads = n_q_heads + n_k_heads;
    rmsnorm_qk_pair_bf16_kernel<<<total_heads, 32, 0, stream>>>(
        (const __nv_bfloat16*)q_in, (const __nv_bfloat16*)q_w, (__nv_bfloat16*)q_out,
        (const __nv_bfloat16*)k_in, (const __nv_bfloat16*)k_w, (__nv_bfloat16*)k_out,
        head_dim, n_q_heads, eps
    );
}

// Clocked variant matching the new 32-thread warp-only design.
__global__ __launch_bounds__(32, 16)
void rmsnorm_qk_pair_bf16_clocked_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ q_w,
    __nv_bfloat16* __restrict__ q_out,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ k_w,
    __nv_bfloat16* __restrict__ k_out,
    unsigned long long* __restrict__ clocks,
    int head_dim, int n_q_heads, float eps
) {
    int head_idx = blockIdx.x;
    int lane = threadIdx.x;
    if (lane == 0) clocks[head_idx * 4 + 0] = clock64();

    bool is_q = head_idx < n_q_heads;
    int local = is_q ? head_idx : (head_idx - n_q_heads);
    const __nv_bfloat16* in_buf  = is_q ? q_in  : k_in;
    const __nv_bfloat16* w_buf   = is_q ? q_w   : k_w;
    __nv_bfloat16* out_buf       = is_q ? q_out : k_out;
    const __nv_bfloat16* x = in_buf + local * head_dim;
    __nv_bfloat16* o = out_buf + local * head_dim;

    float sum_sq = 0.0f;
    for (int i = lane; i < head_dim; i += 32) {
        float v = __bfloat162float(x[i]);
        sum_sq += v * v;
    }
    if (lane == 0) clocks[head_idx * 4 + 1] = clock64();

    #pragma unroll
    for (int o_ = 16; o_ > 0; o_ >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, o_);
    }
    float rms = rsqrtf(sum_sq / (float)head_dim + eps);

    if (lane == 0) clocks[head_idx * 4 + 2] = clock64();

    for (int i = lane; i < head_dim; i += 32) {
        o[i] = __float2bfloat16(__bfloat162float(x[i]) * rms * __bfloat162float(w_buf[i]));
    }

    if (lane == 0) clocks[head_idx * 4 + 3] = clock64();
}

extern "C" void launch_rmsnorm_qk_pair_bf16_clocked(
    const void* q_in, const void* q_w, void* q_out,
    const void* k_in, const void* k_w, void* k_out,
    void* clocks,
    int head_dim, int n_q_heads, int n_k_heads, float eps,
    cudaStream_t stream
) {
    int total_heads = n_q_heads + n_k_heads;
    rmsnorm_qk_pair_bf16_clocked_kernel<<<total_heads, 32, 0, stream>>>(
        (const __nv_bfloat16*)q_in, (const __nv_bfloat16*)q_w, (__nv_bfloat16*)q_out,
        (const __nv_bfloat16*)k_in, (const __nv_bfloat16*)k_w, (__nv_bfloat16*)k_out,
        (unsigned long long*)clocks,
        head_dim, n_q_heads, eps
    );
}

// ============================================================================
// SiLU * mul
// ============================================================================

__global__ void fused_silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        output[idx] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
    }
}

extern "C" void launch_fused_silu_mul_bf16(
    const void* gate, const void* up, void* output,
    int size, cudaStream_t stream
) {
    fused_silu_mul_bf16_kernel<<<(size + 255) / 256, 256, 0, stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
        (__nv_bfloat16*)output, size
    );
}

// ============================================================================
// Residual add
// ============================================================================

__global__ void residual_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

extern "C" void launch_residual_add_bf16(
    const void* a, const void* b, void* output,
    int size, cudaStream_t stream
) {
    residual_add_bf16_kernel<<<(size + 255) / 256, 256, 0, stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        (__nv_bfloat16*)output, size
    );
}

// ============================================================================
// Dtype cast kernels (for TurboQuant which needs F16 Q and outputs F32)
// ============================================================================

__global__ void cast_bf16_to_f16_kernel(
    const __nv_bfloat16* __restrict__ in, __half* __restrict__ out, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = __float2half(__bfloat162float(in[idx]));
}

extern "C" void launch_cast_bf16_to_f16(
    const void* in, void* out, int size, cudaStream_t stream
) {
    cast_bf16_to_f16_kernel<<<(size + 255) / 256, 256, 0, stream>>>(
        (const __nv_bfloat16*)in, (__half*)out, size
    );
}

__global__ void cast_f32_to_bf16_kernel(
    const float* __restrict__ in, __nv_bfloat16* __restrict__ out, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = __float2bfloat16(in[idx]);
}

extern "C" void launch_cast_f32_to_bf16(
    const void* in, void* out, int size, cudaStream_t stream
) {
    cast_f32_to_bf16_kernel<<<(size + 255) / 256, 256, 0, stream>>>(
        (const float*)in, (__nv_bfloat16*)out, size
    );
}
