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

    const __nv_bfloat16* x = input + bid * hidden_size;
    const __nv_bfloat16* r = residual;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __bfloat162float(x[i]);
        if (r) val += __bfloat162float(r[bid * hidden_size + i]);
        sum_sq += val * val;
    }
    smem[tid] = sum_sq;
    __syncthreads();
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / (float)hidden_size + eps);

    __nv_bfloat16* o = output + bid * hidden_size;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __bfloat162float(x[i]);
        if (r) val += __bfloat162float(r[bid * hidden_size + i]);
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
    int smem = threads * sizeof(float);
    fused_rmsnorm_residual_bf16_kernel<<<batch_size, threads, smem, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)residual,
        (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, (__nv_bfloat16*)residual_out,
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
