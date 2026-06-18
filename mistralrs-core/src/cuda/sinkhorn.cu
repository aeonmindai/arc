// Fused Sinkhorn normalization for V4 mHC (manifold-constrained hyper-connections).
//
// RUN-161 throughput: the eager candle path runs the 20-iteration Sinkhorn as
// ~123 separate tensor ops (each a kernel launch + allocation) on a tiny
// [N, hc, hc] = [1, 4, 4] matrix, twice per layer x 43 layers = ~13,000 serial
// launch-bound micro-kernels per decode token. This kernel collapses the entire
// normalization into ONE launch per call: one block per matrix, `hc` threads
// (one per row), the [hc, hc] tile lives in shared memory, all iterations run
// in-block with no global round-trips.
//
// Mirrors `sinkhorn_normalize` in models/dsv4_mhc.rs exactly:
//   1. row softmax: x = exp(c - rowmax) / rowsum ; then x += eps
//   2. initial col normalize: x = x / (colsum + eps)
//   3. (iters-1) more passes: row normalize x/(rowsum+eps), col normalize x/(colsum+eps)
//
// col_sum is the sum DOWN a column (over rows), matching candle's sum_keepdim(dim=1).

#include <cuda_runtime.h>
#include <math.h>

// hc_mult is 4 for V4-Flash; cap at 16 for safety (shared-mem sized at launch).
#define SINKHORN_MAX_HC 16

extern "C" {

// in/out: [n, hc, hc] row-major F32. One block per matrix `n`, `hc` threads.
// Shared memory: hc*hc (matrix tile) + hc (column sums) floats.
__global__ void sinkhorn_normalize_f32_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n,
    int hc,
    int iters,
    float eps
) {
    int batch = blockIdx.x;
    int row = threadIdx.x;
    if (batch >= n || row >= hc) return;

    extern __shared__ float smem[];
    float* mat = smem;          // [hc * hc]
    float* csum = smem + hc * hc; // [hc]

    const float* my_in = in + (size_t)batch * hc * hc + (size_t)row * hc;

    // Each thread owns one row in registers.
    float r[SINKHORN_MAX_HC];
    #pragma unroll
    for (int j = 0; j < hc; ++j) r[j] = my_in[j];

    // ---- 1. stable row softmax, then + eps ----
    float m = r[0];
    for (int j = 1; j < hc; ++j) m = fmaxf(m, r[j]);
    float rs = 0.0f;
    for (int j = 0; j < hc; ++j) { r[j] = expf(r[j] - m); rs += r[j]; }
    for (int j = 0; j < hc; ++j) r[j] = r[j] / rs + eps;

    // publish row to shared
    for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
    __syncthreads();

    // ---- 2. initial column normalize ----
    // column `row` sum = sum over all rows k of mat[k][row]
    float cs = 0.0f;
    for (int k = 0; k < hc; ++k) cs += mat[k * hc + row];
    csum[row] = cs + eps;
    __syncthreads();
    for (int j = 0; j < hc; ++j) r[j] = mat[row * hc + j] / csum[j];
    for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
    __syncthreads();

    // ---- 3. (iters - 1) more row->col passes ----
    for (int it = 0; it < iters - 1; ++it) {
        // row normalize (r already holds this thread's current row)
        float rsum = 0.0f;
        for (int j = 0; j < hc; ++j) rsum += r[j];
        rsum += eps;
        for (int j = 0; j < hc; ++j) r[j] = r[j] / rsum;
        for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
        __syncthreads();

        // column normalize
        float c2 = 0.0f;
        for (int k = 0; k < hc; ++k) c2 += mat[k * hc + row];
        csum[row] = c2 + eps;
        __syncthreads();
        for (int j = 0; j < hc; ++j) r[j] = mat[row * hc + j] / csum[j];
        for (int j = 0; j < hc; ++j) mat[row * hc + j] = r[j];
        __syncthreads();
    }

    // ---- write out ----
    float* my_out = out + (size_t)batch * hc * hc + (size_t)row * hc;
    for (int j = 0; j < hc; ++j) my_out[j] = r[j];
}

void sinkhorn_normalize_f32(
    const void* in,
    void* out,
    int n,
    int hc,
    int iters,
    float eps,
    long long stream
) {
    dim3 grid(n, 1, 1);
    dim3 block(hc, 1, 1);
    size_t shmem = (size_t)(hc * hc + hc) * sizeof(float);
    sinkhorn_normalize_f32_kernel<<<grid, block, shmem, (cudaStream_t)stream>>>(
        (const float*)in, (float*)out, n, hc, iters, eps);
}

} // extern "C"
