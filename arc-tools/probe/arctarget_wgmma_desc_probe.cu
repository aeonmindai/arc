// ArcTarget probe: determine the wgmma shared-memory MATRIX DESCRIPTOR
// encoding EMPIRICALLY, on real Hopper hardware.
//
// WHY THIS EXISTS
// ---------------
// `wgmma.mma_async` (sm_90a) and `tcgen05.mma` (sm_100a+) are the two
// arch-native MMAs the keystone trellis GEMM wants, they are mutually
// exclusive, and both take their B operand through a 64-bit *shared-memory
// matrix descriptor* whose fields are specified semantically ("encoded value",
// "leading dimension byte offset") rather than literally. A descriptor that is
// wrong in any field does not fault: it reads the wrong bytes and the kernel
// returns WRONG NUMBERS. On the moat kernel, silently.
//
// So the encoding is not written from documentation here. This probe builds a
// tile whose correct product is known, sweeps every plausible descriptor
// encoding, and reports which ones reproduce the reference exactly. The
// hardware answers the question; we only read the answer.
//
// WHAT IT PROVES / DOES NOT PROVE
// -------------------------------
//   * PROVES (on the card it runs on): which descriptor field encoding makes
//     wgmma read the tile we think it is reading, for swizzle mode 0 and a
//     K-major B operand.
//   * DOES NOT PROVE anything about sm_100a `tcgen05`, which takes no register
//     operands at all and has its own descriptor. Blackwell is compile-checked
//     separately (see arctarget_tcgen05_compile_probe.cu); we do not rent
//     Blackwell today, and a compile pass is not a measurement.
//
// D18: every terminal outcome prints a line. "No candidate matched" is a
// result, not silence, and the mma.sync control must pass before any wgmma
// result is believed — two wrong runs that agree are not evidence.
//
// Build (on a Hopper box):
//   nvcc -arch=sm_90a -std=c++17 -O3 -o wgmma_desc_probe \
//        arctarget_wgmma_desc_probe.cu
//
// Exit codes: 0 = a unique encoding was identified · 1 = probe answered
// negatively (no candidate, or ambiguous) · 2 = could not run (no device,
// launch failure) — never collapse 2 into 1.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// m64n8k16: A is 64x16 bf16, B is 16x8 bf16 (K-major), D is 64x8 f32.
static constexpr int M = 64;
static constexpr int N = 8;
static constexpr int K = 16;
static constexpr int WG_THREADS = 128;  // one warpgroup

// Candidate encodings for the descriptor's byte-offset fields. The documented
// phrase is "encoded value", which in every published example means
// `(x & 0x3FFFF) >> 4` — but "which x" (row pitch in bytes? core-matrix
// stride? element count?) is the part that is not literal, so we sweep the
// resulting field value directly.
static constexpr uint32_t CAND[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256};
static constexpr int NCAND = sizeof(CAND) / sizeof(CAND[0]);

__device__ __forceinline__ uint64_t make_desc(const void* smem_ptr,
                                              uint32_t lbo_field,
                                              uint32_t sbo_field,
                                              uint32_t base_offset,
                                              uint32_t swizzle) {
    // Bit layout under test. Address in [0,14), leading-dim byte offset in
    // [16,30), stride byte offset in [32,46), base offset in [49,52),
    // swizzle in [62,64). Only the *field values* are swept; if this bit
    // layout itself is wrong, no candidate matches and the probe says so.
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFFu) >> 4) << 0;
    d |= (uint64_t)(lbo_field & 0x3FFFu) << 16;
    d |= (uint64_t)(sbo_field & 0x3FFFu) << 32;
    d |= (uint64_t)(base_offset & 0x7u) << 49;
    d |= (uint64_t)(swizzle & 0x3u) << 62;
    return d;
}

// One wgmma with A from registers and B from a shared-memory descriptor.
// Register-A halves the unknowns: if the result is wrong, the descriptor is
// the only thing it can be.
__device__ __forceinline__ void wgmma_m64n8k16_regA(float (&d)[4],
                                                    const uint32_t (&a)[4],
                                                    uint64_t b_desc) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 1, 0;\n"
        "  wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "  {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// The control: the mma.sync path the Ampere kernel already ships and that we
// trust. If this disagrees with the CPU reference, the harness is broken and
// no wgmma verdict means anything.
__device__ __forceinline__ void mma_m16n8k16(float* c, const uint32_t* a,
                                             uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(lo)) |
           ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}

// Deterministic, small-magnitude, exactly representable in bf16 so the
// comparison can be bit-exact rather than tolerance-based: a tolerance is how
// a wrong descriptor sneaks through.
__device__ __host__ __forceinline__ float a_val(int m, int k) {
    return (float)(((m * 7 + k * 3) % 9) - 4);
}
__device__ __host__ __forceinline__ float b_val(int k, int n) {
    return (float)(((k * 5 + n * 11) % 7) - 3);
}

__global__ void probe_kernel(float* __restrict__ out_wgmma,  // [NCAND*NCAND][M*N]
                             float* __restrict__ out_mma,    // [M*N]
                             int base_offset, int swizzle) {
    // B tile, K-major: 16 k-rows x 8 n-columns of bf16 = 256 B.
    __shared__ __align__(128) __nv_bfloat16 s_b[K * N];
    const int tid = threadIdx.x;
    for (int i = tid; i < K * N; i += WG_THREADS) {
        s_b[i] = __float2bfloat16(b_val(i / N, i % N));
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int row0 = warp * 16;  // this warp's 16 rows of the m64 tile

    // A fragment, mma.sync m16n8k16 layout, for this warp's rows.
    uint32_t a[4];
    a[0] = pack_bf16(a_val(row0 + g, tig * 2), a_val(row0 + g, tig * 2 + 1));
    a[1] = pack_bf16(a_val(row0 + g + 8, tig * 2), a_val(row0 + g + 8, tig * 2 + 1));
    a[2] = pack_bf16(a_val(row0 + g, tig * 2 + 8), a_val(row0 + g, tig * 2 + 9));
    a[3] = pack_bf16(a_val(row0 + g + 8, tig * 2 + 8), a_val(row0 + g + 8, tig * 2 + 9));

    // ---- control: mma.sync over the same data ----
    {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t b0 = pack_bf16((float)__bfloat162float(s_b[(tig * 2) * N + g]),
                                      (float)__bfloat162float(s_b[(tig * 2 + 1) * N + g]));
        const uint32_t b1 = pack_bf16((float)__bfloat162float(s_b[(tig * 2 + 8) * N + g]),
                                      (float)__bfloat162float(s_b[(tig * 2 + 9) * N + g]));
        mma_m16n8k16(c, a, b0, b1);
        out_mma[(row0 + g) * N + tig * 2] = c[0];
        out_mma[(row0 + g) * N + tig * 2 + 1] = c[1];
        out_mma[(row0 + g + 8) * N + tig * 2] = c[2];
        out_mma[(row0 + g + 8) * N + tig * 2 + 1] = c[3];
    }

    // ---- sweep: one wgmma per candidate (lbo, sbo) pair ----
    for (int li = 0; li < NCAND; ++li) {
        for (int si = 0; si < NCAND; ++si) {
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            const uint64_t desc =
                make_desc(&s_b[0], CAND[li], CAND[si], (uint32_t)base_offset, (uint32_t)swizzle);
            wgmma_m64n8k16_regA(d, a, desc);
            float* o = out_wgmma + (size_t)(li * NCAND + si) * (M * N);
            o[(row0 + g) * N + tig * 2] = d[0];
            o[(row0 + g) * N + tig * 2 + 1] = d[1];
            o[(row0 + g + 8) * N + tig * 2] = d[2];
            o[(row0 + g + 8) * N + tig * 2 + 1] = d[3];
            __syncthreads();
        }
    }
}

static void host_reference(float* ref) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += a_val(m, k) * b_val(k, n);
            ref[m * N + n] = acc;
        }
    }
}

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        printf("PROBE_STATUS=NO_DEVICE\n");
        return 2;  // could not answer != answered negatively
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("PROBE_DEVICE=%s sm_%d%d\n", prop.name, prop.major, prop.minor);
    if (prop.major != 9) {
        printf("PROBE_STATUS=WRONG_ARCH need sm_90x for wgmma, got sm_%d%d\n", prop.major,
               prop.minor);
        return 2;
    }

    const int variants = NCAND * NCAND;
    float* d_wgmma = nullptr;
    float* d_mma = nullptr;
    if (cudaMalloc(&d_wgmma, (size_t)variants * M * N * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_mma, (size_t)M * N * sizeof(float)) != cudaSuccess) {
        printf("PROBE_STATUS=ALLOC_FAILED\n");
        return 2;
    }
    cudaMemset(d_wgmma, 0, (size_t)variants * M * N * sizeof(float));
    cudaMemset(d_mma, 0, (size_t)M * N * sizeof(float));

    probe_kernel<<<1, WG_THREADS>>>(d_wgmma, d_mma, /*base_offset=*/0, /*swizzle=*/0);
    // D18 #8: cudaDeviceSynchronize() returns success for a kernel that never
    // launched. The launch error lives in cudaGetLastError and must be read.
    cudaError_t launch_err = cudaGetLastError();
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
        printf("PROBE_STATUS=LAUNCH_FAILED launch=%s sync=%s\n",
               cudaGetErrorString(launch_err), cudaGetErrorString(sync_err));
        return 2;
    }

    float ref[M * N];
    host_reference(ref);
    float* h_mma = (float*)malloc((size_t)M * N * sizeof(float));
    float* h_wgmma = (float*)malloc((size_t)variants * M * N * sizeof(float));
    cudaMemcpy(h_mma, d_mma, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wgmma, d_wgmma, (size_t)variants * M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    int control_bad = 0;
    for (int i = 0; i < M * N; ++i)
        if (h_mma[i] != ref[i]) ++control_bad;
    printf("PROBE_CONTROL_MMA_SYNC=%s mismatches=%d/%d\n", control_bad ? "FAIL" : "PASS",
           control_bad, M * N);
    if (control_bad) {
        // The harness disagrees with arithmetic we already trust; nothing the
        // wgmma sweep says can be believed.
        printf("PROBE_STATUS=HARNESS_INVALID\n");
        return 2;
    }

    int matches = 0, best_hits = -1, best_idx = -1;
    for (int v = 0; v < variants; ++v) {
        const float* o = h_wgmma + (size_t)v * M * N;
        int hits = 0;
        for (int i = 0; i < M * N; ++i)
            if (o[i] == ref[i]) ++hits;
        if (hits > best_hits) {
            best_hits = hits;
            best_idx = v;
        }
        if (hits == M * N) {
            ++matches;
            printf("PROBE_MATCH lbo_field=%u sbo_field=%u base_offset=0 swizzle=0\n",
                   CAND[v / NCAND], CAND[v % NCAND]);
        }
    }
    printf("PROBE_BEST_PARTIAL lbo_field=%u sbo_field=%u hits=%d/%d\n",
           CAND[best_idx / NCAND], CAND[best_idx % NCAND], best_hits, M * N);
    printf("PROBE_MATCH_COUNT=%d of %d candidates\n", matches, variants);

    free(h_mma);
    free(h_wgmma);
    cudaFree(d_wgmma);
    cudaFree(d_mma);

    if (matches == 1) {
        printf("PROBE_STATUS=UNIQUE_ENCODING_IDENTIFIED\n");
        return 0;
    }
    if (matches == 0) {
        // Either the bit layout in make_desc is wrong, or the accumulator
        // fragment layout assumed above is. The partial-hit count separates
        // them: a wrong accumulator layout still reproduces the right VALUES
        // in the wrong places, so hits stays near zero while the multiset of
        // outputs matches; a wrong descriptor reads wrong bytes entirely.
        printf("PROBE_STATUS=NO_ENCODING_MATCHED\n");
        return 1;
    }
    printf("PROBE_STATUS=AMBIGUOUS (several encodings agree; widen the tile so "
           "the offsets can be distinguished)\n");
    return 1;
}
