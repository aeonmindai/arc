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
// Verified 2026-08-17 against docs.nvidia.com/cuda/parallel-thread-execution:
// PTX ISA sections 9.7.16.5.1.2.2 ("Matrix Descriptor Format") and 9.7.16.5.2
// ("wgmma.mma_async") are listed in the table of contents but their bodies are
// TRUNCATED in the published HTML. The bit table and the operand order are not
// obtainable from the doc. So they are not written from documentation here.
//
// WHAT THIS SWEEPS (gen 2)
// ------------------------
// The first cut of this probe swept only the two byte-offset fields, with the
// B tile fixed n-contiguous and `imm-trans-b` fixed at 0. That can only ever
// return NO_ENCODING_MATCHED if either of those two fixed choices is the wrong
// one, which costs a whole box round-trip to learn nothing. This version puts
// all four unknowns in the sweep:
//
//   1. `imm-trans-b`      -> 0 and 1
//   2. B smem layout      -> n-contiguous (s[k*N+n]) and k-contiguous (s[n*K+k])
//   3. leading-dim offset -> 10 candidate field values
//   4. stride offset      -> 10 candidate field values
//
// = 400 wgmma issues, one kernel launch per (layout, trans_b) pair.
//
// It also settles the instruction ARITY, which is the other thing the truncated
// doc will not say: the register-A form takes either
//   d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b        (7)
// or the same with an `imm-trans-a` before it                           (8).
// `-DWGMMA_HAS_TRANS_A=0/1` selects; the runner tries both and reports which
// one ptxas accepted. A wrong arity is a COMPILE error, which is loud — unlike
// a wrong descriptor, which is not. That asymmetry is why arity is settled by
// the compiler and the descriptor by the hardware.
//
// THE POISON POOL
// ---------------
// The B tile is 256 bytes. Placed alone in shared memory, a descriptor with a
// wrong stride can walk off it into whatever happens to be there and — with
// enough zeros around — still reproduce the reference. So the tile is embedded
// at the front of an 8 KB pool pre-filled with a large nonzero sentinel: any
// read outside the tile perturbs the sum, and the comparison is bit-exact
// rather than tolerance-based. A tolerance is how a wrong descriptor sneaks
// through.
//
// WHAT IT PROVES / DOES NOT PROVE
// -------------------------------
//   * PROVES (on the card it runs on): which descriptor field encoding, B
//     layout and trans_b make wgmma read the tile we think it is reading, for
//     swizzle mode 0.
//   * DOES NOT PROVE anything about sm_100a `tcgen05`, which takes no register
//     operands at all and has its own descriptor. Blackwell is compile-checked
//     separately; we do not rent Blackwell today, and a compile pass is not a
//     measurement.
//   * DOES NOT cover swizzled modes (1/2/3). The keystone kernel's decode
//     stage writes its own smem tile, so it can choose to be unswizzled.
//
// D18: every terminal outcome prints a line. "No candidate matched" is a
// result, not silence, and the mma.sync control must pass before any wgmma
// result is believed — two wrong runs that agree are not evidence.
//
// Build (on a Hopper box):
//   nvcc -arch=sm_90a -std=c++17 -O3 -DWGMMA_HAS_TRANS_A=0 \
//        -o wgmma_desc_probe arctarget_wgmma_desc_probe.cu
//
// Exit codes: 0 = a unique encoding was identified · 1 = probe answered
// negatively (no candidate, or ambiguous) · 2 = could not run (no device,
// launch failure) — never collapse 2 into 1.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef WGMMA_HAS_TRANS_A
#define WGMMA_HAS_TRANS_A 0
#endif

// m64n8k16: A is 64x16 bf16, B is 16x8 bf16, D is 64x8 f32.
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

// Shared pool, in bf16 elements. The 128-element (256 B) B tile sits at the
// front; the rest is POISON.
static constexpr int POOL = 4096;
static constexpr float POISON = 64.0f;

// Deterministic, small-magnitude, exactly representable in bf16 so the
// comparison can be bit-exact.
__device__ __host__ __forceinline__ float a_val(int m, int k) {
    return (float)(((m * 7 + k * 3) % 9) - 4);
}
__device__ __host__ __forceinline__ float b_val(int k, int n) {
    return (float)(((k * 5 + n * 11) % 7) - 3);
}

// LAYOUT 0: n-contiguous, s[k*N + n].   LAYOUT 1: k-contiguous, s[n*K + k].
// Both place the tile inside the first 128 elements of the pool.
template <int LAYOUT>
__device__ __host__ __forceinline__ int b_index(int k, int n) {
    return LAYOUT == 0 ? (k * N + n) : (n * K + k);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(lo)) |
           ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}

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

// The operand tail after the b-descriptor. Arity is what the compiler decides;
// see the header note.
#if WGMMA_HAS_TRANS_A
#define WGMMA_TAIL(tb) "p, 1, 1, 0, " #tb ";\n"
#else
#define WGMMA_TAIL(tb) "p, 1, 1, " #tb ";\n"
#endif

// One wgmma with A from registers and B from a shared-memory descriptor.
// Register-A halves the unknowns: if the result is wrong, the descriptor is
// the only thing it can be.
#define DEFINE_WGMMA_REGA(TB)                                                  \
    __device__ __forceinline__ void wgmma_m64n8k16_regA_##TB(                  \
        float (&d)[4], const uint32_t (&a)[4], uint64_t b_desc) {              \
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");              \
        asm volatile(                                                          \
            "{\n"                                                              \
            "  .reg .pred p;\n"                                                \
            "  setp.ne.b32 p, 1, 0;\n"                                         \
            "  wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "           \
            "  {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, " WGMMA_TAIL(TB)              \
            "}\n"                                                              \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                   \
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b_desc));        \
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");       \
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");       \
    }
DEFINE_WGMMA_REGA(0)
DEFINE_WGMMA_REGA(1)

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

template <int LAYOUT, int TRANS_B>
__global__ void probe_kernel(float* __restrict__ out_wgmma,  // [NCAND*NCAND][M*N]
                             float* __restrict__ out_mma,    // [M*N]
                             int base_offset, int swizzle) {
    __shared__ __align__(128) __nv_bfloat16 s_pool[POOL];
    const int tid = threadIdx.x;

    // Poison first, tile second: a descriptor that walks off the tile reads a
    // large nonzero sentinel and cannot accidentally reproduce the reference.
    for (int i = tid; i < POOL; i += WG_THREADS) s_pool[i] = __float2bfloat16(POISON);
    __syncthreads();
    for (int i = tid; i < K * N; i += WG_THREADS) {
        const int k = i / N, n = i % N;
        s_pool[b_index<LAYOUT>(k, n)] = __float2bfloat16(b_val(k, n));
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

    // ---- control: mma.sync over the same shared-memory bytes ----
    {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t b0 =
            pack_bf16(__bfloat162float(s_pool[b_index<LAYOUT>(tig * 2, g)]),
                      __bfloat162float(s_pool[b_index<LAYOUT>(tig * 2 + 1, g)]));
        const uint32_t b1 =
            pack_bf16(__bfloat162float(s_pool[b_index<LAYOUT>(tig * 2 + 8, g)]),
                      __bfloat162float(s_pool[b_index<LAYOUT>(tig * 2 + 9, g)]));
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
            const uint64_t desc = make_desc(&s_pool[0], CAND[li], CAND[si],
                                            (uint32_t)base_offset, (uint32_t)swizzle);
            if constexpr (TRANS_B == 0) {
                wgmma_m64n8k16_regA_0(d, a, desc);
            } else {
                wgmma_m64n8k16_regA_1(d, a, desc);
            }
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

struct ConfigResult {
    int matches = 0;
    int best_hits = -1;
    int best_idx = -1;
    bool ran = false;
};

// Returns 0 on success, 2 if the configuration could not be run at all.
template <int LAYOUT, int TRANS_B>
static int run_config(const char* layout_name, const float* ref, float* d_wgmma,
                      float* d_mma, float* h_wgmma, float* h_mma, ConfigResult& out) {
    const int variants = NCAND * NCAND;
    cudaMemset(d_wgmma, 0, (size_t)variants * M * N * sizeof(float));
    cudaMemset(d_mma, 0, (size_t)M * N * sizeof(float));

    probe_kernel<LAYOUT, TRANS_B>
        <<<1, WG_THREADS>>>(d_wgmma, d_mma, /*base_offset=*/0, /*swizzle=*/0);
    // D18 #8: cudaDeviceSynchronize() returns success for a kernel that never
    // launched. The launch error lives in cudaGetLastError and must be read.
    cudaError_t launch_err = cudaGetLastError();
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess || sync_err != cudaSuccess) {
        printf("PROBE_CONFIG layout=%s trans_b=%d STATUS=LAUNCH_FAILED launch=%s sync=%s\n",
               layout_name, TRANS_B, cudaGetErrorString(launch_err),
               cudaGetErrorString(sync_err));
        return 2;
    }

    cudaMemcpy(h_mma, d_mma, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wgmma, d_wgmma, (size_t)variants * M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    int control_bad = 0;
    for (int i = 0; i < M * N; ++i)
        if (h_mma[i] != ref[i]) ++control_bad;
    printf("PROBE_CONTROL_MMA_SYNC layout=%s trans_b=%d %s mismatches=%d/%d\n", layout_name,
           TRANS_B, control_bad ? "FAIL" : "PASS", control_bad, M * N);
    if (control_bad) {
        // The harness disagrees with arithmetic we already trust; nothing the
        // wgmma sweep says for this config can be believed.
        printf("PROBE_CONFIG layout=%s trans_b=%d STATUS=HARNESS_INVALID\n", layout_name,
               TRANS_B);
        return 2;
    }

    out.ran = true;
    for (int v = 0; v < variants; ++v) {
        const float* o = h_wgmma + (size_t)v * M * N;
        int hits = 0;
        for (int i = 0; i < M * N; ++i)
            if (o[i] == ref[i]) ++hits;
        if (hits > out.best_hits) {
            out.best_hits = hits;
            out.best_idx = v;
        }
        if (hits == M * N) {
            ++out.matches;
            printf("PROBE_MATCH layout=%s trans_b=%d lbo_field=%u sbo_field=%u "
                   "base_offset=0 swizzle=0\n",
                   layout_name, TRANS_B, CAND[v / NCAND], CAND[v % NCAND]);
        }
    }
    printf("PROBE_BEST_PARTIAL layout=%s trans_b=%d lbo_field=%u sbo_field=%u hits=%d/%d\n",
           layout_name, TRANS_B, CAND[out.best_idx / NCAND], CAND[out.best_idx % NCAND],
           out.best_hits, M * N);
    printf("PROBE_CONFIG layout=%s trans_b=%d matches=%d of %d\n", layout_name, TRANS_B,
           out.matches, variants);
    return 0;
}

int main() {
    printf("PROBE_ARITY_BUILT WGMMA_HAS_TRANS_A=%d\n", WGMMA_HAS_TRANS_A);

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

    float ref[M * N];
    host_reference(ref);
    float* h_mma = (float*)malloc((size_t)M * N * sizeof(float));
    float* h_wgmma = (float*)malloc((size_t)variants * M * N * sizeof(float));
    if (!h_mma || !h_wgmma) {
        printf("PROBE_STATUS=ALLOC_FAILED host\n");
        return 2;
    }

    ConfigResult r[4];
    int hard_fail = 0;
    hard_fail |= run_config<0, 0>("n_contiguous", ref, d_wgmma, d_mma, h_wgmma, h_mma, r[0]);
    hard_fail |= run_config<0, 1>("n_contiguous", ref, d_wgmma, d_mma, h_wgmma, h_mma, r[1]);
    hard_fail |= run_config<1, 0>("k_contiguous", ref, d_wgmma, d_mma, h_wgmma, h_mma, r[2]);
    hard_fail |= run_config<1, 1>("k_contiguous", ref, d_wgmma, d_mma, h_wgmma, h_mma, r[3]);

    int total_matches = 0, configs_ran = 0;
    for (const auto& c : r) {
        total_matches += c.matches;
        configs_ran += c.ran ? 1 : 0;
    }
    printf("PROBE_TOTAL matches=%d across %d/4 configs that ran\n", total_matches, configs_ran);

    free(h_mma);
    free(h_wgmma);
    cudaFree(d_wgmma);
    cudaFree(d_mma);

    // "Could not run" outranks any verdict: an environment that failed to
    // answer must never be reported as an answer (D18 rule 2).
    if (configs_ran == 0) {
        printf("PROBE_STATUS=COULD_NOT_RUN\n");
        return 2;
    }
    if (hard_fail) {
        printf("PROBE_STATUS=PARTIAL_ENVIRONMENT_FAILURE ran=%d/4\n", configs_ran);
        return 2;
    }
    if (total_matches == 1) {
        printf("PROBE_STATUS=UNIQUE_ENCODING_IDENTIFIED\n");
        return 0;
    }
    if (total_matches == 0) {
        // Either the bit layout in make_desc is wrong, or the accumulator
        // fragment layout assumed above is. The partial-hit count separates
        // them: a wrong accumulator layout still reproduces the right VALUES
        // in the wrong places, so hits stays near zero while the multiset of
        // outputs matches; a wrong descriptor reads wrong bytes entirely.
        printf("PROBE_STATUS=NO_ENCODING_MATCHED\n");
        return 1;
    }
    printf("PROBE_STATUS=AMBIGUOUS (%d encodings agree; the 16x8 tile cannot "
           "distinguish them — widen N so both offsets bind)\n",
           total_matches);
    return 1;
}
