// Parent system: ArcQuant / TurboQuant — fused block-wise E4M3 activation
// quantizer for DeepSeek-V4's fused MQA key cache (ArcInfer / ArcKV / Fp8).
//
// WHY THIS FILE EXISTS
// -------------------
// `dsv4_kv_fp8::e4m3_codes_cpu` quantized K by copying the scaled block to the
// HOST, casting there (candle has no CUDA F8E4M3 cast) and copying back. On the
// V4-Flash B=1 decode step that is 43 `cuMemcpyDtoHAsync_v2` per token at
// ~109 us of *host* time each (pageable staging => blocking) = 4.81 ms/token,
// and it makes CUDA graph capture impossible: a graph cannot record a blocking
// D2H.
//
// The obvious sync-free replacement (`KvQuantMode::GpuApprox`, ~19 extra
// elementwise candle ops per layer) was MEASURED SLOWER on H200 - in a machine
// bottlenecked on op count, one sync is cheaper than the launches it takes to
// avoid it. So the fix has to be *one* launch, not *fewer syncs*: these two
// kernels collapse the whole quantize (~11 candle ops) and dequantize (~13
// candle ops) chains into a single kernel each.
//
// BIT-PARITY IS THE BAR - HOW IT IS OBTAINED
// -----------------------------------------
// The stored bytes must be identical to what the CPU path already wrote, or
// every cache written before this change decodes differently. Each arithmetic
// step below therefore names the candle op it reproduces:
//
//   amax   = max |x| over the block            <- abs() + max_keepdim()
//   scale  = amax * (float)(1/448) + 1e-12f    <- `(amax / 448.0)?.affine(1,1e-12)`
//                                                 NOTE candle's `Tensor / f64`
//                                                 is `affine(1.0/rhs, 0.0)`, a
//                                                 MULTIPLY by the reciprocal,
//                                                 not a divide.
//   code   = E4M3(x / scale)                   <- broadcast_div + CPU cast
//   value  = lut[code] * scale                 <- index_select + broadcast_mul
//
// Two hazards are handled explicitly:
//
// 1. THIS CRATE COMPILES WITH `--use_fast_math` (mistralrs-quant/build.rs), so
//    a bare `a / b` here would be `div.approx.f32` while candle-kernels (no
//    fast math) emits IEEE `div.rn.f32`, and `-ftz=true` would flush denormals
//    candle keeps. Every float operation below is therefore an explicit
//    `__f*_rn` intrinsic, which the CUDA Math API defines as IEEE-754 and
//    unaffected by `-prec-div`/`-ftz`/`-fmad`. `fabsf`/`fmaxf` are avoided in
//    favour of integer ops on the bit pattern for the same reason.
// 2. The E4M3 rounding is a transcription of NVIDIA's
//    `__nv_cvt_double_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` (cuda_fp8.hpp),
//    which is *also* what the Rust `float8` crate ports in `convert_to_fp8`
//    and therefore what `F8E4M3::from_f32` - candle's CPU cast - computes. It
//    is transcribed rather than #included so the parity claim is auditable
//    against the Rust source in one place, and so it stays pure integer
//    arithmetic that `--use_fast_math` cannot reach.
//
// The DEQUANT side does not convert code->float at all: it indexes the same
// 256-entry `f32` table the Rust side already builds with
// `F8E4M3::from_bits(i).to_f32()` and caches per device, so that half of the
// round trip is bit-exact by construction rather than by argument.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace arc {

// E4M3 max magnitude; `dsv4_kv_fp8::E4M3_MAX`.
#define ARC_KV_E4M3_MAX 448.0
// The warp width the block reduction assumes.
#define ARC_KV_WARP 32

// ---------------------------------------------------------------------------
// Activation dtype <-> f32, matching candle's cast kernels exactly.
//   f32 -> bf16 : candle `cast_f32_bf16` is `out[i] = inp[i]`, i.e. the
//                 `__nv_bfloat16(float)` constructor == `__float2bfloat16`
//                 (inline `cvt.rn.bf16.f32`, immune to -ftz).
//   f32 -> f16  : likewise `__float2half` (inline `cvt.rn.f16.f32`).
// ---------------------------------------------------------------------------
template <typename T> __device__ __forceinline__ float arc_to_f32(T v);
template <> __device__ __forceinline__ float arc_to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <> __device__ __forceinline__ float arc_to_f32<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float arc_to_f32<float>(float v) { return v; }

template <typename T> __device__ __forceinline__ T arc_from_f32(float v);
template <> __device__ __forceinline__ __nv_bfloat16 arc_from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ __forceinline__ __half arc_from_f32<__half>(float v) {
  return __float2half(v);
}
template <> __device__ __forceinline__ float arc_from_f32<float>(float v) { return v; }

// ---------------------------------------------------------------------------
// scale = (amax / 448.0) then +1e-12, reproducing `dsv4_kv_fp8::block_scale`:
//     (amax / E4M3_MAX)?.affine(1.0, 1e-12)
// candle lowers `Tensor / f64` to `affine(1.0/448.0, 0.0)` (bin_trait!(Div, ..,
// |v| 1./v, ..)), i.e. a multiply by the f32-rounded reciprocal, and lowers
// `affine(mul, add)` to `x * mul + add` with `mul`/`add` first narrowed to the
// tensor dtype. `x * 1.0f` is exact, so the second affine is exactly an add,
// and FMA contraction of `x * mul + 0.0f` is exactly the multiply. Both are
// therefore reproduced by one `__fmul_rn` and one `__fadd_rn`.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float arc_kv_block_scale(float amax) {
  const float inv_max = (float)(1.0 / ARC_KV_E4M3_MAX);
  const float eps = (float)1e-12;
  return __fadd_rn(__fmul_rn(amax, inv_max), eps);
}

// ---------------------------------------------------------------------------
// f32 -> E4M3 code byte.
//
// Transcription of NVIDIA `__nv_cvt_double_to_fp8(x, __NV_SATFINITE,
// __NV_E4M3)` (cuda_fp8.hpp), which the Rust `float8` crate ports verbatim as
// `convert_to_fp8`; `F8E4M3::from_f32(x)` is `convert_to_fp8(x as f64,
// SatFinite, E4M3)`, and `f32 -> f64` widening is exact. Every operation here
// is integer, so `--use_fast_math` cannot perturb it.
//
// Kept deliberately branch-shaped like the reference: the cost that mattered
// was a 109 us blocking memcpy, not a dozen integer ops.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t arc_f32_to_e4m3_code(float xf) {
  const double x = (double)xf; // exact
  const uint64_t xbits = (uint64_t)__double_as_longlong(x);

  const uint8_t FP8_MAXNORM = 0x7EU;
  const uint8_t FP8_MANTISSA_MASK = 0x07U;
  const int FP8_EXP_BIAS = 7;
  const int FP8_SIGNIFICAND_BITS = 4;
  const uint64_t FP8_MINDENORM_O2 = 0x3F50000000000000ULL;
  const uint64_t FP8_OVERFLOW_THRESHOLD = 0x407D000000000000ULL;
  const uint64_t FP8_MINNORM = 0x3F90000000000000ULL;
  const uint64_t DP_INF_BITS = 0x7FF0000000000000ULL;
  const uint64_t FP8_DP_HALF_ULP = (uint64_t)1 << (53 - FP8_SIGNIFICAND_BITS - 1);

  const uint8_t sign = (uint8_t)((xbits >> 63) << 7);
  const uint8_t exp =
      (uint8_t)((int)((xbits >> 52) & 0x7FFULL) - 1023 + FP8_EXP_BIAS);
  const uint8_t mantissa =
      (uint8_t)((xbits >> (53 - FP8_SIGNIFICAND_BITS)) & (uint64_t)FP8_MANTISSA_MASK);
  const uint64_t absx = xbits & 0x7FFFFFFFFFFFFFFFULL;

  uint8_t res;
  if (absx <= FP8_MINDENORM_O2) {
    // Zero or underflow.
    res = 0U;
  } else if (absx > DP_INF_BITS) {
    // Preserve NaNs (E4M3 has a single NaN encoding).
    res = 0x7FU;
  } else if (absx > FP8_OVERFLOW_THRESHOLD) {
    // SatFinite.
    res = FP8_MAXNORM;
  } else if (absx >= FP8_MINNORM) {
    // Normal range, round-to-nearest-even.
    res = (uint8_t)((uint8_t)(exp << (FP8_SIGNIFICAND_BITS - 1)) | mantissa);
    const uint64_t round = xbits & ((FP8_DP_HALF_ULP << 1) - 1);
    if ((round > FP8_DP_HALF_ULP) ||
        ((round == FP8_DP_HALF_ULP) && ((mantissa & 1U) != 0U))) {
      res = (uint8_t)(res + 1U);
    }
  } else {
    // Denormal range, round-to-nearest-even.
    const uint8_t shift = (uint8_t)(1 - (int)exp);
    const uint8_t man = (uint8_t)(mantissa | (uint8_t)(1U << (FP8_SIGNIFICAND_BITS - 1)));
    res = (uint8_t)(man >> shift);
    const uint64_t round =
        (xbits | ((uint64_t)1 << (53 - 1))) &
        ((FP8_DP_HALF_ULP << ((uint64_t)shift + 1)) - 1);
    if ((round > (FP8_DP_HALF_ULP << (uint64_t)shift)) ||
        ((round == (FP8_DP_HALF_ULP << (uint64_t)shift)) && ((res & 1U) != 0U))) {
      res = (uint8_t)(res + 1U);
    }
  }
  return (uint8_t)(res | sign);
}

// ---------------------------------------------------------------------------
// FUSED QUANTIZE. One launch replaces narrow/to_dtype/reshape/abs/max_keepdim/
// affine/affine/broadcast_div/cast/reshape/cat.
//
// One CUDA block per token (a token is one (b, h, t) triple); one warp per
// 64-wide quant block. The amax reduction runs on the ABSOLUTE-VALUE BIT
// PATTERN as an unsigned integer: for non-negative IEEE floats the unsigned
// integer order is the float order, so `max` over `bits & 0x7fffffff` is
// exactly `max |x|` with no rounding, no `fabsf`, and no exposure to -ftz.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void arc_kv_fp8_quantize_kernel(
    const T *__restrict__ k,     // [ntok, head_dim] contiguous
    uint8_t *__restrict__ codes, // [ntok, nope]
    T *__restrict__ side,        // [ntok, rope_dim + n_blocks]
    const int head_dim, const int nope, const int rope_dim, const int n_blocks,
    const int block_w, const long ntok) {
  const int side_w = rope_dim + n_blocks;
  const int lane = (int)(threadIdx.x & (ARC_KV_WARP - 1));
  const int warp = (int)(threadIdx.x / ARC_KV_WARP);
  const int n_warps = (int)(blockDim.x / ARC_KV_WARP);

  for (long tok = (long)blockIdx.x; tok < ntok; tok += (long)gridDim.x) {
    const T *krow = k + tok * (long)head_dim;
    uint8_t *crow = codes + tok * (long)nope;
    T *srow = side + tok * (long)side_w;

    for (int blk = warp; blk < n_blocks; blk += n_warps) {
      const int base = blk * block_w;

      unsigned amax_bits = 0U;
      for (int e = lane; e < block_w; e += ARC_KV_WARP) {
        const unsigned b = __float_as_uint(arc_to_f32<T>(krow[base + e])) & 0x7FFFFFFFU;
        amax_bits = amax_bits > b ? amax_bits : b;
      }
#pragma unroll
      for (int off = ARC_KV_WARP / 2; off > 0; off >>= 1) {
        const unsigned o = __shfl_xor_sync(0xFFFFFFFFU, amax_bits, off);
        amax_bits = amax_bits > o ? amax_bits : o;
      }
      const float amax = __uint_as_float(amax_bits);
      const float scale = arc_kv_block_scale(amax);

      for (int e = lane; e < block_w; e += ARC_KV_WARP) {
        const float v = arc_to_f32<T>(krow[base + e]);
        crow[base + e] = arc_f32_to_e4m3_code(__fdiv_rn(v, scale));
      }
      // `amax` IS one of the block's own elements, so narrowing it back to the
      // activation dtype is exact - that is what lets dequant rebuild the
      // identical scale.
      if (lane == 0) {
        srow[rope_dim + blk] = arc_from_f32<T>(amax);
      }
    }

    // The RoPE'd tail is stored verbatim (RoPE is applied before the
    // quantizer), and it is the FIRST `rope_dim` lanes of `side` -
    // `cat(&[k_rope, amax_stored])`.
    for (int i = (int)threadIdx.x; i < rope_dim; i += (int)blockDim.x) {
      srow[i] = krow[nope + i];
    }
  }
}

// ---------------------------------------------------------------------------
// FUSED DEQUANTIZE. One launch replaces flatten/to_dtype/index_select/reshape/
// narrow/to_dtype/reshape/affine/affine/broadcast_mul/reshape/to_dtype/narrow/
// to_dtype/cat/contiguous.
//
// `lut` is the 256-entry f32 table the Rust side builds from
// `F8E4M3::from_bits(i).to_f32()` and caches per device - the SAME tensor the
// candle path passed to `index_select`, so the code->value half is bit-exact by
// construction.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void arc_kv_fp8_dequantize_kernel(
    const uint8_t *__restrict__ codes, // [ntok, nope]
    const T *__restrict__ side,        // [ntok, rope_dim + n_blocks]
    const float *__restrict__ lut,     // [256]
    T *__restrict__ out,               // [ntok, head_dim] contiguous
    const int head_dim, const int nope, const int rope_dim, const int n_blocks,
    const int block_w, const long ntok) {
  const int side_w = rope_dim + n_blocks;
  const int lane = (int)(threadIdx.x & (ARC_KV_WARP - 1));
  const int warp = (int)(threadIdx.x / ARC_KV_WARP);
  const int n_warps = (int)(blockDim.x / ARC_KV_WARP);

  for (long tok = (long)blockIdx.x; tok < ntok; tok += (long)gridDim.x) {
    const uint8_t *crow = codes + tok * (long)nope;
    const T *srow = side + tok * (long)side_w;
    T *orow = out + tok * (long)head_dim;

    for (int blk = warp; blk < n_blocks; blk += n_warps) {
      const int base = blk * block_w;
      const float amax = arc_to_f32<T>(srow[rope_dim + blk]);
      const float scale = arc_kv_block_scale(amax);
      for (int e = lane; e < block_w; e += ARC_KV_WARP) {
        orow[base + e] = arc_from_f32<T>(__fmul_rn(lut[crow[base + e]], scale));
      }
    }

    // `cat(&[k_nope, k_rope])`: the RoPE'd tail follows the nope dims.
    for (int i = (int)threadIdx.x; i < rope_dim; i += (int)blockDim.x) {
      orow[nope + i] = srow[i];
    }
  }
}

} // namespace arc

// ---------------------------------------------------------------------------
// C ABI. `stream_ptr` is candle's stream so the launch is recorded into a CUDA
// graph captured on it (cf. kvwrite.cu; rotary.cu's hardcoded stream 0 is NOT
// capturable). Grid is capped so a long prefill loops instead of launching an
// unbounded grid.
// ---------------------------------------------------------------------------
#define ARC_KV_MAX_BLOCKS 65535

#define ARC_KV_LAUNCH_QUANT(T)                                                 \
  arc::arc_kv_fp8_quantize_kernel<T><<<grid, block, 0, stream>>>(              \
      reinterpret_cast<const T *>(k), codes, reinterpret_cast<T *>(side),      \
      head_dim, nope, rope_dim, n_blocks, block_w, ntok)

#define ARC_KV_LAUNCH_DEQUANT(T)                                               \
  arc::arc_kv_fp8_dequantize_kernel<T><<<grid, block, 0, stream>>>(            \
      codes, reinterpret_cast<const T *>(side), lut,                           \
      reinterpret_cast<T *>(out), head_dim, nope, rope_dim, n_blocks, block_w, \
      ntok)

extern "C" void arc_kv_fp8_quantize(
    const void *k,    // [ntok, head_dim] activation dtype
    uint8_t *codes,   // [ntok, nope]
    void *side,       // [ntok, rope_dim + n_blocks]
    int32_t head_dim, int32_t nope, int32_t rope_dim, int32_t n_blocks,
    int32_t block_w, int64_t ntok,
    void *stream_ptr, // cudaStream_t; null => default stream
    uint32_t dtype    // 0 => f16, 1 => bf16, 2 => f32
) {
  if (ntok <= 0 || n_blocks <= 0) {
    return;
  }
  // One warp per quant block, capped at 256 threads (8 warps).
  int warps = n_blocks < 8 ? n_blocks : 8;
  const int threads = warps * ARC_KV_WARP;
  long blocks = ntok < (long)ARC_KV_MAX_BLOCKS ? ntok : (long)ARC_KV_MAX_BLOCKS;
  dim3 grid((unsigned)blocks);
  dim3 block((unsigned)threads);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (dtype == 0) {
    ARC_KV_LAUNCH_QUANT(__half);
  } else if (dtype == 1) {
    ARC_KV_LAUNCH_QUANT(__nv_bfloat16);
  } else if (dtype == 2) {
    ARC_KV_LAUNCH_QUANT(float);
  }
}

extern "C" void arc_kv_fp8_dequantize(
    const uint8_t *codes, // [ntok, nope]
    const void *side,     // [ntok, rope_dim + n_blocks]
    const float *lut,     // [256]
    void *out,            // [ntok, head_dim]
    int32_t head_dim, int32_t nope, int32_t rope_dim, int32_t n_blocks,
    int32_t block_w, int64_t ntok, void *stream_ptr, uint32_t dtype) {
  if (ntok <= 0 || n_blocks <= 0) {
    return;
  }
  int warps = n_blocks < 8 ? n_blocks : 8;
  const int threads = warps * ARC_KV_WARP;
  long blocks = ntok < (long)ARC_KV_MAX_BLOCKS ? ntok : (long)ARC_KV_MAX_BLOCKS;
  dim3 grid((unsigned)blocks);
  dim3 block((unsigned)threads);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (dtype == 0) {
    ARC_KV_LAUNCH_DEQUANT(__half);
  } else if (dtype == 1) {
    ARC_KV_LAUNCH_DEQUANT(__nv_bfloat16);
  } else if (dtype == 2) {
    ARC_KV_LAUNCH_DEQUANT(float);
  }
}

// ---------------------------------------------------------------------------
// D33 NEGATIVE CONTROL. Same as `arc_kv_fp8_quantize` except the rounding is
// truncated (round-toward-zero) instead of round-to-nearest-even. It exists so
// the bit-parity test can be shown to FAIL on a wrong kernel: a parity check on
// this exact subsystem has already passed vacuously by comparing an
// implementation to itself.
//
// Nothing in the serving path calls this - it is reachable only from
// `arc_kvquant::mutant_quantize_for_parity_test`.
// ---------------------------------------------------------------------------
namespace arc {

// Identical to `arc_f32_to_e4m3_code` except that in the normal range it
// TRUNCATES the mantissa instead of rounding to nearest even - the single most
// plausible way to get an E4M3 quantizer subtly wrong. Everything else (the
// scale, the amax reduction, the layout, the rope copy) is untouched, so a
// parity test that survives this mutant is testing nothing.
__device__ __forceinline__ uint8_t arc_f32_to_e4m3_code_truncating(float xf) {
  const double x = (double)xf;
  const uint64_t xbits = (uint64_t)__double_as_longlong(x);
  const uint64_t absx = xbits & 0x7FFFFFFFFFFFFFFFULL;
  const uint64_t FP8_MINNORM = 0x3F90000000000000ULL;
  const uint64_t FP8_OVERFLOW_THRESHOLD = 0x407D000000000000ULL;
  if (absx < FP8_MINNORM || absx > FP8_OVERFLOW_THRESHOLD) {
    return arc_f32_to_e4m3_code(xf);
  }
  const uint8_t exp = (uint8_t)((int)((xbits >> 52) & 0x7FFULL) - 1023 + 7);
  const uint8_t mantissa = (uint8_t)((xbits >> 49) & 0x7ULL);
  const uint8_t sign = (uint8_t)((xbits >> 63) << 7);
  return (uint8_t)((uint8_t)((uint8_t)(exp << 3) | mantissa) | sign);
}

template <typename T>
__global__ void arc_kv_fp8_quantize_mutant_kernel(
    const T *__restrict__ k, uint8_t *__restrict__ codes, T *__restrict__ side,
    const int head_dim, const int nope, const int rope_dim, const int n_blocks,
    const int block_w, const long ntok) {
  const int side_w = rope_dim + n_blocks;
  const int lane = (int)(threadIdx.x & (ARC_KV_WARP - 1));
  const int warp = (int)(threadIdx.x / ARC_KV_WARP);
  const int n_warps = (int)(blockDim.x / ARC_KV_WARP);

  for (long tok = (long)blockIdx.x; tok < ntok; tok += (long)gridDim.x) {
    const T *krow = k + tok * (long)head_dim;
    uint8_t *crow = codes + tok * (long)nope;
    T *srow = side + tok * (long)side_w;
    for (int blk = warp; blk < n_blocks; blk += n_warps) {
      const int base = blk * block_w;
      unsigned amax_bits = 0U;
      for (int e = lane; e < block_w; e += ARC_KV_WARP) {
        const unsigned b = __float_as_uint(arc_to_f32<T>(krow[base + e])) & 0x7FFFFFFFU;
        amax_bits = amax_bits > b ? amax_bits : b;
      }
#pragma unroll
      for (int off = ARC_KV_WARP / 2; off > 0; off >>= 1) {
        const unsigned o = __shfl_xor_sync(0xFFFFFFFFU, amax_bits, off);
        amax_bits = amax_bits > o ? amax_bits : o;
      }
      const float amax = __uint_as_float(amax_bits);
      const float scale = arc_kv_block_scale(amax);
      for (int e = lane; e < block_w; e += ARC_KV_WARP) {
        const float v = arc_to_f32<T>(krow[base + e]);
        crow[base + e] = arc_f32_to_e4m3_code_truncating(__fdiv_rn(v, scale));
      }
      if (lane == 0) {
        srow[rope_dim + blk] = arc_from_f32<T>(amax);
      }
    }
    for (int i = (int)threadIdx.x; i < rope_dim; i += (int)blockDim.x) {
      srow[i] = krow[nope + i];
    }
  }
}

} // namespace arc

#define ARC_KV_LAUNCH_MUTANT(T)                                                \
  arc::arc_kv_fp8_quantize_mutant_kernel<T><<<grid, block, 0, stream>>>(       \
      reinterpret_cast<const T *>(k), codes, reinterpret_cast<T *>(side),      \
      head_dim, nope, rope_dim, n_blocks, block_w, ntok)

extern "C" void arc_kv_fp8_quantize_mutant(
    const void *k, uint8_t *codes, void *side, int32_t head_dim, int32_t nope,
    int32_t rope_dim, int32_t n_blocks, int32_t block_w, int64_t ntok,
    void *stream_ptr, uint32_t dtype) {
  if (ntok <= 0 || n_blocks <= 0) {
    return;
  }
  int warps = n_blocks < 8 ? n_blocks : 8;
  const int threads = warps * ARC_KV_WARP;
  long blocks = ntok < (long)ARC_KV_MAX_BLOCKS ? ntok : (long)ARC_KV_MAX_BLOCKS;
  dim3 grid((unsigned)blocks);
  dim3 block((unsigned)threads);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (dtype == 0) {
    ARC_KV_LAUNCH_MUTANT(__half);
  } else if (dtype == 1) {
    ARC_KV_LAUNCH_MUTANT(__nv_bfloat16);
  } else if (dtype == 2) {
    ARC_KV_LAUNCH_MUTANT(float);
  }
}
