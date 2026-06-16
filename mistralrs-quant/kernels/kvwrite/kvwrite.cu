// Device-indexed in-place KV-cache write for CUDA-graph-capturable decode.
//
// During single-token decode the running KV cache is a fixed, pre-grown
// buffer `all_data` of shape [B, H, capacity, D] (the contiguous,
// non-paged SingleCache). The new token's K (or V) is `src` of shape
// [B, H, 1, D]. We must write `src` into `all_data` at sequence position
// `positions[b]`.
//
// The position is read from a DEVICE pointer (not a host-baked offset) so
// that a captured CUDA graph can be replayed for successive tokens: the
// host updates `positions` (a tiny H2D copy) outside the capture region,
// and the replayed kernel picks up the new slot. This is the contiguous
// analogue of PagedAttention's reshape_and_cache + slot_mapping.
//
// IMPORTANT (CUDA graph): the launch takes an explicit `stream`. The caller
// passes candle's capture stream so this kernel is recorded into the graph.
// (Contrast rotary.cu, which hardcodes stream 0 — that is fine for eager use
// but would NOT be captured into a graph on candle's stream.)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace arc {

template <typename scalar_t>
__global__ void kvcache_write_kernel(
    scalar_t *__restrict__ all_data,    // [B, H, capacity, D]
    const scalar_t *__restrict__ src,   // [B, H, 1, D]
    const int *__restrict__ positions,  // [B] device, seq slot per batch row
    const int B, const int H, const int capacity, const int D) {
  const long total = (long)B * H * D;
  for (long index = (long)blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += (long)gridDim.x * blockDim.x) {
    const int d = index % D;
    const int hd = index / D;          // == b*H + h
    const int h = hd % H;
    const int b = hd / H;
    const int pos = positions[b];
    // dst element (b, h, pos, d) in the [B,H,capacity,D] contiguous buffer.
    const long dst = ((long)(b * H + h) * capacity + pos) * D + d;
    // src is contiguous [B,H,1,D] so its flat index equals `index`.
    all_data[dst] = src[index];
  }
}

} // namespace arc

#define CALL_KVWRITE(T)                                                        \
  arc::kvcache_write_kernel<T><<<grid, block, 0, stream>>>(                     \
      reinterpret_cast<T *>(all_data), reinterpret_cast<const T *>(src),        \
      positions, B, H, capacity, D);

extern "C" void kvcache_write(
    void *all_data,        // [B, H, capacity, D]
    const void *src,       // [B, H, 1, D]
    const int *positions,  // [B] device
    int32_t B, int32_t H, int32_t capacity, int32_t D,
    void *stream_ptr,      // cudaStream_t (candle capture stream); null => default
    uint32_t dtype         // 0 => f16; 1 => bf16; 2 => f32
) {
  const long total = (long)B * H * D;
  const int threads = 256;
  const int blocks = (int)((total + threads - 1) / threads);
  dim3 grid(blocks > 0 ? blocks : 1);
  dim3 block(threads);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (dtype == 0) {
    CALL_KVWRITE(half);
  } else if (dtype == 1) {
    CALL_KVWRITE(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_KVWRITE(float);
  }
}
