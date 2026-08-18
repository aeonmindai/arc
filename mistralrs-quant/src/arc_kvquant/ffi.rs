use core::ffi::c_void;

extern "C" {
    /// Fused block-wise E4M3 quantize. See `kernels/arc_kvquant/arc_kvquant.cu`.
    ///
    /// * `k`     - `[ntok, head_dim]` contiguous, activation dtype.
    /// * `codes` - `[ntok, nope]` U8 out.
    /// * `side`  - `[ntok, rope_dim + n_blocks]` activation dtype out
    ///             (`cat(&[k_rope, amax])`).
    /// * `stream` - candle's stream, so the launch is recordable into a CUDA
    ///              graph captured on it.
    /// * `dtype` - 0 => f16, 1 => bf16, 2 => f32.
    pub(crate) fn arc_kv_fp8_quantize(
        k: *const c_void,
        codes: *mut u8,
        side: *mut c_void,
        head_dim: i32,
        nope: i32,
        rope_dim: i32,
        n_blocks: i32,
        block_w: i32,
        ntok: i64,
        stream: *mut c_void,
        dtype: u32,
    );

    /// Fused dequantize back to `[ntok, head_dim]` (`cat(&[k_nope, k_rope])`).
    /// `lut` is the 256-entry F32 `F8E4M3::from_bits(i).to_f32()` table.
    pub(crate) fn arc_kv_fp8_dequantize(
        codes: *const u8,
        side: *const c_void,
        lut: *const f32,
        out: *mut c_void,
        head_dim: i32,
        nope: i32,
        rope_dim: i32,
        n_blocks: i32,
        block_w: i32,
        ntok: i64,
        stream: *mut c_void,
        dtype: u32,
    );

    /// D33 negative control: `arc_kv_fp8_quantize` with round-to-nearest-even
    /// replaced by truncation. Test-only; nothing in the serving path calls it.
    pub(crate) fn arc_kv_fp8_quantize_mutant(
        k: *const c_void,
        codes: *mut u8,
        side: *mut c_void,
        head_dim: i32,
        nope: i32,
        rope_dim: i32,
        n_blocks: i32,
        block_w: i32,
        ntok: i64,
        stream: *mut c_void,
        dtype: u32,
    );
}
