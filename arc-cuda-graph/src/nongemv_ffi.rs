//! FFI bindings for clocked variants of non-GEMV kernels (norms, KV cache,
//! turbo paged attention). Used by `gemv_bench` to measure intra-kernel
//! cycle distributions and find where time goes inside each kernel.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

// Force the linker to pull in mistralrs-paged-attn's static CUDA lib so the
// turbo_*_clocked symbols defined in turbo_paged_attention.cu are findable
// from the gemv_bench binary. Reference a public function from the dep so
// cargo includes its rlib + static lib in the link line. Never called.
#[allow(dead_code)]
#[doc(hidden)]
pub fn __force_link_paged_attn() -> *const () {
    mistralrs_paged_attn::turbo_reshape_and_cache as *const ()
}

extern "C" {
    // arc-cuda-graph kernels
    pub fn launch_fused_rmsnorm_residual_bf16_clocked(
        input: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        residual_out: *mut c_void,
        clocks: *mut c_void,
        hidden_size: i32,
        batch_size: i32,
        eps: f32,
        stream: *mut c_void,
    );

    pub fn launch_rmsnorm_qk_pair_bf16_clocked(
        q_in: *const c_void,
        q_w: *const c_void,
        q_out: *mut c_void,
        k_in: *const c_void,
        k_w: *const c_void,
        k_out: *mut c_void,
        clocks: *mut c_void,
        head_dim: i32,
        n_q_heads: i32,
        n_k_heads: i32,
        eps: f32,
        stream: *mut c_void,
    );

    // mistralrs-paged-attn kernels
    pub fn turbo_reshape_and_cache_clocked(
        key: *const c_void,
        value: *const c_void,
        kc: *mut c_void,
        vc: *mut c_void,
        kn: *mut c_void,
        vn: *mut c_void,
        slots: *const i64,
        clocks_k: *mut c_void,
        clocks_v: *mut c_void,
        nt: i32,
        nh: i32,
        hs: i32,
        bs: i32,
        ks: i32,
        vs: i32,
        kbs: i32,
        khs: i32,
        nbs: i32,
        nhs: i32,
        stream: *mut c_void,
        dtype: u32,
    );

    pub fn turbo_paged_attention_v1_bf16out_clocked(
        out_bf16: *mut c_void,
        query: *const c_void,
        kc: *const c_void,
        vc: *const c_void,
        kn: *const c_void,
        vn: *const c_void,
        nkvh: i32,
        scale: f32,
        softcapping: f32,
        bt: *const u32,
        cl: *const u32,
        clocks: *mut c_void,
        bs: i32,
        mcl: i32,
        ns: i32,
        nh: i32,
        hs: i32,
        mbps: i32,
        qs: i32,
        kbs: i32,
        khs: i32,
        nbs: i32,
        nhs: i32,
        stream: *mut c_void,
        qdtype: u32,
    );
}
