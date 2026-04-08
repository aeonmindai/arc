//! Public FFI bindings for the GEMV launcher symbols compiled from
//! `src/cuda/gemv_bf16.cu`. Lives in the library so the standalone
//! `gemv_bench` binary can pull them in via `arc_cuda_graph::gemv_ffi::*`.
//! Without this re-export, the linker would discard the static-lib object
//! files when building a binary that only references them via its own
//! `extern "C"` block.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

extern "C" {
    pub fn arc_launch_gemv_bf16(
        weight: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        m: i32,
        k: i32,
        sm_count: i32,
        stream: *mut c_void,
    );

    pub fn arc_launch_gemv_bf16_f32out(
        weight: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        m: i32,
        k: i32,
        stream: *mut c_void,
    );

    pub fn arc_launch_gemv_bf16_silu_mul_down(
        weight: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        m: i32,
        k: i32,
        stream: *mut c_void,
    );

    pub fn arc_launch_gemv_bf16_dual(
        weight_a: *const c_void,
        weight_b: *const c_void,
        input: *const c_void,
        out_a: *mut c_void,
        out_b: *mut c_void,
        m_a: i32,
        m_b: i32,
        k: i32,
        stream: *mut c_void,
    );

    pub fn arc_launch_gemv_bf16_dual_silu_mul(
        weight_gate: *const c_void,
        weight_up: *const c_void,
        input: *const c_void,
        output_mlp_act: *mut c_void,
        m: i32,
        k: i32,
        stream: *mut c_void,
    );

    pub fn arc_launch_gemv_bf16_dual_silu_mul_2x16(
        weight_gate: *const c_void,
        weight_up: *const c_void,
        input: *const c_void,
        output_mlp_act: *mut c_void,
        m: i32,
        k: i32,
        stream: *mut c_void,
    );

    pub fn arc_launch_gemv_bf16_clocked(
        weight: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        clocks_buffer: *mut c_void,
        m: i32,
        k: i32,
        stream: *mut c_void,
    );

    // Bench harness variants — same templated kernel, different rows-per-block
    // and __launch_bounds__ second argument.
    pub fn arc_launch_gemv_orig_8x4(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_8x6(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_8x8(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_4x8(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_4x12(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_4x16(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_2x16(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_2x24(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_2x32(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_1x16(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_1x32(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_pipe(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_16x2(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
    pub fn arc_launch_gemv_orig_16x3(
        weight: *const c_void, input: *const c_void, output: *mut c_void,
        m: i32, k: i32, stream: *mut c_void,
    );
}
