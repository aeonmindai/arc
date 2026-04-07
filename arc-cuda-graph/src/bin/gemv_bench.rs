//! Standalone GEMV benchmark harness.
//!
//! Allocates real-sized weight tensors with random data, runs the various GEMV
//! kernel variants, times them with cudaEvents, and reports per-shape achieved
//! bandwidth. Also runs the clock64-instrumented variant to extract per-warp
//! cycle counts.
//!
//! Lets us iterate kernel parameters in seconds instead of waiting for a Modal
//! redeploy. No model load required.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

// Import via the library so cargo links the static GEMV lib for us.
use arc_cuda_graph::gemv_ffi::{
    arc_launch_gemv_bf16,
    arc_launch_gemv_bf16_clocked,
    arc_launch_gemv_bf16_dual,
};

extern "C" {
    fn cudaMalloc(ptr: *mut u64, size: usize) -> u32;
    fn cudaFree(ptr: u64) -> u32;
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: u32,
    ) -> u32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> u32;
    fn cudaDeviceSynchronize() -> u32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> u32;
    fn cudaEventCreate(event: *mut *mut c_void) -> u32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> u32;
    fn cudaEventSynchronize(event: *mut c_void) -> u32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> u32;
    fn cudaEventDestroy(event: *mut c_void) -> u32;
}

const KIND_H2D: u32 = 1;
const KIND_D2H: u32 = 2;

unsafe fn alloc(bytes: usize) -> u64 {
    let mut p: u64 = 0;
    let s = cudaMalloc(&mut p, bytes);
    assert_eq!(s, 0, "cudaMalloc({bytes}) failed: {s}");
    p
}

unsafe fn fill_random_bf16(ptr: u64, n_elem: usize) {
    // Cheap deterministic pseudo-random bf16 fill via small CPU buffer.
    // We use a fixed seed and a tiny LCG so the same data shows up each run.
    let mut buf: Vec<u16> = Vec::with_capacity(n_elem);
    let mut x: u64 = 0x12345678;
    for _ in 0..n_elem {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // bf16 small values: top 16 bits of f32(±0.0..1.0)
        let f = ((x >> 32) as u32 as f32) / (u32::MAX as f32) * 0.1 - 0.05;
        let bits = f.to_bits();
        buf.push((bits >> 16) as u16);
    }
    let s = cudaMemcpy(
        ptr as *mut c_void,
        buf.as_ptr() as *const c_void,
        n_elem * 2,
        KIND_H2D,
    );
    assert_eq!(s, 0);
}

unsafe fn time_kernel<F: Fn()>(launch: F, warmup: usize, iters: usize) -> (f32, f32, f32) {
    // Warmup
    for _ in 0..warmup {
        launch();
    }
    cudaDeviceSynchronize();

    // Time individual launches with cudaEvents
    let mut start: *mut c_void = std::ptr::null_mut();
    let mut end: *mut c_void = std::ptr::null_mut();
    cudaEventCreate(&mut start);
    cudaEventCreate(&mut end);

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        cudaEventRecord(start, std::ptr::null_mut());
        launch();
        cudaEventRecord(end, std::ptr::null_mut());
        cudaEventSynchronize(end);
        let mut ms: f32 = 0.0;
        cudaEventElapsedTime(&mut ms, start, end);
        times.push(ms * 1000.0); // → microseconds
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[iters / 2];
    let p10 = times[iters / 10];
    let p90 = times[iters * 9 / 10];

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    (p10, median, p90)
}

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    m: i32,
    k: i32,
}

unsafe fn bench_shape(shape: Shape, sm_count: i32) {
    let bytes_w = (shape.m * shape.k * 2) as usize;
    let bytes_in = (shape.k * 2) as usize;
    let bytes_out = (shape.m * 2) as usize;

    let weight = alloc(bytes_w);
    let input = alloc(bytes_in);
    let output = alloc(bytes_out);

    fill_random_bf16(weight, (shape.m * shape.k) as usize);
    fill_random_bf16(input, shape.k as usize);
    cudaMemset(output as *mut c_void, 0, bytes_out);

    let theoretical_bytes = bytes_w + bytes_in;
    let bw_peak_gb_s = 8000.0; // B200 HBM theoretical
    let theoretical_us = theoretical_bytes as f32 / bw_peak_gb_s / 1000.0;

    let (p10, med, p90) = time_kernel(
        || {
            arc_launch_gemv_bf16(
                weight as *const c_void,
                input as *const c_void,
                output as *mut c_void,
                shape.m,
                shape.k,
                sm_count,
                std::ptr::null_mut(),
            );
        },
        10,
        100,
    );

    let achieved_gb_s = (theoretical_bytes as f32 / 1e9) / (med / 1e6);
    let eff = achieved_gb_s / bw_peak_gb_s * 100.0;

    println!(
        "{:>8}  M={:6} K={:6}  bytes={:7.1}MB  μs(p10/med/p90)={:6.1}/{:6.1}/{:6.1}  achieved={:5.0}GB/s  eff={:5.1}%  theo={:5.1}μs",
        shape.name,
        shape.m,
        shape.k,
        bytes_w as f32 / 1e6,
        p10,
        med,
        p90,
        achieved_gb_s,
        eff,
        theoretical_us,
    );

    cudaFree(weight);
    cudaFree(input);
    cudaFree(output);
}

unsafe fn bench_clocked(shape: Shape) {
    // 8 rows per block in the orig kernel (GEMV_ROWS=8). 8 warps. 4 phases.
    let n_blocks = ((shape.m + 7) / 8) as usize;
    let n_warps_per_block = 8usize;
    let n_phases = 4usize;
    let total_stamps = n_blocks * n_warps_per_block * n_phases;

    let bytes_w = (shape.m * shape.k * 2) as usize;
    let bytes_in = (shape.k * 2) as usize;
    let bytes_out = (shape.m * 2) as usize;
    let bytes_clocks = total_stamps * 8;

    let weight = alloc(bytes_w);
    let input = alloc(bytes_in);
    let output = alloc(bytes_out);
    let clocks = alloc(bytes_clocks);

    fill_random_bf16(weight, (shape.m * shape.k) as usize);
    fill_random_bf16(input, shape.k as usize);
    cudaMemset(output as *mut c_void, 0, bytes_out);
    cudaMemset(clocks as *mut c_void, 0, bytes_clocks);

    // Single launch — we want clean cycle counts, no warmup pollution
    arc_launch_gemv_bf16_clocked(
        weight as *const c_void,
        input as *const c_void,
        output as *mut c_void,
        clocks as *mut c_void,
        shape.m,
        shape.k,
        std::ptr::null_mut(),
    );
    cudaDeviceSynchronize();

    let mut host: Vec<u64> = vec![0; total_stamps];
    cudaMemcpy(
        host.as_mut_ptr() as *mut c_void,
        clocks as *const c_void,
        bytes_clocks,
        KIND_D2H,
    );

    // Aggregate phases
    // For each (block, warp), compute:
    //   first_load_cycles  = phase1 - phase0
    //   inner_loop_cycles  = phase2 - phase1
    //   reduction_cycles   = phase3 - phase2
    //   total_cycles       = phase3 - phase0
    let mut first_loads = Vec::with_capacity(n_blocks * n_warps_per_block);
    let mut inner_loops = Vec::with_capacity(n_blocks * n_warps_per_block);
    let mut reductions = Vec::with_capacity(n_blocks * n_warps_per_block);
    let mut totals = Vec::with_capacity(n_blocks * n_warps_per_block);

    for b in 0..n_blocks {
        for w in 0..n_warps_per_block {
            let row = b * 8 + w;
            if row >= shape.m as usize {
                break;
            }
            let base = (b * n_warps_per_block + w) * n_phases;
            let p0 = host[base + 0];
            let p1 = host[base + 1];
            let p2 = host[base + 2];
            let p3 = host[base + 3];
            if p0 == 0 || p3 == 0 {
                continue;
            }
            first_loads.push(p1.wrapping_sub(p0));
            inner_loops.push(p2.wrapping_sub(p1));
            reductions.push(p3.wrapping_sub(p2));
            totals.push(p3.wrapping_sub(p0));
        }
    }

    fn pct(v: &mut Vec<u64>, p: f64) -> u64 {
        v.sort_unstable();
        let i = ((v.len() as f64) * p).min(v.len() as f64 - 1.0) as usize;
        v[i]
    }

    let k8 = (shape.k / 8) as u64;
    let iters_per_warp = (k8 + 31) / 32; // GEMV_WARP=32
    println!(
        "  CLOCKED  M={:6} K={:6}  warps={}  K8={}  iters/warp={}",
        shape.m, shape.k, totals.len(), k8, iters_per_warp,
    );
    println!(
        "    first_load_cycles  med={:6}  p10={:6}  p90={:6}   (single __ldg pair latency)",
        pct(&mut first_loads.clone(), 0.5),
        pct(&mut first_loads.clone(), 0.1),
        pct(&mut first_loads.clone(), 0.9),
    );
    println!(
        "    inner_loop_cycles  med={:6}  p10={:6}  p90={:6}   ({} iters)",
        pct(&mut inner_loops.clone(), 0.5),
        pct(&mut inner_loops.clone(), 0.1),
        pct(&mut inner_loops.clone(), 0.9),
        iters_per_warp.saturating_sub(1),
    );
    let il_med = pct(&mut inner_loops.clone(), 0.5);
    let cycles_per_iter = il_med / iters_per_warp.saturating_sub(1).max(1);
    println!(
        "    cycles/iter (loop) ≈ {:5}   bytes/iter=32 (1 uint4 weight + 1 uint4 input)",
        cycles_per_iter,
    );
    println!(
        "    reduction_cycles   med={:6}  p10={:6}  p90={:6}",
        pct(&mut reductions.clone(), 0.5),
        pct(&mut reductions.clone(), 0.1),
        pct(&mut reductions.clone(), 0.9),
    );
    println!(
        "    total_cycles       med={:6}  p10={:6}  p90={:6}",
        pct(&mut totals.clone(), 0.5),
        pct(&mut totals.clone(), 0.1),
        pct(&mut totals.clone(), 0.9),
    );

    cudaFree(weight);
    cudaFree(input);
    cudaFree(output);
    cudaFree(clocks);
}

fn main() {
    unsafe {
        let mut sm_count: i32 = 0;
        cudaDeviceGetAttribute(&mut sm_count, 16, 0);
        let mut clock_khz: i32 = 0;
        cudaDeviceGetAttribute(&mut clock_khz, 13, 0); // cudaDevAttrClockRate

        println!("=== arc GEMV bench ===");
        println!("SM count: {}  clock: {} MHz", sm_count, clock_khz / 1000);
        println!();

        // Qwen3-32B shapes (hidden=5120, intermediate=25600, n_heads=64, n_kv=8, head_dim=128).
        // GEMV M = output rows, K = input dim per row.
        let shapes = [
            Shape { name: "qkv",   m: 64*128 + 2*8*128, k: 5120 }, // 10240×5120
            Shape { name: "oproj", m: 5120,             k: 5120 },
            Shape { name: "gate",  m: 25600,            k: 5120 },
            Shape { name: "up",    m: 25600,            k: 5120 },
            Shape { name: "down",  m: 5120,             k: 25600 },
            Shape { name: "lmhead",m: 152064,           k: 5120 },
        ];

        println!("=== throughput sweep (current arc_launch_gemv_bf16 dispatch) ===");
        for &s in &shapes {
            bench_shape(s, sm_count);
        }
        println!();

        println!("=== clock64 instrumentation (gemv_bf16_clocked_kernel, the original 1-warp-per-row variant) ===");
        for &s in &shapes {
            bench_clocked(s);
            println!();
        }

        // Sanity: also dual gate+up combo, the path used in real decode_forward
        println!("=== dual gate+up GEMV (production path) ===");
        let bytes = (25600 * 5120 * 2) as usize;
        let w_a = alloc(bytes);
        let w_b = alloc(bytes);
        let inp = alloc((5120 * 2) as usize);
        let out_a = alloc((25600 * 2) as usize);
        let out_b = alloc((25600 * 2) as usize);
        fill_random_bf16(w_a, 25600 * 5120);
        fill_random_bf16(w_b, 25600 * 5120);
        fill_random_bf16(inp, 5120);
        let (p10, med, p90) = time_kernel(
            || {
                arc_launch_gemv_bf16_dual(
                    w_a as *const c_void,
                    w_b as *const c_void,
                    inp as *const c_void,
                    out_a as *mut c_void,
                    out_b as *mut c_void,
                    25600,
                    25600,
                    5120,
                    std::ptr::null_mut(),
                );
            },
            10,
            100,
        );
        let achieved = (2.0 * bytes as f32 / 1e9) / (med / 1e6);
        println!(
            "  dual_gate_up M_a=25600 M_b=25600 K=5120  bytes=2×{:.1}MB  μs(p10/med/p90)={:6.1}/{:6.1}/{:6.1}  achieved={:5.0}GB/s",
            bytes as f32 / 1e6, p10, med, p90, achieved,
        );
        cudaFree(w_a);
        cudaFree(w_b);
        cudaFree(inp);
        cudaFree(out_a);
        cudaFree(out_b);
    }
}
