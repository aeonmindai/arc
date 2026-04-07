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
    arc_launch_gemv_bf16_dual_silu_mul,
    arc_launch_gemv_orig_8x4,
    arc_launch_gemv_orig_8x6,
    arc_launch_gemv_orig_8x8,
    arc_launch_gemv_orig_4x8,
    arc_launch_gemv_orig_4x12,
    arc_launch_gemv_orig_4x16,
    arc_launch_gemv_orig_2x16,
    arc_launch_gemv_orig_2x24,
    arc_launch_gemv_orig_2x32,
    arc_launch_gemv_orig_1x16,
    arc_launch_gemv_orig_1x32,
    arc_launch_gemv_orig_pipe,
    arc_launch_gemv_orig_16x2,
    arc_launch_gemv_orig_16x3,
};
use arc_cuda_graph::nongemv_ffi::{
    launch_fused_rmsnorm_residual_bf16_clocked,
    launch_rmsnorm_qk_pair_bf16_clocked,
    turbo_reshape_and_cache_clocked,
    turbo_paged_attention_v1_bf16out_clocked,
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

fn percentile(v: &mut Vec<u64>, p: f64) -> u64 {
    if v.is_empty() { return 0; }
    v.sort_unstable();
    let i = ((v.len() as f64) * p).min(v.len() as f64 - 1.0) as usize;
    v[i]
}

fn print_phases(name: &str, phases: &[(&str, Vec<u64>)]) {
    let total_med: u64 = phases.iter().map(|(_, v)| {
        let mut c = v.clone();
        percentile(&mut c, 0.5)
    }).sum();
    println!("  {}: total median cycles ≈ {}", name, total_med);
    for (label, vec) in phases {
        let mut v = vec.clone();
        let med = percentile(&mut v, 0.5);
        let p10 = percentile(&mut v, 0.1);
        let p90 = percentile(&mut v, 0.9);
        let pct = if total_med > 0 { (med as f64 / total_med as f64) * 100.0 } else { 0.0 };
        println!("    {:>14}  med={:7}  p10={:7}  p90={:7}  ({:5.1}% of total)", label, med, p10, p90, pct);
    }
}

unsafe fn read_clocks(buf_ptr: u64, n: usize) -> Vec<u64> {
    let mut h: Vec<u64> = vec![0; n];
    cudaMemcpy(h.as_mut_ptr() as *mut c_void, buf_ptr as *const c_void, n * 8, KIND_D2H);
    h
}

unsafe fn bench_fused_rmsnorm_residual(hidden_size: i32, batch_size: i32, with_residual: bool) {
    let elem = (hidden_size * batch_size) as usize;
    let input = alloc(elem * 2);
    let residual = if with_residual { alloc(elem * 2) } else { 0 };
    let weight = alloc(hidden_size as usize * 2);
    let output = alloc(elem * 2);
    let residual_out = alloc(elem * 2);
    let n_phases = 4;
    let n_blocks = batch_size as usize;
    let clocks = alloc(n_blocks * n_phases * 8);

    fill_random_bf16(input, elem);
    if with_residual { fill_random_bf16(residual, elem); }
    fill_random_bf16(weight, hidden_size as usize);
    cudaMemset(output as *mut c_void, 0, elem * 2);
    cudaMemset(clocks as *mut c_void, 0, n_blocks * n_phases * 8);

    launch_fused_rmsnorm_residual_bf16_clocked(
        input as *const c_void,
        if with_residual { residual as *const c_void } else { std::ptr::null() },
        weight as *const c_void,
        output as *mut c_void,
        residual_out as *mut c_void,
        clocks as *mut c_void,
        hidden_size, batch_size, 1e-6, std::ptr::null_mut(),
    );
    cudaDeviceSynchronize();

    let stamps = read_clocks(clocks, n_blocks * n_phases);
    let mut load = vec![]; let mut reduce = vec![]; let mut write = vec![];
    for b in 0..n_blocks {
        let p0 = stamps[b*4 + 0];
        let p1 = stamps[b*4 + 1];
        let p2 = stamps[b*4 + 2];
        let p3 = stamps[b*4 + 3];
        if p0 == 0 || p3 == 0 { continue; }
        load.push(p1.wrapping_sub(p0));
        reduce.push(p2.wrapping_sub(p1));
        write.push(p3.wrapping_sub(p2));
    }
    let label = if with_residual { "fused_rmsnorm_residual (with residual)" } else { "fused_rmsnorm (no residual)" };
    print_phases(label, &[
        ("load+sumsq", load),
        ("reduction", reduce),
        ("normalize+write", write),
    ]);

    cudaFree(input);
    if residual != 0 { cudaFree(residual); }
    cudaFree(weight); cudaFree(output); cudaFree(residual_out); cudaFree(clocks);
}

unsafe fn bench_qknorm(head_dim: i32, n_q_heads: i32, n_k_heads: i32) {
    let q_size = (head_dim * n_q_heads) as usize;
    let k_size = (head_dim * n_k_heads) as usize;
    let q_in = alloc(q_size * 2);
    let q_w = alloc(head_dim as usize * 2);
    let q_out = alloc(q_size * 2);
    let k_in = alloc(k_size * 2);
    let k_w = alloc(head_dim as usize * 2);
    let k_out = alloc(k_size * 2);
    let total_heads = (n_q_heads + n_k_heads) as usize;
    let n_phases = 4;
    let clocks = alloc(total_heads * n_phases * 8);

    fill_random_bf16(q_in, q_size);
    fill_random_bf16(k_in, k_size);
    fill_random_bf16(q_w, head_dim as usize);
    fill_random_bf16(k_w, head_dim as usize);
    cudaMemset(clocks as *mut c_void, 0, total_heads * n_phases * 8);

    launch_rmsnorm_qk_pair_bf16_clocked(
        q_in as *const c_void, q_w as *const c_void, q_out as *mut c_void,
        k_in as *const c_void, k_w as *const c_void, k_out as *mut c_void,
        clocks as *mut c_void,
        head_dim, n_q_heads, n_k_heads, 1e-6, std::ptr::null_mut(),
    );
    cudaDeviceSynchronize();

    let stamps = read_clocks(clocks, total_heads * n_phases);
    let mut load = vec![]; let mut reduce = vec![]; let mut write = vec![];
    for h in 0..total_heads {
        let p0 = stamps[h*4 + 0];
        let p1 = stamps[h*4 + 1];
        let p2 = stamps[h*4 + 2];
        let p3 = stamps[h*4 + 3];
        if p0 == 0 || p3 == 0 { continue; }
        load.push(p1.wrapping_sub(p0));
        reduce.push(p2.wrapping_sub(p1));
        write.push(p3.wrapping_sub(p2));
    }
    print_phases(&format!("rmsnorm_qk_pair (n_q={}, n_k={}, head_dim={})", n_q_heads, n_k_heads, head_dim), &[
        ("sumsq pass", load),
        ("reduction", reduce),
        ("normalize+write", write),
    ]);

    cudaFree(q_in); cudaFree(q_w); cudaFree(q_out);
    cudaFree(k_in); cudaFree(k_w); cudaFree(k_out);
    cudaFree(clocks);
}

unsafe fn bench_turbo_reshape_and_cache(num_tokens: i32, num_heads: i32, head_size: i32, block_size: i32, num_blocks_total: i32) {
    let kv_size = (num_tokens * num_heads * head_size) as usize;
    let key = alloc(kv_size * 2);
    let value = alloc(kv_size * 2);
    // K cache packed bytes: head_size/2 (4-bit) per element
    let kc_per_block = (num_heads * head_size / 2 * block_size) as usize;
    let kc = alloc(num_blocks_total as usize * kc_per_block);
    // V cache packed bytes: ceil(head_size/10)*4 per element
    let vc_per_block = (num_heads * ((head_size + 9) / 10) * 4 * block_size) as usize;
    let vc = alloc(num_blocks_total as usize * vc_per_block);
    let kn = alloc(num_blocks_total as usize * num_heads as usize * block_size as usize * 2);
    let vn = alloc(num_blocks_total as usize * num_heads as usize * block_size as usize * 2);
    let slots_n = num_tokens as usize;
    let slots_h = vec![0i64; slots_n];
    let slots = alloc(slots_n * 8);
    cudaMemcpy(slots as *mut c_void, slots_h.as_ptr() as *const c_void, slots_n * 8, KIND_H2D);

    fill_random_bf16(key, kv_size);
    fill_random_bf16(value, kv_size);

    let n_blocks = (num_tokens * num_heads) as usize;
    let n_phases = 5;
    let clocks_k = alloc(n_blocks * n_phases * 8);
    let clocks_v = alloc(n_blocks * n_phases * 8);
    cudaMemset(clocks_k as *mut c_void, 0, n_blocks * n_phases * 8);
    cudaMemset(clocks_v as *mut c_void, 0, n_blocks * n_phases * 8);

    let kbs = num_heads * head_size / 2 * block_size;
    let khs = head_size / 2 * block_size;
    let vs_in_stride = num_heads * head_size;
    let nbs = num_heads * block_size;
    let nhs = block_size;
    turbo_reshape_and_cache_clocked(
        key as *const c_void, value as *const c_void,
        kc as *mut c_void, vc as *mut c_void,
        kn as *mut c_void, vn as *mut c_void,
        slots as *const i64,
        clocks_k as *mut c_void, clocks_v as *mut c_void,
        num_tokens, num_heads, head_size, block_size,
        vs_in_stride, vs_in_stride, kbs, khs, nbs, nhs,
        std::ptr::null_mut(), 1, // BF16
    );
    cudaDeviceSynchronize();

    for (which, clocks_buf) in [("tq_cache_k", clocks_k), ("tq_cache_v", clocks_v)] {
        let stamps = read_clocks(clocks_buf, n_blocks * n_phases);
        let mut load = vec![]; let mut norm = vec![]; let mut rotq = vec![]; let mut pack = vec![];
        for b in 0..n_blocks {
            let p0 = stamps[b*5+0];
            let p1 = stamps[b*5+1];
            let p2 = stamps[b*5+2];
            let p3 = stamps[b*5+3];
            let p4 = stamps[b*5+4];
            if p0 == 0 || p4 == 0 { continue; }
            load.push(p1.wrapping_sub(p0));
            norm.push(p2.wrapping_sub(p1));
            rotq.push(p3.wrapping_sub(p2));
            pack.push(p4.wrapping_sub(p3));
        }
        print_phases(which, &[
            ("input load", load),
            ("norm compute", norm),
            ("rotate+quantize", rotq),
            ("pack+write", pack),
        ]);
    }

    cudaFree(key); cudaFree(value); cudaFree(kc); cudaFree(vc);
    cudaFree(kn); cudaFree(vn); cudaFree(slots);
    cudaFree(clocks_k); cudaFree(clocks_v);
}

unsafe fn bench_tq_attn(num_seqs: i32, num_heads: i32, num_kv_heads: i32, head_size: i32, context_len: i32, block_size: i32) {
    let q_size = (num_seqs * num_heads * head_size) as usize;
    let q = alloc(q_size * 2);
    fill_random_bf16(q, q_size);
    let out = alloc(q_size * 2);

    let n_blocks_per_seq = (context_len + block_size - 1) / block_size;
    let total_blocks = (num_seqs * n_blocks_per_seq) as usize;
    // Allocate dummy K/V cache big enough
    let kc_per_block = (num_kv_heads * head_size / 2 * block_size) as usize;
    let vc_per_block = (num_kv_heads * ((head_size + 9) / 10) * 4 * block_size) as usize;
    let kc = alloc(total_blocks * kc_per_block);
    let vc = alloc(total_blocks * vc_per_block);
    let kn = alloc(total_blocks * num_kv_heads as usize * block_size as usize * 2);
    let vn = alloc(total_blocks * num_kv_heads as usize * block_size as usize * 2);
    cudaMemset(kc as *mut c_void, 0, total_blocks * kc_per_block);
    cudaMemset(vc as *mut c_void, 0, total_blocks * vc_per_block);
    cudaMemset(kn as *mut c_void, 0, total_blocks * num_kv_heads as usize * block_size as usize * 2);
    cudaMemset(vn as *mut c_void, 0, total_blocks * num_kv_heads as usize * block_size as usize * 2);

    // block_tables[seq][block_idx] = block_id
    let mbps = n_blocks_per_seq;
    let mut bt_h = vec![0u32; (num_seqs * mbps) as usize];
    for s in 0..num_seqs {
        for b in 0..mbps {
            bt_h[(s*mbps + b) as usize] = (s*mbps + b) as u32;
        }
    }
    let bt = alloc((num_seqs * mbps) as usize * 4);
    cudaMemcpy(bt as *mut c_void, bt_h.as_ptr() as *const c_void, (num_seqs*mbps) as usize * 4, KIND_H2D);
    let cl_h = vec![context_len as u32; num_seqs as usize];
    let cl = alloc(num_seqs as usize * 4);
    cudaMemcpy(cl as *mut c_void, cl_h.as_ptr() as *const c_void, num_seqs as usize * 4, KIND_H2D);

    let n_clock_blocks = (num_seqs * num_heads) as usize;
    let n_phases = 6;
    let clocks = alloc(n_clock_blocks * n_phases * 8);
    cudaMemset(clocks as *mut c_void, 0, n_clock_blocks * n_phases * 8);

    let kbs = num_kv_heads * head_size / 2 * block_size;
    let khs = head_size / 2 * block_size;
    let vbs = num_kv_heads * ((head_size + 9) / 10) * 4 * block_size;
    let vhs = ((head_size + 9) / 10) * 4 * block_size;
    let nbs = num_kv_heads * block_size;
    let nhs = block_size;
    let qs = num_heads * head_size;
    turbo_paged_attention_v1_bf16out_clocked(
        out as *mut c_void, q as *const c_void,
        kc as *const c_void, vc as *const c_void,
        kn as *const c_void, vn as *const c_void,
        num_kv_heads, 1.0/((head_size as f32).sqrt()), 1.0,
        bt as *const u32, cl as *const u32,
        clocks as *mut c_void,
        block_size, context_len, num_seqs, num_heads, head_size,
        mbps, qs, kbs, khs, nbs, nhs,
        std::ptr::null_mut(), 1,
    );
    cudaDeviceSynchronize();

    let stamps = read_clocks(clocks, n_clock_blocks * n_phases);
    let mut q_load = vec![]; let mut qk_dot = vec![]; let mut softmax = vec![]; let mut v_acc = vec![]; let mut out_rot = vec![];
    for b in 0..n_clock_blocks {
        let p0 = stamps[b*6+0];
        let p1 = stamps[b*6+1];
        let p2 = stamps[b*6+2];
        let p3 = stamps[b*6+3];
        let p4 = stamps[b*6+4];
        let p5 = stamps[b*6+5];
        if p0 == 0 || p5 == 0 { continue; }
        q_load.push(p1.wrapping_sub(p0));
        qk_dot.push(p2.wrapping_sub(p1));
        softmax.push(p3.wrapping_sub(p2));
        v_acc.push(p4.wrapping_sub(p3));
        out_rot.push(p5.wrapping_sub(p4));
    }
    print_phases(&format!("tq_attn (ctx={})", context_len), &[
        ("Q load+rotate", q_load),
        ("QK dot products", qk_dot),
        ("softmax", softmax),
        ("V accumulate", v_acc),
        ("inv rotate+write", out_rot),
    ]);

    cudaFree(q); cudaFree(out); cudaFree(kc); cudaFree(vc);
    cudaFree(kn); cudaFree(vn); cudaFree(bt); cudaFree(cl); cudaFree(clocks);
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

        println!("=== variant sweep — find best kernel per shape ===");
        println!("{:>8}  {:>10}  {:>9}  {:>8}  {:>9}", "shape", "variant", "median μs", "GB/s", "% peak");
        for &s in &shapes {
            let bytes = (s.m as usize * s.k as usize * 2) + (s.k as usize * 2);
            let weight = alloc(s.m as usize * s.k as usize * 2);
            let input = alloc(s.k as usize * 2);
            let output = alloc(s.m as usize * 2);
            fill_random_bf16(weight, (s.m * s.k) as usize);
            fill_random_bf16(input, s.k as usize);
            cudaMemset(output as *mut c_void, 0, s.m as usize * 2);

            let variants: &[(&str, unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, i32, i32, *mut c_void))] = &[
                ("orig 16x2", arc_launch_gemv_orig_16x2),
                ("orig 16x3", arc_launch_gemv_orig_16x3),
                ("orig 8x4", arc_launch_gemv_orig_8x4),
                ("orig 8x6", arc_launch_gemv_orig_8x6),
                ("orig 8x8", arc_launch_gemv_orig_8x8),
                ("orig 4x8", arc_launch_gemv_orig_4x8),
                ("orig 4x12", arc_launch_gemv_orig_4x12),
                ("orig 4x16", arc_launch_gemv_orig_4x16),
                ("orig 2x16", arc_launch_gemv_orig_2x16),
                ("orig 2x24", arc_launch_gemv_orig_2x24),
                ("orig 2x32", arc_launch_gemv_orig_2x32),
                ("orig 1x16", arc_launch_gemv_orig_1x16),
                ("orig 1x32", arc_launch_gemv_orig_1x32),
                ("orig pipe", arc_launch_gemv_orig_pipe),
            ];

            let mut best_med = f32::MAX;
            let mut best_name = "";
            for &(name, launch) in variants {
                let (_, med, _) = time_kernel(
                    || launch(weight as *const c_void, input as *const c_void, output as *mut c_void, s.m, s.k, std::ptr::null_mut()),
                    10,
                    100,
                );
                let achieved = (bytes as f32 / 1e9) / (med / 1e6);
                let eff = achieved / 8000.0 * 100.0;
                println!("{:>8}  {:>10}  {:9.2}  {:8.0}  {:8.1}%", s.name, name, med, achieved, eff);
                if med < best_med {
                    best_med = med;
                    best_name = name;
                }
            }
            // Also test the wide dispatcher result for comparison
            let (_, med_wide, _) = time_kernel(
                || arc_launch_gemv_bf16(weight as *const c_void, input as *const c_void, output as *mut c_void, s.m, s.k, sm_count, std::ptr::null_mut()),
                10,
                100,
            );
            let achieved_wide = (bytes as f32 / 1e9) / (med_wide / 1e6);
            println!("{:>8}  {:>10}  {:9.2}  {:8.0}  {:8.1}%", s.name, "dispatch", med_wide, achieved_wide, achieved_wide / 8000.0 * 100.0);
            println!("{:>8}  {:>10}  best={} ({:.2}μs)", s.name, "→", best_name, best_med);
            println!();

            cudaFree(weight);
            cudaFree(input);
            cudaFree(output);
        }
        println!();

        println!("=== clock64 instrumentation (gemv_bf16_clocked_kernel, the original 1-warp-per-row variant) ===");
        for &s in &shapes {
            bench_clocked(s);
            println!();
        }

        // ── Non-GEMV kernels: clock64 phase breakdown ──
        println!("=== non-GEMV clock64 phase breakdown ===");
        // Qwen3-32B shapes: hidden=5120, num_heads=64, num_kv_heads=8, head_dim=128, batch=1
        bench_fused_rmsnorm_residual(5120, 1, false);
        println!();
        bench_fused_rmsnorm_residual(5120, 1, true);
        println!();
        bench_qknorm(128, 64, 8);
        println!();
        bench_turbo_reshape_and_cache(1, 8, 128, 16, 256);
        println!();
        bench_tq_attn(1, 64, 8, 128, 256, 16);
        println!();
        bench_tq_attn(1, 64, 8, 128, 1024, 16);
        println!();
        bench_tq_attn(1, 64, 8, 128, 4096, 16);
        println!();

        // Bench the new dual+silu_mul fused kernel (production MLP path)
        {
            println!("=== dual gate+up + silu+mul fused GEMV (production MLP) ===");
            let bytes_pair = (25600 * 5120 * 2 * 2) as usize;
            let w_g = alloc(25600 * 5120 * 2);
            let w_u = alloc(25600 * 5120 * 2);
            let inp = alloc(5120 * 2);
            let out = alloc(25600 * 2);
            fill_random_bf16(w_g, 25600 * 5120);
            fill_random_bf16(w_u, 25600 * 5120);
            fill_random_bf16(inp, 5120);
            cudaMemset(out as *mut c_void, 0, 25600 * 2);
            let (p10, med, p90) = time_kernel(
                || arc_launch_gemv_bf16_dual_silu_mul(
                    w_g as *const c_void, w_u as *const c_void,
                    inp as *const c_void,
                    out as *mut c_void,
                    25600, 5120, std::ptr::null_mut(),
                ),
                10, 100,
            );
            let achieved = (bytes_pair as f32 / 1e9) / (med / 1e6);
            println!(
                "  dual_silu_mul M=25600 K=5120  bytes=2×262.1MB  μs(p10/med/p90)={:6.1}/{:6.1}/{:6.1}  achieved={:5.0}GB/s  eff={:5.1}%",
                p10, med, p90, achieved, achieved/8000.0*100.0,
            );
            cudaFree(w_g); cudaFree(w_u); cudaFree(inp); cudaFree(out);
        }

        // Bench dual gate+up across multiple shapes — comparison path
        println!("=== dual gate+up GEMV (no silu_mul, for comparison) ===");
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
