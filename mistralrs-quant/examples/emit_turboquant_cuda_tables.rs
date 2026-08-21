//! Emit the CUDA `__constant__` tables for TurboQuant at a given head dimension.
//!
//! The CUDA kernels cannot call `generate_signs`/`cached_generated_codebook` at
//! runtime, so the tables are checked in as `__constant__` arrays. This example
//! is the *only* sanctioned way to produce them: it uses the same crate code the
//! Rust compressor uses, so the emitted tables cannot drift from the format by
//! construction.
//!
//! `turboquant_cuda_tables_match_crate` in `mistralrs-quant/src/turboquant/cuda_tables.rs`
//! re-derives these values at test time and fails if the checked-in CUDA source
//! disagrees, so a hand-edit of the `.cu` is caught by CI.
//!
//! Usage: `cargo run -p mistralrs-quant --example emit_turboquant_cuda_tables -- 512`

// `get_codebook` is the same entry point `TurboQuantLayout::new` uses, so the
// emitted table is the one the Rust compressor actually quantizes against —
// static for 64/128/256, numerically generated elsewhere.
use mistralrs_quant::turboquant::codebook::get_codebook;
use mistralrs_quant::turboquant::wht::generate_signs;

const SEED: u64 = 42;

fn emit_signs(dim: usize) {
    let signs = generate_signs(SEED, dim);
    println!("__constant__ float TQ_SGN_{dim}[{dim}] = {{");
    for row in signs.chunks(16) {
        let cells: Vec<String> = row
            .iter()
            .map(|s| format!("{:>2}", if *s > 0.0 { "1" } else { "-1" }))
            .collect();
        println!("    {},", cells.join(","));
    }
    println!("}};");
}

fn emit_codebook(dim: usize, bits: u32) {
    let cb = get_codebook(dim, bits);
    let n = cb.centroids.len();
    println!("__constant__ float TQ_CB{bits}_{dim}[{n}] = {{");
    for row in cb.centroids.chunks(4) {
        let cells: Vec<String> = row.iter().map(|c| format!("{c:.12}f")).collect();
        println!("    {},", cells.join(","));
    }
    println!("}};");

    let nb = cb.boundaries.len();
    println!("__constant__ float TQ_BD{bits}_{dim}[{nb}] = {{");
    for row in cb.boundaries.chunks(4) {
        let cells: Vec<String> = row.iter().map(|c| format!("{c:.12}f")).collect();
        println!("    {},", cells.join(","));
    }
    println!("}};");
}

fn main() {
    let dims: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("head dim must be an integer"))
        .collect();
    let dims = if dims.is_empty() { vec![512] } else { dims };

    for dim in dims {
        println!("// ==== head_dim = {dim} (seed {SEED}) ====");
        emit_signs(dim);
        for bits in [3u32, 4u32] {
            emit_codebook(dim, bits);
        }
        println!();
    }
}
