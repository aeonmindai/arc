//! The contract between the Rust TurboQuant compressor and the CUDA kernels.
//!
//! The kernels cannot call [`super::wht::generate_signs`] or
//! [`super::codebook::get_codebook`] at run time, so the sign diagonals and
//! Lloyd–Max codebooks are checked into
//! `mistralrs-paged-attn/src/cuda/turbo_paged_attention.cuh` as `__constant__`
//! arrays. Two copies of the same numbers is a drift hazard: if the CUDA sign
//! table stops matching the Rust one, every compressed head decodes to noise,
//! and it decodes to noise *quickly* — a broken kernel reports a better tok/s,
//! not a worse one.
//!
//! [`turboquant_cuda_tables_match_the_generator`] closes that. It parses the
//! header and re-derives every table from the same functions the compressor
//! uses. It is a CPU test about a *format*, not a claim about kernel behaviour
//! on a GPU.

/// Head dimensions the CUDA kernels are instantiated for.
///
/// Mirrors the `case` arms of `TQ_ATTN_BY_HS` / `TQ_CACHE_BY_HEAD_DIM` in
/// `turbo_paged_attention.cu`, and [`cuda_head_dims_match_the_kernel_dispatch`]
/// pins the two together. Callers gate on this before dispatching so an
/// unsupported width becomes a refusal rather than an untouched output buffer.
///
/// This is narrower than what the *format* supports:
/// [`super::layout::TurboQuantLayout`] handles any width `>= MIN_BLOCK_DIM` by
/// block decomposition. The kernels take only the widths that decompose to a
/// single power-of-two block, which is every entry here.
pub const TURBOQUANT_CUDA_HEAD_DIMS: [usize; 4] = [64, 128, 256, 512];

/// Whether the CUDA kernels have an instantiation for `head_dim`.
pub fn cuda_supports_head_dim(head_dim: usize) -> bool {
    TURBOQUANT_CUDA_HEAD_DIMS.contains(&head_dim)
}

/// The seed the kernels' sign tables were generated with. Matches
/// [`super::TurboQuantConfig`]'s default.
pub const CUDA_TABLE_SEED: u64 = 42;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::codebook::get_codebook;
    use crate::turboquant::wht::generate_signs;

    const HEADER: &str =
        include_str!("../../../mistralrs-paged-attn/src/cuda/turbo_paged_attention.cuh");
    const KERNEL: &str =
        include_str!("../../../mistralrs-paged-attn/src/cuda/turbo_paged_attention.cu");
    /// The FFI wrapper's own copy of the list. `mistralrs-paged-attn` has no
    /// dependency on this crate (and should not grow one just to share a
    /// four-element array), so the two are kept equal by this test instead.
    const FFI_WRAPPER: &str =
        include_str!("../../../mistralrs-paged-attn/src/cuda/backend/paged_attention.rs");

    /// Pull `static __constant__ float NAME[N] = { ... };` out of the header.
    fn table(name: &str) -> Vec<f32> {
        let needle = format!("float {name}[");
        let start = HEADER
            .find(&needle)
            .unwrap_or_else(|| panic!("{name} is not declared in turbo_paged_attention.cuh"));
        let open = HEADER[start..].find('{').expect("no opening brace") + start;
        let close = HEADER[open..].find("};").expect("no closing brace") + open;
        HEADER[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim_end_matches('f')
                    .parse::<f32>()
                    .unwrap_or_else(|e| panic!("{name}: cannot parse {s:?}: {e}"))
            })
            .collect()
    }

    fn assert_same(name: &str, cuda: &[f32], rust: &[f32]) {
        assert_eq!(
            cuda.len(),
            rust.len(),
            "{name}: CUDA table has {} entries, the generator produces {}",
            cuda.len(),
            rust.len()
        );
        for (i, (c, r)) in cuda.iter().zip(rust).enumerate() {
            assert!(
                (c - r).abs() < 1e-9,
                "{name}[{i}]: CUDA has {c}, the generator produces {r}. \
                 The tables in turbo_paged_attention.cuh are generated — re-run \
                 `cargo run -p mistralrs-quant --example emit_turboquant_cuda_tables -- \
                 64 128 256 512` rather than editing them by hand."
            );
        }
    }

    /// Every `__constant__` table in the CUDA header equals what the Rust
    /// compressor would produce at that dimension.
    ///
    /// This is the guard that makes the two implementations one format. Without
    /// it, the four hand-copied sign tables in the tree (`.cu`, `.cuh`,
    /// `turbo_wht.cu`, `.metal`) had nothing asserting they agreed with the
    /// generator or with each other.
    #[test]
    fn turboquant_cuda_tables_match_the_generator() {
        for dim in TURBOQUANT_CUDA_HEAD_DIMS {
            assert_same(
                &format!("TQ_SGN_{dim}"),
                &table(&format!("TQ_SGN_{dim}")),
                &generate_signs(CUDA_TABLE_SEED, dim),
            );
            for bits in [3u32, 4u32] {
                let cb = get_codebook(dim, bits);
                assert_same(
                    &format!("TQ_CB{bits}_{dim}"),
                    &table(&format!("TQ_CB{bits}_{dim}")),
                    cb.centroids,
                );
                assert_same(
                    &format!("TQ_BD{bits}_{dim}"),
                    &table(&format!("TQ_BD{bits}_{dim}")),
                    cb.boundaries,
                );
            }
        }
    }

    /// A sign table that is all `+1` is not a rotation at all — it would make
    /// `D·H·D` collapse to a plain Hadamard transform and quietly change the
    /// coordinate distribution the codebooks assume. Guards against a table
    /// being zeroed or stubbed rather than merely wrong.
    #[test]
    fn cuda_sign_tables_are_actually_mixed() {
        for dim in TURBOQUANT_CUDA_HEAD_DIMS {
            let s = table(&format!("TQ_SGN_{dim}"));
            let neg = s.iter().filter(|&&x| x < 0.0).count();
            assert!(
                neg > dim / 8 && neg < dim - dim / 8,
                "TQ_SGN_{dim} has {neg}/{dim} negative signs — that is not a random diagonal"
            );
            assert!(
                s.iter().all(|&x| x == 1.0 || x == -1.0),
                "TQ_SGN_{dim} contains a value that is not ±1"
            );
        }
    }

    /// The kernel dispatch and [`TURBOQUANT_CUDA_HEAD_DIMS`] must agree.
    ///
    /// If a `case` is added to the CUDA switch without being added here the
    /// Rust gate rejects a width the kernel could serve; if it is removed from
    /// CUDA but left here, a launch falls through the switch and leaves the
    /// output buffer untouched. That second failure mode is silent, which is
    /// why this test exists rather than a comment.
    #[test]
    fn cuda_head_dims_match_the_kernel_dispatch() {
        for dim in TURBOQUANT_CUDA_HEAD_DIMS {
            assert!(
                KERNEL.contains(&format!("TQ_ATTN_BY_BS({dim}, ")),
                "head dim {dim} is in TURBOQUANT_CUDA_HEAD_DIMS but TQ_ATTN_BY_HS \
                 in turbo_paged_attention.cu has no case for it"
            );
            assert!(
                KERNEL.contains(&format!("TQ_CACHE_BY_DTYPE(FN, {dim},")),
                "head dim {dim} is in TURBOQUANT_CUDA_HEAD_DIMS but TQ_CACHE_BY_HEAD_DIM \
                 in turbo_paged_attention.cu has no case for it"
            );
        }
        // And nothing the kernel dispatches is missing from the Rust list.
        for dim in [64usize, 128, 256, 512, 1024] {
            let in_kernel = KERNEL.contains(&format!("TQ_ATTN_BY_BS({dim}, "));
            assert_eq!(
                in_kernel,
                cuda_supports_head_dim(dim),
                "head dim {dim}: kernel dispatch says {in_kernel}, \
                 TURBOQUANT_CUDA_HEAD_DIMS says {}",
                cuda_supports_head_dim(dim)
            );
        }
    }

    /// The Rust FFI wrapper refuses exactly the widths the kernel cannot serve.
    ///
    /// The wrapper is the last gate before the launch, so if its list is wider
    /// than the kernel's the caller reads an uninitialized output buffer.
    #[test]
    fn ffi_wrapper_head_dims_match() {
        let start = FFI_WRAPPER
            .find("pub const TURBO_HEAD_DIMS")
            .expect("TURBO_HEAD_DIMS is not declared in the paged-attn FFI wrapper");
        let open = FFI_WRAPPER[start..].find('[').unwrap() + start;
        let open = FFI_WRAPPER[open + 1..].find('[').unwrap() + open + 1; // skip the [usize; N] type
        let close = FFI_WRAPPER[open..].find(']').unwrap() + open;
        let declared: Vec<usize> = FFI_WRAPPER[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("head dim is not an integer"))
            .collect();
        assert_eq!(
            declared,
            TURBOQUANT_CUDA_HEAD_DIMS.to_vec(),
            "mistralrs-paged-attn's TURBO_HEAD_DIMS has drifted from \
             TURBOQUANT_CUDA_HEAD_DIMS"
        );
    }

    /// The packed byte counts the CUDA cache layout assumes must be divisible
    /// by the x=16 swizzle group, or a decode lane's run straddles two groups
    /// and the vectorized load in `tq_load_klane` reads the wrong bytes.
    #[test]
    fn packed_k_bytes_are_swizzle_aligned() {
        for dim in TURBOQUANT_CUDA_HEAD_DIMS {
            let k_bytes = super::super::packed_size(dim, 4);
            assert_eq!(
                k_bytes % 16,
                0,
                "head dim {dim}: {k_bytes} packed K bytes is not a multiple of the x=16 group"
            );
            // Each of the 32 lanes owns a whole, power-of-two-sized run.
            let per_lane = k_bytes / 32;
            assert!(
                per_lane > 0 && per_lane.is_power_of_two() && 16 % per_lane == 0,
                "head dim {dim}: {per_lane} bytes per lane does not divide the 16-byte group"
            );
        }
    }
}
