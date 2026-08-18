use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/gather_rope.cu");
    println!("cargo:rerun-if-changed=src/cuda/sampling.cu");
    println!("cargo:rerun-if-changed=src/cuda/sampling_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/decode_loop.cu");
    println!("cargo:rerun-if-changed=src/cuda/decode_kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda/gemv_bf16.cu");
    // RUN-163: FlashMLASparse — vendored from sgl-project (MIT). See
    // src/cuda/flashmlasparse/LICENSE-MIT for attribution.
    println!("cargo:rerun-if-changed=src/cuda/flashmlasparse/indexer_score.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashmlasparse/topk_radix.cu");

    // ArcTarget (D16): this crate had no architecture handling at all, while
    // `src/cuda/sampling_kernel.cu` names sm_89 / sm_90 / sm_100 as its
    // targets — so the file claimed three architectures and the build produced
    // one. `ARC_CUDA_ARCHS` makes the list authoritative and the archive is
    // verified against it after the build.
    let requested_archs = arc_target::build::requested_archs()
        .unwrap_or_else(|e| panic!("ArcTarget: invalid ARC_CUDA_ARCHS: {e}"));

    let mut builder = cudaforge::KernelBuilder::new()
        .source_glob("src/cuda/*.cu")
        .source_glob("src/cuda/flashmlasparse/*.cu")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--compiler-options")
        .arg("-fPIC");

    if let Some(archs) = &requested_archs {
        let (primary, extra) =
            arc_target::build::split_primary(archs).unwrap_or_else(|e| panic!("ArcTarget: {e}"));
        builder = builder.compute_cap_arch(&primary);
        for gencode in extra {
            builder = builder.arg(&gencode);
        }
    }

    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target = std::env::var("TARGET").unwrap();
    let out_file = if target.contains("msvc") {
        build_dir.join("arccudagraph.lib")
    } else {
        build_dir.join("libarccudagraph.a")
    };
    builder
        .build_lib(&out_file)
        .expect("Build arc-cuda-graph kernels failed!");
    arc_target::build::verify_and_export(&out_file, requested_archs.as_deref());

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=arccudagraph");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=curand");

    Ok(())
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
fn main() -> Result<()> {
    Ok(())
}
