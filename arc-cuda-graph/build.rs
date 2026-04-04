use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/gather_rope.cu");
    println!("cargo:rerun-if-changed=src/cuda/sampling.cu");
    println!("cargo:rerun-if-changed=src/cuda/decode_loop.cu");

    let mut builder = cudaforge::KernelBuilder::new()
        .source_glob("src/cuda/*.cu")
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
        .build_lib(out_file)
        .expect("Build arc-cuda-graph kernels failed!");

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
