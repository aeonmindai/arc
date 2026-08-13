#[cfg(feature = "cuda")]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

fn main() {
    set_git_revision();

    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        println!("cargo:rerun-if-changed=build.rs");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        // sinkhorn.cu is EXCLUDED from this fast-math builder and compiled
        // separately below: it must be bit-identical to candle-kernels (which
        // build with plain -O3, no fast math), and --use_fast_math rewrites
        // expf -> __expf and IEEE division -> approximate reciprocals. The
        // kernel source carries an `#error` guard against being re-globbed
        // under fast math. See mistralrs-core/src/cuda/sinkhorn.cu.
        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("src/cuda/*.cu")
            .exclude(&["sinkhorn.cu"])
            .out_dir(&build_dir)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--verbose")
            .arg("--compiler-options")
            .arg("-fPIC");

        // Check if CUDA_COMPUTE_CAP < 80 and disable bf16 kernels if so.
        // bf16 WMMA operations and certain bf16 intrinsics are only available on sm_80+.
        if let Some(compute_cap) = builder.get_compute_cap() {
            if compute_cap < 80 {
                builder = builder.arg("-DNO_BF16_KERNEL");
            }
        }

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();

        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrscuda.lib")
        } else {
            build_dir.join("libmistralrscuda.a")
        };

        builder
            .build_lib(out_file)
            .expect("Build mistral-core failed!");
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrscuda");

        // Dedicated IEEE (no fast math) builder for sinkhorn.cu — bit-identity
        // with candle-kernels requires accurate expf + div.rn.f32; --fmad=false
        // additionally forbids FMA contraction so rounding matches candle's
        // unfused op chain exactly. Own subdirectory so its build cache never
        // mixes with the fast-math builder's.
        let sinkhorn_dir = build_dir.join("sinkhorn_ieee");
        let mut sinkhorn_builder = cudaforge::KernelBuilder::new()
            .source_files(vec!["src/cuda/sinkhorn.cu"])
            .out_dir(&sinkhorn_dir)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--fmad=false")
            .arg("--verbose")
            .arg("--compiler-options")
            .arg("-fPIC");
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            sinkhorn_builder = sinkhorn_builder.arg("--compiler-options");
            sinkhorn_builder = sinkhorn_builder.arg(cuda_nvcc_flags_env);
        }
        let sinkhorn_out = if target.contains("msvc") {
            sinkhorn_dir.join("mistralrssinkhornieee.lib")
        } else {
            sinkhorn_dir.join("libmistralrssinkhornieee.a")
        };
        sinkhorn_builder
            .build_lib(sinkhorn_out)
            .expect("Build sinkhorn IEEE kernel failed!");
        println!("cargo:rustc-link-search={}", sinkhorn_dir.display());
        println!("cargo:rustc-link-lib=mistralrssinkhornieee");

        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // nothing to link to
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }
}

fn set_git_revision() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=MISTRALRS_GIT_REVISION={commit}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(ref_path) = head.strip_prefix("ref:") {
            let ref_path = ref_path.trim();
            if !ref_path.is_empty() {
                println!("cargo:rerun-if-changed=.git/{}", ref_path);
            }
        }
    }
}
