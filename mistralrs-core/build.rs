#[cfg(feature = "cuda")]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

/// CUDA sources whose output must be BIT-IDENTICAL to candle-kernels, and
/// which therefore must be compiled without `--use_fast_math`. Each carries a
/// bit-identity contract in its file header.
#[cfg(feature = "cuda")]
const IEEE_SOURCES: &[&str] = &["sinkhorn.cu", "swiglu_clamp.cu", "qnorm.cu"];

/// nvcc args for the normal (fast-math) `src/cuda/*.cu` builder.
#[cfg(feature = "cuda")]
const FAST_MATH_ARGS: &[&str] = &[
    "-std=c++17",
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--verbose",
];

/// nvcc args for the IEEE (no fast-math) builder.
#[cfg(feature = "cuda")]
const IEEE_ARGS: &[&str] = &[
    "-std=c++17",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--fmad=false",
    "--verbose",
];

/// The guard that the `#error` idiom cannot be: nvcc 12.4 defines NO
/// preprocessor macro for `--use_fast_math` in either compilation pass (probed
/// directly on sm_90 with this toolchain), so a `.cu`-side `#if
/// defined(__USE_FAST_MATH__)` — as sinkhorn.cu carries today — never fires.
///
/// This asserts the invariant at the only place it can regress: the flag sets
/// in this file. Adding `--use_fast_math` to `IEEE_ARGS`, or dropping a source
/// from the fast-math builder's exclusion list, becomes a hard build failure
/// instead of silent numeric drift in generated tokens.
///
/// Proved red: temporarily appending `"--use_fast_math"` to `IEEE_ARGS` makes
/// `cargo build -p mistralrs-core --features cuda` panic here before nvcc is
/// ever invoked.
#[cfg(feature = "cuda")]
fn assert_ieee_kernel_flags(sources: &[&str], fast_math_args: &[&str], ieee_args: &[&str]) {
    assert!(
        fast_math_args.contains(&"--use_fast_math"),
        "the general src/cuda/*.cu builder lost --use_fast_math; if that is \
         intentional, IEEE_SOURCES no longer needs a separate builder"
    );
    assert!(
        !ieee_args.contains(&"--use_fast_math"),
        "BIT-IDENTITY VIOLATION: the IEEE builder was given --use_fast_math. \
         {sources:?} must match candle-kernels bit for bit, and fast math \
         rewrites expf -> ex2.approx.ftz.f32 and div.rn.f32 -> \
         div.approx.ftz.f32 (measured, nvcc 12.4/sm_90). This changes \
         generated tokens."
    );
    for src in sources {
        let path = std::path::Path::new("src/cuda").join(src);
        assert!(
            path.exists(),
            "IEEE_SOURCES lists {src}, which does not exist at {}. It would be \
             silently dropped from the build while still being excluded from \
             the fast-math glob.",
            path.display()
        );
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    set_git_revision();

    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        println!("cargo:rerun-if-changed=build.rs");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        // These sources are EXCLUDED from this fast-math builder and compiled
        // separately below: they must be bit-identical to candle-kernels
        // (which build with plain -O3, no fast math), and --use_fast_math
        // rewrites expf -> __expf and IEEE division -> approximate
        // reciprocals. See mistralrs-core/src/cuda/{sinkhorn,swiglu_clamp}.cu.
        //
        // NOTE: the `#if defined(__USE_FAST_MATH__)` `#error` in sinkhorn.cu is
        // VACUOUS. nvcc 12.4 defines no preprocessor macro for
        // --use_fast_math in either the host or the device pass (probed on
        // sm_90: __USE_FAST_MATH__, __CUDA_FAST_MATH__, __FAST_MATH__,
        // __CUDACC_FAST_MATH__, __CUDA_PREC_DIV__, __CUDA_FTZ__ are all
        // undefined with and without the flag), so that guard can never fire.
        // `assert_ieee_kernel_flags` below is the guard that does.
        assert_ieee_kernel_flags(IEEE_SOURCES, FAST_MATH_ARGS, IEEE_ARGS);

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("src/cuda/*.cu")
            .exclude(IEEE_SOURCES)
            .out_dir(&build_dir);
        for a in FAST_MATH_ARGS {
            builder = builder.arg(a);
        }
        builder = builder.arg("--compiler-options").arg("-fPIC");

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

        // Dedicated IEEE (no fast math) builder for the bit-identity kernels —
        // bit-identity with candle-kernels requires accurate expf +
        // div.rn.f32; --fmad=false additionally forbids FMA contraction so
        // rounding matches candle's unfused op chain exactly. (Verified on
        // nvcc 12.4/sm_90 that --fmad=false does NOT perturb libdevice's
        // accurate expf expansion, which uses explicit fma.rn.f32 rather than
        // source-level contraction.) Own subdirectory so its build cache never
        // mixes with the fast-math builder's.
        let sinkhorn_dir = build_dir.join("sinkhorn_ieee");
        let mut sinkhorn_builder = cudaforge::KernelBuilder::new()
            .source_files(
                IEEE_SOURCES
                    .iter()
                    .map(|f| format!("src/cuda/{f}"))
                    .collect::<Vec<_>>(),
            )
            .out_dir(&sinkhorn_dir);
        for a in IEEE_ARGS {
            sinkhorn_builder = sinkhorn_builder.arg(a);
        }
        sinkhorn_builder = sinkhorn_builder.arg("--compiler-options").arg("-fPIC");
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
