#[cfg(feature = "cuda")]
#[allow(unused)]
fn cuda_version_from_build_system() -> (usize, usize) {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .expect("Failed to execute `nvcc`");

    if !output.status.success() {
        panic!(
            "`nvcc --version` failed.\nstdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().nth(3).unwrap();
    let release_section = version_line.split(", ").nth(1).unwrap();
    let version_number = release_section.split(' ').nth(1).unwrap();

    match version_number {
        "13.1" => (13, 1),
        "13.0" => (13, 0),
        "12.9" => (12, 9),
        "12.8" => (12, 8),
        "12.6" => (12, 6),
        "12.5" => (12, 5),
        "12.4" => (12, 4),
        "12.3" => (12, 3),
        "12.2" => (12, 2),
        "12.1" => (12, 1),
        "12.0" => (12, 0),
        "11.8" => (11, 8),
        "11.7" => (11, 7),
        "11.6" => (11, 6),
        "11.5" => (11, 5),
        "11.4" => (11, 4),
        v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
    }
}

/// The root of the CUDA kernel tree, relative to the package manifest dir.
/// `build.rs` hands `cudaforge` the glob `{KERNEL_GLOB_ROOT}/*/*.cu`, and
/// watches this directory so Cargo re-runs the script when that glob's result
/// would change. Kept in sync by `tests/cuda_kernel_build_guard.rs`.
#[cfg(feature = "cuda")]
const KERNEL_GLOB_ROOT: &str = "kernels";

/// Checked-in expected size of the discovered kernel set. See the file itself
/// for why it exists.
#[cfg(feature = "cuda")]
const EXPECTED_KERNEL_COUNT_FILE: &str = "kernels/EXPECTED_KERNEL_COUNT";

/// Enumerate exactly what `kernels/*/*.cu` matches: `.cu` files one directory
/// level below the glob root. Deliberately mirrors the glob rather than
/// recursing, so a kernel buried deeper (which the glob would silently skip)
/// shows up as a count mismatch instead of vanishing.
#[cfg(feature = "cuda")]
fn discover_kernel_sources() -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let Ok(top) = std::fs::read_dir(KERNEL_GLOB_ROOT) else {
        return found;
    };
    for entry in top.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Ok(inner) = std::fs::read_dir(entry.path()) else {
            continue;
        };
        for file in inner.flatten() {
            let path = file.path();
            if path.extension().is_some_and(|e| e == "cu") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Read the expected count: first line that is neither blank nor a `#` comment.
#[cfg(feature = "cuda")]
fn read_expected_kernel_count() -> usize {
    let raw = std::fs::read_to_string(EXPECTED_KERNEL_COUNT_FILE).unwrap_or_else(|e| {
        panic!("cannot read `mistralrs-quant/{EXPECTED_KERNEL_COUNT_FILE}`: {e}")
    });
    raw.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .and_then(|l| l.parse::<usize>().ok())
        .unwrap_or_else(|| {
            panic!(
                "`mistralrs-quant/{EXPECTED_KERNEL_COUNT_FILE}` has no bare integer line; \
                 it must contain the expected `{KERNEL_GLOB_ROOT}/*/*.cu` count"
            )
        })
}

/// Hard-fail the build when the discovered kernel set is not the expected size.
///
/// Cargo cannot tell you a kernel is missing: a glob that matches fewer files
/// just... matches fewer files. Without this gate the only signal that a
/// kernel was dropped is the `Compiling N of M kernels` line in the log, which
/// nobody reads on a green build. Asserting the count turns a silent no-op
/// into an error at the moment the mistake is made.
#[cfg(feature = "cuda")]
fn assert_kernel_set_intact() {
    let discovered = discover_kernel_sources();
    let expected = read_expected_kernel_count();
    if discovered.len() != expected {
        let listing = discovered
            .iter()
            .map(|p| format!("  {}", p.display()))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "\n\
             mistralrs-quant: CUDA kernel set changed.\n\
             `{KERNEL_GLOB_ROOT}/*/*.cu` matched {} sources, but \
             `{EXPECTED_KERNEL_COUNT_FILE}` expects {expected}.\n\n\
             Discovered:\n{listing}\n\n\
             If this change is intended, set the count in \
             `mistralrs-quant/{EXPECTED_KERNEL_COUNT_FILE}` to {}.\n\
             If it is NOT intended, a kernel is being silently dropped from the \
             build -- check that it sits exactly one directory below \
             `{KERNEL_GLOB_ROOT}/` (the glob does not recurse further).\n",
            discovered.len(),
            discovered.len(),
        );
    }
    // Positive engagement signal: prove the gate ran, on green builds too.
    println!(
        "cargo:warning=mistralrs-quant: kernel-set guard OK -- {expected} sources under `{KERNEL_GLOB_ROOT}/*/*.cu`"
    );
}

fn main() -> Result<(), String> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_scalar_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_vector_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_qtip_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_wmma_kernels)");
    // SageAttention SM89/SM90/SM100 INT8 + FP8 kernels (vendored from
    // SageAttention upstream). When the compiled GPU compute cap is high
    // enough, the kernels are linked in; otherwise the Rust side falls back
    // to the software path.
    println!("cargo::rustc-check-cfg=cfg(has_sage_sm89_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_sage_sm90_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_sage_sm100_kernels)");

    #[cfg(feature = "cuda")]
    {
        use std::{path::PathBuf, vec};
        const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

        println!("cargo:rerun-if-changed=build.rs");

        // ── Watch the kernel glob ROOT, not just its current contents. ──────
        // The kernels below are discovered by `source_glob("kernels/*/*.cu")`,
        // which is evaluated only when this script RUNS. Cargo re-runs a build
        // script solely for paths it named via `rerun-if-changed`, and
        // cudaforge names one path per *already-resolved* source — so a kernel
        // that does not exist yet can never be named.
        //
        // Consequence before this line existed: adding `kernels/<d>/<new>.cu`
        // did not re-run this script, the glob was never re-evaluated, the
        // kernel was never compiled, and the build was STILL GREEN — `extern
        // "C"` declarations fail only at link time and an rlib build does not
        // link. Naming the directory makes Cargo re-run on any add / edit /
        // delete anywhere beneath it (verified: a file-list trigger does not).
        //
        // STILL BROKEN, DELIBERATELY NOT FIXED HERE — edited `.cuh` headers.
        // This line makes Cargo re-run the script for them, but cudaforge's
        // incremental cache keys only on the `.cu`'s own content hash, gpu arch
        // and args hash (`CacheEntry`, cudaforge 0.1.5 `src/hash.rs`); it never
        // hashes headers — `collect_headers` is exported but never called. So
        // touching e.g. `kernels/qtip/qtip_codebook.cuh` still yields "All
        // kernels up-to-date, skipping compilation" and the OLD codebook stays
        // linked in. Until that is fixed, `touch` the dependent `.cu` (or clear
        // OUT_DIR) after editing a header, and check the `Compiling N of M`
        // line actually moved. Tracked separately; the fix is to fold a hash of
        // `kernels/**/*.cuh` into the nvcc args so `args_hash` changes.
        //
        // Guarded by `tests/cuda_kernel_build_guard.rs` — do not remove.
        println!("cargo::rerun-if-changed={KERNEL_GLOB_ROOT}");
        assert_kernel_set_intact();

        // SageAttention kernel sources — recompile if any change.
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/sm89_qk_int8_sv_f8_attn.cu");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/sage_quant.cu");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/sage_dispatch.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/sage_utils.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/attn_utils.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/qk_int_sv_f8_cuda_sm89.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/fused_kernels.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/cp_async.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/math.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/mma.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/wgmma.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/permuted_smem.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/numeric_conversion.cuh");
        println!("cargo:rerun-if-changed=src/sage_cuda/kernels/reduction_utils.cuh");

        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("kernels/*/*.cu")
            .out_dir(build_dir.clone())
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

        let compute_cap = builder.get_compute_cap().unwrap_or(80);
        // ======== Handle optional kernel compilation via rustc-cfg flags
        let cc_over_80 = compute_cap >= 80;
        let cc_over_89 = compute_cap >= 89;
        let cc_over_90 = compute_cap >= 90;
        let cc_over_100 = compute_cap >= 100;

        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");
            println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
            println!("cargo:rustc-cfg=has_vector_fp8_kernels");
            // QTIP CUDA dequantize + activation rotation (uses __nv_bfloat16 ops).
            println!("cargo:rustc-cfg=has_qtip_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
            println!("cargo:rustc-cfg=has_mxfp4_wmma_kernels");
        }
        // SageAttention compile-time gating. The SM89 path needs FP8 MMA
        // instructions, which require SM >= 8.9 (Ada / L4 / RTX 4090) at
        // minimum. Lower SMs would still link, but the kernel would refuse to
        // run, so we gate on compile target.
        if cc_over_89 {
            println!("cargo:rustc-cfg=has_sage_sm89_kernels");
        }
        if cc_over_90 {
            println!("cargo:rustc-cfg=has_sage_sm90_kernels");
        }
        if cc_over_100 {
            println!("cargo:rustc-cfg=has_sage_sm100_kernels");
        }
        // MXFP4 is always enabled with CUDA (uses LUT-based dequantization)
        println!("cargo:rustc-cfg=has_mxfp4_kernels");

        // Exclude SageAttention sources from the main builder; they need
        // different SM targets (SM89/SM90/SM100) and a separate include path.
        let mut excluded_files = if cc_over_80 {
            vec!["dummy_*.cu", "*_dummy.cu"]
        } else {
            vec!["marlin_*.cu", "*_fp8.cu", "*_fp8_gemm.cu", "*_wmma.cu"]
        };
        // sage_*.cu / sm89_*.cu live under src/sage_cuda/kernels/ and are
        // already outside the kernels/*/*.cu glob; the explicit exclude here
        // is defense-in-depth in case the glob is broadened later.
        excluded_files.push("sm89_*.cu");
        excluded_files.push("sage_*.cu");
        builder = builder.exclude(&excluded_files);

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrsquant.lib")
        } else {
            build_dir.join("libmistralrsquant.a")
        };
        builder
            .build_lib(out_file)
            .expect("Build mistral quant lib failed!");
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrsquant");
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

        let (major, minor) = cuda_version_from_build_system();
        println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"");

        Ok(())
    }

    #[cfg(feature = "metal")]
    {
        use std::path::PathBuf;
        use std::process::Command;
        use std::{env, str};

        const METAL_SOURCES: [&str; 16] = [
            "bitwise",
            "blockwise_fp8",
            "bnb_dequantize",
            "f8q8",
            "fused_glu",
            "hqq_dequantize",
            "hqq_bitpack",
            "mxfp4",
            "quantized",
            "scalar_fp8",
            "scan",
            "sdpa_with_sinks",
            "softmax_with_sinks",
            "sort",
            "copy",
            "turbo_wht",
        ];
        const HEADER_SOURCES: [&str; 5] = ["utils", "bf16", "scan_impl", "sort_impl", "copy_impl"];
        // Include-only headers (not compiled directly, just tracked for changes)
        const INCLUDE_ONLY: [&str; 2] = ["float8", "float4"];
        for src in METAL_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in HEADER_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in INCLUDE_ONLY {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        println!("cargo::rerun-if-changed=build.rs");

        // Check if precompilation should be skipped
        // https://github.com/EricLBuehler/mistral.rs/pull/1311#issuecomment-3001309885
        println!("cargo:rerun-if-env-changed=MISTRALRS_METAL_PRECOMPILE");
        let skip_precompile = env::var("MISTRALRS_METAL_PRECOMPILE")
            .map(|v| v == "0" || v.to_lowercase() == "false")
            .unwrap_or(false);

        if skip_precompile {
            println!(
                "cargo:warning=Skipping Metal kernel precompilation (MISTRALRS_METAL_PRECOMPILE=0)"
            );
            // Write a dummy metallib file to satisfy the include_bytes! macro
            let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
            std::fs::write(out_dir.join("mistralrs_quant.metallib"), []).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_ios.metallib"), []).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_tvos.metallib"), []).unwrap();
            return Ok(());
        }

        enum Platform {
            MacOS,
            Ios,
            TvOS,
        }

        impl Platform {
            fn sdk(&self) -> &str {
                match self {
                    Platform::MacOS => "macosx",
                    Platform::Ios => "iphoneos",
                    Platform::TvOS => "appletvos",
                }
            }

            fn metal_std(&self) -> &str {
                // Use Metal 3.1 unified standard for all platforms.
                // This fixes Xcode 26+ where the default Metal standard may be too low.
                // https://github.com/EricLBuehler/mistral.rs/issues/1844
                //
                // Note: tvOS devices with A15+ (Apple TV 4K 3rd gen) support Metal 3.1.
                match self {
                    Platform::MacOS | Platform::Ios | Platform::TvOS => "metal3.1",
                }
            }
        }

        fn compile(platform: Platform) -> Result<(), String> {
            let current_dir = env::current_dir().expect("Failed to get current directory");
            let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
            let working_directory = out_dir.to_string_lossy().to_string();
            let sources = current_dir.join("src").join("metal_kernels");

            // Compile metal to air
            let mut compile_air_cmd = Command::new("xcrun");
            compile_air_cmd
                .arg("--sdk")
                .arg(platform.sdk())
                .arg("metal")
                .arg(format!("-std={}", platform.metal_std()))
                .arg(format!("-working-directory={working_directory}"))
                .arg("-Wall")
                .arg("-Wextra")
                .arg("-O3")
                .arg("-c")
                .arg("-w");
            for metal_file in METAL_SOURCES {
                compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
            }
            for metal_file in HEADER_SOURCES {
                compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
            }
            compile_air_cmd
                .spawn()
                .expect("Failed to compile air")
                .wait()
                .expect("Failed to compile air");

            let mut child = compile_air_cmd.spawn().expect("Failed to compile air");

            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        panic!("Compiling metal -> air failed. Exit with status: {status}")
                    }
                }
                Ok(None) => {
                    let status = child
                        .wait()
                        .expect("Compiling metal -> air failed while waiting for result");
                    if !status.success() {
                        panic!("Compiling metal -> air failed. Exit with status: {status}")
                    }
                }
                Err(e) => panic!("Compiling metal -> air failed: {e:?}"),
            }

            // Compile air to metallib
            let lib_name = match platform {
                Platform::MacOS => "mistralrs_quant.metallib",
                Platform::Ios => "mistralrs_quant_ios.metallib",
                Platform::TvOS => "mistralrs_quant_tvos.metallib",
            };
            let metallib = out_dir.join(lib_name);
            let mut compile_metallib_cmd = Command::new("xcrun");
            compile_metallib_cmd.arg("metal").arg("-o").arg(&metallib);

            for metal_file in METAL_SOURCES {
                compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
            }
            for metal_file in HEADER_SOURCES {
                compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
            }

            let mut child = compile_metallib_cmd
                .spawn()
                .expect("Failed to compile air -> metallib");

            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        panic!("Compiling air -> metallib failed. Exit with status: {status}")
                    }
                }
                Ok(None) => {
                    let status = child
                        .wait()
                        .expect("Compiling air -> metallib failed while waiting for result");
                    if !status.success() {
                        panic!("Compiling air -> metallib failed. Exit with status: {status}")
                    }
                }
                Err(e) => panic!("Compiling air -> metallib failed: {e:?}"),
            }

            Ok(())
        }

        compile(Platform::MacOS)?;
        compile(Platform::Ios)?;
        compile(Platform::TvOS)?;

        Ok(())
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    Ok(())
}
