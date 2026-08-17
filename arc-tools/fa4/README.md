# FlashAttention-4 in Arc — feasibility, routes, and the open gate

## The problem in one line

FA4 is written in **CuTeDSL** and JIT-compiled at runtime through `cute.compile()`. It exposes **no C or C++ API**, so Arc cannot link it the way `candle-flash-attn` (FA2) and `candle-flash-attn-v3` (FA3) link their C++ kernels. The property that makes FA4 trivial for vLLM and SGLang — they are Python — is exactly what makes it hard here.

## Scope — read this before writing a release note

**FA4 is a fleet-wide lever, never a DeepSeek-V4 one.** V4 sets `SdpaParams::sinks` on all 43 layers (`deepseek4.rs:1188-1197`), `run_attention` returns into `sinks_attn` at `attention/mod.rs:133` **before** the flash branch, and `sinks.rs:78` gates the fused sinks kernel on `head_dim ∈ {64,80,96,112,128,192,256}` — V4's is **512**. No V4 layer reaches any FlashAttention kernel of any generation, and no build flag changes that. FA2/FA3 cap head_dim at 256 as well.

Value, if any, is for the head_dim ≤ 256 majority: Llama/Qwen/Mistral-class GQA models.

## Nothing like this exists yet

Checked 2026-08-17, all four negative:

| Where | Result |
|---|---|
| `EricLBuehler/mistral.rs` master | `flash-attn` + `flash-attn-v3` only (`mistralrs-core/Cargo.toml:143-144`). Code search for `flash-attn-v4`, `flash_attn_v4`, `CuTeDSL`, `flash_attn.cute` → **0 hits each**. No PR or issue. |
| `huggingface/candle` main | Crates are `candle-flash-attn`, `candle-flash-attn-v3`. No v4. **0 hits**. |
| `aeonmindai/candle` (our fork) | candle **0.9.2**, base `c3bb5bf`, **112 commits behind** upstream main (0.11.0). Nothing to rebase onto. |
| crates.io | No crate binds FA4 or the CuTeDSL. Closest are CUTLASS **C++** template binders (`baracuda-cutlass`, `atomr-accel-cutlass`), which are nvcc/NVRTC-driven. |

A working binding would be the **first FA4 binding in Rust anywhere**.

## The three cubin routes (research-established, needs on-box confirmation)

Contrary to the initial assumption that driver interposition would be required, **cubin access is documented and first-class** in the CuTeDSL. Three routes exist:

1. **`compiled.__cubin__`** — `cute.compile()` returns an object carrying `__cubin__`, `__ptx__`, `__sass__`, `__mlir__` as documented public attributes (CuTeDSL `guides/debugging.rst`). Cheapest to verify.
2. **`CUTE_DSL_KEEP=cubin`** + `CUTE_DSL_DUMP_DIR` — writes `{fn}.{arch}.cubin` to disk. (`CUTE_DSL_KEEP_CUBIN=1` is the deprecated form.) Leanest to automate in a bake step.
3. **`export_to_c()` / `dump_to_object()`** — emits a linkable `.o` + `.h` exposing `__tvm_ffi_<name>`. **This is what FA4 ships in production and the only route that gives Arc a Rust binding with no Python at runtime and no serve-time JIT.** ABI-safe; the one to build on if it confirms.

### Traps already identified

- **The DSL file cache holds `.mlir` bytecode, not cubins** (`CUTE_DSL_CACHE_DIR`, default `/tmp/{user}/cutlass_python_cache`). The cubin is embedded but needs MLIR bindings to extract. **Do not build a "scrape the cache dir" plan.**
- **NVIDIA's AOT docs are wrong about the runtime entry point.** They show `CuteDSLRT_Module_Load(&module, "path.o")`; `nm -D libcute_dsl_runtime.so` on the shipped 4.7.0 wheel shows **that symbol does not exist**. The real one is `CuteDSLRT_Module_Create_From_Bytes`. **Bind against the wheel's header, not the docs.**
- `--enable-tvm-ffi` combined with `--host-target` raises an error.
- **`.artifacts.CUBIN` is contested** — bytes in one reading of the source, a path in another. Prefer the documented `__cubin__`; the probe resolves this empirically.
- FA4 pins `nvidia-cutlass-dsl>=4.6.2`; the API surface moves between minors. **Pin the wheel** in any binding.

## Hopper caveat — the reason to keep expectations low

FA4's optimization target is Blackwell. Research indicates **Hopper decode is a regression versus FA3**, not merely unbenchmarked. A pre-registered decision threshold of **1.15x over FA3** was recorded *before* measuring, as the level at which building an AOT toolchain would pay for itself. The build proceeded on an explicit product call rather than on that threshold being met — that is a deliberate, recorded choice, not a post-hoc justification.

Also open upstream: the **FA4 sm90 SplitKV gap** (Dao-AILab PR #2415, open and stale) bounds long-context Hopper decode. It is a self-contained patch that could be carried in-fork.

## What is in this directory

### `fa4_probe.py`

One script, six independent sections, each failing soft so a single run yields the whole picture:

1. Environment + SM detection.
2. Which of FA2 / FA3 / FA4 / the DSL actually import.
3. **Empirical head_dim envelope** — sweeps 32…576 and reports what FA4 accepts on *this* GPU (in particular whether 512 is rejected, confirming V4 is out of reach).
4. **JIT cost** — `cute.compile()` cold vs warm, the number the AOT path must beat.
5. **AOT routes** — tries all three above, inspects `libcute_dsl_runtime.so` symbols, and emits a machine-readable verdict.
6. **FA2 vs FA3 vs FA4 benchmark** at fleet shapes, reporting the FA4/FA3 ratio per cell.

Writes `--json` (default `/tmp/fa4_probe.json`). Writes nothing into the repo.

```bash
# MANDATORY on a fresh runcrate image: it ships pip 22.0.2 with no numpy, and
# flash-attn's metadata generation then dies with
#   TypeError: canonicalize_version() got an unexpected keyword argument
#              'strip_trailing_zero'
python3 -m pip install -q --upgrade pip setuptools wheel
python3 -m pip install -q numpy

pip install flash-attn-4
python3 arc-tools/fa4/fa4_probe.py --json /tmp/fa4_probe.json
```

`--quick` runs two benchmark shapes instead of six; `--skip-bench` skips section 6.

## The gate before any Rust is written

The probe's section-5 verdict decides the binding design:

- `USE_ROUTE3_EXPORT_TO_C__no_python_at_runtime` → build a normal `build.rs` + FFI crate against the exported `.o`. This is the good outcome.
- `USE_CUBIN_AOT__cuModuleLoadData_from_rust` → bake-time cubin export plus a driver-API launcher; needs the kernel param layout resolved, which is the expensive part.
- `NO_CUBIN_ACCESS_ON_THIS_VERSION__investigate` → stop and re-scope.

**No Rust FA4 code should land before that verdict exists.** Wiring a feature flag whose implementation is `unimplemented!()` is precisely the "wired but dead code" category the BACKLOG already tracks.

## Related

- `aeonmindai/arc#88` — FA2/FA3 runtime dispatch; FA3 kept opt-in.
- `aeonmindai/candle#6` — port of `huggingface/candle#3606`, the FA3 causal fixes. **Unverified on hardware.**
