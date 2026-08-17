# FlashAttention-4 in Arc — feasibility, routes, and the open gate

## The problem in one line

FA4 is written in **CuTeDSL** and JIT-compiled at runtime through `cute.compile()`. It exposes **no C or C++ API**, so Arc cannot link it the way `candle-flash-attn` (FA2) and `candle-flash-attn-v3` (FA3) link their C++ kernels. The property that makes FA4 trivial for vLLM and SGLang — they are Python — is exactly what makes it hard here.

## Scope — two deliverables, and they must never be conflated in a claim

**1. Vanilla FA4 binding (head_dim ≤ 256) — fleet-wide, NOT a V4 lever.**
V4 sets `SdpaParams::sinks` on all 43 layers (`deepseek4.rs:1188-1197`), `run_attention` returns into `sinks_attn` at `attention/mod.rs:133` **before** the flash branch, and `sinks.rs:78` gates the fused sinks kernel on `head_dim ∈ {64,80,96,112,128,192,256}` — V4's is **512**. No V4 layer reaches a stock FlashAttention kernel of any generation. FA2/FA3 cap head_dim at 256 too. Value here is the Llama/Qwen/Mistral-class GQA majority.

**2. FA4 extended to MLA — head_dim 512, MQA, with sinks — IS V4's fused kernel.**
This is the single largest identified V4 speed lever. V4 today runs unfused matmul + `softmax_with_sinks` on all 43 layers. Extending the CuTeDSL schedule to d=512 is what produces that kernel, and it is why the durable substrate was chosen over a per-arch hand-written one.

**Keep these separated in every PR, doc, and release note.** Deliverable 1 makes no V4 claim. V4 value appears only at rung 3 and above.

## Why CuTeDSL and not FlashMLA

Hand-written CUDA (FlashMLA) is sm90-locked: adopting it means re-porting *and* re-applying our extras (sinks, N-region reads) at every hardware generation, permanently downstream of someone else's release cadence. CuTeDSL retargets because tile shapes are parameters. This is a product decision — the durable substrate over the fast patch — recorded here so it is not re-litigated, and revisited only if **rung 1 fails**.

## The ladder

Each rung is independently testable. Build in order.

| Rung | What | V4? |
|---|---|---|
| **1** | **THE GATE.** `export_to_c()` → linkable `.o` with `__tvm_ffi_<name>`, callable from Rust, no Python at runtime. | — |
| 2 | Vanilla FA4 callable from Arc at head_dim 64/128, output matching the Python reference. Proves the binding independently of kernel novelty. | no |
| 3 | Extend to **head_dim 512 with MQA**. The wall is ~228 KB SM shared memory against a ~320 KB naive tile budget at d=512. The exploit V4 hands us and stock FA cannot assume: **64 query heads, 1 KV head** — load K/V once, reuse across all heads. Study `absorbed_mqa_decode` (`dsv4_attention.rs:125`), which already avoids materialising 512-wide K/V per head at `t_q == 1`. | **yes** |
| 4 | **Attention sinks in the softmax.** No FA and no FlashMLA release has this. The sink is a **per-head scalar in the denominator — zero cache bytes**, not a KV region. | **yes** |
| 5 | **N-region read.** V4 reads 2 regions: raw sliding window (128) ++ compressed KV (ratio 4 = CSA, 128 = HCA; ratio 0 = window-only on layers **{0,1,43}** — 43, not 42; layer 42 is CSA). Must compose with PR #90's segment allocator, whose finding is that the existing gather kernel treats **a segment as a row**, needing zero `.cu` changes. Read `memory/mission/wave61-CL-segment-allocator.md` first. | **yes** |
| 6 | Wire into `dsv4_attention` and measure against the unfused `softmax_with_sinks` baseline. **That delta is the deliverable number.** | **yes** |

### Not starting cold

- `arc-cuda-graph/src/cuda/flashmlasparse/{indexer_score.cu,topk_radix.cu}` — a port of SGLang's FlashMLASparse (`fp8_paged_mqa_logits` family, sm90 decode sparse), specialised to BF16. That is the sparse-selection half; it also documents the V4Indexer contract and interface shape.
- Dao-AILab discussion **#1474**, *"How to Extend FlashAttention to Nearly Infinite HeadDim and Achieve Fully Fused MLA?"* — upstream is asking exactly the rung-3 question and it is unanswered. Read before designing rung 3; solving it is genuinely novel and worth upstreaming.

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

## Rung 1 — run this first, alone

```bash
bash arc-tools/fa4/rung1_gate.sh 2>&1 | tee /tmp/arc_fa4_rung1/gate.log
```

Bootstraps pip/numpy/`nvidia-cutlass-dsl`, exports a trivial `@cute.jit` kernel, describes the resulting ABI, then generates a Rust harness, links it against the `.o`, runs it, and `ldd`s it. Four checks:

| | Check | Meaning |
|---|---|---|
| C1 | Rust links the `.o` and resolves `__tvm_ffi_<name>` | callable |
| C2 | the symbol is a TEXT symbol | real code, not a stub |
| C3 | the linked binary has **no** `libpython` | no Python at runtime |
| C4 | the binary runs to completion | usable |

Send back `/tmp/arc_fa4_rung1/manifest.json`, `/tmp/arc_fa4_rung1/link_verdict.json`, and `rustc.log` if the link failed.

**Verdicts:**

- `GATE_PASSES__export_to_c_is_linkable_from_rust_without_python` → the substrate bet is sound; start rung 2.
- `LINKS_BUT_DRAGS_PYTHON__investigate` → determine whether `libpython` comes from the harness or the runtime `.so`.
- `GATE_FAILS__no_rust_callable_object` / `NO_OBJECT__GATE_FAILS` → **STOP AND REPORT IMMEDIATELY.** FlashMLA-per-arch becomes the only road and the substrate decision gets revisited on this evidence.

The probe assumes nothing about the API surface: `export_to_c` is tried in four argument shapes and `dump_to_object` in two, each outcome recorded separately, and the generated header is dumped **verbatim** — the header is the ABI contract, and reading it beats guessing at it. One run yields everything needed to design rung 2 whether it passes or fails.

The exported kernel is a trivial no-op **on purpose**: rung 1 tests the toolchain, not the kernel, so DSL-syntax drift surfaces loudly instead of being mistaken for an export failure. Stage C takes the symbol's *address* rather than calling it — invoking needs the TVM FFI argument pack, which is rung 2; `nm` type `T` already proves it is code.

**No Rust FA4 code lands before this verdict exists.** A feature flag whose implementation is `unimplemented!()` is precisely the "wired but dead code" debt the BACKLOG tracks.

## Related

- `aeonmindai/arc#88` — FA2/FA3 runtime dispatch; FA3 kept opt-in.
- `aeonmindai/candle#6` — port of `huggingface/candle#3606`, the FA3 causal fixes. **Unverified on hardware.**
