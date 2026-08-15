# Wave 28 — AZ: the 8-token cap on the fused MoE gather

**Scope.** `mistralrs-quant/src/qtip/**` only. No GPU rented; every claim is
CONFIRMED (source read, `file:line`), DERIVED (arithmetic shown), or labelled
UNMEASURED. Base: `origin/master` @ `57bd1ba70`. Branch
`perf/moe-gather-token-cap`.

---

## 0. The one-paragraph answer

The cap of 8 is **conservatism, not correctness** — and it was applied to two
rungs whose over-cap behaviour is not remotely comparable. The kernel's only
real limit is `grid.y = n_pairs ≤ 65535`, i.e. **8191 tokens at top-8**: three
orders of magnitude above the cap. Nothing about shared memory, registers or
occupancy depends on the token count. The bitshift rung's cap is legitimate
(above it sits the amortizing grouped GEMM); the **LUT rung's cap was pure
loss** (above it sits a host-orchestrated dequantize-materialize loop), and the
LUT rung is the one the measured H200 artifact serves. That rung's boundary is
now derived from the traffic ratio between the two paths (~512 tokens at top-8
of 256 rather than 8), the structural limit is enforced as an error instead of
silently returning zeros, and the dedup work turns out to be already done — on
the other rung.

---

## 1. Why the cap is 8 — CONSERVATISM, with evidence

### 1a. Provenance

`git log -S "ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS" -- mistralrs-quant/src/qtip/`
returns three commits; the oldest (`ba026e9d1`) is the fork's squashed import,
so the `8` has no standalone introducing commit and no measurement attached to
it. The only later commit that touched it (`56053aaab`, "FP8 blockwise GEMV
decode kernel + QTIP dequant-materialize decode guard") did not choose the
value — it *codified* it, adding
`DECODE_REGIME_MAX_TOKENS: usize = 8` (`qtip/mod.rs:90`) whose own doc says it
"Matches the default of `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS`". The constant was
named after the magic number, not the other way round.

### 1b. What the authors say the cap is for

The kernel header is explicit that this is a *regime* judgement, not a limit
(`kernels/qtip/qtip_gather_gemv.cu:12-17`):

> Scope: the b=1 (and small-batch / MTP-draft) DECODE regime, where there are
> few (token, slot) pairs. It does one independent gemv per pair, so it does
> NOT reuse a dequantized expert across many tokens — that token-grouping win
> only matters at prefill, which keeps the existing grouped path.

and the LUT dispatch call site said the host path
"stays the PREFILL path (better expert reuse across many tokens)".

**There is one genuine correctness constraint in the area, and it is not a
token bound.** RUN-161: the LUT rung's fallback `gather_forward_cuda` reads the
router to the host (`qtip/mod.rs:2580-2584`,
`indices.to_device(&Device::Cpu)?…to_vec1()`). Under CUDA-graph capture that
read is recorded-not-executed ⇒ garbage expert ids ⇒ out-of-bounds weight read.
So the decode regime must *never* fall back. That argues for the fused path
being used **more**, not less. It cannot justify a cap of 8.

### 1c. The two rungs' fallbacks are not comparable

| rung | `--isq` | over-cap path | what it is |
|---|---|---|---|
| `QtipLayer` (LUT) | `qtip2` | `gather_forward_cuda` (`qtip/mod.rs:2567`) | D2H router sync; `dequantize_rotated_cuda` per **distinct** expert into a `HashMap<usize, Tensor>` held live simultaneously (`:2621-2634`); then per expert `index_select` + `matmul` + `index_add`, the last of which reallocates the whole output each iteration (`:2659-2675`) |
| `Qtip2bLayer` | `qtip2b` | `grouped_gemm_2b_cuda` (`cuda_ops.rs:1689`) | on-device histogram → scans → grouped scatter → persistent tensor-core tile loop, **zero host syncs** (`:1823-1841`) |

`mistralrs-core/src/pipeline/isq.rs:231` maps `"qtip2" | "qtip"` to
`IsqType::QtipBitshift2`, and `mistralrs-quant/src/mxfp4/mod.rs:375-383`
routes everything that is not `IsqType::Qtip2b` to `QtipLayer`. FACTS records
the measured H200 run as an "in-situ **qtip2** W=32 bake" and the published
artifact as `DeepSeek-V4-Flash-UQFF-qtip2`. **The served model is on the rung
with no grouped kernel.** That answers AY's open question #2: the fix for what
was measured is "raise the cap", not "finish the grouped kernel" — but only
because the grouped kernel exists on the *other* rung (§4).

---

## 2. The true limit — arithmetic

Launch geometry, both gather-GEMV kernels
(`qtip_gather_gemv.cu:437-441`, `qtip_bitshift.cu:548-562`):

```
grid  = (ceil(n_rows / (WARPS_PER_BLOCK * ROWS_PER_WARP)), n_pairs, 1)
block = (WARPS_PER_BLOCK * 32, 1, 1)            = 256 threads
smem  = ROWS_PER_BLOCK * packed_per_row          staged iff <= 48 KiB
```

with `WARPS_PER_BLOCK = 8`, `ROWS_PER_WARP = 2` ⇒ `ROWS_PER_BLOCK = 16`.

| candidate bound | value | depends on tokens? |
|---|---|---|
| shared memory / block | `16 * packed_per_row` B. V4 gate/up `K=4096` ⇒ `packed_per_row = 4096/4 = 1024` B ⇒ **16 KiB**, under the 48 KiB staging threshold | **no** — weight shape only. Over 48 KiB the launcher passes `stage_packed = 0` and reads from global: degradation, not failure (`qtip_gather_gemv.cu:435-436`) |
| registers / thread | `ROWS_PER_WARP = 2` accumulators + 2 row pointers + 2 scales + `GROUP = 4` `float2` staged activations + trellis state; `__launch_bounds__(256)` caps allocation | **no** — all compile-time |
| occupancy | more pairs ⇒ more blocks ⇒ strictly better | **no** |
| **`grid.y`** | **65535** (CUDA C Programming Guide, "Technical Specifications per Compute Capability"; unchanged for every CC ≥ 2.0) | **yes** |

⇒ **`n_pairs = n_tokens × n_experts_per_tok ≤ 65535`**, i.e. **8191 tokens** at
top-8, **10922** at top-6. The cap of 8 was ~1000× below the first thing that
breaks.

**And what breaks is silent.** The launchers are `extern "C"` and discard the
`<<<>>>` status; the Rust wrappers return `dev.alloc_zeros(..)`
(`cuda_ops.rs:461`, `:485`, `:509`, and the qtip2b sibling). A launch past `grid.y` fails
with `cudaErrorInvalidConfiguration`, nothing is written, and the caller gets a
**zero MoE layer that looks like a valid tensor**. Fixed: `check_gather_gemv_pairs`
(`qtip/gather_policy.rs`) bails before both launches.

### The LUT rung's derived boundary

Per **distinct** expert, in units of that expert's packed 2-bit bytes
(`n_rows * in_features / 4`):

```
fused    : 1 packed read per (token, slot) PAIR              -> n_pairs units
fallback : BF16 dequantize write (n*k*2) + cuBLAS read (n*k*2)
           = 4*n*k bytes / (n*k/4) packed bytes              -> 16 units per DISTINCT expert
```

With `E(n) = E · (1 − (1 − k/E)^n)` distinct experts, the fused path moves less
memory exactly while `n·k ≤ 16·E(n)`. Left side linear, right side concave and
saturating at `16E` ⇒ a single crossover:

| routing | crossover |
|---|---|
| top-8 of 256 | **511 tokens** (`4088 ≤ 4096`; at 512, `4096 > 4095.9996`) |
| top-6 of 256 (V4-Flash) | **682 tokens** |

Both are comfortably inside the 65535-pair structural limit, so the derived
boundary can never ask the kernel for something it cannot do — asserted in
`derived_boundary_never_exceeds_the_structural_limit` across
`top_k ∈ {1,2,6,8,16,64} × E ∈ {1,8,64,128,256,1024,8192}`.

**What the model deliberately does not claim** (three unmodelled terms, two of
which point the other way — all UNMEASURED):

1. *Favours the fallback:* its 16.8 MB BF16 expert tensor may stay L2-resident
   between the dequantize and the GEMM, making its real HBM traffic ~1 unit
   rather than 16.
2. *Favours the fused path:* the fallback pays a D2H sync + `O(E(B))` kernel
   launches + `O(E(B))` multi-MB allocations **per MoE call**, ~120 calls/step
   on V4. At B=13 (`E ≈ 86`) that is ~41k launches/step. This term grows with
   batch and is the most likely single explanation for the measured 197.2 ms
   per-sequence step cost.
3. *Favours the fallback:* the fused GEMV is trellis-decode-serialisation-bound
   at ~9–15% of peak bandwidth (FACTS, wave16-AF) while the fallback streams.
   Folding only this in moves the crossover down to ~45–60 tokens.

The traffic ratio of 16 is the only one of the four models with no free
parameter, which is why it is the one in the code. §6 says how to settle it.

---

## 3. What changed

* **`qtip/gather_policy.rs` (new)** — the analysis above as executable policy:
  `CUDA_MAX_GRID_DIM_Y`, `GATHER_GEMV_MAX_PAIRS`, `check_gather_gemv_pairs`,
  `expected_distinct_experts`, `lut_fused_gather_preferred`,
  `lut_fused_gather_max_tokens`, `ondevice_max_tokens_override`.
* **LUT rung** (`qtip/mod.rs:3182-3225`) — boundary is now derived per call from
  `(n_tokens, n_experts_per_tok, num_experts)`: **8 → ~511 tokens** at top-8 of
  256, ~682 at top-6. Pure function of shapes, so it costs no device sync and
  stays capture-safe. `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS` still overrides, still
  clamped by the structural limit.
* **RUN-161 floor** — `n_tokens ≤ DECODE_REGIME_MAX_TOKENS` fuses
  unconditionally regardless of routing shape, so no exotic `(k, E)` can push
  decode onto the capture-unsafe fallback.
* **bitshift rung** (`bitshift.rs:1488-1512`) — **cap deliberately unchanged at
  8**, now with the asymmetry documented and the structural clamp applied.
  Raising it would replace a kernel whose cost tracks *distinct experts* with
  one that is linear in *pairs*. It shares the env override.
* **Silent-zeros guard** — `check_gather_gemv_pairs` in both
  `gather_gemv_cuda` and `gather_gemv_2b_cuda`.
* **The knob is now logged** — `ondevice_max_tokens_override` emits one
  `tracing::info!` when set, and the LUT fallback emits one `tracing::warn!`
  naming the token count and the boundary that decided it. Addresses wave27-AY
  §7 ("the measured sweep cannot be reproduced from its own record") for the
  one env var this change touches.

**Still falls back** (unchanged, by design): >511/682 tokens on the LUT rung
(long prefill chunks); >8 tokens on the bitshift rung (to the grouped GEMM);
anything with `ARC_NO_QTIP_ONDEVICE_MOE` set; non-CUDA storage; dtypes outside
BF16/F16/F32; kernels not compiled in.

---

## 4. Dedup at `bitshift.rs:1311` — SCOPED, because it is already built (on the other rung)

The ask was "group tokens by expert, issue one grouped GEMM per expert".
`kernels/qtip/qtip_grouped_gemm.cu` **is** that kernel, and it is finished:
on-device histogram → prefix scans → ragged tile map → grouped scatter
(`launch_qtip2b_moe_route`), then a persistent `mma.m16n8k16` tile loop with
`QG_TILE_M = 16` pairs per m-tile, each expert's bytes cp.async-staged once per
tile. Zero host syncs, so it is capture-eligible. FACTS records grouped-GEMM
parity **5/5 on hardware** (s3).

So the honest verdict is in two halves:

* **Bitshift rung — done.** Dedup exists and is what runs above 8 tokens. The
  only open question is whether its crossover is at 8 or lower, which is a
  measurement (`examples/qtip_grouped_curve.rs` exists precisely for it, and is
  currently unrunnable because its fixture calls the D4-banned
  `QtipMode::Greedy` — BACKLOG, not this change).
* **LUT rung — a rewrite, not a contained change.** `QtipLayer` has no grouped
  kernel and cannot borrow the bitshift one: different codebook (K=4/L=16
  computed Gaussian via `qtip_decode_state`, V=2 weights per state) vs the
  bitshift MCG K=2/V=1 stream, so `q2b_decode_smem` and the
  `window_state_2b` random-access state identity that lets the GEMM decode any
  `(row, k)` in ~4 ALU ops both need re-deriving. **Specified rewrite:**
  port `qtip_grouped_gemm.cu` to a second template instantiation whose
  `decode_smem` evaluates the L=16 state recurrence and calls
  `qtip_decode_state`, keeping the routing kernels and tile geometry verbatim;
  add the K=4 class to `ExpertBpwTable`/`TrellisBpw` (`grouped.rs`) and a
  `grouped_gemm_lut_cuda` wrapper mirroring `grouped_gemm_2b_cuda`. The L=16
  recurrence needs `QTIP_WARMUP_SYMS = 4` symbols of replay per random access
  rather than the bitshift rung's closed-form window, which is the one genuinely
  new piece of design.

**The cheaper alternative, and it is one flag:** bake V4 with `--isq qtip2b`
instead of `--isq qtip2` and the served model lands on the rung that already
has the amortizing kernel. Both rungs are 2 bits/weight; qtip2b carries 20/20
CUDA parity and 5/5 grouped parity on hardware. This is agent BB's territory
(bake config / model card) — flagged, not touched.

**And the ceiling this does not lift.** Even with the cap raised, the fused
path is one GEMV per (token, expert) pair, so its cost is *linear in tokens*:
it converts a fallback that degrades with batch into a path that is merely flat
per token. The `8B/E(B)` amortisation the fleet thesis needs (4.07× at B=128,
8.0× at B=256) comes only from the grouped kernel. Raising the cap should stop
aggregate throughput *falling*; it will not on its own make it rise.

---

## 5. Tests, and proof they can fail

**CPU / CI-gating** (`qtip/gather_policy.rs`, 5 tests, run in the standard
`cargo test -p mistralrs-quant` lane):

| test | asserts |
|---|---|
| `expected_distinct_experts_matches_the_published_curve` | `E(B)` for `B ∈ {1,8,32,128,256}` matches the FACTS table; degenerate inputs finite |
| `lut_boundary_is_where_the_traffic_model_says` | boundary ∈ [480,560] at top-8/256 and [640,730] at top-6/256; flips exactly once; **every count in {1,8,9,13,16,32,64,128} now fuses** |
| `decode_regime_is_fused_unconditionally` | RUN-161 floor holds for a shape the traffic model alone would reject |
| `derived_boundary_never_exceeds_the_structural_limit` | 42 `(k, E)` combinations |
| `pair_guard_rejects_exactly_above_the_grid_limit` | ok at 65535, **Err** at 65536, message names the context |

**Mutation-proved** (each mutation applied to `master`+change, tests re-run,
mutation reverted):

| mutation | result |
|---|---|
| `CUDA_MAX_GRID_DIM_Y` 65535 → 131071 | `pair_guard_rejects_exactly_above_the_grid_limit` **FAILED** |
| `DEQUANT_TRAFFIC_RATIO` 16.0 → 1.0 | `lut_boundary_is_where_the_traffic_model_says` **FAILED** |
| delete the RUN-161 decode floor | `decode_regime_is_fused_unconditionally` **FAILED** |
| `GATHER_GEMV_MAX_PAIRS` → `usize::MAX` | rejected at **compile time** (`arithmetic_overflow` in the guard test) |

Four mutations, four distinct failures, one test each. No test is vacuous.

**GPU-session** (`qtip/mod.rs`, `#[cfg(feature = "cuda")]`):

* `cuda_fused_gather_matches_dequantize_fallback_across_the_old_cap` — for
  `n_tokens ∈ {1, 8, 9, 16, 32, 64, 128}`, calls `gather_forward_cuda_ondevice`
  and `gather_forward_cuda` directly on the same layer/activations/routing and
  requires cos sim ≥ 0.999 **and** max abs error ≤ 2% of the fallback's range.
  It **carries its own anti-vacuity control**: each iteration also compares the
  fused output against the fallback run on a *shifted* expert assignment and
  requires cos < 0.9. An all-zero, constant, or routing-independent output
  fails that control, so the parity assertion above it cannot pass for free.
* `cuda_fused_gather_errors_past_the_grid_limit_instead_of_returning_zeros` —
  drives 65536 pairs into the fused path and requires an **error whose message
  names `grid.y`**, i.e. the "force the fused path beyond its safe limit and
  confirm you get a failure, not silent garbage" mutation, encoded as a test
  rather than performed once by hand.

⚠️ Both GPU tests `eprintln!` and return `Ok(())` when `Device::new_cuda(0)`
fails, matching the file's existing convention — which is exactly the
"vacuously green on the wrong box" pattern BACKLOG item 2 names. They are
GPU-session tests; do not read a green CI lane as evidence they ran.

---

## 6. What needs a GPU

One run, in this order. Nothing here needs a re-bake.

1. **Re-run the wave26-AX B-sweep on this branch, unmodified.** The LUT rung
   now fuses through B=128+ by default. Prediction: the 8→13 token cliff
   disappears and the B=32 row stops being the worst. If aggregate still falls,
   the fused GEMV's linear-in-pairs cost dominates earlier than the traffic
   model says and the boundary belongs at ~45–60 tokens (§2, term 3) — set
   `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS=48` and re-run to confirm before changing
   the constant.
2. **Sweep the boundary directly:** `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS ∈
   {8, 16, 48, 128, 512}` at fixed B=32. The minimum of that curve is the real
   crossover and settles which of the four models in §2 is right. The knob now
   logs itself, so the result is reproducible from its own log.
3. **Run the two CUDA-gated tests** —
   `cargo test -p mistralrs-quant --features cuda cuda_fused_gather`. Both are
   new; neither has ever executed. `…_errors_past_the_grid_limit_…` is also the
   first empirical check that a `grid.y` overflow returns zeros rather than
   raising, i.e. that the guard was necessary.
4. **Only then** consider the bitshift-rung crossover (fix the D4-banned
   `QtipMode::Greedy` fixture in `qtip_grouped_curve.rs` first).

---

## 7. Surfaced, not shipped

> **Noticed:** the gather-GEMV kernels put pairs on `grid.y` (bounded 65535) and
> row-tiles on `grid.x` (bounded 2^31−1) — the assignment is backwards. Swapping
> them raises the structural limit from 8191 tokens to ~2^31 and costs one line
> per kernel, but it is a `.cu` change that cannot be compiled or tested in a
> CPU session. Not needed at any batch we can schedule. Worth a separate change?

> **Noticed:** `QtipLayer::gather_forward_cuda` holds every distinct expert's
> dequantized BF16 tensor live simultaneously in a `HashMap`
> (`qtip/mod.rs:2621-2634`). At E(B)=163 distinct experts × 16.8 MB that is
> ~2.7 GB per MoE call, and `out_flat = out_flat.index_add(...)` reallocates the
> full output once per expert. Even as a prefill path this is worth an arena +
> a single scatter. Worth a separate change?

> **Noticed:** the fleet-relevant fix for the *served* model may be a bake flag,
> not a kernel: `--isq qtip2b` puts V4 on the rung that already has the
> amortizing grouped GEMM (§4). Same 2 bits/weight, 20/20 + 5/5 hardware parity.
> Requires a re-bake and a re-published artifact. Agent BB's scope.

> **Noticed:** three sibling env vars remain silent — `ARC_NO_QTIP_ONDEVICE_MOE`,
> `ARC_NO_QTIP_GROUPED_MOE`, `ARC_WARN_DEQUANT_MATERIALIZE`. This change logged
> only the one it touched. Worth a separate change?
