# wave63-CN — ArcFlash: naming the kernel, and rung 0

**Date:** 2026-08-17
**System name:** **ArcFlash**, under `ArcAttention` (`TAXONOMY.md` §2.3).
**Rung reached:** 0 of 6 — shipped, CPU-exact, hardware-unmeasured.

---

## The name

Jish's standing rule is that no subsystem is left without an absolute parent
name. `ArcAttention/Flash` was already taken, and it means the wrong thing: it
is the **capability envelope for Dao's FA2/FA3**, i.e. a wrapper around someone
else's kernel. Arc's own kernel therefore needs its own name, and reusing
"Flash" for both would guarantee the two get conflated in exactly the claims
that matter.

```
Arc → ArcInfer → ArcAttention → ArcFlash
                                ├── ArcFlash/Plan    shape → path + the D18 engagement record
                                ├── ArcFlash/Fold    GQA/MQA expansion as a reshape, never a copy
                                ├── ArcFlash/Tile    query-row tiling under a byte budget
                                └── ArcFlash/Cute    the fused CuTeDSL d=512 kernel — NOT BUILT
```

`ArcFlash/Cute` is named here so rungs 3–6 have a home. **No code claims it
exists.** `arc-tools/fa4/README.md` forbids landing Rust before a verdict, and
that still holds.

---

## What rung 0 changes

`mistralrs-core/src/attention/arcflash.rs` (new), wired at the two sites that
performed the GQA expansion:

* `attention/backends/sinks.rs` — every DeepSeek-V4 layer, because head_dim 512
  is outside the fused sinks kernel's `{64,80,96,112,128,192,256}`.
* `attention/mod.rs::run_attention_noflash` — every model that fails
  `can_use_flash`, now that `repeat_kv` has been moved *inside* the cuBLASLt arm
  instead of running ahead of all three.

Two structural facts, both verified against the pinned candle
(`aeonmindai/candle` @ `c3bb5bf`) rather than recalled:

1. **`Tensor::reshape` is a view when the input is contiguous** — the fold costs
   nothing. `repeat_kv` maps `x[b,kv,s,d] → y[b, kv*n_rep + rep, s, d]`, i.e.
   heads are kv-major, so `q[B,H,Tq,D] → [B,Hkv,g*Tq,D]` is exactly the same
   bytes and both GEMM operands then carry batch `B*Hkv`.
2. **`broadcast_matmul` is not an alternative** — candle concretises the
   broadcast operand via `.contiguous()` (its own `// TODO: Avoid concretising
   the broadcasted matrixes`), which would reintroduce the copy.

The trick is not new in this tree: `absorbed_mqa_decode`
(`models/dsv4_attention.rs`) already does it at `t_q == 1` and is pinned by
`absorbed_decode_matches_repeat_kv_reference`. ArcFlash generalises it to any
`t_q` and any GQA ratio.

**Query-row tiling is exact, not an approximation.** Each output row is an
independent softmax, so splitting the query axis computes the same rows — no
online rescaling, and sinks are untouched because a sink is a per-head constant
in each row's denominator. Tiling the *key* axis would need rescaling; ArcFlash
deliberately does not.

---

## Numbers — label each one

**ARITHMETIC (not measured).** One layer's attention allocations at V4 prefill,
`B=1, H=64, D=512, T=1400, bf16`:

| | Old | ArcFlash rung 0 |
|---|---|---|
| GQA expansion of K and V | 175 MiB | **0** |
| score matrices | 478 MiB (2 × full `[B,H,T,T]`) | 64 MiB (2 × tile, 187 rows) |
| folded query tile | — | 12 MiB |
| K/V as given | — | 3 MiB |
| **peak** | **~653 MiB** | **~78 MiB** |

**≈8.3×.** Pinned by `v4_prefill_peak_bytes_drop_by_an_order_of_magnitude`,
which asserts >8× and refuses to pass if the shape does not actually tile.

**MEASURED (CPU only, D14-labelled):** 385 `mistralrs-core` lib tests green,
including the 17 pre-existing `dsv4_attention` tests, which now execute through
ArcFlash.

**NOT MEASURED:** every GPU number. No wall-clock, no VRAM, no tok/s claim is
made. `arc-tools/arcflash/rung0_gpu_validate.sh` is staged for main.

---

## Mutation results — four applied, four caught

The trap that hit five PRs in five disguises is a test that cannot fail. Each
mutation was applied to the shipped source and reverted:

| Mutation | Caught by |
|---|---|
| GQA heads treated as rep-major instead of kv-major | `fold_matches_repeat_kv_across_gqa_ratios` (max abs diff 1.5) |
| mask does not follow the query tile | `query_tiling_is_exact_and_the_mask_follows_the_tile` |
| sink dropped from the softmax denominator | 2 ArcFlash tests **and 3 pre-existing V4 tests** |
| vendor flash disabled, so everything falls to Tile | the D18 **control** arm |

The third is the load-bearing one: it shows the existing V4 suite genuinely
covers this path, so ArcFlash is not tested only by its own author's tests.

---

## D18

`note()` increments a process-wide counter per path and logs
`ArcAttention: first dispatch on <path>` once. That line is the hardware signal:
**present in the arm that uses the path, absent in the arm that does not.**

The test instrument is a `#[cfg(test)]` thread-local mirror, not the atomics —
`cargo test` runs test fns on parallel threads, so a control arm reading the
process-wide counter would be reading other tests' dispatches. That was a real
failure during development, not a hypothetical.

`rung0_gpu_validate.sh` checks **both directions**: V4 must log
`arcflash-tile`, and a GQA control must not.

---

## What rung 0 does NOT do

* **It does not make FA2 stop being the default for GQA models.** That default
  exists because the pinned `candle-flash-attn-v3` has two specific defects —
  NULL `tile_count_semaphore` on dense causal, and a window clamp that rewrites
  `causal` to full attention on varlen causal (`huggingface/candle#3606`). Both
  live in `aeonmindai/candle`, **a different repository**, so they cannot be
  fixed from this branch. Flipping the default is a two-patch job in the fork
  plus a benchmark, and `fa3_preferred()` is already the single switch.
* **It is not the fused kernel.** ArcFlash/Tile is three GEMMs and a softmax
  with bounded working set. The fused d=512 MQA-with-sinks kernel is rungs 3–5.

## Next rung

Rung 2 (vanilla FA4 binding) is unblocked per wave62-CM. For rung 3, read
`flashinfer/cute_dsl/attention/dsa/hca_fp8.py` (4,001 lines, BSD-3-Clause)
first: it is V4 HCA in CuTeDSL, differing from Arc's target essentially in
`arch_str`. ~26% ports as-is and carries the semantics; ~17% is deleted rather
than ported (p_cor corrects tcgen05→TMEM, which Hopper does not have); ~57% is
genuine redesign. **D17** — the KV element format must be a kernel parameter
(BF16 / FP8 / TurboQuant), not hardcoded BF16 — enters at rung 3, not before;
adding a dead enum now would be exactly the wired-but-dead debt BACKLOG tracks.

## Surfaced, not shipped

* `dsv4_attention.rs` builds `k_cat` and `v_cat` with two identical
  `Tensor::cat` calls, despite `V == K` being a V4 invariant — a full redundant
  copy of the union per layer per step.
* `chunked_attention` narrows a rank-4 mask on dim 2 unconditionally; a mask
  that is size-1 on the query axis would be mis-selected. ArcFlash's
  `narrow_mask_rows` guards this; the older helper does not.
