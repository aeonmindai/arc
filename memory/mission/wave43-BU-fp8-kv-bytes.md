# wave43 — BU — FP8 K storage + the rANS probe

Branch `perf/fp8-kv-and-bytes`, based on `master` @ `372976933`.
**No GPU.** Every byte count below is read off source or driven through the real
code on CPU; nothing here is measured on hardware. The one *hardware* number in
this document is the rANS probe, which is measured on the real published
artifact (bytes downloaded from `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`).

---

## 1. TARGET 1 — is FP8 K bit-exact for V4? **CONFIRMED, with one correction**

BK's claim (`wave33-BK-attention-kv.md` §5, echoed in
`CEILINGS.json:KV_AND_XS_LADDER.with_FP8_K_storage`) was:

> `act_quant_kv_nope` already E4M3-round-trips K's 448 nope dims with 64-wide
> blocks every forward and then stores the result in BF16. Storing the E4M3
> bytes is **bit-exact** with what is stored today. Target **586**
> B/token/layer ⇒ B ≈ 699.

### The round trip — CONFIRMED

`deepseek4.rs:2264-2306` (pre-change), reached from `deepseek4.rs:1445` on every
layer of every forward:

```rust
let kb    = k_nope.to_dtype(F32)?.reshape([.., nope/64, 64])?;
let amax  = kb.abs()?.max_keepdim(D::Minus1)?;
let scale = (amax / 448.0)?.affine(1.0, 1e-12)?;
let scaled = kb.broadcast_div(&scale)?;
let rt = scaled.to_device(Cpu)?.to_dtype(F8E4M3)?.to_dtype(F32)?.to_device(&dev)?;
let q = rt.broadcast_mul(&scale)?;
let k_nope_q = q.reshape(orig_dims)?.to_dtype(k.dtype())?;   // <-- widened to BF16
Tensor::cat(&[&k_nope_q, &k_rope], D::Minus1)?
```

* the E4M3 round trip: `deepseek4.rs:2296-2302` — real, unconditional, 64-wide
  blocks, exactly `nope = 512 - 64 = 448` dims;
* the BF16 storage: `deepseek4.rs:2304` widens it back, and
  `deepseek4.rs:1605` → `append_kv_mqa` → `SingleCache::append`
  (`kv_cache/single_cache.rs:129-164`, buffer allocated with `src.dtype()`);
* the PagedAttention gate: `PagedCacheType::F8E4M3 -> DType::F8E4M3` lives at
  `paged_attention/cache_engine.rs:18-40` and is selectable only as
  `--pa-cache-type f8e4m3`, while `DeepSeekV4Loader::supports_paged_attention`
  returns `false` (`pipeline/loaders/normal_loaders.rs:3231-3275`).

So the premise holds: **Arc already pays the quantization error and then pays
full BF16 to store the result.**

### The correction — 586 would be LOSSY; 590 is the bit-exact number

The reference's 584-byte row spends **7 bytes** on UE8M0 (power-of-two) block
scales. Arc's scale is `amax/448 + 1e-12`, a general F32. Round-tripping through
a UE8M0 exponent is a *different* scale, so `code * ue8m0_scale != code * scale`
and the reconstructed BF16 differs from what is stored today. That is a quality
change, small but real, and the brief said to stop rather than buy batch with
one.

The bit-exact layout stores **`amax` itself, at activation precision** — which
is free, because `amax` **is one of the block's own elements** and is therefore
already exactly representable in BF16. `scale` is then recomputed at read time
with the *same two candle ops*, so it is reproduced exactly.

| | K half | V half | B/token/layer |
|---|---|---|---|
| pre-wave33 (duplicate V) | 512 BF16 | 512 BF16 | 2048 |
| wave33 (V a 1-wide marker) | 512 BF16 | 1 BF16 | 1026 |
| **wave43 (packed)** | **448 E4M3 codes, U8** | **64 rope + 7 `amax`, BF16** | **590** |
| SGLang DSV4 reference | 448 FP8 | 128 rope + 7 UE8M0 + 1 pad | 584 |

**590, not 586.** The 6-byte difference is the price of exactness. It costs
0.7% of the batch-size win and buys a provable no-op.

**Mutation-proved:** replacing the exact scale with a UE8M0-rounded one — i.e.
building the reference's 584-byte layout — makes
`kv_fp8_roundtrip_is_bit_exact_vs_reference` FAIL. The test has teeth against
exactly the change it is there to prevent.

---

## 2. What was built to decouple it from PagedAttention

Nothing about `PagedCacheType` was touched. The packed layout is provided
directly on the `NormalCache` path, using the two halves of the `Normal` slot
V4 already owns:

**`mistralrs-core/src/models/dsv4_kv_fp8.rs` (new).**
`quantize_k(k, rope_dim, mode) -> Option<V4PackedK>` is now the *only*
implementation of the FP8-QAT round trip. It returns the parts instead of
throwing them away:

* `codes` — `[B, H, T, 448]` **U8**, one E4M3 code byte per nope dim;
* `side` — `[B, H, T, 71]` activation dtype: the rope tail verbatim, then the
  7 block `amax`.

`V4PackedK::dequant` rebuilds the BF16 tensor through a 256-entry E4M3 decode
table (`index_select`), so reconstruction is *exactly* candle's `F8E4M3 -> F32`
cast. `act_quant_kv_nope` is no longer a separate function — the dense tensor
the paged/dummy/graph arms need is literally `quantize_k(..).dequant(..)`, so
the two paths cannot drift.

Deliberate choices, each with a reason:

* **U8, not `DType::F8E4M3`.** candle has no CUDA `F8E4M3` cast kernel (the
  comment at the old `deepseek4.rs:2282` records the "named symbol not found"
  failure); the only CUDA reader of an `F8E4M3` tensor in-tree is
  `mistralrs_quant::blockwise_fp8::fp8_blockwise_dequantize`, which demands
  rank-2 contiguous inputs. U8 + `index_select` uses only ops with guaranteed
  kernels on every backend and imposes no layout constraint.
* **The `amax`, not the scale.** `amax` is lossless in the activation dtype;
  the scale is not.
* **No new `KvCache` variant.** `codes` goes in the K half and `side` in the V
  half of the existing `KvCache::Normal` slot. `KvCache::append`
  (`kv_cache/mod.rs:124-172`) and both cache managers are dtype- and
  width-agnostic, so batching clone-in/clone-out, truncation, prefix reuse and
  MTP rollback keep working with **zero** changes to shared cache
  infrastructure. This is the same device wave33 used for the V marker and the
  `xs` history slot.

**The one seam that had to open: `Dsv4AttentionConfig::raw_prefix`.**
A naive packed cache dequantizes the whole context every step, which at
B≈700 / 2048 ctx writes ~63 GB per step — it would cost more bandwidth than the
storage saves, and turn a win into a regression. The fix is to reconstruct only
the span attention can reach. `dsv4_attention` inferred the absolute query
position from the cache length (`q0 = t_k_full - t_q`), so a pre-narrowed K
silently corrupted both the window mask and the compressed-block causality
threshold — the exact blocker BK named in §6. `raw_prefix` says how many raw
keys the caller already dropped; `t_k_full = raw_prefix + k.dim(2)` and
everything downstream is unchanged. `raw_prefix = 0` is every other path, and
is byte-identical to the previous behaviour.

Decode therefore dequantizes `window + t_q - 1 = 128` columns per layer per
step, not `T`. At 2048 ctx that is 1/16th of the context, and it replaces a
`narrow().contiguous()` copy `dsv4_attention` was already paying — the net extra
traffic is one read of 448 B/token over the window.

**This also unblocks BK §6** (capping the *stored* raw KV at `window`, worth a
further ~30× on this term). Not done here; `raw_prefix` is the prerequisite it
was missing.

### Gates

> 🔴 **CORRECTED 2026-08-16 by wave49-BZ.** This section shipped the gate
> **defaulted ON** (`!(v == "0")`, so unset meant on) and it was merged without
> the GPU exercise this same document called for below. The first V4 forward on
> a commit containing it died — every request, including the engine's dummy run
> — with `dtype mismatch in slice-set, lhs: BF16, rhs: U8` (wave48-BY).
> **FP8 K storage is now OPT-IN: off unless `ARC_V4_FP8_KV=1`.** Read every
> "default" below as "what you get with `ARC_V4_FP8_KV=1`".

`ARC_V4_FP8_KV=1` turns packing on; unset (and any other value) leaves it off.
It is also
automatically off under `ARC_V4_CAPTURE_PROBE`, because the CUDA-graph decode
arm writes through `mistralrs_quant::kvwrite::write_kv_inplace`, which is
instantiated for F16/BF16/F32 only (`kvwrite/mod.rs:98-116`). `append_graph_kv_mqa`
now bails with that instruction rather than failing inside the kernel dispatch.
The graph arm is itself behind `ARC_V4_CAPTURE_PROBE` (`pipeline/normal.rs:1528`),
so nothing in the default serving path is affected.

---

## 3. New numbers

Per token, all 43 layers, plus the post-PR#59 `xs` state (20,224 B/token):

| | B/token/layer | B/token | at 2048 ctx | max B in ~65 GB |
|---|---|---|---|---|
| before PR #59 | 2,048 | 424,018 | 868 MB/seq | ~68 |
| after PR #59 | 2,048 | 108,288 | 222 MB/seq | ~266 |
| after PR #63 | 1,026 | 64,342 | 132 MB/seq | ~493 |
| **after this PR** | **590** | **45,594** | **93.4 MB/seq** | **~696** |

`43 * 590 + 20,224 = 45,594`; `45,594 * 2048 = 93.38 MB`; `65e9 / 93.38e6 = 696`.
Same 65 GB budget and decimal convention `FACTS.md` uses. **Arithmetic, not
measured.** 1.41× more concurrent sequences than PR #63, 10.2× than this morning.

Against `CEILINGS.json`'s saturation table, B≈696 sits at an aggregate ceiling of
**~45,200 tok/s on one H200** (the file tabulates B=699 → 45,400).

---

## 4. Tests

New, all CPU, all in the default `cargo test -p mistralrs-core` lane:

`models::dsv4_kv_fp8::tests`
* `kv_fp8_roundtrip_is_bit_exact_vs_reference` — holds a **verbatim copy** of
  the pre-change `act_quant_kv_nope` and asserts the packed round trip is
  identical **bit pattern for bit pattern** (`bf16::to_bits`), on *both*
  quantizer arithmetics (CPU-exact and the `ARC_GPU_ACT_QUANT` on-device form).
* `kv_fp8_fixture_discriminates` — D12. Asserts the fixture is not one an
  identity `dequant` would pass (quantization changes it), not one a constant
  `dequant` would pass (>32 distinct codes used), and not one a single global
  scale would pass (>4 distinct block scales).
* `kv_fp8_narrow_then_dequant_matches_dequant_then_narrow` — the property the
  O(window) decode rests on.
* `e4m3_table_is_the_cast` — the decode table is the E4M3 grid, all 256 codes.
* `kv_fp8_declines_geometries_without_whole_blocks`.

`models::deepseek4::kv_footprint_tests`
* `v4_kv_bytes_per_token_per_layer` — **the byte count**, 300 real appends
  through `append_kv_mqa` on a slot built as `NormalCache::new` builds V4's.
  Asserts K half = 448 B, V half = 142 B, total **590**, whole-model **25,370**,
  and — measured from the same fixture with the quantizer switched off — that
  the dense fallback is still exactly 1026. Three D12 discrimination asserts
  (code byte narrower than an activation; non-empty rope tail; >1 block/token),
  each naming the wrong implementation it rules out.
* `packed_cache_reconstructs_the_stored_keys` — end-to-end: drives the packed
  store through the real `KvCache` and asserts every reconstructed row is
  bit-identical to what the pre-packing path stored, then that the decode span
  is a suffix of it.

`models::dsv4_attention::tests`
* `raw_prefix_is_equivalent_to_passing_the_whole_cache` — 4 (ratio, t_k, t_q)
  cases across Standard/CSA/HCA, output equality, plus a
  `narrowed_at_least_once` guard so the sweep cannot pass with `base == 0`
  everywhere.
* `raw_prefix_past_the_reachable_span_is_refused`.

### Mutation proofs (all three run, all three observed)

1. **Revert the layout** — `append_kv_mqa` stores dense again
   (`packed.filter(|_| false)`) ⇒ `v4_kv_bytes_per_token_per_layer` FAILS with
   `left: 1024, right: 448` (the old 512×BF16 K half, i.e. 1026 B/token/layer),
   and `packed_cache_reconstructs_the_stored_keys` FAILS on the layout assert.
2. **Make it lossy** — dequantize with a power-of-two (UE8M0) scale, i.e. build
   the reference's 584-byte layout ⇒
   `kv_fp8_roundtrip_is_bit_exact_vs_reference` FAILS on `CpuExact`.
3. **Ignore the seam** — `t_k_full = t_k_given` (drop `raw_prefix` from the
   absolute position) ⇒ `raw_prefix_is_equivalent_to_passing_the_whole_cache`
   FAILS while all 16 other `dsv4_attention` tests still pass, so the new test
   is the only thing standing between a packed decode and a shifted window.

Green: `cargo check -p mistralrs-core`; `cargo test -p mistralrs-core
-p mistralrs-quant` (292 + 12 + 231 + 2, 0 fail); the scoped clippy lane clean;
`mistralrs-core` clippy count **183, down from 184 on master**. `rustfmt` applied
to the new file only.

---

## 5. TARGET 2 — the rANS / entropy-coding probe: **a measured NO**

Measured on the **real published artifact**, not a synthetic fixture:

* `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` shard `qtip2-7.uqff` (541,597,883 B,
  downloaded whole) — one QTIP blob `mtp.7`, `blocks` U8 `[256, 4096, 512]`,
  **1.074 billion K=4 symbols / 2.147 billion weights**;
* shard `qtip2-0.uqff` tensor `17` (HTTP range request, first 8 MB of its
  `blocks`) — `blocks` U8 `[256, 2048, 1024]`, `in_features = 4096`, a
  structurally different 256-expert stack, **16.8 million symbols sampled**.

| | order-0 | order-1 | order-2 |
|---|---|---|---|
| `mtp.7` symbol entropy (of 4.000 bits) | **3.99987** | 3.99903 | 3.98962 |
| implied saving on the symbol stream | **0.003%** | 0.024% | 0.259% |
| shard-0 layer 17 | **3.99980** | 3.99860 | — |

The 16-symbol histogram is flat to within ±0.2 points of uniform
(6.117%–6.442% against 6.25%). Byte-level order-0 entropy is 7.99890 of 8.

**This is QTIP working as designed.** The Hadamard incoherence rotation makes
the input approximately Gaussian and the trellis maps it onto a near-uniform
16-symbol alphabet; an entropy coder on top has nothing left to take.

**Decode cost — and why the answer is worse than "0%".** rANS decode is
inherently sequential: symbol *i* needs the decoder state after symbol *i-1*.
The serving path does not read the stream sequentially — the grouped GEMM reads
`blocks[e, n, :]` for arbitrary `(e, n)`. Preserving that random access means
one independent rANS stream per row, each carrying a state flush (typically
4 bytes). On a 512-byte row that framing is **+0.78%** — 150–250× the entropy it
would recover. **rANS makes this artifact strictly bigger, before a single
cycle of decode arithmetic is spent.** The probe is closed: no.

**Two things the probe did surface, both unbuilt:**

* `row_scales` are F32: 4,194,304 B of a 541,597,795 B layer = **0.77%**. At
  BF16 that is ~0.39% of the whole artifact, and unlike rANS it is free at
  decode (the dequant already multiplies by a per-row scalar). Whether BF16
  row scales cost quality is unmeasured.
* The `lut` is F32 `[65536, 2]` = 524,288 B and is written **per layer**, though
  the code and docs both call it the *shared* Gaussian trellis LUT. Across the
  artifact's QTIP blobs that is on the order of **90 MB of byte-identical
  duplication (~0.12%)**. Deduplicating it is a UQFF-format change.

---

## 6. TARGET 3 — indexer-K MQA collapse: scope only

**The BACKLOG entry is real, and its cost line is half right.**

`dsv4_indexer.rs:324-327`:

```rust
let indexer_k = indexer_k_mqa      // [B, T_c, D] — ONE key set, MQA
    .unsqueeze(1)?
    .expand((b, self.n_heads, t_c, self.head_dim))?
    .contiguous()?;                // <-- materialises n_heads identical copies
```

With V4's `index_n_heads = 64`, `index_head_dim = 128` (`deepseek4.rs:340-345`,
confirmed in the shipped `config.json` fixture at `deepseek4.rs:4684-4686`),
`.contiguous()` turns 256 B/slot into 16,384 B/slot — **64×, confirmed**. At
T_c = 512 (2048 ctx, ratio 4) that is 8.4 MB written and re-read per CSA layer
per forward instead of 131 KB.

**Correction to the BACKLOG wording:** it is 64× on *memory and DRAM traffic*,
not on compute. Each head's score is a different dot product (`q` differs per
head), so the FLOPs are inherently per-head either way. The win is that 64 heads
would read one 131 KB key set out of L2 instead of a 8.4 MB one out of DRAM.

**What it would take** (not built):

* Rust reference path: `indexer_logits` (`dsv4_indexer.rs:401-420`) does
  `q.matmul(&k_t)`. Swapping to `broadcast_matmul` against a `[B, 1, D, T_c]`
  key lets the `.contiguous()` go. Roughly a one-line change plus dropping the
  `expand`.
* CUDA path: `arc-cuda-graph/src/cuda/flashmlasparse/indexer_score.cu` indexes
  `k_base = k + bh * t_c * HEAD_DIM` with `bh = b * H + h` (lines 139, 221, 329,
  381 — four kernel variants: BF16/F32 × templated/generic). The fix is a
  `k_head_stride` parameter, `0` for MQA, so `k_base = k + b * k_batch_stride +
  h * k_head_stride`. Four one-line edits, plus the host wrapper's shape
  validation in `arc-cuda-graph/src/flashmlasparse.rs:242-249`, plus a CPU
  reference parity test. **Needs a GPU to validate.**

**But it costs nothing today.** BK's "the Lightning Indexer is loaded and never
called" is confirmed: `V4Indexer` is constructed at `deepseek4.rs:1134-1144` and
stored at `:918`, and there is **no call site** in the forward path (the only
other mentions in the file are doc comments and test scaffolding). So the 64× is
latent cost that fires the day the indexer is wired, not a cost being paid now.
Fix it *with* the wiring, not before.

---

## 7. What needs a GPU

Everything in §1–§4 is verified on CPU; the arithmetic is dtype- and
backend-independent, but three things are exercised on CUDA for the first time
by this change and are unverified:

1. **A U8 `SingleCache`** — `Tensor::zeros`/`slice_set`/`narrow` on a U8
   `[B,1,T,448]` buffer. candle's CUDA backend covers this (`copy2d_u8` at
   `cuda_backend/mod.rs:2367`, `const_set` at `:1578`, strided copy at `:2540`),
   but no Arc test has run it.
2. **`index_select` on the 256-entry LUT** with U32 indices over a
   `[B*1*window*448]` index tensor, 43 times a step.
3. **The H2D leg of the code extraction.** The CPU-exact quantizer already
   synced per layer; it now carries back U8 instead of F32 — a quarter of the
   bytes — but the code path is new.

The cheap first check on a box: run the existing generation smoke with
`ARC_V4_FP8_KV=1` and with it **unset** (the default since wave49-BZ), and diff
the greedy token stream. They must
be **identical**, because the change is bit-exact; any divergence is a CUDA-path
bug, not a quality question. Then re-measure the batch sweep — the claim to
falsify is 493 → ~696 concurrent sequences at 2048 ctx.

> 🔴 **wave49-BZ addendum.** One of the three items above has since been
> exercised — on CPU, where it should have been exercised in the first place.
> The U8 `SingleCache` never got as far as `copy2d_u8`: the engine installs a
> **preallocated BF16 `[1,1,cap,512]` buffer** as `all_data` before the first
> append (`engine/add_request.rs:464` → `kv_cache/mod.rs:802`), so the packed
> append met it at `single_cache.rs:161` and `slice_set` rejected the dtype.
> Fixed in `SingleCache::append`, which now rebuilds a preallocated buffer the
> source does not fit while the cache is still empty. Items 2 and 3
> (`index_select` over the LUT, the U8 H2D leg) remain **GPU-only and
> unexercised**.

Not needed: any re-bake, any quality re-measurement, any PagedAttention work.
