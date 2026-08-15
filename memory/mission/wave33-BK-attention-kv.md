# wave33 — BK — V4 attention KV cache layout and size

Branch `perf/attention-kv-footprint`, based on `master` @ `3460656d3`.
No GPU. Every number below is either read off source (`file:line` given) or
arithmetic over source; nothing here is measured on hardware.

---

## 1. The claim: "Arc stores 1,024 B/token/layer vs the reference's 584"

**Half-confirmed, and the Arc side was under-counted by exactly 2×.**

### Reference side — 584 is REAL, and V4-specific

`research/code/06_foundation/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:93-111`

```python
    def get_bytes_per_token(self) -> int:
        dim_per_token = (
            self.qk_nope_head_dim                                   # 448, FP8 E4M3
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize  # 64 * 2 (BF16)
            + self.qk_nope_head_dim // self.quantize_block_size     # 448/64 = 7 UE8M0 scales
            + self.scale_pad                                        # 1
        )
```
with its own assert at `:108-111`:
```
assert bytes_per_token == 448 + 64 * 2 + 8, (
    "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
    "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token")
```
Constants at `:73-76` (`scale_pad = 1`, `quantize_block_size = 64`,
`rope_storage_dtype = bfloat16`), re-asserted in the write kernel
(`layers/attention/dsv4/index_buf_accessor.py:16-30`: nope 448 / rope 64 /
scales 7). Duplicated in the sizing path at
`model_executor/pool_configurator.py:373`.

**584 = 448 + 128 + 7 + 1.** It is NOT "576 fp8 + 8 scale" — that misreading
comes from 448+64 = 512 = `kv_lora_rank`. The V3.2/DSA family's fp8 figure is
**656**, a different number for a different model
(`model_executor/model_runner_kv_cache_mixin.py:173-182`).

### Arc side — 2,048 B/token/layer, not 1,024

Geometry (`mistralrs-core/src/models/deepseek4.rs:4140-4157`, the verbatim
V4-Flash `config.json` fixture): `head_dim = 512`,
`num_key_value_heads = 1` (MQA), `num_hidden_layers = 43`,
`torch_dtype = bfloat16`.

The live path is the **non-paged** in-process cache —
`DeepSeekV4Loader::supports_paged_attention` returns `false`
(`pipeline/loaders/normal_loaders.rs:3231-3275`) — so the KV slot is
`KvCache::Normal { k: SingleCache, v: SingleCache }`
(`kv_cache/mod.rs:46-68`), built by
`NormalCache::new(43, max_position_embeddings)` at `deepseek4.rs:3442`.

Pre-fix append, `deepseek4.rs:1459` then `:1586`:
```rust
let v = k.copy()?;                             // V is a bit-identical copy of K
...
let (k_cached, v_cached) = kv_cache.append(&k, &v)?;   // BOTH stored
```
`SingleCache::all_data` is a tensor in the activation dtype
(`kv_cache/single_cache.rs:9-16`, allocated with `src.dtype()`), so per token
per layer:

```
K:  1 kv_head * 512 dims * 2 B (BF16) = 1024 B
V:  1 kv_head * 512 dims * 2 B (BF16) = 1024 B   <-- a duplicate of K
                                        ------
                                        2048 B
```

Cross-check: 2048 * 43 = **88,064 B/token**, which is exactly the KV figure
already recorded in `FACTS.md` ("88,064 B/token KV (43 layers)"), and
88,064 / 108,288 = **81.3%** — matching BJ's own "attention KV is ~81% of what
remains". BJ's *ratio* was right; BJ's per-layer number counted only the K half.

**Verdict: Arc was 2,048 vs the reference's 584 = 3.51×, not 1.75×.**

The paged path allocates the same 2×: key block
`(n_kv_heads, k_head_dim/x, block_size, x)` and value block
`(n_kv_heads, v_head_dim, block_size)` —
`paged_attention/cache_engine.rs:440-484` — with
`k_head_dim = v_head_dim = 512` from `deepseek4.rs:3484-3485`. Dead for V4
today, but it would carry the same duplication if turned on.

## 2. Cause

Candidate (a) from the brief: **storing K and V separately when one latent
serves both.** V4 has a single fused `wkv` projection; `v = k.copy()` is
unconditional and the comment at `deepseek4.rs:1449-1458` says so
("K and V come from the same wkv tensor"). The `copy()` (rather than `clone()`)
exists only to dodge a `PagedAttention::reshape_and_cache` RwLock self-deadlock
— on a path V4 does not use.

Candidate (b), dtype, is the *remaining* 1026/584 = 1.76× (see §5).
Candidate (c), padding to 512, is **not** a cause: 448+64 = 512 exactly.
Candidate (d), broadcast-to-`n_heads`, is **not** present on the KV store path
(`num_kv_heads = 1` all the way into the cache). PR #55's indexer-K broadcast is
a separate, still-open issue — and the indexer is not even called (§7).

## 3. Shipped

### (i) V is a marker, not a duplicate — `2048 -> 1026` B/token/layer

`deepseek4.rs`: new `V4_V_MARKER_WIDTH = 1`, `v4_v_marker`, `append_kv_mqa`,
`append_graph_kv_mqa`, `require_normal_kv_slot`. The V half of the slot now
stores a `[B, 1, T, 1]` zero marker; `dsv4_attention` is handed the cached K as
both K and V, which is what it was already receiving (`v == k`).

The V half cannot simply be dropped: `NormalCacheManager::clone_in_cache` /
`clone_out_cache` unwrap `v.all_data` unconditionally
(`kv_cache/mod.rs:431,478,616,664`) when batching sequences in and splitting
them out. A 1-wide marker keeps every batching / chunking / truncation path
working. **This is the same device this file already used for the `xs` history
slot** (see the R2/R3 comment at `deepseek4.rs:1341-1342`: "`v` is a
`[B, T, 1]` zero marker kept in lockstep because the cache managers require
both sides populated") — so it is a proven pattern here, not a new one. Zero
changes to shared cache infrastructure.

Side effect: the `k.copy()` moved into the paged arm, so the live decode path
no longer materialises a `[B,1,T,512]` copy per layer per step either.

### (ii) Raw attention working-set narrowing — decode reads `window`, not `T`

`dsv4_attention.rs`: new `raw_keep_span(t_q, window, t_k_full)`. Query row `r`
sits at absolute position `q0 + r` and attends raw key `j` iff
`q0 + r - window < j <= q0 + r`; over `r in [0, t_q)` the union is the trailing
`t_q + window - 1` keys. Everything earlier is `-inf` on **every** row, so
dropping it is an identity. K, V and the caller's mask are narrowed to that span
before the union is built; `kp` now carries absolute positions so the masks are
unchanged.

Why it matters: the cost of carrying the dead columns was never the mask — it
was the `Tensor::cat` (which copies the whole raw cache, twice, every decode
step) and the scores GEMM. At 2048 ctx a CSA decode step goes from
`2048 + 512 = 2560` union columns to `128 + 512 = 640`; HCA from `2064` to
`144`. Prefill is untouched (there the reachable union *is* the whole cache).

This is the **read** side only. The store is still grown to full length, because
`q0` is inferred from the cache length — see §6.

### (iii) `ARC_TIME_DECODE` retargeted

- The emit block lived **inside the `use_4d_mhc` arm** of
  `DeepSeekV4::forward`. A checkpoint without the global mHC head ran with
  `ARC_TIME_DECODE=1` and printed nothing. Now a single `emit_decode_profile`
  call covers both arms, and the 3-D arm logs a one-shot warning saying it has
  no component timers (they live in `DecoderLayer::forward_4d`) instead of being
  silently dead. **Instrumentation that is present, plumbed, documented and
  silent — case #10.**
- The log line carried **no batch geometry**, so a B=64 profile and a B=1
  profile were indistinguishable after the fact. It now stamps
  `b= t= tokens= forward_total= (ms/token)`. The 49%/16%/16%/16% split we have
  been steering by was taken at b=1 on the abandoned cudnn build and the log
  itself did not record either fact.
- The gate was `std::env::var_os` **per timer call** — 9 components x 43 layers
  = ~390 environment scans per forward *when the profiler is off*. Now a
  `OnceLock`, matching `absorbed_decode_disabled()` in `dsv4_attention.rs`.

The profiler now yields a self-describing B=1 and B=64 profile for free on the
next GPU session. It still syncs per component, so `forward_total` is an upper
bound on the real step time — that was already true and is documented at
`deepseek4.rs:2212-2216`.

## 4. Numbers

| | B/token/layer | B/token (43 layers) |
|---|---|---|
| Arc before | 2,048 | 88,064 |
| **Arc after** | **1,026** | **44,118** |
| SGLang DSV4 reference | 584 | 25,112 |

Whole per-token context budget (KV + the post-PR#59 `xs` state, 20,224 B/token):

| | B/token | at 2048 ctx | max B in ~65 GB |
|---|---|---|---|
| before PR #59 | 424,018 | 868 MB/seq | ~74 |
| after PR #59 | 108,288 | 222 MB/seq | ~293 |
| **after this PR** | **64,342** | **132 MB/seq** | **~493** |

(~65 GB usable = 141 − 68 weights − ~8, the same budget `FACTS.md` uses;
decimal GB/MB, same convention. Arithmetic, not measured.)

**1.68× more concurrent sequences at 2048 ctx.** B=256 now has ~1.9× headroom
rather than ~1.1×.

## 5. FP8 KV — feasible, and it would be LOSSLESS, but it is not wired

**An FP8 KV path exists in-tree**: `PagedCacheType::F8E4M3 -> DType::F8E4M3`
(`paged_attention/cache_engine.rs:18-40`), selectable as `--pa-cache-type
f8e4m3`, with block shapes computed from `dtype.size_in_bytes()`
(`cache_engine.rs:455-465`).

**V4 cannot reach it.** It is behind PagedAttention, and
`DeepSeekV4Loader::supports_paged_attention` returns `false`
(`normal_loaders.rs:3231-3275`) for two reasons that remain true: (a)
`cache_write_and_gather` returns a varlen pack `[1, H, sum(seqlen), D]` that
`dsv4_attention` would read as one sequence (hence `v4_paged_dispatch_precheck`
refusing `bs > 1`); (b) the `xs` history lives in extra `NormalCache` slots that
the engine's PagedAttention arm never clones per sequence.

**The interesting part: for V4 the quality cost is ZERO, not "unmeasured".**
`act_quant_kv_nope` (`deepseek4.rs:2168-2210`) **already** round-trips K's 448
nope dims through E4M3 with 64-wide blocks and an fp32 scale, on every layer of
every forward, and then stores the result back in BF16. Arc is already paying
the quantization error and then paying full BF16 to store it. Storing the E4M3
bytes and the 7 block scales directly is **bit-exact with what is stored
today** — it is a serialization change, not a precision change. (The
`ARC_GPU_ACT_QUANT=1` variant is an approximation of E4M3, so the exactness
claim holds for the default CPU-roundtrip path; the GPU path would need its
own check.)

Target layout, mirroring the reference:
`448 (E4M3) + 64*2 (rope BF16) + 7 (UE8M0 scales) + 1 (pad) + 2 (marker) = 586`
B/token/layer — i.e. parity with the reference's 584, and
`43*586 + 20,224 = 45,422` B/token ⇒ ~93 MB/seq at 2048 ctx ⇒ **B ≈ 699**.

**What breaks:** not the attention kernels' semantics —
sliding-window + `attn_sink` + CSA/HCA folding all live in `dsv4_attention`,
which consumes an ordinary `[B,1,T,D]` tensor. Not the RoPE order either: RoPE
is applied *before* `act_quant_kv_nope` and only to the last 64 dims, which the
layout keeps in BF16 exactly as the reference does. What breaks is **the storage
type**: `SingleCache` holds one dtype for the whole row, so a 448-E4M3 +
64-BF16 + 7-U8 row needs either a byte-packed `SingleCache` (the reference packs
it into one `uint8` row of 584) plus a dequant on read, or a second cache half.
**Scoped, not started** — this is exactly the "requires touching the storage
contract" case the brief said to plan rather than half-migrate.

## 6. Also scoped, not started: cap the stored raw KV at the window

The strongest remaining win, and the one the reference already takes.
`dsv4_attention`'s raw branch can only ever reach the trailing
`window + t_q - 1` keys (§3.ii, now proven by `raw_keep_span_is_exactly_the_
reachable_set`). Distant context is served by the *compressed* branch, which is
rebuilt from `XsRollingCache`, not from the raw KV. So the raw KV cache does not
need to be `O(T)` at all — it needs to be `O(window)`.

The reference does exactly this: `pool_configurator.py:397` charges raw KV at
`swa_full_tokens_ratio * kv_bytes * num_layers_total`, with separate 1/4 and
1/128 pools for the CSA and HCA layers.

**What blocks it:** `dsv4_attention` derives the absolute query position as
`q0 = t_k_full - t_q`, i.e. from the cache length. Cap the cache and `q0`
becomes wrong, which silently corrupts both the sliding-window mask and the
compressed-block causality threshold. The fix is to thread the true position
(`seqlen_offsets`, already in `Attention::forward`) into `dsv4_attention` rather
than infer it — a signature change to the module and every one of its 15 tests.
Worth doing; it turns V4's per-sequence raw KV from 132 MB at 2048 ctx (and
growing) into a fixed ~11 MB.

## 7. Surfaced, not shipped (for BACKLOG)

- **The Lightning Indexer is loaded and never called.** `V4Indexer` is
  constructed per CSA layer (`deepseek4.rs:1131-1143`, stored at `:917`) and has
  **zero call sites** in the forward path. `dsv4_attention`'s module docs say so
  ("The Indexer + sparse-gather kernel is a long-context **speed** layer to add
  on top — not required for correctness"), but the weights are loaded and the
  module is fully tested. Wired-but-dead, instance #10-ish.
- The indexer scores against **compressed** keys (`T_c = T_full / ratio`,
  `dsv4_indexer.rs:249-270`), not raw tokens. That is load-bearing for §6:
  capping the raw KV does **not** foreclose the sparse long-context path.
- `mistralrs-core/src/kv_cache/mod.rs:263` — `set_turboquant_head_dim` is
  `#[allow(dead_code)]` "not yet wired into the active path". TurboQuant KV is
  unreachable for V4 anyway (`head_dim = 512`, kernels want 128).

## 8. Tests

New:
- `deepseek4::kv_footprint_tests::v4_kv_bytes_per_token_per_layer` — drives 300
  appends through the real `append_kv_mqa` on a slot built as
  `NormalCache::new` builds V4's, and asserts K half = 1024 B, V half = 2 B,
  total 1026, whole-model 44,118, and that the two halves stay length- and
  capacity-synchronised (the invariant `clone_in_cache` relies on).
- `append_kv_mqa_returns_k_for_both_sides` — the returned tensor is K's
  geometry and K's contents, and the marker is 1-wide zeros.
- `non_normal_kv_slot_is_refused`.
- `dsv4_attention::tests::raw_keep_span_is_exactly_the_reachable_set` — brute
  forces the reachable set from the per-row window rule and compares.
- `dsv4_attention::tests::keys_outside_the_retained_span_cannot_influence_decode`
  — perturbing any dropped key is bit-identical; perturbing the oldest RETAINED
  key moves the output (so the span is not over-wide).

**D12 — fixture discrimination, asserted in the tests themselves:**
- the byte test asserts `HEAD_DIM != V4_V_MARKER_WIDTH`, because if the marker
  were `head_dim`-wide a duplicate-V implementation and the marker
  implementation would measure identically and the test could not tell them
  apart;
- the span sweep asserts `narrowed_at_least_once`, because the sweep would
  otherwise pass while `raw_keep_span` was the identity on every case visited.

**Mutation-proved (both run, both observed):**
- reverting `append_kv_mqa` to the duplicate-V layout (`let marker = k.copy()?`)
  ⇒ `v4_kv_bytes_per_token_per_layer` FAILS with `left: 1024, right: 2`
  (i.e. the old 2048 B/token/layer) and
  `append_kv_mqa_returns_k_for_both_sides` FAILS on the marker dims;
- off-by-one in `raw_keep_span` (`keep - 1`) ⇒ **8 of 15** `dsv4_attention`
  tests fail, including the two pre-existing decode tests
  (`standard_decode_window_boundary_exact`,
  `union_decode_matches_scalar_reference`).

Green: `cargo check -p mistralrs-core`; `cargo test -p mistralrs-core`
(278 + 12 + 2 pass, 0 fail); scoped clippy lane clean.
`rustfmt` applied to the two touched files only, and the pre-existing
(unrelated) reformat churn in `deepseek4.rs` was reverse-applied so the diff is
changes only — verified by diffing the residual fmt delta against the
pre-existing one.
