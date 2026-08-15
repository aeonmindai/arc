# Wave 29 — BC: V4 + paged attention. The stated reason is wrong, and the real answer is still no.

**Scope.** V4's attention / KV-cache integration with the MLA paged path:
`models/deepseek4.rs` attention + KV wiring, `normal_loaders.rs:3231`, the MLA
paged helpers. No GPU rented. Base: `origin/master` @ `57bd1ba70`. Starting
point: `wave28-BA-scheduler-batch.md` §1e (four steps, "a real project").
Every claim below is CONFIRMED (code read, `file:line`) or labelled SUSPECTED.

---

## 0. The one-paragraph answer

**NOT FEASIBLE, and the reason on the flag was never the reason.** BA's §1a
accepted the flag's own explanation — head_dim=512 has no case in
`pagedattention.cuh:714`'s `switch (head_size)`. That switch is reached only
from `PagedAttention::forward`, and **V4 never calls it**: the single
`paged_attn.` call site in the whole model is
`cache_write_and_gather` (`deepseek4.rs:1424`), which runs `reshape_and_cache` +
`gather_kv_cache` and never dispatches on head size. head_dim=512 does not bind
on V4's code path at all. The MLA paged path V2/V3 use cannot rescue it either,
for two independent reasons: 448 is not an instantiable `HEAD_DIM_CKV` (it does
not compile), and — the one that actually matters — `flashinfer_mla_decode`
computes **dense causal** attention while **all** of V4's layers are
sliding-window + attention-sink, with CSA/HCA folding a second, compressed key
set into the *same* softmax. No V4 layer computes the function that kernel
computes. So the flag stays `false`; what changes is that the file now says why,
and two silent-corruption paths behind it are now loud.

---

## 1. Geometry — does the MLA paged path accept V4's shape?

### 1a. `concat_and_cache_mla` — YES, it is shape-agnostic

`mistralrs-paged-attn/src/cuda/concat_and_cache_mla_kernel.cu:53-64`:
`kv_lora_rank`, `kpe_head_dim` and `block_size` are all **runtime `int32_t`
arguments**; the kernel is a strided copy loop
(`for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x)`, `:39-42`) with
`dim3 block(std::min(max_dim, 512))`. 448 + 64 would work today. The Rust
wrapper (`backend/mla.rs:10-225`) checks only dtype agreement.

### 1b. `flashinfer_mla_decode` — NO, and it is a compile-time no

- `mistralrs-paged-attn/src/cuda/flashinfer_mla_decode.cu:12-13`:
  `constexpr uint32_t HEAD_DIM_CKV = 512; constexpr uint32_t HEAD_DIM_KPE = 64;`
  — passed as **template arguments** to
  `BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE, ...>`
  (`:47-50`). Only that one instantiation exists in the binary.
- `backend/mla.rs:263-265` mirrors it on the Rust side:
  `if head_dim_ckv != 512 || head_dim_kpe != 64 { bail!("flashinfer_mla_decode
  is compiled for head dims 512/64, ...") }`.
- **Adding a 448 instantiation does not compile.**
  `flashinfer/attention/decode.cuh:1107-1108`:
  ```
  constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
  constexpr uint32_t bdx          = HEAD_DIM_CKV / vec_size_ckv;
  ```
  At bf16, `HEAD_DIM_CKV = 448` gives `vec_size_ckv = max(8, 14) = 14`, and
  `vec_t<nv_bfloat16, 14>` trips
  `static_assert(vec_size % 8 == 0, "Invalid vector size")` —
  `flashinfer/vec_dtypes.cuh:1566` (bf16) and `:1362` (half). The generic
  vector type is `int4 data[vec_size / 8]`; 14 is not expressible.
  **BA §1e step 2 said this "needs a GPU" to answer. It does not — it is a
  `static_assert` in a vendored header, readable on a laptop.** 512 works
  (`vec_size_ckv = 16`, `bdx = 32`, `vec_size_kpe = 64/32 = 2`); the next legal
  width below it is 256, not 448.
- **Padding 448 → 512 is dimensionally legal** but pointless once §2 lands, and
  it would also be *wrong* for V4's value side: the MLA kernel's output is
  `HEAD_DIM_CKV`-wide (`decode.cuh:1082`) and uses `ckv` as V (`:880`), so the
  k_pe dims contribute to scores but never to the output. V4's V **is** the
  full 512 including the RoPE'd tail (`deepseek4.rs:1357-1364`, `let v =
  k.copy()`), and the model then *inverse*-RoPEs those tail dims out of the
  attention output (`:1544-1548`). An MLA-shaped 448-wide result has no tail to
  de-rotate.

### 1c. V4's MQA layout is not the blocker people assumed

`deepseek4.rs:12-20` (module docs) is explicit: **"V4 is not an MLA variant"**.
There is no `kv_lora_rank`, no `kv_b_proj`, no `kv_a_proj_with_mqa` — one fused
`wkv` (`hidden=4096 → head_dim=512`) produces a single MQA head broadcast over
64 Q heads, RoPE applied in place to the last 64 dims, and `V == K`. That is
actually *easier* than MLA: V2/V3 need `mla/forward.rs:190-202`'s
`w_uk`/`w_uv_t` absorption to get into and out of latent space; V4's weights are
already absorbed, so the two `broadcast_matmul`s around the kernel call would
simply drop out. **Geometry was never the hard part.**

---

## 2. The kernel computes the wrong function — this is the real blocker

`flashinfer_mla_decode.cu:33` fixes the attention variant:
`using AttentionVariant = DefaultAttention<false, false, false, false>;`
and `flashinfer/attention/variants.cuh:31-32` names those parameters
`<use_custom_mask, use_sliding_window, use_logits_soft_cap, use_alibi>`. So the
compiled kernel is: dense causal, **no sliding window, no custom mask, no
attention sinks, one key set**.

V4 has no layer that wants that:

- `models/dsv4_attention.rs:9-46` (module docs, ported from the reference
  `Attention.forward`): V4 attention is **"a single online softmax over a union
  of key sets"**. Standard (`compress_ratio == 0`, the ratio-0 layers) is
  **sliding-window + `attn_sink`, "NOT dense causal!"** — running those layers
  dense-causal is the long-context repetition collapse RUN-161 fixed. CSA
  (ratio 4) and HCA (ratio 128) add a **compressed** key set to the *same*
  softmax.
- `dsv4_attention.rs:281-322` builds that union mask, and
  `deepseek4.rs:1432-1441` shows **every** dispatch arm — paged, graph, plain —
  routing through it.

There is no per-layer escape hatch. Enabling `flashinfer_mla_decode` for V4
would require the sliding-window variant, sink columns in the denominator, and
a second key set with its own strided-position RoPE — i.e. a new kernel, not a
wiring change. **Stopping here, per the brief.**

---

## 3. So why is the flag actually false? — corrected on the branch

`normal_loaders.rs:3231` said:

> V4 MLA uses head_dim=512, which exceeds the PagedAttention kernel's supported
> head sizes (64/80/96/112/128/192/256) ... (RUN-161)

**Inapplicable.** That switch lives in `pagedattention.cuh:714`, reached from
`paged_attention(...)` at `paged_attention/layers/paged_attention.rs:341`,
inside `PagedAttention::forward` (`:71`). `grep -n "paged_attn\." deepseek4.rs`
returns **exactly one line — `:1424`, `cache_write_and_gather`**. V4 uses
PagedAttention as block-allocated KV **storage** and then runs its own
`dsv4_attention` over the gathered result. `PagedAttention::forward` is never
called, so its head-size table never applies.

The comment is replaced on this branch with the three findings (§1, §2, §4) so
the next agent does not re-derive them. **The value is not `false` any more —
the value was always `false`; the *reason* is.**

### What the flag actually gates, and what flipping it would unlock

`pipeline/normal.rs:344-346` — `if !supports_paged_attention { paged_attn_config
= None; }` — is the single gate. Of BA §5's three dead capabilities:

- **Ragged batching: no.** Paged KV would remove the length-bucketing
  *requirement*, but see §4a — the V4 gather is single-sequence.
- **CUDA-graph autonomous decode: no, not from this flag alone.**
  `normal.rs:1855-1862` bails on `cache_config == None`, but `:1771-1779` is
  explicit above it: *"this method does NOT yet prime per-step paged attention
  metadata into the runner ... the runner intentionally never finishes a real
  decode batch and the engine continues to fall back to `step()`"*. The
  `prime_for_step` bridge from the engine's `PagedAttentionMeta` does not exist.
  Paged attention is **necessary but not sufficient**; the brief's framing that
  the flag alone unlocks graph decode is one gate short.
- **Fused GPU sampler: no** — it is reachable only through that same runner
  (`arc-cuda-graph/src/sampling_cuda.rs`).

---

## 4. The two things that would actually break if the flag were flipped

Both are silent. Both are now loud on this branch.

### 4a. The gather returns a varlen *pack*, not a batch

`paged_attention.rs:508-509`:
```rust
let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
```
`gather_kv_cache` returns `[N_total, H, D]` with `N_total = sum_i(seqlen_i)`;
the bare `unsqueeze(0)` makes that `[1, H, N_total, D]`. The function's own doc
(`:380-388`) admits it: *"The current V4 deployment target (single-stream rental
serving) keeps batch_size == 1 ... Multi-tenant batching with CSA/HCA needs a
per-seq slice + dispatch loop (audit §8 P0 item 9, deferred)."*

`dsv4_attention` reads `t_k` as **one** sequence's cached length and derives
`q0 = t_k - t_q`, the sliding-window band and the compressed-block causality
from it (`dsv4_attention.rs:270-294`). Hand it the pack and every query attends
over the concatenation of all sequences' keys, at wrong absolute positions.

**Added:** `v4_paged_dispatch_precheck(bs)` in `deepseek4.rs`, called at
`:1424` before the gather, refusing `bs > 1` by name.

### 4b. The `xs` compressor history is NOT paged and cannot be

V4 keeps the compressor-input history in **extra `NormalCache` slots past the KV
entries** (`deepseek4.rs:3331-3356`), one per compressor layer, `[B, T, hidden]`
on seq dim 1. Its own comment states the mechanism it depends on:

> *the engine's NormalCacheManager iterates `metadata.num_hidden_layers`, which
> pipelines derive from the CACHE length — so these extra entries are cloned
> in/out per sequence exactly like KV*

That derivation is real (`normal.rs:1214-1218`: `num_hidden_layers =
normal.lock().0.len()`), and so is the dependency. **The engine's PagedAttention
arm never issues `CacheInstruction::In`/`Out`** — compare `engine/mod.rs:393-400`
(DefaultScheduler arm, In/Out) with `:556+` (`SchedulerOutput::PagedAttention`,
neither). Under paging, `NormalCacheManager::clone_in_cache` never runs, and the
per-sequence `xs` history collapses into **one buffer shared by every
sequence** — R3's exact regression, reintroduced, silently.

**Answer to "paged, unpaged, or separate?": UNPAGED, and it must stay
alongside — as its own project.** It cannot ride the KV block tables: the KV
cache is `[num_blocks, num_kv_heads, head_size/x, block_size, x]` keyed by a
slot mapping over KV rows, and `xs` is a `hidden`-wide row per token per
*compressor* layer, a different width on a different layer set. Paging it means
a second block table, a second slot mapping and a second allocator.

**And it is the binding constraint on batch, not the KV.** Per token per layer:
`xs` is `4096 × 2 B = 8,192 B`; K+V is `512 × 2 B × 2 = 2,048 B` — exactly
**4.0× per layer**, and 3.8× in total (335,954 vs 88,064 B/token) only because
fewer layers carry a compressor than carry KV. **Paging the KV and not the `xs`
history buys back the smaller fifth of the footprint.** Any credible route to
B=128–256 has to page `xs` — or stop materialising it, which is the better
question: the compressor consumes the history only through
`forward_from_xs` → `wkv_gate(x)` over the largest `ratio`-multiple prefix
(`deepseek4.rs:703-757`), and a compressed rolling state would be `ratio`× (4×
or 128×) smaller than the raw `xs` it is recomputed from every step.

---

## 5. What changed

1. `mistralrs-core/src/kv_cache/mod.rs` — `first_mismatched_cache_len()` +
   a `debug_assert!` at the top of `NormalCacheManager::clone_in_cache`
   (BA §7 item 1). Debug-only: the scan walks every sequence's whole cache
   vector and the scheduler is what enforces the invariant.
2. `mistralrs-core/src/models/deepseek4.rs` — `v4_paged_dispatch_precheck(bs)`,
   called at the PagedAttention arm (`:1424`).
3. `mistralrs-core/src/pipeline/loaders/normal_loaders.rs:3231` — value
   unchanged (`Ok(false)`), rationale replaced with §1/§2/§4.
4. Tests (below).

Deliberately **not** done: the flag is not flipped, no kernel is added, no
`mistralrs-quant/qtip/**` or `scheduler/**` or model-card file is touched.

---

## 6. The tests, and the proof each can fail (D12)

Six tests, three mutations, each run and each observed to fail.

`kv_cache::clone_in_cache_invariant_tests`
- `uniform_cache_lens_are_accepted`
- `mismatched_cache_lens_are_detected`
- `mismatch_is_found_on_any_layer_and_single_seqs_pass`
- `clone_in_cache_refuses_a_length_mismatched_batch` — **drives the real
  `NormalCacheManager::clone_in_cache`** through a `StubPipeline`
  (`CacheManagerMixin + MetadataMixin`), not a copy of the condition.
  `#[cfg(debug_assertions)]`-gated, since `debug_assert!` compiles out of
  release test builds.

> **Mutation 1** — delete the `debug_assert!` from `clone_in_cache`:
> ```
> test kv_cache::...::clone_in_cache_refuses_a_length_mismatched_batch - should panic ... FAILED
> panicked at mistralrs-core/src/kv_cache/mod.rs:393:45:
> called `Option::unwrap()` on a `None` value
> note: panic did not contain expected string
>       panic message: "called `Option::unwrap()` on a `None` value"
>  expected substring: "must share current_seq_len"
> ```
> That failure *is* the argument for the assert: without it the batch still
> dies, 80 lines later, saying nothing about sequence lengths. Reverted; passes.

`models::deepseek4::tests`
- `v4_paged_varlen_pack_leaks_keys_across_sequences` — builds the exact tensor
  the gather returns (two sequences' K concatenated on the seq axis, B still 1)
  and shows sequence 0's decode output **changes**. Also asserts the honestly
  batched form (`q` at B=2 vs the B=1 pack) does not survive: today it panics in
  `attention/backends/cpu.rs:485` with `range end index 56 out of range for
  slice of length 48` — which is why a named guard beats the status quo.
- `v4_paged_dispatch_precheck_refuses_multi_sequence_batches` — the guard as
  installed; asserts the refusal names the offending batch size.

> **Mutation 2** — `if false && bs > 1` in `v4_paged_dispatch_precheck`:
> ```
> test models::deepseek4::tests::v4_paged_dispatch_precheck_refuses_multi_sequence_batches ... FAILED
> bs>1 must be refused: cache_write_and_gather packs, it does not batch: ()
> ```
> **Mutation 3** — leak test attends over `k0` instead of the pack:
> ```
> test models::deepseek4::tests::v4_paged_varlen_pack_leaks_keys_across_sequences ... FAILED
> the two-sequence pack reproduced sequence 0's own answer exactly (max_diff=0)
> ```
> Both reverted; all six pass.

**Honest limit.** These prove the guards, not that a paged V4 path is taken —
because there is no paged V4 path to take, and a test asserting on a path that
does not exist is exactly the vacuity D12 is about. The existing RUN-167 tests
at `deepseek4.rs:4941+` already say so in their own header: *"The PagedAttention
dispatch arm itself is CUDA-only at runtime ... so we cannot exercise it on a
CPU build."* They test the post-gather tensor, not the dispatch — BACKLOG
"wired but dead" entry nine stands, and this branch does not clear it.

---

## 7. What needs a GPU

**Nothing, for anything in this document.** The three questions BA deferred to
hardware were all answerable from the tree:

1. *"Does `flashinfer_mla_decode` accept `kv_lora_rank=448`? Not answerable
   from the Rust side."* — Answerable, and no: `vec_dtypes.cuh:1362,1566`.
   That is a ~15-minute rental not spent.
2. Whether the head-size switch binds — no: V4 never calls the kernel that
   has one.
3. Whether the MLA path could serve V4 — no, for reasons in the vendored
   headers and in V4's own module docs.

The *only* GPU item this branch creates: the `debug_assert!` runs on every
decode step of every batched request in a debug build. If a debug rental ever
shows it in a profile, gate it behind an env var. Release builds compile it out.

---

## 8. Surfaced, not shipped

> **Noticed:** BA §1e's four-step plan ("port V4's attention onto
> `mla_decode_forward`") is not implementable and should be struck from the
> backlog before someone budgets a session for it. The replacement item, if
> anyone wants paged V4 decode, is *"a V4 decode kernel with sliding window +
> sinks + a second key set"* — a kernel project, sized like the trellis
> grouped-GEMM, not a wiring project. Worth a separate change?

> **Noticed:** the `xs` compressor history is recomputed from a raw
> `[B, T, hidden]` buffer every decode step, at 4× the per-layer cost of the KV
> it sits beside, when the compressor only ever consumes a `ratio`-strided
> reduction of it (`deepseek4.rs:703-757`). A rolling compressed state would cut
> the dominant per-token memory term by 4–128× and is independent of paging.
> That, not paged KV, is the batch-size lever. Worth a separate change?

> **Noticed:** `cache_write_and_gather` gathers the **entire** context every
> layer, every decode step, precisely to undo paging before handing a dense
> tensor to `dsv4_attention`. Under paged attention V4 would pay block
> allocation *and* a full O(context) gather per layer per token. Nothing
> measures that; the flag being false has hidden it. Worth a separate change?
