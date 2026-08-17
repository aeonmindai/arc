# wave61-CL — The segment allocator: a KV read primitive that is a list of runs

**Branch:** `feat/segment-kv-allocator` off `d4cf5a2d7` (master). **Draft PR #TBD.**
**Hardware used: NONE.** Everything below is code, arithmetic, or a CPU unit test.
**Nothing here is an acceptance measurement.** See §7 for the exact GPU ask.

---

## 0. The claim, in one line

PagedAttention's read contract is `(block_table, context_len)` — one start, one
length. Three workstreams are blocked on a read that is **a list of runs**. This lands
that primitive, proves the one-run case is byte-identical to what ships today, and puts
it behind `ARC_SEGMENTED_KV`, default OFF.

## 1. 🔑 The load-bearing discovery: the gather kernel already accepts it

I expected this to need a new CUDA kernel. It does not.

`mistralrs-paged-attn/src/cuda/gather_kv_cache_kernel.cu:40` takes:

```c
const int32_t *block_table,  // [batch, max_blocks]
const int32_t *cu_seq_lens,  // [batch + 1]
```

and per output token binary-searches `cu_seq_lens` for its row, then reads
`block_table[row * stride + (token_id - cu[row]) / block_size]`.

**Nothing in that kernel requires a row to be a sequence.** A row is an independent
`(block-table, length)` run gathered into a contiguous output slice. Therefore
**a segment is a row**: `B` sequences × `S` segments flattens to a gather with `B*S`
rows, and the `.cu` file does not change. This is why the whole thing is tractable.

The two constraints this imposes are real and are encoded as invariants
(`Segment::check`): a row must begin on a block boundary (the kernel derives
`slot = (token_id - cu[row]) % block_size`), and a run whose true start is mid-block
must gather the whole block and name the dead prefix — `lead_pad`.

## 2. What was actually blocking V4 — corrected

The brief said V4 reads "a window, plus a compressed summary, **plus an attention
sink**", and that ratio-0 layers are `{0, 1, 42}` + MTP. Both are wrong:

* **The sink is not a KV region.** It is a per-head learned scalar `[1,H,1,1]` F32 that
  enters `softmax_with_sinks` as a **denominator-only column**
  (`models/dsv4_attention.rs:149`, scalar reference `:1008`). Zero cache bytes, zero
  gather, nothing to allocate. V4 reads **two** regions, not three.
* **The ratio-0 set is `{0, 1, 43}`** — layers 0 and 1 plus the MTP slot. **Layer 42 is
  CSA (ratio 4).** Pinned by `yarn_applies_to_compressed_layers_only`
  (`deepseek4.rs:6567`) and stated at `deepseek4.rs:377-382`. Two stale comments in the
  tree still say "0, 1, 42" (`deepseek4.rs:35-36`, `dsv4_attention.rs:12`) — **not fixed
  here**, flagged for a separate change.

The corrected V4 read, per CSA/HCA layer, per call:

| region | extent | anchor | stride | present on |
|---|---|---|---|---|
| **R1** raw window | suffix `[t_k_full - keep, t_k_full)`, `keep = min(t_q+127, t_k_full)` (`raw_keep_span`, `dsv4_attention.rs:272`) | **end** — head slides | 1 row/token | every layer |
| **R2** compressed | prefix `[0, t_c)`, read in full, never windowed (`:419-434`) | **start** — grows at tail | 1 row per `ratio` tokens | ratio 4 (21 layers), 128 (20 layers) |

Two contiguous runs, opposite anchors, different strides. That is exactly a 2-segment
table — and it is why the one-run contract cannot express it: one run cannot be anchored
at both ends at two strides.

### The payoff is a deletion, not an addition

Today `dsv4_attention.rs:431-432` does `Tensor::cat(&[k, comp], 2)` — physically
materialising a fresh `[B, 1, keep + t_c, 512]` buffer **twice** (K and V) per layer per
step, to fake a single run. A segment plan emits the two runs adjacent in one gather
output: same bytes, no copy. Making the read primitive segment-shaped **removes** that
`cat` rather than adding work.

## 3. The in-tree precedent this replaces

`PagedAttentionInputMetadata` (`pipeline/inputs_processor.rs:92`) already carries a
hardcoded two-table special case: `block_tables`/`context_lens` (windowed) **and**
`full_block_tables`/`full_context_lens` (unwindowed), so a per-layer sliding-window model
picks one *or* the other. Two tables, chosen between — never summed. This is precisely the
"per-model special case at the allocator layer" that vLLM and SGLang carry. A segment list
subsumes it: pick-one-of-two is the 1-segment case of read-all-of-N.

## 4. What was built

| piece | file | what it is |
|---|---|---|
| `RegionKind`, `Segment`, `SegmentTable`, `SegmentPlan`, `flatten` | `paged_attention/segment.rs` (new) | the data structure + the kernel-ready flattening |
| `SegmentedAllocator` | `paged_attention/segmented_allocator.rs` (new) | allocation/advance/rollback/slide/free over a caller-owned `BlockPool` |
| `Tracking::{Flat, Segmented}` + `ARC_SEGMENTED_KV` | `paged_attention/kv_cache_manager.rs` | the flag and the routing |

**No new memory owner.** `SegmentedAllocator` takes `&mut BlockPool` in every method, so
one pool and one global free list still back every region of every sequence.

### Group ids: namespaced by *stored value*, not by read pattern

`RegionKind::group_id()` feeds the existing `BlockHashWithGroupId`. The rule is one line:
**group id is the compression ratio; 0 means uncompressed.**

* `Dense`, `Window`, `Sink` → **0**, deliberately shared. All three store raw K/V,
  block-aligned in the same absolute frame, so a block written by one is bit-identical to
  a block written by another — a sliding window *should* reuse a dense prefix-cache hit.
* `Compressed { ratio }` → `ratio`. CSA (4) and HCA (128) get distinct groups because
  their stored bytes differ. `RegionKind::compressed()` refuses `ratio < 2` so a
  compressed region can never collide with the raw group.

### Composition with `kv_sharing/`, not duplication of it

No second tree was built. `kv_sharing::RadixTree` is generic over its symbol and already
has a `PagedTree<P> = RadixTree<BlockHash, P>` substrate. The segment allocator's sharing
unit is expressed in the terms that tree already consumes — a `BlockHash` chain plus a
group id — so a `PagedTree` keyed per region kind drops in without touching either side.
`SegmentedAllocator::region_key` is that adapter.

Byte accounting follows `kv_sharing/layout.rs`'s rule exactly: **width is a property of
the block, never a global**. Every `Segment` carries its own `KvBlockLayout`; `bytes()`
sums per segment. Pinned by `a_v4_shaped_table_is_billed_per_region` — a raw+HCA table
billed at one global width is wrong by the compression ratio.

### The two capabilities the one-run contract cannot express

1. **Per-sequence, per-region advance.** `advance`/`rollback` name a
   `(request_id, segment_index)` pair and read no other request. This is the allocator-level
   precondition for wave59-CJ §3's MTP ceiling, where one shared dense cache forces
   `next_u = u + a + 1 − min_i(u_i + a_i)`, and one sequence rejecting its first draft
   ratchets every other sequence's tail to the window.
   **It is a precondition, NOT the fix** — `mtp_pipeline.rs` still rolls one shared cache.
   Pinned by `one_sequences_rollback_does_not_touch_another_sequences_advance` (5 seqs, 7
   steps, one permanent laggard: the other four advance `16 → 44` regardless).
2. **Head release.** `slide` returns blocks a window has left behind so they go back on
   the free list *while the sequence still runs*. The one-run contract cannot: its block
   table must stay anchored at token 0 for `block_table[offset / block_size]` to address,
   so every block ever written stays charged. Pinned by
   `a_sliding_window_returns_its_head_blocks_to_the_pool` — 40 decode steps on a 16-token
   window converge to ≤5 blocks and free-list accounting closes exactly.

## 5. No-regression: the strongest available form

Not "close enough" — **the same bytes**, and **the same tests**.

1. `degenerate_batch_flattens_to_the_legacy_block_table_and_cu_seqlens` — a batch of
   one-`Dense`-segment tables flattens to exactly what `get_block_table` (ids, zero-padded
   to `max_blocks`) and `build_cu_seqlens_kv_from_context_lens` (prepend 0, cumsum) build.
2. `degenerate_path_matches_the_legacy_manager_step_for_step` — the legacy
   `KVCacheManager` and the segmented allocator driven through the *same* trace against
   equal pools: block tables, slot mappings and free-block counts agree at every step,
   **including across a prefix-cache hit**, where touch/evict accounting is easiest to get
   subtly wrong.
3. **The pre-existing `KVCacheManager` suite passes unmodified against the segmented
   backing.** `ARC_SEGMENTED_KV=1 cargo test -p mistralrs-core --lib` → **377 passed, 0
   failed**, identical to the flag-off run. Fourteen of those are the original
   `kv_cache_manager::tests`, not rewritten.

### The benchmark

⚠️ **HOST-SIDE BOOKKEEPING ONLY. NOT A VALIDATION. NOT A GPU RESULT (D14).**
`segmented_backing_allocator_overhead`, `--ignored`, release, darwin/arm64, 64 seqs ×
256 decode steps × 512-token prompts:

```
flat         1384875 ns total,  84 ns/step-seq
segmented    1381042 ns total,  84 ns/step-seq
ratio        1.00x
```

This number's only job is to catch a *complexity* regression (a scan where there was a
map, an alloc per decode step). It says nothing about throughput and must never be quoted
as evidence the segmented path works. The assertion is deliberately loose (`< 4x`) so it
cannot become a CI flake.

### Mutation runs — no test here is vacuous

```
group_id: Compressed{ratio} => ratio  ⟶  => 0
  regions_do_not_alias_in_the_prefix_cache ......................... FAILED
slide: drain the dead blocks but return none
  a_sliding_window_returns_its_head_blocks_to_the_pool ............. FAILED
  left: 19  right: 59   (blocks leaked)
allocate_dense: drop num_evictable accounting + the touch
  degenerate_path_matches_the_legacy_manager_step_for_step ......... FAILED
  left: 25  right: 30   ("touch/evict accounting must match after a prefix hit")
```

## 6. What is NOT done, deliberately

* **`supports_paged_attention` for V4 is still `false`, and I did not touch it.** The
  allocator is necessary and not sufficient. Flipping it now would repeat the FP8-KV
  merge: a default-on path nobody ran. The two remaining blockers are unchanged from
  `normal_loaders.rs:3231` §3 — (a) `dsv4_attention` must consume per-segment boundaries
  instead of reading the varlen pack as one sequence (`v4_paged_dispatch_precheck` refuses
  `bs > 1` rather than corrupt it), and (b) the compressor's `xs` history has no block
  table.
* **`XsRollingCache.tail` is explicitly out of scope.** It is `hidden`-wide (4096), on a
  different layer set, bounded by `span_groups*ratio + margin` tokens rather than by
  context. A third storage class, not a KV region; it cannot ride these block tables.
  wave29-BC §4b already ruled it "unpaged, and its own project" — that still holds.
* **The MTP ceiling is not lifted.** Per-sequence tables are the precondition;
  `plan_batch_step` still advances one shared cache by `min_i(u_i + a_i)`.
* **The multi-region API has no production caller.** `open_region`, `advance`, `slide`,
  `region_key` are carried under an explicit `#[allow(dead_code)]` with a comment saying
  so. They are tested, not live. `ARC_SEGMENTED_KV=1` today drives only the degenerate
  path — which is the point: it makes the new allocator measurable on a model that
  already works, before anything depends on the rest.

## 7. 🔴 The GPU ask

Everything above is CPU. Under D14 **none of it is validated.** One box, one model that
already works today (any dense Llama-class), two runs:

```bash
# A: control
mistralrs bench -m <dense-model> --pa-memory-mb <N> 2>&1 | tee /tmp/flat.txt
# B: same box, same model, same flags, segmented backing
ARC_SEGMENTED_KV=1 mistralrs bench -m <dense-model> --pa-memory-mb <N> 2>&1 | tee /tmp/seg.txt
```

**The one number:** decode tok/s in B as a fraction of A. The claim under test is
`B/A ≥ 0.99` and identical generated text. If B ≠ A on output, the degenerate path is not
degenerate and this design is wrong at the root — which is a valid and cheap outcome to
discover, because it needs no V4, no kernel work and no new model.

## 8. Surfaced, not shipped

1. **Two stale in-tree comments assert the wrong ratio-0 layer set** (`deepseek4.rs:35-36`,
   `dsv4_attention.rs:12` both say "layers 0/1/42"); the config and the test say `{0,1,43}`
   and that layer 42 is CSA. Worth a one-line correction — not made here to keep this diff
   to the allocator.
2. **`get_computed_blocks` carries a `debug_assert!` that multi-group lookup is
   unimplemented** (`kv_cache_manager.rs:264`). Multi-region prefix-cache hits need it, and
   the group ids now exist to make it meaningful. Own change.
3. **`KVCacheManager` is never told the model's KV geometry**, so `set_kv_layout` defaults
   to the neutral 1-bit layout and `allocated_bytes()` is not a real byte count until a
   caller supplies one. Deliberate — inventing a width would make the number wrong
   invisibly — but it means byte reporting is inert until wired.
