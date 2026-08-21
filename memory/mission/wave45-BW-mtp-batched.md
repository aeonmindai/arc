# wave45-BW — MTP at every batch size, and the two forwards that halved it

Branch `feat/mtp-batched`, stacked on `feat/mtp-working` (PR #71). Scope fence
respected: no `mistralrs-quant/src/qtip/**`, no KV/`xs` byte-layout, no
`device_map.rs` / `distributed/**`.

---

## 1. Why `take_fast_path` was B=1-gated — REAL, and about the cache, not MTP

`mtp_pipeline.rs`, old gate: `input_seqs.len() == 1`. The comment blamed "no
batched draft path in Tier A". That is not the constraint. The constraint is
one level down:

* `kv_cache/single_cache.rs:161` — `SingleCache::append` does
  `ad.slice_set(src, self.dim, self.current_seq_len)`. **One offset for the
  whole batch.**
* `kv_cache/mod.rs:390-604` — `clone_in_cache` builds ONE dense batched cache
  and takes `seqs[0]` as the template for `current_seq_len`.
* `kv_cache/mod.rs:353` — `first_mismatched_cache_len` exists precisely to
  catch two sequences at different lengths in one batch, and says the failure is
  silent (`CACHE_GROW_SIZE` is 512, so the shapes still match).
* `scheduler/default_scheduler.rs:115-157` — exactly **one** length-bucket runs
  per step.

So: speculative decode makes per-sequence accept lengths diverge; a dense
shared-offset cache cannot represent per-sequence lengths; and if you let token
lengths diverge, the scheduler shatters the cohort into one bucket per accept
length and idles all but one. **Genuine constraint, not laziness — but a
constraint of the dense cache, which is code, not physics.**

## 2. The fix: fused single-forward + a fixed window with an uncached tail

Two changes, and they are inseparable.

**(a) Fusion.** The old shape ran **TWO** target forwards per step: one 1-token
decode to materialise `T0` and its hidden state, then the verify forward. That
caps the multiplier at `(depth+1)/2` — **at depth 1 that is exactly 1.0, i.e.
`--mtp-depth 1` could never have been faster than no MTP at all.** The seed the
next chain needs (`h_{L-2}`) is always inside the window the last verify already
covered — on a rejection at slot `j` the committed length is `C+j+1` so
`L-2 = C+j-1`; on a full accept it is the window's last input. Carrying it on
`DraftKv::seed` removes the first forward outright. SGLang reaches the same
place from the other side, skipping the *last* draft forward because the
draft-extend it must run anyway already produced that token
(`eagle_worker.py:871-873`).

**(b) The window.** Cache length `C` stays **uniform** across the batch; token
lengths diverge instead. Sequence `i` carries an uncached committed tail
`u_i ∈ [1, w]`, `w = depth + 1`, and feeds a fixed-width window of `u_i` real
tokens then `w - u_i` drafts. After verify the shared cache rolls back to
`C + min_i(u_i + accepted_i)` — the one length every sequence can prove — and
each sequence's surplus stays committed as *tokens*, returning next step as a
longer tail, which is exactly what `w - u` drafts leaves room for.

The invariant closes: `u ≥ 1` ⇒ the cache advances every step (no stall), and
`u_i + a_i ≤ w` ⇒ `u' ≤ w + 1 - min ≤ w` (a tail can never outgrow the window).
Pinned by `batched_window_invariant_closes_at_every_depth_and_accept_pattern`
over >5,000 (depth, accept-pattern) combinations.

**(c) Scheduler.** `BucketKey` now keys on `Sequence::cache_bucket_len()` (the
cache length) rather than `len()`. For every non-speculative path this is the
*identical partition* — `cache_len ≡ len - 1` — so nothing else moves. It is
what keeps a batched MTP cohort whole.

**(d) Batched drafting.** Within a group of equal `u`, every sequence sits at the
same absolute chain start `C + u - 2` and needs the same `w - u` drafts, so one
`[G, 1, hidden]` MTP-block forward per chain step drafts for all of them. `u`
ranges over `[1, w]`, so a batch of any size splits into at most `depth + 1`
groups. Per-sequence draft KVs are concatenated into one batched cache
(`batch_draft_caches`) and split back (`split_draft_cache`) — the MTP block is a
single decoder layer, so this is `clone_in_cache` at ~1/43 of its cost.

## 3. PagedAttention exclusivity — **unimplemented branch, and moot for V4**

Not a genuine incompatibility. One thing is missing: the rollback.
`truncate_cache_by` walks a `NormalCache` and calls `set_len` per layer; under
paging the K/V lives in the block table the PagedAttention KV manager owns, the
`EitherCache::Normal` entries are unused, and that truncate would be a **silent
no-op** leaving a rejected draft's K/V addressable. Refusing is correct until
the paged free-list step exists.

The reference proves it is the cheap case. SGLang frees exactly the complement
of its accept set (`eagle_info.py:488-490`), and at **topk == 1** — a linear
chain, which is what DeepSeek MTP *is* in SGLang's shipped config
(`server_args.py:7611-7627`: `(3, 1, 4)` for every DeepSeek arch) — the accepted
tokens are a contiguous prefix of the allocated run, so rollback is **pure
truncation, no KV movement** (`eagle_info.py:492-501`). Only tree drafting needs
the `move_kv_cache` compaction (`:505-545`). Draft-phase slots are never leaked
at all: they are allocated against a backed-up allocator state that is
unconditionally restored (`eagle_worker.py:613-617` → `:727`,
`allocator.py:71-75`).

Further — and this inverts the framing in `CEILINGS.json` — **paged KV is a
*better* substrate for batched MTP than the dense cache**, because a per-sequence
block table makes per-sequence lengths free and the entire uncached-tail scheme
above unnecessary. That is why vLLM/SGLang can do batched spec decode at all.

**Moot for V4, twice over.** `DeepSeekV4Loader::supports_paged_attention` returns
`false` (`loaders/normal_loaders.rs`, wave29-BC rationale: `flashinfer_mla_decode.cu`
fixes `HEAD_DIM_CKV=512` as a template constant and computes dense causal
attention while every V4 layer is sliding-window + sink), so the engine never
hands V4 a `CacheBackendMetadata::PagedAttention`. And `mtp_decode_kit` has
**exactly one** implementation in the tree (`deepseek4.rs:4250`; the trait
default at `loaders/normal_loaders.rs:104` returns `None`), so the MTP wrapper
cannot currently wrap a model that pages. **The two features have never met and
this guard has never fired.** `--paged-attn off` is still required for V4 MTP —
but for the unrelated reason that `auto` = ON on CUDA would put the engine on a
backend V4 does not support.

## 4. "Best MTP" — ranked, one implemented, the rest scoped with numbers

| rank | item | expected gain | status |
|---|---|---|---|
| 1 | **Fuse the two target forwards** | **×2 on the multiplier at every depth**; depth 1 goes from 1.00 to 2.00 max | **implemented** |
| 2 | **Batched fast path** | MTP applies at B>1 at all — the difference between a single-user lever and a fleet lever | **implemented** |
| 3 | Depth 2→3 | reference accept-length 2.30 (non-simulated) at their `num_steps=3`; ours is `depth`-configurable already, clap-capped at 8 vs `XS_TAIL_MARGIN_TOKENS=16` | **already available**, default still 0 |
| 4 | Tree / multi-candidate drafting | **NOT recommended for V4.** SGLang ships `topk=1` (linear chain) for *every* DeepSeek arch (`server_args.py:7611-7627`); tree is the Llama/Grok default `(5,4,8)`. Tree also forces the `move_kv_cache` compaction on paged rollback and is unsupported by their overlap scheduler | scoped, **declined** |
| 5 | EAGLE-3 | **wrong target for V4.** Needs weights the DeepSeek MTP checkpoint does not ship: `fc` (3H→H aggregation), `fc_norm.{0,1,2}`, `d2t`/`t2d` vocab maps, and its own `lm_head` over `draft_vocab_size` (`llama_eagle3.py:148-173,277-327`); plus a *target-side* 3-layer capture (`deepseek_v2.py:2567-2574`) and a separate SpecForge training run. MTP ships in the base checkpoint | scoped, **declined for V4** |
| 6 | Stochastic-sampling verification (rejection sampling) | today acceptance is greedy-argmax equality; the step's one non-drafted token does go through the real sampler | scoped, backlog |

## 5. What the reference does that we do not

* **Ragged verify as a dense `(B, steps+1)` `-1`-padded `accept_index` matrix**
  plus a `num_correct_drafts` vector, resolved by one CUDA kernel with one block
  per sequence (`eagle_info.py:277-281`, `eagle_utils.cu:290-311`). We resolve
  on the host in a `B`-length loop — fine at our step times, a target later.
* **Per-sequence KV freeing** instead of a batch-minimum rollback
  (`eagle_info.py:488-501`). Ours costs the surplus as a longer tail next step;
  theirs costs nothing because the KV is paged.
* **`backup_state`/`restore_state`** on the allocator so draft KV is borrowed,
  never leaked (`allocator.py:71-75`).
* **Adaptive depth** from an EMA of accept length
  (`docs/advanced_features/adaptive_speculative_decoding.md:91`).
* **A stated acceptance floor in CI.** DeepSeek V4 MTP: `acc_length > 2.30`
  non-simulated (`test/manual/dsv4/test_dsv4_pro_mtp.py:261`), `> 2.85` with
  `SGLANG_SIMULATE_ACC_LEN=3`. V3: `> 2.8` (`test_deepseek_v3_mtp.py:88`).

## 6. Tests

* `batched_window_invariant_closes_at_every_depth_and_accept_pattern` — >5,000
  patterns, depths 1-8: `keep ≥ 1`, `next_uncached ∈ [1, w]`.
* `batched_ragged_accept_is_token_identical_to_the_b1_reference` — five
  sequences with different agreement rates decode together and alone; the
  per-sequence token streams must match exactly. **Asserts its own
  non-degeneracy first** (D12): the run must contain a step with differing
  accept lengths *and* a step where some tail exceeded 1, or the fixture proves
  nothing.
* `batched_rollback_mutation_max_instead_of_min_is_caught` — resolving on the
  MAXIMUM valid extent (the plausible "keep as much cache as possible" mistake)
  drives a tail to 0, which is a sequence whose committed tokens sit in the
  cache as K/V computed from a rejected draft.
* `window_verify_row_off_by_one_destroys_acceptance` — a one-row shift is the
  invisible defect (no crash, no corruption, just collapsed acceptance).
* `batched_acceptance_is_reported_per_batch_size` — all-accept reads 1.0,
  all-reject reads 0.0, per batch size; per-user and aggregate multipliers are
  both on the marker.
* Everything from PR #71 still passes: 296 + 13 + 2 tests green.

## 6b. A latent prefix-cache over-claim the ragged tail exposed (fixed)

`search_for_matching_cache` calls `set_len(match_len)` on the stored layers, and
`SingleCache::try_set_len` checks **capacity, not validity** — it extends a
length into slots that were never written. Plain decode never trips it because
two facts cancel exactly: `cache_len == len - 1`, and a full-length match is
rejected by the `new_toks.is_empty()` guard, so `match_len ≤ len - 1 ==
cache_len`. Safe by arithmetic coincidence, not by design.

Batched MTP breaks the coincidence: a sequence can finish with
`cache_len == len - u`, `u` up to `depth + 1`, while `match_len` can still reach
`len - 1` — a later request sharing that prefix would be served uninitialised
K/V. Fixed twice over: `add_sequence` keys by `toks[..min(len, cache_len + 1)]`
(a **no-op when `u == 1`**, so no existing path changes), and retrieval declines
any match longer than `layer.current_seq_len()`.

## 6c. Batched prefill primed only row 0 — batched MTP would have been dead on arrival

The prompt path seeded the draft KV only when `input_seqs.len() == 1`. In a real
serve that is the common case *reversed*: `batch_load_probe.py --batches 128`
makes 128 requests arrive together, they prefill together, and under the old code
**not one of them would have had a seed** — so every decode step would have
declined to draft, losslessly and silently, at exactly zero speedup. A batched
MTP that never drafts is indistinguishable in a log from a batched MTP that
works badly, which is how this class of thing survives a GPU session.

Now every row is primed. `make_prompt_chunk` right-pads to the batch's longest
prompt, so row `i`'s real positions are `[0, len_i)`, and
`extend_draft_kv_row`'s own `i ≤ L-2` bound already stops there.

## 7. Measurement

`MTP[agg] …` plus one `MTP[b=<B>] …` line per observed batch size, each
carrying `tok_per_step` (**per user** — the number that multiplies the ceiling),
`tok_per_batch_step` (**aggregate**), and `mean_batch`. `mean_batch` collapsing
toward 1 while the load probe reports a large batch is the tell that the cohort
fragmented. Recipe: `RUNBOOK_8 §S3c` (new). `bench` still forces
`--max-seqs 1`, so a bench run remains a **B=1 diagnostic row only** (D2).

## 8. Projection (LABELLED PROJECTION — nothing here is measured)

`CEILINGS.json` per-user ceilings: **83 tok/s at B=64, 68 at B=128.** MTP emits
`k` tokens per step without reading more bytes, so it multiplies straight
through. Draft overhead is `depth` single-layer forwards against a 43-layer
target ⇒ `1 + depth/43`.

| | k=2 (depth 2) | k=3 (depth 3) |
|---|---|---|
| effective multiplier | 2 / (1+2/43) = **1.91** | 3 / (1+3/43) = **2.80** |
| B=64 per user | 83 × 1.91 = **159** | 83 × 2.80 = **232** |
| B=128 per user | 68 × 1.91 = **130** | 68 × 2.80 = **190** |

**k=2 clears Jish's 100-tok/s-at-any-batch target on a single card.** The
reference's own non-simulated floor for V4 MTP is k = 2.30
(`test_dsv4_pro_mtp.py:261`), which lands between these columns. Measured
acceptance remains **UNMEASURED** on our stack — no number is claimed here.

## 9. Corrections owed to CEILINGS.json

`levers_that_BEAT_the_table.MTP_speculative_decode` currently says the fast path
is B=1 only and that "MTP and PagedAttention are mutually exclusive". Both
should now read: the fast path runs at every batch size (wave45-BW); the
PagedAttention conflict is an unimplemented rollback branch that is **moot for
V4** because V4 does not page and V4 is the only model with an MTP head.
