# wave62-CN — MTP at batch: the cohort barrier removed, and the one wire that is left

**Branch:** `feat/mtp-per-sequence-kv` off `02e441e22`. **PR #92.**
**Hardware used: NONE.** Everything below is code, arithmetic, or a CPU unit test.
**Nothing here is an acceptance measurement.** See §6 for the GPU ask.

---

## 0. The correction this acts on

wave59-CJ found the ceiling correctly and it was over-generalised into "MTP can't work at
batch". It can. The narrow, true statement is:

> **Arc's MTP stalls at batch because the cohort shares ONE DENSE CACHE with a cohort-wide
> MIN-ROLLBACK. That is an Arc implementation choice, not a property of speculative decoding.**

PR #86 pushed the wrong way — it *enforces* the lockstep. Correct as a bug fix, wrong as a
destination. This removes the need for it. **#86 does not need to be reverted and is not
touched here**: `enforce_shared_cache_lockstep` is a `KvAdvance::Cohort` postcondition and
stays exactly that. Under `KvAdvance::PerSequence` it is not reached, because
`plan_per_sequence_step` never puts two sequences on one length in the first place.

## 1. 🔑 Verified: no production engine uses a cohort minimum

Read from pinned `main` (vLLM `cc7cf71`, SGLang `c82e928`, TRT-LLM `8efd46e`), not from
memory. All three key the advance on a **per-request scalar**, inside a per-request loop:

| engine | the line |
|---|---|
| vLLM v1 | `v1/core/sched/scheduler.py:1836-1848` — `num_rejected = num_draft_tokens - num_accepted; request.num_computed_tokens -= num_rejected` |
| SGLang | `speculative/eagle_worker_common.py:587-588` `new_seq_lens = batch.seq_lens + accept_lens`; committed at `batch_result_processor.py:686-687` `req.kv_committed_len += num_accept_tokens` |
| TensorRT-LLM | `pyexecutor/resource_manager.py:1330-1346` rewinds `request.py_rewind_len` per request |

A batch-wide reduction over accept lengths appears in **none** of them (`grep` for
`num_accepted.min` / `min(accept_len` returns only unrelated hits — vLLM's cascade-attention
common-prefix pick, SGLang's per-request `finished_len` clamp, TRT-LLM's warmup floor).

**What happens to a rejected draft's K/V: nothing.** vLLM and SGLang neither free nor zero
it — the request's length shrinks so attention stops reading those slots, and the next
step's window overwrites the same physical positions (SGLang says so at
`eagle_worker_common.py:450`: "trailing unaccepted slots stay and are freed as overshoot").
TRT-LLM is the only one that reclaims, and only when a rewind empties a whole block
(`kvCacheManager.cpp:4562-4600`). Prefix-cache correctness is kept by *capping the commit*
at the verified length (vLLM `kv_cache_manager.py:562-565`), never by erasing KV.

**Lockstep is real — but only in dense/padded batching outside paged engines.** It is the
stated motivation of EMS-SD ([arXiv 2405.07542]) and "Batch Speculative Decoding Done Right"
([arXiv 2510.22876]). That is exactly the regime Arc is in. Jish is right about production
engines; wave59 was right about Arc; the two were never in conflict.

## 2. The barrier, stated as one equation

`plan_batch_step` keeps `keep = min_i(u_i + a_i)`, so `next_u_i = u_i + a_i + 1 - keep`.
`plan_per_sequence_step` keeps `committed_i = u_i + a_i` and `next_u_i = 1`. Pinned by
`both_plans_agree_on_what_each_sequence_can_vouch_for`:

```
cohort.keep == min_i(per_seq.committed[i])
```

**The cohort keep IS the batch minimum of the per-sequence commits.** That single `min` is
the whole defect, and it is now the only difference between the two rules — everything else
(drafting, verify, `verify_proposed`, `window_verify_row`) is literally the same code path,
which is what makes the measurements in §3 attributable.

## 3. 📊 Measured — arithmetic on the production plan functions, **not hardware**

`tok_per_step_at_b128_is_bounded_by_the_cohort_rule_and_freed_by_the_per_sequence_rule`,
B=128, depth 3 (`w=4`), 200 steps, one fixture whose sequences accept at content-dependent
rates:

| rule | tok/step | vs B=1 |
|---|---|---|
| cohort (today) | **1.1297** | −23.6% |
| per-sequence | **1.4783** | **0.0000** (exact, `< 1e-9`) |
| same 128 sequences run alone (B=1) | 1.4783 | — |

⚠️ These are **fixture numbers, not V4's**. The absolute values depend on the oracle's
acceptance profile; the *ratio* is the claim. Read `tok_per_step`, never `accept_rate` — a
saturated sequence drafts 0, contributes `proposed = 0`, and leaves `accept_rate`
flattering while `tok_per_step` collapses.

⚠️ **A correction to wave59-CJ §3.** It said saturation is a fixed point and `keep = 1` is
certain at B=128. The dynamics are milder: when *every* sequence reaches `u = w` the minimum
is `w`, so the whole cohort resets to `u = 1` at once and the system **oscillates** rather
than latching. The cost is 24% of `tok_per_step` on this fixture, not a total collapse.
Directionally wave59 was right and the fix is the same; the magnitude was overstated.

## 4. What was built

| piece | file | what it is |
|---|---|---|
| `plan_per_sequence_step`, `PerSeqStepPlan` | `pipeline/mtp_pipeline.rs` | the production rule; `next_uncached ≡ 1`, no reduction over the batch |
| `KvAdvance` + `MtpSpeculativePipeline::kv_advance()` | same | latched once, logged once, `ARC_MTP_PER_SEQ_KV=1`, default OFF |
| `cache_supports_per_sequence_advance` | same | names the first slot that cannot carry its own length |
| per-sequence commit in `step` | same | after the POST op splits the batched buffer, each sequence's own cache is set to `cache_len + u_i + a_i`; **no reduction across the batch, and no shadow state — see below** |
| `KvCache::supports_per_sequence_len` / `kind_name` | `kv_cache/mod.rs` | the capability, with the reason each of the four variants gives |
| `front_pad_kv_cache`, `front_align_batch` | `kv_cache/mod.rs` | left-alignment: shift each row's live run so it **ends** at the batch max |
| `RaggedKvLens`, `make_left_padded_causal_mask` | `layers_masker.rs` | the `[B,1,t_q,k]` additive mask a left-aligned batch needs |

### 🔑 No second map keyed by sequence id

The per-sequence state **is** `Sequence::normal_cache`'s own `current_seq_len`. The commit
happens in `step`, in the same scope that computed it, from a plain `Vec` indexed by the
batch row, immediately after `handle_post_cache_op` has split the batched buffer back into
per-sequence storage. An earlier draft staged the lengths in a
`HashMap<seq_id, usize>` on the pipeline; that was removed. A shadow copy of a length the
cache also holds is precisely the divergence wave59-CJ §1 documents, reappearing in a new
place — the bug being fixed, in a new hiding spot. (Thanks to the PR #90 author for the
catch.)

Steps whose POST op is not `CacheInstruction::Out` take the cohort rule, because there is no
per-sequence storage to write to on those steps.

### 🔑 The paged route needs none of this — and is a separate change

`KVCacheManager::trim_request_to_num_tokens` under `ARC_SEGMENTED_KV=1` already routes to
`SegmentedAllocator::rollback(pool, request_id, 0, num_tokens)`
(`paged_attention/kv_cache_manager.rs:445-448`) — **per-request, absolute token count, no
new API needed**. PagedAttention therefore has no cohort barrier to remove. What blocks
paged MTP is upstream: `MtpSpeculativePipeline::step`'s fast path only accepts
`CacheBackendMetadata::DefaultInstructions`, so a paged batch never reaches that rollback.

⚠️ And PR #90 makes V4's read *expressible*, not paged: `cache_write_and_gather` still
builds `cu_seqlens_kv` from per-sequence context lens and must consume
`SegmentPlan.cu_seqlens` (B×S rows) instead, and the compressor's `xs` history has no block
table at all. Do not assume V4 end-to-end from the segment allocator landing.

### 🔑 Why left-alignment is the dense-cache answer

`SingleCache::append` writes every sequence's new K/V at **one** shared offset
(`single_cache.rs:225`). Right-padding a ragged cohort puts each row's hole in the middle,
where the next append lands. Left-alignment puts every row's live run against the **same
end column**, so that one shared offset is simultaneously correct for all of them — the
dense batching code in `NormalCacheManager::clone_in_cache` needs **no change at all**, and
`ensure_uniform_batch_cache_lens` passes for the same reason it always did. What the caller
gains is that the length handed back on the way *out* may differ per sequence.

The dead prefix does not grow without bound: it is `max_j L_j − L_i`, and sequences in one
batch accept at statistically the same rate, so the spread is a `sqrt(steps)` random walk,
not a linear one (≈6.6% at 1000 steps, B=128).

## 5. 🔴 The honest gap — this does NOT move the B=128 number yet

`KvAdvance::PerSequence` is **unreachable in production today**, deliberately, and the
refusal is logged by name. Two things block it and both are named in code:

1. **`XsRolling` cannot carry its own length** (`KvCache::supports_per_sequence_len`).
   V4's compressor slots hold two time bases (`comp` rows + a raw `tail`) plus scalar
   `tokens`/`base` that are one number for the whole batch, and per-sequence advance needs
   `advance` to append a **different number of rows per sequence** — which one shared append
   offset cannot express. Same gap wave61-CL §6 names ("the compressor's `xs` history has no
   block table") and wave29-BC §4b ruled its own project. **This is now the third wave to
   land on it. It is the keystone.**
2. **No model threads the ragged mask** (`target_masks_ragged_batches`). `front_align_batch`
   already produces `lead_pad` per sequence and `make_left_padded_causal_mask` already turns
   it into the mask; what is missing is carrying it from the pipeline through
   `inputs_processor` into the model forward, where the mask is rebuilt from
   `seqlen_offsets` alone today. `dsv4_attention::compose_caller_mask` already accepts rank
   4, so the consuming end is done.

**Refusing is not conservatism, it is the only correct behaviour.** A zero-filled dead
prefix is *not* a masked prefix — a zero K row scores logit 0 and takes real softmax weight.
Turning this on without (2) would be a wrong answer nothing downstream catches, which is the
exact failure this project already paid for once with FP8 KV.

**MTP has exactly one model in the tree** (`mtp_decode_kit` → `deepseek4.rs:4250`; the trait
default returns `None`), so there is no second model to exercise the per-sequence path on
either. The order is forced: (1) → (2) → the number.

### What was ruled out, with the reason

* **Advance the shared cache by `w` and mask the dead rows.** V4's raw branch is a
  ~128-token *suffix* window; 54% of it would be dead rows, so the effective window shrinks
  to ~59 real tokens. Not token-identical. Dead on arrival.
* **A wider window `W = max_i(u_i) + depth`.** `u_i` grows by `a_i` per step whenever any
  sequence sits at `u=1, a=0`, so `W` grows linearly and verify compute with it.
* **Discard the accepted-but-uncached surplus.** Throughput becomes `min_i a_i` ⇒ 1.0
  tok/step at B=128. Strictly worse than today.
* **Bucket the cohort by accept length.** `select_running_bucket` runs one bucket per step,
  so a B=128 cohort becomes ≤4 buckets of ~32 and only one runs. This is wave59's option (B)
  under another name.

## 6. 🔴 The GPU ask

Nothing here changes a default, so the check is a **no-regression** one, on the box that is
already serving V4:

```bash
# A: control (current master behaviour — the flag is OFF)
mistralrs bench -m <v4> -b 1 -b 128 2>&1 | tee /tmp/a.txt
# B: same box, same flags, per-sequence advance REQUESTED
ARC_MTP_PER_SEQ_KV=1 ARC_MTP_LOG_ACCEPTANCE=1 mistralrs bench -m <v4> -b 1 -b 128 2>&1 | tee /tmp/b.txt
```

**The one number:** decode tok/s in B as a fraction of A, at B=1 and B=128. The claim under
test is `B/A == 1.00` at both — B must be byte-identical to A, because the refusal in §5
means B takes the cohort path. **What B additionally proves is that the refusal fires and
names `XsRolling`**: `grep "cannot honour it" /tmp/b.txt`. If that line is absent the probe
is wrong; if `B/A != 1.00` something in this diff touched the live path and must be found.

## 7. Gates

`cargo check -p mistralrs-core` green · `cargo test -p mistralrs-core --lib` **377 passed,
0 failed** (was 366) · scoped clippy lane green · **zero rustfmt drift**, checked
like-for-like on `/tmp` copies against the pre-change baseline (all three files 0 → 0).

### Mutation runs — no test here is vacuous

```
plan_per_sequence_step: committed = (u+a).min(w).min(batch_min)
  both_plans_agree_on_what_each_sequence_can_vouch_for ............ FAILED
  per_sequence_plan_ignores_every_other_sequence .................. FAILED
plan_per_sequence_step: next_uncached follows the cohort rule
  per_sequence_advance_is_token_identical_to_the_b1_reference ..... FAILED
  per_sequence_advance_never_lets_a_tail_ratchet_at_b128 .......... FAILED
  tok_per_step_at_b128_… ......................................... FAILED
  per_sequence_plan_ignores_every_other_sequence .................. FAILED
front_pad_single: slice_set at offset 0 instead of `lead`
  front_pad_moves_the_live_run_to_the_end_and_zeroes_the_prefix ... FAILED
make_left_padded_causal_mask: drop the `j < lead` condition
  left_padded_mask_kills_the_dead_prefix_and_the_future_… ......... FAILED
  a_single_query_row_still_gets_its_ragged_mask .................. FAILED
cache_supports_per_sequence_advance: never refuse
  the_capability_probe_names_the_slot_that_cannot_carry_its_own_length  FAILED
```

The no-ratchet test carries its own control in the same test: the **cohort** rule on the
identical fixture must still ratchet tails to the window, or the premise is gone and the
whole workstream is unnecessary.

## 8. Surfaced, not shipped

1. **`XsRollingCache` per-sequence state is the keystone** (§5.1), and the per-row append it
   needs is precisely a `slot_mapping` — the primitive every paged engine has and Arc's
   dense path does not. PR #90's gather kernel already accepts "a row is any
   (block-table, length) run", so the read side exists; the write side does not.
2. **The ragged-mask wire** (§5.2) is small and independent of (1) — it would let any plain
   dense model take the per-sequence path, if one ever gets an MTP head.
3. **`MtpAcceptance::from_fused_verify(d, n_acc)` records `proposed = d`**, so a saturated
   sequence contributes a 0-proposed step and keeps `accept_rate` flattering. Carried over
   from wave59-CJ §6.3, still worth an explicit `drafting_sequences / batch` counter.
