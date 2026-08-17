# wave56-CG — the three wave51-CB serving crashes: root-caused, fixed, mutation-proven on CPU

**Branch:** `fix/kv-length-invariants` off `46ea6948d` (origin/master, PR #76 head —
the exact binary wave51-CB served on the H200).
**Hardware needed: none.** All three are shape/length invariants and all three are
reproduced byte-for-byte on CPU. No rental was taken; no rental is required to land this.

---

## 0. The headline correction

All three panics were attributed to `SingleCache::append`. **None of them are in
`append`.** `kv_cache/mod.rs:498` and `:499` are the two `slice_set` calls at the
bottom of **`NormalCacheManager::clone_in_cache`**:

```rust
498:  batch_k.slice_set(&src_k, 0, offset).unwrap();
499:  batch_v.slice_set(&src_v, 0, offset).unwrap();
```

`clone_in_cache` builds ONE dense batched cache per layer, allocating it from
`seqs[0]`'s tensor dims and `slice_set`-ing every other sequence in along dim 0.
`Tensor::slice_set` demands an exact match on **every other dimension**. So the
real contract was never "sequences share a length" — it was "every sequence's
per-layer buffer is dimensionally identical to `seqs[0]`'s", and nothing enforced
either statement in the shipped binary.

**Why V4 and only V4.** V4's cache vector is not homogeneous: `DeepSeekV4::new`
builds 43 K/V entries followed by **41 `KvCache::XsRolling` compressor histories**
(`deepseek4.rs:3737-3766`), and `GeneralMetadata::num_hidden_layers` is derived
from the *cache length* (`normal.rs:1214`), so `clone_in_cache` batches all 84.
The two tensors it batches for an `XsRolling` slot are:

| line | side | tensor | dim 1 is | quantised? |
|---|---|---|---|---|
| 498 | K | `xs.comp.all_data` | `comp.capacity_seq_len` | yes — 64, then +512 blocks |
| 499 | V | `xs.tail` | `tokens - base` | **NO — exact, changes every token** |

For a `KvCache::Normal` slot the batched dims are `[1, n_kv_heads, capacity, head_dim]`,
whose dim 1 is the head count. **V4's K/V halves batch happily at any pair of
lengths** — which is why the panic only ever appeared on the compressor slots, and
why the scheduler's slot-0 check can never see it. That is asserted as a test:
`kv_slots_alone_cannot_see_the_divergence`.

---

## 1. Crash (B) — `kv_cache/mod.rs:498`, `576 <> 64`, ordinary decode path

**Not** a head dim vs a step count. Both numbers are `XsRollingCache::comp`
**capacities**: `init_rows = 64` (`xs_rolling.rs:118`) and `64 + CACHE_GROW_SIZE`
= **576** after exactly one growth.

Two facts compose into the bug:

1. `comp` holds one row per `ratio` tokens, so a CSA layer (`ratio = 4`) crosses
   64 rows at **260 tokens**. **The batch sweep's sequences were ~132 tokens
   (68-token prompts, 64 decode) — every `comp` buffer in it was 64 wide, so they
   always matched.** GSM8K generated up to 2048. That is the whole of "zero
   crashes in the 505-request sweep, two in ~1,300 GSM8K requests"; it is
   length-dependent, not load-dependent.
2. `SingleCache::reset` (`single_cache.rs:52-55`) clears `current_seq_len` and
   `all_data` but **not `capacity_seq_len`**, and the next `append` re-allocates at
   that retained capacity. V4's attention layer resets the compressor slot at the
   start of every prompt (`deepseek4.rs:1361`, `seqlen_offsets.iter().all(|&o| o == 0)`),
   and `clone_out_cache` then stamps the batch's metadata onto every sequence in
   it. **So a prompt batch scheduled after a long-context batch hands brand-new
   sequences a 576-wide buffer, while one scheduled after a short batch hands
   them 64.**

3. **How two different widths come to meet.** `clone_out_cache` stamps the
   *batched* slot's `capacity_seq_len` onto every sequence that passes through a
   batch, so the pipeline slot's capacity is monotonically non-decreasing for the
   life of the process and active sequences converge on it. **But
   `select_running_bucket` runs exactly ONE bucket per step and waitlists the
   rest** (`default_scheduler.rs:81-157`), so a waiting sequence's cache is frozen
   at 64 while the running bucket grows the pipeline slot to 576. When the buckets
   later coalesce — which is what `COALESCE_PAYBACK_STEPS` exists to make happen —
   the two widths meet in one `clone_in_cache`. This step is inferred from the
   scheduler's own documented behaviour rather than observed in the log; steps 1
   and 2 are read directly off the code and are what the CPU test reproduces.

⇒ **Two sequences at the identical length can hold different-width compressed-row
buffers.** The scheduler cannot prevent it: it agrees on every length there is to
bucket on. The extra width is preallocation slack holding nothing.

**This also explains the sweep/GSM8K split twice over**: the sweep's sequences were
all the same length (one bucket, no waitlisting) *and* too short to grow any
buffer. GSM8K at `--concurrency 16` with 1,319 distinct prompt lengths is many
buckets, constantly coalescing, with generations well past 260 tokens.

**Verdict: not a ragged batch. A tolerance bug.** `KvCache::Normal` is
slack-tolerant to ±511 tokens by accident of `CACHE_GROW_SIZE`; `comp` was
tolerant to ±0 across a growth boundary.

## 2. Crash (A) — `kv_cache/mod.rs:499`, `18 <> 22` / `19 <> 23`, MTP at B=8

Line 499 is the **V** half = `xs.tail`, `[B, tokens - base, hidden]`. Its width is
`T - ratio * floor((T - margin) / ratio)` for `span_groups = 1`. For V4's **HCA**
layers (`ratio = 128`, `span_groups = 1`, `margin = XS_TAIL_MARGIN_TOKENS = 16`)
that is **18 at T=274 and 22 at T=278**; **19 at T=275 and 23 at T=279**. Both
observed pairs, both exactly **four tokens** apart.

**The hypothesis in the brief (`depth+1` draft tokens counted into one length but
not the other) is close but off by one in its framing.** The gap is not a
miscount — it is the *designed* divergence:

* `mtp_pipeline.rs:2503-2515` — the batched verify commits `1 + accepted_i` tokens
  per sequence but rolls the ONE dense cache back to `min_i(u_i + accepted_i)`:
  *"each sequence's surplus stays committed as TOKENS and comes back as a longer
  uncached tail next step."*
* `sequence.rs:786-806` — `cache_bucket_len` therefore buckets on the **cache**
  length, deliberately, *"the one that keeps a batched MTP cohort whole when its
  accept lengths differ"* — and it reads `.flatten().next()`, i.e. **cache slot 0
  only**.

Both decisions are individually correct. **Jointly they are unsound the moment a
cache slot's batched width tracks tokens rather than the cache length** — which is
exactly what `xs.tail` does. At depth 2 the per-step surplus is 0..2 tokens, so a
4-token spread is two steps of ordinary ragged acceptance. B=1 cannot produce it
(nothing to diverge from), which is why B=1 measured cleanly at `accept_rate=0.4194`.

**Verdict: a genuinely ragged batch.** The two sequences hold different compressor
history and *cannot* share one dense buffer. Padding would fabricate history. It
must be refused — by name, and not on the engine task.

## 3. Crash (C) — `engine/mod.rs:428`, `SendError { .. }` — **symptom, and independently a defect**

`engine/mod.rs:428:25` is the *expansion site* of `handle_pipeline_forward_error!`;
the `unwrap` was inside the macro body at `utils/mod.rs`:

```rust
seq.responder().send(Response::ModelError(..)).await.unwrap();
```

**Cause or symptom: symptom.** It can only fire after some other error has already
put the batch on the failure path — it is never first. **But it is also a real,
separate fleet defect**: the receiver is the HTTP handler's channel, and any client
that has already given up (timeout, abort, dropped connection) has closed it. So
reporting a model error to a departed client escalated *one* failed request into a
dead engine and orphaned all 16 in flight. **Error reporting is the one path that
must not be able to fail loudly.** Note the three sibling macros immediately above
it (`handle_seq_error!` et al.) already did this correctly — only this one unwrapped.

---

## 4. What was fixed, and where the invariant now lives

**One place: `NormalCacheManager::clone_in_cache` (`kv_cache/mod.rs`).**

1. **`CacheManager::clone_in_cache` and `Pipeline::clone_in_cache` now return
   `Result<()>`** (12 implementors + 5 call sites). `Pipeline::step` already returns
   `Result`, so a refusal flows to `handle_pipeline_forward_error!`, which fails the
   **requests** and keeps the engine running.
2. **`ensure_uniform_batch_cache_lens` runs in RELEASE.** It was a `debug_assert!`
   — present in CI, **absent from the binary that served on the H200**. It now
   returns an error naming the slot, both lengths and the sequence index, and
   explains that the scheduler only buckets on slot 0.
3. **Slack vs content is made explicit** (`struct BatchSrc`). Every slot declares
   which of its two tensors is a grown *capacity* buffer (`SingleCache::all_data`,
   `comp.all_data`) and which is live *content* (`xs.tail`, `TurboQuant::current_data`).
   Capacity dims are widened to the **batch maximum** and zero-padded
   (`pad_slack`); content dims must match exactly or `reconcile_batch_dims`
   refuses with a message that names the dim and both widths. This also generalises:
   a `KvCache::Normal` pair at different capacities would have panicked identically
   and now cannot.
4. **`capacity_seq_len` is read back from the buffer that now exists**, not from
   the `seqs[0]` template — a stale 64 against a 576-wide buffer would make the next
   `SingleCache::append` growth try to slice a wide buffer into a narrow one.
5. **`engine/mod.rs::step_catching_panics`** wraps all three `pipeline.step` call
   sites: a *panic* anywhere in a forward becomes an `Err` on the existing recovery
   path. The specific panics are fixed at source; this is the backstop for the ones
   nobody has found yet, and it is what makes item 6 below true in general.
6. **`utils::send_response_or_log`** replaces the two unwrapped responder sends, and
   the recovery path's `evict_all_caches().unwrap()` (both copies) is now logged.

**Consequence for MTP at B=8:** the panic is gone and the engine survives, but a
ragged V4 cohort is now *refused* rather than served. **MTP at batch on V4 is not
yet working — it is now failing safely instead of fatally.** The remaining work is
to make the xs slots advance in lockstep with the KV cache under ragged acceptance
(or to bucket on the widest slot); that is a follow-up, and it is now bounded by a
test that says exactly what "correct" means. **Do not read this PR as "MTP at batch
is fixed."**

---

## 5. Regression tests, all mutation-proven

Built on a production-shaped fixture per **DOCTRINE D12**: preallocated K/V
`all_data` (not `KvCache::new_normal`'s `None`), and `XsRolling` state produced by
running the real `XsRollingCache::advance`. Each test also asserts its fixture
*discriminates* by running the legacy operation (`legacy_batch_error`) and requiring
the exact historical message — so the reproduction is re-checked on every CI run,
not just once.

| test | reproduces |
|---|---|
| `xs_comp_capacity_slack_does_not_kill_the_batch` | `shape mismatch on dim 1, 576 <> 64` |
| `ragged_xs_tail_is_refused_by_name_not_panicked` | `shape mismatch on dim 1, 18 <> 22` |
| `kv_slots_alone_cannot_see_the_divergence` | the K/V halves batch fine at 274 vs 278 — why slot-0 bucketing is blind |
| `clone_in_cache_refuses_a_length_mismatched_batch` | de-`debug_assertions`-gated; now asserts a value, in every profile |
| `send_response_or_log_tests::a_departed_client_does_not_take_the_engine_down` | `SendError { .. }` |
| `step_panic_containment_tests::a_panicking_step_becomes_an_error_not_an_engine_death` | any panic → `Err` |

**Mutation runs (fix reverted, tests re-run):**

```
kv_cache/mod.rs:683:54  called `Result::unwrap()` on an `Err` value: shape mismatch on dim 1, 576 <> 64
kv_cache/mod.rs:684:54  called `Result::unwrap()` on an `Err` value: shape mismatch on dim 1, 18 <> 22
utils/mod.rs:38:36      called `Result::unwrap()` on an `Err` value: SendError { .. }
engine/mod.rs           a_panicking_step_becomes_an_error_not_an_engine_death ... FAILED
```

Lines 683/684 are 498/499 after the rewrite — the same two `slice_set` calls, the
same two messages, on CPU.

---

## 6. Gates

`cargo check --workspace` green · `cargo test -p mistralrs-core` **332 passed, 0
failed** · scoped clippy lane (`arc-bench arc-engine arc-cuda-graph arc-cli
mistralrs-quant`) green · `mistralrs-core` clippy clean in every file touched ·
**zero new rustfmt drift** (per-file diff counts compared against `master`;
`engine/mod.rs` 7→7, `pipeline/mod.rs` 22→22, `normal.rs` 17→17, others 0→0 —
`rustfmt` was never run on a `mod.rs`, per the 87-file mass-reformat hazard).

---

## 7. Surfaced, not shipped

1. **`Sequence::cache_bucket_len` reading `.flatten().next()` (slot 0) is the
   structural hole.** The fix makes a batch it mis-forms fail cleanly instead of
   fatally; it does not make the scheduler stop forming one. A key that covers
   every populated slot would close it — at the cost of shattering ragged MTP
   cohorts, which is the throughput cliff `cache_bucket_len` exists to avoid.
   **That trade is a decision, not a bug fix.**
2. **The engine reboots lazily**, on the next `get_sender` (`lib.rs:938-940`).
   wave51-CB §3.2 observed the MTP engine reboot and then serve nothing at 0% GPU.
   Panic containment means the reboot is needed far less often; it does not fix the
   reboot.
3. **`DefaultScheduler::schedule` can put prompt and completion sequences in the
   same selected bucket** (both lists are non-empty in the same engine iteration),
   after which the completion branch's `last_completion_ids` check can leave
   `pre_op = CacheInstruction::Nothing` over a pipeline cache the prompt step just
   overwrote. Not proven to fire, not touched here. Worth its own look.
4. **FP8 KV still has no INFO line when it engages** (wave51-CB §5) — a correctness
   check whose only positive control is `getenv` is weaker than it needs to be.
