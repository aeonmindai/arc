# wave63-CO — The `xs` keystone: per-row compressor state

**Branch:** `feat/xs-per-sequence`, stacked on `feat/mtp-per-seq-kv` (PR #92) at `24613e87a`.
**Hardware used: NONE.** Everything below is code, arithmetic, or a CPU unit test.
**Nothing here is an acceptance measurement.** See §6 for the GPU ask.

---

## 0. The blocker, in one sentence

> V4's compressor history `xs` carried a single `tokens`/`base` pair for the whole batch, so
> no sequence could advance its compressed KV independently of the others.

Three workstreams arrived at it from three directions and each ruled it out of its own scope:
PR #90 §6 ("the compressor's `xs` history has no block table"; "a third storage class, not a
KV region"), PR #92 §5.1 ("`XsRolling` cannot carry its own length … **third wave to land on
it; it is the keystone**"), wave29-BC §4b ("unpaged, and its own project"). This is that
project.

## 1. 🔑 The load-bearing discovery: the two time bases need OPPOSITE anchors

I expected this to need one mechanism. It needs two, and the reason is geometric.

`XsRollingCache` holds `comp` (completed compressed blocks) and `tail` (the retained raw
window). Under a ragged batch they behave nothing alike:

| | anchor | what a ragged batch costs it | what it needs |
|---|---|---|---|
| `comp` | **start** — block `j` covers absolute tokens `[jr, (j+1)r)` for every row | rows are at different block counts | a **per-row append column** |
| `tail` | **end** — a suffix window that slides | rows retain different widths | one shared append offset ⇒ **left-alignment** |

`tail` is exactly PR #92's case and takes PR #92's answer: end-anchor it, so every row's live
run finishes at the same column and the single offset `Tensor::cat(&[tail, xs_new], 1)` uses
is simultaneously right for all of them.

`comp` is **not** that case, and forcing it into that shape would have been the wrong call.
Left-aligning `comp` would need a per-row data shift every step *and* a two-sided mask. Left
alone, start-anchored, it needs neither — and it gets its mask for free:

> A row's live blocks are `[0, tokens_i / ratio)`, and the compressed branch's existing
> causality threshold is `b < floor((q_abs + 1) / ratio)` (`dsv4_attention.rs:419-434`).
> Since `q_abs + 1 <= tokens_i`, that threshold **already** excludes exactly the columns a
> shorter row has not reached. No extra mask term — only a per-row query position.

So the ragged `comp` read costs zero new masking, and the whole per-row problem reduces to
one primitive: a per-row append column.

## 2. 🔑 The second discovery: the window geometry is a function of the RESIDUE alone

The naive ragged path is one compressor call per batch row — 128 calls per layer per step,
which would cost more than the raggedness buys. It does not have to be.

For row `i` at `tokens_i = q_i * ratio + r_i`, the slice handed to the compressor starts at
physical offset `off_i = W + (1 - span) * ratio - r_i` and is
`len_i = ((r_i + t_new)/ratio - 1 + span) * ratio` wide. **Both depend on `r_i` only.** Two
rows at the same residue need byte-for-byte the same window and differ *only* in the
destination block `q_i`.

Therefore: group by residue, one compressor call per group, scatter the outputs to per-row
columns. A uniform batch is exactly one group. A ragged one is at most
`min(B, ratio, spread + 1)` groups — and under MTP the spread is the verify window
(`depth + 1`), so **≤ 4 calls, not 128**. Pinned by
`window_geometry_depends_only_on_the_token_residue` (8 rows, 2 residues ⇒ 2 groups).

## 3. What was built

| piece | file | what it is |
|---|---|---|
| `tokens: Vec<usize>` / `base: Vec<usize>` | `kv_cache/xs_rolling.rs` | the per-row state, **on the cache** — no second map keyed by sequence id |
| `plan_xs_advance` / `XsAdvancePlan` / `XsCompressGroup` | same | the whole step as pure arithmetic: groups, new lengths, retained width, scatter band |
| `scatter_comp_rows` | same | the `slot_mapping`-shaped write: `index_add` onto zero + written/not-written select, **bounded to the column band** the plan says every write falls in (`max g_target − min g_done`, 1–2 columns) so it is `O(B · spread)`, not `O(B · capacity)` |
| `advance_uniform` | same | the pre-existing scalar path, **verbatim** |
| `split_row` | same | the inverse of batch assembly; restores the per-sequence invariant `W == tokens − base` from the END of the shared window |
| `supports_per_sequence_len` → `xs_per_sequence_enabled()` | `kv_cache/mod.rs` | `ARC_V4_XS_PER_SEQ`, **default OFF** |
| `pad_slack(.., at_front)`, `BatchSrc::v_slack_at_front`, `xs_rows` | same | end-anchored reconciliation + carrying every sequence's own lengths into the batch |
| `first_mismatched_cache_len_inner(.., xs_per_seq)` | same | the uniformity precondition, exempting compressor slots **and nothing else** |

### 🔑 Why `advance` keeps a verbatim uniform branch

`advance` dispatches on `rows_uniform()`. Every B=1 sequence and every uniform batch runs the
pre-existing code, character for character. "Token-identical with the flag off" is therefore
a property of the control flow, not something a test has to keep re-establishing — and the
uniform layout is the special case `W == tokens − base`, under which end-anchored and
start-anchored coincide exactly, so the reinterpretation of `tail` is a no-op there.

### 🔑 The lengths have exactly one owner

Per-row state lives in `XsRollingCache`, written only by `advance`, `set_len`, `set_row_lens`
and `split_row`. There is no `HashMap<seq_id, …>` anywhere — the divergence wave59-CJ §1
documents and PR #92 removed, not reintroduced in a new hiding spot.

### What is refused, on purpose

* **A shared `set_len` on a ragged cache.** The window is end-anchored, so each row would need
  its own narrow; one call cannot express it. Split first (`split_row`) and truncate the
  per-sequence caches — which is what the per-sequence commit already does.
* **Reshaping a *ragged* cache to a different batch width.** Which rows survived is the
  caller's knowledge. Uniform reshape (batch grow/shrink/reset) stays exact and is unchanged.
* **A batch whose widest retained window exceeds its shortest sequence.** `set_row_lens`
  refuses by name, from a fallible path, so the requests fail and the engine task does not.

## 4. No-regression, and the identity claim

`cargo test -p mistralrs-core --lib` → **398 passed, 0 failed** (was 377) ·
`--test synthetic_load_smoke` → **13 passed** · workspace `cargo check --tests` green ·
scoped clippy lane green · **zero rustfmt drift**, checked like-for-like on `/tmp` copies
(xs_rolling 0→0, kv_cache/mod 0→0, deepseek4 30→30, mtp_pipeline 0→0).

The identity tests run the **real `V4Compressor`**, give every row its **own** input stream
(sharing one stream would make every row's block `j` cover identical tokens and hide any
row mix-up), and compare each row against the same sequence advanced alone:

| test | what it pins |
|---|---|
| `batched_ragged_xs_is_token_identical_to_the_b1_reference_csa` | 4 rows at 4 residues mod `ratio`, CSA/overlapping |
| `..._hca` | ratio 128, where almost every row completes no block |
| `two_rows_one_compressor_call_two_destinations` | the `slot_mapping` case: one compressor call, two `comp` columns |
| `a_ragged_batch_stays_token_identical_across_many_steps` | 24 steps, re-batched each step, incl. two rows sharing a residue a block apart |
| `a_rolled_back_row_keeps_its_own_resume_point_through_a_batch` | `set_len` leaves `base` high; batching must not loosen it |
| `splitting_a_batched_row_restores_the_per_sequence_window` | the split takes this row's share from the END |
| `a_uniform_batch_still_matches_the_b1_reference` | the control — the untouched scalar path |

⚠️ These are **CPU unit tests on the real compressor with fixture weights, not hardware**
(D14). They establish token-identity of the cache; they are not a throughput measurement.

### Mutation runs — no test here is vacuous

```
scatter_comp_rows: drop the per-row destination (every row writes at band_start)
  two_rows_one_compressor_call_two_destinations .................... FAILED
  a_ragged_batch_stays_token_identical_across_many_steps ........... FAILED
plan_xs_advance: keep_from without the `.max(base)` floor
  a_rolled_back_row_keeps_its_own_resume_point_through_a_batch ..... FAILED
split_row: take the row's share from the FRONT of the shared window
  splitting_a_batched_row_restores_the_per_sequence_window ......... FAILED
advance: always take the uniform path
  5 tests ......................................................... FAILED
front_align_batch: keep the lead from EVERY slot (incl. trailing xs)
  front_align_reads_the_dead_prefix_from_a_kv_slot_not_a_trailing_xs_one  FAILED
first_mismatched_cache_len_inner: never exempt the compressor slot
  the_uniformity_check_exempts_xs_slots_and_nothing_else ........... FAILED
batch assembly: front-pad the xs window at the BACK
  3 tests ......................................................... FAILED
```

⚠️ The first mutation **survived my initial fixtures** and the fixtures were wrong, not the
mutation: every row happened to land on the same `comp` column, so the per-row destination was
never exercised. `two_rows_one_compressor_call_two_destinations` exists because of that, and
the multi-step fixture's starting lengths were changed to put two rows at one residue a block
apart. Two more (`.max(base)`, the split anchor) survived until the rollback and split tests
were added. **Three of eight mutations initially survived.** Recording that because a mutation
table nobody failed is a table nobody ran.

### A pre-existing bug this surfaced

`front_align_batch` kept whatever `lead` the **last** slot returned. On DeepSeek V4 the 41
compressor slots come last, so it reported "no dead prefix" for every sequence — and PR #92's
consumer would have built no mask for a batch that needs one. Fixed and pinned.

## 5. 🔴 The honest gap — this still does NOT move the B=128 number

`KvAdvance::PerSequence` remains unreachable, and the refusal is still logged by name. PR #92
named two blockers. **This removes the first.** The second is untouched and is now the only
one:

> **No model threads the ragged mask** (`MtpSpeculativePipeline::target_masks_ragged_batches`,
> which returns `Err` unconditionally). `front_align_batch` produces `lead_pad`;
> `make_left_padded_causal_mask` turns it into the `[B,1,t_q,k]` additive mask; what is
> missing is carrying it from the pipeline through `inputs_processor` into the forward, where
> the mask is rebuilt from `seqlen_offsets` alone.

For V4 specifically that wire needs a **per-row `q0`**, because `dsv4_attention` derives one
scalar absolute query position from `t_k_full - t_q` and uses it for *both* branches. Under
left-alignment that position overstates row `i`'s true position by `lead_i`, which makes the
raw window still correct (`q` and `k` shift together) but the compressed threshold too
**permissive** — a row would attend blocks it has not reached. A rank-4 caller mask can remove
those columns (`compose_caller_mask` already accepts rank 4, and removal is all that is
needed, since the module's own mask is an AND), so the consuming end is not the problem. The
missing piece is the same wire, extended to cover the compressed columns as well as the raw
ones. **Deliberately not started here** — it is a model/inputs-processor change, not a cache
change, and stacking it would make this diff unreviewable.

**So the observable effect of turning `ARC_V4_XS_PER_SEQ=1` on today is: the refusal moves.**
`cache_supports_per_sequence_advance` stops naming `XsRolling`; `kv_advance` still returns
`Cohort`, now citing the mask. That is the correct and only safe outcome — a zero-filled dead
prefix is not a masked prefix, and serving from it would be a wrong answer nothing downstream
catches (the FP8-KV failure, again).

### What was ruled out, with the reason

* **Left-align `comp` like `tail`.** Needs a per-row data shift every step (the alignment
  changes whenever the row's block count does) *plus* a two-sided mask, and gives up the
  "physical column = absolute block" identity that makes the existing threshold sufficient.
  Strictly more work and more mask, for nothing.
* **One compressor call per batch row.** 128 calls per layer per step. §2 makes it ≤ 4.
* **Scatter over the whole `comp` buffer.** `B · capacity · head_dim` ≈ 33M elements per layer
  per step at B=128. The band bound makes it ≈ 200K.
* **`Tensor::scatter` / `scatter_set`.** Needs `indexes.shape() == source.shape()` against a
  destination that is a row *subset*, so it would need an `index_select` out and an
  `index_add` back — strictly more than the flat `index_add`-onto-zero used instead.

## 6. 🔴 The GPU ask

Nothing here changes a default, so the check is a **no-regression** one, on the box already
serving V4:

```bash
# A: control (current behaviour — both flags off)
mistralrs bench -m <v4> -b 1 -b 128 2>&1 | tee /tmp/a.txt
# B: same box, same flags, per-row xs state requested
ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1 ARC_MTP_LOG_ACCEPTANCE=1 \
  mistralrs bench -m <v4> -b 1 -b 128 2>&1 | tee /tmp/b.txt
```

**The one number:** decode tok/s in B as a fraction of A, at B=1 and B=128. The claim under
test is `B/A == 1.00` at both, because the refusal in §5 means B still takes the cohort path.

**What B additionally proves is that the refusal has MOVED**: `grep "cannot honour it"
/tmp/b.txt` must now cite the ragged mask and must **not** contain `XsRolling`. If it still
says `XsRolling`, the flag is not reaching the cache. If `B/A != 1.00`, something in this diff
touched the live path and must be found.

## 7. Surfaced, not shipped

1. **The ragged-mask wire is now the single blocker** (§5) and for V4 it needs a per-row `q0`
   in `dsv4_attention`, not just the `[B,1,t_q,k]` K/V mask PR #92 built. Own change, and the
   next one on this critical path.
2. **`XsRollingCache::advance` still recompresses on the CPU-side host loop per layer.** The
   ragged path adds ≤ 3 extra small kernel launches per layer per step (index_select, cat,
   index_add). At 41 layers that is ~120 launches/step — worth folding into the CUDA-graph
   capture before quoting a B=128 number, and worth measuring before assuming it matters.
3. **`plan_xs_advance` is `pub(crate)` and not re-exported**, so the integration-test crate
   cannot pin the group arithmetic directly; it is covered indirectly through the model tests.
   Exporting it would let `synthetic_load_smoke` assert group counts.
