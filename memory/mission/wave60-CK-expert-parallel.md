# wave60-CK — expert parallelism, stage 0 + stage 1 (EP=2)

**Date:** 2026-08-17 · **Base:** `master` @ `c763123be` · **Branch:** `feat/expert-parallel-ep2`
**NO GPU WAS RENTED. $0.00 SPENT. NO BOX EXISTS.** Everything below is code +
unit tests + arithmetic. **Nothing here is a measurement of Arc on multiple
GPUs, because Arc has still never run a forward pass on more than one GPU.**

Grading, as wave44-BV: `[measured]` = on hardware, `[computed]` = arithmetic,
`[tested]` = asserted by a unit test on this branch.

---

## 0. Headline

**The expert-parallel contract is implemented, wired end to end, and its
arithmetic is proven by mutation-tested unit tests. The number Jish asked for
— per-user tok/s at B=128 on two cards — is NOT in this document, because no
hardware ran.**

Two policy changes arrived mid-wave and both are reflected here:

1. **CPU-only validation is banned** (Jish: *"don't do any fucking CPU work,
   ban CPU only gpu"*). The 2-rank in-process tests below are therefore
   labelled what they are — **unit tests of the arithmetic**, not validation.
   They exist so that a rental measures speed instead of debugging correctness.
   The question EP exists to answer — per-collective latency on NVLink — is
   untouched by them.
2. **`runcrate` is called by the coordinator only.** This agent called
   `runcrate ps` **once**, before that rule arrived; it returned
   `session expired — refresh_token_already_used` and was not retried.

---

## 1. Stage 0 — what was broken and is now fixed

### 1.1 🔴→🟢 The CUDA wrappers validated device **kind**, not **ordinal**

`mistralrs-quant/src/qtip/cuda_ops.rs` took its stream from one tensor
(`blocks.device()`) and then only checked the others with
`matches!(.., Device::Cuda(_))`. Activations on `cuda:0` with expert weights on
`cuda:1` passed that check and the kernel launched **on the wrong device's
stream** — silent corruption, not an error. Unreachable while `mapper.map`
co-locates everything first; it is exactly the invariant EP breaks.

**Fixed in all 10 multi-operand wrappers** — `dequantize_rotated`,
`fused_gemv`, `gather_gemv`, `rotate_x`, `rotate_weight_rows`, `quantize_rows`,
`dequantize_2b`, `fused_gemv_2b`, `gather_gemv_2b`, `grouped_gemm_2b` — via a
new `mistralrs-quant/src/qtip/device_guard.rs`. Its core is a **pure function
over `DeviceLocation`**, so the ordinal comparison is unit-tested without a
GPU. Error now names both ordinals:
`"QTIP gather gemv CUDA: x_rotated is on cuda:0 but blocks is on cuda:1"`.
8 tests; mutation A (`if lhs == rhs` → `if true`, i.e. back to a kind check)
is caught by `kind_check_would_pass_where_ordinal_check_fails`.

### 1.2 🔴→🟢 No NVLink peer access — every cross-GPU copy staged through host RAM

Neither candle nor cudarc ever calls `cuCtxEnablePeerAccess` (CUDA Programming
Guide, *Multi-GPU Systems*: peer-to-peer copies are staged through the host
without it). That penalty is paid **today**, by the existing layer-wise
pipeline map's hidden-state handoff, not just by EP.

New `mistralrs-quant/src/cuda_peer.rs`: `enable_peer_access(&[Device])` →
`PeerAccessReport` (per ordered pair: enabled / already-enabled / unsupported),
with a non-cuda stub. Symbols verified against the **actual** cudarc 0.19.4
source in the registry, not from memory — `cuDeviceCanAccessPeer`
(`driver/sys/mod.rs:10287`), `cuCtxEnablePeerAccess` (`:10133`),
`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704` (`:6605`). Reached through
candle's re-export `candle_core::cuda::cudarc`; no new dependency.

**It now has a call site** (the subagent flagged it would otherwise be dead
code, which would have been the thirteenth "wired but never invoked" case):
`DeviceMapSetting::into_mapper` calls it for every multi-CUDA map and **logs
the report at WARN when any pair is not peered**. A run quietly staging every
cross-GPU copy through host RAM looks identical to a healthy one except in the
numbers — that is the failure class this repo keeps paying for.

⚠️ **Not compile-verified under `--features cuda`**: cudarc's `build.rs` panics
on darwin (`nvcc --version` failed). The driver-API half was type-checked
against the real crate in a throwaway probe; the candle-side half is
read-verified only. **This is the first thing the rental must check.**

### 1.3 🟢 The offline `tid2eid` expert→device distribution — computed, and wired

wave44-BV §3.2: V4's first `num_hash_layers` (3) layers route by
`gate.tid2eid`, a fixed `[vocab, top_k]` table, so their expert load is a
closed-form property of the tokenizer — "free to check and nobody has".

- `arc-engine/src/expert_placement.rs`: `expert_loads_from_tid2eid`,
  `expert_loads_weighted` (measured token histogram),
  `plan_placement(cfg, loads)` — **this is what makes `arc-engine`'s `ep_size`
  do something**, and `compare_to_contiguous` for the before/after.
- Runtime half: `deepseek4.rs::build_expert_parallel_plan` derives a balanced
  placement from the layer's own `tid2eid` when `ARC_EP_PLACEMENT=balanced`,
  contiguous otherwise. `MoeGate` is now built **before** the experts so the
  table is available to place them.
- On a deliberately skewed table the planner removes **>99% of the imbalance**
  (contiguous ratio >1.4 → planned <1.001) `[tested]`, and on a uniform table it
  reports **no improvement available** — the D12 pair that keeps the first
  assertion from passing on any input.

**Not yet run against the real published table.** It needs
`gate.tid2eid` from the artifact, which is on the box, not here.

### 1.4 🟢 Balancedness is now measurable, not assumed

`ExpertParallelPlan` ships `BalancednessCounter`, which records **two**
quantities per rank, because they answer different questions:

- **tokens per rank** — the compute term, comparable to TensorRT-LLM's
  published 1.564 imbalance ratio at EP=32;
- **distinct (layer, expert) pairs per rank** — the **weight-byte** term, which
  is the one Arc is bound by and the one the +5% bound below applies to.

Off unless `ARC_EP_BALANCE=1`: recording reads `topk_ids` back to the host, and
a D2H sync in the routing path is incompatible with CUDA-graph capture and with
overlapping the all-to-all (wave44-BV §4.3).

---

## 2. 🔴 A CORRECTION TO wave44-BV §3.2 — the imbalance bound

The design note tabulates a **per-N spread**: +4.8% / +4.6% / +4.2% at
N=2/4/8, B=128. **That spread does not follow from the definition the note
states.** Once coverage exceeds `E/N` — i.e. once a rank's whole slice can be
woken — the hottest rank reads `E/N` experts against a balanced `distinct/N`:

```
(E/N) / (distinct/N) = E / distinct        ← N cancels
```

Recomputed for V4's 256 experts, top-6 `[computed, tested]`:

| B | coverage | **bound, every N** | note's figure |
|---|---|---|---|
| 16 | 31.6% | +100% (N=2) / +217% (N=8) | +87% / +136% |
| 64 | 78.1% | **+28.1%** | +23…27% |
| **128** | **95.2%** | **+5.05%** | +4.2…4.8% |
| 256 | 99.8% | **+0.23%** | +0.2% |

The corrected bound is **slightly worse** than claimed and the per-N variation
is an artefact. **The conclusion is unchanged and is the part that matters:**
at the batch sizes where EP is worth doing, routing skew cannot cost more than
~5% of the weight-read term ⇒ **ship stage 1 without EPLB, instrument for it.**
The test asserts the corrected numbers, so a future edit that "restores" the
note's figures fails.

*(This is the measured-vs-predicted answer the brief asked for at the level it
can honestly be given today: the +4.2% prediction is refuted as an arithmetic
claim. Whether real routing hits the bound is a hardware question and the
counter now exists to answer it.)*

---

## 3. Stage 1 — EP=2, what was built

### 3.1 The contract

For token `t`, slot `j`, EP=1 computes `y[t] = Σ_j w[t,j]·E_{g[t,j]}(x[t])`.
Rank `r` computes the partial `y_r[t] = Σ_{j : owner(g)=r} w[t,j]·E_g(x[t])`,
and the combine is `y = Σ_r y_r`. Every slot is owned by **exactly one** rank,
so the sum reproduces EP=1 up to float re-association.

Unowned slots are **not removed** — they are pinned to local expert 0 with
weight exactly `0.0`. That keeps every tensor shape identical across ranks
(which is what lets the combine be a plain sum) and costs compute, **not weight
bytes**: local expert 0 is essentially always already resident at the batch
sizes where EP matters. Named here because it is a real, deliberate
inefficiency and stage 3 should remove it.

### 3.2 New code

| Piece | Location |
|---|---|
| `ExpertPlacement` (contiguous / explicit / LPT-balanced), `ExpertParallelPlan`, `localize`, `BalancednessCounter`, the imbalance bound | `mistralrs-core/src/moe/expert_parallel.rs` (new) |
| `ExpertSubset` — which experts a rank loads; slices dim 0, or **refuses by name** | `mistralrs-quant/src/distributed/layers.rs` |
| `QuantMethod::select_experts` + impls for `QtipLayer` **and `Qtip2bLayer`** (V4's serving rung) | `mistralrs-quant/src/lib.rs`, `qtip/mod.rs`, `qtip/bitshift.rs`, `qtip/grouped.rs` |
| EP in the MoE forward + the deferred-UQFF-slice guard | `mistralrs-core/src/moe/experts.rs` |
| `ep_size` in the model config, `effective_ep_size()`, `build_expert_parallel_plan`, `tid2eid_expert_loads` | `mistralrs-core/src/models/deepseek4.rs` |
| `IsqModel::apply_pending_expert_parallel_slice` + its call after load | `pipeline/isq.rs`, `pipeline/normal.rs` |
| Offline planner | `arc-engine/src/expert_placement.rs` (new) |
| Ring collective algebra + the tests that would have caught the ring bug | `mistralrs-quant/src/distributed/mod.rs` |

### 3.3 `ep_size` now does something

`arc-engine/src/deepseek_v4.rs:48`'s `ep_size` was deserialized and never read
— the twelfth "wired but never invoked" case in this repo. Now:

- the **model** config (`mistralrs-core`) gained `ep_size` (serde default 1),
  so the published `config.json` field reaches the forward path;
- `effective_ep_size()` lets `ARC_EP_SIZE` override it without editing a
  published config;
- `build_expert_parallel_plan` **refuses** when `ep_size != comm.world_size()`
  rather than silently degrading — wave44-BV §1.6, "a device list on the wrong
  kind of run is an error, not a silent no-op";
- `arc-engine`'s field drives the offline planner `[tested]`.

### 3.4 Two refusals that exist to prevent silent wrongness

1. **`ExpertSubset::require_all`** — expert formats whose leading dimension
   cannot be sliced (AFQ, stacked blockwise-FP8, MXFP4, per-expert INT4)
   **error at load** naming the format, instead of loading all 256 experts on
   every rank. "EP appeared to work and bought nothing" is a capacity bug that
   looks exactly like a healthy run.
2. **`pending_expert_subset`** — a UQFF artifact holds **every** expert, so the
   slice cannot happen at construction. `MoEExperts` records the subset and
   **`forward` refuses to run** until `apply_pending_expert_parallel_slice`
   narrows the deserialized stacks. Loud failure, not a quiet 2× memory bill.

---

## 4. Tests, and how each was mutation-proved (DOCTRINE D12)

**41 new tests. All green. Every EP claim below was proved to fail under a
targeted break, and the break reverted.**

### 4.1 The collective-correctness tests — the ones that would have caught the ring bug

`ring_algebra` (`distributed/mod.rs`), available **without** the `ring`
feature so CI actually runs it:

- `single_exchange_ring_is_a_complete_sum_only_at_world_size_two` — asserts the
  collective's **result equals the true sum** at N=2, and asserts it **does
  not** at N=4/8, naming what it computes instead (`self + left neighbour`).
- `the_construction_guard_agrees_with_the_algebra` — for every world size 2..16,
  the guard `RingComm::from_device` gates on must agree with what the collective
  actually computes. **A guard that disagreed with the algebra is exactly how
  the original bug survived review.** `RingComm::from_device` now calls this
  predicate instead of carrying its own inline `> 2` check.
- `completeness_requires_world_size_minus_one_exchanges` — pins the *shape* of
  any future fix: the guard may only be relaxed by adding exchange rounds (or
  moving to NCCL), never by widening the accepted sizes.
- `an_expert_parallel_combine_that_drops_a_rank_is_detectable` — the EP-shaped
  version, asserting the dropped result stays **finite and plausible**, which is
  why an equality assertion is required and a smoke test is not.

> ⚠️ **Honest limit.** These test the *algebra* of the collective, not the TCP
> socket path: `ring_ops` uses a process-global `OnceLock` pair of streams, so
> two ranks cannot coexist in one test process. The real NCCL/ring transport is
> unexercised until hardware.

### 4.2 The EP=2 equivalence tests, and the three mutations

| Test | What it pins |
|---|---|
| `ep2_reproduces_ep1_output` | combined EP=2 == EP=1 |
| `ep2_reproduces_ep1_under_a_permuted_placement` | same, with rank 0 owning experts {1,3} — local 0 ≠ global 0, so the remap is exercised non-trivially |
| `ep4_reproduces_ep1_output` | one expert per rank |
| `dropping_one_ranks_partial_changes_the_answer` | the combine is load-bearing |
| `a_placement_that_is_not_a_partition_changes_the_answer` | double-counting a rank is detectable |
| `skipping_the_global_to_local_remap_changes_the_answer` | the remap is load-bearing |
| `a_pending_uqff_expert_slice_refuses_to_run` | the UQFF guard fires and clears |

**Fixture discrimination:** every expert computes a *different* function
(expert `e` has gate pre-activation `e+1`). With identical experts — the shape
the pre-existing fixture used — routing to the wrong expert would be invisible.

**Mutations actually applied, run, and reverted:**

| # | Break | Tests that caught it |
|---|---|---|
| **A** | drop the global→local remap (`local_idx[g] = 0`) | `localize_masks_unowned_slots_and_remaps_owned_ones`, `ep2_reproduces_ep1_output`, `ep2_reproduces_ep1_under_a_permuted_placement` |
| **B** | never mask unowned slots (`mask[g] = 1.0` always) | those three **plus** `masks_partition_the_routing_slots_across_ranks` and `ep4_reproduces_ep1_output` |
| **C** | combine keeps unmasked weights (discard `localize`'s weights) | `ep2_reproduces_ep1_output`, `ep4_reproduces_ep1_output`, `ep2_reproduces_ep1_under_a_permuted_placement` |

Note **`ep4` survived mutation A**: with one expert per rank the local index is
always 0, so the remap is correct by coincidence there. Recorded because it is
the kind of accidental pass that makes a suite look stronger than it is — the
EP=2 tests are the ones carrying that assertion.

---

## 5. 🔴 What is NOT done — the honest gap

1. **No hardware number.** Per-user tok/s at B=128 on 2 GPUs vs the 1-GPU
   baseline of **1.09 `[measured, wave51-CB]`** is **not measured**. §6 is the
   exact script.
2. **NCCL is still not compiled into the production build**
   (`deploy/modal_b200.py:26` is `--features 'cuda flash-attn'`). Stage 1 runs
   the combine through whatever `Comm` exists; with the shipped build that is
   `Comm::Dummy`, whose `sum_all_reduce` is `Ok(xs.clone())` — **so a 2-rank
   run without `--features nccl` (or `ring`) would silently drop the other
   rank's partial.** The refusals in §3.4 do not cover this; the world-size
   check does (`ep_size != comm.world_size()` ⇒ error), because Dummy reports
   `world_size = 1`. **That check is the only thing standing between a
   mis-built binary and a fluent, wrong model. Verify it on the box first.**
3. **Two NCCL landmines from wave44-BV §1.5 are still live and will confound a
   measurement if not handled:**
   - `pipeline/normal.rs:374-380` — when `use_nccl`, the mapper is overwritten
     with `DummyNccl` and **auto device mapping is skipped entirely**;
   - `attention/mod.rs:232` — **silently forces `naive_sdpa` when NCCL is on.**
     Turning NCCL on therefore turns flash-attention **off**, so an EP=2 vs
     1-GPU comparison would be measuring two different attention kernels.
     **This must be neutralised or controlled for before any tok/s number is
     quoted.**
4. **Expert slicing is implemented for**: per-expert unquantized, stacked
   unquantized, the FP8-config-without-scales fallback, and — via
   `select_experts` — `QtipLayer` / `Qtip2bLayer` after UQFF deserialize.
   **Not** for AFQ, MXFP4, stacked blockwise-FP8, or per-expert INT4, which
   refuse by name.
5. **The UQFF slice is post-load**, so each card reads the full 74 GB artifact
   and then narrows to ~37 GB. Peak load-time memory is unchanged; steady-state
   halves. Fine on 80 GB H100 (74.2 GB fits, RUN-161) but it is **not** a
   capacity win at load, and on a 2×H100 box the margin is thin.
6. **`cargo check --features cuda` has never run on this branch.**

---

## 6. The rental, and the one number

**Box: 2× H100 SXM5 on one node** (the fleet has no 2×H200; SXM5 is the
NVLink/NVSwitch form factor, so the latency question is still answered
honestly, and an H100 result is the more conservative fleet claim).

**Hard preflight, first command, before the clock matters:**

```bash
nvidia-smi topo -m     # GPU0<->GPU1 must read NV#. SYS/PHB/PIX ⇒ STOP, re-provision.
```

Then, in order — each step gates the next:

```bash
set -euxo pipefail
cd /root/arc && git fetch origin && git checkout feat/expert-parallel-ep2

# 1. The thing that has never compiled: CUDA + NCCL.
cargo build --release -p mistralrs-cli --features "cuda flash-attn nccl"
cargo test -p mistralrs-quant --features cuda device_guard cuda_peer -- --nocapture

# 2. Peer access on real silicon — the report must say every pair is peered.
#    (Needs the ~40-line examples/peer_access.rs; see §5.)
cargo run --release -p mistralrs-quant --features cuda --example peer_access

# 3. EP=1 control on THIS binary, so the comparison is same-binary.
ARC_EP_SIZE=1 <serve + wave34-BL sweep at B=128>

# 4. EP=2. The world-size check must pass, i.e. Comm must NOT be Dummy.
ARC_EP_SIZE=2 <2-rank launch> <same sweep>

# 5. Acceptance: greedy, same seed, same prompt, EP=2 vs EP=1 → token-identical.
# 6. ARC_EP_BALANCE=1 for one short run → measured imbalance vs the +5.05% bound.
```

**The one number: decode tok/s per user at B=128, EP=2, against 1.09
`[measured]` on one card and a 126 tok/s/user saturated floor `[computed,
wave44-BV §2.6].`** Secondary: per-collective latency (the single most
uncertain quantity in the design — no intra-node NVLink low-latency figure
exists in any vendored reference), and the measured imbalance ratio.

**Cost:** 2×H100 SXM5 at **$9.22/hr**. Steps 1–2 ≈ 30 min. Steps 3–6 ≈ 1.5 h
including the 74 GB pull. **≈ 2 h ≈ $18.50** against a **$127.16** balance.

**Refutation, pre-committed:** if EP=2 measures per-user **below ~2.0 tok/s**
at B=128 (i.e. under ~2× the single-card 1.09), the bottleneck is not expert
weight bytes and the whole EP thesis needs re-deriving before spending on
EP=4/8. If `nvidia-smi topo -m` says PCIe, the run is void.

---

## 7. Changes in this PR

Stage 0: ordinal guards in all 10 QTIP CUDA wrappers; `cuda_peer` +
its call site in `device_map.rs`; the offline `tid2eid` planner; the
balancedness counter.
Stage 1: `ExpertPlacement` / `ExpertParallelPlan` / `ExpertSubset`;
`select_experts` on both trellis rungs; EP in the MoE forward; `ep_size` wired
from config and env; the deferred-UQFF-slice guard; the ring collective
algebra and its tests.

Deliberately **not** touched: `mtp_pipeline.rs`, `Sequence::cache_bucket_len`,
`kv_sharing/`, `supports_paged_attention`. Upstream `mistralrs-*` files were
**not** mass-reformatted (fork policy, `fab114fe3`): the incidental rustfmt
churn in `deepseek4.rs` (341 lines) and `distributed/layers.rs` (854 lines) was
reverted, leaving only semantic hunks.

Green: `cargo check --workspace`; `cargo test -p mistralrs-core -p
mistralrs-quant -p arc-engine`; the scoped clippy lane.
