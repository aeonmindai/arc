# THE GPU LADDER, REORDERED BY THE CENSUS

**Session 8, 2026-08-19. Needs Jish's approval before any GPU is touched.**
Companion to `CENSUS_SESSION8.md`. Every row cites a measured number or is labelled an estimate.

---

## What the census changed about the plan

| I previously believed | The census established |
|---|---|
| Chunked prefill is a quick TTFT win | **It is NEGATIVE** — ~3× worse until the expert gather is fixed (§3.1) |
| Fixing MTP acceptance is worth ×5.53 | **×1.55** against the measured base. It is a multiplier on a 6× problem |
| The aggregate ceiling is a batching-policy problem | **64× KV over-retention** + a ragged fix that never reached V4 |
| The trellis ceiling is one re-bake away | **≤1.41 inst/wt is not established**; and one of my three "free" levers does not exist |
| Prefix caching works by default | **The default command line silently disables it** |
| The 2-bit format is our speed story | **It is our CAPACITY story.** bpw cancels out when instruction-bound |

---

## Rung 0 — costs ZERO GPU, blocks everything above it

**None of this needs a card. All of it must land before GPU money is spent, because each one is
either a prerequisite instrument or a measurement we currently cannot interpret.**

| # | item | why it gates the rung above | cost |
|---|---|---|---|
| 0.1 | **Per-position MTP acceptance telemetry** | `MtpAcceptance.accepted` is a scalar (`mtp_pipeline.rs:977`). Without it, `p₁≈p₂≈0.54` (distribution mismatch) and `p₁≈0.75,p₂≈0.35` (chain compounding) are indistinguishable — **any acceptance work is a guess, and guesses cost rental** | 0.5 sess, 0 GPU-h |
| 0.2 | **Token-level prefix-cache hit rate** | `engine/logger.rs:110` is request-level and reads **100%** when one 32-token block of 2048 is reused. Counters exist (`kv_sharing/radix.rs:167,177`) with zero readers ⇒ **no Arc run can currently report whether the cache hit**, so every prefill measurement is uninterpretable | 0.25 sess, 0 GPU-h |
| 0.3 | **Non-degenerate V4 test fixture** | The live fixture is **all-zeros** (`synthetic_load_smoke.rs:521`) and `v4_e2e.rs` is `#[ignore]`d on V3-era shapes ⇒ every "outputs unchanged" assertion is **vacuous**. Two guards already passed their own mutations this session | 1 sess, 0 GPU-h |
| 0.4 | **Graph-mode mask wiring** | `set_graph_mode_mask` (`layers.rs:2284,2294`) is **called from nowhere**; the unwritten tail is attended as **zero-padding, which takes softmax weight**. **Every quality number on the CUDA-graph arm is invalid until this lands** | 1 sess, 0 GPU-h |
| 0.5 | **SASS census of the trellis geometries** | Converts the entire §1.3 ladder from estimate to measurement. Method validated to 1.3% vs ncu. Hand-counting undercounts by **2.05×** | 0.5 sess, 0 GPU-h (CI runner) |
| 0.6 | **Merge #133 / decide the serving rung** | The shipped artifact **cannot reach the grouped GEMM at all** — above 8 tokens it dequantizes every distinct expert to BF16 in HBM (16 traffic units vs 1). **Every throughput number assumes a kernel we do not ship** | 0.5 sess, 0 GPU-h |

**Rung 0 total: ~3.75 agent-sessions, ZERO GPU hours, zero dollars.**

---

## Rung 1 — the cheapest real wins. **~2.5 GPU-h ≈ $12**

Ordered by measured-value-per-GPU-hour. Every one is code that already exists.

| # | item | expected | evidence | GPU-h |
|---|---|---|---|---|
| 1.1 | **Point ragged decode at V4** | **18.83 → 61.22 tok/s, ×3.25** | Measured. Gated behind unset `ARC_V4_XS_PER_SEQ`; the merged fix `64b3ff379` touched only a file V4 never reaches | 0.5 |
| 1.2 | **Stop the default from disabling the prefix cache** | prompt reuse turns on at all | `--pa-cache-type` unset ⇒ TurboQuant ⇒ prefix caching disabled wholesale (`engine/mod.rs:249-259`) | 0.25 |
| 1.3 | **Measure PR #145** (cache visible at end-of-prefill, not end-of-generation) | K concurrent requests on a shared prompt: `K × 5.90 s` → `5.90 s + (K−1)ε` | Landed, tested, unmeasured | 0.5 |
| 1.4 | **Kill the 430 provably-dead launches** | −430/token, no new CUDA | `dsv4_mhc.rs:323` re-casts what `:266` already cast (86); `:335,336`→`:376,377` round-trip and are discarded (344) | 1.0 |
| 1.5 | **MTP depth 2→3** | +9% (1.84 → 2.00 tok/step) | One flag. Matches SGLang's V4 recipe. Ceiling at current p is +19% | 0.25 |

**Rung 1 total: 2.5 GPU-h ≈ $12.** Every item is switched-off or unmeasured code, not new work.

---

## Rung 2 — the aggregate wedge. **~6 GPU-h ≈ $29**

| # | item | expected | GPU-h |
|---|---|---|---|
| 2.1 | **Per-layer windowed KV retention** | **361 MB → 5.65 MB per sequence (64×)**; at B=32, **11.6 GB → 0.18 GB.** Unblocks batch width and removes the exclusive-card requirement | 4 |
| 2.2 | **Aggregate sweep B=1→256** with 2.1 + 1.1 live | the actual capacity number, which we have never measured with either fix on | 2 |

⚠️ **2.1 must NOT use `KvCache::Rotating`** — six blockers, two of which defeat the purpose: it turns
**ragged batching off** (undoing 1.1) and **kills the CUDA-graph decode arm**. Use `raw_prefix` +
a `first_cached` base on `SingleCache`.

---

## Rung 3 — the keystone. **~8 GPU-h ≈ $39**

| # | item | expected | GPU-h |
|---|---|---|---|
| 3.1 | **QTIP MoE expert gather → a real batched GEMM** | **71.3% of an N=128 prefill step.** Also the gate on chunked prefill, and on the whole prefill story | 8 |

**This is the single highest-value item in the program and it unlocks two others.** Neither of the
two current paths is a batched GEMM (a no-dedup GEMV with 3.15× redundant reads, and a host-synced
per-expert dequantize loop), which is why the tuning knob between them does nothing.

**Only after 3.1:** chunked prefill becomes affordable at C≈256 (must be a multiple of the block
size). Before 3.1 it is ~3× negative. **Do not reorder these two.**

---

## Rung 4 — the format decision. **~1 GPU-h + a re-bake ≈ $5–20**

Blocked on Rung 0.5 (the SASS census). **Do not spend the re-bake on `sum2`.**

| candidate | est. X | est. B=256 | note |
|---|---|---|---|
| `sum2` | 15.85 **measured** | 1,477 | the rejected number |
| K8/V4/L12, 32 KB bf16 LUT | ~4.7 est | ~4,977 | LUT fits shared; V=4 = one `LDS.64` per mma B-operand pair |
| + row-scale hoist | ~3.7 est | ~6,321 | the one surviving "free" lever |

⚠️ **Two censuses disagree on whether the re-bake gets cheaper or stays flat at K8/V4/L12.
Unresolved. The recommendation depends on it.**

---

## What NOT to do — each has a reason, not a preference

| do not | because |
|---|---|
| **Chunked prefill before 3.1** | ~3× negative; chunking multiplies steps, the gather is billed per step |
| **Chase MTP acceptance before 0.1** | you cannot tell distribution mismatch from chain compounding; you would be renting a card to guess |
| **Chase tree speculation** | **vLLM is chain-only too**, and SGLang's V4 recipe is `topk=1` — also a chain. Not table stakes |

### Two Arc systems whose SCOPE is now known — these are not verdicts (D1)

**D1: a scoping result is never a verdict.** Both of the following are novel Arc systems. Neither is
ranked down; each has a measured bound and a named condition under which it becomes worth building.

| system | measured bound today | what would make it pay |
|---|---|---|
| **EPLB / expert placement** | On **one** card there is one rank owning all 256 experts, so cross-rank balancing has nothing to balance. At B=128, ~244 distinct experts × 6.29 MB = **1.53 GB/layer against a 60 MB L2 — 25× over**, so no permutation improves reuse. Arc already hard-bounds imbalance at **+5.05%** (`expert_parallel.rs:18-19`) | **It pays the moment `ep_size > 1`.** EP is on master (`deepseek4.rs:2211`) but env-gated with no CLI flag, and **Arc has never run a forward pass on two GPUs**. Build the rebalancer *when* the second card runs — the reference gains (1.49× prefill / 2.54× decode) are multi-node numbers and are only reachable there |
| **`expert_affinity`** | Zero call sites; presumes an oracle for *next-step* routing that does not exist; headroom is **4.8%** because 243.7 of 256 experts (95.2%) are already distinct at B=128 | Two conditions change it: (a) a **smaller effective batch**, where distinct-expert count falls and affinity has room; (b) a **real predictor** replacing the oracle — the closed-form `tid2eid_expert_loads` (`deepseek4.rs:2185`) already predicts load from the tokenizer with zero GPU for V4's 3 hash-routed layers. **Extending that to the other 40 is the thing to build**, and it is Arc-only — both references need a profiling pass |
| **Copy SGLang's 8192 chunk** | no-op at N=2048, and their per-token cost is **29.5× lower**. Arc's equivalent is ~277 |
| **Use `KvCache::Rotating` for 2.1** | undoes 1.1 and kills the graph arm |
| **Quote their per-GPU totals as comparable** | `(input+output)/GPU` at 8192/1024 = a **9× multiplier**, on top of a TP≥4 divisor and a 2.97× spec multiplier |

---

## Budget

| rung | GPU-h | ≈ cost | gives |
|---|---|---|---|
| 0 | **0** | **$0** | the instruments; without them the rest is unmeasurable |
| 1 | 2.5 | $12 | ×3.25 aggregate, prompt reuse on, −430 launches |
| 2 | 6 | $29 | 64× KV, and the first honest capacity number |
| 3 | 8 | $39 | the keystone; unblocks prefill and chunking |
| 4 | 1 + bake | $5–20 | the format decision, on measured SASS |
| **total** | **~17.5** | **≈ $85** | |

**Rung 0 first, and it is free.** Rungs 1–2 are ~8.5 GPU-h / $41 and contain every measured win we
already own but have never switched on.

---

## The claim to make when this is done

**Not** "Arc is faster than SGLang" — on per-user latency we are behind, and their headline number is
inflated ~9× by metric definition before any engineering.

**Instead:** *DeepSeek-V4-Flash, 284B parameters, served from a single H200.* At FP8 the model is
284 GB and at MXFP4 it is 151 GB — **neither fits.** At Arc's 2.09 bpw it is 74 GB, leaving ~65 GB
for KV. **No competitor format can make that claim at all.** That is capacity per node, it is the
wedge, and the format has already won it.
