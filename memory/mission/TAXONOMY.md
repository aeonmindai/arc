# Arc SYSTEM TAXONOMY — the names we own

**Set by Jish, 2026-08-17:** *"we're making our own systems here: ArcInfer,
ArcAttention, QTIP, TurboQuant... any future / additional model / architecture /
gpu support should be integrated into our named systems... name everything that
deserves a name and no subsystem should be left without an absolute parent
system name. This helps us understand what we're working on."*

**The hard rule: every subsystem resolves upward to a named parent, and every
parent resolves to Arc.** If something has no parent, either the parent set is
incomplete or the thing is misplaced. Resolve it; do not leave it dangling.

**Status of names.** Jish seeded four: **ArcInfer**, **ArcAttention**, **QTIP**,
**TurboQuant**. The rest are named here by the agent, per Jish's standing
instruction that agents decide rather than escalate. QTIP and TurboQuant keep
their paper names by Jish's explicit decision — see *Attribution* at the end.

**This document names systems. It does NOT rename code.** No crate, symbol, env
var or feature flag changes in this pass — six workstreams are live and a mass
rename would be reckless. Migration proposal is the last section.

---

## 🔵 SESSION-8 CORRECTIONS (2026-08-19) — four claims in this file were wrong

Each is fixed **in place** below; they are listed here only so a reader who
already quoted the old text knows to re-read.

1. **Expert parallelism IS on master, and it has TWO doors.** The old
   `ArcMoE/EP 🔵 "NOT ON MASTER"` row was stale — PR #89 merged (`610c4506b`),
   feature `fce33ae22`. Its replacements were wrong in both directions: EP is
   neither "env-gated only" (`ep_size` is a serde `config.json` field,
   `deepseek4.rs:317-318`) nor "a config field, not an env var" (`ARC_EP_SIZE`
   overrides it in `effective_ep_size()`, `:371-375`). **Both doors are real.**
   The gap is a missing CLI/server flag, not a missing feature. *(§2.5)*
2. **`CrossPrefixMeter` is not dead** — it records in production; only its
   readout is test-only. *(§2.2)*
3. **head_dim 512 is not the blocker — 448 is.** 512 compiles; the failure is
   `vec_size 14 % 8 ≠ 0` at 448. *(§2.3)*
4. **The prefix-cache-disable is narrower than written** — CUDA **and**
   PagedAttention **and** standard layout **and** head_dim 128. V4 is MLA, so it
   does **not** apply to the flagship. *(§4.2)*

**Arc has never run a forward pass on more than one GPU.** EP, the pipeline
shard and every multi-GPU claim in this file are code-complete and unrun. Write
"code-complete, never run", never "supported".

**Companions, both on branch `docs/census-session8` until it merges:**
`memory/mission/CENSUS_SESSION8.md` (the complete Arc/SGLang/vLLM census, zero
GPU hours) and `memory/mission/LADDER_POST_CENSUS.md` (the GPU ladder reordered
by it). **Read the census before quoting any ceiling out of this file.**

> 📁 **WHERE THE MISSION RECORD LIVES.** `00_RESUME_HERE.md`, `FACTS.md`,
> `KERNEL_RULES.md`, `COMPETITIVE_TEARDOWN.md` and this file are the record. The
> **live, agent-writable copies** are in the agent memory directory
> (`~/.claude/projects/-Users-jish-Documents-GitHub-arc/memory/mission/`); the
> copies in this repo are the **durable mirror**, synced **2026-08-19**. They had
> already drifted once — the repo's TurboQuant box was correct and the live one
> was a revision behind, which is exactly the failure this note exists to make
> visible. **When they disagree, reconcile before quoting either; re-sync in the
> same PR as any mission-record change.**

---

## THE TREE

```
Arc — the inference engine
├── ArcServe      the front door: how a request reaches the runtime
├── ArcInfer      the runtime: request → tokens                    [Jish]
│   ├── ArcSched      the serving loop, admission, batching policy
│   ├── ArcKV         all key/value memory
│   ├── ArcAttention  attention math + kernel dispatch             [Jish]
│   ├── ArcSpec       speculative decoding
│   ├── ArcMoE        mixture-of-experts serving
│   ├── ArcGraph      GPU-autonomous decode (CUDA graphs)
│   ├── ArcSample     token sampling
│   └── ArcBoost      serving-side quality (training-free)
├── ArcModels     architecture support — where new models land
├── ArcQuant      compression
│   ├── QTIP          trellis weight quantization                  [Jish]
│   ├── TurboQuant    KV compression                               [Jish]
│   └── ArcBake       the offline quantize pipeline
├── ArcKernels    the GPU kernel substrate — where new GPUs land
│   └── ArcTarget     arch matrix + dispatch (SM80/89/90/100)
├── ArcFormat     the model artifact: UQFF + the serving overlay
├── ArcLab        measurement: profiler, benchmarks, ops tooling
└── ArcGate       correctness gates + release discipline
```

Two systems are deliberately **substrates**, consumed by several parents:
`ArcKernels` (both ArcInfer and ArcQuant call into it) and `ArcGate`
(cross-cutting). Each subsystem still has exactly **one owning parent** — the
one listed — so the rule holds; the arrows out are consumers, not co-owners.

---

## 1. ArcServe — the front door

| Subsystem | Does | Lives in |
|---|---|---|
| ArcServe/HTTP | OpenAI-compatible routes, streaming, web UI | `mistralrs-server-core/`, `mistralrs-server/`, `mistralrs-web-chat/` |
| ArcServe/CLI | the `arc` binary: `run`/`serve`/`bench`/`validate` | `arc-cli/src/main.rs` |
| ArcServe/SDK | Rust façade + Python bindings | `mistralrs/`, `arc-engine/src/lib.rs`, `mistralrs-pyo3/` |
| ArcServe/MCP | Model Context Protocol client | `mistralrs-mcp/` |
| ArcServe/Search | web-search + custom-tool sidecar, BM25 rerank | `mistralrs-core/src/engine/search_request.rs` |

## 2. ArcInfer — the runtime

### 2.1 ArcSched — the serving loop *(named here; was unnamed)*

| Subsystem | Does | Lives in |
|---|---|---|
| ArcSched/Loop | owns pipeline+scheduler+prefix-cache locks; splits each iteration into **decode then prompt** sub-steps (never merged, so profiler subtrees stay separate); `step_catching_panics` so one bad batch fails alone | `mistralrs-core/src/engine/mod.rs` |
| ArcSched/Admit | tokenize, build `Sequence`s, consult the prefix cache, hand to scheduler | `mistralrs-core/src/engine/add_request.rs` |
| ArcSched/Bucket | `DefaultScheduler`: FCFS + length bucketing; exactly one bucket runs per step because the dense batch cache is built once. **`select_running_bucket` is the coalescing policy** (`COALESCE_PAYBACK_STEPS = 256`) that fixed the measured "32 running / 32 waiting" H200 stall | `mistralrs-core/src/scheduler/default_scheduler.rs` |
| ArcSched/Paged | `PagedAttentionScheduler`: `max_num_seqs` cap, block-hash prefix tracking, starvation at `WAITING_TIMEOUT = 64` | `mistralrs-core/src/paged_attention/scheduler.rs` |
| ArcSched/Sequence | `Sequence`/`SequenceGroup` state machine: tokens, logprobs, stop reasons, multimodal ranges | `mistralrs-core/src/sequence.rs` |
| ArcSched/Telemetry | `IntervalLogger` — throughput, prefix hits, running/waiting counts | `mistralrs-core/src/engine/logger.rs` |
| ArcSched/Chunked ⚠️ | Sarathi-Serve chunked prefill — **Tier A, NOT wired into the loop** | `arc-engine/src/sarathi.rs` |
| ArcSched/Affinity ⚠️ | `AffinityBatcher`, groups requests by predicted routing — **NOT wired** | `arc-engine/src/expert_affinity.rs` |

### 2.2 ArcKV — key/value memory *(named here; was unnamed)*

`mistralrs-core/src/kv_sharing/mod.rs` says in its own doc: *"Internal name only
— the public/product name is deliberately not decided here."* **This decides
it: ArcKV/Share.**

| Subsystem | Does | Lives in |
|---|---|---|
| **ArcKV/Share** | radix-tree KV sharing, generic over key symbol (token ids *or* block hashes) and payload | `mistralrs-core/src/kv_sharing/mod.rs`, `radix.rs` |
| ArcKV/Share/Evict | `EvictionScorer`; `LruScorer` baseline and **`ValueAwareScorer`** — recompute cost × reuse probability × staleness. The named differentiator: every SGLang policy is a function of time or hit count only | `kv_sharing/evict.rs` |
| ArcKV/Share/Content | `ContentDigest`/`Fnv1a128`/`ContentIndex`, `CrossPrefixMeter` measuring reuse the tree *misses*; refuses causally-invalid cross-prefix hits. ⚠️ **It RECORDS in production — only the READOUT is test-only.** An earlier "`CrossPrefixMeter` is dead" claim is **wrong** and must not be repeated: a metric with no reader is not a metric that does not run | `kv_sharing/content.rs` |
| ArcKV/Share/Layout | `KvElemType` + `KvBlockLayout` — element width is a **per-block** property, so BF16 and FP8 blocks coexist with exact byte accounting | `kv_sharing/layout.rs` |
| ArcKV/Paged | `BlockPool` (O(1) free list), `BlockHash` (chained content hash), `KVCacheManager`, `CacheEngine`, `PagedCacheType`, `EncoderCacheManager` | `mistralrs-core/src/paged_attention/` |
| ArcKV/Dense | `KvCache`/`NormalCache`/`SingleCache`/`RotatingCache`/`HybridCache`; clone-in/clone-out dense batch cache | `mistralrs-core/src/kv_cache/` |
| ArcKV/Xs | **`XsRollingCache`** — bounded rolling window of the V4 compressor input. At 8192 B/token/layer on 41 of 43 layers it, not KV, was capping batch size | `mistralrs-core/src/kv_cache/xs_rolling.rs` |
| ArcKV/Fp8 | `V4PackedK` — E4M3 code + block `amax` instead of a widened BF16 copy. Bit-exact (V4 is FP8-QAT, so this is a serialization change). **Opt-in: `ARC_V4_FP8_KV=1`** | `mistralrs-core/src/models/dsv4_kv_fp8.rs` |
| ArcKV/Prefix | `PrefixCacheManagerV2` over two substrates (token radix, paged tree) | `mistralrs-core/src/prefix_cacher.rs` |
| ArcKV/Segments 🔵 | **NOT ON MASTER — open in PR #90** ("segment tables: a KV read primitive that is a list of runs"). No `SegmentAllocator` type is in the tree today; the shipped "segment" vocabulary is `SegmentRef`/`SegmentDigest` in `content.rs`, which is *hashing, not allocation* | — |

### 2.3 ArcAttention *(Jish's name)*

| Subsystem | Does | Lives in |
|---|---|---|
| ArcAttention/Dispatch | `Sdpa::run_attention` — the one dispatch site. Order: **sinks first**, then CUDA flash, then CPU flash, then naive. Also `chunked_attention` and the "(1,1) placeholder mask ⇒ non-causal" guard | `mistralrs-core/src/attention/mod.rs` |
| ArcAttention/Flash | FA2/FA3 capability envelopes + runtime dispatch (FA2: mult-of-8 ≤256, softcap; FA3: {64,128,256}, no softcap) | `attention/backends/flash.rs` |
| ArcAttention/Sinks | per-head sinks; **`head_dim ∈ {64,80,96,112,128,192,256}` gate** | `attention/backends/sinks.rs` |
| ArcAttention/V4 | V4 hybrid: one online softmax over a union of key sets. ratio 0 → sliding-window+sink, 4 → CSA, 128 → HCA; `absorbed_mqa_decode` | `mistralrs-core/src/models/dsv4_attention.rs` |
| ArcAttention/Indexer | **Lightning Indexer** — per-query top-k over compressed K on CSA layers | `models/dsv4_indexer.rs`; kernels `arc-cuda-graph/src/cuda/flashmlasparse/` |
| ArcAttention/MLA | shared MLA for DeepSeek V2/V3 + GLM4-MoE-lite; absorbed decode | `mistralrs-core/src/mla/` |
| ArcAttention/Sage | SageAttention INT8-QK / FP8-PV | `mistralrs-quant/src/sage_cuda/`, `arc-engine/src/sage.rs` |
| ArcAttention/MoBA | block routing + sparse mask — Tier A | `arc-engine/src/moba.rs` |
| ArcAttention/FA4 🔵 | **NO KERNEL EXISTS ANYWHERE — not even an open PR.** Feasibility ladder + probe scripts only; its own README forbids landing Rust before a verdict ("a feature flag whose implementation is `unimplemented!()` is precisely the wired-but-dead debt the BACKLOG tracks"). This is the future home of a fused head_dim-512 attention path | `arc-tools/fa4/` |

> 🔑 **V4 cannot use FlashAttention, at any generation.** Sinks are set on all 43
> layers, so dispatch takes the sinks branch before flash is considered; and the
> sinks gate excludes head_dim 512, which is what V4 uses. V4 runs the **unfused
> matmul + `softmax_with_sinks`** path. Any doc implying otherwise is wrong.
>
> 🔵 **CORRECTED 2026-08-19 — "head_dim 512 is the blocker" is INVERTED.**
> **512 compiles.** The width that fails is **448**, on `vec_size 14 % 8 ≠ 0`.
> The dispatch-order sentence above stands on its own and is unaffected; what is
> retracted is the *kernel-instantiation* half — do not cite 512 as an
> instantiation ceiling, and do not plan work around removing a cap that is not
> there. **The 448 record is the real constraint.**

### 2.4 ArcSpec — speculative decoding *(named here; was unnamed)*

| Subsystem | Does | Lives in |
|---|---|---|
| ArcSpec/MTP | the real implementation: `MtpHiddenCapture` (side-channel so `h_proj` and `e_proj` get *different* signals), full-block vs fallback path, `verify_proposed`, `MtpAcceptance` bucketed by batch | `mistralrs-core/src/pipeline/mtp_pipeline.rs` |
| ArcSpec/DraftKV | per-sequence draft KV where slot `k` is an **absolute** position; `batch_draft_caches`/`split_draft_cache` fold and unfold per-sequence caches around one batched forward. ⚠️ master still rolls the shared cache to the batch **minimum** (cohort min-rollback); PR #92 removes it in favour of per-sequence advance, worth ~+31% by its own arithmetic | same file |
| ArcSpec/Generic | draft/target pipeline, `try_make_mtp_pipeline` routing | `mistralrs-core/src/pipeline/speculative.rs` |
| ArcSpec/Eagle | EAGLE-3 draft heads — Tier A | `arc-engine/src/eagle3.rs` |
| ArcSpec/MagicDec ⚠️ | long-context spec; draft on a sliding window — **not plumbed** | `arc-engine/src/magicdec.rs` |

> **Never quote `accept_rate`.** Measured session 8: at B=128 `accept_rate` holds
> ~0.43 (looks healthy) while `tok_per_step` collapses to **1.0558** — a
> saturated sequence drafts 0, contributing `proposed=0`, so the ratio flatters
> while the yield dies. **`tok_per_step` is the honest metric.**

### 2.5 ArcMoE *(named here; was unnamed)*

| Subsystem | Does | Lives in |
|---|---|---|
| ArcMoE/Experts | `MoEExperts` — gate-external unified experts layer; backends fused/fast/slow; TP shard + all-reduce; `swiglu_clamp` | `mistralrs-core/src/moe/experts.rs` |
| ArcMoE/Route | **per-model, no shared router**: grouped-noaux-tc top-k + `sqrtsoftplus` in deepseek3/4; separate routers in qwen3/glm4/phi3.5/mixtral | `mistralrs-core/src/models/*.rs` |
| ArcMoE/Gather | the **token cap** and fused-gather-vs-fallback policy per rung | `mistralrs-quant/src/qtip/gather_policy.rs`, `grouped.rs` |
| ArcMoE/TD | **TD-MoE** Tucker decomposition of stacked experts, installed as a post-ISQ hook | `arc-engine/src/td_moe.rs`, `td_moe_loader.rs`, `mistralrs-quant/src/td_moe_factored/` |
| ArcMoE/Sparse | dReLU activation sparsity stats — Tier A | `arc-engine/src/turbo_sparse.rs` |
| ArcMoE/AnyMoE | training-time MoE adapters over a dense model — orthogonal to serving | `mistralrs-core/src/amoe/` |
| ArcMoE/EP ⚠️ | ✅ **ON MASTER since PR #89 merged** (`610c4506b`; feature commit `fce33ae22`) — stage 0 + stage 1, EP=2. `Moe::new` calls `build_expert_parallel_plan` (`deepseek4.rs:2227`, called at `:2300`) → `MoEExperts::new_expert_parallel` (`:2308`). **TWO doors, both real; neither is the CLI.** (1) **`ep_size` is a deserialized `config.json` field** — `#[serde(default = "default_ep_size")]` at `:317-318`, default `1` at `:140` — so a checkpoint whose `config.json` carries `ep_size: 2` shards with **no environment variable set**. (2) **`ARC_EP_SIZE` overrides that field at run time** — `effective_ep_size()` at `:371-375` reads the env var first and falls back to the config field, and `ARC_EP_PLACEMENT=balanced` (`:2184`) picks the placement. **What is missing is a CLI/server flag** (`grep -rn 'ep_size\|expert_parallel' mistralrs-cli/ mistralrs-server-core/` → nothing), **not a config door.** Dark by default only because the published V4-Flash `config.json` carries `ep_size: 1` and `effective_ep_size() <= 1` returns `ExpertParallelPlan::single` (`:2236`). `ep_size != comm.world_size()` is an error, not a degrade (`:2238-2242`). Weight-read imbalance is hard-bounded at **+5.05%** at B=128 (`expert_parallel.rs:18-19`), which is why stage 1 ships without EPLB | `mistralrs-core/src/moe/expert_parallel.rs` (847 lines), `mistralrs-core/src/models/deepseek4.rs`, `mistralrs-quant/src/distributed/` |

> 🔴 **ARC HAS NEVER RUN A FORWARD PASS ON MORE THAN ONE GPU.** EP, the
> layer-wise pipeline shard, `SumAllReduce` and every multi-GPU number in this
> file are **code-complete and unrun**. At `world_size=1` the code still takes
> `Comm::Dummy(rank=0)` and `SumAllReduce` = clone, so *"the EP tests pass"*
> certifies the single-rank degenerate case only.
> ⇒ **Say "code-complete, never run." Never "supported."**
>
> **The corollary for EPLB / expert placement is a SCOPE, not a verdict (D21):**
> on **one** card there is one rank owning all 256 experts, so cross-rank
> balancing has nothing to balance, and at B=128 ~244 distinct experts ×
> 6.29 MB = **1.53 GB/layer against a 60 MB L2 — 25× over**, so no permutation
> improves reuse either. **It pays the moment `ep_size > 1`.** Build the
> rebalancer *when* the second card runs.
>
> ⚠️ **This row has now been wrong in BOTH directions — do not restore either.**
> *"NOT ON MASTER"* was stale by a merge (fixed in #140). #140 then replaced it
> with *"env-gated only … `ARC_EP_SIZE` … the only door"*, and a session-8 draft
> swung the other way with *"`ep_size` is a `config.json` field, **not** an env
> var"*. Both halves are wrong: the cited file contradicts each of them three
> lines from the other — `ep_size` is a serde field (`:317`) that
> `effective_ep_size` falls back to (`:374`) *after* consulting `ARC_EP_SIZE`
> (`:372`). **Both doors exist; the CLI is the one that does not.** Before
> marking a subsystem 🔵 or writing "env-only"/"not an env var", (a) `git
> merge-base --is-ancestor` the PR's merge commit, and (b) read the struct
> *and* the `std::env::var` call — an env var that *overrides* a config field
> neither replaces it nor is replaced by it.

### 2.6 ArcGraph — GPU-autonomous decode

Existing in-code name: *"Arc GPU-Autonomous Decode"*. Crate `arc-cuda-graph/`.
Bypasses Candle entirely: `dedicated.rs` (warmup→capture→replay on a
non-blocking stream), `decode_forward.rs` (raw device-pointer forward, cuBLASLt,
zero per-step alloc), `autonomous.rs` (CUDA ≥12.4 WHILE conditional-graph node =
**one `cuGraphLaunch` per generation**; older CUDA = pre-captured body at
~2.5 µs/step), `graph.rs` (private memory pool so KV/weights are unaffected),
`weights.rs` (model-agnostic pointer extraction via `IsqModel`).

### 2.7 ArcSample *(named here; was unnamed)*

`mistralrs-core/src/sampler.rs` + `arc-cuda-graph/src/sampling_{cpu,cuda}.rs`.
`sampling_cpu.rs` is the **bit-exact reference** the GPU kernel mirrors.

> 🔴 **Measured session 8:** GPU radix top-k falls back to CPU on **every token**
> (`tensor_device_ptr: unsupported dtype I32`, ~10/sec continuous) — a
> device→host round trip per token in the decode loop.

### 2.8 ArcBoost

Existing in-code name (`mistralrs-core/src/arc_boost.rs`). Training-free
serving-side quality: `ConfidenceTracker` (DeepConf rolling-mean-logprob,
lowest-group confidence), `should_early_stop` online culling.

## 3. ArcModels — architecture support *(named here)*

**This is where any new model or architecture lands.** `mistralrs-core/src/`:
`models/`, `vision_models/`, `diffusion_models/`, `speech_models/`, plus
`pipeline/loaders/`. ArcModels/V4 (`deepseek4.rs`, `dsv4_*.rs`, `dsv4_mhc.rs` —
mHC split-Sinkhorn residual mixing) is the only architecture Arc has served
end-to-end. ArcModels/Schema = offline safetensors validators
(`arc-engine/src/weight_schema.rs`).

**V4 layer map (code-verified, `deepseek4.rs:5181`):** layers 0/1 standard, even
2..=42 **CSA (ratio 4)**, odd 3..=41 **HCA (ratio 128)**, slot 43 standard = the
**MTP block**. ⇒ **the ratio-0 set is exactly `{0, 1, 43}`.** Layer 42 is CSA;
it is *not* in the set. "`{0,1,42}`" is a retracted audit error.

## 4. ArcQuant — compression *(named here)*

### 4.1 QTIP — trellis weight quantization *(Jish's name)*

Two **rungs** share one L=16 format family, both 2 bits/weight:

| Rung | Geometry | Codebook | ISQ flag |
|---|---|---|---|
| `qtip2` | K=4 / V=2 | 512 KB Gaussian LUT | `--isq qtip2` |
| `qtip2b` | K=2 / V=1 | **computed** MCG/3INST, no LUT | `--isq qtip2b` |

`qtip2b` is the serving rung, decided by measurement: GEMV is flat (315 tok/s at
B=64, 317 at B=128) while the grouped kernel climbs (322 → 527). Crossover B=64.

| Subsystem | Does | Lives in |
|---|---|---|
| QTIP/Format | pack/unpack, UQFF serde, MoE gather dispatch, bake header | `mistralrs-quant/src/qtip/mod.rs` |
| QTIP/Bitshift | the `qtip2b` rung end-to-end, MCG codeword, CPU beam | `qtip/bitshift.rs` |
| QTIP/Search | exhaustive trellis DP **and** beam. `ARC_QTIP_BEAM` — **unset ⇒ exhaustive** | `qtip/viterbi.rs` |
| QTIP/Codebook | `QtipCodebook::{Gaussian, Mcg}`; **default `Gaussian`** | `qtip/mod.rs`, `kernels/qtip/qtip_codebook.cuh` |
| QTIP/Rotate | Hadamard incoherence, block ≤128. **Imports TurboQuant's WHT** — the two systems share one rotation primitive | `qtip/mod.rs` → `turboquant/wht.rs` |
| QTIP/Hessian | **diagonal only**, `ARC_QTIP_HESSIAN`, default **off** | `qtip/viterbi.rs` |
| QTIP/Tune | GEMV variant autotune across two generations | `qtip/tune.rs` |

> 🔵 **LDLQ and EoRA are NOT implemented.** LDLQ appears only in prose explaining
> that QTIP's per-block objective is the *diagonal* of what LDLQ captures. EoRA
> is named only as a future consumer in `calibration/mod.rs`. Do not describe
> either as part of the stack.
>
> **D4 — greedy is banned forever.** `QtipMode::Greedy` hard-errors in every
> build; UQFF refuses greedy-stamped payloads at load.

### 4.2 TurboQuant — KV compression *(Jish's name)*

`mistralrs-quant/src/turboquant/` (algorithm) + `arc-turbo/` (cache type,
re-export). L2 norm → randomized Walsh–Hadamard `D·H·D` → Lloyd-Max scalar
coding. Presets Default (K4/V3), Balanced (K3/V3), Aggressive (K3/V2).

| Subsystem | Lives in |
|---|---|
| TurboQuant/Codebook | `codebook.rs` (shipped dims 64/128/256), `generate.rs` (runtime Lloyd-Max for arbitrary dims) |
| TurboQuant/WHT | `wht.rs` — the shared `D·H·D`, also used by QTIP/Rotate |
| TurboQuant/Layout | non-power-of-two head_dim decomposition (80→64+16, 112→64+32+16, 192→128+64) |
| TurboQuant/Cache | `arc-turbo/src/cache.rs` — packed indices + norms + FP16 recent window |
| TurboQuant/Kernels | `kernels/turboquant/turbo_wht.cu`, `mistralrs-paged-attn/.../turbo_paged_attention.cu` |

> 🔵 **CORRECTED 2026-08-17 — this box previously read "TurboQuant is NOT on
> any default path, and has NEVER been measured." Both halves were false.**
> Jish caught it. Read the replacement carefully; it is deliberately split.
>
> **It IS the paged default.** `defaults::PAGED_CACHE_TYPE` is
> `PagedCacheType::TurboQuant` and `--pa-cache-type` carries no clap default, so
> a standard-layout **head_dim-128** model on CUDA takes K4/V3 with no flag and
> `cache_type_explicit = false`. `turboquant_stays_on_supported_geometry` in
> `cache_engine.rs` asserts exactly this. Off that envelope the default
> auto-falls back to `Auto` with a warning, and an explicit request hard-errors
> instead. The prefix cache auto-disables under TurboQuant, so default-path
> users lose prefix reuse silently — ⚠️ **NARROWED 2026-08-19: that requires
> CUDA AND PagedAttention AND standard layout AND head_dim 128. V4 is MLA, so
> "the default config silently disables prefix caching" does NOT apply to the
> flagship.** It is a real defect on the head_dim-128 fleet and a
> non-event on V4; a doc that states it unqualified is over-claiming. Only the
> **eager** path is opt-in
> (`ARC_TURBOQUANT_KV=1`, default off). **No kernel exists at head_dim 512, so
> V4 does not take it.**
>
> **It HAS been measured — narrowly.** `4eba13905` (2026-04-06): *"55 tok/s
> with TurboQuant = 46% over Candle baseline"*, **B200**, correct output.
> Harness `deploy/modal_b200.py` names the model: `MODEL="Qwen/Qwen3-32B"`,
> `gpu="B200"`, `--pa-cache-type turboquant`. wave61 recorded this commit and
> still concluded "never measured"; the model it called "unstated" was one file
> away. Eight CUDA correctness defects were fixed against that hardware on
> 2026-04-02 (`143b5ab20` V-cache stride, `fd0074792` Q·K warp deadlock).
>
> **What is still genuinely unmeasured, and must not be inflated:** the run was
> b=1, one card, one model, head_dim 128, `Default` preset. The "46%" compares
> Arc's whole dedicated decode path with Candle's and **isolates nothing about
> TurboQuant** — there is no A/B against an unquantized cache. There is **no
> quality evaluation at any preset**. The former "4.27× measured end-to-end" is
> **format arithmetic, never a forward pass** — retracted 2026-08-17 and it
> stays retracted; likewise the **1,026 → 260 B/token** V4 figure, which is
> design arithmetic over code that has never run.
>
> **Codebook density:** `∝ (1-x²)^((d-3)/2)`, i.e. `(x+1)/2 ~ Beta((d-1)/2,
> (d-1)/2)`. The old "Beta(d/2, d/2)" prose was off by a half in both shape
> parameters; the tables were always right. Pinned by
> `generator_reproduces_shipped_tables` to 1e-9.
>
> **In flight:** PR #94 (CUDA kernels at head_dim 64/128/256/**512**, Hopper +
> Blackwell) and PR #98 (TurboQuant KV storage for V4, `V4CachedK`). If those
> land, the "head_dim 128 only" and "V4 does not take it" statements above
> become stale — **re-check this section before quoting it**. A kernel existing
> does not make *that width* measured: 64/256/512 would be compiled and unrun,
> and only head_dim 128 has hardware behind it.

### 4.3 ArcBake — the offline pipeline *(named here)*

calibrate (`ARCCALIB` artifact, `.arccalib`) → quantize (`arc quantize --isq`) →
serialize shards + `residual.safetensors` → publish → serve as an overlay.
Layer-parallel multi-GPU bake (`bake_devices.rs`, byte-identical-artifact
invariant), OOM seatbelt (`bake_budget.rs`, `ARC_BAKE_HEADROOM` default 8%).
Ops scripts live in ArcLab/Ops, not here.

**Bake cost is a real product issue — customers bake too.** Measured: 241 s/layer
beam W=256 on H200 pre-#40; the published artifact baked at 370–376 s/layer on a
$1.49/hr A100. The lever is the **codebook** (≈1.68×, quality-neutral), not the
search and not the unpack (unpack measured at 1.0% of layer time).

### 4.4 ArcQuant/Legacy

Inherited upstream methods, unbranded, still supported: GPTQ, AWQ, HQQ, Marlin,
MXFP4, AFQ, blockwise-FP8, bitsandbytes, GGUF/GGML. `mistralrs-quant/src/`.

## 5. ArcKernels — the GPU substrate *(named here)*

| Subsystem | Contents |
|---|---|
| ArcKernels/Trellis | `mistralrs-quant/kernels/qtip/` — **`qtip_grouped_gemm.cu`** (the keystone: W2A16 batched-MoE grouped GEMM, on-device expert sort, cp.async double buffering, in-register 3INST decode → `mma.sync`), `qtip_gemv.cu`, `qtip_gather_gemv.cu`, `qtip_dequantize.cu`, `qtip_bitshift*.cu`, `qtip_beam.cu`, `qtip2b_beam.cu`, `qtip_exact_fp.cuh` (defeats `--use_fast_math` so CPU/GPU bakes are bit-identical) |
| ArcKernels/Attention | `mistralrs-paged-attn/src/cuda/` — paged v1/v2, `flash_attn_sinks.cu`, `flashinfer_mla_decode.cu` (head_dim 512 **decode-only** MLA), `flashinfer/` |
| ArcKernels/KV | `reshape_and_cache`, `copy_blocks`, `gather_kv_cache`, `gather_mla_cache`, `concat_and_cache_mla`, `update_kvscales` |
| ArcKernels/Decode | `arc-cuda-graph/src/cuda/` — `decode_kernels.cu`, `decode_loop.cu`, `gemv_bf16.cu`, `sampling_kernel.cu`, `flashmlasparse/` |
| ArcKernels/Model | `mistralrs-core/src/cuda/` — `moe_gemm.cu`, `moe_gemm_wmma.cu`, `moe_gemv.cu`, `sinkhorn.cu`, `sort.cu`, `ssm.cu`, `gdn.cu` |
| **ArcTarget** | **where new GPU support lands.** Build matrix + dispatch: `mistralrs-quant/build.rs` emits SM89/90/100; `has_qtip_kernels` at cc≥80 |

> **D16 — every kernel targets Hopper AND Blackwell.** Today most kernels are
> written to an Ampere baseline (`__CUDA_ARCH__ >= 800`), so they *run* on H200
> and B200 while exploiting neither — no wgmma on Hopper, no tcgen05/tmem on
> Blackwell. **We are arch-portable by being arch-naive.** Compliance means
> arch-specialised paths with a correct fallback, in the build matrix *and* the
> dispatch, and stating which arch was **measured** vs merely compiled.
>
> **D17 — the moat is our byte formats, so kernels touching them must be ours.**
> Trellis weights are a *state machine*, not a dtype: decoding symbol N requires
> walking 1..N-1, so no stock CUTLASS grouped GEMM can read them. Score any
> upstream kernel on *"can it read our bytes"*, never on *"can it express our
> attention shape"*.
>
> 🔵 **NARROWED 2026-08-19 — "there is no runtime LUT on the GPU."** True only
> of `qtip_gather_gemv.cu`. **`qtip_gemv.cu:266` still selects a stored Gaussian
> LUT when `cb_mult == 0`, and that is the SHIPPED DEFAULT**
> (`QtipCodebook::DEFAULT = Gaussian`, `mod.rs:570`). The in-tree doc at
> `mod.rs:596-598` is broader than the code supports. A lever that "does not
> exist" on one kernel may be the live path on another — **name the file.**
>
> ⚠️ **The shipped `qtip2` artifact cannot reach the grouped GEMM at all.**
> Above 8 tokens it dequantizes every distinct expert to BF16 in HBM —
> **16 traffic units against the grouped path's 1**. Every grouped-GEMM
> throughput number in this file and in `FACTS.md` **assumes a kernel we do not
> ship**. Say so whenever one is quoted.

## 6. ArcFormat — the artifact *(named here)*

| Subsystem | Does | Lives in |
|---|---|---|
| ArcFormat/UQFF | the on-disk format. `UQFF_VERSION` **0.3.0**; major must match, minor ≤. Arc's extensions: 0.2.1 QTIP 3-D stacked-expert payloads, 0.3.0 the **search stamp** (greedy-stamped payloads refused at load) | `mistralrs-quant/src/utils/uqff.rs` |
| **ArcOverlay** | **named here — it had no name in code OR prose.** An Arc artifact is an *overlay*: `--from-uqff <shard0>` layered over `-m <source checkpoint>`, plus `residual.safetensors` for everything not quantized. **You need the source checkpoint too** — 74 GB of artifact over 159.63 GB of public source | `mistralrs-core/src/pipeline/isq.rs`, `mistralrs-cli/src/args/model.rs` |
| ArcFormat/Publish | HF upload + generated model cards | `scripts/generate_uqff_card.py`, `docs/model-cards/` |

## 7. ArcLab — measurement *(named here)*

| Subsystem | Does | Lives in |
|---|---|---|
| ArcLab/Profiler | hierarchical span tree; **three structurally distinct time channels — wall / device (CUDA events) / sync (host blocked)**. Built because a 4-bucket split could not locate a 150× gap. JSON is the durable artifact. ⚠️ **`mark_unreachable` — the registry that is supposed to list dark features — is ITSELF DARK:** inert unless `ARC_PROFILE=1`, and **4 of its 6 `site` strings have drifted** away from the code they name. The instrument for finding switched-off code is switched off, with stale labels | `arc-profiler/` |
| ArcLab/Bench | **AA-AgentPerf**: recorded multi-turn agentic-coding trajectories replayed against any `Vendor` with tool responses injected for determinism; SLO tiers, ramp-then-binary-search for max passing concurrency | `arc-bench/`, `arc-cli/src/bench/` |
| ArcLab/Ops | **no Rust** — shell/Python: `preflight.sh`, `cuda_compile_check.sh`, rental playbooks, quality gates (ppl, GSM8K, longctx, coherence), bake drivers, harvest | `arc-tools/` |
| ArcLab/Validate | `arc validate --target-hbm` pre-flight HBM residency; offline weight-schema check | `arc-cli/src/validate.rs`, `arc-engine/src/weight_schema.rs` |

> `arc-tools/` is **not a Cargo crate** and is not a workspace member. Docs that
> list it beside the `arc-*` crates should say so.

## 8. ArcGate — correctness gates *(named here)*

The system that decides whether a green result means anything. Its named
concern, from **D18**:

> **SILENT SUCCESS IS THE HOUSE FAULT.** The mechanical form:
> **the absence of a signal was read as a specific signal.**
> A status check must distinguish *"not observed"* from *"observed negative"* —
> never collapse them into a default.

Named after the same bug appeared **eight times in one session** across kernels,
harnesses, tooling and CI: a clamp that turned causal into full attention; a
launcher that `return`ed leaving the output buffer uninitialised; a missing
`DType::I32` arm causing a per-token CPU fallback visible only as an unread
WARN; a harness printing `RESULT=OK` while every batch row was `None`; a
transient `gh` failure read as "resolved". Vigilance was at maximum in two of
those cases and did not prevent either — **that is the argument for mechanical
gates over another fix.**

| Subsystem | What it enforces |
|---|---|
| ArcGate/Canary | a coherence canary (questions a working model cannot miss) gates timing. **A broken box reports a BETTER number, not a worse one** — garbage decodes fast |
| ArcGate/Exit | separate exit codes: `0` pass · `1` genuine failure (the only strategy signal) · `2` environment could not answer |
| ArcGate/Parity | CPU↔CUDA bit-for-bit trellis parity; `qtip_exact_fp.cuh` exists to make it achievable |
| ArcGate/Tripwire | exhaustive `match DType` + `device_ptr_supports_dtype` so a new candle variant **breaks compilation** |
| ArcGate/Doctrine | D4 greedy ban in code; UQFF search stamp; `ensure_uniform_batch_cache_lens` promoted from `debug_assert!` to release |
| ArcGate/CI | the scoped clippy lane, CUDA type-check lane, fmt |

---

## NAMING RULES FOR WHAT COMES NEXT

1. **Every new subsystem declares its parent in its module doc**, first line.
   No orphans. If none of the eight parents fit, the parent set is wrong —
   raise it, don't invent a ninth silently.
2. **New model / architecture → ArcModels.** New GPU / arch → **ArcTarget**
   (under ArcKernels). These two exist so "future support" always has a home.
3. **`Arc` prefix for systems we own outright.** Names carried from papers
   (QTIP, TurboQuant) stay bare — that is the signal that an external
   algorithm is underneath.
4. **Name the thing, not the file.** `ValueAwareScorer` is a good name;
   `mod.rs` is not a system.
5. **Mark reality in the name's neighbourhood**, using the three tags this
   document uses: **⚠️ wired but not on the live path**, **🔵 does not exist
   yet**, **🔴 measured problem**. A name must never imply shipped.

## ATTRIBUTION — what is ours, what is the paper's

Keeping paper names is Jish's decision. It costs nothing *provided* the
provenance stays legible, and the model for that is already in the tree:
`mistralrs-core/src/kv_sharing/NOTICE` breaks SGLang-derived work into
**ADAPTED** (algorithm reimplemented, control flow follows, data structures do
not — with line-level `<-` mappings), **VERBATIM: none**, and **OURS**
(`ValueAwareScorer`, `layout.rs`, `content.rs`, subsumption dedup).

**Every system carrying an external name should have that breakdown.** Today:

| System | Origin | Ours |
|---|---|---|
| QTIP | Tseng et al., Cornell (ICLR'25) — trellis + incoherence processing | both rungs' Rust+CUDA implementation, the `qtip2b` bitshift geometry, beam search on GPU, the grouped GEMM, bit-exact CPU↔CUDA parity, the UQFF search stamp |
| TurboQuant | Zandieh et al., Google Research (ICLR'26), arXiv:2504.19874 — WHT + Lloyd-Max KV | the Rust implementation, runtime codebook generation for arbitrary dims, the non-power-of-two layout decomposition |
| ArcKV/Share | SGLang radix cache (Apache-2.0) | `ValueAwareScorer`, per-block element widths, content identity + cross-prefix meter, generic `Symbol` key |
| ArcKV/Paged | vLLM v1 `BlockPool`/`KVCacheManager` | port, plus Arc's cache-type plumbing |
| ArcAttention/Indexer | DeepSeek V4 Lightning Indexer; FlashMLASparse kernels MIT from sgl-project | the Rust `V4Indexer` and its CUDA parity |
| ArcModels/V4 | DeepSeek V4 technical report (mHC, MTP) | the whole Rust implementation |

**Gap to close (follow-up):** only `kv_sharing/` has a directory `NOTICE`.
QTIP, TurboQuant and FlashMLASparse rely on the root `NOTICE` plus scattered
prose. Each deserves the same ADAPTED / VERBATIM / OURS breakdown.

## FOLLOW-UP — code renames, deliberately NOT done here

Names above are **documentation-level only**. A mechanical rename during six
concurrent workstreams would collide with every open PR. Proposed, in
dependency order, each its own PR:

1. **Zero-risk, do first:** add a parent-system line to the top of each
   crate/module `//!` doc (`//! Parent: ArcInfer / ArcKV`). Pure comment.
2. **Low risk:** `kv_sharing` → the module doc adopts **ArcKV/Share** as the
   product name it says it is waiting for. No symbol changes.
3. **Medium:** introduce `arc-tools` as a real workspace member (or document
   that it is scripts only — it is currently listed as a crate in the README
   and is not one).
4. **Deferred, needs a migration note:** env var families are inconsistent
   (`ARC_QTIP_*`, `ARC_V4_*`, `ARC_TURBOQUANT_KV`, `ARC_PROFILE*`, plus
   `MISTRALRS_*` survivors). Any rename must keep the old name working for at
   least one release and warn — bake and serve scripts on rented boxes hardcode
   them, and a silent rename is exactly the D18 failure shape.
5. **Never in one pass:** renaming `mistralrs-*` crates. Upstream-merge
   compatibility is a stated design goal; `git merge upstream/master` currently
   works cleanly.
