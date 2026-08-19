# THE COMPLETE CENSUS — Arc vs SGLang vs vLLM

**Session 8, 2026-08-19. Ordered by Jish: *"You must find everything before touching a gpu,
compare with sglang, completely, two codebases completely compared. Not a single thing left."***

**SHAs.** Arc `cca5e5c6e` (= `origin/master`) · SGLang `ef490853` · vLLM `f1178f3a`.
Every Arc claim below was read at that SHA by an agent that opened the file. Where a claim is
second-hand it is marked **UNVERIFIED**.

**GPU hours spent producing this document: ZERO.**

---

## 0. THE TWO CLAIMS MUST NEVER SHARE A TABLE AGAIN

This is the most important line in the file.

> **When you are instruction-bound, bits-per-weight cancels out of the throughput equation.**
> `achieved = ceiling(bpw) × budget(bpw)/X`, `ceiling ∝ 1/bpw`, `budget ∝ bpw`
> ⇒ `achieved = 23,409/X` at B=256 (`1,992/X` at B=1) — **bpw-invariant**.

Compressing further does **not** make Arc faster. It decides whether the model *exists on the card*:

| format | bpw | V4-284B weights | fits one H200 (141 GB)? |
|---|---|---|---|
| **Arc QTIP** | **2.09** | **74.2 GB** | **YES** — ~65 GB left for KV ⇒ max_B 696 |
| MXFP4 / Marlin (best competitor weight-only) | 4.25 | 150.9 GB | **no** |
| FP8 (what SGLang actually serves) | 8.0 | 284 GB | **no** |

**SGLang's 308.6 tok/s on V4 runs on two GH200s. No competitor format puts this model on one
card at all.** That is the ×4–8 capacity-per-node wedge, and the format has already won it.

The speed claim is a **separate** claim, and it is 15–25× behind. **Conflating the two produced
the "16,600 → 1,477" framing that Jish rejected.** Separate them in every future document.

---

## 1. THE CEILING — corrected

### 1.1 The retracted claim (mine)

I briefed that **K4/V4/L12 → 1.38 inst/wt "at the 1.41 budget", so the full 16,602 is recoverable.**
**Wrong.** `bits/weight = K/V` (`mistralrs-quant/src/qtip/mod.rs:376-381`, stated in source and
confirmed by the byte geometry). K4/V4 is **1 bpw**, and the budget scales *with* bpw, so its own
budget is **0.674** — 1.38 is **2.05× over**, not at. I compared a 1-bpw kernel to a 2-bpw budget.

**The correct 2-bpw candidate is K=8 / V=4 / L=12.** LUT = 4096 × 4 × 2 B = **32 KB**, fits shared.
V=4 is principled, not arbitrary: for `mma.m16n8k16.bf16` each thread's B fragment is 4 contiguous-k
bf16 values, so a V=4 bf16 entry is **8 bytes = one `LDS.64` = a complete mma B-operand pair**, zero
conversion, zero packing. V=8 would need L≥16, putting the table back at 1 MB.

### 1.2 The budget is a band, not a point

`1.41` appears **only** in `COMPETITIVE_TEARDOWN.md:100`; it is not in `FACTS.md`.
It assumes 25.9 T-inst/s = **87.3% of sm_90 peak**, while `BUDGET_V4_B1.md:571` measures this kernel
family at **~70% issue efficiency** ⇒ **1.13**. **Honest band: 1.13 – 1.41 inst/weight.**

### 1.3 The ladder (measured rows marked)

| configuration | X (inst/wt) | B=1 | **B=256** | needs |
|---|---|---|---|---|
| shipped LUT rung — **MEASURED** | 35.52 | 56 | **659** | — |
| + PR #129 — **MEASURED** | 28.91 | 69 | 810 | merged |
| `sum2` — **MEASURED** | 15.85 | 126 | 1,477 | re-bake |
| **`sum2` floor, NO format change** | ~6.5 | 307 | **3,601** | 3 levers (§1.4) |
| K4/V2/L13, 32 KB bf16 LUT, permute folded | ~8.5 | 235 | 2,764 | re-bake |
| **K8/V4/L12, 32 KB bf16 LUT, permute folded** | ~4.7 | 424 | **4,977** | re-bake |
| ⤷ + row-scale hoisted out of the k-loop | ~3.7 | 538 | **6,321** | re-bake |
| ⤷ + tight addressing + larger TILE_N | **~1.9–2.4** | 830–1,050 | **9,700–12,300** | re-bake + kernel |
| full ceiling | ≤1.41 | 1,413 | 16,602 | — |

**VERDICT: ≤1.41 is NOT established. Best source-derived floor is ~1.9–2.4 — 1.4–1.7× short of
the full 16,602.** But **Jish is right that 1,477 is wrong**: the answer is **~3,600 free,
~6,300 for a re-bake, ~9,700–12,300 with kernel work = 15–19× over today's 659.**

### 1.4 The levers — **three claimed, ONE survives intact.** Corrected 2026-08-19 by source read.

1. ~~**Fold the nibble-reverse permutation into the LUT at load time.**~~ **DEAD — there is no
   runtime LUT on the GPU at all.** `qtip_gather_gemv.cu:318`: `(void)lut; // codebook now computed
   in-register, param kept for ABI stability`. `qtip/mod.rs:596-598` agrees: the codebook is
   *"Always materialized for the CPU search and for the artifact; **the GPU decode paths compute
   instead**."* The qtip2b family has no LUT either (`q2b_decode` = IMAD+LOP3+2×H2F+FADD).
   **There is nothing to fold a permutation into.** This is a property of the *proposed* K8/V4/L12
   design (which reintroduces a LUT), not a free-standing win on shipped code.
   *Side finding: a dead LUT-reading kernel still sits at `:215-230` with a stale ABI parameter — delete it.*
2. **Hoist the row scale out of the k-loop — SURVIVES, confirmed exactly.**
   `qtip_gather_gemv.cu:411-412` does `fmaf(w.x * scale, xg[j].x, a)` with `scale` loop-invariant
   (set `:365`, read `:393`, outside the k-loop). Per symbol V=2 weights, 2 FMUL + 2 FMA ⇒
   **1 FMUL/weight removable, −1.0 inst/wt.** Costs bit-exactness only (FP reassociation, which nvcc
   will not do). Same pattern repeats in the dead kernel at `:216-217, :229-230`.
3. **Random-access window on the decode GEMV — NOT A PORT.** `window_state_2b` /
   `grouped.rs:432` are **K=2 pair-reversal only**; no nibble-reversal (K=4) variant exists anywhere
   in Rust or CUDA. This requires **deriving and property-testing a new bijection**, not moving a
   proven one. **But the prize is bigger than first stated:** `qtip_gather_gemv.cu:396-401` replays
   `QTIP_WARMUP_SYMS = L/K = 4` symbols to decode `GROUP = 4` — a **1:1 warm-up-to-useful-work
   ratio**, ~5–6 ops per replayed symbol, all of it removable.

### 1.4b 🔴 Half of the last ladder rung is already REFUTED by a compiled in-tree experiment

`qtip_grouped_gemm.cu:135-142`, verbatim:
> **🔴 THE "ADDRESS ARITHMETIC" LEVER DOES NOT EXIST.** … A variant 3 was built that hoists every
> shared base pointer out of both loops and keeps the index math in 32 bits. It produces
> **BYTE-IDENTICAL SASS to variant 2** — total 672, IMAD 115, every count the same.

⇒ **nvcc already does it.** The "+tight addressing" half of the `~1.9–2.4` rung is dead on arrival;
only the `TILE_N` half survives. **That rung must be re-priced.**

*(Same file `:126` re-confirms the 672 inst/thread → 21.00 inst/wt anchor is genuine.)*

### 1.4c Consequence for §1.3

The "**~3,600 at B=256 with no format change and no re-bake**" row rested on three levers.
**One is dead (no LUT exists), one needs new derivation + property tests, one survives.**
That row is now an **overestimate of what is free** and must be recomputed once the SASS numbers land.
The re-bake rows are unaffected — they reintroduce a LUT by construction, which is what makes
lever #1 meaningful again.

### 1.4d Open disagreement between two censuses — unresolved, decides the re-bake target

- Census A: K8/V4/L12 makes the **exhaustive** Viterbi dramatically cheaper (~16–60 s/layer) because
  backtrace traffic falls 32× at V=4 and `FACTS.md:921` says that kernel is bandwidth-bound.
- Census B: bake cost is `2^L × 2^K` transitions/step ⇒ K4/L16 = K8/L12 = **1.05 M, identical**;
  K8/V4/L12 does **not** inflate or deflate bake time.

**Both cannot be right** — transitions/step is not total wall-time when V changes the number of steps
*and* the traffic per step. **Unresolved; §1.6's "re-bake is cheaper" claim is contingent on this.**

### 1.5 `L=16` is confirmed as the disease — verbatim

`qtip_gather_gemv.cu:98-102`: *"at L=16 the LUT is 2^16 * V * 4 = 512 KB, which does NOT fit in
48 KB shared memory, so the prior 'stage LUT to shared' path was dead and every per-symbol weight
lookup was a dependent, data-scattered GLOBAL load. ncu attributed the kernel's stall to
long_scoreboard."* Note it compares against **48 KB**, never the 227 KB opt-in — and even at bf16
(256 KB) L=16 does not fit either.

### 1.6 Re-bake cost — cheaper than today, and it buys quality

`FACTS.md:672-678`, `:905`: qtip2 beam W=256 = **372.0 s/layer A100 / 241 s/layer H200**;
exhaustive = 510 s/layer H200.
- Today's re-bake (for `sum2`): 241 × 43 = **2.88 h ≈ $14**.
- **K4/V2/L13:** Viterbi nodes 2^16 → 2^13 (8×) ⇒ **exhaustive ≈ 64 s/layer ⇒ ~0.8 h.**
- **K8/V4/L12:** branch metrics per weight identical, backtrace traffic 32× lower, and the exhaustive
  kernel is bandwidth-bound (`FACTS.md:921`) ⇒ **~16–60 s/layer, 0.2–0.7 h.** *(arithmetic, not measured)*

⇒ **Both candidates re-bake for LESS than `sum2` does AND make the exhaustive Viterbi affordable —
itself a quality gain over the shipping W=256 beam.**
**Recommendation: do not spend the re-bake on `sum2`. Spend it on K=8/V=4/L=12 with a shared bf16 LUT.**

### 1.7 🔴 The shipped artifact cannot reach the fast kernel at all

`qtip2-*.uqff` ships the **LUT rung**. The grouped GEMM on master is **`qtip2b` only**; the LUT-rung
one exists solely on the unmerged `perf/qtip-lut-grouped-gemm` branch (PR #133).
`gather_policy.rs:10` documents the absence. Above `DECODE_REGIME_MAX_TOKENS = 8` (`qtip/mod.rs:115`)
the shipped rung falls to `gather_forward_cuda` (`mod.rs:3649`): **host D2H sync of the router, then
dequantize every distinct expert to BF16 in HBM holding them all live**, then index_select + matmul +
index_add per expert — priced in-tree at **16 traffic units per distinct expert vs 1 per pair**
(`gather_policy.rs:66-70`).
**So even 659 is optimistic — the B=256 number assumes a kernel the shipped artifact cannot reach.**
Fix: merge #133, or ship `qtip2b` as the serving rung (its bake is now cheapest, `FACTS.md:905`).

### 1.8 The decisive next step costs ZERO GPU hours

Compile K8/V4/L12 and run `nvcc -cubin` + `cuobjdump -sass`. The method is already validated in-tree
to **1.3%** against ncu (static 35.06 vs ncu 35.52). **Hand-counting from C++ undercounts SASS by
2.05×** on this exact kernel — a census subagent made that error and was retracted. **Compile, never count.**

---

## 2. THE AGGREGATE CEILING — root-caused, two switched-off causes

### 2.1 🔴 V4 keeps full-context raw K/V for a 128-token window — 56×

- `deepseek4.rs:4099` → `NormalCache::new_plain(num_hidden_layers, max_position_embeddings)` = **full context**
- `deepseek4.rs:4138` passes `sliding_window: None`
- but `default_sliding_window()` = **128** (`:347-349`), and `raw_keep_span` proves nothing beyond
  `t_q + 127` is reachable
- **`RotatingCache` is fully implemented** (`kv_cache/mod.rs:81-85`) **and unused for V4**

At 8192 ctx × 43 layers: **Arc 361 MB/seq vs SGLang 6.4 MB/seq.** At B=32: **12.9 GB vs 1.2 GB.**

**This explains BOTH "V4 maps 140 of 143.7 GB ⇒ exclusive card" AND the B=32 ceiling.**

⚠️ **`compress_ratios` is `[0,0,4,128,...]`, so some layers genuinely need more than 128.
A uniform window would corrupt output. The fix must be per-layer.**

### 2.2 🔴 The merged ragged-batching fix never applied to V4 — one env var away

`supports_paged_attention()` → **false** (test-pinned `normal_loaders.rs:5691-5699`) ⇒
`normal.rs:345-347` nulls paged config ⇒ **`DefaultScheduler`**. The merged fix `64b3ff379` touched
**only** `paged_attention/scheduler.rs` — a file V4 never reaches.
`DefaultScheduler`'s own ragged path (`default_scheduler.rs:267,281-287`) is gated on
`ragged_decode_supported()` ← `batch_can_be_ragged`, which **refuses `XsRolling` unless
`ARC_V4_XS_PER_SEQ`** (`kv_cache/mod.rs:126-138,1200-1203,1492`), default **false**.
V4 builds `XsRolling` at `deepseek4.rs:4111`.

⇒ **The measured 18.83 → 61.22 tok/s (×3.25) is behind ONE UNSET ENV VAR. Shipped 19.6 @ B=32 is
the flag-off number.**

### 2.3 What V4's actual scheduler does not have

- **no token budget** — `sequence_fits` **discards its `_seq` argument** (`default_scheduler.rs:602-606`);
  32 prompts of 256 tokens and 32 of 32K are the same decision
- **no KV-capacity admission test** — `kv_cache_manager()` returns `None` (`:643`)
- **no preemption**
- **no priority queue** — sorts by **ID** (`:535`)
- admission is by **sequence count only**
- its only prefill-stall control is the undocumented env var `ARC_PREFILL_MAX_SEQS`
  (`:104`, enforced `:498`, `:541-546`), whose own doc records **K=32 → one prompt step of 43.2 s =
  50.3% of the profiled window; K=128 → zero tokens for 70 s** (`:88-96`)

---

## 3. PREFILL

### 3.1 ⚠️ SEQUENCING: chunked prefill would make things ~3× WORSE today

Chunking does **not** reduce FLOPs — it is latency shaping. It multiplies prefill **steps** by `N/C`,
and **the QTIP MoE expert gather is 71.3% of an N=128 prefill step**, billed per step.

At N=128: step = 128 × 11.98 ms = 1533 ms, gather = **1093 ms**. Chunking N=2048 into 16 × 128 pays
16 × 1093 ≈ **17.5 s of gather alone — ~3× the entire current 5.90 s prefill.** Even C=512 costs
~4.4 s vs ~1.1 s.

Arc's own ms/token curve corroborates: **11.98 (N=128) / 11.48 (512) / 11.81 (1024) / 8.23 (2048)** —
small shapes cost ~1.46× more per token.

**SGLang states the coupling structurally: `server_args.py:7014` makes the chunk size *be* the MoE
dispatch-tokens-per-rank. Chunk size is a MoE parameter, not a scheduler knob.**

⇒ **Fix the expert gather FIRST. Then C = 128–256 becomes affordable.**
And SGLang's 8192 default would be a **no-op** for Arc (N=2048 < 8192, one chunk). Their per-token
cost is 0.0975 ms vs our 2.880 ms — **29.5×** — so Arc's equivalent chunk is **~277 tokens**, ~32×
smaller than theirs. Copying their default would be wrong twice over.

### 3.2 🔴 V4's prefix cache populates only when a request FINISHES

`add_sequence` has exactly two callers, both inside `if let Some(reason) = is_done`
(`pipeline/sampling.rs:256-265`), while lookup is at admission (`engine/add_request.rs:650`).
SGLang caches per chunk (`radix_cache.py:516`); vLLM per step (`block_pool.py:225`).
⇒ **K simultaneous requests on the same cold prefix ALL miss: `K × 5.90 s` instead of
`5.90 s + (K−1)×ε`.** Highest-leverage prefill fix for V4; needs no paged attention. 1–2 sessions.

### 3.3 🔴 For V4 the ENTIRE block-level prefix cache is dead code

`paged_attention/{block_pool,block_hash,kv_cache_manager,scheduler}.rs` — **2,972 lines** — never
execute for V4 (chain in §2.2). Confirmed in-tree by
`pipeline/mod.rs:926-934` `mark_unreachable("paged_attention", …)`.
`ARC_V4_PAGED_ATTN=1` is **measured broken**: `normal_loaders.rs:3334-3355` records an A100-80G run
producing **zero tokens** (`1 query rows against only 0 keys` → `CUBLAS_STATUS_INTERNAL_ERROR` →
`CUDA_ERROR_ILLEGAL_ADDRESS`); `v4_paged_dispatch_precheck` refuses `bs > 1` (`:3330`).
**All V4 prefix-cache work must land in `prefix_cacher.rs` / `kv_sharing/`.**

### 3.4 🔴 TurboQuant and prefix caching are mutually exclusive

`engine/mod.rs:249-259` disables prefix caching **wholesale** when
`cache_type.supports_prefix_cache()` is false — "gathering packed TurboQuant blocks is not supported
yet". **The compressed-KV moat and the prefix cache cannot both be on.** 2–3 sessions for
dequant-on-gather.

### 3.5 Every prefill number we hold is cache-COLD

`bench.rs:82` `.with_prefix_cache_n(0) // Disable prefix cache for benchmarking`.
So 5.90 s @ N=2048 and 2.880 ms/prompt-token are cache-cold, and **no measurement of Arc's
cache-hit path exists.**

### 3.6 The hit-rate metric can read 100% while saving almost nothing

`engine/logger.rs:110` = `100 * prefix_cache_hits / total_new_seqs`, incremented **once per sequence**
(`add_request.rs:668`). A workload reusing one 32-token block of 2048 reports **100%**.
Token-level counters exist (`kv_sharing/radix.rs:167,177`, surfaced `prefix_cacher.rs:184`) with
**zero production readers**. **No Arc run to date could report whether the cache hit** ⇒ every
prefill measurement is uninterpretable. 1 session, 0 GPU-h.
⚠️ When wiring it, **exclude self-matches** or a chunked N=2048 prompt at C=256 will report ~87%
against a cold cache (SGLang handles this at `radix_cache.py:730-735`).

### 3.7 Arc truncates the TAIL, which guarantees a cache miss

`add_request.rs:283-290` keeps the tail; SGLang keeps the head (`managers/utils.py:213`).
Changing token 0 means the shared system prompt never matches. Reported only by a `warn!`, never in
the response. *(Correction: an earlier claim that "Arc is the only engine that discards prompt
content" was WRONG — SGLang has `--allow-auto-truncate`. vLLM is the only one that always rejects.)*

---

## 4. DECODE — Arc has no overlap mechanism of any kind

| capability | SGLang | vLLM | Arc |
|---|---|---|---|
| overlap scheduler | `overlap_utils.py:246`, **ON by default** (`server_args.py:963-970`) | `config/scheduler.py:148` | **ABSENT** |
| two-batch / dual-batch overlap | `batch_overlap/two_batch_overlap.py` | `v1/worker/ubatching.py` | **ABSENT** |
| multi-stream | yes | yes | **ABSENT** — no `Stream::new` anywhere |
| multi-step decode | `--num-continuous-decode-steps` | — | **ABSENT** |
| CUDA-graph capture | live | live | **DARK** (`arc-cuda-graph/src/graph.rs:71-81`) |

`engine/mod.rs:448-451` (in-tree): *"The pipeline mutex is held across the whole step, including the
serial `responder.send().await` loop at the end of it."* Sampling, detokenisation and response send
are all **inside** the GPU-step critical section.

**This lines up exactly with the measured 49% GPU-idle at B=1.**

### 4.1 The cast bill — 1,571/token, and where it comes from

`hc_pre`/`hc_post` (`dsv4_mhc.rs:236-338`, `:351-397`) do **10 real casts per call**, and `forward_4d`
calls mHC **4× per layer** (`deepseek4.rs:3229,3258,3268,3286`) ⇒ **20/layer × 43 = 860 = 54.7% of
the 1,571.** Two of them (`:335`,`:336`) are **discarded four lines later**; `:323` re-casts what
`:266` already cast.

**Why the references have no counterpart:** their kernels take bf16 in, accumulate in a register-level
`float`, write bf16 out, **in one kernel** — there is no intermediate F32 *tensor*, so there is nothing
to cast. SGLang's `mhc_pre_big_fuse` (`mhc.py:223`) does the whole of `hc_pre` in one kernel.
**Arc's 860 mHC casts are pure artefacts of expressing a fused kernel as ~50 candle ops.**

**430 launches/token (27.4% of the cast bill) are provably dead work, removable with zero new CUDA:**
`dsv4_mhc.rs:323` (86) + keeping `post`/`comb` in F32 across `:335,336`→`:376,377` (344).
**1 session, ~2 GPU-h.**

Launch budget: 9,131 → 3,831 (drop the ~5,300 casts/copies) → **~1,300–1,400** with mHC collapsed =
**6.8–7.9× fewer launches.**

### 4.2 Fusion Arc HAS but does not use on V4

`fused_add_rmsnorm` (kernel `arc-cuda-graph/src/cuda/decode_kernels.cu:46`, caller
`decode_forward.rs:275`) is **DARK** — the non-candle decode path is refused twice at
`weights.rs:54-55` and `:60-61` (*"256 experts × 3 projections … is not 7"*).

vLLM wrote **single kernels for exactly V4's shapes** that Arc expresses as many launches:
`silu_and_mul_clamp` (`activation_kernels.cu:298`) for V4's `swiglu_limit=10.0` — Arc uses **7 launches**;
`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:408` — Arc uses **≥6**.
vLLM also ships **15 generic fusion passes** including `NoOpEliminationPass`
(`utility/noop_elimination.py:18`), which eliminates *exactly the class of op that is 58% of Arc's
launches*. Hand-written Arc equivalent: ~37 sessions + ~155 GPU-h.

---

## 5. SAMPLING — the 517 KB blocking D2H is the smallest part of it

Per **sequence per token** at `temperature>0` with penalties (`sampler.rs:1352-1397`):
1 blocking **517,120 B logits D2H** (`:1352`) · `vec![0.0; V]` (`:1071`) · O(V) host penalty loop
(`:1080-93`) · `Tensor::from_vec` 517,120 B (`:1057`) · CPU divide + softmax, 2 × 517,120 B
(`:1383-84`) · `probs.to_vec1()` (`:1385`) · `Vec<(u32,f32)>` of V = **1,034,240 B** (`:299`) ·
`WeightedIndex::new` O(V) (`:951`) · **`rng.lock()` global mutex** (`:953`).
Logprobs add another 517,120 + 1,034,240 (`:445-446`) **and disable every GPU path** (`:1254`).

⇒ **≈4.6 MB host traffic per sequence per token ⇒ 132 MB D2H + 1.2 GB host traffic per decode step
at B=256.** Both references: **0 bytes** — their token never leaves the GPU.

Outside the sampler: `sequence.rs:1091-1101` runs `gs_find` over the **entire** accumulated output for
**every** stop string **every** token = **O(L²) per request**; `parse_text_tools` runs up to **3×** per
streaming token (`sampling.rs:70,134,167`).

**Both references do top-p by rejection sampling with NO SORT** (`topk_topp_sampler.py:487-89`);
Arc host-sorts (`sampler.rs:999,1012-19`) after a 517 KB sync.

### 5.1 Correctness defects found, all shipping

| defect | evidence |
|---|---|
| **Frozen Gumbel noise** — every token of a request sampled with identical noise | `sampler.rs:602-615` caches and `.clone()`s; `Sampler` built once per request (`add_request.rs:392`) |
| **`min_p` silently skipped** when `top_p` ∉ (0,1) | `sampler.rs:1001-1003` returns before `:1021-39`; replicated deliberately at `:719-745` |
| **`logit_bias` is a DEAD WIRE** — 200 OK with silently unbiased output | 28 hits, all writes/decls, **zero reads**; hard-nulled `chat_completion.rs:643`, `completions.rs:258` |
| **Speculative verify is not rejection sampling** — wrong for `T>0` | `pipeline/sampling.rs:696-698` + `mtp_pipeline.rs:1943` exact-`u32` equality; no `min(1,p/q)`, no `max(0,p−q)` residual |
| **MTP has no temperature gate at all** — drafted positions are silently greedy | `mtp_pipeline.rs:3489-3499` emits with `logprob: 0.0`; only the correction token is sampled (`:3515`). **Violates D4.** |
| **Device "top-p" never excludes the tail** | `sampling.cu:137-150` thresholds `top_p·u` over the **untruncated** CDF in id order; smaller `top_p` ⇒ *more* low-id bias |
| **`O(vocab × kept)` sampler** | `sampling_kernel.cu:272` `cap = vocab` when `top_k<=0`, and `add_request.rs:301-304` defaults `top_k = -1` ⇒ **129 M reads in ONE block** |

---

## 6. SPECULATION — a 1.55× multiplier on a 6× problem

**The 2-bit draft-head hypothesis is DEAD.** `floor_mtp_isq` (`mtp_pipeline.rs:165`) raises sub-int8 to
an 8-bit floor; it is live (`isq.rs:864`, gated `:770`, V4 tail override `deepseek4.rs:4634`); and fix
`07766cfa1` **is an ancestor of** the measured binary `46ea6948d`. The floor was already in place.
**Close this line.**

**The real measured number is `accept_rate=0.4194`, `tok_per_step=1.8387`** (depth 2, B=1, H200,
`wave51-CB-the-measurement.md:166-186`). *(The "p ≈ 0.485" I circulated appears nowhere in the tree.)*

SGLang's CI floor **for the same model** is `acc_length > 2.85` at depth 3, **topk=1 — also a linear
chain, architecturally the same as Arc's** (`test/manual/dsv4/test_dsv4_flash_mtp_tp8.py:107`).

Solving `E[accepted] = Σ pⁱ`:

| | γ | accept length | per-position p |
|---|---|---|---|
| **Arc, measured** | 2 | **1.839** | **0.544** |
| SGLang V4-Flash CI floor | 3 | 2.85 | **0.777** |
| SGLang DSpark (V4-Pro) | 5 | 4.678 | ~0.88 |

**The gap is per-position quality, not depth.** At p=0.544 Arc's asymptote is 2.19; depth 2→3 buys
only +9%, 2→4 buys +13.6%, ceiling +19%.

**⚠️ The ×5.53 framing does not survive the measured base.** To hit 313 tok/s/user at accept length
2.85 you need a base decode of **110 tok/s**; Arc measures **17.99**. Acceptance 1.84→2.85 is worth
**1.55×**. **The remaining 6.1× is the decode disease — speculation is a 1.55× multiplier sitting on
top of a 6× problem.**

**Cheapest path, in order:**
1. **Per-position acceptance telemetry — 0.5 session, 0 GPU-h.** `MtpAcceptance.accepted` is a scalar
   `usize` (`mtp_pipeline.rs:977`); **that scalar is why the gap cannot be diagnosed.**
   `p₁≈p₂≈0.54` ⇒ distribution mismatch; `p₁≈0.75, p₂≈0.35` ⇒ chain compounding (a code fix).
   Today we would be guessing, and guessing costs rental.
2. **Synthetic acceptance harness — 0.5 session.** `verify_proposed` (`arc-engine/src/mtp.rs:151`) is
   a pure function; forcing an accept length is ~30 lines. Prices the prize before paying for it.
   *(Correction: this does NOT remove the drafter — vLLM's `rejection_sample_method: synthetic`
   still runs it at full cost and only replaces the accept/reject decision.)*
3. **Depth 2→3 — free, one flag.** +9%.
4. **Then the head** — expensive (needs ≥3 H200s for an FP8 target), so do 1–2 first.

---

## 7. MULTI-GPU — code-complete, never run

**Arc has never run a forward pass on more than one GPU** (`wave60-CK-expert-parallel.md:5-6`).

**Correction to TAXONOMY.md:145** — it says expert parallelism is "NOT ON MASTER — open in PR #89".
**PR #89 merged** (`610c4506b`, feature `fce33ae22`). It IS on master: `deepseek4.rs:2211`
`build_expert_parallel_plan` → `:2284` → `MoEExperts::new_expert_parallel` `:2292`, plus
`moe/expert_parallel.rs` (847 lines). **But env-gated only** (`ARC_EP_SIZE`, `ARC_EP_PLACEMENT`) with
**no CLI flag** — another silent env var.

| | Arc |
|---|---|
| Tensor parallel | **HAVE, never run** (`distributed/layers.rs:154,479,826`) |
| Pipeline parallel | **DARK** — `NcclPipelineParallelMapper` is `#[allow(dead_code)]`, **never constructed** |
| Expert parallel | **HAVE, env-gated, V4-only** |
| EPLB | instrumentation only (`BalancednessCounter`); no rebalancer — **deliberate**, imbalance hard-bounded +5.05% |
| DP attention | **ABSENT** |
| PD disaggregation | **ABSENT — zero files** |
| Cache-aware routing | **ABSENT** |

**Single-card EPLB is vacuous** — one card is one rank owning all 256 experts. At B=128 ~244 distinct
experts × 6.29 MB = **1.53 GB/layer = 25× the 60 MB L2**; no permutation changes that. Hot/cold
residency only matters at BF16 (553 GB) — **which is exactly why the 2-bit rung exists.**
`arc-engine/src/expert_affinity.rs` has **zero call sites** and presumes an oracle for *next-step*
routing that does not exist; ceiling is 4.8%. **Do not wire it.**

---

## 8. HOW THEIR PUBLISHED NUMBERS ARE BUILT — do not quote them naively

`deepseek-v4-benchmarks.jsx:4`, verbatim:
> `// tokens_per_sec_per_gpu is total (input+output) tok/s/GPU = output/GPU × (isl+osl)/osl.`

Every cell is `isl: 8192, osl: 1024` ⇒ a **9.0× multiplier over output-only throughput**.
**"5,464 tok/s/GPU" is 607 output tok/s/GPU.** On top of that, V4-Flash at FP8 is ~284 GB vs an
H200's 141 GB, so **every per-GPU figure is a TP≥4 aggregate divided by the GPU count** — and TP is
**not recorded in the benchmark rows**, which is a real provenance weakness in their table.

**Three independent inflation factors: the input-token term (9×), the TP divisor, and the speculation
multiplier (2.97×).** Almost certainly the same mechanism behind the unreproducible
"2.2k tok/s per H200".

**The claim "PD disaggregation is required for every published ≥1.5k tok/s/GPU figure" is FALSE** —
their table has 77 cells, **zero** mentions of disaggregation, 67 of them `nodes: "single"`.

**The one directly comparable number** is per-user latency, which does not divide across GPUs:
SGLang H200/FP8 at `max_concurrency: 1` reports `tpot_ms: 3.26` ⇒ **306.7 tok/s/user** — the 308.6
figure — and it is a **multi-GPU (TP≥4) single-node** result, not single-card.

⇒ **State Arc's target as single-card, output-tokens-only, with batch size and ISL/OSL attached.
Stop comparing Arc's single-card output tok/s to their per-GPU total tok/s: they differ by ~9× in
definition before any engineering.**

---

## 9. WHERE ARC IS AHEAD (name these; they are unclaimed assets)

| asset | evidence |
|---|---|
| **2.008 bpw fused tensor-core GEMM** | Only tree with one; both upstreams floor at 4.125 for weight-only. `grep -ri "qtip\|trellis"` over both = **zero** |
| **Cost-aware eviction** | `kv_sharing/evict.rs:120,143` — measured recompute-ns × reuse prob ÷ staleness, fed by real prefill timings, refuses to fabricate when unobserved. SGLang's 7 policies are all recency/frequency; vLLM is LRU-only |
| **Exact spherical-marginal TurboQuant codebooks** | vs vLLM's Gaussian d→∞ approximation. Never claimed publicly |
| **top-nσ sampling, device-resident** | `sampler.rs:1200-1222`. Absent from both |
| **DRY sampling** | `sampler.rs:1102-1185`. Absent from both |
| **Closed-form expert load from the tokenizer** | `deepseek4.rs:2185` `tid2eid_expert_loads` — zero GPU, where both references need a profiling pass. Covers V4's 3 hash-routed layers |
| **Sampler inside the CUDA graph** | Arc's graph body **includes** the sampler (`autonomous.rs:510-590`); **both references keep it outside**. Design leads — has never run |
| **AOT compilation** | Rust; their JIT costs p99 11.5 s or −65% decode to avoid |
| **Three correctness guards with no counterpart** | ragged dead-prefix poisoning (`prefix_cacher.rs:228-235`), over-claim validation before mutation (`:428-453`), per-row logits at `new_len` not `padded.len()` (`inputs_processor.rs:301-309`) |
| **CUDA-graph-safe routing** | `qtip/mod.rs:3630-3636` refuses the D2H fallback under capture to avoid a real MMU fault — better documented than either reference |

---

## 10. SEQUENCING — what must precede what

1. **QTIP MoE expert gather** ⟶ blocks chunked prefill (§3.1), and is 71.3% of prefill
2. **Per-position MTP telemetry** ⟶ blocks any acceptance work (§6)
3. **Token-level cache hit rate** ⟶ blocks evaluating any prefill change (§3.6)
4. **Graph-mode mask wiring** ⟶ blocks any quality claim on the CUDA-graph arm
5. **Merge #133 / ship qtip2b** ⟶ blocks every inst/weight argument (§1.7)
6. **Do NOT** wire `expert_affinity`, build EPLB for one card, or chase tree speculation
   (vLLM is chain-only too)

---

## 11. RETRACTIONS ISSUED THIS SESSION

| claim | status |
|---|---|
| "K4/V4/L12 → 1.38 inst/wt is at budget, full 16,602 recoverable" | **RETRACTED** — 1 bpw vs a 2-bpw budget (§1.1) |
| "MTP acceptance p ≈ 0.485" | **RETRACTED** — appears nowhere in the tree; real is 0.4194 |
| "The 2-bit draft head explains the acceptance gap" | **RETRACTED** — fix predates the measurement |
| "Acceptance is worth ×5.53" | **RETRACTED** — 1.55× against the measured base |
| "The synthetic harness runs with no drafter" | **CORRECTED** — drafter still runs at full cost |
| "PD disaggregation is behind every ≥1.5k/GPU figure" | **RETRACTED** — it is the 9× metric definition |
| "`paged_attention/scheduler.rs:310-311`'s comment is false" | **STALE** — fixed and merged (`64b3ff379`) |
| "`graph.rs` logs capture is IMPOSSIBLE" | **STALE** — now implements capture-once |
| "head_dim 512 is the blocker" | **INVERTED** — 512 compiles; **448** fails (`vec_size 14 % 8 ≠ 0`) |
| "SGLang uses `page_size=1`" | **WRONG for V4** — overridden to **256** |
| "Arc is the only engine that discards prompt content" | **WRONG** — SGLang has `--allow-auto-truncate` |
| "Arc QTIP is ~10 inst/wt" (a subagent's hand-count) | **REJECTED** — 2.05× low vs measured SASS in the same file |
| "Expert parallelism is not on master" (TAXONOMY.md:145) | **WRONG** — PR #89 merged |

**The methodological rule that produced most of these: a number you did not compile is an estimate,
and hand-counting from C++ undercounts SASS by 2.05× on this kernel family.**
