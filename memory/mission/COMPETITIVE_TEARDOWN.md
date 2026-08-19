# WHY SGLANG'S PYTHON GETS 308 tok/s AND ARC'S RUST GETS 14.3

**Read before any performance planning.** Source-verified against sglang `c14312a6` and vllm `f1178f3a`
(both 2026-08-18) plus our own tree. **Every claim carries file:line.** Session 8, 2026-08-19.

Reference numbers (2× GH200, FP8 KV + **MXFP4 experts / Marlin**):
[dnhkng.github.io GH200 part 4](https://dnhkng.github.io/posts/gh200-benchmarking-part-4-dsv4-released/)
| | SGLang | vLLM | **Arc (1× H200)** |
|---|---|---|---|
| single-request decode | **308.6 tok/s** | 275.9 | **14.3** |
| aggregate peak | **630** @ concurrency 4 | — | **19.6** @ B=32 |
| TTFT p50 | **0.284 s** | p99 11.5 s | 1.80 s @512 |
| prefill | **10,262 tok/s** | 7,112–10,323 | ~284 |

---

## 0aa. 🔴🔴 SUPERSEDED IN PART BY `CENSUS_SESSION8.md` (2026-08-19) — SIX MORE RETRACTIONS

**This file is the largest single carrier of claims retracted at the close of session 8.** Each is
struck **in place** below; this list exists so a reader who already quoted one knows to re-read.

| in this file | status |
|---|---|
| §"MTP: acceptance `p ≈ 0.485`" (§0b) | **RETRACTED** — it appears nowhere in the tree. Measured: **0.4194**, B=1 only |
| §"our 2-bit draft head explains it" (§0b) | **RETRACTED** — the `floor_mtp_isq` fix `07766cfa1` is an **ancestor** of the measured binary `46ea6948d` |
| §"×5.53 if the MTP suspicion lands ⇒ 313 tok/s" (§0b) | **RETRACTED — ×1.55** against the measured base |
| §"K4/V4/L12 → 1.38 inst/wt = AT the budget" (§"DOES ANY TRELLIS PARAMETERISATION REACH PARITY") | **RETRACTED** — 1 bpw scored against a 2-bpw budget (`bpw = K/V`, `qtip/mod.rs:376-381`) |
| §"16,600 → 1,477" (§"AGGREGATE") | **REJECTED FRAMING** — capacity and speed are two claims and must never share a table |
| §"`page_size=1`" (§4) | **WRONG FOR V4** — overridden to **256** |
| the ≥1.5k tok/s/GPU figures (§0) | it is the **metric definition**, not PD disaggregation — see §0a |

## 0a. 🔑 THE TWO CLAIMS MUST NEVER SHARE A TABLE AGAIN

> **When you are instruction-bound, bits-per-weight CANCELS OUT of the throughput equation.**
> `achieved = ceiling(bpw) × budget(bpw)/X`, `ceiling ∝ 1/bpw`, `budget ∝ bpw`
> ⇒ **`achieved = 23,409/X` at B=256 — bpw-invariant.**

Compressing further does not make Arc faster. **It decides whether the model exists on the card.**

| format | bpw | V4-284B weights | fits one H200 (141 GB)? |
|---|---|---|---|
| **Arc QTIP** | **2.09** | **74 GB** | **YES** |
| MXFP4 / Marlin (best competitor weight-only) | 4.25 | 151 GB | no |
| FP8 (what SGLang actually serves) | 8.0 | 284 GB | no |

**Every SGLang V4 number at the top of this file runs on TWO GH200s. No competitor format puts this
model on one card at all.** ⇒ **The capacity wedge is already won by the format. Speed is a separate
claim and we are 15–25× behind on it.** Conflating them produced the "16,600 → 1,477" framing Jish
rejected. **Two claims, two tables, always.**

**And their per-GPU totals are not comparable to ours by construction:** `(input+output)/GPU` at
isl 8192 / osl 1024 is a **9× multiplier**, on top of a TP≥4 divisor and a 2.97× spec multiplier.
**It is the metric definition, not PD disaggregation** — that attribution is **retracted**.

## 0. 🚨 THREE RETRACTIONS — READ BEFORE QUOTING ANYTHING ABOVE OR BELOW
1. **"vLLM Wide-EP 2.2k tok/s per H200" is NOT CITABLE.** The blog gives no ISL/OSL, concurrency or
   GPU count. The only fully-specified published figure is llm-d's **1,573 output tok/s per DECODE
   GPU** — at 2K/2K, concurrency 2048, **32× H200 + InfiniBand, DP16+DP16.** Never quote 2.2k.
2. **The "36× prefill gap" is wrong, and so was its first replacement (14.8×).** vLLM measures
   **2,070 tokens → 0.291 s**, a matched length against our N=2048 → 5.90 s ⇒ **20.3× on their two
   GPUs, 10.2× per GPU.** The 36× compared 2 GPUs at 8K against 1 GPU at 512.
3. **🔑 THE HEADLINE COMPARISON WAS A CATEGORY ERROR AGAINST OURSELVES.**
   **DeepSeek-V4-Flash with speculation OFF on 2× GH200 is 92.9 tok/s = 46.5 per GPU.**
   Their k=6 buys **2.97×** to reach 275.9/308.6. **Our 14.3 (no drafter running) vs their 46.5
   per-GPU no-spec is 3.25×, NOT 21×.**

## 0b. 💰 THE LADDER AND THE BUDGET — 18 GPU-h / **$87**, of which the first 7.5 h / **$36** carries most of it

### DECODE b=1: 14.99 → **56.6 base** → **109** with today's MTP → ~~**313** if the MTP suspicion lands~~
🔴 **The 313 is RETRACTED (2026-08-19)** — it rests on `p=0.485` (fitted, not measured) and on a
2-bit-draft-head theory refuted by commit order. Against the measured `0.4194`, acceptance work is
worth **×1.55**, so the honest top of this ladder is **~88**, not 313. **56.6 and 109 stand.**
| # | fix | term attacked | × | tok/s | status |
|---|---|---|---|---|---|
| R1 | fused FP8 KV kernel | blocking D2H 43.9→1.0 | 1.08 | 16.2 | **BUILT, no PR** |
| R2+3 | arena + graph capture + device WHILE node | **49% GPU idle → ~0** | **2.13** | **34.6** | **BUILT, PR open** |
| R4 | gather 35.52→28.91 inst/wt | trellis | 1.05 | 36.2 | **BUILT, PR open** |
| | **BUILT-ONLY SUBTOTAL — NOBODY HAS COMPOSED THESE** | | **×2.41** | **36.2** | |
| R5 | `sum2` → 15.85 inst/wt | trellis | 1.17 | 42.3 | needs **re-bake** |
| R6 | cast/elementwise tax (5,300 launches ≥8 ms; **the router re-casts `self.weight` to F32 every layer every step**, `deepseek4.rs:2000`, self-flagged in-code) | −75% | 1.34 | **56.6** | **NEW** |

**R2+3's ×2.13 is externally bracketed:** vLLM PR #12222 measures no-graph→graph on a 256-expert MoE
at **1.94× (Marlin) / 2.53× (Triton) / 2.96× (+`moe_align`)**. ⇒ **56.6 on ONE H200 = 1.22× their
per-GPU no-spec 46.5.** Past that, 200 tok/s needs a further ×1.83 and it lives in **`sdpa` at
7.10 ms ≈ 364× its own KV-bandwidth floor** — and **V4 can use no FlashAttention at any generation**
(sinks on all 43 layers, head_dim 512 vs the kernel's fixed 512/64 and V4's 448 tripping its
`static_assert`). **That kernel must be ours.**

### 🔴 MTP — THIS WHOLE SECTION IS RETRACTED (2026-08-19). The surviving half is below it.

~~Fitting `p+p²+p³ = 0.8387` gives **p ≈ 0.485**~~ · ~~**our draft head is 2-bit quantised**~~ ·
~~**at p=0.92: 56.6 × 5.53 = 313 tok/s**~~ — **all three retracted.**

1. **`p ≈ 0.485` was FITTED, then quoted as measured. It appears nowhere in the tree.** The only
   acceptance number that exists is **`accept_rate = 0.4194`** (26/62 accepted,
   `tok_per_step = 1.8387`), **B=1 only** — `wave51-CB-the-measurement.md:166-186`.
2. **The 2-bit-draft-head theory is dead.** `floor_mtp_isq` (`07766cfa1`) is an **ancestor** of the
   binary that produced `0.4194` (`46ea6948d`) — **the gap was measured with the fix already in.**
   The "one flag and one hour A/B" was an A/B against a state that no longer existed.
3. **Against the measured base, fixing acceptance is worth ×1.55, not ×5.53.** It is a multiplier on
   a **6× problem**, not a solution to one. Budgeting it as the highest-value GPU-minute was wrong.

**What survives, and it is the actionable half:** ⚠️ **per-position telemetry does not exist** —
`MtpAcceptance.accepted` is a scalar (`mtp_pipeline.rs:977`), so `p₁≈p₂≈0.54` (distribution
mismatch) and `p₁≈0.75, p₂≈0.35` (chain compounding) are **indistinguishable**. **That instrument
gates all acceptance work and costs zero GPU hours.** Also standing: **depth is not the lever** —
SGLang's V4 recipe is `topk=1`, a chain, and **vLLM is chain-only too**, so tree speculation is not
table stakes. **MTP depth 2→3 is +9% (1.84 → 2.00 tok/step) for one flag.**

### PREFILL (N=2048): 121 → **725 tok/s**, of which **347 is already built**
| rung | s | tok/s | status |
|---|---|---|---|
| baseline | 16.87 | 121 | MEASURED |
| + grouped GEMM | 9.61 | 213 | **BUILT** |
| + FP8 cuBLASLt | 5.90 | **347** | **BUILT** — *two built rungs alone = ×2.86* |
| + stride fixes ported to the LUT kernel | 4.87 | 420 | **NEW, high confidence** |
| + TILE_M 16→64 | 3.20 | **640** | **NEW** |
| + `sum2` | 2.82 | 725 | needs re-bake |

**Two NEW rungs found, neither needing a re-bake or a format change:**
- `qtip_grouped_gemm_lut.cu:204-205` declares `s_x[2][16][64]` (**128 B row stride**) and
  `QLG_WP_STRIDE = 32` — **verbatim the 8-way and 2-way shared-memory bank conflicts PR #124 already
  measured and fixed on the sibling kernel for +39.4%.** #133 branched off an older master and never
  got them.
- **`QLG_TILE_M = 16` caps trellis-decode amortisation.** The sibling's header says *"TILE_M IS A
  NON-LEVER IN THE REGIME WE SERVE"* — **true for decode, FALSE for prefill**, which #133 just put
  this kernel family on. m-tiles/expert = 3 at N=2048, 12 at 8K, 48 at 32K ⇒ **×3 at 2K, ×4–6 at
  exactly the 8K–32K lengths they benchmark.**

**Also unowned:** vLLM's `moe_align_block_size` costs **317.8 µs → 39.8 µs** (PR #19572). At 43
layers naive bookkeeping is ~19 ms/step ⇒ **an Arc grouped GEMM must own the sort/pad, not just the
matmul.**

### ⚠️ AGGREGATE — a CORRECTION TO `CEILINGS.json`
**Its 16,600 tok/s at B=256 is bandwidth-only and silently assumes the format is FREE TO DECODE.**
| B | bw ceiling | at X=35.52 | at X=15.85 (`sum2`) |
|---|---|---|---|
| 32 | 3,816 | **151** | 339 |
| 256 | **16,602** | **659** | **1,477** |
⇒ **The 16,600 headline is 25× unreachable with today's format+kernel and 11× with `sum2`.**

> 🔴 **DO NOT PRESENT THIS AS "16,600 → 1,477". Jish rejected that framing and he was right.**
> Both columns are *speed*; the capacity claim is a **different claim** and does not belong in this
> table (§0a). Stating the speed gap is correct — **stating it as if it were the wedge is not.**
> `sum2`'s 1,477 is also not the ceiling of the format work: the compiled ladder puts K8/V4/L12 at
> **4.375 inst/wt**, and the honest read of the whole program is **~15–19× over today's 659 with
> re-bake plus kernel work** — scope, not a promise, and priced at ~$30 of re-bake.
**But the trellis tax is a LOW-CONCURRENCY tax that AMORTISES with B:** at B=256, `sum2` lands within
**1.07×** of llm-d's 1,573-per-decode-GPU — **which takes 32 H200 + InfiniBand. Arc's claim is one
card.** Path: measured 19.6 @B=32 → ×3.25 ragged (**MEASURED 18.83→61.22**) → ×1.82 MTP per-seq KV →
×2.09 prefill = **~242 tok/s at B=32.** All three BUILT; two unmerged, one merged-but-gated-OFF.

### DOES ANY TRELLIS PARAMETERISATION REACH PARITY? **NO — AND THE LADDER IS NOW COMPILED.**
🔴 **RETRACTED 2026-08-19: ~~"K4/V4/L12, 32 KB → 1.38 inst/wt = AT the budget ⇒ 1,413 tok/s b=1"~~.**
`bits/weight = K/V` (`qtip/mod.rs:376-381`), so **K4/V4 is 1 bpw** and the budget scales *with* bpw
— its own budget is **0.674**, and 1.38 is **2.05× over, not at**. A 1-bpw kernel was scored against
a 2-bpw budget. **The budget is also a band, not a point: 1.13 – 1.41** (the 1.41 assumes 87.3% of
sm_90 issue peak; this kernel family measures ~70%).

**Replaced by compiled numbers** (`nvcc -cubin`, CUDA 12.4.131, inner loop by unroll differencing —
all three geometries **2 bpw**, so the scoring error cannot recur):

| geometry | sm_90 | sm_80 | static smem |
|---|---|---|---|
| K4/V2/L16 computed — **SHIPPED** | **15.125** | 14.812 | 0 B |
| K4/V2/L13 bf16 LUT | **11.250** | **10.250** | 32,768 B |
| K8/V4/L12 bf16 LUT | **5.375** | **4.625** | 32,768 B |
| **K8/V4/L12 + row-scale hoist** | **4.375** | **3.625** | 32,768 B |

**3.46× fewer instructions than shipped — and still 3.1–3.9× short of the budget.** Both LUTs fit
the 48 KB static limit (no `cudaFuncSetAttribute`); occupancy **62.5%**, **register**-limited.
🔴 **And the re-bake gets ~8× MORE expensive, not cheaper** — the production baker is **beam**
(213 s/layer A100), not exhaustive (8,257), and is issue-bound; `(n/V)×W×2^K` ⇒ **~1,700 s/layer
≈ 20 h ≈ $30**. **`L=16` remains the disease** (512 KB never fits shared, which is why RUN-161 went
computed). Both are format changes (D17). **`sum2` needs no `L` change** and still has headroom
inside its own arm (15.85 measured), **but do not spend the re-bake on it.**
**Calibration that this is engineering, not physics: SGLang's own 308.6 is 48% of MXFP4's
instruction ceiling and 44% of its bandwidth ceiling — they are ~2× off too.**

### 💰 BUDGET (H200 @ $4.85/h; V4 maps 140 of 143.7 GB ⇒ exclusive card)
| h | $ | rung |
|---|---|---|
| 3.0 | 14.55 | compose R1+R2+R3 — **arena eviction defect FIRST** (gate fails past the 8 GB cap) |
| ~~1.0~~ | ~~4.85~~ | 🔴 **CANCELLED — "MTP draft-head unquantised A/B, the p=0.485 test, ×2.9" is RETRACTED.** The fix is already in the measured binary. **Its replacement costs ZERO GPU: per-position acceptance telemetry, which gates all acceptance work.** |
| 1.0 | 4.85 | stride fixes ported to the LUT kernel + bit-parity |
| 1.5 | 7.27 | TILE_M sweep at N=2048/**8192** + parity |
| 1.0 | 4.85 | merge-and-confirm the three open perf PRs on **one** binary |
| | | **↑ first five rows = 7.5 h / $36.39 ⇒ decode ~109–313, prefill ~640** |
| 1.5 | 7.27 | cast/elementwise fusion A/B |
| 2.0 | 9.70 | MTP per-seq KV at B=1/8/32/128 |
| 1.0 | 4.85 | chunked prefill's own acceptance gate (**never run**) |
| 4.0 | 19.40 | **`sum2` re-bake + GSM8K — the ONLY format decision here** |
| 2.0 | 9.70 | aggregate ladder B=8/32/128, post-everything |
| **18.0** | **$87.30** | |

**Caveats:** Marlin counts are PTX-level ±20%, no SASS. R2+3's ×2.13 needs three unmerged branches to
compose *and* the arena bug fixed — least proven, though externally bracketed. The ×1.41 stride
transfer is from the sibling kernel, not measured on this one. `sdpa` 364× assumes 2,048 context.
Aggregate ×2.09 is derived from PR #125's `run_max 52/128`, not measured end-to-end.

## 1. 🔴 HOST CUDA CALLS PER DECODE STEP — **~10–30 vs ~34,800. A 1,000–3,000× GAP.**

**Theirs: ONE `cuGraphLaunch` covers the ENTIRE model forward.**
`model_executor/runner_backend/full_cuda_graph_backend.py:156`; captured body is the whole
`forward()` at `runner/decode_cuda_graph_runner.py:1188`. Plus **≤18 small tail-fills — skipped
entirely when batch size hits a bucket** (`cuda_graph_buffer_registry.py:232`), 1–2
`torch._foreach_copy_` (`:464`), and 1–2 **fused** Triton metadata kernels
(`kernels/ops/attention/metadata.py:600` — *"replaces 4-5 sequential CUDA kernels with 1-2"*).

**Ours: 9,131 launches + 25,622 alloc/free/H2D = ~34,800** (`BUDGET_V4_B1.md`).
**Allocation during replay is STRUCTURALLY zero for them** — a graph-private pool
(`runner_utils/pool.py:74`) on top of PyTorch's caching allocator.

> **The language is irrelevant. Their Python runs ONCE PER STEP as orchestration; our Rust issues
> nine thousand launches per token.**

## 2. 🔴🔴 THE GRAPH IS ONLY HALF OF IT — **the sampled token never leaves the GPU**

Theirs: the token is **scattered into a device buffer** (`managers/overlap_utils.py:555`) and step
N+1's `input_ids` is a **device gather** (`:107`). The result D2H runs **on a side stream, consumed
one iteration later** (`scheduler.py:3737-3744` → `batch_result_processor.py:203`).
**Overlap is ON BY DEFAULT** (`server_args.py:970`).

**Ours: `logits.to_vec1()` — a 517 KB BLOCKING device→host copy — EVERY TOKEN**
(`mistralrs-core/src/sampler.rs:437, 838, 1288`).

> ⚠️ **This serialises host and device BEFORE any kernel-count argument applies.** The decode loop
> **cannot run ahead of the GPU** while the sampler drags 129,280 floats to the host every token.
> **Highest-leverage item on this page, and it is not a kernel.**
> We *have* `arc_launch_sampler_bf16` (device-resident) — **gated off.**

## 3. 🔑 inst/weight — THE MOAT, STATED NUMERICALLY

| path | **inst/weight** | bits/weight | source |
|---|---|---|---|
| Marlin MXFP4, **sm_90 (H200/GH200)** | **2.875** (23 ops → 8 weights); 3.875 w/ E8M0 scale | 4.25 | `csrc/…/marlin_moe_wna16/dequant.h:433-449,469-470`; `marlin_template.h:1366,1409` |
| MXFP4 native, **sm_100** | **0** — `tcgen05` block-scaled MMA eats e2m1+e8m0 directly | 4.25 | `csrc/…/fp4/mxfp4_blockwise_moe_kernel.cu:183-210` |
| Marlin INT4 GPTQ fp16 | 1.125 | 4.25 | `dequant.h:121-142` |
| **Arc QTIP trellis** | **28.91** (35.52 shipped, **15.85 w/ `sum2`**) | **2** | `BUDGET_V4_B1.md` |

> **We store 2.1× denser and pay ~10× the decode instructions on Hopper — and INFINITELY more on
> Blackwell, where MXFP4 costs ZERO because the tensor core eats it natively.**

**And they decode each weight EXACTLY ONCE per GEMM**, fusing dequant into the mainloop and reusing
it across `thread_m_blocks×16` rows of A (`marlin_template.h:1377-1423`). **Ours is ~99% trellis
decode, so the density advantage is currently unrealised.** H200 dispatch to Marlin confirmed —
DeepGEMM FP4 needs sm_100/120 (`fused_moe/experts/deep_gemm_moe.py:426-428`), Triton path commented
out (`oracle/mxfp4.py:345-347`).

## 4. 🔴 RAGGED BATCHING — **our blocker is ONE COMMENT AND ONE MAP KEY, not a kernel**

**Theirs: varlen + block table, ONE kernel call for a mixed 500-prefill + N-decode batch** —
`cu_seqlens_q=[0,500,501,502,503]`, `seqused_k`, `block_table`
(`vllm/v1/attention/backends/flash_attn.py:1176-1199`; ragged build `gpu_model_runner.py:2042-2049`).
**Sequence length is NEVER a graph shape**: bucketing is **batch-size only**
(`server_args.py:5143-5148`), ~~`page_size=1` (`arg_groups/overrides.py:2396`)~~ ⚠️ **RETRACTED
2026-08-19 — for V4 `page_size` is OVERRIDDEN to 256**, so "they run page_size=1" is wrong for the
only model we are comparing against; page table sized to static max, kernel bounded by
`cache_seqlens` (`metadata.py:616-618`).

**Ours:**
- `paged_attention/scheduler.rs:310-311` buckets prompts to equal length, commented
  *"required for correct flash attention varlen operation"* — **that is FALSE. `cu_seqlens` exists
  precisely to avoid it.**
- `scheduler/default_scheduler.rs:179` partitions running decodes by **exact KV cache length**
  (`sequence.rs:800`) ⇒ staggered arrivals each land in their own bucket ⇒ **19.6 tok/s at B=32.**
- **We ALREADY build varlen `cu_seqlens_kv`** for the prefix-cache path
  (`pipeline/inputs_processor.rs:163-167`). ⇒ **HAVE-IT-DARK, not lack-it.**

## 5. SPECULATIVE DEPTH IS **NOT** WHERE THE GAP LIVES
SGLang's DeepSeek default is `(steps=3, topk=1, draft=4)` — **a chain, not a tree**
(`arg_groups/speculative_hook.py:839`); accepted ceiling `spec_steps+1`
(`speculative/eagle_info.py:43-47`). **"DSpark k=6" is a VERIFY WINDOW of 6 = gamma 5 + bonus, not
depth 6** (`docs/cookbook/…/DeepSeek-V4.mdx:599`).
**Our 1.93 / 1.67 / 1.55 / 1.06 curve is the NORMAL shape, not a defect.** At their depth our b=1
is worth **~2× ⇒ ~28 tok/s against 308.** ⇒ **The 21× lives in the 66.68 ms step, not in speculation.**
*(Useful: they disable speculation entirely at bs≥64, driven by an EMA on measured accept length —
`speculative/adaptive_spec_params.py:22-46, 216-235`.)*

## 6. PREFILL CHUNKING — and a correction to our own diagnosis
⚠️ **RETRACTED: "prefill excludes decode" is NOT the differentiator.** SGLang runs prefill
*instead of* decode too (`managers/scheduler.py:3128-3137`) — **structurally identical to our
`paged_attention/scheduler.rs:309`.** Only **vLLM V1** truly co-schedules.
**The differentiator is the UNBOUNDED CHUNK.** SGLang auto-sets `chunked_prefill_size = 8192` on
H200-class (`server_args.py:4861`, fallback 4096 at `:4886`); over-long prompts are **truncated to
`rem_chunk_tokens` and re-admitted next iteration** (`managers/schedule_policy.py:1186-1197`,
continuation `scheduler.py:3280-3282`). True mixing is opt-in `--enable-mixed-chunk`, **default
False** (`:981-985`).
**vLLM V1 has no chunk size at all** — one shared `token_budget` (default
`max_num_batched_tokens=2048`, `config/scheduler.py:42`) spent across running **and** waiting in the
same `schedule()` (`v1/core/sched/scheduler.py:496,525,751`), commented *"There's no 'decoding phase'
nor 'prefill phase' in the scheduler"* (`:478-487`). **At that default a 512-token prompt leaves
1,536 tokens of budget for concurrent decodes in the same forward.**
**Ours: `prompt_chunksize` is a comment only** (`inputs_processor.rs:192`) ⇒ **5,876 ms prefill @ N=512.**

## 7. 🚨 STRATEGIC — **TurboQuant KV IS NO LONGER DIFFERENTIATED**
**vLLM ships it in-tree:** `vllm/v1/attention/backends/turboquant_attn.py` (**1,272 lines**), four
presets including `turboquant_k3v4_nc`, selectable via `--kv-cache-dtype`
(`vllm/config/cache.py:28-30`, `quantization/turboquant/config.py:20-40`).
**The trellis weight format IS still unique** — `grep -ri "qtip\|trellis"` over both trees returns
**zero**. ⇒ **Half the stated moat is now upstream. Re-plan the moat around the trellis, not the KV.**

## 8. THE 448 RECORD HOLDS — but SGLang wrote a model-specific backend rather than bending the model
`decode.cuh:1107` gives `vec_size_ckv=14`, failing `vec_dtypes.cuh:1362`'s `%8==0`. **They did not
work around the generic kernel — they wrote `DeepseekV4AttnBackend` asserting `head_dim==512`
(448 nope + 64 rope, `deepseek_v4_backend.py:525-528`) calling FlashMLA with sparse indices, FP8 KV,
sinks and a second key set (`:1708-1723`) — exactly the four features our record says the generic
kernel lacks.** ⇒ **The answer to "why can't V4 use paged attention" is that nobody wrote the
model-specific backend. They did.**

---

## HAVE-IT / HAVE-IT-DARK / LACK-IT, mapped to our measured terms
| capability | Arc | our own number |
|---|---|---|
| full-forward graph capture | **DARK** — `arc-cuda-graph/src/graph.rs:68-80` logs *"capture is IMPOSSIBLE"*, candle on the legacy NULL stream. Gate now open, 2 blockers named | **49% of the step is GPU idle** |
| device-resident token relay | **LACK** in serving (`sampler.rs:437`); `arc_launch_sampler_bf16` exists, gated off | **517 KB blocking D2H/token** |
| caching / graph allocator | **DARK** — arena on a branch, leaks past its 8 GB cap (`driver_allocs` 1,401/step at step 400+) | 11,436 allocs/token |
| paged KV / block tables | **DARK** — runner proven generic; V4 takes `DefaultInstructions` | — |
| ragged / varlen batching | **DARK** — varlen path exists for prefix-cache; scheduler buckets by choice | 19.6 tok/s @ B=32 |
| chunked prefill | **LACK** — comment only | 5,876 ms prefill @ N=512 |
| mixed prefill+decode step | **LACK** — *but so does SGLang*; only vLLM V1 | — |
| MTP / speculative | **HAVE**, depth comparable to their default | 1.93 @ b=1 |
| trellis weight format | **UNIQUE** | 28.91 inst/weight |
| TurboQuant KV | **NO LONGER UNIQUE** — vLLM ships it | — |

## ⚠️ UNVERIFIED
`sgl-kernel`, DeepGEMM and FlashInfer sources are wheel/external — **SGLang-side inst/weight is
unverified**; the vLLM Marlin counts are **source-derived, not ptxas-verified (±0.25)**.
