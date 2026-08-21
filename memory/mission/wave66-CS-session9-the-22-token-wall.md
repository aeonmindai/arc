# wave66-CS — V4 cannot serve past ~22 tokens, and every single-user throughput number this project has ever recorded was taken inside a crashing run

Parent system: **ArcInfer / ArcKV** (the blocker), with findings in ArcQuant,
ArcGraph, ArcKernels and ArcGate.

Status: **the blocker is open.** One of the seven items below was fixed and
merged tonight (§2, PR #209). The rest are surfaced with owners and probes.

---

## Provenance — read this before quoting anything here

**Every line number in this file was verified against master `9c127a2b1`**, not
against the ref the work was done on. Master moved four times tonight:

```
cc5487ad3  TCFRAG-2B on the tensor cores                        (#203)   <- the serving run ran HERE
02edd31cc  decode raw-window mask is all-ones, -424 launches    (#206)
56c9a99ba  dense-layer inventory: fluent garbage, no fault      (#208)
bb30559a5  register the V4 compressed-KV re-RoPE recompute      (#207)
9c127a2b1  ARC_QTIP_TCFRAG opt-in                               (#209)   <- master now
```

The **serving measurements in §1 were taken at `cc5487ad3`**, on an H200. The
**source claims in §2–§7 are read at `9c127a2b1`**. Where the two disagree —
and for TCFRAG they do, because #209 landed between them — both states are
written out. Nothing in this file was measured tonight on a GPU except §1.

Citations into **candle** carry a file and a symbol and **no line number** on
purpose: candle is not vendored here, so a bare `device.rs` plus a line number
suffix-matches `arc-profiler/src/device.rs` (268 lines) and the citation gate
reports it as out-of-range — which is `SYMBOL-ROT`, the one verdict with no
waiver path. Same convention as wave65-CR, and the gate reproduced the failure
against this very sentence before it was reworded. Do not add them back.

---

# 1. 🔴🔴🔴 THE BLOCKER — V4 CANNOT SERVE. It dies of OOM after 22–30 generated tokens.

**Measured at `cc5487ad3`, H200, batch = 1 decode, 3/3 runs, two independent
arms.** The failure is `CUDA_ERROR_OUT_OF_MEMORY`, and the card is genuinely
full when it happens: **143,151 MiB of 143,771 MiB used**.

Per-token `memory.used`, one run:

| generated token | MiB used |
|---|---|
| 1 | 142,927 |
| 3 | 142,959 |
| 11 | 143,023 |
| 15 | 143,087 |
| **20** | **143,151** |
| 22–30 | **OOM** |

That is **≈11 MiB per generated token**, in ~32 MiB granules.

### It is a LEAK, not a fixed-size overrun

Three observations separate the two, and only the leak explanation survives all
three:

1. **The death token moves with available headroom.** 30 tokens with room, 25
   without, **0 on a process already poisoned by §2**. A fixed-size overrun
   would die at the same token every time.
2. **Memory is never released across requests.** One run ended at **142,959 MiB
   used and the next *started* there.** Nothing between requests gives it back.
3. **The backtrace names an allocation that grows with context**, not a
   constant-size one:

```
CudaSlice<float8::F8E4M3>
  <CudaDevice as BackendDevice>::alloc_uninit
    <Tensor>::cat_contiguous::<&Tensor>
```

### Where the bytes come from

`cat_contiguous` is candle's (`candle-core/src/tensor_cat.rs`, symbol
`cat_contiguous`) — not in this tree, hence no line number. The Arc-side caller
whose width tracks KV length is `compressed_kv_from_rows`
(`deepseek4.rs:1409`), which hands the whole compressed row set to
`forward_at_positions` (`layers.rs:1826`) and concatenates it back together
every step. wave65-CQ derived that at **~22 MB/token of pure re-RoPE traffic**
at 2,048 ctx, from tensor shapes. **That derivation and this measurement are
the same shape and the same order of magnitude, but they are not the same
number and must not be quoted as confirming each other** — 11 MiB/token is what
the card retained, ~22 MB/token is what was moved.

Two mechanisms make retention permanent rather than transient:

* **Every decode step presents a byte size nothing asks for twice**, because the
  size tracks context length. candle's caching allocator files each one
  permanently (candle `AllocCache`, symbol `evict_if_over_capacity` — see §7 for
  why its bound never engages).
* **Freed bytes go into the CUDA async pool, and `cuMemGetInfo`/`nvidia-smi`
  count pool-held memory as used** — the tree says so itself at
  `trim_cuda_memory_pools` (`memory_usage.rs:6-16`). That trim runs **once**,
  post-ISQ (`normal.rs:1432`), and **never between requests**. That is why run
  N+1 starts at run N's high-water mark.

### ⛔ THE CONSEQUENCE, IN THE STRONGEST TERMS THE EVIDENCE SUPPORTS

> **Every single-user throughput number this project has ever recorded was
> measured over ≤30 tokens before a crash.** Not "on an old build" — on a build
> that could not reach token 31.

The best clean fragment measured tonight was **37.35 tok/s over 20 tokens,
`finish=length`, coherent output**. That is a **20-token fragment, not a
steady-state rate**, and it must never be quoted as one. A rate measured over a
window that ends in an OOM is a rate measured while the allocator is filling —
the tail that would drag it down is exactly the part that never ran.

**Anything in `FACTS.md` or `00_RESUME_HERE.md` of the form "V4 b=1 = N tok/s"
inherits this.** See the FACTS retraction block dated 2026-08-21.

### Serve config when measured — and it corrects our own record

```
kv-cache = eager   (PagedAttention OFF)
prefix-cache = ON
max-seqs = 32
```

**Prefix caching was ON.** The unqualified claim *"the default config silently
disables prefix caching"* was already **NARROWED** on 2026-08-19 to "CUDA **and**
PagedAttention **and** standard layout **and** head_dim 128". Tonight's run is
the first direct observation confirming that narrowing on the flagship: V4's
loader returns `false` from `supports_paged_attention`
(`normal_loaders.rs:3269`, whose own comment opens *"Still `false`"*), so the
paged-only predicate is never reached. And the reason it is paged-only is
structural — **TurboQuant is a *paged* cache type**, an enum variant
(`TurboQuant`, `cache_engine.rs:31`) of `PagedCacheType`, so it cannot be
selected on a model that declines PagedAttention.

⇒ **The narrowed row stands; the original claim stays retracted.** Do not
re-broaden it.

### Owner

**PR #213** (`fix/arckv-decode-vram-leak`) is open with a plan: the retention
cap is `min(1 GiB, max(64 MiB, free/8))` planned from reported headroom and
re-armed during decode. Its own reasoning is that the existing 1 GiB constant is
**unreachable** — V4 loads to 142,927 MiB of 143,771, so decode starts with
**~844 MiB free** and a 1,024 MiB cap can never be crossed. **Retention was
bounded on paper and unbounded in practice.**

⚠️ **#213 is UNVERIFIED on hardware.** Its own falsification criterion is
written down and should be honoured: `ARC_ALLOC_CACHE_STATS=32` must show `held`
plateauing at ~105 MiB **and** `free/step` going non-zero within the first few
tokens **and** a generation passing 30 tokens. If `held` plateaus and it still
OOMs, the victim allocation is larger than the headroom and the divisor must not
be re-tuned to hide it.

### 🔴 UNRESOLVED — this measurement and PR #182's do not reconcile, and the gap is ~60 GB

**Both are H200 measurements of the same allocator, and they disagree by two
orders of magnitude on how long a generation survives:**

| | PR #182's leg | tonight, `cc5487ad3` |
|---|---|---|
| tokens generated | **2,600** | **22–30** |
| outcome | **no OOM, in every leg** | **`CUDA_ERROR_OUT_OF_MEMORY`, 3/3** |
| growth rate | **6.04 MiB/token** (+15,700 MiB unbounded) | **~11 MiB/token** |
| free VRAM implied at decode start | **≥ 15.7 GB** | **844 MiB** |
| config named in the artefact | `--max-seq-len 4096` | `max-seqs=32`, prefix-cache ON |

A run cannot grow by 15,700 MiB on a card with 844 MiB free. **So the two runs
did not start from the same residency, and the difference is ~60 GB.** The
likeliest explanation is the serve configuration — `max-seqs=32` and an active
prefix cache against #182's leg, which names neither — but **that is a
hypothesis, not a finding.**

> ⚠️ **This matters because it may change the fix.** If the 142,927 MiB starting
> residency is driven by a batch-32 KV budget and a prefix cache holding
> `n_on_device` entries, then the OOM is a **concurrency-budget defect** at
> least as much as a per-token leak, and re-tuning the allocator cap (PR #213)
> treats a symptom. **Do not let #213 land as "the OOM fix" without this
> settled.**

**The probe is cheap and needs no new instrument:** re-run tonight's arm at
`--max-seqs 1` with the prefix cache off, and read `memory.used` at token 1. If
it starts near 80 GB rather than 143 GB, the config is the cause. Also
establish what model and flags #182's leg used — the PR body does not name the
model, and while its 17–19.8 tok/s and its *"auto-device-map plans 140 GB of a
143 GB card"* both point at V4, **that is inference, not a record.**

### Two smaller retentions named, not fixed

* `prefix_cacher.rs:302` stores `seq.normal_cache().to_vec()`; after
  `clone_out_cache` those tensors are `chunk()` **views** of the cohort's
  batched buffer (`chunk`, `kv_cache/mod.rs:2032`), and candle's `narrow` shares
  storage. One entry pins the whole B-row buffer. **Invisible at B=1** (the
  chunk is degenerate), so it is *not* the measured per-token leak.
* The F8E4M3 dtype in the backtrace is worth one line of caution: V4's FP8 KV
  **code storage** is opt-in — `v4_fp8_kv_enabled` (`deepseek4.rs:3149`) is
  `var == Some("1") && !capture_probe` (`fp8_kv_enabled_from`,
  `deepseek4.rs:3165`), and the caller filters on it (`append_kv_mqa`,
  `deepseek4.rs:2968`). **Whether the failing run had `ARC_V4_FP8_KV=1` set is
  NOT established from the artefacts I have.** Either the run set it, or an
  F8E4M3 `cat` is reachable on a path we believe is off. **That is an open
  question with a cheap probe: re-run with the variable explicitly unset and
  read the backtrace dtype.** Do not assume the first branch.

---

# 2. ✅ FIXED TONIGHT — TCFRAG poisoned the process, and it was default-ON

`qtip2b_tcfrag.cu` was merged by **PR #203** (`cc5487ad3`) — the exact ref §1's
serving run was taken at. Its own header, verbatim, at line 7 of
`qtip2b_tcfrag.cu`:

```
// ⚠️  UNVERIFIED ON HARDWARE — NEVER RUN.
```

The full block is `qtip2b_tcfrag.cu:6-14` and continues *"No part of this file
has executed on a GPU. It has been compiled … and nothing more."*

**And it was the default.** The pre-#209 gate matched only the *off* spellings:

```rust
// at cc5487ad3 .. bb30559a5
*ON.get_or_init(|| match std::env::var("ARC_QTIP_TCFRAG").as_deref() {
    Ok("0") | Ok("off") | Ok("false") | Ok("no") => false,
    _ => true,          // <- unset, and any typo, selected the never-run kernel
})
```

The stated reasoning was *"a typo in an env var must not silently change which
kernel serves production."* **That reasoning is correct and was applied in the
wrong direction**: a typo landed on the unverified side.

**Fixed by PR #209, merged tonight** (branch commit `29fd0da`, merge
`9c127a2b1` — note the merge is a combined diff, so `git show 9c127a2b1 --
<path>` prints **nothing**; use the branch commit or `git diff bb30559a5
9c127a2b1`). `tcfrag2b_enabled` (`cuda_ops.rs:1612`) now delegates to
`tcfrag2b_enabled_from` (`tcfrag2b.rs:272`), which is `value == Some("1")` —
opt-in, and only that exact value. Polarity is pinned by a test,
`tcfrag_gate_is_opt_in_and_only_literal_one_enables` (`tcfrag2b.rs:652`).

### ⚠️ Two corrections to how this was first reported

**(a) The poisoning is real, but it is NOT a panic.** The allocating
initializer is `tcfrag_words` (`bitshift.rs:1334`), a
`OnceLock<Option<Tensor>>` whose body calls `tcfrag2b_repack_cuda`
(`bitshift.rs:1348`). Every failure inside it is `?`/`bail!`, caught by an
`Err(e)` arm that logs and returns `None` (`bitshift.rs:1357-1362`). There is no
`unwrap`, `expect`, `panic!` or `assert` on that path. **The mechanism is that
`OnceLock::get_or_init` caches the `None` permanently** — one transient
allocation failure disables TCFRAG for that weight's entire process lifetime
with no retry. Same blast radius, different fault. *Writing "panics inside the
initializer" would send the next reader looking for a panic that is not there.*

**(b) "The live b=1 trellis path" was true at `cc5487ad3` and is not true
now.** The dispatch sites are the b=1 ones — dense single-token decode
(`fused_gemv_2b_cuda`, `bitshift.rs:1423`, guarded by `n_tokens == 1`) and MoE
on-device gather decode (`gather_gemv_2b_cuda`, `bitshift.rs:1712`) — and the
`.cu` header itself claims *"the whole of the b=1 decode path"*. After #209 the
default is OFF, so at `9c127a2b1` it carries **zero production traffic**. Phrase
it as "was the default b=1 trellis path at `cc5487ad3`; now opt-in."

### The decline evidence, and its limits

**Reported from the box: TCFRAG declined 4 of 4 loads and never once
succeeded.** The log line that would say so is the only `warn!` on that path:

> `"qtip2b TCFRAG repack declined for [{num_experts}, {n_rows}, {packed_per_row}] (k_in={}): {e}. Keeping the shipped GEMV."`

at `bitshift.rs:1358`. ⚠️ **That warn emits no count and aggregates nothing, and
the string "4 of 4" appears nowhere in this repo — `memory/` contains no mention
of TCFRAG at all before this file.** So "4 of 4" is a **run observation held
outside the tree**, not something a later reader can re-derive from source. If
it is going to be load-bearing, the run log belongs in `arc-tools/`.

### The second reason, which was itself the bug

`tcfrag2b.rs:151-158` documents an **UNMEASURED fp16 overflow risk** — fp16
overflows at 65,504 where bf16 does not; the K=4 probe peaked at 3.3, *"but that
number belongs to that probe, not to DeepSeek-V4"*; above
`TCFRAG2B_MAX_FP16_ACTIVATION` (`tcfrag2b.rs:155`) the mma sees an infinity.

**At `cc5487ad3` that paragraph named the kill switch as its remedy:** *"The
remedy in this change is the kill switch (`ARC_QTIP_TCFRAG=0`), not a silent
clamp."* **A remedy that was opt-out is not a remedy.** #209 rewrote it: the
current text names the remedy as *not enabling the path*, and
`tcfrag2b.rs:160-163` explicitly repudiates the kill-switch framing. *(Line
range corrected: **151-158**, not 151-157 — the sentence runs through 158.)*

---

# 3. 🔴 TWO host round-trips per decode step, not one — and this RETRACTS "op count is not the binding constraint"

Both are on master at `9c127a2b1`:

| # | what | where |
|---|---|---|
| 1 | `cudaStreamSynchronize(self.stream)` after every `cuGraphLaunch` | `cudaStreamSynchronize` (`graph.rs:362`) |
| 2 | greedy `argmax` + a 4-byte D2H `to_scalar::<u32>()` | `argmax` (`sampler.rs:1479`) |

Site 1 is *deliberate and correct* as written — a fault during graph execution
is asynchronous, so discarding the sync's return turns an illegal access into a
silently poisoned context. It is nonetheless a full host round-trip per replay.
Site 2 is a second one, in the sampler, on the same step.

### Why this matters more than it looks

`CAPTURE_LANE.md` recorded a **1,137× cut in launch APIs** (3,961 → 3.8 per
decode step) that bought **~8%** (33.4 → 34.2 tok/s), and the conclusion drawn
was that **op count is retired as a lever**.

> 🔴 **That conclusion is RETRACTED.** With two host syncs per step surviving
> the capture, the experiment did not test "few ops vs many ops" — it tested
> "few ops with two round-trips vs many ops with two round-trips." **The
> round-trips, not the op count, were the invariant.** An 8% result from that
> comparison is not evidence about op count either way.

What survives from CAPTURE_LANE is the *measurement* (3,961 → 3.8; 33.4 → 34.2)
and the finding that kernel **duration** is a large share. What does not survive
is the inference that launch reduction is exhausted.

### The falsifying arm exists

Branch **`arcgraph/device-decode-loop`** @ `1b6949244` carries commit
`31e634f06`, *"device decode loop — N steps, N cuGraphLaunch, **zero host
syncs**"*. It adds `is_greedy_trivial` to the sampler as the exact precondition
for substituting an on-device argmax for the host one. **That is the arm that
settles §3, and it has not been measured.** Until it is, neither "op count is
the lever" nor "op count is retired" is supported.

---

# 4. 🔴 The dedicated / autonomous decode tier cannot accept V4 — architecturally, not by configuration

This is not a flag that is off. The data structures cannot describe the model.

**`LayerWeights` (`weights.rs:430`) has seven non-optional projection slots** —
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
The count is a constant, `DENSE_PROJS_PER_LAYER` (`weights.rs:44`), whose doc
says *"`LayerWeights` has exactly these seven slots"*. The only `Option` fields
in the struct are `q_norm`/`k_norm`, and those are norms. **There is no MLA slot
and no expert slot.**

**`decode_forward` (`decode_forward.rs:700`) computes a fused QKV projection**
(`qkv_fused`, `decode_forward.rs:775`) **plus gate/up/silu/down**
(`gate_proj`, `decode_forward.rs:958`; `down_proj`, `decode_forward.rs:970`) in
a single flat loop with one fixed op sequence per layer. No routing, no expert
loop, no KV compression.

**`DecodeConfig` (`weights.rs:521`) carries twelve scalars and no layer kind.**
A per-layer kind does exist — `LayerCensus` (`weights.rs:61`) — but **only on
the refusal path**: it counts layers to produce the error text at
`weights.rs:152`, *"a MoE or MLA layer is not describable by DecodeConfig"*. It
classifies in order to reject; it does not parameterise.

> ⇒ **Accepting V4 here would trade a named refusal for a wrong-tensor read.**
> The refusal is the correct behaviour today. Making the tier *useful* for V4
> needs a layer **kind** threaded through `DecodeConfig` → `LayerWeights` →
> `decode_forward`, which is a design change, not a wiring fix.

### And even for a model it *can* describe, capture is not wired

`normal.rs:2619` is an **unconditional `return Ok(None)`** inside
`autonomous_decode` (`normal.rs:2379`). The comment above it
(`normal.rs:2568-2601`) is explicit: *"CAPTURE IS NOT WIRED. This return is
unconditional, and it is guarding work that was described and never written —
not a leftover."* The two missing pieces:

1. **`has_cached_kv_info` has no definition anywhere in the workspace.**
   ⚠️ *Correction to the first report: it is not "zero occurrences" — there are
   **three**, at `normal.rs:2370`, `normal.rs:2575` and `normal.rs:2604`, and
   all three are prose or a log string. `grep "fn has_cached_kv_info"` returns
   nothing.* And the comment that once claimed it had been added has **already
   been corrected in-tree**: `normal.rs:2575` now reads *"The comment this
   replaces said it was 'added in the same change as the accessor below'. It was
   not — there is no such method on `DedicatedDecodePath` anywhere in the
   workspace."* Cite `:2575` as the refutation, not as the false claim.
2. **`paged_state_factory` is un-buildable at the call site.** The parameter is
   `capture_via_decode_forward` (`autonomous.rs:213`), bound
   `F: Fn() -> PagedAttentionState`. The only producer of a state with live
   device pointers is `build_paged_attn_state` (`dedicated.rs:1085`), which
   takes `&mut self` *and* an already-built `&PagedAttentionState`. At
   `autonomous_decode` neither exists: the context carries host-side `Vec`s
   only, and `self.autonomous_runner` is already mutably borrowed, so a second
   `&mut self` borrow to form the closure will not compile.
   `arc-cuda-graph/src/lib.rs:165` says the same independently.

⚠️ **Provenance correction: `ba026e9d1` did not introduce this method.** It
resolves, but it is a **root/import commit** — 1,430 files, 604,635 insertions,
0 deletions — and `git rev-list --max-parents=0 HEAD` lists it. `git log -S "fn
autonomous_decode"` returning only that SHA means "present since the initial
import", which is true and vacuous. The commit that authored the current
"CAPTURE IS NOT WIRED" analysis is **`6e58c5dbf`**. Cite that one.

---

# 5. 🔴 The b=1 FP8 GEMV is INSTRUCTION-bound — this retracts the "4% memory-controller utilisation ⇒ latency-bound" story

The kernel is `fp8_gemv_warp` (`blockwise_fp8_gemm.cu:144`). Line 199 of that
file, the continuation of a statement opened at 198:

```c
const float scale =
    __ldg(&weight_scale[scale_row_offset + k / block_size_x]);
```

`block_size_x` is a **kernel parameter, plain signed `int`, not `const`**
(`blockwise_fp8_gemm.cu:150`), and it stays `i32` all the way out through the
Rust FFI. NVIDIA GPUs have no integer-divide instruction, so nvcc expands this
to a reciprocal-and-fixup sequence; the *signed* form needs the extra
sign-fixup. It runs **once per four weight bytes** — one `uint32_t` weight load
at `blockwise_fp8_gemm.cu:171`, one divide at `:199`, same loop body. There is a
second identical divide in the scalar remainder loop at
`blockwise_fp8_gemm.cu:215`.

b=1 reaches it because `use_gemv` is `m <= fp8_gemv_max_m()` (default **4**,
`mistralrs-quant/src/blockwise_fp8/ops.rs:914`) and
`mistralrs-quant/src/blockwise_fp8/ops.rs:1140` selects it.

### Two precision corrections to how this was first stated

* **It is NOT loop-invariant.** `k = k_base + lane*4` and `k_base` advances by
  128 per trip, so the quotient changes every iteration. Nothing is hoistable.
  *(This strengthens the argument — do not write "loop-invariant".)*
* **Warp-uniformity holds for the shipped configuration, not by
  construction.** At `block_size_x == 128` all 32 lanes span
  `k_base..k_base+124` and compute the identical quotient. The only precondition
  the dispatcher enforces is `block_size_x % 4 == 0`
  (`mistralrs-quant/src/blockwise_fp8/ops.rs:1139`), and at 64 the warp
  diverges. Say "warp-uniform at the 128-wide block size this path actually
  runs."
* **Loads are already `__ldg` and already 32-bit** (`blockwise_fp8_gemm.cu:171`),
  not scalar. The honest framing of the fix is **32-bit → 128-bit widening**,
  not "scalar → vectorised."

### The arithmetic — corrected, and it still lands

The original form was `132 SM × 4 sched × 32 lanes × 1.98 GHz = 33.4 T inst/s ÷
(45 inst / 4 B) = 2.97 TB/s`. Each step reproduces: 132 × 4 × 32 = 16,896 lanes;
× 1.98e9 = **3.345e13**; ÷ 45 × 4 = **2.97 TB/s**.

🔴 **But 128 lanes/SM/clk is the FP32 warp-issue rate, not the INT32-ALU rate.**
GH100's SM has 128 FP32 units and only **64 INT32**. An emulated signed `idiv`
is IADD3/LOP3/SHF/ISETP-dominated — the 64-wide pipe. This repo already carries
the right figure: `00_RESUME_HERE.md:729` reads *"~1.67e13 INT32-lane/s"*, and
132 × 64 × 1.98e9 = 1.673e13 reproduces it exactly. On that roof the ceiling is
**1.49 TB/s**.

> **The defensible statement is a RANGE: 1.5 TB/s (INT32-pipe bound) to 3.0 TB/s
> (absolute warp-issue ceiling), against a 4.8 TB/s nameplate and a ~4.0 TB/s
> achievable memory roof. Both ends are under the roof, so the instruction-bound
> verdict holds *a fortiori*.** Quoting 2.97 alone is the most generous end and
> invites the correct objection that it used the FP32 rate.

⚠️ **"45 instructions per 4 bytes" is DERIVED, not measured.** It back-solves
from the PR's own prose ("15-20 instructions", "~40% of the loop body";
17.5/0.40 ≈ 44). Self-consistent, but it is an estimate of an estimate, and
`00_RESUME_HERE.md:725` states the house rule: *"prototype → disassemble →
confirm the instructions actually move → THEN predict."* **The probe that closes
this is a `cuobjdump -sass` count of the `fp8_gemv_warp` inner loop**, with
precedent tooling on branch `perf/qtip-sass-census`.

### The replacement

**PR #210** (`perf/fp8-gemv-wide-b1` @ `4b3f35a24`), open. Adds `fp8_gemv_wide`
behind `ARC_FP8_GEMV_WIDE=1`, **default OFF**: `uint4` loads, `k >> scale_shift`
with the host passing `log2(block_size_x)` and refusing non-powers-of-two, a
2-deep explicit software pipeline, four accumulators. Its own body says
**UNVERIFIED ON HARDWARE**, the `.cu` is uncompiled, and it is **not
bit-identical** (f32 re-association). Its projected ~1.6× on the kernel /
~1.3 ms/token is a **projection**. Do not let it be quoted as a measurement.

⚠️ **And the 16.0% share it multiplies has the same provenance problem as §6's
27.2%** — see the FACTS retraction.

---

# 6. 🔴 Four instrument failures — the tally grows, and two of my own framings needed correcting

### (a) `mark_unreachable` is a no-op unless `ARC_PROFILE=1` — TRUE, **but not new**

⚠️ **Attribution first: this was already on the record.**
`00_RESUME_HERE.md:231` has said since session 8 that *"`mark_unreachable` — the
registry for dark features — is ITSELF dark: inert unless `ARC_PROFILE=1`"*.
**What is new here is the consequence, which nobody drew, and the exact site
list.** Filing it as a fresh discovery would be its own kind of drift.

`mark_unreachable` (`arc-profiler/src/lib.rs:467`) returns immediately unless
`enabled()`, and the gate is `std::env::var("ARC_PROFILE").ok().as_deref() ==
Some("1")` (`arc-profiler/src/lib.rs:124`), latched once per process behind a
`Once`. `ARC_PROFILE=true`, `=yes` and `=0` all read as OFF.

⇒ **Every reachability probe in `normal.rs` has been dark in ordinary runs, and
we have been reading their silence as evidence.** That is the same fault as the
MHC "no output ≠ no execution" retraction, by a different road.

⚠️ *Correction: there are **6** call sites in `normal.rs` — `:1556`, `:1635`,
`:1921`, `:2404`, `:2474`, `:2602` — not seven; the seventh grep hit
(`normal.rs:2399`) is prose. Workspace-wide: 14 invocations, 11 of them
non-test.*

### (b) `ci_cuda.yaml` gates nothing — TRUE

`ci_cuda.yaml:3` is the whole `on:` block, and it contains only
`workflow_dispatch:`. No `push`, no `pull_request`. A redundant job-level guard
sits at `ci_cuda.yaml:12`, and the runner is
`[self-hosted, Linux, ARM64, gpu, cuda]` at `ci_cuda.yaml:18`. Its own comment
says *"Keep it manual so normal PRs are not blocked."*

**The real lane is `cuda-typecheck`, at `cuda_compile_check.yaml:337`** — job
name at `:338`, `runs-on: ubuntu-latest` at `:339`, triggers at
`cuda_compile_check.yaml:109` (`push` to master, `pull_request` on `'**'`,
`workflow_dispatch`), path-filtered on `.rs`/`.cu`/`.cuh`/`Cargo.toml`.

⚠️ *Scope note: branch-protection "required check" configuration is not in the
repo, so "gates nothing" is an inference from the trigger block — a sound one,
since a workflow that never runs on a PR cannot be a passing required check, but
it is an inference.*

### (c) `ARC_NO_DEFERRED_FREE` — 🔴 **MY FRAMING WAS OVERSTATED. Corrected here.**

The claim was that it "makes a disabled capture indistinguishable from a clean
one." At `9c127a2b1` that is too strong, and the reasons matter:

* It has **four occurrences, all in one file, and zero production readers** —
  `arc-cuda-graph/examples/capture_probe.rs:19` is the only read, and the value
  it binds is used for exactly one thing: the `println!` on the next line, which
  prints `deferred_free=false`. **The run does self-identify.**
* Its *behaviour* is not implemented here at all. It lives in the pinned candle
  fork, which is not vendored.

**What is true is the weaker version, and it is still a real hazard:** the
success banner is identical either way. `capture_probe.rs:88` prints
`CAPTURE+LAUNCH OK. max|captured-eager| = …` on both paths, and the only
discriminator is one token **69 lines earlier**. Write it as *"the success
banner is identical and the only discriminator is an easily-missed earlier
line."* Not "indistinguishable."

*Counter-evidence in the same neighbourhood, so this is not filed as an open
D18 case:* `arc-cuda-graph/src/lib.rs:124` now logs `INERT` instead of a success
line, and `arc-cuda-graph/src/graph.rs:161` calls `mark_unreachable` on the
NULL-stream path.

### (d) Presence-tested env flags, where `=0` turns a feature ON — TRUE, with the label corrected

**21 distinct env flags are read presence-only** (`.is_some()` ×17,
`.is_none()` ×9 across 26 sites). ⚠️ **But only 18 of the 21 are `ARC_*`** — the
other three are legacy `V4_`-prefixed (`V4_NAN_DEBUG` `deepseek4.rs:3334`,
`V4_STATS` `:3364`, `V4_TRACE` `:3449`). Widening to the same bug class via
`env::var(..).is_ok()/.is_err()` adds four more `ARC_*` names and reaches **22
`ARC_*`**; restricting to shipping `src/` gives **16**. Either framing reaches
"21+"; **do not attribute all 21 to `ARC_*`**.

Representative sites, each with its flag on the cited line:
`ARC_MOE_SLOW` (`experts.rs:268`) · `ARC_TIME_DECODE` (`deepseek4.rs:3283`) ·
`ARC_SYNC_ISQ` (`isq.rs:34`) · `ARC_COLLAPSE` (`deepseek4.rs:2219`).

> ⇒ **Any past A/B that used `FLAG=0` as its "off" leg was never a
> comparison — both arms ran with the feature on.** This is not hypothetical:
> `normal.rs:216` records exactly that happening to `ARC_NO_DEDICATED_DECODE`,
> where two harnesses passed `=0` as their control.

**The canonical fix already exists in-tree and these 18 do not use it**:
`env_flag_is_set` (`normal.rs:232`) over `env_flag_value` (`normal.rs:252`),
pinned by `zero_and_its_spellings_are_off_not_merely_present`
(`normal.rs:2977`), which asserts `["0","false","no","off","OFF"," 0 ","False",""]`
all read as OFF. **PR #212 (`arcgate/env-flag-value-semantics`) is the sweep and
is open.**

---

# 7. PR #182 is a PRECONDITION, not a fix — and its hazard is currently MASKED by §1

`perf/alloc-cache-bounded` (#182) is the arc-side landing of the bounded
allocator; **its candle side is `b2a4dd80` → `89ab14ef`, one of the two lines
merged into `9586979d`** on `aeonmindai/candle` (the other being the capture
branch's `9211966e`). Both sides are already ancestors of arc master.

**Before it there was no eviction at all** — the only ways memory went back to
the driver were `set_alloc_cache_enabled(false)` and
`drain_alloc_cache_and_free()`, and one long generation has one prefill, so
nothing ever drained. #182 measured that as **+15,700 MiB over 2,600 tokens
unbounded vs +2,004 MiB capped, for 0.15 ms/token.**

> **So #182 converts a bounded memory leak into a correctness risk that is live
> exactly when a graph is live.** Still the right trade — but a trade, and now
> on the record as one. The hazard is wave65-CR / **PR #211**: LRU eviction
> stores `(bytes, ptr)` and **cannot express "this address is baked into a
> `CUgraphExec`"**. A `cuGraphLaunch` executes baked addresses and calls no
> `cache_take`, so replay never refreshes a tick — the graph's buffers age
> monotonically from the moment capture ends, while eager verify-replays keep
> stamping every *other* size.

### 🔴 The masking, and why it is the most misquotable thing in this file

**That hazard is currently masked by the OOM in §1.** A process that dies at
token 22 never keeps a `CUgraphExec` alive long enough for its baked buffers to
age to the front of the LRU queue.

> ⇒ **Any probe of hazard 1 today returns a green the instrument never had a
> chance to fail.** `alloc_cache_stats().evicted_at_demand_size == 0` is the
> probe `9586979d` adds for exactly this, and it will read zero for the wrong
> reason until §1 is fixed.
>
> **Fixing the OOM UN-MASKS it. #211's pin must land before capture is defaulted
> on.**

Not reachable in the shipping configuration today for two independent reasons:
capture is behind `ARC_V4_CAPTURE_PROBE`, unset by default; and `arc-cuda-graph`
drains the cache before every `cuMemPoolDestroy`
(`drain_alloc_cache_and_free`, `graph.rs:56`; `drain_alloc_cache`,
`graph.rs:214`). Neither survives capture being defaulted on.

---

## Open, with the probe named

| # | question | the probe that settles it | owner |
|---|---|---|---|
| **1** | **🔴 Why does decode start at 142,927 MiB here and ≥15.7 GB free in #182's leg?** Blocks calling #213 "the OOM fix". | re-run at `--max-seqs 1`, prefix cache off; read `memory.used` at token 1. Separately: name #182's model and flags | **UNOWNED** |
| 1 | Does #213's re-armed cap let a generation pass 30 tokens? | `ARC_ALLOC_CACHE_STATS=32`: `held` plateaus ~105 MiB **and** `free/step` non-zero **and** run passes 30 tok | PR #213 |
| 1b | Was `ARC_V4_FP8_KV=1` set on the failing run, or is an F8E4M3 `cat` reachable with it off? | re-run with the variable explicitly unset; read the backtrace dtype | UNOWNED |
| 3 | Are the two host syncs the real limiter, not op count? | measure `arcgraph/device-decode-loop` @ `1b6949244` against master, same binary | UNOWNED |
| 5 | Is `fp8_gemv_warp` really ~45 inst / 4 B? | `cuobjdump -sass` on the inner loop; tooling on `perf/qtip-sass-census` | PR #210 |
| 5b | Does `fp8_gemv_wide` compile, and what does it measure? | first CUDA CI run is the syntax gate; then an A/B | PR #210 |
| 7 | Does eviction reach a live graph's baked buffers? | `evicted_at_demand_size == 0` — **only meaningful after §1 is fixed** | PR #211 |

## Related

* **§1** → PR #213, and wave65-CQ (`wave65-CQ-v4-compressed-rope-recompute.md`)
  for the `cat` that sets the rate.
* **§2** → PR #203 (introduced), PR #209 (fixed, merged).
* **§3** → `CAPTURE_LANE.md` (the retracted conclusion), branch
  `arcgraph/device-decode-loop`.
* **§5** → PR #210, PR #200/#201 (the FP8 crossover).
* **§6** → PR #212.
* **§7** → PR #211 / wave65-CR, PR #182, PR #205 (capture lane).
* Filed here rather than as a GitHub issue because `aeonmindai/arc` has issues
  disabled; `memory/mission/` is the tracker this project actually uses.
