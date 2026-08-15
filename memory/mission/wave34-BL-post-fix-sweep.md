# wave34-BL — post-fix sweep on the PUBLISHED artifact

**Session:** 2026-08-15, Runcrate H200 (Helsinki), instance `arc-s9-h200`
**Rate:** **$4.85/hr**. Every `$/Mtok` below carries this rate.
**Binary:** arc **`3460656d3`** (origin/master), `cuda flash-attn`, **no `cudnn`**.
Built on-box in **6 m 36 s**.
**Card:** NVIDIA H200, `nvidia-smi` reports **143,771 MiB**, driver 580.126.09,
700 W limit; 44 vCPU, 185 GB RAM, 1.5 TB disk. CUDA toolkit 12.4.

**Box health gate: full PASS** — `sustained power 292W (41% of 700W limit,
peak 294W) >= 200W floor`, `SM clocks 1980/1980 MHz (100%) under load`,
PCIe gen 5/5 x16/x16, 44 cores, 1301 GB free. (The first gate run reported
`WARN no synthetic load available`; CUDA was installed but not on `PATH`, so
the power check — the one that catches the s5a transfer-starved signature —
had silently not run. Re-run with `--burn nvcc` after exporting
`/usr/local/cuda/bin`. **A gate that skips its own most important check must
be treated as a failed gate, not a passed one.**)

---

## 0. The artifact loaded — this is the first sweep on the PUBLISHED artifact

Unlike wave26-AX, which measured an **in-situ `ARC_QTIP_BEAM=32` bake**, this
session measured the published **W=256** artifact
`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` as a customer would load it.

```
mistralrs serve -p 1234 --max-seqs 256 --prefix-cache-n 0 \
  -m /workspace/src \
  -a deepseekv4 \
  --from-uqff /workspace/uqff/qtip2-0.uqff \
  --chat-template chat_templates/deepseek_v4.json
```

| evidence | value |
|---|---|
| `Auto-discovered 8 UQFF shard files (from 1 specified)` | yes |
| `Applying ISQ` in the log | **absent** — a real UQFF load, not a re-quantize |
| Cold load, `Loading` → `Server listening` | 21:39:32Z → 21:41:43Z = **2 m 11 s** |
| Dummy run | **completed in 3.19 s** (it fails on the bake box; it passes here) |
| Resident after load | **78,865 MiB of 143,771** |
| Integrity, verified before serving | 8 shards ✓, `residual.safetensors` **1,293,806,700 B exactly** ✓, 46 source shards ✓ |

Downloads: overlay **74,190,223,041 B**, source **159,630,129,669 B**,
sustained **~280–334 MB/s**.

> The `-m` must point at the SOURCE checkpoint; the artifact is an overlay.
> That is unchanged from wave26-AX and is documented on the model card.

---

## 1. Harness trust — the concurrency assertion can fail, and did not

Both halves of DOCTRINE D12's requirement:

- **Offline, before renting**: `test_batch_load_probe.py` — `ALL PASS`,
  including `mutation(serial): probe exited 1, effective_B 1/4, mean in-flight
  0.76, headline suppressed — the assertion CAN fail` and
  `mutation(cap 2 of 8): FAIL at default gate`.
- **On this box**: `CONC[B=8] effective_B=8 mean_inflight=5.67
  server_prefill_batch=3 verdict=pass`.

`--max-seqs 256` was set on the serve process for every row below, so no row
is a silently-capped B=32 run. `effective_B` is carried on every row.

---

## 2. THE SPEED TABLE [measured]

**Protocol.** One H200 @ **$4.85/hr**. Model: DeepSeek-V4-Flash, the
**PUBLISHED** `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` artifact (qtip2, Viterbi
**beam W=256**, hadamard-128, mse) overlaid on the source checkpoint. Server:
`--max-seqs 256 --prefix-cache-n 0`, chat template supplied. Probe:
`arc-tools/quality/batch_load_probe.py`, `/v1/chat/completions` streaming,
distinct ~68-token prompts, **64 decode tokens, 1 rep**, `--max-ctx 545000`,
temperature 0.

> **Protocol note, load-bearing for the comparison.** 64 decode tokens / 1 rep
> is **wave26-AX's protocol, chosen deliberately so the rows compare line by
> line** — it is *not* the runbook's richer `--reps 3 --max-tokens 256`.
> The pre-fix baseline was also taken on an **in-situ `ARC_QTIP_BEAM=32` bake**,
> not on this published W=256 artifact. Both differences are stated on every
> comparison below.

### 2.1 The table

| B | prefill agg tok/s (TTFT-derived, **lower bound**) | prefill agg tok/s (server-instrumented, **compute only**) | decode tok/s per user (p50) | **decode AGGREGATE tok/s** | TTFT p50/p95 (s) | $/Mtok @ $4.85/hr | effective_B | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 *(diagnostic only — never a headline, D2)* | 77.81 | 78.14 | 16.02 | **16.27** | 0.822 / 0.822 | 82.80 | 1 | exempt |
| 8 | 73.17 | 79.50 | 4.11 | **26.96** | 4.258 / 7.187 | 49.97 | 8 | pass |
| 16 | 75.18 | 81.76 | 2.37 | **30.65** ← peak | 7.367 / 12.989 | **43.96** | 16 | pass |
| 32 | 78.30 | 84.30 | 1.25 | **30.59** | 16.611 / 26.711 | 44.04 | 32 | pass |
| 64 | 79.40 | 85.66 | 0.59 | **30.41** | 27.590 / 52.689 | 44.30 | 64 | pass |
| 128 | 103.66 | 114.34 | 0.27 | **28.86** | 48.467 / 80.698 | 46.68 | 128 | pass |
| 256 | 128.32 | 156.78 | 0.08 | **19.02** | 70.659 / 130.385 | 70.83 | 256 | pass |

```
BATCHSWEEP[s9_sweep_a]: peak decode 30.65 tok/s @B=16 (effective_B=16) | errors 0
BATCHSWEEP[s9_sweep_b]: peak decode 28.86 tok/s @B=128 (effective_B=128) | errors 0
```

**Every row ran at full requested concurrency.** `effective_B == B` on all seven
rows, every verdict `pass`, **0 errors out of 505 requests**. `--max-seqs 256`
was set, so no row is a silently-capped B=32 measurement. **No row was dropped
and none needed to be**: the KV guard printed `KV[B=256]: est ~33792 tokens`
against the 545,000-token budget and **no `WARNING[KV]` fired on any row**.
`WARN[CLIENT]` never fired either.

**B=256 fitted, measured on the card:** peak resident **114,665 MiB of 143,771**
(≈ 29 GB spare) with 256 sequences in flight. PR #59's arithmetic held.

### 2.2 What the fixes bought [measured, same probe, same 64-token/1-rep protocol]

| B | pre-fix agg (wave26-AX, in-situ **W=32** bake) | post-fix agg (this session, published **W=256** artifact) | ratio |
|---|---|---|---|
| 1 | 15.35 | 16.27 | 1.06× |
| 8 | 14.83 | 26.96 | **1.82×** |
| 16 | 10.31 | 30.65 | **2.97×** |
| 32 | 5.07 | 30.59 | **6.03×** |
| 64 | 8.14 | 30.41 | **3.74×** |
| 128 | *never ran* — 3.91× out of memory before PR #59 | 28.86 | n/a |
| 256 | *never ran* | 19.02 | n/a |

**The headline change is the SHAPE, not any single row.** Pre-fix, peak
aggregate was at **B=1** and aggregate *fell* with batch — the opposite of the
capacity claim. Post-fix, peak aggregate is at **B=16**, and it is **2.00× the
pre-fix peak** at **half the cost per token** ($43.96 vs $87.77 /Mtok).
Under **D2** the headline is now an actual batch row rather than a single stream.

> Two protocol differences are folded into every ratio above and neither is
> controlled for: the pre-fix rows came from an **in-situ `ARC_QTIP_BEAM=32`
> bake**, these from the **published W=256 artifact**; and this box is a
> different Helsinki rental. b=1 moved only 15.35 → 16.27 (**1.06×**), which is
> the evidence that the artifact/box swap is *not* what produced the 3–6×
> gains at batch — those rows are the fixes.

### 2.3 Does aggregate still equal 1 / per-sequence-step-time? — **the sharp test**

**Answer: the identity is unfalsifiable, and the signature it was standing in
for is BROKEN below B=128 and BACK at B=256.**

First, the honest caveat, because it changes what the test means.
In wave27-AY's table `step_ms = running / aggregate` and
`ms-per-seq-in-step = step_ms / running`, so **`ms-per-seq-in-step ≡ 1000 /
aggregate` is an algebraic identity of that construction** — it holds for any
numbers whatsoever and no measurement can refute it. What was falsifiable, and
what actually carried the diagnosis, is the **shape** of that per-sequence cost
as B grows. Pre-fix it was **flat** (65.1 → 67.4 ms, i.e. batching bought
literally nothing) and then **doubled** across the MoE gather cap
(97.0 → 197.2 ms).

| B | agg tok/s | ms per seq in step (= 1000/agg) | pre-fix ms per seq in step |
|---|---|---|---|
| 1 | 16.27 | **61.5** | 65.1 |
| 8 | 26.96 | **37.1** | 67.4 |
| 16 | 30.65 | **32.6** | 97.0 |
| 32 | 30.59 | **32.7** | 197.2 ← the cap crossing |
| 64 | 30.41 | **32.9** | 122.8 |
| 128 | 28.86 | **34.7** | — |
| 256 | 19.02 | **52.6** | — |

- **The MoE-cap doubling is gone.** There is no longer any discontinuity between
  B=8 and B=32. PR #56 did what it was predicted to do.
- **Amortization now exists, but only up to B=16**: per-seq cost falls
  61.5 → 32.6 ms, a **1.89×** gain, then is flat within 1% across
  B=16/32/64 (32.6 / 32.7 / 32.9).
- **So the prediction in the brief was right and should be recorded as right:**
  "stops collapsing, does not start scaling." Aggregate plateaus at
  **~30.5 tok/s** from B=16 to B=64 instead of collapsing to 5.07.
- **A NEW wall appears at B=256** — per-seq cost rises 34.7 → 52.6 ms. That one
  is not the MoE cap; see §2.4.

### 2.4 The B=256 regression is host-CPU starvation, and it was caught on the card

While B=256 was decoding, with `256 running, 0 waiting` in the scheduler:

```
mistralrs process   100.0% CPU   (ONE core of 44; load average 1.13)
GPU                 0–4% util, 121–123 W of a 700 W limit
```

For contrast, B=64 and B=128 ran the same box at **420–476 W and 97–100% GPU**.
The client was idle, so this is **not** `WARN[CLIENT]` — the server is
host-bound and the GPU is starved. This is the `per_step_host_overhead` item
CEILINGS.json already lists as OPEN; what is new is that it becomes the
*dominant* term at B=256 and drags aggregate **below** the B=128 row.

**Ruled out, with evidence:** the CPU sampler fallback. The log carries exactly
**2** `GPU radix top-k sampling failed; falling back to CPU: tensor_device_ptr:
unsupported dtype I32` lines in the whole session, both from the startup dummy
run and none during any measured row — the probe sends `temperature=0`, whose
greedy path returns before the radix branch. wave26-AX flagged this as "the
single highest-value thing to check next"; it is now **measured and refuted**,
matching the refutation already recorded in CEILINGS.json.

### 2.5 Against the physics ceiling (CEILINGS.json, read before quoting)

Those ceilings assume **uniform random routing**, which *maximises* distinct
experts woken — so they are **lower bounds on the ceiling**, not ceilings.

| B | measured agg | ceiling agg | gap |
|---|---|---|---|
| 1 | 16.27 | 1,413 | 87× |
| 16 | 30.65 | 3,144 | 103× |
| 64 | 30.41 | 5,289 | 174× |
| 256 | 19.02 | 16,602 | 873× |

**Every one of those gaps is implementation, not physics.** None of it is a law;
all of it is code we control. The fixes measured here closed 3–6× of it at
mid-batch in one day.

---

## 3. Component profile — captured at B=1 AND B=64 [measured]

Captured on a **separate serve process** with `ARC_TIME_DECODE=1`, so the sweep
above is unpolluted. **The profiler's syncs inflate absolute times** — read the
percentages, not the milliseconds. Build is `cuda flash-attn`, **no `cudnn`**.

**The profile this replaces was taken on a cudnn build we abandoned at −62%
decode, at b=1 only.** It said `mla_attn 49% · mhc_attn_pre 16% ·
mhc_ffn_pre 16% · moe 16%`. It is superseded.

### B=1 (`forward_total = 67.24 ms`)

| component | ms | share |
|---|---:|---:|
| `mla_attn` | 24.45 | **36%** |
| `moe` | 23.65 | **35%** |
| `mhc_ffn_pre` | 6.95 | 10% |
| `mhc_attn_pre` | 6.92 | 10% |
| `mix_post_attn` | 2.64 | 4% |
| `mix_post_ffn` | 2.63 | 4% |

MLA interior: `q_proj 2.57 · kv_proj_rope 0.91 · invrope_oproj 2.94`
⇒ SDPA itself ≈ **18.0 ms**, i.e. ~74% of `mla_attn` and ~27% of the step.

### B=64 (`forward_total = 210.00 ms`, settled sample)

| component | ms | share |
|---|---:|---:|
| `moe` | 103.92 | **49%** |
| `mla_attn` | 82.86 | **39%** |
| `mhc_ffn_pre` | 8.76 | 4% |
| `mhc_attn_pre` | 8.69 | 4% |
| `mix_post_ffn` | 2.98 | 1% |
| `mix_post_attn` | 2.80 | 1% |

MLA interior: `q_proj 23.79 · kv_proj_rope 7.81 · invrope_oproj 17.44`.
Earlier, less-settled samples in the same run put `moe` as high as **64%**
(`forward_total = 512.74 ms`).

### What the profile says about where the next month goes

> 🔑 **`moe` share GROWS with batch: 35% at B=1 → 49–64% at B=64, while
> `mhc_attn_pre` and `mhc_ffn_pre` COLLAPSE from 10% to 4%.**

That is the component-level fingerprint of the same fact §2.3 measured
end-to-end: the fused MoE path is **flat per token and does not amortize** (one
GEMV per `(token, expert)` pair, no dedup), so its cost scales with batch while
everything around it amortizes. **At fleet batch sizes MoE is the dominant term
and it is the term that does not get cheaper with batch.** The trellis
grouped-GEMM with cross-token dedup — already named the keystone in the goal
audit — is now confirmed as the right next target by an *end-to-end profile at
batch*, not by a b=1 microbench on an abandoned build.

Second target, from §2.4: **per-step host overhead**, which is what caps B=256.

### MTP acceptance — why three sessions got empty files

**Nothing emitted, and the reason is now known and is not a logger bug.**
`ARC_MTP_LOG_ACCEPTANCE=1` was set on the profiling server. The serve log says:

```
UQFF artifact contains 3 MTP decoder block tensors but the MTP block is
not loaded (`--mtp-depth 0`); skipping them.
```

**The artifact ships the MTP tensors; the server does not load them by
default.** So acceptance is unmeasurable on a default serve at any batch — not
because the counter has no callers (PR #36 fixed that), but because the block
is switched off by a flag default. Measuring it needs `--mtp-depth > 0`, which
changes decode semantics and was out of scope for this session (Part C was
capture-only). CEILINGS.json calls MTP "the only lever that beats the
large-batch per-user wall on one card" — and §2.1 shows that wall clearly
(per-user 0.59 tok/s at B=64), so this is now a specific, cheap, unclaimed
experiment: **serve once with `--mtp-depth 1` and re-run the same sweep.**

---

## 4. QUALITY — GSM8K re-measured, and 87.0 is retired [measured]

```
GSM8K[chat] greedy (temperature=0) n=100: 96/100 = 96.0% (95% CI ±3.8pp)
| degenerate loops: 1 | truncated: 0
```

**Protocol, stated in full because comparisons keep going wrong:** n=100,
**0-shot chat** (the default; `--eight-shot` is the opt-in and was NOT used),
2048-token cap, **seed 161**, temperature 0, ONE scored request at a time,
on the **published** `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` artifact, arc
`3460656d3`. Result file `arc-tools/quality/results/gsm8k_s9.json`.

| | void 87.0 (session 3) | **this session** |
|---|---|---|
| accuracy | 87.0% (87/100), ±6.6 pp | **96.0% (96/100), ±3.8 pp** |
| degenerate | 2 | **1** |
| truncated at the 2048 cap | 9 | **0** |
| mean completion tokens | 528.4 | **148.5** |
| bake | session-3 GPU-Viterbi, **not** the published artifact | the **published W=256** artifact |
| decode math | **superseded** — pre-PR #35 | current master |

**87.0% is now retired, not beaten.** It was measured on a different bake *and*
on decode math that PR #35 subsequently changed (the SwiGLU clamp was missing on
4 of the 5 expert paths including the shared expert — which every token
traverses in every layer — and YaRN was being applied to the ratio-0 layers
{0,1,43} that must not have it). The brief anticipated the new number could
land either side of 87.0 and that lower would not be a regression. **It landed
9 points higher**, and the two collapse indicators moved with it: truncations
went 9 → **0** and mean completion length 528.4 → 148.5 tokens. That is the
signature of a model that now terminates its reasoning instead of rambling into
the cap — consistent with the clamp and YaRN fixes, though this session did not
isolate which fix contributed what.

**On the reference.** DeepSeek's published **90.8** for V4-Flash-Base is
**8-shot EM** — a different and *easier* protocol than the 0-shot chat used
here. The two are **not directly comparable**, and 96.0 should not be reported
as "beating 90.8". What can be said is that Arc's 2-bit artifact scores 96.0 on
the harder 0-shot protocol, and that **DOCTRINE D6's standing commitment of
≥90 GSM8K is met on our own protocol for the first time** — with the caveat
that ±3.8 pp at n=100 is a wide band and n=100 is a subset, not the full test
split.

### 4.1 Coherence / facts / math [measured]

```
COHERENCE: coherence6 5/6 | facts 21/22 | math 8/8
| June anchor: 6/6 coherence (commit 6102b4d84)
```

**Reported as measured, including the two that moved the wrong way.**
coherence6 is **5/6** against a 6/6 anchor and facts is **21/22** against a
prior 22/22. Both are one-item differences on very small batteries, so neither
is distinguishable from noise at this n — but neither is an improvement, and
they sit alongside a 9-point GSM8K gain, so they should not be quietly folded
into a good-news summary. `arc-tools/quality/results/coherence_s9.json`.

---

## 5. What this session changes, and what it hands to the next one

**Answered.** The four fixes work. The collapse that wave26-AX found is gone;
peak aggregate doubled and moved onto a real batch row; B=128 and B=256 became
reachable; and the quality number that was void is replaced by a much better
one taken on the published artifact under a stated protocol.

**The three things the next session should take, in order:**

1. **Trellis grouped-GEMM with cross-token dedup.** The component profile now
   says it at batch, end to end: `moe` is 35% of the step at B=1 and **49–64%
   at B=64**, and it is the one term that does not amortize. Everything else
   shrinks with batch; MoE grows. This is the keystone and it is now measured
   rather than argued.
2. **Per-step host overhead.** It is what caps B=256 — one core at 100%, GPU at
   0–4% and 121 W. Until it moves, the largest batches are *worse* than the
   middle ones, which inverts the fleet story exactly where the fleet story
   should be strongest.
3. **MTP with `--mtp-depth > 0`.** The artifact ships the tensors; the default
   serve does not load them. This is the only lever CEILINGS.json identifies
   that beats the large-batch per-user wall — and §2.1 measures that wall at
   **0.59 tok/s per user at B=64**, which is the number a customer would feel.
   One flag, one re-sweep.

**Also worth someone's time (surfaced, not shipped):**

- The **computed codebook (PR #46) is still not active at serve time** — it
  needs a bake, so none of the 3.86–4.03× it measured on the decode GEMV is in
  any number on this page. The next bake should carry it.
- **b=1 barely moved** (15.35 → 16.27, and 14.56 on the profiling build). The
  b=1 gap to the 1,413 tok/s physics ceiling is **87×** and is dominated by
  per-step host overhead, i.e. the same item as (2).
- Prefill now scales monotonically to **156.78 tok/s** at B=256 and never
  became the bottleneck.

## 6. Cost

| | |
|---|---|
| Instance | `arc-s9-h200`, H200, Helsinki, **$4.85/hr** |
| Created → deleted | 2026-08-15T21:08:22Z → 22:47:29Z (**99 min**) |
| **Spend** | balance $84.93 → **$77.33** = **$7.60** |
| Teardown | **DELETED**, confirmed by `runcrate ps` returning no instances |

Raw logs, result JSONs, both decode profiles, the scheduler lines and
`box_health.json` are in the session harvest `s9_results.tgz`.
