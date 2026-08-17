# wave51-CB — THE MEASUREMENT: `qtip2b` served, swept, and scored on one H200

**Session:** 2026-08-16, Runcrate H200 (Helsinki), instance `arc-s15-measure`
(created **21:02:31Z**, `running` **21:06:55Z**, self-destruct **ARMED 21:07:05Z**
for 25,200 s).
**Rate:** **$4.85/hr**. Every `$/Mtok` below carries this rate.
**Binary:** arc **`46ea6948d`** (origin/master, PR #76 head), `cuda flash-attn`,
**no `cudnn`**. Built on-box, `BUILD_RC=0`, ~6 min.
**Card:** NVIDIA H200, 143,771 MiB, driver 580.126.09, 44 vCPU, 178 GB RAM,
1.5 TB disk, CUDA toolkit 12.4, PCIe gen 5 ×16.

**What was never true before this session:** `qtip2b` had **never been served**.
wave48-BY died on `dtype mismatch in slice-set` before a single forward;
wave50-CA baked it and lost the box to an auth lockout before loading it. This
is the first end-to-end measurement of the rung the program decided to serve.

---

## 0. Box health gate — full PASS, before any measurement

```
PASS GPU 0: NVIDIA H200, driver 580.126.09, 143771 MiB
PASS PCIe gen 5/5, width x16/x16      PASS nproc=44      PASS disk free 1078GB
PASS sustained power 286W (40% of 700W limit, peak 287W) >= 200W floor
PASS SM clocks 1980/1980 MHz (100%) under load
=== VERDICT: PASS (0 fail, 0 warn) ===
```

The s5a signature (99% util at 132 W) is what this gate exists to catch. It did
not fire; the box was healthy for every number below.

---

## 1. The artifact loads, and that is itself a first [measured]

Downloads: overlay **74,121,021,083 B** (15 files, `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b`,
**public**), source **159,630,129,665 B** (`deepseek-ai/DeepSeek-V4-Flash`, 46
weight shards). Both landed in **~9 min** (≈430 MB/s combined).
Integrity verified **before** serving: 8 `qtip2b-*.uqff` shards ✓,
`residual.safetensors` **1,293,806,700 B exactly** ✓, 46 source shards ✓.

```
mistralrs serve -p 1234 -m /workspace/src -a deepseekv4 \
  --from-uqff /workspace/uqff/qtip2b-0.uqff \
  --chat-template chat_templates/deepseek_v4.json \
  --max-seqs 256 --prefix-cache-n 0
```

| evidence | value |
|---|---|
| `Auto-discovered 8 UQFF shard files (from 1 specified)` | yes |
| `Applying ISQ` in the log | **absent** (`grep -c` = 0) — a real UQFF load |
| ISQ artifacts loaded | **517 tensors, 17.20 s** |
| Cold load → `Server listening` | 21:32:37Z → 21:34:42Z = **2 m 05 s** |
| `/health` | **200** |
| Resident after load | **78,801 MiB of 143,771** |

🔑 **PR #76's KV preallocation fix is now proven on hardware, not just on CPU.**
Every prompt step in this session ran through `SingleCache::append`; the
`dtype mismatch in slice-set, lhs: BF16, rhs: U8` and `shape mismatch on dim 3,
512 <> 1` failures that killed wave48-BY never appeared, at any batch size, in
either the sweep or the 1,319-problem eval.

**Harness trust (D12), both halves:**
- offline, before renting: `test_batch_load_probe.py` → `ALL PASS`, including
  `mutation(serial): probe exited 1 … the assertion CAN fail`;
  `test_degeneracy.py` → all assertions hold.
- on the box: `CONC[B=8] effective_B=8 mean_inflight=6.61 server_prefill_batch=3
  verdict=pass`, `SELFTEST_RC=0`.

---

## 2. THE SWEEP [measured] — the headline

**Protocol.** One H200 @ **$4.85/hr**. Published `qtip2b` overlay on the source
checkpoint. Server `--max-seqs 256 --prefix-cache-n 0`, V4 chat template. Probe
`arc-tools/quality/batch_load_probe.py`, `/v1/chat/completions` streaming,
distinct ~68-token prompts, **64 decode tokens, 1 rep**, `--max-ctx 545000`,
temperature 0 — **wave34-BL's protocol exactly, so the rows compare line by
line with the `qtip2` rung.** Sweep wall time 21:36:10Z → 21:45:42Z = **9 m 32 s**,
`SWEEP_RC=0`.

| B | prefill agg tok/s (TTFT-derived, **lower bound**) | prefill agg tok/s (server-instrumented, **compute only**) | decode tok/s per user (p50) | **decode AGGREGATE tok/s** | TTFT p50/p95 (s) | $/Mtok @ $4.85/hr | effective_B |
|---|---|---|---|---|---|---|---|
| 1 *(diagnostic only — never a headline, D2)* | 170.56 | 171.58 | 17.99 | **18.27** | 0.375 / 0.375 | 73.74 | 1 |
| 8 | 178.39 | 216.11 | 5.67 | **41.43** | 1.642 / 2.948 | 32.52 | 8 |
| 16 | 196.49 | 235.59 | 3.965 | **54.75** | 3.135 / 5.321 | 24.61 | 16 |
| 32 | 231.84 | 266.67 | 2.87 | **74.52** | 5.530 / 9.020 | 18.08 | 32 |
| 64 | 260.92 | 293.96 | 1.82 | **91.46** | 9.914 / 16.031 | 14.73 | 64 |
| 128 | 277.94 | 308.49 | 1.09 | **106.36** | 18.669 / 30.096 | 12.67 | 128 |
| **256** | 285.85 | 314.72 | 0.53 | **111.69 ← peak** | 27.779 / 58.529 | **12.06** | 256 |

```
BATCHSWEEP[s15_qtip2b_sweep]: peak decode 111.69 tok/s @B=256 (effective_B=256)
 | prefill @peak 285.85 tok/s | per-user p50 @peak 0.53 tok/s
 | TTFT p50/p95 @peak 27.779/58.529s | $/Mtok @peak 12.06 ($4.85/hr) | errors 0
```

**`effective_B == B` on all seven rows, every verdict `pass`, 0 errors out of
505 requests.** No `WARNING[KV]` (the guard printed `KV[B=256]: est ~33792
tokens` against the 545,000 budget), no `WARN[CLIENT]`.

### 2.1 🔑 THE PREDICTION UNDER TEST — CONFIRMED

**Does aggregate rise past B=64? YES.** 91.46 → **106.36** (B=128, +16.3%) →
**111.69** (B=256, +5.0%). The curve is **monotonically increasing across all
seven rows** — the first time in this program's history that has been true.

The kernel-level crossover (gemv **flat** 315→317 tok/s from B=64 to B=128;
grouped **climbing** 322→527) **survived contact with the full serving path.**
On the `qtip2` rung the same probe measured aggregate *peaking at B=16* and
falling 37% by B=256; on `qtip2b` it climbs to the last row we can fit.

### 2.2 Against the `qtip2` rung — same probe, same protocol, same box class

| B | `qtip2` (wave34-BL) | **`qtip2b` (this session)** | ratio |
|---|---|---|---|
| 1 *(diagnostic)* | 16.27 | 18.27 | 1.12× |
| 8 | 26.96 | 41.43 | 1.54× |
| 16 | 30.65 ← its peak | 54.75 | 1.79× |
| 32 | 30.59 | 74.52 | 2.44× |
| 64 | 30.41 | 91.46 | 3.01× |
| 128 | 28.86 | 106.36 | 3.69× |
| 256 | 19.02 | **111.69** | **5.87×** |
| **peak** | **30.65 @B=16** | **111.69 @B=256** | **3.64×** |
| **best $/Mtok** | **$43.96** | **$12.06** | **3.65× cheaper** |

**b=1 moved only 1.12× — that is the control.** The 3–6× at batch is the
grouped kernel amortizing, not a box or bake windfall. Two uncontrolled
differences still ride along (different rung ⇒ different bake; different
rental), and b=1 is the evidence that neither explains the batch rows.

**Prefill also rises with batch on this rung** — 171.58 → 314.72 tok/s
server-instrumented (1.83×), where `qtip2` went 78.14 → 156.78. The prefill
figures are quoted in both forms because they are not the same measurement: the
TTFT-derived number includes HTTP, queueing and scheduling (a **lower bound**);
the server-instrumented number is compute-only (an **upper reference**).

### 2.3 What it still is not

Against the physics ceiling (`CEILINGS.json`, uniform-routing ⇒ these are
**lower bounds on the ceiling**): B=64 91.46 vs 5,289 (**58×**), B=128 106.36 vs
8,701 (**82×**), B=256 111.69 vs 16,602 (**149×**). The gaps narrowed from
174×/301×/873× on `qtip2`, and every remaining one is implementation, not
physics.

**Per-user decode is the cost of the batch rows.** 17.99 tok/s at b=1 falls to
1.09 at B=128 and 0.53 at B=256, against a bandwidth ceiling of 68 and 65. TTFT
p95 at B=256 is 58.5 s. **No configuration measured in this session puts a user
anywhere near 100 tok/s** — see §3, which was supposed to be the lever, and §5.

---

## 3. MTP AT BATCH — MEASURED AT B=1, **BROKEN ABOVE IT** [measured]

Speculative decode is the only lever that raises **per-user** decode on one
card. It has never been measured in this program. It is measured now, and the
result is half a number and one defect.

Serve: `ARC_MTP_LOG_ACCEPTANCE=1 … --mtp-depth 2 --paged-attn off
--prefix-cache-n 0 --max-seqs 256`. All four zeroing conditions satisfied; the
log confirms both halves:
`V4 MTP: full decoder block loaded (attention + MoE)` and
`MTP speculative decode engaged (depth=2); target advertised an MTP head`.

### 3.1 B=1 — the first MTP acceptance number this program has ever had

```
MTP[b=1] accept_rate=0.4194 accepted=26 proposed=62 steps=31 drafted_steps=31
         committed=57 tok_per_step=1.8387 batch_steps=31 mean_batch=1.0000
         tok_per_batch_step=1.8387
```

- **`accept_rate` = 0.4194** — 26 of 62 drafted tokens survived verification at
  depth 2. Reference floor for DeepSeek V4 MTP is **2.30 accepted-per-verify**
  in SGLang's non-simulated CI; at depth 2 our equivalent is **0.84
  accepted-per-verify** (26/31 steps). **The draft head is well below the
  reference.**
- **`tok_per_step` = 1.8387** — 1.84 emitted tokens per target forward. Applied
  to the b=1 diagnostic row that is 17.99 → **~33 tok/s per user, projected**,
  and it is *only* a projection: the multiplier and the throughput were not
  measured in the same run.
- `drafted_steps == steps` (31/31): the draft KV primed on every step. The
  wave42-BT/#71 failure mode (`drafted_steps=0`) is gone.

### 3.2 B≥8 — 🔴 THE ENGINE PANICS. Reproduced from a clean engine.

```
thread '<unnamed>' panicked at mistralrs-core/src/kv_cache/mod.rs:499:54:
called `Result::unwrap()` on an `Err` value: shape mismatch on dim 1, 18 <> 22
WARN mistralrs_core: Engine MTP-speculative(depth=2, target=…) is dead, rebooting
```

- First hit during the B=8 batch of the S3c sweep. **Reproduced on a freshly
  loaded server with a single B=8 probe**: same site, `19 <> 23`. Both deltas
  are **exactly 4** — a rollback length disagreeing with the cache by
  `(depth + 1) + 1`.
- **The reboot does not recover.** The engine restarts, logs
  `Successfully rebooted engine`, then serves nothing: subsequent requests hang
  with the GPU at 0% and 118 W. Two later probes (B=2, B=4) hung on the rebooted
  engine and produced **no** numbers — they are **not** evidence about those
  batch sizes and are not reported as such.
- ⇒ **PR #73's "MTP runs at every batch size" does not hold on the real serving
  path.** The batched ragged-verification rollback and `SingleCache` disagree
  about the committed length. `mtp_pipeline`'s own tests build their caches the
  way wave49-BZ already flagged — never against a preallocated `all_data` — so
  the disagreement cannot surface in CI.
- ⇒ **Item ⑥ of this session's charter is UNMEASURABLE, not zero:** per-user
  decode at B=128 with MTP cannot be measured until this is fixed. What we can
  say from measured parts: B=128 per-user without MTP is **1.09 tok/s**, and the
  best multiplier MTP has ever shown here is **1.84×** ⇒ ~2 tok/s. **The
  100-tok/s-per-user target is not met on one card at B=128, and MTP as it
  stands is not the thing that would meet it** (the bandwidth ceiling at B=128
  is 68 tok/s — even a perfect implementation needs the aggregate gap closed
  first).

---

## 4. FULL-SET GSM8K ON `qtip2b` — **96.3% ± 1.0pp** [measured]

```
python3 run_gsm8k.py --all --concurrency 16 --max-tokens 2048 --seed 161 \
  --checkpoint-every 10 --out results/gsm8k_qtip2b_full.json
```

**1,319 problems, 0-shot chat, greedy (t=0), seed 161, 2048-token cap**, served
by the same standard (non-MTP) server as the sweep. Main pass 22:17:19Z →
00:07:10Z (**1 h 49 m 51 s**); the crash-retry pass 00:09:20Z → 00:13:32Z.

| | |
|---|---|
| **Accuracy** | **1270 / 1319 = 96.3%**, 95% CI **± 1.0 pp** |
| degenerate loops | **0 / 1319** |
| truncated | **0 / 1319** |
| request errors, final | **0 / 1319** |
| mean completion | **157.8 tokens** |
| quality gate | **GATE[OK]** |

**⚠️ Two numbers exist and both are reported, because the difference is a
defect, not a protocol choice.** The first pass returned **1240/1319 = 94.0%
(± 1.3 pp) with 34 request errors** — 32 of those were requests in flight when
the engine died and rebooted (§4.1), plus 2 isolated failures. Those 34 items
carried no model output at all; they were **re-run against the recovered engine**
(the errored records were stripped so `--resume` picked exactly them, backup
kept as `gsm8k_qtip2b_full.prefill_backup.json`), which produced the complete
1,319-answer set above. **94.0% is the number if you charge every crash to the
model; 96.3% is the number for the model itself.**

- **Against our own history:** confirms and tightens wave34-BL's n=100 **96.0%
  ± 3.8 pp** (the `qtip2` rung) to **96.3% ± 1.0 pp** on `qtip2b` — the CI
  narrowed 3.8×, and the 2-bit rung switch cost nothing measurable in quality.
- **Against the reference:** the base model card publishes **90.8 8-shot EM**.
  Ours is **0-shot chat**. Different, harder protocol on our side; the numbers
  are **not directly comparable** and 96.3 must never be quoted as "beating"
  90.8 without that sentence attached.
- D6's ≥90 commitment is now met with a CI that can actually separate it from
  the anchor.

### 4.1 🔴 THE ENGINE DIED TWICE MID-EVAL — two distinct panics [measured]

Neither is MTP-related; this was the ordinary decode path with speculative
decode off.

| time | site | message | cost |
|---|---|---|---|
| 22:18:42Z | `mistralrs-core/src/kv_cache/mod.rs:498:54` | `shape mismatch on dim 1, **576 <> 64**` | 16 requests → HTTP 500 |
| 23:31:41Z | `mistralrs-core/src/engine/mod.rs:428:25` | `SendError { .. }` (unwrap on a closed response channel) | 16 requests → HTTP 500 |

- **576 = V4's KV width (512 + 64); 64 is the RoPE half alone.** The same file
  and the same `unwrap` family as the MTP panic in §3.2 (line 499 vs 498).
  wave49-BZ fixed the *dtype* contract in `SingleCache::append`; this is a
  **width** disagreement that survives it, and it is reachable **without MTP**.
- Both times the engine rebooted **and then served normally** — ~1,200
  subsequent problems completed, so the reboot recovers on the plain path (it
  does **not** on the MTP path, §3.2).
- **Frequency: 2 deaths in ~1,300 requests / 1 h 50 m of sustained
  `--concurrency 16` with 2048-token generations.** Zero appeared in the
  505-request sweep, whose completions are 64 tokens — **long generations or
  sustained concurrency are what surface it.**
- ⇒ This is a **serving-reliability defect on the default path**, and it is the
  single most important thing this session found that was not being looked for.

---

## 5. FP8 KV ON CUDA — **TOKEN-IDENTICAL** [measured]

`ARC_V4_FP8_KV=1` has never run on CUDA; `index_select` over the E4M3 LUT and
the U8 H2D leg were CPU-proven only (wave49-BZ / PR #76).

**Protocol.** Same box, same binary, same artifact, same serve flags; the only
difference is the environment variable. 5 fixed prompts, greedy (`temperature 0`,
`top_p 1.0`), 96-token cap, one request at a time. The OFF arm was captured on
the sweep's server (variable unset); the ON arm on a server started with
`ARC_V4_FP8_KV=1`, verified present in the serving process itself
(`/proc/<pid>/environ` → `ARC_V4_FP8_KV=1`, `ARC_V4_CAPTURE_PROBE` absent, which
is the only other gate in `v4_fp8_kv_enabled`).

```
FP8_IDENTICAL=5/5
FP8_VERDICT=TOKEN_IDENTICAL
completion_tokens both arms: [35, 74, 17, 96, 66]
```

**Every completion is byte-identical and every token count matches.** The
acceptance criterion PR #76 set for promoting FP8 KV from opt-in toward default
is met on hardware.

**What it does not cover, stated so nobody over-claims it:** 5 prompts / 288
generated tokens, single-stream, short context. It does not test FP8 KV at
batch, at long context, or with the sliding-window eviction path. And there is
**no log line** confirming the branch was taken — the evidence is the process
environment plus the source's single gate. **Add an INFO line when FP8 KV
engages**; a correctness check whose positive control is `getenv` is weaker than
it needs to be.

---

## 6. Cost, and the protocol that kept the box alive

| | |
|---|---|
| Instance | `arc-s15-measure`, H200 Helsinki, **$4.85/hr** |
| Created → deleted | **2026-08-16T21:02:31Z → 2026-08-17T00:21:51Z** = **3 h 19 m 20 s** |
| **Spend** | **≈ $16.11** by arithmetic (3 h 19 m 20 s × $4.85). Balance moved **$43.36 → $27.21** = $16.15; `runcrate ps` also lists **another session's** A100 (`arc-w53-paged`, $1.49/hr, `Active usage $0.39`), which this session did not create and did not touch |
| Self-destruct | **ARMED 21:07:05Z** (7 h), never needed |
| Teardown | `runcrate instances delete` → confirmed absent from `runcrate ps` |

Time went: bootstrap+downloads+build 17 min · health gate 3 min · load+self-test
6 min · **sweep 10 min** · MTP 25 min · **GSM8K 114 min** · FP8 8 min · harvest
5 min. **The eval is 57% of the bill; every measurement in §2 and §3 together
cost about $3.**

**The lockout protocol held.** One long-lived on-box collector wrote
`/root/status.txt` every 60 s and the laptop read it at 3–10 minute intervals —
no batches of `runcrate` calls, no concurrent OAuth refreshes, no lost session.
This is the direct fix for what cost wave50-CA its artifact.

---

## 7. What this session changes

1. **`qtip2b` is proven as a serving rung, not just a bake.** It loads, it
   batches, it holds 256 concurrent sequences, and it is **3.64× the peak
   aggregate and 3.65× cheaper per token** than the rung we published.
2. **The crossover thesis is confirmed end-to-end.** Aggregate rises through
   every batch row; "serve the truck, not the van" is now a measurement.
3. **PR #76's KV fix is hardware-proven** — 1,800+ requests, zero dtype/width
   failures on the sweep path.
4. **MTP is half-shipped**: real at B=1 (1.84 tok/step, 41.9% acceptance),
   **panics at B=8**. PR #73's claim does not hold on the serving path.
5. **A KV width panic and a `SendError` panic exist on the default path** and
   cost 32 of 1,319 eval requests. Nothing in CI can see either.
6. **Quality is settled**: 96.3% ± 1.0 pp on the full set, 0 degenerate, 0
   truncated.
7. **FP8 KV is correct on CUDA** for single-stream greedy decode.

### Next, in the order the numbers argue for
1. **Fix the two default-path panics** (`kv_cache/mod.rs:498` width, `engine/mod.rs:428`
   `SendError`). A 2-in-1,300 crash rate is a fleet-blocker independent of speed.
2. **Fix MTP at batch** (`kv_cache/mod.rs:499`, rollback length off by
   `depth + 2`) — it is the only per-user lever on one card, and per-user is
   where we are weakest (0.53 tok/s at B=256).
3. **Re-run the sweep past B=256** — the curve was still rising at the last row
   we measured, and the highest footprint sampled during the sweep was
   **105,095 MiB of 143,771** (60 s polling, so a floor on the peak, not the peak).
4. **Then the aggregate gap**: 111.69 vs a 16,602 tok/s bandwidth ceiling is
   149×, all of it implementation.
