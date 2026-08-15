# GPU SESSION 8 RUNBOOK — measure the fleet, from the published artifact

**What this session is for:** producing the **first end-to-end batched serving
measurement Arc has ever taken.** Every speed number in the record today is
either b=1 (5.4 → 13.99 → 14.58 tok/s across sessions) or a kernel microbench
(grouped-GEMM step time flat ~63.5 ms from B=16→64 ⇒ ~1,006 aggregate tok/s
⇒ ~15.7 tok/s/user at B=64 ⇒ ≈$1.36/Mtok at $4.92/hr). **The product thesis is
fleet capacity economics — aggregate tok/s per node and $/Mtok — and it has
never been measured through the actual server.** That is the whole job.

**Box:** one Runcrate **H200, 141 GB HBM** (≥24 cores, ≥720 GB disk),
**$4.92/hr = $0.082/min**. **NOT the 80 GB A100** — see §3, the KV arithmetic:
after the ~68 GB artifact an A100 leaves ~12 GB, which caps the sweep at
roughly B=24 at any useful context and makes the headline row unreachable.

**Prerequisite:** the UQFF is **published** to HF
`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` (HF org is `aeonmind`; the GitHub org
is `aeonmindai` — they differ). This session **pulls the 68 GB artifact and
loads it in ~11 s**. There is **no bake and no 149 GB source download.** If the
artifact is not published, this session does not start — run session 6/7 first.

Estimated clean run: **≈ 2:05 for ≈ $10.3.**

---

## 0. Standing rules that apply before anything else

- **NEVER DEBUG ON A PAID BOX.** Any ABORT-IF below means: harvest, then
  decide hold-vs-delete on the arithmetic in §8. Diagnosis happens on a laptop
  where compute is free.
- **DOCTRINE D10 — never leave a GPU idle.** §9 teardown is not optional and is
  never cut for time. Check `list_instances` on every resume.
- **DOCTRINE D2 — batch-first.** b=1 is a **diagnostic row only**. It never
  appears as a headline, in a PR title, or in a slide.
- **DOCTRINE D12 — fixtures lie, and so do harnesses.** Everything in §5 exists
  because a probe that measures nothing still prints a number.
- Runcrate mechanics that have cost us aborts: `ssh_execute` **mangles
  heredocs** → use `file_upload`; `ssh_execute` **drops with exit 255** on any
  command containing `sleep` ≳ 40 s → poll instead; **kill by PID, never
  `pkill -f <pattern>`** (it matches and kills your own SSH session — if you
  must pattern-match, use the bracket trick `foo[b]ar`).

---

## 1. Timeline and cost model

$4.92/hr = **$0.082/min**. Cumulative from *instance creation*.

| # | Step | Wall | Cum | $ step | Cum $ |
|---|---|---|---|---|---|
| S0 | **Box health gate — before the 68 GB pull** | 4m | 0:04 | 0.33 | **0.33** |
| S1 | Build (binary cache hit) ∥ pull 68 GB UQFF from HF | 20m | 0:24 | 1.64 | 1.97 |
| S2 | Serve + load gate (~11 s load) + **concurrency self-test** | 8m | 0:32 | 0.66 | 2.63 |
| S3 | **Speed sweep B ∈ {1,8,16,32,64,128}** — THE deliverable | 25m | 0:57 | 2.05 | 4.68 |
| S4 | Sustained-mode confirmation at the two best B | 10m | 1:07 | 0.82 | 5.50 |
| S5 | **GSM8K n=100, 0-shot, 2048-cap, seed 161** | 40m | 1:47 | 3.28 | 8.78 |
| S6 | coherence6 + facts/math | 8m | 1:55 | 0.66 | 9.44 |
| S7 | Tar + **teardown (NEVER CUT)** | 10m | 2:05 | 0.82 | **10.26** |

S5 is the long pole and is **independent of S3/S4** — if the session must be
cut short, S3+S4 (the thing that has never been measured) is what survives, and
S5 is deferred. Never the other way round.

---

## 2. Step-by-step

Wall-time estimates assume a healthy box. Each step names its **ABORT-IF**
precisely enough to act on offline, without re-renting.

### S0 — Box health gate (4 min) — RUNS BEFORE THE 68 GB PULL

```bash
bash /root/box_health_gate.sh --json /srv/arcstatus/box_health.json --burn-secs 60
```

We lost ~1.5 h and ~$7 to box `s5a` (NY H200) sitting at **99% GPU utilization
while drawing 132 W of a 700 W limit** — transfer-starved, invisible to
`nvidia-smi utilization`, and a 6× slowdown. The gate's POWER check is the one
that catches it.

> **ABORT-IF** the gate exits non-zero. In particular:
> - `FAIL sustained power <N>W ... < 200W floor` — the s5a signature.
>   **DELETE the instance and re-rent in a DIFFERENT region.** Do not debug it,
>   do not "see how it goes". Pull `box_health.json` first — it is the only
>   artifact of a failed gate and it is what tells you offline which region to
>   avoid next.
> - `FAIL nproc=<n> < 16` or `FAIL disk free <n>GB < 600GB` — wrong box type,
>   re-rent to spec.

### S1 — Build ∥ pull the artifact (20 min, parallel)

Run both at once; the download is the critical path.

```bash
# A: binary — cached build if the tree is unchanged, else ~25 min
cargo build --release -p mistralrs-cli --features "cuda flash-attn"
# NOTE: do NOT add the `cudnn` feature. Measured -62% decode on V4
# (5.45 vs 14.58 tok/s, session 4).

# B: the artifact — 68 GB, NOT the 149 GB source
huggingface-cli download aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 \
  --local-dir /workspace/uqff --max-workers 16
```

> **ABORT-IF** the HF download 404s or resolves to a repo under `aeonmindai`
> (GitHub org) rather than `aeonmind` (HF org) — the artifact is not published;
> this session cannot run. Delete the box; it is $2 spent, not $10.
> **ABORT-IF** the download sustains < 200 MB/s for 5 min — 68 GB will not land
> inside the budget. Delete and re-rent in another region.

### S2 — Serve, load gate, and the concurrency self-test (8 min)

```bash
/root/arc/target/release/mistralrs serve -p 1234 \
  --from-uqff /workspace/uqff \
  --max-seqs 128 \
  --prefix-cache-n 0 \
  --max-seq-len 4096 --max-batch-size 128
```

**`--max-seqs 128` is load-bearing and is the single most likely way this
session silently produces garbage.** mistral.rs defaults `--max-seqs` to **32**
(`mistralrs-cli/src/args/mod.rs:414-416`). Serve without this flag and the
B=64 and B=128 rows quietly become B=32 rows: the probe still returns, the
numbers still look plausible, and nothing in a naive harness reveals it. The
probe's concurrency check (§5) catches it — but set the flag anyway.

`--prefix-cache-n 0` keeps prefill real; the probe's prompts are distinct from
token 0 regardless, but a prefix-cache hit sets the server's
`total_prompt_time_sec` to zero and blanks the prefill instrumentation.

Then the **self-test**, ~90 s, before any measurement is trusted:

```bash
# a. the check must PASS on a correctly-served box
python3 batch_load_probe.py --batches 8 --reps 1 --max-tokens 32 --label selftest
# expect: CONC[B=8] effective_B=8 ... verdict=pass       -> exit 0

# b. and it must FAIL when the server is forced to serialise.
#    Restart the server with --max-seqs 1, re-run, then restore --max-seqs 128.
python3 batch_load_probe.py --batches 8 --reps 1 --max-tokens 32 --label selftest_serial
# expect: FAIL: CONCURRENCY[B=8] effective_B=1 ...       -> exit 1
```

> **ABORT-IF (a) does not print `verdict=pass`** — the server is not batching.
> Read `effective_B`: `1` means strictly serial (suspect the xs_history
> per-sequence fix, PR #21, is missing from this binary — the old per-model
> buffer corrupts or crashes with >1 sequence in flight); `32` with
> `--max-seqs 128` set means the flag did not take.
> **ABORT-IF (b) exits 0.** The concurrency assertion is then non-functional
> and **every number this session produces is unfalsifiable**. This is a
> harness bug, not a box bug: delete, fix on CPU against
> `test_batch_load_probe.py`, re-rent.
> **ABORT-IF** UQFF load takes > 3 min (expected ~11 s) — the artifact is being
> re-quantized rather than loaded; check for `Applying ISQ` in the server log,
> which must NOT appear on a `--from-uqff` path.

### S3 — The speed sweep (25 min) — **THE DELIVERABLE**

```bash
python3 batch_load_probe.py \
  --batches 1,8,16,32,64,128 \
  --reps 3 --max-tokens 256 --warmup-tokens 32 \
  --max-ctx 153000 \
  --cost-per-hour 4.92 \
  --label s8_sweep
```

Produces exactly one table. **Copy it out verbatim; do not re-derive numbers.**

| B | prefill tok/s | decode tok/s per user | decode aggregate tok/s | TTFT p50/p95 | $/Mtok | effective_B |
|---|---|---|---|---|---|---|
| 1 | | | | | | *(diagnostic only)* |
| 8 | | | | | | |
| 16 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |
| 128 | | | | | | |

Fill from the probe's own markers — `PREFILL[B=n]`, `BATCH[B=n]`, `CONC[B=n]`
— never by eyeballing the JSON.

**Reporting rules, non-negotiable:**
- **b=1 is a diagnostic row.** It exists to compare against the 14.58 tok/s
  history. It is never the headline (**D2**).
- **`effective_B` is a required column.** If it is below B, the row measures
  `effective_B`, and the table must say so. **We report the largest batch that
  ACTUALLY RAN — never a silently dropped row.**
- **Prefill and decode never merge into one rate.** The probe prints the
  prefill number with the words *"derived from TTFT + prompt len; includes
  queueing — a LOWER bound"* next to it, plus a server-instrumented
  compute-only figure. Carry both labels into the writeup.
- **$/Mtok is printed with the rate it used** (`$/Mtok X @ $4.92/hr`). Quote
  the rate every time. A $/Mtok without its $/hr is not a number.

> **ABORT-IF** the probe exits 1 with `FAIL: CONCURRENCY[B=...]` — see S2.
> **ABORT-IF** `WARN[CLIENT] ... probe used N% of a CPU core` fires at B=128.
> The **client** is then the bottleneck and the tok/s are an *understatement*,
> not a server result. Re-run B=128 alone with `--max-tokens 128`; if it still
> fires, split the load across two probe processes and sum them.
> **ABORT-IF** `WARNING[KV]` fires — the batch exceeds `--max-ctx`; expect
> eviction. Record it, drop to the largest B that does not warn, and state the
> cap in the table rather than deleting the row.

### S4 — Sustained-mode confirmation (10 min)

One-shot batches measure a burst; production is a closed loop. Confirm the two
best B from S3 under sustained load:

```bash
python3 batch_load_probe.py --batches 64,128 --duration 120 \
  --max-tokens 256 --cost-per-hour 4.92 --label s8_sustained
```

> **ABORT-IF** sustained aggregate is < 70% of the one-shot aggregate at the
> same B. That gap is a scheduler//admission problem, not noise, and the
> one-shot number is then the one that is misleading — report the sustained
> number as the headline and file the gap.

### S5 — Quality re-measure (40 min)

```bash
python3 run_gsm8k.py --n 100 --shots 0 --max-tokens 2048 --seed 161 --label s8
```

**87.0% is PROVISIONAL and must be re-measured.** PR #35 changed the decode
math on every token's path: the SwiGLU clamp was missing on **4 of the 5 expert
paths including the shared expert**, and YaRN was being applied to the ratio-0
layers **{0, 1, 43}** that must not have it.

**Protocol, stated because comparisons keep going wrong:** n=100, **0-shot
chat**, 2048-token cap, **seed 161**, ONE scored request at a time (the batch
probe is the deliberate exception to that rule; scored evals are not).

> **The reference publishes 90.8 with 8-shot.** That is a *different, easier*
> protocol. **Always state the shot count when comparing.** A 0-shot number
> against an 8-shot number is not a regression, and reporting it as one has
> burned us before.

Then coherence6 and facts/math:

```bash
python3 run_coherence.py --label s8
```

> **ABORT-IF** GSM8K lands below ~80% — a >7-point drop from the provisional
> 87.0 is a decode-math regression, not noise at n=100. Harvest and stop; do
> not spend the rest of the session on speed numbers for a broken model.

---

## 3. KV budget arithmetic — **does B=128 fit in 141 GB?**

**Answer: yes at the sweep's context, no at long context.** The arithmetic,
from the repo rather than from memory:

**Headroom.** 141 GB (H200) − 68 GB (artifact) = **73 GB raw**. Reserve ~8 GB
for the CUDA context, prefill activations at B=128, logits and fragmentation
⇒ **~65 GB usable for cache.**

**Per-token cost.** V4-Flash is **MQA, not MLA** — there is no `kv_lora_rank`
and no compressed latent; the config carries `num_key_value_heads = 1` and
`head_dim = 512` (`mistralrs-core/src/models/deepseek4.rs`, config const
`V4_FLASH_CONFIG_JSON` at ~line 4025, copied from the HF card). **All 43 layers
keep a full cache** — the ratio-0 layers {0,1} differ only in RoPE, and "43" in
that set is the MTP slot, not a real layer. KV is **BF16** on CUDA CC≥8.0, and
there is **no KV quantization on this path**.

Two caches, and the second is the big one:

| cache | formula | B/token |
|---|---|---|
| attention KV, 43 layers | `43 x 2(K,V) x 1 head x 512 x 2 B` | 88,064 |
| compressor `xs` history, 41 compressed layers | `41 x (4096 + 1) x 2 B` | 335,954 |
| **total** | | **424,018 ≈ 414 KiB/token** |

The KV formula is the repo's own
(`paged_attention/config.rs:62-70` `kv_cache_elements_per_token`, times layers
times dtype size, as in `tuning.rs:425-436`). **PagedAttention is disabled for
V4** — `DeepSeekV4Loader::supports_paged_attention()` returns `false` because
head_dim=512 exceeds the kernel's supported sizes — so **every `--pa-*` flag is
silently inert here** and the cache is contiguous, grown in 512-token chunks.

**Per sequence, and the resulting cap** (65 GB usable, allocation rounded up to
the 512-token grow granularity):

| context C (prompt + decode) | per-seq | max B | **B=128?** |
|---|---|---|---|
| 320 → **512 alloc** (the S3 sweep: ~64 prompt + 256 decode) | 0.217 GB | ~299 | **YES** — uses 27.8 GB of 65 |
| 1024 | 0.434 GB | ~149 | **YES** — 55.6 GB, tight |
| 2048 | 0.868 GB | **~74** | **NO** — largest sweep row is **B=64** |
| 4096 | 1.737 GB | ~37 | **NO** — largest sweep row is **B=32** |

**So: B=128 fits and must be run at the S3 sweep's context.** Beyond roughly
**1,150 tokens per sequence B=128 stops fitting**, and at a 2048-token context
the largest feasible batch is ~74 — i.e. the sweep tops out at B=64 and the
table must say *why*, not omit the row. `--max-ctx 153000` in S3 is exactly
65 GB / 424,018 B, so the probe's KV guard warns at the real boundary.

> **Noticed:** the `xs` history cache is **3.8× the KV cache itself** and is
> what actually caps batch size — 41 layers × full `[B,T,4096]` BF16 hidden
> states. The reference stores 584 B/token/layer of KV where Arc stores 1,024
> plus 8,194 of history. Halving it (fp8, or recomputing) would roughly
> **4× the feasible batch at long context**. Worth a separate change?

---

## 4. Baseline row — the in-class comparison (**D3**)

**VERDICT: no single-H200 in-class baseline exists for DeepSeek-V4-Flash on any
engine, because the model does not fit on one H200. That is not a hole in our
measurement — it is the result.** Report it as such, with sources, and do not
fabricate a substitute.

Evidence:

| claim | source |
|---|---|
| Native checkpoint ≈ **160 GB** (46 shards) — exceeds 141 GB before any cache | `huggingface.co/deepseek-ai/DeepSeek-V4-Flash` file listing |
| SGLang/vLLM/NVIDIA all ship recipes, so this is **capacity, not support** | `docs.nvidia.com/dynamo/dev/recipes/deepseek-v4-flash` |
| Smallest **published H200** config = **4× H200** (DP4+TP1+EP) | NVIDIA Dynamo recipe, above |
| LMSYS day-0: "Flash (285B): H200 with **tensor parallelism of 4**" | `lmsys.org/blog/2026-04-25-deepseek-v4/` |
| The one published W4A16 quant is **143 GB**, card states verbatim: *"TP=2 is the only validated configuration. TP=1 OOMs on a single 141 GB H200."* | `huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8` |
| `nvidia/DeepSeek-V4-Flash-NVFP4` is **Blackwell-only** — H200 is Hopper, no FP4 tensor cores | `huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4` |
| H200 = 141 GB @ **4.8 TB/s** | `nvidia.com/en-us/data-center/h200/` |

The checkpoint already ships at ~4.5 bits/param (experts natively FP4), so
"just quantize to 4-bit" is **already spent**. Arc's ~68 GB artifact is
**≈1.9 bits/param** — under half the smallest published quantized checkpoint —
which is precisely what makes single-H200 serving possible at all.

**Therefore the defensible baseline claim is a footprint claim, and it is
strong:** the same model, at published-recipe quality, needs **4 H200s**
(NVIDIA/LMSYS) or at minimum **2** (the W4A16 card's own words). Arc targets
**1**. State $/Mtok with the GPU count attached — a 4×H200 node is
**$19.68/hr**, not $4.92 — because that is the fleet-capacity comparison the
whole thesis rests on.

**Second baseline, and the one an engineer should argue with — the roofline.**
It needs no third party and cannot be gamed:

- artifact 68 GB / 284 B params ⇒ **1.92 bits/param**
- decode at large B touches ~all 256 routed experts ⇒ ~68 GB read per step
  ⇒ 68 GB ÷ 4.8 TB/s = **14.2 ms/step floor**
- ⇒ ceiling **≈ 4,500 tok/s at B=64**, **≈ 9,000 tok/s at B=128**
- measured microbench 63.5 ms/step at B=64 ⇒ ~1,006 tok/s = **~22% of
  roofline**. S3's real number goes in the same column.

State plainly what each proves: the footprint row proves **node capacity**
(the fleet wedge); the roofline proves **headroom** and prices future kernel
work. Neither proves kernel-efficiency parity with SGLang — only running
SGLang on the same box would, and it cannot run this model there.

**Do NOT** compare against Blackwell NVL72 rack numbers (different silicon,
sharded experts, different economics), and **do not** use the aggregator blogs
claiming "INT4 V4-Flash on a single H200 at ~34 tok/s" or "a single 80 GB
card" — those are arithmetically impossible against the 143 GB measured
artifact and that whole family of sources is unusable.

*Optional, only if the box is already paid for and S3–S6 are complete:* serve
**gpt-oss-120b** (117B/5.1B active, MXFP4, ~61 GB — fits one H200) under
SGLang on the SAME box and run the same protocol. That yields a same-box,
same-protocol **engine-efficiency reference**. It is **not** a same-model
baseline — 2.5× fewer active params — and must never be labelled as one.

---

## 5. Why the probe's numbers are trustworthy this time

`arc-tools/quality/batch_load_probe.py`. Our dominant defect class has been
tests and probes that pass while measuring nothing — seven instances in one day
(**BACKLOG**, "VACUOUS / MISSING TESTS"; **DOCTRINE D12**). The shape that
would bite here is exact: *a probe that requests B=64 but whose requests
execute sequentially prints a plausible number that is really 64 × b=1.*

So the probe measures its own concurrency and can fail:

- **Signal:** per-request **decode** windows `[first token, last token]`, swept
  for peak overlap. Under continuous batching they coincide; under serial
  execution request *i* emits nothing until *i−1* finishes, so they are
  disjoint and overlap collapses to 1. Client *submit* windows are deliberately
  not used — the barrier makes those overlap even when the server serialises,
  which would be a check that can never fail.
- **Corroboration, server-side:** the engine stamps every sequence in one
  prefill step with that step's wall time (`engine/mod.rs`), surfaced as
  `usage.total_prompt_time_sec`. Requests sharing a stamped duration *and*
  starting to decode together shared a step, so `server_prefill_batch` is an
  independent, engine-sourced batch size.
- **Gate:** hard `FAIL:` + **exit 1** below `--min-concurrency-frac × B`
  (default 0.5, floor 2); `WARN[CONCURRENCY]` plus an `effective_B` column when
  the server ran fewer than asked. A failed row is **excluded from `peak`** so
  a serialised batch can never become the headline.
- **Mutation-tested, both offline and on GPU.** `test_batch_load_probe.py`
  serialises the mock behind a global lock and asserts the probe **exits 1**
  with `effective_B=1`; a second mutation caps the mock at 2-of-8 (the realistic
  `--max-seqs` bug) and asserts it is both detected and reported. S2(b) repeats
  the proof on real hardware with `--max-seqs 1`.

Also guarded: `WARN[CLIENT]` when the probe's own CPU use suggests the *client*
is the bottleneck (a real risk at B=128, and it would understate the server);
`--no-stream` refuses to claim a prefill/decode split or any concurrency at all
and says so.

**Greppable markers** (poll these, never the prose):
`BATCH[B=n]` · `PREFILL[B=n]` · `CONC[B=n]` · `BATCHSWEEP[label]` ·
`WARNING[KV]` · `WARN[CONCURRENCY]` · `WARN[CLIENT]` · `FAIL:` · `GSM8K[...]`

---

## 6. What "done" looks like

1. The §3 table, filled, with `effective_B` on every row and `$/Mtok` carrying
   `@ $4.92/hr`.
2. Prefill and decode reported **separately**, each with its method label.
3. A stated answer to "did B=128 run?" — with `effective_B` as the evidence.
4. GSM8K n=100 0-shot at seed 161, replacing the provisional 87.0, **with the
   shot count stated** next to any comparison to the reference's 90.8/8-shot.
5. The §4 baseline paragraph, sources attached.

---

## 7. Harvest (runs before teardown, unconditionally)

```bash
tar czf /root/s8_results.tgz \
  /srv/arcstatus/box_health.json \
  arc-tools/quality/results/ \
  /root/logs/
```

A box that dies with nothing pulled makes the next attempt blind and we pay
twice for the same ignorance. **Harvest first, always — including on the
health-gate abort path**, where `box_health.json` is exactly what tells you
offline which region to avoid.

## 8. Hold-or-delete arithmetic

| | cost |
|---|---|
| holding an idle box | **$0.082/min** |
| re-entry (boot ~10 min + cached binary <1 min + 68 GB pull ~15 min) | **~26 min ≈ $2.13** |

**HOLD if the fix is shorter than ~26 minutes of work. DELETE if it is
longer.** "Never debug on a paid box" means never *sit and think* on one; it
does not mean throwing away 25 minutes of paid setup to fix a typo.

## 9. Teardown — NEVER CUT

```bash
# 1. pull the tarball
# 2. Runcrate: delete_instance <id>
# 3. list_instances -> confirm nothing is left running
```

**DOCTRINE D10.** The delete is confirmed by a `list_instances` that comes back
without this box, not by having called delete.
