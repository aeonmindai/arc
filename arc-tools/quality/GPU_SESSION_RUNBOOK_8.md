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
is `aeonmindai` — they differ). If the artifact is not published, this session
does not start — run session 6/7 first.

> 🔴 **THE ARTIFACT IS AN OVERLAY. YOU NEED THE SOURCE CHECKPOINT TOO.**
> An earlier revision of this runbook said "no bake and **no 149 GB source
> download**". **That was wrong and it cost a session.** The UQFF repo ships a
> `config.json` and a tokenizer, which makes it look standalone; it is not. Its
> only non-quantized weight file, `residual.safetensors`, is 1.29 GB — 1.7% of
> the repo. Arc **builds the model from the source checkpoint and overlays the
> quantized layers from the shards.**
>
> Confirmed in code, not from memory: with `--from-uqff` set, `normal.rs:679-687`
> loads the ISQ layers by deserialization and leaves the base expert weights as
> `DummyLayer` placeholders — but the **non-ISQ** weights (embeddings, norms,
> lm_head, compressor) still come through the source VarBuilder. So the source
> tree must be on disk. The expert bulk is *not* read from it, which is why this
> is still not a bake.
>
> **Two downloads, both required** (sizes read from the HF API, 2026-08-15):
> | repo | bytes | note |
> |---|---:|---|
> | `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` | 74.19 GB | 8 shards + `residual.safetensors` — **all 9 or it fails** |
> | `deepseek-ai/DeepSeek-V4-Flash` | 159.63 GB | public, ungated; the overlay base |
> | **total** | **233.8 GB** | budget disk ≥ 400 GB |
>
> Downloading the source **as the overlay base is expected and correct**. What
> RULE ZERO forbids is downloading it in order to *re-quantize*.

Estimated clean run: **≈ 2:17 for ≈ $11.2.** (Up from the old ≈2:05/$10.3, which
omitted the 159.63 GB source pull and assumed a warm-cache load time.)

---

## 0. Standing rules that apply before anything else

- 🔴 **RULE ZERO — IF THE ARTIFACT DOES NOT LOAD, STOP. DELETE THE BOX.
  REPORT.** Do not bake. Do not quantize. Do not re-derive the artifact by any
  route. **There is no fallback path any agent is authorized to invent.** A
  clean "it failed, here is the exact error and the exact command" is a GOOD
  outcome and costs ~$2 instead of ~$15.
  *This rule exists because a session improvised exactly that fallback: unable
  to load the artifact, it downloaded the source checkpoint and ran a full
  in-memory quantization on a $4.85/hr H200, burning ~$10 to produce nothing.*
- 🔴 **PREFLIGHT ON THE LAPTOP, BEFORE RENTING.** Every check that does not
  need a GPU is free and must happen first — HF token identity, repo
  reachability, artifact completeness, and **that the GPU provider session is
  actually authenticated** (`runcrate ps`). A session that discovers a broken
  login *after* renting has paid for the discovery. See §10.
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
| P | **Preflight on the laptop (§10) — FREE, before renting** | — | — | **0.00** | **0.00** |
| S0 | **Box health gate — before the 234 GB pull** | 4m | 0:04 | 0.33 | **0.33** |
| S1 | Build ∥ pull **234 GB** (74 overlay + 160 source) from HF | 30m | 0:34 | 2.46 | 2.79 |
| S2 | Serve + load gate (**~3 min cold**) + **concurrency self-test** | 10m | 0:44 | 0.82 | 3.61 |
| S3 | **Speed sweep B ∈ {1,8,16,32,64,128,256}** — THE deliverable | 25m | 1:09 | 2.05 | 5.66 |
| S4 | Sustained-mode confirmation at the two best B | 10m | 1:19 | 0.82 | 6.48 |
| S5 | **GSM8K n=100, 0-shot, 2048-cap, seed 161** | 40m | 1:59 | 3.28 | 9.76 |
| S6 | coherence6 + facts/math | 8m | 2:07 | 0.66 | 10.42 |
| S7 | Tar + **teardown (NEVER CUT)** | 10m | 2:17 | 0.82 | **11.24** |

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

# B: the overlay — 74 GB (private; needs the token)
huggingface-cli download aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 \
  --local-dir /workspace/uqff --max-workers 16

# C: the overlay BASE — 160 GB. REQUIRED. See the prerequisite box above.
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /workspace/src --max-workers 16
```

Token: upload `/Users/jish/.config/arc/env`'s `HF_TOKEN` to the box as
`/root/.hf_token` with `file_upload`, `chmod 600`, and `export
HF_TOKEN=$(cat /root/.hf_token)`. **Never echo it.** The overlay repo is
private; the source repo is public and ungated.

> **ABORT-IF** the HF download 404s or resolves to a repo under `aeonmindai`
> (GitHub org) rather than `aeonmind` (HF org) — the artifact is not published;
> this session cannot run. Delete the box; it is $2 spent, not $10.
> **ABORT-IF** the download sustains < 200 MB/s for 5 min — 234 GB will not land
> inside the budget. Delete and re-rent in another region.
> **ABORT-IF** either download is incomplete. Verify **before** serving —
> the failure mode downstream is the misleading `DummyLayer` error, not a
> "missing file" message:
> ```bash
> ls /workspace/uqff/qtip2-*.uqff | wc -l   # must be 8
> ls -l /workspace/uqff/residual.safetensors # must exist, 1,293,806,700 B
> ls /workspace/src/model-*.safetensors | wc -l  # must be 46
> ```

### S2 — Serve, load gate, and the concurrency self-test (8 min)

🔴 **The invocation below is the one VERIFIED on hardware** (517 tensors,
12.94 s load). The previous revision of this runbook printed
`serve --from-uqff /workspace/uqff` with **no `-m`, no `-a`, and a directory
where a file belongs** — three independent errors, each of which alone produces
the `DummyLayer` error in §"the one error you are most likely to hit". Do not
re-derive this command; copy it.

```bash
/root/arc/target/release/mistralrs serve -p 1234 \
  -m /workspace/src \
  -a deepseekv4 \
  --from-uqff /workspace/uqff/qtip2-0.uqff \
  --chat-template chat_templates/deepseek_v4.json \
  --max-seqs 256 \
  --prefix-cache-n 0 \
  --max-seq-len 4096 --max-batch-size 128
```

Four things that are each load-bearing:

* **`-m` points at the SOURCE checkpoint, never at the UQFF directory.** The
  overlay is not a model. This is the single most common way to hit `DummyLayer`.
* **`-a deepseekv4`** — the architecture is stated explicitly.
* **`--from-uqff` takes the FIRST SHARD FILE, not a directory**
  (`mistralrs-cli/src/args/model.rs:91-95`: *"UQFF file(s) to load from. Shards
  are auto-discovered: specifying the first shard … automatically finds …"*).
  Success logs `Auto-discovered 8 UQFF shard files (from 1 specified)`.
* **`--chat-template chat_templates/deepseek_v4.json`** — present in the
  measured-working invocation (wave26-AX §2). The probe and every scored eval
  post to `/v1/chat/completions` and rely on the server-side template.

**Why `-m` decides it, from the source** (`paths.rs:365-386`): weight files are
picked as `if !safetensors.is_empty() { safetensors } else if uqff_residual …`.
With `-m` at the artifact the only match is `residual.safetensors` — which holds
575 tensors (embeddings, norms, router gates, compressor) and **zero attention
projections, zero shared-expert weights, no `gate.tid2eid`**. With `-m` at the
source, the 46 shards supply everything the overlay does not carry.

Full documentation of the artifact, including the failure table, is in
`docs/model-cards/deepseek-v4-flash-uqff-qtip2.md`. **Read it before the box is
running**, not after.

> 🔴 **ABORT-IF the load fails — RULE ZERO.** In particular
> `Error: DummyLayer not replaced at index 1, layer Some(0) after
> load_from_artifacts` means a quantizable layer never got its weights. Check,
> in this order: (1) is `-m` the source and not the overlay? (2) are all 8
> shards **and** `residual.safetensors` present? Then **STOP, delete the box,
> and report the exact command and error.** Do not attempt to produce the
> weights by any other means.

**`--max-seqs` is load-bearing and is the single most likely way this
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
#    Restart the server with --max-seqs 1, re-run, then restore --max-seqs 256.
python3 batch_load_probe.py --batches 8 --reps 1 --max-tokens 32 --label selftest_serial
# expect: FAIL: CONCURRENCY[B=8] effective_B=<1 or 2> ... -> exit 1
```

Both halves were verified on an A6000 before this runbook was written
(wave25-AV §2). Note `--max-seqs 1` measured `effective_B=2`, not 1 — the
engine still overlaps one sequence's tail with the next sequence's first token.
**Anything ≤ 2 is the serialised signature; what matters is that (b) exits 1.**

> **ABORT-IF (a) does not print `verdict=pass`** — the server is not batching.
> Read `effective_B`: `1`–`2` means strictly serial (suspect the xs_history
> per-sequence fix, PR #21, is missing from this binary — the old per-model
> buffer corrupts or crashes with >1 sequence in flight); `32` with
> `--max-seqs 256` set means the flag did not take.
> **ABORT-IF (b) exits 0.** The concurrency assertion is then non-functional
> and **every number this session produces is unfalsifiable**. This is a
> harness bug, not a box bug: delete, fix on CPU against
> `test_batch_load_probe.py`, re-rent.
> **ABORT-IF** UQFF load takes **> 8 min** — the artifact is being re-quantized
> rather than loaded. The reliable signal is not the clock but the log: grep for
> `Applying ISQ`, which **must NOT appear** on a `--from-uqff` path.
>
> ⚠️ **The old threshold here was "> 3 min (expected ~11 s)" and it was wrong in
> both halves.** The 12.94 s figure on the model card was measured on the A100
> that had just baked the artifact, with the file cache warm. The only
> cold-cache measurement we have is **3 m 10 s** (wave26-AX §2, load 14:52:47 →
> serving 14:55:57) — i.e. the old rule would have **aborted a perfectly healthy
> load.** Expect **~3 min cold**, seconds warm.

### S3 — The speed sweep (25 min) — **THE DELIVERABLE**

```bash
python3 batch_load_probe.py \
  --batches 1,8,16,32,64,128,256 \
  --reps 3 --max-tokens 256 --warmup-tokens 32 \
  --max-ctx 545000 \
  --cost-per-hour 4.92 \
  --label s8_sweep
```

🔴 **Serve with `--max-seqs 256` for this sweep** (§S2 shows 128). The flag
defaults to **32**; if it is below the largest B, those rows silently become
B=32 rows reporting a believable number. See §5 — 190.01 vs 198.34 tok/s, 4%
apart, indistinguishable by inspection.

Produces exactly one table. **Copy it out verbatim; do not re-derive numbers.**

| B | prefill tok/s | decode tok/s per user | decode aggregate tok/s | TTFT p50/p95 | $/Mtok | effective_B |
|---|---|---|---|---|---|---|
| 1 | | | | | | *(diagnostic only)* |
| 8 | | | | | | |
| 16 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |
| 128 | | | | | | |
| 256 | | | | | | |

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
python3 batch_load_probe.py --batches 128,256 --duration 120 \
  --max-tokens 256 --cost-per-hour 4.92 --label s8_sustained
```

> **ABORT-IF** sustained aggregate is < 70% of the one-shot aggregate at the
> same B. That gap is a scheduler//admission problem, not noise, and the
> one-shot number is then the one that is misleading — report the sustained
> number as the headline and file the gap.

### S5 — Quality re-measure (40 min)

```bash
# 0-shot is the DEFAULT (--eight-shot is the opt-in). There is no --shots flag
# and no --label flag; the run is named through --out.
python3 run_gsm8k.py --n 100 --max-tokens 2048 --seed 161 \
  --out results/gsm8k_s8.json
```

> **The previous revision printed `--shots 0 … --label s8`. Neither flag
> exists** (`run_gsm8k.py:141-161`) and argparse rejects the whole command, so
> the step fails instantly on a paid box. `--eight-shot` is the only shot
> control; omitting it *is* 0-shot. Requires `data/gsm8k_test.jsonl` — run
> `bash fetch_data.sh` first.

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
# Again: no --label. `run_coherence.py` takes --out and --skip-facts only.
python3 run_coherence.py --out results/coherence_s8.json
```

> **ABORT-IF** GSM8K lands below ~80% — a >7-point drop from the provisional
> 87.0 is a decode-math regression, not noise at n=100. Harvest and stop; do
> not spend the rest of the session on speed numbers for a broken model.

---

## 3. KV budget arithmetic — **does B=256 fit in 141 GB?**

> 🔴 **REWRITTEN FOR PR #59 (merged 2026-08-15 20:13:30Z).** Every figure in
> this section before that PR — `424,018 B/token`, "largest sweep row is
> **B=64**", `--max-ctx 153000` — described the *pre-rolling* `xs` cache and is
> now wrong by **3.91×**. Left in place it does active harm: `--max-ctx 153000`
> makes the probe fire `WARNING[KV]` at B=128 and B=256, and §S3's own ABORT-IF
> then instructs the operator to *"drop to the largest B that does not warn"* —
> i.e. the stale number would have thrown away exactly the two rows the session
> exists to produce. Use the numbers below.

**Answer: yes — B=256 fits at 2048 context, with ~2.2 GB spare.** The
arithmetic, from the repo rather than from memory
(`memory/mission/wave30-BE-rolling-xs.md`, the PR's own derivation):

**Headroom.** 141 GB (H200) − 74.18 GB (artifact) − ~8 GB reserve (CUDA
context, prefill activations, logits, fragmentation) ⇒ **~59 GB usable for
cache.** *(The pre-#59 text said "141 − 68 = 73 raw ⇒ 65 usable". 68 GB
understates the artifact: the HF listing totals **74.19 GB** and the model card
measures **75.7 GB resident on load**. 59 GB is the figure PR #59 sized
against and is the one used throughout this section.)*

**Per-token cost.** V4-Flash is **MQA, not MLA** — there is no `kv_lora_rank`
and no compressed latent; the config carries `num_key_value_heads = 1` and
`head_dim = 512` (`mistralrs-core/src/models/deepseek4.rs`, config const
`V4_FLASH_CONFIG_JSON` at ~line 4025, copied from the HF card). **All 43 layers
keep a full cache** — the ratio-0 layers {0,1} differ only in RoPE, and "43" in
that set is the MTP slot, not a real layer. KV is **BF16** on CUDA CC≥8.0, and
there is **no KV quantization on this path**.

Two caches. **The second used to be the big one; PR #59 is what changed that.**

| cache | formula | B/token |
|---|---|---|
| attention KV, 43 layers | `43 x 2(K,V) x 1 head x 512 x 2 B` | 88,064 |
| compressor `xs`, 41 layers — **pre-#59**, verbatim `[B,T,4096]` history | `41 x 4096 x 2 B` | ~~335,872~~ |
| compressor `xs`, 41 layers — **post-#59**, rolling compressed state | 41.4 MB/seq @2048 ctx ÷ 2048 | **20,224** |
| **total, post-#59** | | **108,288 ≈ 106 KiB/token** |

The `xs` term is now **0.23×** the KV cache instead of **3.8×** it — a **3.91×**
cut in per-token footprint (423,936 → 108,288). PR #59 keeps the *compressed
rows* plus a bounded raw tail rather than the whole raw history, on the grounds
that the history is a recompute buffer and not state
(`mistralrs-core/src/kv_cache/xs_rolling.rs`).

The KV formula is the repo's own
(`paged_attention/config.rs:62-70` `kv_cache_elements_per_token`, times layers
times dtype size, as in `tuning.rs:425-436`). **PagedAttention is disabled for
V4** — `DeepSeekV4Loader::supports_paged_attention()` returns `false` because
head_dim=512 exceeds the kernel's supported sizes — so **every `--pa-*` flag is
silently inert here** and the cache is contiguous, grown in 512-token chunks.

**Per sequence, and the resulting cap** (59 GB usable, 108,288 B/token):

| context C (prompt + decode) | per-seq | max B post-#59 | max B pre-#59 | **B=256?** |
|---|---|---|---|---|
| 320 → **512 alloc** (the S3 sweep: ~64 prompt + 256 decode) | 0.055 GB | ~1,064 | ~271 | **YES** — 14.2 GB of 59 |
| 1024 | 0.111 GB | ~532 | ~135 | **YES** — 28.4 GB |
| 2048 | 0.222 GB | **~266** | ~68 | **YES** — **56.8 GB, ~2.2 GB spare** |
| 4096 | 0.444 GB | ~133 | ~34 | **NO** — largest row is **B=128** |

**So the whole sweep B ∈ {1,8,16,32,64,128,256} fits**, comfortably at the S3
sweep's own ~320-token context and still (tightly) at a full 2048. B=256 was
**3.9× out of reach on memory** before #59; it is the first batch large enough
to reach the expert-amortisation regime, since `E(B) = 256·(1−(1−8/256)^B)`
puts the 8× point at B≈256.

🔴 **`--max-ctx` must be updated with this section.** The guard is *"server
KV/context budget in tokens"*, i.e. usable-bytes ÷ bytes-per-token:

| | usable ÷ B/token | `--max-ctx` |
|---|---|---|
| pre-#59 (**stale — do not use**) | 65e9 / 424,018 | ~~153000~~ |
| **post-#59** | 59e9 / 108,288 | **545000** |

Passing the stale 153000 fires a spurious `WARNING[KV]` on the B=128 and B=256
rows and, per S3's ABORT-IF, would get them dropped. S3 below carries 545000.

> **Noticed:** with `xs` no longer dominant, **attention KV is now 81% of the
> per-token budget** and is the next thing capping batch at long context. V4 is
> MQA with `head_dim=512` and 43 full-cache layers; the reference stores 584
> B/token/layer where Arc stores 1,024. FP8 KV would roughly double feasible B
> again. Worth a separate change?

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

**Why this is not paranoia.** On the A6000 validation box, a server capped at
`--max-seqs 4` and probed at B=16 reported an aggregate of **190.01 tok/s**.
The genuine B=16 figure on the same box minutes earlier was **198.34 tok/s** —
a **4% difference**. A silently-capped sweep and a real one are
indistinguishable by inspection. `effective_B` is the only thing that tells
them apart, which is why it is a required column in §3.

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
| re-entry (boot ~10 min + cached binary <1 min + 234 GB pull ~20 min) | **~31 min ≈ $2.54** |

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

---

## 10. Preflight — run ALL of this on the laptop, before renting anything

Every check here is free and each one has, at least once, been discovered the
expensive way. **Do not create an instance until all four pass.**

**1. Is the GPU provider session alive?** *(This is the one that blocked
wave31-BF: both the Runcrate MCP server and the CLI had expired refresh tokens,
`runcrate login` needs an interactive browser, and the session could not rent a
box at all. Discovered for $0 because it was checked first; discovered after
renting it would have been billed.)*

```bash
runcrate ps               # must list instances, not "session expired"
runcrate billing balance  # must return a balance
```

If this fails with `session expired — run: runcrate login`, **the session
cannot proceed and must stop here.** Re-auth is interactive and only the user
can do it.

**2. Is anything already running and billing?** `runcrate ps` again — DOCTRINE
D10 applies across sessions, not just within one. An instance left up by a
previous session bills at $0.082/min whether or not anyone is looking at it.

**3. Is the HF token valid and does it see the private org?**

```bash
set -a; . ~/.config/arc/env; set +a
curl -s https://huggingface.co/api/whoami-v2 -H "Authorization: Bearer $HF_TOKEN" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['name'], [o['name'] for o in d.get('orgs',[])])"
# expect: heydryft ['aeonmind']
```

The HF org is **`aeonmind`**; the GitHub org is **`aeonmindai`**. A token that
does not list `aeonmind` cannot read the private overlay.

**4. Are both repos reachable and is the overlay complete?**

```bash
set -a; . ~/.config/arc/env; set +a
for R in aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 deepseek-ai/DeepSeek-V4-Flash; do
  curl -s "https://huggingface.co/api/models/$R?blobs=true" \
    -H "Authorization: Bearer $HF_TOKEN" \
  | python3 -c "
import json,sys; d=json.load(sys.stdin); s=d.get('siblings',[])
print('$R', len(s), 'files', round(sum(x.get('size',0) or 0 for x in s)/1e9,2), 'GB')"
done
# expect: overlay 15 files 74.19 GB   |   source 73 files 159.63 GB
```

Verified 2026-08-15: overlay HTTP 200, `private: True`, 15 files, 74.19 GB with
all 9 weight files byte-matching the model card; source HTTP 200, public,
ungated, 73 files, 159.63 GB.
