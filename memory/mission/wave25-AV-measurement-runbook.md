# wave25-AV — the measurement harness, and proving it measures something

Branch `perf/measurement-runbook-8` off `origin/master` (a71a9c046). Scope was
the measurement harness and its runbook ONLY — no kernel, no `qtip/`, no bake
path. Validation used a **$0.55/hr A6000** (deleted; see §5). The protected
bake box `d7d5d4ba-f5b2-4a13-bba2-b432d5a7a5c4` was never contacted.

---

## 1. What the probe was actually missing

`arc-tools/quality/batch_load_probe.py` (merged in PR #23, never trusted
end-to-end) already had aggregate decode tok/s, per-request p50/p95, TTFT
p50/p95 and a `$/Mtok`. Against the five stated requirements, three gaps:

| requirement | state before | state now |
|---|---|---|
| prefill and decode SEPARATE | **absent.** Only TTFT. No prefill rate of any kind. | three prefill numbers, each labelled with its method (§3) |
| aggregate AND per-user | present | present |
| TTFT p50 and p95 | p50 and p95 in JSON; only p50 printed | both printed on `BATCH[...]` |
| `$/Mtok` from a cost arg | `--gpu-cost-hr`; rate printed only on the sweep line | `--cost-per-hour` (old name aliased); rate printed next to **every** `$/Mtok` |
| greppable markers | `BATCH[` `BATCHSWEEP[` `WARNING[KV]` | + `PREFILL[` `CONC[` `WARN[CONCURRENCY]` `WARN[CLIENT]` `FAIL:` and a non-zero exit |

**But the real gap was the one that makes all of the above worthless: nothing
verified the server ran the batch concurrently.** The probe fired B threads
behind a barrier and reported whatever came back. A server executing those
requests one at a time produces a plausible aggregate that is really B × b=1,
and *nothing in the output contradicted it*.

This is not hypothetical on mistral.rs. **`--max-seqs` defaults to 32**
(`mistralrs-cli/src/args/mod.rs:414-416`), so a B=64 or B=128 sweep against a
default-launched server is silently a B=32 sweep. The runbook's sweep would
have shipped that number.

---

## 2. Proving concurrency is real — and that the check can fail

**Signal chosen: overlap of per-request DECODE windows** `[first token, last
token]`, swept for peak simultaneity.

Why that and not the obvious thing: the client's *submit* windows `[t0, t_end]`
all overlap **by construction**, because the launch barrier releases every
thread together — a check built on those could never fail. Decode windows have
the opposite property: under serial execution request *i* emits nothing until
*i−1* finishes, so the windows are disjoint and peak overlap collapses to 1.
Ties are broken close-before-open so two merely-touching windows never count as
overlapping.

**Independent corroboration, server-sourced.** `engine/mod.rs` stamps every
sequence in a prefill step with that whole step's wall time, surfaced as
`usage.total_prompt_time_sec`. Requests sharing a stamped duration *and*
starting to decode together shared a step, so `server_prefill_batch` is a batch
size reported by the engine, not inferred by the client.

**Gate.** Below `--min-concurrency-frac × B` (default 0.5, floor 2) →
`FAIL:` + **exit 1**, and the row is excluded from `peak` so a serialised batch
can never become the headline. Above that but below B → `WARN[CONCURRENCY]`
plus an `effective_B` column, because a capped run is a real measurement *of a
different B* and must be reported as such, never dropped.

### The mutation tests — four, two of them on real hardware

Offline (`test_batch_load_probe.py`, no GPU, no pip):

| mutation | mechanism | asserted result |
|---|---|---|
| strict serialisation | mock holds a global lock per request | exit **1**, `effective_B=1`, mean in-flight 0.71–0.76, headline suppressed |
| partial cap | mock `Semaphore(2)` at B=8 | exit **1**, `effective_B=2`, `min_required=4`; with `--min-concurrency-frac 0` it still WARNs and still reports 2-of-8 |

On the A6000, against the real server:

| mutation | mechanism | observed |
|---|---|---|
| serialise | `--max-seqs 1`, probe B=8 | exit **1** · `FAIL: CONCURRENCY[B=8] effective_B=2 (mean in-flight 1.96)` · per-user p50 **95.6 tok/s** ≈ the b=1 rate of 116 — the "B copies of b=1" pathology, visible |
| realistic cap | `--max-seqs 4`, probe B=16 | exit **1** · `FAIL: CONCURRENCY[B=16] effective_B=6 (mean in-flight 4.51)` |

**The `--max-seqs 4` run is the one that justifies the whole exercise.** Its
naive aggregate was **190.01 tok/s at "B=16"**. The genuine B=16 number
measured minutes earlier on the same box was **198.34 tok/s**. A capped run and
a real one are *within 4% of each other*. Nothing but the concurrency check
distinguishes them.

---

## 3. Prefill and decode, never blended

Three numbers, because no single one is honest on its own:

- `prefill_agg_tok_s` — **derived from TTFT + prompt length**:
  `sum(prompt_tokens) / (last TTFT − batch release)`. Client wall-clock, so it
  includes HTTP, queueing and scheduler time: a conservative **lower bound**,
  and the table's headline. The script prints the words *"derived from TTFT +
  prompt len; includes queueing — a LOWER bound"* next to it.
- `prefill_agg_server_tok_s` — **real server instrumentation** from
  `total_prompt_time_sec`, clustered into prefill steps. Compute-only **upper**
  reference.
- `prefill_per_req_tok_s_p50` — server per-sequence rate, which **understates**,
  since one sequence's "rate" is its share of a shared batched step.

**Boundary verified, not assumed** (Principle 2). Curled the real server:
`usage` is serialised on *every* chunk but is `null` on all of them except the
final one, which carries `total_prompt_time_sec: 0.015` and
`prompt_tokens: 36`. The probe's falsy-check on `usage` handles the nulls
correctly. Had this been assumed rather than probed, the null-on-every-chunk
shape would have silently produced a prefill number from the first chunk.

---

## 4. Cheap-box validation — the numbers

A6000 48 GB, CUDA build (6m07s), Qwen2.5-0.5B-Instruct, `--paged-attn off`,
`--max-seqs 32`, `--cost-per-hour 0.55`:

| B | prefill agg tok/s | decode per-user p50 | decode agg tok/s | TTFT p50/p95 | $/Mtok | effective_B |
|---|---|---|---|---|---|---|
| 1 | 4561 | 116.18 | 117.41 | 0.017/0.018 | 1.30 | 1 *(exempt)* |
| 4 | 3737 | 68.98 | 223.94 | 0.038/0.084 | 0.68 | 4 |
| 8 | 2512 | 46.87 | **318.13** | 0.070/0.648 | **0.48** | 8 |
| 16 | 2543 | 13.26 | 198.34 | 0.123/0.768 | 0.77 | 16 |

Every requirement confirmed on real hardware: **aggregate rises while per-user
degrades** (117→224→318 vs 116→69→47); **prefill is an order of magnitude off
decode** and never blended; **TTFT populated at every B**; **`effective_B` = B**
throughout; **`$/Mtok` carries its rate**.

Aggregate *falls* at B=16 on this box — a 0.5B model on an A6000 saturates
early. `WARN[CLIENT]` did not fire, so the probe was not the bottleneck. This
is a property of the toy model, not of the harness.

`--paged-attn off` was required: PagedAttention crashed on sm_86 with
`CUDA error at src/cuda/pagedattention_v1_bf16.cu:30: invalid argument`. Not in
scope here, and **not a blocker for the H200 session** — PagedAttention is
disabled for V4-Flash anyway (head_dim=512 exceeds the kernel's supported
sizes). Logged to BACKLOG rather than chased.

---

## 5. Spend and teardown

Instance `946ac888-6943-44b2-8ac2-3a973aa94ac0`, A6000 Montreal, $0.55/hr,
created 01:16:10Z, deleted ~01:35Z ⇒ **≈19 min ≈ $0.17**, against a $3 budget.
`delete_instance` returned `{"deleted": true}` and a follow-up
`list_instances` shows only the protected bake box. **DOCTRINE D10 satisfied.**

(The workspace balance delta over that window was $0.596, but that includes the
concurrently-running $1.49/hr bake box; $0.17 is this agent's share.)

No A30 was on offer. The A6000 was chosen for **28 cores** — build time, not
GPU class, dominates the cost of this validation. Worth stating plainly: what
the harness validation actually needed was a fast many-core Linux box with a
CUDA toolkit; the GPU mattered only to keep the served path faithful.

---

## 6. KV arithmetic for the H200 session — does B=128 fit?

**Yes at the sweep's context; no beyond ~1,150 tokens/sequence.** V4-Flash is
**MQA, not MLA** (`num_key_value_heads=1`, `head_dim=512`, no `kv_lora_rank` —
`deepseek4.rs`, const `V4_FLASH_CONFIG_JSON`), BF16, PagedAttention disabled:

- attention KV, 43 layers: `43 × 2 × 1 × 512 × 2 B` = **88,064 B/token**
- compressor `xs` history, 41 layers: `41 × (4096+1) × 2 B` = **335,954 B/token**
- **total 424,018 B/token ≈ 414 KiB/token**

141 − 68 = 73 GB raw, ~65 GB usable after ~8 GB for context/activations:

| context | per-seq | max B | B=128? |
|---|---|---|---|
| 512 (the sweep) | 0.217 GB | ~299 | **YES**, 27.8 GB |
| 1024 | 0.434 GB | ~149 | **YES**, tight |
| 2048 | 0.868 GB | **~74** | **NO** → top row is B=64 |
| 4096 | 1.737 GB | ~37 | **NO** → top row is B=32 |

> **Surfaced, not shipped:** the `xs` history cache is **3.8× the KV cache** and
> is what actually caps batch size. Halving it (fp8, or recompute) would roughly
> **4× the feasible batch at long context**. Separate change.

---

## 7. In-class baseline (D3) — verdict

**No single-H200 baseline exists for DeepSeek-V4-Flash on any engine, because
the model does not fit on one H200.** That is the result, not a gap.

- native checkpoint ≈**160 GB** > 141 GB before any cache
- smallest published H200 config = **4× H200** (NVIDIA Dynamo; LMSYS day-0 used
  TP4) — this is capacity, not support: SGLang/vLLM/Dynamo all ship recipes
- the one published W4A16 quant is **143 GB** and its card states verbatim:
  *"TP=2 is the only validated configuration. TP=1 OOMs on a single 141 GB H200."*
- `nvidia/…-NVFP4` is **Blackwell-only**; H200 is Hopper
- the checkpoint already ships ~4.5 bits/param, so "just quantize to 4-bit" is
  **already spent**

Arc's ~68 GB artifact is **≈1.9 bits/param** — under half the smallest
published quantized checkpoint — which is exactly what makes single-H200
serving possible. So the defensible baseline is a **footprint** claim (1 GPU vs
a published 4, with `$/Mtok` stated per-node at $19.68/hr for 4×H200) plus a
**roofline** that needs no third party: 68 GB ÷ 4.8 TB/s = 14.2 ms/step floor
⇒ ~4,500 tok/s at B=64; the 63.5 ms/step microbench is **~22% of roofline**.

Rejected: Blackwell NVL72 rack numbers (D3), and the aggregator blogs claiming
"INT4 V4-Flash on a single H200 at ~34 tok/s" — arithmetically impossible
against the 143 GB artifact, so that whole source family is unusable.

Optional stretch, clearly labelled if ever run: gpt-oss-120b (~61 GB, fits one
H200) under SGLang on the **same box, same protocol** = an *engine-efficiency*
reference, **not** a same-model baseline (2.5× fewer active params).
