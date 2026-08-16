# wave52-CC — hierarchical token-path profiler (`arc-profiler`)

**Order (Jish, verbatim):** *"make a flowchart of every checkpoint a token passes
through and how much each checkpoint/step costs in ms or ns, like a profiling,
and I mean everything, cuda graph, attention, whatever, profile everything,
don't leave anything, every step and every substep, like a tree. If you can make
a profiler that can be reused, that's even better. And document this whole shit,
and this profiling should report in an html and a json, the html should be
highly detailed and interactive."*

**Branch:** `feat/token-path-profiler` · **PR:** draft, base `master`
**Scope fence honoured:** new crate + instrumentation call sites + report
generator only. **No model math, kernel, or scheduling behaviour was changed.**
**No GPU rented.** Everything below was built and tested on CPU.

---

## 1. Why this was the highest-value work

Measured on H200 serving `qtip2b`: **91.5 tok/s @B=64 → 106.4 @B=128 → 111.7
@B=256**, against a **~16,600 tok/s** physics ceiling at B=256 (`CEILINGS.json`)
— **~150x below**, and nobody could say where the time went.

Three prior attempts and their exact ceilings:

| Instrumentation | Gave | Why it could not close it |
|---|---|---|
| `ARC_TIME_DECODE` | 4 buckets (`moe/mla_attn/mhc_attn_pre/mhc_ffn_pre`) | Four buckets cannot find a 150x gap. Worse: it `device.synchronize()`s **twice per timed call = 774 full syncs per token** at 43 layers, so it perturbs what it measures. |
| `STEP_us TOTAL/fwd/sample/other` | a host/forward split | **Exists only on the PagedAttention arm** (`pipeline/mod.rs:1088`), which **V4 provably never takes** — `DeepSeekV4Loader::supports_paged_attention()` returns `false` (`normal_loaders.rs:3231`) ⇒ `cache_config == None` ⇒ engine always issues `DefaultInstructions`. V4 logged **no** host/forward split at all. |
| wave36-BN's fit | `k=105.9 ms, a=24.26 ms/seq, c=0.109 ms` | Quadratic found + fixed (PR #67). **The linear `a` — ~69 points of a B=64 step — remained UNATTRIBUTED** because nothing measured below "forward". |

---

## 2. The trap, and how it was avoided

**CUDA is asynchronous.** A host timer around a launch measures the *launch*
(µs) for work that takes ms. A profiler built only on `Instant::now()` reports
~0 for the GPU and blames the host for everything — and looks authoritative
doing it. Given the two extremes already measured on this system (GPU at 0–4%
util / 121 W of 700 at B=256, while one CPU core of 44 pegged at 100%), a
profiler that cannot tell "host busy" from "host waiting" would have been
actively misleading.

So there are **three channels, structurally separated**:

| channel | field | how measured |
|---|---|---|
| wall | `wall_ns` | `Instant` |
| device | `device_ns` | `cudaEventRecord` / `cudaEventElapsedTime` on **candle's** stream |
| sync | `sync_ns` | `Instant`, only on spans declared `sync_span` |

- A `Host` node has **no API** by which `device_ns` can be written. Its device
  column is purely a roll-up of children.
- `busy_self_ns = wall_self_ns − sync_ns`. Waiting is never reported as compute.
- Events are recorded but **resolved lazily**, once per step after a single
  stream flush — querying an event inside a span would serialise the pipeline
  and change the measurement.

**`device_ns` is stream-ELAPSED, not kernel-busy** — stated in the docs, the
crate docs, and every report's `run.notes`. That is deliberate: a node's
`device_self_ns` (own stream time minus children's) is then exactly *the stream
time belonging to no instrumented kernel*, i.e. **launch starvation, measured
rather than argued.**

---

## 3. What was built

```
arc-profiler/
  src/lib.rs        gate, span guards, step lifecycle, snapshot, self-calibration, GPU self-test
  src/tree.rs       span-tree accumulator; derives self-times + device roll-up in ONE place
  src/device.rs     DeviceTimer trait; CUDA event pool (recycled); NullTimer
  src/report.rs     the JSON schema (`arc-profile/1`) + independent `recheck()`
  src/html.rs       splices the JSON into the template at one marker
  src/template.html 75 KB self-contained interactive page, zero external assets
  src/bin/arc_profile_report.rs   N JSONs -> one comparison page (no GPU needed)
  src/tests.rs      24 unit tests incl. the mutation proofs
  tests/end_to_end.rs  full V4-shaped tree -> JSON + HTML -> read back and check
```

Registered in the workspace, in `mistralrs-core`'s deps, and in **both CI lanes**
(`fmt` and the scoped `clippy -D warnings`).

### Nodes instrumented (43 distinct, ~1,000 span opens per decode step)

`step` → `scheduler.lock`, `scheduler.schedule`, `decode`/`prompt` →
`pipeline.lock`, `pipeline.step` → `input_prep` (+`input_prep.h2d_per_seq`
[sync], `input_prep.cat`), `cache.pre_op`→`clone_in_cache`(+`clone_in.alloc`,
`clone_in.slice_set`), `forward`→`model`→ `embed`, `causal_mask`,
`mhc.lift_3d_to_4d`, `layers`→`layer`→ {`device_map.map`, `mhc_attn_pre`,
`input_layernorm`, `mla_attn`→{`compressor_advance`, `q_proj`, `q_rmsnorm`,
`kv_proj`, `kv_norm`, `rope`, `kv_fp8_quant`, `kv_fp8_dequant`,
`compressed_kv_build`, `kv_cache_append`, `kv_cache_span`, `sdpa`, `inv_rope`,
`o_proj`}, `mix_post_attn`, `mhc_ffn_pre`, `post_attention_layernorm`,
`moe`→{`moe.gate`→{`gate.router_gemm`, `gate.scoring`, `gate.topk`,
`gate.renormalize`}, `moe.experts`→`experts.fast`→{`experts.gate_proj`,
`experts.up_proj`, `experts.swiglu`, `experts.down_proj`,
`experts.weighted_sum`}, `moe.shared_expert`}, `mix_post_ffn`}, `mhc_head`,
`final_norm`, `extract_logits`, `lm_head`; then `logits_d2h` [sync],
`logits_split`, `cache.post_op`→`clone_out_cache`(+`clone_out.chunk`,
`clone_out.rebuild_per_seq`), `sample_and_dispatch`→{`sample.join_all`
(+`sample.logits_cast`, `sample.ctx_clone`), `finish_or_add_toks`→{`stop_check`,
`detokenize`, `seq.add_token`, `group_lock.is_streaming`, `response.send`}};
plus `gpu_drain` [sync] auto-inserted at step end.

Every named suspect from prior investigations has a node: the `get_mut_group!`
busy-wait, the unconditional `CacheInstruction::Out`, `seq.get_toks().to_vec()`
per sequence per step, the serial `responder.send().await` under the pipeline
mutex, the per-sequence GPU `Tensor::new` (CLAUDE.md pitfall #5), the logits D2H.

### 🔴 The old four-bucket component percentages are OVER-ATTRIBUTED TO SDPA

Placing spans against the real call sites surfaced a defect in the **existing**
`ARC_TIME_DECODE` timers that invalidates every component percentage this
program has quoted from them. "SDPA" was never measured — it is **derived by
subtraction**, `sdpa = mla_attn − (q_proj + kv_proj_rope + invrope_oproj)`
(`deepseek4.rs:2497-2507`; the emitted line says so itself:
`"(sdpa=mla_attn-these)"`). That is only valid if those three cover everything
in `mla_attn` bar the kernel. They do not:

- `MLA_NS[1]` "kv_proj_rope" wraps **only** `self.wkv.forward_autocast(xs)`
  (`:1433`) — `kv_norm`, `apply_rope_inplace`, the FP8 K quant/dequant,
  `append_kv_mqa` and `cached.span(...)` all leak into the residual;
- `MLA_NS[2]` "invrope_oproj" wraps **only** the o_proj block (`:1762`) —
  `forward_inverse_tail` (`:1750`) is **outside** it, i.e. the inverse RoPE the
  name claims to include leaks in too;
- `compressor_advance` and `compressed_kv_from_rows` sit inside `mla_attn` and
  are wrapped by no MLA timer at all.

⇒ The "SDPA" residual is the attention kernel **plus** RoPE, inverse RoPE,
kv_norm, the KV append + span, the FP8 round trip, and the compressor. SUPERSEDE:
the b=1 split `mla_attn 49% (q_proj 22 ms, kv_proj_rope 7.7 ms, invrope_oproj
16.6 ms, rest = SDPA)` and the `fp8_matmul 31.5% / qtip_dequantize 26.5%`
framing built on it. The B=64 `mla_attn 39%` **total is sound** (it wraps the
whole attention call) — only the decomposition inside it is wrong, so "MoE grows
with batch, everything else collapses" survives; "attention is mostly SDPA" does
not. `arc-profiler` measures all fourteen MLA sub-ops directly with **no
subtraction anywhere**; the §7 GPU run replaces the superseded numbers, and
until it lands, quote neither set.

### Declared UNREACHABLE on V4 (striped in the HTML, never a silent zero)

| path | why | site |
|---|---|---|
| `paged_attention` | `supports_paged_attention()==false` ⇒ `cache_config==None` ⇒ the whole PagedAttention step arm, `graph_wrapped_forward`, dedicated decode and CUDA-graph replay are off the path | `pipeline/mod.rs:1088` |
| `cuda_graph.capture_probe` | gated on `ARC_V4_CAPTURE_PROBE`, unset by default; even when set the replay branch **discards its output** | `normal.rs:1554` |
| `cuda_graph.autonomous_decode` | bails on `cache_config == None` — always, on V4 | `normal.rs:1844` |
| `experts.fused` / `experts.fast` | one backend wins at load; the report names which | `moe/experts.rs:97`/`:103` |

⇒ **No CUDA graph capture or replay executes in a default V4 run.** Jish's
"profile cuda graph" is answered as *"it does not run, here is the line that
bails"*, which is the honest answer.

🔴 **Headline in its own right:** three independent bails, all rooted in
`supports_paged_attention() == false` ⇒ `cache_config == None`, mean the
graph/megakernel path on V4 is **UNREACHABLE, not merely deferred**. Any plan
treating "turn on CUDA graphs" as a tuning step is mis-scoped — it is a
prerequisite project (give V4 a cache config, or a capture path that needs
none). Equally: no V4 measurement to date can have been affected by graph
capture, in either direction.

---

## 4. Overhead — measured in both states, not asserted

`cargo test -p arc-profiler --release overhead_is_measured -- --nocapture`

| state | ns per call site |
|---|---|
| **OFF** (`ARC_PROFILE` unset) | **2.9 ns** — one relaxed atomic load + branch |
| **ON** | **92.6 ns** — open + close, registry lookup, atomics |

At ~1,000 spans/step: **0.09 ms/step host cost** = 0.004% of a 2,105 ms B=64
step, 0.15% of a 61 ms b=1 step. A test **fails** if the off-state exceeds 50 ns
— the tripwire for the `env::var_os`-per-timer regression that once cost ~390
environment scans per forward with profiling *disabled*.

Every report carries its own `overhead` block and the HTML raises a banner above
5%.

🔴 **The device half is NOT yet measured.** Each device span issues two
`cudaEventRecord`s, which cost stream time, and that has never run on hardware.
The runbook's step 2 is the A/B that settles it. Until then the b=1 figure is a
**lower bound** and is labelled as such.

---

## 5. Tests — 26 total, all green on CPU

`cargo test -p arc-profiler` → **24 unit + 1 e2e + 1 doc, 0 failed.**
`cargo test -p mistralrs-core` → **310 + 13 passed, 0 failed** (no regression).

**Known-truth tests:** a 25 ms span measures 24–80 ms; a 3-level nest lands
~5/10/10 ms in the right self-times; 5 calls aggregate with a max that reflects
the one long call.

**Mutation proofs (D12 — this repo has found 7+ tests that passed while their
assertion was unreachable):**

1. `device_ns_comes_from_the_event_timer_not_the_clock` asserts device (40 ms
   from a fake timer) ≥100x wall. **Its mutation**,
   `device_ns_would_equal_wall_ns_if_we_timed_launches`, installs a
   `LaunchOnlyTimer` that reproduces the bug exactly (host-clock timestamps) and
   asserts the two columns **collapse to a ratio of ~1** — i.e. the first
   assertion has teeth.
2. `a_well_formed_tree_reconciles` asserts zero violations **and** that
   `wall_self + Σ children == parent` exactly. **Its mutation**,
   `reconciliation_flags_a_child_that_exceeds_its_parent`, makes a child claim
   2x its parent and asserts the check fires; `reconciliation_flags_a_broken_
   device_rollup` does the same on the device channel.
3. `tolerance_is_a_band_not_a_rubber_stamp`: +1% passes, +50% fails. Without
   both halves the 2% tolerance could be any number and no test would notice.
4. `a_device_span_with_no_timer_is_unmeasured_not_zero` — a device span with no
   backend increments `unresolved_device_spans` and the header names the
   backend, so **missing never reads as zero**.
5. `unreachable_paths_are_labelled_not_silently_zero`, `warmup_steps_are_
   discarded`, `the_depth_limit_truncates_and_says_so`,
   `sync_time_is_waiting_and_is_not_counted_as_busy_host_time`,
   `device_self_time_exposes_stream_gaps_between_children`.

`Profile::recheck()` re-derives reconciliation **from the node table alone**, so
tests verify the artifact rather than trusting the accumulator that wrote it.

**HTML verified in a real browser** (Playwright, served over http): renders,
zero page-level console errors, all panels present — icicle + metric switch,
sortable/filterable table, health panel reading *"Checked and clean… within
2.00%; 0 misnested; 0 unresolved"*, unreachable table with reason + site,
overhead block, and the summary line *"host busy … · host blocked on GPU … ·
GPU executing … over a … wall"*.

---

## 6. Known limits (also in `docs/engineering/PROFILING.md` §8)

1. **Per-sequence sampler time on the rayon pool is not decomposed.** For B>1
   the sampler runs via `tokio_rayon::spawn` and the B futures are polled
   interleaved under `join_all`; a span inside a concurrently-polled future
   would interleave with its siblings and corrupt the tree. `sample.join_all`
   carries the batch wall time; the per-sequence host prologue is broken out.
   The report states this in `run.notes`.
2. Work on other threads is not in the tree (would become a spurious root).
3. `device_ns` is stream-elapsed; per-kernel occupancy/bandwidth needs Nsight.
4. Single device — multi-GPU device-mapped runs report device time for the
   attached device only.
5. Aggregated by default; `ARC_PROFILE_UNROLL=1` for per-layer nodes.
6. Misnesting is **counted, not prevented**; non-zero ⇒ the tree shape is not
   trustworthy and the health panel says so.

---

## 7. WHAT NEEDS A GPU — the exact commands

Full runbook in `docs/engineering/PROFILING.md` §9. Summary: **one H200,
~45 min, ≈$4.** Build **without cudnn** (−62% decode).

```bash
# 1) PROVE the device timer is not timing launches. Do this FIRST.
ARC_PROFILE=1 ARC_PROFILE_SELFTEST=1 ./target/release/mistralrs run \
  -m $SRC -a deepseekv4 --from-uqff $UQFF/qtip2-0.uqff 2>&1 | tee selftest.log
#   "device selftest: ... ratio=NNx — PASS/FAIL"
#   ratio > 10 => proceed.  ratio ~ 1 => STOP, the device column is void.

# 2) Overhead A/B on hardware (the device half CPU cannot measure):
#    same batch/seed, ARC_PROFILE unset vs =1, record both tok/s.

# 3) The three profiles. --max-seqs MUST equal the batch (mistral.rs defaults
#    to 32; a B=64 sweep against a default server is silently B=32).
for B in 1 64 256; do
  ARC_PROFILE=1 ARC_PROFILE_WARMUP=8 ARC_PROFILE_STEPS=100 \
  ARC_PROFILE_LABEL="B$B" ARC_PROFILE_OUT=/root/profiles \
    ./target/release/mistralrs serve -p 1234 -m $SRC -a deepseekv4 \
      --from-uqff $UQFF/qtip2-0.uqff --max-seqs $B &
  sleep 60; python3 scripts/batch_probe.py --batch $B --tokens 200
done

# 4) Pull BEFORE deleting the box, then merge into one page (no GPU):
runcrate cp <box>:/root/profiles ./profiles
cargo run -p arc-profiler --bin arc-profile-report -- -o profiles/sweep.html \
  profiles/B1.json profiles/B64.json profiles/B256.json
```

**Pre-committed reading, so the result cannot be rationalised after the fact:**

| observation | conclusion |
|---|---|
| `pipeline.step` wall ≫ `forward`, excess is `busy_self` | host compute dominates → `cache.post_op`, `sample_and_dispatch`, `input_prep` |
| excess is `sync_ns` (`logits_d2h`/`gpu_drain`) | host is **waiting**; the GPU is the constraint |
| `forward.device_self_ns` large, leaves small | **launch starvation** — the case CUDA graphs would attack |
| `mla_attn.device_ns ≈ moe.device_ns` at B=64 | contradicts the standing "MoE 49–64% at batch" finding — re-check before publishing either |
| violations non-empty or `misnested_spans > 0` | **quote nothing from that run** |

---

## 8. Files

New: `arc-profiler/**` (9 files), `docs/engineering/PROFILING.md`, this log.
Modified: `Cargo.toml`, `mistralrs-core/Cargo.toml`, `.github/workflows/ci.yml`,
`engine/mod.rs`, `pipeline/{mod,normal,sampling,inputs_processor}.rs`,
`kv_cache/mod.rs`, `models/deepseek4.rs`, `moe/experts.rs`.
**477 insertions / 105 deletions across mistralrs-* — no mass reformat.**
