# Token-path profiling

`arc-profiler` is a hierarchical, reusable profiler for Arc's inference path. It
answers one question that no previous instrumentation in this repository could:
**where does a decode step actually go — host compute, host waiting, or GPU
execution — and in which node of the call tree.**

- **Crate:** `arc-profiler/`
- **Artifacts:** one JSON (durable, machine-readable) and one self-contained
  interactive HTML page (a *view* of the JSON)
- **Enable:** `ARC_PROFILE=1`
- **Cost when off:** measured at **2.9 ns per call site** (see
  [Overhead](#overhead))

---

## 1. Why the previous instrumentation was not enough

| What existed | What it gave | Why it could not close the gap |
|---|---|---|
| `ARC_TIME_DECODE` (`deepseek4.rs:2430`) | four buckets: `moe / mla_attn / mhc_attn_pre / mhc_ffn_pre` (+3 MLA sub-timers) | Four buckets cannot locate a 150x gap. It also `device.synchronize()`s twice per timed call — **774 full device syncs per token** at 43 layers — so it changes the run it measures. |
| `STEP_us TOTAL/fwd/sample/other` (`pipeline/mod.rs`, PagedAttention arm) | a host/forward split | **V4 never takes that arm.** `DeepSeekV4Loader::supports_paged_attention()` returns `false`, so the engine always issues `DefaultInstructions`. V4 logged no host/forward split at all. |
| wave36-BN's fit | `k=105.9 ms, a=24.26 ms/seq, c=0.109 ms` | The quadratic term was found and fixed (PR #67). **The linear `a` — roughly 69 points of a B=64 step — stayed unattributed** because nothing measured below the level of "forward".

The measured starting point this profiler exists to explain: aggregate decode
**91.5 tok/s @ B=64 → 106.4 @ B=128 → 111.7 @ B=256** on an H200 serving
`qtip2b`, against a physics ceiling at B=256 of **~16,600 tok/s**
(`memory/mission/CEILINGS.json`).

---

## 2. The one thing you must understand before reading a number

**CUDA is asynchronous.** A host timer around a kernel launch measures the
*launch* — microseconds — for work that may take milliseconds. A profiler built
only on `Instant::now()` reports ~0 for the GPU and blames the host for
everything, *and it looks authoritative while doing it*.

So this profiler reports **three separate channels**, and they are structurally
prevented from bleeding into each other:

| Channel | Field | What it is | How it is measured |
|---|---|---|---|
| **wall** | `wall_ns` | wall-clock inside the span | `std::time::Instant` |
| **device** | `device_ns` | GPU stream time | `cudaEventRecord` / `cudaEventElapsedTime` on candle's stream |
| **sync** | `sync_ns` | host **blocked** on the GPU | `Instant`, on spans declared as `sync_span` |

Consequences you should internalise:

- For a **device** node, `wall_ns` is the *launch* cost. It is normally three
  orders of magnitude smaller than `device_ns`. The HTML tags these nodes
  "launch" and warns in the tooltip; do not read the wall column as GPU time.
- A `Host` node has **no API by which `device_ns` can be written**. Its device
  column is purely the roll-up of its children.
- `busy_self_ns = wall_self_ns − sync_ns` is host time genuinely spent
  computing. Waiting is never reported as computing.

### `device_ns` is stream-*elapsed*, not kernel-busy

`cudaEventElapsedTime(a, b)` is the wall time **on the stream** between the two
records. If the host cannot keep the stream fed, the stream idles between
kernels and that idle is *inside* the interval.

This is deliberate, because of what the tree then gives you for free:

> `device_self_ns` (a node's stream time minus its children's) is the stream
> time belonging to **no instrumented kernel** — launch gaps plus anything
> uninstrumented. A large `forward.device_self_ns` with small leaves **is**
> launch starvation, measured rather than argued.

What `device_ns` is **not**: SM occupancy, achieved bandwidth, or per-kernel
time under overlap. Those need Nsight. See [Limits](#7-limits).

### Proving it is not timing launches

Two independent proofs, both runnable:

1. **On a laptop, in CI.** `arc-profiler`'s test suite installs a
   `LaunchOnlyTimer` that reproduces the bug exactly — it timestamps on the host
   clock — and asserts the device and wall columns collapse together
   (`device_ns_would_equal_wall_ns_if_we_timed_launches`). Its sibling
   (`device_ns_comes_from_the_event_timer_not_the_clock`) asserts that with a
   real event timer they must differ by >100x. Neither passes for free.
2. **On the GPU.** `arc_profiler::device_selftest(&device)` issues 32 large
   matmuls inside one device span **without synchronising** and returns
   `{ launch_wall_ns, device_ns, ratio }`. A correct profiler reports
   `ratio ≫ 10`; a launch-timing one reports `ratio ≈ 1`. Run it in the GPU
   session (see [Runbook](#8-runbook-for-the-next-gpu-session)) and record the
   number.

---

## 3. Running it

```bash
# Minimal
ARC_PROFILE=1 mistralrs serve -p 1234 -m <model> --max-seqs 64

# Typical: 4 warmup steps discarded, auto-write after 100 recorded steps
ARC_PROFILE=1 \
ARC_PROFILE_WARMUP=4 \
ARC_PROFILE_STEPS=100 \
ARC_PROFILE_LABEL=B64 \
ARC_PROFILE_OUT=/root/profiles \
  mistralrs serve -p 1234 -m <model> --max-seqs 64
```

Writes `/root/profiles/B64.json` and `/root/profiles/B64.html`.

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `ARC_PROFILE` | unset | `1` enables. Anything else, or unset, disables. |
| `ARC_PROFILE_DEPTH` | `12` | Max span nesting depth. 12 reaches V4's deepest node (`…moe.experts.experts.fast.experts.gate_proj`). Lower runs cheaper; spans beyond the limit are dropped **and the report says the tree is truncated**. |
| `ARC_PROFILE_WARMUP` | `4` | Steps discarded before recording. Warmup carries allocator growth, autotune, and lazy kernel loads. |
| `ARC_PROFILE_STEPS` | unset | Auto-write the report after N recorded steps. Essential for a server that never exits. |
| `ARC_PROFILE_OUT` | `./arc-profile` | Output directory. |
| `ARC_PROFILE_LABEL` | `run` | File stem and the run's name in the HTML. |
| `ARC_PROFILE_UNROLL` | unset | `1` gives one node per layer index instead of one aggregated `layer` node. |
| `ARC_PROFILE_NO_CALIBRATE` | unset | `1` skips the startup self-calibration. |
| `ARC_PROFILE_SELFTEST` | unset | `1` runs the device-timer proof at load and records the verdict in `run.notes`. |

### Comparing several runs in one page

```bash
cargo run -p arc-profiler --bin arc-profile-report -- \
  -o batch-sweep.html b1.json b64.json b256.json
```

With two or more runs the page gains a comparison view keyed by node path:
per-run wall, device, and wall-per-token, plus a delta column against the first
run. A node absent from a run renders `—`, never `0`.

---

## 4. The node tree

Paths are dotted, from the root. `step` is one engine iteration.

```
step                                        engine/mod.rs — one scheduler iteration
├─ scheduler.lock                           contention on the scheduler mutex
├─ scheduler.schedule                       bucket selection / batch formation
├─ decode                                   the completion branch
│  ├─ pipeline.lock                         waiting for the pipeline mutex
│  └─ pipeline.step
│     ├─ input_prep                         process_inputs
│     │  ├─ input_prep.h2d_per_seq  [sync]  Tensor::new on the GPU device, once PER SEQUENCE
│     │  └─ input_prep.cat                  the B-way concat
│     ├─ cache.pre_op                       clone_in_cache when batch composition changed
│     │  └─ clone_in_cache
│     │     ├─ clone_in.alloc      [device]  2 fresh device allocs per layer
│     │     └─ clone_in.slice_set  [device]  2 device copies per sequence per layer
│     ├─ forward
│     │  └─ model                           DeepSeekV4::forward
│     │     ├─ embed               [device]
│     │     ├─ causal_mask                  returns None at decode (tgt_len == 1)
│     │     ├─ mhc.lift_3d_to_4d   [device]
│     │     ├─ layers
│     │     │  └─ layer                     aggregated: calls = n_layers x steps
│     │     │     ├─ device_map.map        [device]
│     │     │     ├─ mhc_attn_pre          [device]
│     │     │     ├─ input_layernorm       [device]
│     │     │     ├─ mla_attn              [device]
│     │     │     │  ├─ compressor_advance [device]  XsRolling history advance
│     │     │     │  ├─ q_proj             [device]
│     │     │     │  ├─ q_rmsnorm          [device]  per-head RMS norm before RoPE
│     │     │     │  ├─ kv_proj            [device]  fused wkv
│     │     │     │  ├─ kv_norm            [device]
│     │     │     │  ├─ rope               [device]
│     │     │     │  ├─ kv_fp8_quant       [device]  opt-in, ARC_V4_FP8_KV=1
│     │     │     │  ├─ kv_fp8_dequant     [device]
│     │     │     │  ├─ compressed_kv_build[device]
│     │     │     │  ├─ kv_cache_append    [device]
│     │     │     │  ├─ kv_cache_span      [device]
│     │     │     │  ├─ sdpa               [device]  dsv4_attention (window ∧ compressed)
│     │     │     │  ├─ inv_rope           [device]  NOT inside ARC_TIME_DECODE's timer
│     │     │     │  └─ o_proj             [device]  grouped LoRA
│     │     │     ├─ mix_post_attn         [device]
│     │     │     ├─ mhc_ffn_pre           [device]
│     │     │     ├─ post_attention_layernorm [device]
│     │     │     ├─ moe                   [device]
│     │     │     │  ├─ moe.gate           [device]
│     │     │     │  │  ├─ gate.router_gemm[device]  re-casts the weight to F32 every call
│     │     │     │  │  ├─ gate.scoring    [device]  sqrt(softplus) for V4
│     │     │     │  │  ├─ gate.topk       [device]  NoAuxTc / hash routing
│     │     │     │  │  └─ gate.renormalize[device]
│     │     │     │  ├─ moe.experts        [device]
│     │     │     │  │  └─ experts.fast    [device]  (or .fused / .slow — see below)
│     │     │     │  │     ├─ experts.gate_proj  [device]
│     │     │     │  │     ├─ experts.up_proj    [device]
│     │     │     │  │     ├─ experts.swiglu     [device]
│     │     │     │  │     ├─ experts.down_proj  [device]
│     │     │     │  │     └─ experts.weighted_sum [device]
│     │     │     │  └─ moe.shared_expert  [device]
│     │     │     └─ mix_post_ffn          [device]
│     │     ├─ mhc_head            [device]  learned 4D→3D collapse
│     │     ├─ final_norm          [device]
│     │     ├─ extract_logits      [device]
│     │     └─ lm_head             [device]
│     ├─ logits_d2h                [sync]    ONE batched D2H (PR #67); also the step's sync point
│     ├─ logits_split                        per-sequence views of the host tensor
│     ├─ cache.post_op                       CacheInstruction::Out — UNCONDITIONAL every step
│     │  └─ clone_out_cache
│     │     ├─ clone_out.chunk        [device]
│     │     └─ clone_out.rebuild_per_seq
│     └─ sample_and_dispatch
│        ├─ sample.join_all                  B futures; see Limits
│        │  ├─ sample.logits_cast            squeeze + F32 cast
│        │  └─ sample.ctx_clone              seq.get_toks().to_vec() PER SEQUENCE PER STEP
│        └─ finish_or_add_toks
│           ├─ stop_check                    stop tokens / max length / stop strings
│           ├─ detokenize                    tok_trie().decode_ext
│           ├─ seq.add_token
│           ├─ group_lock.is_streaming       the get_mut_group! busy-wait
│           └─ response.send                 serial responder.send().await, mutex held
├─ prompt                                    the prefill branch (same sub-tree)
└─ gpu_drain                      [sync]     stream flush at end of step, inserted by the profiler
```

`clone_in_cache` / `clone_out_cache` loop over `num_hidden_layers`, which for V4
is **not 43**: the cache vector also carries one `XsRolling` compressor-history
slot per CSA/HCA layer, so both loops run `43 + n_compressed` times.

### Nodes that are unreachable on V4 — and why

The report carries these explicitly, as striped zero-time nodes plus an entry in
the `unreachable` list with a `file.rs:LINE`. **Zero-because-unreached and
zero-because-fast must not look the same**, which is why they are declared at
the branch that bails rather than left silent.

| Path | Why it never runs on V4 | Site |
|---|---|---|
| `paged_attention` | `DeepSeekV4Loader::supports_paged_attention()` returns `false`, so `cache_config` is `None` and the engine always issues `DefaultInstructions`. The whole `CacheBackendMetadata::PagedAttention` arm — and with it `graph_wrapped_forward`, the dedicated decode path, and CUDA graph replay — is off the path. | `pipeline/mod.rs:1088` |
| `cuda_graph.capture_probe` | Gated on `ARC_V4_CAPTURE_PROBE`, unset by default. Even when set, the replay branch discards its output and is a latency measurement only. | `normal.rs:1554` |
| `cuda_graph.autonomous_decode` | Bails on `cache_config == None`, which is always true for V4 (same root cause as above). | `normal.rs:1844` |
| `experts.fused` / `experts.fast` | Exactly one expert backend is selected at load; the other two are declared unreachable naming which one won. | `moe/experts.rs:97`/`:103` |

**So: no CUDA graph capture or replay executes in a default V4 run.** If a
future change makes one reachable, the node stops being striped and starts
carrying time — which is the point of declaring it rather than omitting it.

---

## 5. Reading the HTML

Everything is inline; the page opens with no network.

- **Icicle chart** of the span tree. The **metric switch** (`wall` / `device` /
  `sync` / `busy_self`) re-lays out the chart — this is the control that
  separates "the host was busy" from "the GPU was busy" visually. Click a node
  to zoom; breadcrumb to go back; Esc resets. Hover for exact ns, calls, per-call
  mean/min/max, all four channels, and geometry (`b` / `t` / `tokens`).
- **Colour by kind**: host / device / sync, with a legend. Device nodes carry a
  "launch" tag so their wall column is never misread.
- **Hatching**: −45° with a ⊘ marker means *never ran* (unreachable); +45° means
  *ran, but below the timer floor on this metric*. Neither renders as invisible.
- **Sortable table** of every node — path, kind, calls, wall total/self, device
  total/self, sync, busy self, % of root wall, mean, min, max — with a path
  filter. Clicking a row selects it in the icicle and vice versa.
- **Health panel**, always visible: reconciliation violations, `misnested_spans`,
  `unresolved_device_spans`, and the unreachable list. When clean it says so
  explicitly — "reconciles within 2.00% — checked and clean" — because *checked
  and clean* and *never checked* must be distinguishable.
- **Overhead banner** if the profiler's own cost exceeds 5% of a step.
- **Summary bar**: one plain-English line such as
  `host busy 1,842 ms · host blocked on GPU 61 ms · GPU executing 88 ms over a 1,903 ms wall`.

---

## 6. The JSON schema

`schema: "arc-profile/1"`. Top level: `run` (provenance), `totals`, `overhead`,
`nodes[]`, `unreachable[]`, `reconciliation`.

Each node:

```jsonc
{
  "id": 12, "parent": 7, "name": "sdpa",
  "path": "step.decode.pipeline.step.forward.model.layers.layer.mla_attn.sdpa",
  "depth": 8, "kind": "device",          // root | host | device | sync
  "calls": 4300,
  "wall_ns": 0, "wall_self_ns": 0,       // wall_self = wall − Σ children wall
  "device_ns": 0, "device_self_ns": 0,   // rolled up through host parents
  "sync_ns": 0, "busy_self_ns": 0,       // busy_self = wall_self − sync
  "min_wall_ns": 0, "max_wall_ns": 0,
  "geom": { "b": 64, "t": 1, "tokens": 64 },
  "reachable": true, "note": null,
  "children": [13, 14]
}
```

`Profile::recheck(tolerance_pct)` re-derives the reconciliation check from the
node table **alone**, so a consumer can re-verify a JSON someone hands them
without trusting the writer. The crate's own tests use that path rather than the
accumulator's.

---

## 7. Overhead

Measured through the real public API by flipping the real gate — not estimated,
not argued from the code. Reproduce with:

```bash
cargo test -p arc-profiler --release overhead_is_measured -- --nocapture
```

| State | Cost per instrumented call site | Measured on |
|---|---|---|
| **OFF** (`ARC_PROFILE` unset) | **2.9 ns** | Apple M-series, release, 20k iterations |
| **ON** | **92.6 ns** (open + close, registry lookup, atomics) | same |

The off-state cost is one relaxed atomic load and a predictable branch. In
particular there is **no `env::var_os` per timer call** — an earlier iteration of
the V4 timers did exactly that and paid ~390 environment scans per forward *with
profiling disabled*. A test fails if the off-state cost exceeds 50 ns, which is
the regression tripwire for that mistake.

Every run's report carries its own `overhead` block:
`enabled_ns_per_span`, `disabled_ns_per_span`, `spans_per_step`, and
`enabled_overhead_pct = enabled_ns_per_span × spans_per_step ÷ mean_step_wall`.
**If that exceeds 5% the HTML raises a banner saying the profile is materially
measuring itself.**

At V4's roughly 1,000 spans per decode step, the host-side cost is ≈ 0.09 ms per
step: ~0.004% of a 2,105 ms B=64 step, ~0.15% of a 61 ms b=1 step.

⚠️ **The host half is measured; the device half is not, yet.** Each device span
also issues two `cudaEventRecord`s, which cost stream time. That has never been
measured on hardware. The runbook below includes the A/B that settles it, and
until it is run, treat the enabled-overhead figure at b=1 as a lower bound.

---

## 8. Limits — what this cannot see, and why

1. **Per-sequence sampler time on the rayon pool is not decomposed.** For `B>1`,
   `Sampler::sample` runs via `tokio_rayon::spawn`, off the engine thread, and
   the `B` futures are polled interleaved under `join_all`. A span opened inside
   a concurrently-polled future would interleave with its siblings and corrupt
   the tree shape, so `sample.join_all` reports the batch's wall time and the
   per-sequence *host prologue* (`sample.logits_cast`, `sample.ctx_clone`) is
   broken out separately. The report states this in `run.notes`.
2. **Work on other threads is not in the tree.** A span opened on a rayon worker
   has no parent on that thread's stack and would become a spurious root. Nothing
   currently instruments such regions; if something does, it must be reviewed
   against this constraint.
3. **`device_ns` is stream-elapsed, not kernel-busy** (§2). For per-kernel
   occupancy, bandwidth, or overlap analysis, use Nsight Systems/Compute.
4. **Single device.** The timer is bound to one candle stream by
   `attach_device`. Multi-GPU device-mapped runs will report device time only for
   the attached device; host spans are unaffected.
5. **Aggregated by default.** One `layer` node covers all layers; per-layer
   variance shows only as `min`/`max` unless `ARC_PROFILE_UNROLL=1`.
6. **Depth truncation is silent in the tree but loud in the header.** Spans past
   `ARC_PROFILE_DEPTH` are dropped and `run.notes` says how many.
7. **Misnesting is counted, not prevented.** A span opened on one thread and
   closed on another still accumulates into the correct node (identity is bound
   at open) but ticks `reconciliation.misnested_spans`. **A non-zero value there
   means the tree shape is not trustworthy** — read the health panel first.
8. **`ARC_TIME_DECODE` still perturbs.** It is independent of this profiler and
   still synchronises twice per timed call. Do not enable both when the numbers
   matter.

---

## 9. Runbook for the next GPU session

Produces a full three-batch profile of the served `qtip2b` artifact. This is the
run that attributes the ~69% linear term and the 150x gap.

**Prerequisites:** an H200 (141 GB — the artifact needs 74.18 GB plus KV), the
published UQFF, and a build **without `cudnn`** (−62% decode).

```bash
# 0) Build. cudnn deliberately omitted.
cargo build --release --features "cuda flash-attn"

# 1) Prove the profiler is not timing launches. Do this FIRST — every device
#    number below is void if the ratio is not >> 10.
#    The selftest runs at pipeline construction, so any short invocation
#    triggers it and it costs nothing but the load.
ARC_PROFILE=1 ARC_PROFILE_SELFTEST=1 \
  ./target/release/mistralrs run -m $SRC -a deepseekv4 \
    --from-uqff $UQFF/qtip2-0.uqff 2>&1 | tee selftest.log
# Look for:  "device selftest: launch_wall=... device=... ratio=...x — PASS/FAIL"
# ratio > 10  => CUDA events are measuring execution.   PROCEED.
# ratio ~ 1   => STOP. The device column is void; report the ratio and stop.
# "NOT RUN"   => no CUDA timer attached; device columns will be unmeasured.

# 2) Overhead A/B on hardware — the device half the CPU measurement cannot cover.
#    Same batch, same seed, profiler off then on. Record both tok/s.
for P in "" "ARC_PROFILE=1"; do
  env $P ./target/release/mistralrs serve -p 1234 -m $SRC -a deepseekv4 \
      --from-uqff $UQFF/qtip2-0.uqff --max-seqs 64 &
  SRV=$!
  sleep 60
  python3 scripts/batch_probe.py --batch 64 --tokens 128   # asserts effective_B
  kill $SRV
done
# If enabling the profiler moves aggregate tok/s by more than a few percent,
# say so in the report and re-run the profile at ARC_PROFILE_DEPTH=6.

# 3) The three profiles. --max-seqs MUST equal the batch: mistral.rs defaults to
#    32, and a B=64 sweep against a default server is silently a B=32 sweep.
for B in 1 64 256; do
  ARC_PROFILE=1 \
  ARC_PROFILE_WARMUP=8 \
  ARC_PROFILE_STEPS=100 \
  ARC_PROFILE_LABEL="B$B" \
  ARC_PROFILE_OUT=/root/profiles \
    ./target/release/mistralrs serve -p 1234 -m $SRC -a deepseekv4 \
      --from-uqff $UQFF/qtip2-0.uqff --max-seqs $B &
  SRV=$!
  sleep 60
  python3 scripts/batch_probe.py --batch $B --tokens 200   # asserts effective_B == B
  wait $SRV || kill $SRV
done

# 4) Per-layer detail, B=64 only, if the aggregate layer node hides something.
ARC_PROFILE=1 ARC_PROFILE_UNROLL=1 ARC_PROFILE_LABEL=B64-unrolled ... (as above)

# 5) Pull the artifacts BEFORE deleting the box.
runcrate cp <box>:/root/profiles ./profiles

# 6) One page, three runs, side by side. No GPU needed.
cargo run -p arc-profiler --bin arc-profile-report -- \
  -o profiles/batch-sweep.html \
  profiles/B1.json profiles/B64.json profiles/B256.json
```

### Reading the result — what would refute what

| Observation | Conclusion |
|---|---|
| `step.decode.pipeline.step` wall ≫ `forward` wall, and the excess is `busy_self` | Host compute dominates. Look at `cache.post_op`, `sample_and_dispatch`, `input_prep`. |
| The excess is `sync_ns` (mostly `logits_d2h` / `gpu_drain`) | The host is **waiting**. The GPU is the constraint; look at the device column. |
| `forward.device_self_ns` large, leaves small | **Launch starvation**: the stream idles between kernels. This is the case CUDA graphs would attack. |
| `layer.mla_attn.device_ns` ≈ `layer.moe.device_ns` at B=64 | Contradicts the standing finding that MoE grows to 49–64% at batch — re-check before publishing either. |
| `reconciliation.violations` non-empty, or `misnested_spans` > 0 | **Do not quote any number from this run** until it is explained. |
| `unresolved_device_spans` > 0 with a CUDA backend attached | Some device spans produced no time. Their zeros are missing data, not fast kernels. |

**Cost:** one H200 for roughly 45 minutes including load (~13 s per load, three
servers, one probe each) ≈ **$4**. Delete the box immediately afterwards.
