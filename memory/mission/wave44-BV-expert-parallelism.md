# wave44-BV — multi-GPU expert parallelism for V4's MoE

**Date:** 2026-08-16 · **Base:** `master` @ `372976933` · **Branch:** `docs/expert-parallelism-design`
**NO GPU WAS RENTED. $0.00.** A paid bake is live; nothing here touched it.
Everything below is source audit + reference-code citation + arithmetic against
`CEILINGS.json` and `FACTS.md`. Every number is graded `[measured]`,
`[computed]` or `[projected]`. Nothing here is a measurement of Arc on
multiple GPUs, because Arc has never run on multiple GPUs for inference.

---

## 0. Headline

**Expert parallelism is the only lever that raises the per-user ceiling at
batch, comms does not eat the win on NVLink, and the first rung that clears
Jish's 100 tok/s target is TWO cards, not eight.**

Four things this document establishes:

1. **Arc has no expert parallelism and no working collective.** The `Comm`
   that is constructed in every production run is `Comm::Dummy(rank=0,
   world_size=1)`; `SumAllReduce` is `Ok(xs.clone())`. The only LIVE multi-GPU
   path is **layer-wise pipeline sharding**, which moves one hidden state per
   layer boundary and pools *capacity*, not *bandwidth*. NCCL exists but is
   not compiled into the shipped binary. §1.
2. **The all-to-all is cheap on NVLink and expensive everywhere else.**
   At B=128, N=8 the dispatch+combine moves **43.5 MB per GPU per step**
   against **2.15 ms of weight reading** — 0.53 ms of comms on NVLink (20% of
   the step), 3.5 ms on InfiniBand (62%). **Bandwidth is not the problem;
   86 synchronising collectives per step is.** §2.
3. **Arc does not need EPLB at the batch sizes where EP matters.** Arc's
   dominant cost is *which experts are woken* (weight bytes), not *how many
   tokens each got. At B=128, 95.2% of all 256 experts are woken, so a GPU
   cannot read more than the experts it owns: **weight-read imbalance is
   hard-bounded at +4.2% at B=128 and +0.2% at B=256**, regardless of routing
   skew. The reference systems all ship EPLB; Arc can skip it in stage 1 and
   revisit only if it becomes compute-bound. §3.
4. **Expert parallel beats both tensor parallel and plain replication for V4,
   on per-user latency AND on $/Mtok, at every concurrency checked.** §5.

And the thing that must not be lost: **EP multiplies whatever the single card
delivers.** Measured today at B=128 is **0.27 tok/s per user / 28.86 aggregate**
[measured, wave34-BL] against a 68 / 8,701 single-card ceiling — a ~300× gap
that is entirely implementation. EP×8 turns 0.27 into ~2.2, not into 100.
**EP is necessary and not sufficient, and it is not first.** §6.

---

## 1. What exists — LIVE / CONDITIONAL / DEAD

Three unrelated multi-GPU systems live in this repo. Exactly one runs in a
production inference request.

| System | Status | Gate |
|---|---|---|
| Layer-wise pipeline sharding (`LayerDeviceMapper`) | 🟢 **LIVE** | none — default |
| NCCL / ring tensor parallelism | 🟡 **CONDITIONAL** — not compiled in prod | `--features nccl`/`ring` |
| Multi-GPU UQFF bake (`--bake-devices`) | 🟢 **LIVE**, quantize-time only | flag / `ARC_BAKE_DEVICES` |
| Expert parallelism / MoE sharding / all-to-all | 🔴 **ABSENT** | — |

### 1.1 🟢 LIVE — layer-wise pipeline sharding

`DeviceMapSetting::into_mapper` (`mistralrs-core/src/device_map.rs:83-279`)
builds one `Device` entry **per layer**, extended in contiguous runs
(`device_map.rs:223`). `LayerDeviceMapper::map` is one line —
`input.to_device(&self.mappings[layer])` (`device_map.rs:319`) — called
per-layer in every model forward, for V4 at
`mistralrs-core/src/models/deepseek4.rs:3758` (4-D mHC path) and `:3782`.

Auto-mapping (`pipeline/loaders/auto_device_map.rs:192-501`) greedily fills each
device with as many layers as `layer_size + kv_cache_bytes` allows (`:440-451`).
CLI surface: `-n / --device_layers "0:10;1:20"`
(`mistralrs-cli/src/args/model.rs:117-120`), parsed at
`mistralrs-server-core/src/mistralrs_for_server_builder.rs:1117-1156`; `None`
→ `Auto` (`:1154`). PagedAttention *is* device-map aware and splits the KV
cache per device (`pipeline/normal.rs:527-530`,
`paged_attention/cache_engine.rs:174,189,221-226`).

**What it buys and what it does not.** Each GPU owns a contiguous slab of
layers; the only cross-GPU traffic is one hidden-state handoff per boundary.
That pools **HBM capacity**. It does **not** pool bandwidth for a single step:
every card still reads every expert its slab's tokens woke, and the cards run
**sequentially** — card 1 is idle while card 0 does layers 0-21. For V4 on a
card the model already fits on, pipeline parallelism is close to worthless for
decode latency.

Crucially, `LayerDeviceMapper::get_comm_for` (`device_map.rs:372-380`) hands
every layer a fresh `world_size=1` Dummy `Comm`. **This is pipeline
parallelism with zero collectives.**

### 1.2 🔴 The `Comm` in production is always Dummy

- `Comm::from_device` (`mistralrs-quant/src/distributed/mod.rs:110-134`) falls
  through to `Ok(Self::Dummy(...))` at `:131` with no nccl/ring feature.
- `DummyComm` (`:330-354`): `rank() -> 0`, `world_size() -> 1`.
- `SumAllReduce` dummy impl (`:1076-1084`): **`sum_all_reduce(xs) -> Ok(xs.clone())`**.
- `use_nccl()` (`:92`) is `false` because of the `cfg!(feature = "nccl")`
  conjunct.
- **Production build:** `deploy/modal_b200.py:26` —
  `cargo build --release -p mistralrs-cli --features 'cuda flash-attn'`. No
  `nccl`, no `ring`. The only nccl build is a manual-dispatch-only CI lane
  (`.github/workflows/ci_cuda.yaml:12,24`).

`RowParallelLayer` / `ColumnParallelLayer` are **not optional TP wrappers — they
are the model's Linear layers**, used in ~45 model files (V4 at
`deepseek4.rs:968,978,1054,1074`). At `world_size == 1` they collapse to plain
linears: `Shard::Simple { world_size: 1 }` is short-circuited at
`mistralrs-quant/src/safetensors.rs:509-512`.

One live cost: `RowParallelLayer::forward` (`distributed/layers.rs:223`) calls
`sum_all_reduce` **unconditionally** — no `world_size > 1` guard — so every
row-parallel forward pays a `.contiguous()` + `Arc` clone even single-rank.
`mistralrs-core/src/moe/experts.rs:470` *does* guard it. Inconsistent.

### 1.3 🔴 Dead code inventory (present, plumbed, documented, zero call sites)

| Item | Location |
|---|---|
| `AllGather` — all 4 impls + wrapper, exported at `lib.rs:74` | `distributed/mod.rs:411-463, 617-764, 947-1065, 1086-1097` |
| `BarrierLike::wait` + all 3 impls; `barrier_all`/`barrier_crossnode` fields constructed, never used | `distributed/mod.rs:48-57`, `socket.rs:60-84, 133-154, 41-42, 109-110` |
| `NcclPipelineParallelMapper` (self-declared `#[allow(dead_code)]`) | `device_map.rs:519-586` |
| `ep_size` config field — deserialized from HF `config.json`, never read | `arc-engine/src/deepseek_v4.rs:48` |

That `ep_size` field is the closest thing to expert parallelism in the repo: a
serde default that nothing reads. Exhaustive greps for `expert_parallel`,
`moe_ep`, `all_to_all`, `alltoall`, `AllToAll` return **zero hits workspace-wide.**

What *does* exist is TP sharding of the MoE **intermediate** dimension —
`moe/experts.rs:257,262,267,304,311,320,327`, all
`shard(dim, comm.rank(), comm.world_size())`, no-ops at `world_size=1`. The
**expert count is never divided by world size** anywhere.

### 1.4 🔴 Two landmines found in the LIVE cross-GPU path

**(a) Arc has no NVLink peer-to-peer path.** Every cross-GPU tensor move goes
`Tensor::to_device` → `transfer_to_device`
(`candle-core/src/cuda_backend/mod.rs:1354`) → `dst_stream.clone_dtod(...)` →
`cudarc-0.19.4/src/driver/safe/core.rs:1624` → `memcpy_dtod`. **Neither candle
nor cudarc ever calls `cuCtxEnablePeerAccess`** — grep for
`EnablePeerAccess`/`can_access_peer` returns nothing in either. NVIDIA:
*"If your devices do not support peer-to-peer memory access or if it is not
enabled with cudaDeviceEnablePeerAccess(), the peer-to-peer copies are staged
through the host which entails a performance penalty."*
([CUDA Programming Guide, Multi-GPU Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html))
⇒ **today's pipeline-parallel hidden-state handoff is staged through host RAM,
and any EP built on `to_device` would be too.** Enabling peer access (or
adopting NCCL) is stage 0 of any EP work, and it is also a free speedup for the
pipeline path that exists now.

**(b) The CUDA MoE wrappers validate device *kind*, not *ordinal*.**
`mistralrs-quant/src/qtip/cuda_ops.rs:413-421` takes its stream from
`blocks.device()` and then only checks `matches!(x_2d.device(), Device::Cuda(_))`;
same at `:1785-1793` for the grouped GEMM. Under a multi-GPU map, activations on
`cuda:0` with expert weights on `cuda:1` pass this check and launch on the wrong
device's stream. Unreachable today because the top-level `mapper.map` co-locates
first — but it is exactly the invariant EP breaks. **Not fixed here: `qtip/**`
is fenced while the bake runs.** Carried as stage-0 work in §6; this document is
its record until it lands.

**(c) FIXED IN THIS PR — the ring all-reduce silently computed a wrong sum.**
`distributed/mod.rs` `SumAllReduce::run` performs exactly one exchange (send
right, read left) and `sum_all_reduce` returns `xs + delta`. That is a complete
sum only at `world_size == 2`, yet `RingComm::from_device` accepted **any
power of two ≥ 2**. At world_size 4/8/16 it returned `self + left neighbour`
with no error — wrong numerics, not a failure. Now rejected at construction
with a message naming NCCL as the alternative. This is in scope and topical:
`ring` needs no NCCL and is the first thing someone reading this document
would reach for.

### 1.5 🟡 CONDITIONAL — the NCCL path, and what it would cost to turn on

All rank setup is in `mistralrs-core/src/distributed.rs`.
`prepare_distributed_mapper` (`:224`) has three call sites
(`pipeline/normal.rs:723-724`, `vision.rs:616-617`, `embedding.rs:552-553`),
each guarded `if use_nccl || cfg!(feature = "ring")` — **unreachable in the
production build.** Rank 0 calls `Id::new()` (`:288`), **spawns N-1 copies of
its own executable** (`:293-311`) and waits on `mistralrs_daemon.sock`
(`:314-326`). Request fan-out to daemons (`engine/mod.rs:1060-1087`) is gated
the same way.

Two sharp edges if this is ever enabled:
- `normal.rs:374-380` — when `use_nccl`, the mapper is **overwritten** with
  `DummyNccl` and **auto device mapping is skipped entirely**.
- `attention/mod.rs:232` silently forces `naive_sdpa` when NCCL is on. Turning
  on NCCL today would therefore turn *off* the flash-attention path.
- `is_daemon()` (`distributed.rs:28-36`) reads its env flag whenever
  `cfg!(feature="cuda") && !cfg!(feature="ring")` — **true for the production
  cuda-only build.** Nothing sets the var, but if it leaked in, `lib.rs:775-776`
  parks the process in `loop {}` forever.

Env surface, all in `distributed/mod.rs` and `distributed.rs`:
`MISTRALRS_NO_NCCL`, `MISTRALRS_MN_LOCAL_WORLD_SIZE`,
`MISTRALRS_MN_GLOBAL_WORLD_SIZE`, `MISTRALRS_MN_HEAD_NUM_WORKERS`,
`MISTRALRS_MN_HEAD_PORT`, `MISTRALRS_MN_WORKER_SERVER_ADDR`,
`MISTRALRS_MN_WORKER_ID`, `RING_CONFIG`, `__MISTRALRS_DAEMON_INTERNAL`.
No torchrun-style `RANK`/`WORLD_SIZE` anywhere. There is **no**
`--tensor-parallel`, `--tp-size`, `--world-size` or `--nccl` CLI flag.

### 1.6 🟢 LIVE but unrelated — `--bake-devices` (PR #45)

`mistralrs-cli/src/args/quantize.rs:145-151`, backed by
`mistralrs-quant/src/utils/bake_devices.rs`, consumed in the ISQ pass at
`pipeline/isq.rs:428,461,942,992`. **1.97× on 2 GPUs, byte-identical** [measured].
It spreads *quantization work* across devices and never touches a forward pass.
Two patterns worth reusing verbatim for EP:
- **Work-stealing, not pre-assignment** — a worker takes the next layer when it
  frees up, so a cheap layer does not leave a device idle.
- **A device list on the wrong kind of run is an error, not a silent no-op**
  (`isq.rs:461-467`). EP should refuse just as loudly rather than quietly
  running on one card.

---

## 2. The communication cost, priced

### 2.1 Model constants — one correction to the brief

| quantity | value | source |
|---|---|---|
| MoE layers | **43, not 40** | `first_k_dense_replace=0`, `moe_layer_freq=1` ⇒ every layer is MoE (`deepseek4.rs:30`, `:2694-2697`, asserted `:4576`) |
| hidden_size | 4096 | `deepseek4.rs:4480` |
| experts / top-k / shared | 256 / 6 / 1 | `:4488`, `:4492`, `:4489` |
| moe_intermediate_size | 2048 | `:4487` |

Structural check on `CEILINGS.json`'s derived split: 3 × 4096 × 2048 × 256 × 43
= **277.0e9** routed params ⇒ 1.082e9 per expert across all layers, against
CEILINGS' algebraically-derived 1.084e9. **Independently confirmed.** Bytes per
expert 283 MB, dense 1.697 GB — both stand.

### 2.2 The dispatch/combine model

Under EP a token is sent **once per destination rank** (not once per expert —
this is what DeepEP's dispatch does), and each rank returns **one** locally
summed partial. With 6 experts spread over N ranks:

```
E[remote ranks per token] = (N-1) · (1 − (1 − 1/N)^6)
bytes/token/layer         = 2 · E[remote] · hidden · dtype_bytes
```

| N | E[remote ranks] | bytes/token/layer (BF16 both legs) |
|---|---|---|
| 2 | 0.984 | 16,128 |
| 4 | 2.466 | 40,404 |
| 8 | 3.858 | 63,216 |
| 16 | 4.816 | 78,905 |

### 2.3 Bytes per decode step — **the answer to "does comms eat the win"**

43 MoE layers. Per-GPU **egress** is total ÷ N. [computed]

| B | N | total MB/step | per-GPU egress MB | @450 GB/s | @150 GB/s | @25 GB/s |
|---|---|---|---|---|---|---|
| 64 | 4 | 111.2 | 27.8 | 62 µs | 185 µs | 1,112 µs |
| 64 | 8 | 174.0 | 21.7 | 48 µs | 145 µs | 870 µs |
| **128** | **4** | **222.4** | **55.6** | **124 µs** | **371 µs** | **2,224 µs** |
| **128** | **8** | **347.9** | **43.5** | **97 µs** | **290 µs** | **1,740 µs** |
| 256 | 4 | 444.8 | 111.2 | 247 µs | 741 µs | 4,448 µs |
| 256 | 8 | 695.9 | 87.0 | 193 µs | 580 µs | 3,479 µs |

H200 SXM is NVLink 4.0, **900 GB/s bidirectional per GPU** = 450 GB/s per
direction, NVSwitch full mesh
([NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)). 150 GB/s is a
conservative *achieved* all-to-all figure; DeepEP measures 726/740 GB/s
dispatch/combine NVLink on SM100 EP8
([DeepEP README](https://github.com/deepseek-ai/DeepEP)), so 150 is pessimistic
by a wide margin and the table's middle column is a floor, not a forecast.

**Against 2.15 ms of weight reading at B=128/N=8, 97–290 µs of all-to-all is
4.5%–13% of the step. Bandwidth does not eat the win.**

### 2.4 What *does* cost — 86 synchronising collectives per step

43 layers × (dispatch + combine) = **86 collectives per decode step**. At a
plausible 10 µs each intra-node that is **0.86 ms — three times the bandwidth
term.** This, not bytes, is the engineering problem.

Per-GPU weight reads under pure EP (dense/attention replicated, i.e. attention
data-parallel): [computed]

| B | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| 64 | 58.31 GB / 12.15 ms | 30.00 / 6.25 | 15.85 / 3.30 | 8.77 / 1.83 |
| **128** | **70.71 GB / 14.73 ms** | 36.21 / 7.54 | **18.95 / 3.95** | **10.32 / 2.15** |
| 256 | 74.03 GB / 15.42 ms | 37.86 / 7.89 | 19.78 / 4.12 | 10.74 / 2.24 |

> The brief's 8.83 GB / 1.84 ms / 543 tok/s for N=8 assumed the **dense** term
> shards too. It does not under pure EP — 1.697 GB is replicated on every card.
> Correct figure **10.32 GB / 2.15 ms / 465 tok/s**. The brief was ~17%
> optimistic and otherwise right. Sharding the dense as well (attention TP)
> recovers that 17% but adds an all-reduce per attention block — see §5.

### 2.5 Realistic per-user tok/s at B=128 — **the requested answer**

Step = HBM + a2a bandwidth + 86 × per-collective latency, no overlap. [projected]

| | N=2 | N=4 | N=8 |
|---|---|---|---|
| HBM | 7.54 ms | 3.95 ms | 2.15 ms |
| a2a bandwidth @150 GB/s | 0.30 ms | 0.37 ms | 0.29 ms |
| a2a latency (86 × 10 µs) | 0.86 ms | 0.86 ms | 0.86 ms |
| **total** | **8.70 ms** | **5.18 ms** | **3.30 ms** |
| **per-user tok/s** | **115** | **193** | **303** |
| aggregate tok/s | 14,715 | 24,716 | 38,779 |
| vs 68 on one card | 1.7× | 2.8× | **4.5×** |

Fabric sensitivity at B=128, N=8: [computed]

| fabric | comms | share of step | per-user | vs 1 card |
|---|---|---|---|---|
| NVLink 4, theoretical 450 GB/s, 5 µs/coll | 0.53 ms | 20% | 373 | 5.5× |
| NVLink 4, conservative 150 GB/s, 10 µs/coll | 1.15 ms | 35% | 303 | 4.5× |
| PCIe 5 p2p 50 GB/s, 20 µs/coll | 2.59 ms | 55% | 211 | 3.1× |
| PCIe 5 contended 25 GB/s | 3.46 ms | 62% | 178 | 2.6× |
| InfiniBand 400 Gb/s (46 GB/s, DeepEP-measured) | 3.53 ms | 62% | 176 | 2.6× |

**Verdict, stated plainly: comms does NOT eat the win, but the fabric decides
how much of it survives.** On NVLink the 8-card win is ~4.5–5.5×; on PCIe or
across nodes it collapses to ~2.6×, and comms becomes the majority of the step.
**Expert parallelism for Arc is an intra-node, NVLink-only design.** A
cross-node EP=16 build is not worth attempting before the intra-node one is
measured. The `nvidia-smi topo -m` check is a hard preflight gate on any rental.

### 2.6 The per-user FLOOR, and why N=2 is the interesting rung

Once every expert is woken (B ≳ 256), bytes/step stops growing. The saturated
per-user bandwidth floor by N is: [computed]

| N | GB/card | ms | **per-user tok/s at ANY batch** |
|---|---|---|---|
| 1 | 74.20 | 15.46 | **65** |
| **2** | **37.95** | **7.91** | **126** |
| 4 | 19.82 | 4.13 | 242 |
| 8 | 10.76 | 2.24 | 446 |
| 16 | 6.23 | 1.30 | 771 |

**One card can never serve 100 tok/s/user past B≈43** (10 ms budget = 48 GB =
163.5 experts = B=42; aggregate 4,200 tok/s). **Two cards clear it at any
batch.** That is the cheapest possible answer to Jish's target and it should be
the first thing built.

### 2.7 The FLOP guardrail — the assumption most likely to break this

`CEILINGS.json` models bandwidth only. Per-card compute under EP-8 is
13e9 × 2 × B / 8 FLOP: [computed]

| effective TFLOPS | B=128 | B=256 | B=512 |
|---|---|---|---|
| 700 (near BF16 peak) | 0.59 ms | 1.19 ms | 2.38 ms |
| 400 | 1.04 ms | 2.08 ms | 4.16 ms |
| **100** | **4.16 ms** | **8.32 ms** | 16.64 ms |

At 400+ TFLOPS the step stays bandwidth-bound and §2.5 holds. **At 100 TFLOPS
the EP-8 step is compute-bound and 465 tok/s is unreachable no matter how good
the comms are.** Arc has no measured effective-TFLOPS number for the grouped
trellis kernel — FACTS' "15% of peak" and "~22%" figures are bandwidth-side and
microbench-side respectively and neither answers this. **Measuring the grouped
kernel's achieved FLOPS is a cheap, unclaimed prerequisite for sizing EP, and it
can be done on the single card we already rent.**

Related, and worth correcting upstream: `CEILINGS.json`'s
`PAST_SATURATION` row `B_699 = 45,400 tok/s` ignores the FLOP bound entirely.
Single card, bandwidth pins at 15.42 ms while compute grows as B; the two cross
at **B ≈ 415** even at a generous 700 TFLOPS. At B=699 compute is 25.9 ms, so
the aggregate ceiling is **~27,000, not 45,400**. The headline 16,600 at B=256
is unaffected (compute 9.5 ms < 15.4 ms bandwidth) and stands.

---

## 3. Load imbalance

### 3.1 What the reference implementations do — replicate AND rebalance

SGLang, vLLM and TensorRT-LLM independently converged on **DeepSeek's EPLB
algorithm**: greedily replicate hot experts, then LPT bin-pack physical experts
onto GPUs, hierarchically so replicas stay node-local.

- SGLang's copy is verbatim: *"This file is copied from
  https://github.com/deepseek-ai/EPLB/blob/main/eplb.py"* —
  `sglang/python/sglang/srt/eplb/eplb_algorithms/deepseek.py:1`.
  `replicate_experts()` at `:55-83` repeatedly picks `argmax(weight / logcnt)`;
  `balanced_packing()` at `:7-52`; `rebalance_experts_hierarchical()` at
  `:86-168`. Budget flag `--ep-num-redundant-experts`, **default 0**
  (`server_args.py:609`). Rebalance every **1000 iterations**
  (`eplb_manager.py:48-53`, `server_args.py:614`), chunked by layer so it does
  not stall a forward (`eplb_manager.py:80-87`); weights move by NCCL
  isend/irecv (`expert_location_updater.py:227-311`).
- vLLM reimplements the same three primitives
  (`vllm/distributed/eplb/policy/default.py:23,76,104,275`), rebalances every
  **3000 steps** (`config/parallel.py:60`), and admits the cost:
  *"EPLB uses redundant experts that need to fit in GPU memory... For
  DeepSeekV3, this is approximately 2.4 GB for one redundant expert per EP
  rank"* (`docs/serving/expert_parallel_deployment.md:187`).
- TensorRT-LLM "Wide-EP" publishes **the only measured imbalance number in the
  tree**: DeepSeek-R1 at EP=32, mean 1024 tokens/rank, **average imbalance
  ratio 1.564** — the hottest rank receives 1.56× the mean — with per-layer
  worst case 2.46 at layer 58
  (`tensorrt_llm/examples/wide_ep/ep_load_balancer/README.md:49-70`).
  Recommended budget `num_slots = total_experts + EP_size` (288 at EP=32).
- **DeepSpeed is the outlier and simply accepts imbalance**, dropping tokens
  over a capacity factor (`deepspeed/moe/sharded_moe.py:162-171,189,294`).
  That is the training-era design point and is not what anyone serves.

Also worth stealing later: SGLang's **DeepEP Waterfill** dispatches the *shared*
expert as an extra routed expert to the least-loaded rank
(`sglang/python/sglang/srt/layers/moe/deepep_waterfill.py:14`,
`server_args.py:5846-5857`, "Supported on DeepSeek-V3/R1 with EP >= 2"). V4 has
exactly one shared expert per layer traversed by every token, so this applies
directly.

### 3.2 Why Arc is structurally different — and can skip EPLB in stage 1

**Arc's dominant per-step cost is weight bytes, and a GPU cannot read more
experts than it owns.** At high coverage that puts a *hard bound* on
weight-read imbalance that no routing distribution can exceed: [computed]

| B | expert coverage | max weight-read imbalance N=2 / N=4 / N=8 |
|---|---|---|
| 16 | 31.6% | +87% / +167% / +136% |
| 32 | 53.2% | +81% / +75% / +65% |
| 64 | 78.1% | +27% / +25% / +23% |
| **128** | **95.2%** | **+4.8% / +4.6% / +4.2%** |
| **256** | **99.8%** | **+0.2% / +0.2% / +0.2%** |

This is not an estimate with an assumed distribution — it is `min(experts
owned, E(B))` against the balanced `E(B)/N`. **At the batch sizes where EP is
worth doing, routing skew cannot make the weight-read term more than ~4% worse.**
At B ≤ 64 it very much can, but at B ≤ 64 a single card is nearer the target
anyway.

Skew still hits two smaller terms: the per-card GEMM work and the hot rank's
a2a receive volume. Applying TRT-LLM's **measured 1.564×** to both: [projected]

| kernel efficiency | B=128 balanced → skewed | B=256 balanced → skewed |
|---|---|---|
| 400 TFLOPS | 303 → 289 tok/s (**5% worse**) | 272 → 199 (37% worse) |
| 100 TFLOPS | 188 → 128 (47% worse) | 102 → 68 (51% worse) |

**Conclusion: EPLB's value for Arc is a function of kernel efficiency, not of
routing.** While the step is bandwidth-bound at B=128, skew costs ~5% and EPLB
is not worth a rebalancing subsystem plus redundant-expert memory. If the
grouped kernel stays slow, or once B ≥ 256 pushes the step compute-bound, skew
costs ~40-50% and EPLB becomes the next lever. **Ship without it; instrument
for it.** The cheap instrumentation is SGLang's "balancedness" metric —
mean/max tokens per rank, `expert_distribution.py:1036-1046` — which is one
counter per rank per layer and tells us whether we ever need the machinery.

One V4-specific wrinkle: **layers 0, 1 and 2 do not route by score at all.**
They use TD-MoE hash routing — `gate.tid2eid`, an I64 `[vocab=129280, top_k=6]`
table loaded iff `layer_idx < num_hash_layers` (`deepseek4.rs:1750, 1766-1775`),
forward at `:1854-1877`. Their expert assignment is a **deterministic function
of token id**, so their load distribution is a fixed property of the tokenizer
and the corpus and can be computed offline, exactly, with no GPU. If those three
layers turn out badly skewed, they can be balanced statically at bake time by
permuting expert→GPU assignment. **This is free to check and nobody has.**

---

## 4. Interaction with what we already have

### 4.1 The `qtip2b` grouped kernel — confirmed a natural fit, with one caveat

**Confirmed: the grouping is by expert, on device.**
`mistralrs-quant/kernels/qtip/qtip_grouped_gemm.cu:3-5` — *"tokens are sorted
by expert ON-DEVICE"*; pipeline at `:33-46` (per-expert histogram → exclusive
scan → ragged per-expert tile map → grouped scatter); the GEMM reads
`tile_expert[mt]` and `offsets[expert]`/`offsets[expert+1]` at `:349-352` and
indexes weights by `expert` at `:296, :372`. Rust entry
`grouped_gemm_2b_cuda` at `qtip/cuda_ops.rs:1722`.

**Why this is the right substrate for EP:** expert sharding is a slice of the
leading dimension of the expert-stacked weight tensors. Weights are already
stored that way — `QtipLayer` carries `blocks: [E, N, packed_per_row]`,
`row_scales: [E, N]`, plus a **shared** `lut` and **shared** `rotation_signs`
(`qtip/mod.rs:700-755`). Sharding is `E → E/N` on two tensors with the shared
tensors replicated. The kernel's per-expert offset table then simply describes
32 local experts instead of 256. **No kernel change is required to run
expert-parallel** — only a remapping of global expert id → (rank, local id),
which is what the routing kernel `launch_qtip2b_moe_route`
(`cuda_ops.rs:1858-1874`) already computes offsets over.

**Caveat, and it is the one that decides the schedule:** the grouped kernel is
on the **`qtip2b` rung only**. The rung V4 is baked and served on today
(`qtip2` → `QtipLayer`) has no grouped kernel at all
(`qtip/gather_policy.rs:8-12`, `qtip/mod.rs:3534-3537`). V4's live decode path
is `MoEExpertsBackend::Fast` (`moe/experts.rs:84-112`, chosen because V4 carries
a `quantization_config` at `deepseek4.rs:4501`) → `gather_forward_cuda_ondevice`
(`qtip/mod.rs:2834-2884`) → `gather_gemv_cuda`, which launches **one independent
trellis GEMV per (token, expert) pair** mapped to `grid.y`
(`cuda_ops.rs:360`, pair guard `:409`). **No grouping, no expert reuse.**
That is the "flat, does not scale" path FACTS' crossover table measures at
315 → 317 tok/s from B=64 → B=128.

⇒ **EP and the `qtip2b` re-bake are the same decision.** Building EP on the
GEMV path would shard 86 collectives onto a path that already refuses to
amortize. The bake currently running is what makes EP worth building.

Structural limit to respect: `GATHER_GEMV_MAX_PAIRS = 65535`
(`qtip/mod.rs:119-126`, enforced `cuda_ops.rs:409`) — over the limit the launch
**silently returns zeros**. EP *helps* here: pairs per card fall by N.

### 4.2 `FusedExperts` takes no `comm` at all

`FusedExperts::new` (`mistralrs-quant/src/distributed/layers.rs:1598`) has **no
`comm` parameter** — call sites `moe/experts.rs:352, :378`. `PackedExperts::new`
(`layers.rs:1031`) does take one but **bails** if `world_size != 1` with a quant
config (`:1045-1050`). So the expert-holding types have no notion of a world.
This is the single largest concrete API change EP requires, and it is a
constructor signature plus a leading-dim slice — not a rewrite.

`Moe::new` is already device-map aware (`deepseek4.rs:2034-2037, 2051, 2063,
2081-2086`; `moe/experts.rs:195` pins `vb.pp("experts").set_device(...)`), so the
loading side already knows which device a layer's experts belong on. **The
forward is not** — `Moe::forward` (`deepseek4.rs:2095`) and
`MoEExperts::forward` (`moe/experts.rs:452`) take no mapper and assume
co-location.

### 4.3 Hot-path debris to clean before adding collectives

`deepseek4.rs:1804, 1980, 1991, 2004, 2101, 2128` are live env-gated
diagnostics (`ARC_COLLAPSE`, `ARC_SOFTMAX_ROUTE`, `ARC_ROUTE_TOP1`,
`ARC_CAPTURE_MOE_INPUT`) inside the gate/MoE forward, **including a per-call
`to_vec2` device-to-host read at `:2131`**. A D2H sync inside the routing path
is incompatible with both CUDA-graph capture and any overlapped all-to-all.

---

## 5. The honest alternative — EP vs TP vs plain replication

**V4 fits on one H200 (74.19 GB in 141 GB).** That is the fact that makes this
question different from the reference systems', which need 8+ cards just to
*hold* DeepSeek-V3 at FP8. Arc is never forced to shard. So plain replication is
a real competitor and has to be beaten, not assumed away.

### 5.1 Tensor parallelism

Sharding every matrix needs an all-reduce **after every** attention block and
every MLP — 2 per layer, 86 per step for V4, the same collective count as EP,
but each all-reduce carries the **full** `[B, hidden]` activation from **every**
rank rather than a routed subset. And for V4 it is actively wrong-shaped:
V4's K/V is a **single fused MQA head** (`wkv`, `hidden=4096 → head_dim=512`,
broadcast across all 64 Q heads — `deepseek4.rs:12-19`). TP=N over one KV head
**duplicates the KV cache N times** or does not shard at all. This is exactly
why SGLang pairs DP attention with EP MoE for MLA models:
*"DPA addresses these limitations by applying data parallelism specifically to
the attention component"* (`sglang/docs/advanced_features/dp_dpa_smg_guide.md:29-51`,
motivation at `:19-25`).

Given `CEILINGS.json`'s `KV_AND_XS_LADDER` says **memory for context, not
bandwidth, is what sets max B**, a topology that multiplies the KV cache is
disqualified on Arc's actual binding constraint.

TP does have one advantage worth keeping: it shards the 1.697 GB dense term,
worth ~17% of the N=8 step (§2.4). That is a stage-3 refinement, not a
foundation.

### 5.2 Plain replication (data parallel), and why EP beats it

N independent replicas, no comms at all, each serving U/N users. Priced against
EP-N with full comms from §2.5, at $4.85/GPU-hr: [computed]

| U | N | DP per-user | DP agg | DP $/Mtok | **EP per-user** | EP agg | **EP $/Mtok** |
|---|---|---|---|---|---|---|---|
| 128 | 2 | 82 | 10,538 | $0.256 | **115** | 14,715 | **$0.183** |
| 128 | 4 | 119 | 15,263 | $0.353 | **193** | 24,716 | **$0.218** |
| 128 | 8 | 195 | 24,986 | $0.431 | **303** | 38,779 | **$0.278** |
| 256 | 2 | 68 | 17,377 | $0.155 | **107** | 27,409 | **$0.098** |
| 256 | 8 | 119 | 30,526 | $0.353 | **272** | 69,620 | **$0.155** |
| 512 | 8 | 82 | 42,150 | $0.256 | **235** | 120,150 | **$0.090** |

**EP wins on per-user latency AND on $/Mtok at every point checked, comms
included.** The mechanism is simple and worth stating because it is the whole
argument: a DP replica at B=16 still reads 80 distinct experts to serve 16
users — terrible amortization. EP pools all 128 users against **one** copy of
the expert set, so the same expert byte serves 8× more tokens.

### 5.3 Verdict

**Expert parallel with data-parallel attention.** Same topology the entire
industry converged on for DeepSeek-class models — vLLM's recommended 8×H200
DeepSeek-V3 line is literally `--tensor-parallel-size 1 --data-parallel-size 8
--enable-expert-parallel` (`vllm/docs/serving/expert_parallel_deployment.md:70-78`);
SGLang's is `--tp 8 --dp-size 8 --ep 8 --enable-dp-attention --moe-a2a-backend
deepep` (`sglang/docs/advanced_features/dp_dpa_smg_guide.md:68-79`).

Two honest caveats against over-reading that consensus:
- SGLang's **recommended** single-node V3.2 config is TP+DP **without** `--ep`;
  EP+DP is listed as an alternative (`sglang/docs/basic_usage/deepseek_v32.md:43-63`),
  with *"DP Attention is better for large concurrency"* / *"TP attention is
  better for low latency"*. EP is a large-concurrency play, which is exactly
  Arc's fleet thesis but is not a universal win.
- Both engines force **`ep_size == tp_size`** once a real a2a backend is chosen
  (`sglang/python/sglang/srt/server_args.py:3229-3236`;
  `sglang/docs/advanced_features/expert_parallelism.md:25`). Hybrid EP<TP is
  only supported on the all-gather fallback. Arc should not invent a more
  general topology than the reference maintains.

One design constraint to inherit from day one: SGLang's a2a mode is
`auto` → **`low_latency` for decode, `normal` for prefill**
(`server_args.py:5736-5741`, resolved at `layers/moe/utils.py:151-158`), and
**`normal` mode force-disables CUDA graphs** (`server_args.py:3229-3232`) while
low-latency is graph-compatible
(`sglang/docs/advanced_features/expert_parallelism.md:23`). Given Arc's
CUDA-graph work, **the decode path must be the graph-compatible one from the
start.** Also: DeepEP low-latency assumes decode batch ≤ 256 per rank
(`token_dispatcher/deepep.py:621-624`) — comfortably inside Arc's range.

---

## 6. Staged plan

Ordering principle, stated bluntly: **EP multiplies the single card. At B=128
Arc measures 0.27 tok/s/user against a 68 ceiling [measured, wave34-BL]. EP-8
turns 0.27 into ~2.2.** The 300× is implementation and is already in flight
(the `qtip2b` bake, the grouped kernel, the host decode loop). **EP is stage 3
of the throughput programme, not stage 1 — but it is unconditionally required,
because no amount of single-card work gets past 68 tok/s/user at B=128.**

**Stage 0 — free, no GPU, no EP.** (a) Enable CUDA peer access, or route
cross-device moves through a real p2p path, so the *existing* pipeline-parallel
mapper stops staging through host RAM (§1.4a). (b) Fix the ordinal check in the
MoE CUDA wrappers (§1.4b) — after the bake releases `qtip/**`. (c) Compute the
layer-0/1/2 `tid2eid` load distribution offline (§3.2) — pure table arithmetic.
(d) Add the mean/max tokens-per-would-be-rank counter so we learn whether EPLB
is ever needed. *Buys: no throughput. Removes three ways stage 1 could fail for
reasons unrelated to EP.*

**Stage 1 — EP=2, intra-node, correctness first.** Shard `E → E/2` on the
expert-stacked `blocks`/`row_scales`; replicate `lut`, `rotation_signs`, the
shared expert and all dense weights; attention stays data-parallel. Dispatch and
combine as explicit p2p exchanges — no collective library, no EPLB, no overlap.
Gate: **bit-comparable logits against the single-card run**, then the wave34-BL
sweep. *Buys: the per-user floor moves 65 → 126 [computed] — the first
configuration that clears 100 tok/s/user at any batch. Validates the entire
sharding contract at the N where the all-to-all is a single pairwise exchange
and imbalance is bounded at +4.8%.*

**Stage 2 — EP=4 and EP=8 on the same code path.** Only the expert→rank map and
the number of peers change. Gate: per-user ≥ 190 at N=4 and ≥ 300 at N=8, B=128.
*Buys: 193 and 303 tok/s/user [projected]. If the measured numbers land far
under, §2.7's FLOP guardrail is the first suspect and the answer is kernel work,
not more cards.*

**Stage 3 — overlap and fusion.** Split dispatch/combine into launch/wait halves
the way SGLang does (`deepep.py:351-369`) so the shared expert and the attention
of the next microbatch fill the gap; keep the decode path CUDA-graph capturable.
*Buys: up to the 0.86 ms latency term — 3.30 → ~2.5 ms at N=8, ~303 → ~400
tok/s/user. SGLang claims "up to 2x" for two-batch overlap
(`expert_parallelism.md:133`); do not bank that until measured.*

**Stage 4 — only if instrumentation says so: EPLB, dense TP, cross-node.**
Redundant experts (~283 MB each for V4, an order of magnitude cheaper than
DeepSeek-V3's 2.4 GB); TP on the dense term for the last 17%; cross-node last,
because §2.5 says the fabric halves the win.

---

## 7. Hardware to validate, and what it costs

**Nothing here needs hardware until stage 1, and stage 0 needs none at all.**

**Stage 1 validation — 2 × H200 on one NVLink/NVSwitch node.**
- Hard preflight: `nvidia-smi topo -m` must show `NV#` between the two cards.
  A PCIe-only pair invalidates the run (§2.5) — check before starting the clock.
- Work: pull the published 74 GB UQFF, load EP=2, parity-check logits against
  the single-card artifact, then run the wave34-BL sweep B ∈ {1,8,16,32,64,128,256}.
- Time: wave34-BL's full single-card sweep cost **$7.60** at 1×H200 [measured].
  Two cards at $4.85/hr = $9.70/hr; allow 1.5–2 h including download and the
  parity gate.
- **Cost: $15–20.** Against the $49.97 remaining budget, this is the single
  highest-information purchase available — it either confirms a 1.7× per-user
  win and the whole sharding contract, or it refutes the model cheaply.

**Stage 2 validation — 8 × H200 HGX, ~1 h, ≈ $39.** Do not buy this before
stage 1 returns. If stage 1's measured step time exceeds the §2.5 projection by
more than ~30%, the cause is per-collective latency or kernel FLOPS, and both
are diagnosable on the 2-card box for a quarter of the money.

**Free and unclaimed, no GPU:** the `tid2eid` load distribution (§3.2), the
`nvidia-smi topo -m` preflight script, and the offline expert→rank permutation.

---

## 8. What would refute this

- **`nvidia-smi topo -m` shows PCIe, not NVLink, on the rented pair.** Then §2.5
  row 4 applies, the 8-card win is 2.6× not 4.5×, and EP is worth much less.
- **Per-collective latency lands above ~40 µs.** 86 × 40 µs = 3.4 ms would
  exceed the entire N=8 HBM term and make stage 2 pointless without stage 3's
  overlap first. This is the single most uncertain number in the document:
  DeepEP publishes 163 µs dispatch / 318 µs combine per layer, but those are
  **RDMA over InfiniBand at 128 tokens/rank, hidden 7168, top-8**
  ([DeepEP](https://github.com/deepseek-ai/DeepEP)) — not intra-node NVLink,
  and DeepSeek overlaps them across two microbatches precisely because
  43 × 481 µs would be 20 ms/step. **No intra-node NVLink low-latency figure
  was found in any vendored reference or public doc.** Stage 1 must measure it.
- **The grouped kernel's effective FLOPS is nearer 100 than 400 TFLOPS.**
  Then §2.7 says the EP-8 step is compute-bound at B=128 and the ceiling is
  ~188 not ~465, and skew starts costing ~47%. Measurable on one card.
- **Real routing is far more concentrated than uniform.** This biases
  *favourably* for bytes (fewer experts woken ⇒ CEILINGS' table is a lower
  bound on the ceiling) and *unfavourably* for skew. Both directions are
  unmeasured; the coverage curve has never been measured on this model.

---

## 9. Changes in this PR

- `mistralrs-quant/src/distributed/mod.rs` — `RingComm::from_device` now
  rejects `world_size > 2` instead of silently computing a partial sum (§1.4c).
  In scope (`distributed/**`), verified with `cargo check -p mistralrs-quant
  --features ring`.
- This document.

Deliberately **not** changed: `qtip/**` (bake fenced), MTP (agent BT),
KV/`xs`/FP8 storage (agent BU). The ordinal check of §1.4b lives in
`qtip/cuda_ops.rs` and is left to whoever picks up stage 0.
