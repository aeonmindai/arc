# Hardware lessons: running this on rented GPUs

Arc's quantization and benchmarking work runs on rented cloud GPUs, billed by the
hour, deleted when the work finishes. This document records what that costs and
what goes wrong, so the next person does not pay to rediscover it.

Every item here was found the expensive way. Evidence grades are the same as in
[QUANTIZATION_PERFORMANCE.md](QUANTIZATION_PERFORMANCE.md): **[measured]**,
**[derived]**, **[source-verified]**, **[projected]**, **[published]**.

## The cost model

| card | price | what it was used for |
|---|---|---|
| H200 141 GB | $4.92/hr | the reference bakes, quality runs, fit proof |
| A100 / L40 48 GB | $0.97–1.49/hr | bakes, after the memory fix below |
| A30 24 GB | $0.39/hr | kernel A/B experiments — ratios only, never absolutes |

Cumulative spend across the first four measurement sessions was **≈$123**
[measured], all on H200 at $4.92/hr. Two later kernel questions that would each
have cost $10–20 of H200 time were answered on an A30 for **$0.39** and **$0.07**
respectively, because the questions were about *ratios* (kernel A vs kernel B,
byte-identity, occupancy) and ratios transfer between cards while absolute times
do not.

**Rule that follows: decide, before renting, whether the question is a ratio or
an absolute.** Ratio questions belong on the cheapest card that runs the kernel.

### Re-entry arithmetic (why "just delete it" is sometimes wrong)

An idle box at $4.92/hr costs **$0.082/min**. Getting back to a working state
from scratch costs boot ~10 min + a 149 GB model pull ~35 min + build ~25 min ≈
**70 min ≈ $5.74** before any work happens [measured]. So the standing rule
"never debug on a paid box" means *never sit and think on one* — not *throw away
70 minutes of paid setup to fix a typo*. Below roughly 70 minutes of fix work,
holding beats re-entering; after the artifact has been uploaded the re-entry cost
drops sharply and the threshold moves with it.

In one session, **four aborts cost ~$0.30 total** [measured], because each fix
took minutes and the box was held rather than destroyed.

## 1. Check the driver against the toolkit *before* you build

A rented box came with an image carrying **CUDA toolkit 13.1** on a driver
(580.173) whose maximum supported runtime is **CUDA 13.0**. Everything built
cleanly. The failure appeared only when the first kernel launched:
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` — after a 25-minute bootstrap that included
a 149 GB model download [measured]. Recovery meant installing the 13.0 toolkit,
cleaning the kernel crates, and rebuilding everything.

**Gate:** compare `nvidia-smi`'s reported CUDA version against `nvcc --version`
as step zero, before the download and before the build. It costs one command.

## 2. The bad-box lottery is real, and it looks like a code regression

The same binary and the same weights ran **~6× slower on one rental than on
another** [measured]. A bisect against an older binary on the *slow* box showed
that binary was equally slow there — so the slowdown was **environmental, not
ours**. Hours were spent chasing a regression that did not exist.

The signature of a starved box: **GPU at 99 % utilization but drawing only
~132 W of 700 W**, clocks at maximum, temperatures fine. High utilization with
low power means the kernel is resident but stalled — for a transfer-heavy
workload, starved on host↔device traffic.

**Gate:** after boot, run a short bake and check *both* layers/minute and
`nvidia-smi power.draw` under load. A card drawing <200 W of 700 W under a
sustained bake is a bad rental: tear it down, rent another, do not debug it.

### But low power is not *by itself* proof of a bad box

The healthy H200 beam bake draws **261 W of 700 W** [measured] — 37 % of TDP,
against a **37.5 %** measured occupancy. That kernel is latency-bound by design;
its power draw tracks its occupancy. Read the power number together with what the
kernel is supposed to be doing:

| observation | reading |
|---|---|
| 99 % util, 132 W / 700 W, transfer-heavy kernel | starved box — re-rent |
| 100 % util, 261 W / 700 W, latency-bound kernel at 37.5 % occupancy | healthy, working as designed |
| 100 % util, 154–170 W against a 165 W limit, 1395–1440 MHz of 1440 max, 51 °C | healthy (A30 under sustained load) |

## 3. Health gates must sample *under confirmed load*

Three separate gates fired false alarms in one program, all with the same root
cause — **the gate sampled before the thing it was measuring had ramped**:

1. A power gate sampled 35 s after *launching* a burn, before the burn ramped;
   it read **80 W / 0 % util** and declared the box unhealthy. The same box
   measured **437 W at 100 % util at 1980 MHz** seconds later [measured].
2. A pace gate required ≥2 layer markers at t=3 min and killed a bake whose GPU
   was at 100 % and whose memory was climbing normally. Layer 1 carries one-time
   setup, so a 3-minute sample measures startup, not pace. The t=10 min gate on
   the same run was the honest one and was left armed.
3. The same class of error produced a published bake-rate number that was wrong
   — see [TESTING_DISCIPLINE.md](TESTING_DISCIPLINE.md#antidote-4-measure-rates-by-differencing-consecutive-markers).

**Rule:** sample in a loop while *confirming* the load is present (utilization
above a threshold), never as a single early sample; and never let a gate's window
include one-time setup.

## 4. `cudnn` costs 62 % of decode throughput on this workload

Single-stream decode, same model, same box, two builds [measured]:

| build | b=1 decode |
|---|---|
| without `cudnn` | **14.58 tok/s** |
| with `cudnn` | 5.45 tok/s |

**−62 %.** This is a negative result worth as much as a positive one: the
"obviously beneficial" optional dependency is a large regression here. Arc builds
without `cudnn` and the project's build documentation says so.

(b=1 decode is a kernel diagnostic, not a headline metric — production batches are
32–128 — but the A/B is valid: same kernels, same weights, one feature flag.)

## 5. ISQ thread policy: 24 host threads thrashing one GPU

A bake logged `Applying immediate ISQ in parallel on 24 threads` with the GPU at
99 %, and ran at **4–9 minutes per layer with no warning of any kind** — 24 CPU
threads submitting quantization work to a single device, contending [measured].
A fast bake on the same code printed `1 threads`.

The immediate fix was to force single-threaded ISQ submission
(`MISTRALRS_ISQ_SINGLETHREAD=1`, later a hardcoded policy returning 1 for QTIP on
a GPU device — `mistralrs-quant/src/lib.rs`, `isq_thread_policy`).

**The interesting part is what happened next.** Once the search kernel got much
cheaper, the hypothesis appeared that the single-thread safeguard had become the
limiter — that it was strangling the host-side INT4→BF16 expert unpack, with
"3–5× remaining headroom" [projected]. Two things killed that:

1. **A call-graph trace showed the unpack was never inside the ISQ pool.** The
   pool is only ever `spawn`ed onto, never `install`ed, so the caller is not in
   it; and layer construction runs sequentially on the loader thread. The unpack
   was already using rayon's global pool at full width. [source-verified]
2. **Direct instrumentation measured the unpack at 2.5 s of a 241 s layer — 1.0 %.**
   [measured]

The projected 3–5× was worth 1.01×. What did ship is a **separation of concerns**
that is correct under either answer: a dedicated, process-wide unpack pool
(`ARC_UNPACK_THREADS`) so that *host unpack width* is independent of *GPU
submission width*, plus two per-layer timing log lines that split each layer into
its host half and its device half. The wall-time gain was zero, and the code says
so.

**Rule:** two different resources were being controlled by one knob. Name them
separately even when the fix is a no-op today, and instrument the split before
projecting a speedup from either.

## 6. The bake OOM — the single most expensive defect

A 43-layer bake died at **layer 28** with `CUDA_ERROR_OUT_OF_MEMORY` on a 140 GB
H200, two hours and ~$10 in, leaving a **4 KB** output directory [measured].
Until it was fixed no artifact could be produced — by us or by any customer
quantizing this model.

### Root cause, in two parts

**Primary — retention.** The fused-expert constructor picks the mapped CUDA
device as the quantization target and passes it to the ISQ apply. That makes the
"move the result back to host" condition false, so **every quantized expert stack
stays resident on the GPU**. For this model that is `3 × 537 MB = 1.61 GB/layer`,
**69 GB over 43 layers** — exactly the size of the 68 GB artifact, and exactly the
measured 1.7–1.9 GB/layer baseline growth [measured + derived]. Correct behaviour
for a *serve*; pure cost for a *bake*, which constructs the model only so it can
be serialized.

**Accelerant — a 4.3 GiB transient nobody wrote.** The per-chunk loop does
`weight.narrow(0, i, 16)?.reshape(...)`, which is a **view**: the layout names 16
experts, the storage still holds all 256. Candle's `to_device` copies the
**entire backing storage** and clones the layout offset
[source-verified, `candle-core/src/tensor.rs:2368` in the pinned fork]. So each
chunk uploaded **4.295 GiB where 268 MiB was needed** — 48 allocate/free cycles of
a 4.3 GiB block per layer, plus 48 cycles of the ~6 GiB search scratch, churning
around a resident set growing in 537 MB steps. That fragmentation is what broke
the growth slope from 1.71 to **4.45 GB/layer between layers 22 and 24**
[measured]; extend that slope and you reach 140 GB at layer 28.

The search kernel was **not** the cause: the beam allocates one ~6 GiB trace, the
exhaustive DP allocates 6.06 GiB plus 2 × 189 MB. One block versus three, same
magnitude — not an 80 GB difference.

### The fix

| change | effect |
|---|---|
| bake-to-host switch | during a UQFF bake the quantize still runs on the accelerator, but the packed result is materialized on the **host**, reusing the move-back path that already existed |
| contiguous chunk materialization | `force_contiguous()` on partial chunks, so `to_device` ships the chunk (268 MiB) rather than the whole stack (4.295 GiB) |
| VRAM budget guard | samples device usage once per fused MoE layer and bails when `used + remaining × growth` would exceed the card, printing every term; slope is a trailing 4-layer mean and it stops only after **5 consecutive** over-budget projections, so a one-off allocator jump cannot trip it |
| streaming shard writer | writes ≤10 GB shards as it goes instead of buffering the whole 68 GB artifact in host RAM before splitting |
| PagedAttention off for `quantize` | it sizes its KV cache from *free* VRAM — which this fix makes large — for a command that never emits a token |

**Result: a flat ~10.5 GiB peak across all 43 layers** [derived from the per-layer
working set: 0.27 chunk BF16 + 0.54 F32 + 0.54 rotated + 6.00 search trace + 0.04
packed, plus ~3 GiB resident]. That is a 13× margin on a 141 GB H200, an 8×
margin on an 80 GB H100, and it is what makes a **$0.97–1.49/hr card able to do
the job that previously needed a $4.92/hr one** — a 3–5× reduction in the cost of
every future bake.

### Two honest limits, both pinned as tests

- **The budget guard would not have saved that run.** The steady 1.71 GiB/layer
  stretch through layer 22 genuinely projects to ~85–95 GB on a 140 GB card — it
  fits. The run died to a late nonlinearity no early projection could see. The
  retention fix is what fixes it; the guard is a seatbelt for the general case.
  A test named for exactly this records it so nobody later assumes otherwise.
- **Making a load-time OOM cost only the tail is not feasible without a format
  change**, because UQFF tensor names are positional indices into the model's
  layer list, which does not exist until the model is fully constructed. Flagged
  rather than forced.

### Related: the artifact used to be written only at the end

Before the streaming shard writer, the output directory stayed at ~1 MB until the
final serialize — a bake killed at layer 40 of 43 lost **everything**, with no
partial resume. Price that into any decision to interrupt a long run, and check
whether the tool you are running buffers or streams before you rely on being able
to stop it.

## 7. Most session failures are in the scaffolding, not the engine

In the session with the most aborts, **four of four aborts were in the
session-driver script written that same day; zero were in Arc** [measured]. The
driver had an 11-scenario, 147-assertion dry-run harness, and it passed all of
them, because the harness asserted that the right *banner* was printed rather
than that the right *action* was taken.

The response was to delete the driver and run the session as a sequence of direct
remote commands. Concrete recurring hazards, all cheap to avoid:

- Killing processes **by PID, never by pattern** — a pattern-based `pkill`/`pgrep`
  matches the invoking command line and kills the session issuing it.
- Remote-exec transports may drop long-running commands; poll on a short interval
  instead of sleeping inside a remote command.
- Heredocs do not reliably survive remote-exec; upload scripts as files.
- If your harvest step destroys credentials (it should), every resume must
  re-supply them — and the step that needs them must **fail loudly** when they
  are absent rather than skipping.
- A parity gate that runs *every* test matching a broad pattern will abort on an
  unrelated pre-existing failure in a subsystem the gate is not about. Narrow the
  filter to the tests the gate is actually asserting.

## Checklist

Before renting:

- [ ] Is the question a **ratio** or an **absolute**? Ratios go on the cheapest card.
- [ ] Is the peak memory known, and does it fit a cheaper card?

First five minutes on the box:

- [ ] `nvidia-smi` CUDA version vs `nvcc --version` — before download, before build.
- [ ] Compute capability meets the kernels' requirement (and the tests **fail**,
      not skip, if it does not — see [TESTING_DISCIPLINE.md](TESTING_DISCIPLINE.md)).
- [ ] Health burn: sample power **in a loop, under confirmed load**. Below ~30 %
      of TDP for a throughput-bound kernel ⇒ re-rent.
- [ ] Build without `cudnn`.
- [ ] Check the thread-count line in the first 30 s of a bake.

During:

- [ ] Measure rates by differencing consecutive markers, never a running average.
- [ ] Watch device memory growth per layer, not just the final number.

On failure:

- [ ] Harvest artifacts **before** deleting; a deleted box with no artifacts makes
      the next attempt blind. The tarball is the deliverable of a failed session.
- [ ] Fix on a free machine, then return. Hold the box only if the fix is shorter
      than re-entry, and put a hard idle timeout on the hold.

## Provenance

Internal agent logs: `wave14-AJ-session6.md` (driver/toolkit mismatch, session
aborts), `wave15-AM-unpack.md` (unpack call-graph trace and instrumentation),
`wave16-AF-beam-perf.md` (occupancy and power correlation),
`wave19-AP-gmin-exhaustive.md` (A30 health telemetry), plus the standing
`FACTS.md` hardware ledger.

Pull requests: **#20**/**#25** (bake thread policy and the box health gate),
**#39** (unpack pool separation + per-layer timing), **#41** (the OOM fix:
bake-to-host, contiguous chunks, streaming shard writer, VRAM budget guard).

In-tree source: `mistralrs-quant/src/lib.rs` (`isq_thread_policy`,
`expert_unpack_threads`), `mistralrs-quant/src/utils/bake_budget.rs`,
`mistralrs-quant/src/qtip/bake_memory_tests.rs`, and `docs/CONFIGURATION.md` for
the environment variables named here.
