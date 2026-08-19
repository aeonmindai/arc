> ## 🔴 STATUS: the heap fix in this PR **DOES NOT FIX THE OBSERVED CRASH**. Measured, not suspected.
>
> The A/B ran on a fresh binary (all three fix markers verified present in it), gated on an empty GPU, provenance asserted:
>
> ```
> graphs_off : 0 glibc diagnostic(s) over 24 tokens, server ALIVE, 0 new Xids
> graphs_on  : "corrupted double-linked list", died, 0 tokens
> ```
>
> **Identical to the pre-fix result.**
>
> **Why**: the crash occurs at `cuGraphInstantiate`, and `cuMemPoolDestroy` appears **zero** times in the failing log — the pool is *retained* on the success path. So none of the six drains this PR adds ran before the crash. **The A/B did not test this fix; it tested a path this fix does not touch.**
>
> I had a mechanism that explained the *shape* of the evidence (capture-only, varying diagnostic, moving abort site) and stopped there, instead of checking whether its trigger occurs on the failing path. It does not. The missing check was one grep: `cuMemPoolDestroy` count in the failing log.
>
> **What this PR's heap change still is**: a real latent UB fix — dangling cache entries after pool destruction, and the `CapturedGraph::drop` field-ordering hole where `output` reached the cache after its pool was destroyed. Both are genuine. Neither is the observed crash. Reviewed and merged on that basis, or held; not merged as "fixes the corruption".
>
> The mutation was **not** run: reverting a fix that changed nothing proves nothing.
>
> **Where the evidence now points.** The corrupting write precedes `cuGraphInstantiate`, which is the first thing after capture to allocate host memory and therefore the first to walk the arena. The only capture-time event left is the one present in every log all night: the **`172032`-byte allocation that misses the warm pool during capture** — `4096 × 21 × 2`, the `xs` compressor history — which candle's own warning calls "graph will be unstable". That makes **zero-allocation capture the actual fix rather than the elegant one**, and **that requirement is currently UNOWNED** — see below.

## Reviewer guide — what each part claims, and on what evidence

This PR is several things. They stand or fall independently and should be judged that way; nothing here needs the heap fix to be correct.

| # | Change | Claim | Evidence | Verdict basis |
|---|---|---|---|---|
| 1 | **Latent UB: drain the alloc cache before destroying the private pool** (`graph.rs`, fork `88d86a20`) | Closes a real use-after-free: cached pointers outlive the pool they came from, and `CapturedGraph::drop` handed `output` to the cache *after* destroying the pool, because fields drop after the Drop body. | Code reading + the fork's own contract. **NOT tested to fix anything** — `cuMemPoolDestroy` runs zero times on the failing path, so its drains never fire there. | Review as **latent-only UB hardening**. Merge if the reasoning holds; do **not** merge as "fixes the corruption". |
| 2 | **2b: address-stable graph inputs** (`layers.rs`, `normal.rs`) | Capture bakes device pointers, so per-step reallocation made every replay read a dropped tensor. This is why replay output was discarded. | Mechanism is `cuStreamBeginCapture` semantics; the discard was already in the code with a comment saying why. | Necessary for any correct replay. Independent of the corruption. |
| 3 | **Replay output used, behind a verified comparison** (`normal.rs`, `graph.rs`) | First N replays also run eager and compare `max|Δ| ≤ 1e-4`; one divergence latches replay off for the process. | Design; not exercised, since capture still cannot launch. | Review the *policy*, not a result. |
| 4 | **Architecture guard** (`weights.rs`) | The dense decode extractor would silently compute the wrong model for V4/Mixtral/Phi-2; a guard refuses by contract instead of by luck. | 9 tests incl. a distribution that defeats a length check; fixtures mutation-verified. | Self-contained. Judge on the tests. |
| 5 | **KV preallocation + pin** (`single_cache.rs`) | `append_graph` never got wave48-BY's preallocation correction; and its buffer must not resize under a captured graph. | Reproduced the exact H200 error; 9 tests, two mutations each caught by one test. | Self-contained. |
| 6 | **D18 log honesty** (`lib.rs`, `graph.rs`) | Three subsystems reported success while doing nothing; `replay()` discarded a sync error. | Direct from logs. | Self-contained. |
| 7 | **Measurement tooling** (`arc-tools/`) | A script built one binary and measured another; the PTX gate isn't on master. | Reproduced live (`git revision: ab42c4508` vs checkout `7fbdfcfdb`). | Self-contained, and arguably the most reusable part. |
| 8 | **Pre-warm probe** (`normal.rs`, `arc-tools/`) | A **probe, not a fix**. Superseded by the `xs` chain's real pin. | Not run. | **Do not merge as a solution.** Labelled so in code and output. |

A live V4 server logged `CUDA graph: NULL stream, capture disabled` and then, on the very next line, `CUDA graph runner initialized`. Pulling that thread found four things: why capture was off, why it could never have produced a correct token, why it killed the process when forced on, and a measurement bug that made an earlier number unattributable.

---

## 1. Why ArcGraph was inert

Capture is disabled because **candle runs on the legacy default stream**, which CUDA forbids capturing. That is a launch-time choice, not a lost stream: `mistralrs_for_server_builder.rs:1095` already selects `Device::new_cuda_with_stream(0)` — but only under `ARC_CAPTURE_STREAM`, unset by default. Three stacked default-off gates in total (`ARC_CAPTURE_STREAM`, `ARC_V4_CAPTURE_PROBE`, `ARC_CANDLE_ALLOC_CACHE`); nothing captures without all three.

**Confirmed on hardware**: with the flag set, `CUDA graph: non-null stream on device 0, capture enabled`, proceeding to `ARC capture: V4 forward RECORDED (bs=1)`.

## 2. The host heap corruption — a latent UB found while chasing it (NOT the crash)

Forcing capture on did not produce a graph. It killed the process:

```
ARC capture: V4 forward RECORDED (bs=1); instantiating + launching
malloc_consolidate(): invalid chunk size
```

That is **glibc aborting on host heap corruption**, not a CUDA error return — none of `end_capture_and_cache`'s bails logged, because nothing returned an error.

**The defect.** The caching allocator stores `(bytes, ptr)` with **no record of which memory pool a pointer came from** (`device.rs:52`). It is device-global and lives for the process. The private pool is **per-capture**. The two lifetimes are unrelated and the cache is the longer one.

1. `begin_capture` installs the private pool as device default (`graph.rs`, `cuDeviceSetMemPool`).
2. cudarc allocates via `cuMemAllocAsync` (`cudarc/result.rs:683`), which draws from the **current default pool** — so every cache miss during capture is served **from the private pool**.
3. `CudaStorage::drop` → `byte_len_and_leak()` → `cache_put` parks it in `deferred`.
4. The pool is destroyed — on error paths immediately, on success in `CapturedGraph::drop`.
5. A later `cache_take` hands the pointer out again; its eventual Drop calls `cuMemFreeAsync` **into a pool that no longer exists**.

`cuMemPoolDestroy` with outstanding allocations is undefined behaviour, and freeing into a destroyed pool corrupts the driver's host-side bookkeeping — which lives in the process's glibc arena.

**⚠️ READ THE BANNER FIRST. This was offered as the root cause and it is not.** It explains the *shape* of the evidence, which is why it was convincing — and the shape is not the same as the mechanism firing on the failing path. It does not fire there: `cuMemPoolDestroy` runs zero times before the crash. What follows is why it looked right, kept for the record:
- **Confined to capture**, because the private pool only exists during capture. That is exactly where the measured A/B boundary falls, and this says *why*.
- **The diagnostic varies between runs** (`malloc_consolidate` vs `corrupted double-linked list`) — a stray free trips whichever check the allocation history reaches first.
- **The abort site moves**, for the same reason — which is why the bisect was abandoned *before* this was found, and why it would have named a different innocent bystander each sample.

### Two catches that are sharper than the finding

- **Draining only `deferred` would have been insufficient.** A buffer still live at capture end — the captured graph's own **output tensor** — drops when `capturing` is already `false` and therefore lands in `free`. Both lists can hold private-pool pointers and there is no provenance to separate them, so the obvious fix would have left the bug live on the success path, i.e. the common one.
- **`CapturedGraph::drop` is the whole bug in miniature.** Rust drops fields *after* the Drop body runs, so `output` — allocated from that pool — was handed to the cache after the pool was destroyed. That reads as correct in review indefinitely.

### The fix (Option A, minimal) — closes the latent UB, does NOT fix the crash

Ordering. `CapturedGraph::drop` now: destroy exec (no replay can reference the addresses) → release output (its storage reaches the cache) → **drain while the pool is alive** → destroy the pool.

Every runner path that destroys an *installed* pool drains first: the four error paths in `end_capture_and_cache`, `cancel_capture`, and the Drop above — **six sites, not ten**. The other three (`create_private_pool` attribute failure, `cuDeviceGetMemPool` failure, `cuDeviceSetMemPool` failure) are **provably pre-install**: the pool was created but never became the device default, so `cuMemAllocAsync` cannot have served from it and the cache holds no pointer into it. Draining there would flush a healthy cache for nothing. Each carries a comment saying so, so the fix is not "completed" later by adding them.

**Second defect in the same finding, also fixed: the cache was never drained.** `set_alloc_cache_enabled(true)` was called once and nothing ever called it with `false`, so every buffer it held — including private-pool allocations from every capture — was retained for the process lifetime.

Cross-repo: the fork change landed **first** (`aeonmindai/candle` @ `88d86a20`, adding `drain_alloc_cache_and_free`), and Arc is pinned to that **explicit rev** rather than the branch — a moving branch pointer has turned green builds red here before with no commit in this repo to blame.

### ➡️ The follow-up that makes this question disappear

Both Option A and Option B patch around the fact that **capture allocates at all**. If capture makes zero allocations, the pool-lifetime mismatch cannot arise.

That is reachable, and measured: the remaining capture-time miss is **exactly one buffer**, `172032` bytes = `4096 × 21 × 2` — `hidden_size × total_tokens` in BF16, where 21 = 10 prompt tokens + 11 warmup/deferred steps. It is the `xs` compressor history.

> ⚠️ **ATTRIBUTION — corrected twice, final state.** An earlier revision credited the `min=1 → min=48` gate result to the batch-invariance chain. That was wrong: they wrote no code this session and do not own `xs_rolling.rs`. My correction then **overshot** and said the requirement was *unowned* — also wrong. Verified against the commit log: `xs_rolling.rs` has an **active owner** with four commits today — `8bc6af45c` (roll the xs compressor state), `4cad4976e` (per-row xs compressor state, +925 lines), `73a0c72e3` (*a ragged xs window may hold rows shorter than itself*, +226 lines — the AFTER commit of that gate result), `6e408eb2e` (pin the window width). The constant-allocation-size requirement has been routed to that chain with this PR's evidence, sequenced after their throughput ladder. Recorded in full because a retraction that overshoots is still a false record, and this one was retracted twice.

**Zero-allocation capture makes both A and B unnecessary.** `arc-tools/arcgraph_miss_gate.sh` in this PR is the standalone gate for it — exits 2 on unprovable, decomposes sizes into the buffer owner's terms, and states the requirement as *allocation size constant per decode step*, which is strictly stronger than "correct on ragged batches".

## 3. Why a replay could never have been correct — 2b

`cuStreamBeginCapture` bakes the device pointers it was launched with, and both graph inputs were freshly allocated every step, so a replay read the address of a tensor that had already been dropped. That is why `normal.rs` discarded replayed logits and ran an eager forward for the real ones — **no token Arc has ever emitted came from a graph replay**.

Both inputs now live at stable addresses updated in place via `Tensor::slice_set`. Positions stage through a constant-size `from_vec` (that buffer is the *source*, so its address is never one the graph depends on); token ids are device-to-device, adding no host sync. Positions alone would not have sufficed — a replay reading this step's positions against last step's ids is still wrong, just less visibly.

**The replay's output is now used, but it earns it.** The first `ARC_GRAPH_VERIFY_REPLAYS` (default 3) replays also run eager and compare `max|Δ|` against a tight `1e-4`; both paths run identical kernels on identical inputs, so a loose band would wave through exactly the stale-address failure being guarded. One divergence latches replay output off for the process. A graph returning plausible-but-wrong logits is worse than no graph.

## 4. The architecture guard — a refusal, not a repair

The decode-path extraction failure is `tensor_device_ptr requires CUDA tensor`, the **non-CUDA-storage arm** — *not* PR #96's missing `DType::I32` dtype arm. Different failures in the same helper, and #96 does not fix this one.

More importantly it must **not** be "fixed". `DecodeConfig` describes a dense Llama-shaped model and `extract_model_weights` indexes `layers[1 + i*7 + k]` positionally; V4 is MLA + a 256-expert MoE + untagged MTP entries. If extraction ever started succeeding, V4 decode would route into a dense-transformer kernel stack and return garbage with no error. `check_dense_layer_inventory` converts that luck into a contract, and runs *before* the two `dequantize_w()` probes — a refusal that lands after two large BF16 allocations is not a refusal. Newly refused: V4, Mixtral, Phi-2; none is a regression, all three were already being silently mis-indexed.

## 5. D18 — three subsystems reporting success while doing nothing

- `try_init_graph_runner` printed "CUDA graph runner initialized" unconditionally, including immediately after capture was disabled. Now reports state: `ARCGRAPH STATUS: capture_possible=… captured=… replayed=…`.
- `try_init_autonomous_runner` logged "graph capture deferred until first decode" at **info**. Verified against the workspace: `AutonomousDecodeRunner::capture` has **zero call sites**. That deferral never ends.
- `replay()` discarded `cudaStreamSynchronize`'s return, so an async fault during graph execution became stale output bytes with no error.

## 6. The measurement bug that nearly voided the numbers

`measure_v4_prefill_curve.sh:241-244` built `-p arc-cli` — whose binary is `arc` — and ran `target/release/mistralrs`, from `mistralrs-cli`. The only check was `[ -x "$BIN" ]`, which any leftover binary passes. My first run measured `git revision: ab42c4508` while my checkout was `7fbdfcfdb`.

This makes *measurements* lie, silently, and **biased toward "no effect"** — a stale binary yields a clean flat result indistinguishable from an honest negative. `arc-tools/lib/build_and_verify.sh` enforces four steps: build the package that produces the binary; take the path from cargo's own artifact stream; refuse unless the binary carries a marker only the code under test has; and once the server is up, **assert it reports the commit that was built**. Steps 3 and 4 are different guarantees — a stale binary that is a different recent build already containing those markers passes 3 while every number came from the wrong commit.

Also: `arc-tools/gpu_box_preflight.sh` is **not on master** (only on the unmerged `wave61/box-preflight-shared-prefix`), so the prefill-curve script could not pass its own gate from a clean checkout. The probe here accepts either that gate or a PTX-JIT receipt, and **refuses if neither ran**.

## 7. [MEASURED] b=1 decode

| quantity | measured |
|---|---|
| b=1 decode, 24 tokens, temp 0 | **15.11 / 15.39 / 15.45 / 15.51 / 15.90 tok/s** (five runs, two boxes) |
| eager V4 forward, sync'd | **49.9 – 59.7 ms** |
| ⇒ share of the step inside the forward | **~88%** |
| bandwidth floor (3.396 GB ÷ 4.8 TB/s) | 0.71 ms |

The last three are provenance-asserted. **This corrects a live claim:** the eight host costs in FACTS were measured at B=256, where the engine loop dominates. At b=1 they are ~12% of the step. "Overhead-bound" now has to name a batch size — and graphs have a ~88% ceiling at b=1 against ~26% at B=256, the opposite of the intuition.

There is deliberately **no graphs-vs-no-graphs tok/s**, and no `T_graph`. Capture records but cannot launch, so there is nothing to time, and a synthesised figure would be worth less than nothing.

## Tests

`check_dense_layer_inventory`: 9 tests, including an inventory whose **total** is exactly `1 + num_layers*7` but whose distribution is 8+6 — a length-check guard passes it and mis-assigns every pointer from layer 1 on. Fixtures verified by neutering the guard (7 of 9 fail; the 2 survivors are the one expecting `Ok` and the one preceding the mutation point).

`single_cache`: 9 tests covering the preallocation correction, the graph-buffer pin, and — the one that matters — that the **eager path still serves what the pinned graph refuses**, because a refusal that kills the request is a worse bug than the one it fixes.

`use_capture_stream`: 5 tests including the wave49-BZ bug in test form (`""`, `"true"`, `"yes"`, `" 1"` must none of them enable it).

Every mutation listed above was run and observed to fail the intended test.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
