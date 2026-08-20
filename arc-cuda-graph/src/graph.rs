//! Capture-once CUDA graph with private memory pool for isolation.
//!
//! The private pool ensures graph-owned memory doesn't interfere with
//! the rest of the application (PagedAttention KV cache, model weights, etc.).
//!
//! Sequence:
//! 1. Save device's current memory pool
//! 2. Create private pool, set RELEASE_THRESHOLD=UINT64_MAX (never release)
//! 3. Install private pool as device default
//! 4. cuStreamBeginCapture → forward pass → cuStreamEndCapture
//! 5. Restore original pool
//! 6. cuGraphInstantiate
//! 7. On replay: cuGraphLaunch (graph uses its own pool addresses)

#[cfg(feature = "cuda")]
use crate::ffi::*;
#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;
#[cfg(feature = "cuda")]
use candle_core::{Device, Tensor};
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
struct CapturedGraph {
    exec: CUgraphExec,
    /// `Option` so [`Drop`] can release it BEFORE the pool dies. Its storage was
    /// allocated from `pool` during capture, and dropping it returns that
    /// pointer to the device alloc cache — which must then be drained while the
    /// pool is still alive.
    output: Option<Tensor>,
    /// Private memory pool for this graph's allocations
    pool: CUmemoryPool,
    /// Needed to drain the alloc cache before `pool` is destroyed.
    device: Device,
}

#[cfg(feature = "cuda")]
impl Drop for CapturedGraph {
    fn drop(&mut self) {
        // ORDER IS THE FIX. Previously this destroyed the exec and the pool and
        // left `output` to drop afterwards (fields drop after the Drop body),
        // which returned a private-pool pointer to the device alloc cache
        // AFTER that pool no longer existed. The cache then handed it out again.
        //
        // 1. exec first: once it is destroyed no replay can reference the
        //    captured addresses, so freeing them is safe.
        unsafe {
            cuGraphExecDestroy(self.exec);
        }
        // 2. release the captured output so its storage reaches the cache.
        drop(self.output.take());
        // 3. drain the cache while the pool is STILL ALIVE, so every buffer
        //    allocated from it is freed into the pool that owns it.
        if let Device::Cuda(cd) = &self.device {
            cd.drain_alloc_cache_and_free();
        }
        // 4. only now is the pool unreferenced.
        unsafe {
            cuMemPoolDestroy(self.pool);
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
    /// Kept so every pool-destroying path can drain the alloc cache first.
    device: Device,
    stream: CUstream,
    device_ordinal: CUdevice,
    graphs: HashMap<usize, CapturedGraph>,
    enabled: bool,
    warmup_remaining: u32,
    /// Successful `replay()` calls. Together with `graphs.len()` this is the
    /// only honest answer to "did ArcGraph do anything this run" — see
    /// [`CudaGraphRunner::status_line`]. A runner that constructed, logged
    /// "initialized", and then did nothing reports `captured=0 replayed=0`.
    replays: u64,
    /// Replays still owed a side-by-side eager comparison before the graph's
    /// output is trusted as this step's logits.
    ///
    /// Turning replay output ON is the one change in this subsystem that can
    /// corrupt tokens silently: a graph reading a stale address returns
    /// plausible logits, not an error. So the first `verify_remaining` replays
    /// run the eager forward too and compare; only after they agree does the
    /// replay output get used on its own. Set by `ARC_GRAPH_VERIFY_REPLAYS`
    /// (default 3); 0 means trust immediately and is not recommended.
    verify_remaining: u32,
    /// Set once a verification has failed — latches replay off permanently.
    verify_failed: bool,
    /// RUN-161: deferred-free warmup passes still owed. After the eager warmups
    /// (which only fill the alloc cache to peak-live), forwards must run with
    /// the device in capture mode so the cache grows to the FULL per-forward
    /// allocation count (every alloc distinct, frees deferred).
    /// Generic: the caller toggles the device's capture mode; the runner just
    /// tracks how many passes are owed. See `try_take_deferred_pass`.
    ///
    /// This was a `bool` — exactly one pass. That is only sufficient if the
    /// per-forward allocation set is the SAME at every decode step. It is not
    /// obviously so: a single pass runs at kv_len = N and capture then runs at
    /// kv_len = N+1, so any buffer whose size tracks context length is a size
    /// the cache has never seen, and it becomes an unstable graph memory node.
    /// A measured V4 capture showed four distinct sizes missing at capture time
    /// (131072, 135168, 16896, 132 bytes) after the single pass had run.
    /// Making the count tunable turns "how many passes are enough?" into a
    /// measurement instead of an assumption — and if no finite number drives
    /// the miss count to zero, that is itself the answer: the sizes are
    /// context-dependent and the real fix is shape-constant buffers, not more
    /// warmup.
    deferred_passes_remaining: u32,
    /// Deferred passes still available to `grant_extra_deferred_pass`. Counts
    /// down independently of `deferred_passes_remaining` so that "keep warming
    /// while the profile grows" cannot become an unbounded loop.
    deferred_extra_budget: u32,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphRunner {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphRunner {}

#[cfg(feature = "cuda")]
impl CudaGraphRunner {
    pub fn new(device: &Device, warmup_steps: u32) -> candle_core::Result<Self> {
        Self::new_with_passes(device, warmup_steps, Self::default_deferred_passes())
    }

    /// Number of deferred-free warmup passes, from `ARC_GRAPH_DEFERRED_PASSES`.
    ///
    /// Default 4, not the historical 1. One pass only suffices if every decode
    /// step allocates the same set of sizes, and V4's does not: the rolling
    /// compressor's retained tail is rebuilt at width `tokens - base`, and
    /// `base` jumps a whole `ratio` at a group boundary while `tokens` climbs by
    /// one, so the size cycles through `ratio` consecutive values (measured:
    /// `4096 × {18,19,20,21}`). One pass warms one phase of that cycle; the
    /// captured step lands on another phase and allocates a size the pool has
    /// never held. `ratio` is 4 on the shipped V4 configuration.
    ///
    /// This is a floor, not the answer — `grant_extra_deferred_pass` keeps
    /// extending while the observed size profile is still growing, so a model
    /// with a longer cycle is covered without retuning this number.
    /// See `deferred_passes_remaining`.
    pub fn default_deferred_passes() -> u32 {
        std::env::var("ARC_GRAPH_DEFERRED_PASSES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
    }

    /// Upper bound on passes granted by `grant_extra_deferred_pass`, from
    /// `ARC_GRAPH_DEFERRED_MAX` (default 24). The bound is what keeps a model
    /// whose allocation sizes grow monotonically with context length from
    /// warming forever: it will exhaust the budget, and the capture-miss assert
    /// will then refuse the capture and name the sizes. That refusal is the
    /// useful result — it says the fix is shape-constant buffers (or allocator
    /// size-class bucketing), not more warmup.
    pub fn default_deferred_max() -> u32 {
        std::env::var("ARC_GRAPH_DEFERRED_MAX")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(24)
    }

    pub fn new_with_passes(
        device: &Device,
        warmup_steps: u32,
        deferred_passes: u32,
    ) -> candle_core::Result<Self> {
        let Device::Cuda(cuda_dev) = device else {
            candle_core::bail!("CudaGraphRunner requires a CUDA device");
        };
        let stream = cuda_dev.cuda_stream().cu_stream();

        if stream.is_null() {
            // Not a bug in this crate, and not a lost stream: candle's
            // `BackendDevice::new` binds `CudaContext::default_stream()`, which
            // is the legacy default stream (handle 0). `cuStreamBeginCapture`
            // rejects it by CUDA's own rules, so capture is genuinely
            // impossible on this device — permanently, for this process.
            //
            // The remedy is a *launch-time* decision, so name it here: the
            // operator reading this line is the only one who can act on it.
            tracing::warn!(
                "ArcGraph INERT: candle is on the legacy default (NULL) stream, which CUDA \
                 forbids capturing. No CUDA graph will be captured or replayed for the life of \
                 this process, and decode will run entirely eagerly. To enable capture, restart \
                 with ARC_CAPTURE_STREAM=1 (server_builder init_device -> \
                 Device::new_cuda_with_stream, a real capturable stream). Note capture also \
                 requires ARC_V4_CAPTURE_PROBE=1 and ARC_CANDLE_ALLOC_CACHE=1; all three are \
                 unset by default."
            );
            arc_profiler::mark_unreachable(
                "cuda_graph.capture",
                "candle is on the legacy default (NULL) stream; cuStreamBeginCapture is refused, \
                 so no graph is captured or replayed. Set ARC_CAPTURE_STREAM=1 to bind a \
                 capturable stream.",
                "arc-cuda-graph/src/graph.rs:71",
            );
            return Ok(Self {
                device: device.clone(),
                stream,
                device_ordinal: 0,
                graphs: HashMap::new(),
                enabled: false,
                warmup_remaining: 0,
                replays: 0,
                verify_remaining: 0,
                verify_failed: false,
                deferred_passes_remaining: 0,
                deferred_extra_budget: 0,
            });
        }

        // Get device ordinal from the stream's context
        let ordinal = match device.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id as CUdevice,
            _ => 0,
        };

        tracing::info!("CUDA graph: non-null stream on device {ordinal}, capture enabled");

        Ok(Self {
            device: device.clone(),
            stream,
            device_ordinal: ordinal,
            graphs: HashMap::new(),
            enabled: true,
            warmup_remaining: warmup_steps,
            replays: 0,
            verify_remaining: Self::default_verify_replays(),
            verify_failed: false,
            deferred_passes_remaining: deferred_passes,
            deferred_extra_budget: Self::default_deferred_max(),
        })
    }

    /// Free every cached buffer while the private pool is still alive.
    ///
    /// MUST precede every `cuMemPoolDestroy`. Allocations that miss the cache
    /// during capture come from the private pool (`cuMemAllocAsync` draws from
    /// the device's current default pool), and the cache records no pool
    /// provenance — so a pointer outliving its pool is handed out again and
    /// eventually freed into a pool that no longer exists. That is the host
    /// heap corruption.
    fn drain_alloc_cache(&self) {
        if let Device::Cuda(cd) = &self.device {
            cd.drain_alloc_cache_and_free();
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.warmup_remaining == 0
    }

    /// Whether capture is physically possible on this device — i.e. the stream
    /// is capturable and no capture attempt has failed and latched `enabled`
    /// off.
    ///
    /// Distinct from [`is_enabled`](Self::is_enabled), which additionally
    /// requires warmup to have finished. Callers reporting *status* want this
    /// one; callers deciding whether to capture *now* want `is_enabled`.
    pub fn capture_possible(&self) -> bool {
        self.enabled
    }

    /// Graphs actually captured and instantiated, keyed by batch size.
    pub fn graphs_captured(&self) -> usize {
        self.graphs.len()
    }

    /// Successful graph replays so far.
    pub fn replays(&self) -> u64 {
        self.replays
    }

    /// The one line that distinguishes a working ArcGraph from an inert one.
    ///
    /// D18: "CUDA graph runner initialized" is emitted by a runner that will
    /// never capture anything, so it carries no information. This does:
    /// `captured=0 replayed=0` is a subsystem that did nothing, and says so in
    /// terms nobody can mistake for success.
    pub fn status_line(&self) -> String {
        format!(
            "ARCGRAPH STATUS: capture_possible={} captured={} replayed={} \
output_trusted={} verify_remaining={} verify_failed={}",
            self.enabled,
            self.graphs.len(),
            self.replays,
            self.replay_output_trusted(),
            self.verify_remaining,
            self.verify_failed
        )
    }

    /// Returns `true` exactly once, after warmup completes and before the graph
    /// is captured: the caller should then run ONE forward with the device's
    /// caching allocator in capture mode (`set_capture_mode(true)` ... forward
    /// ... `set_capture_mode(false)`) to grow the free pool to the full
    /// per-forward allocation count. Generic across models.
    pub fn try_take_deferred_pass(&mut self) -> bool {
        if self.enabled && self.warmup_remaining == 0 && self.deferred_passes_remaining > 0 {
            self.deferred_passes_remaining -= 1;
            true
        } else {
            false
        }
    }

    /// Ask for one more deferred-free warm pass, spending from a fixed budget.
    /// Returns whether one was granted.
    ///
    /// The caller grants a pass whenever the last one taught the allocator a
    /// size it had not seen, and stops when a pass adds nothing. That turns
    /// "how many passes cover the allocation cycle?" from a constant that has
    /// to be right into an observation that terminates on its own, while
    /// `deferred_extra_budget` guarantees termination even when the sizes never
    /// converge.
    pub fn grant_extra_deferred_pass(&mut self) -> bool {
        if self.deferred_extra_budget == 0 {
            return false;
        }
        self.deferred_extra_budget -= 1;
        self.deferred_passes_remaining += 1;
        true
    }

    /// Deferred passes still available to `grant_extra_deferred_pass`.
    pub fn deferred_extra_budget(&self) -> u32 {
        self.deferred_extra_budget
    }

    /// How many replays must be proven against an eager forward before the
    /// graph's output is trusted. `ARC_GRAPH_VERIFY_REPLAYS`, default 3.
    pub fn default_verify_replays() -> u32 {
        std::env::var("ARC_GRAPH_VERIFY_REPLAYS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3)
    }

    /// Does this replay still owe a side-by-side eager comparison?
    pub fn needs_verification(&self) -> bool {
        !self.verify_failed && self.verify_remaining > 0
    }

    /// May the replay's output be used as the step's real logits?
    pub fn replay_output_trusted(&self) -> bool {
        !self.verify_failed && self.verify_remaining == 0
    }

    /// Record one replay that matched the eager forward.
    pub fn record_verification_pass(&mut self) {
        if self.verify_remaining > 0 {
            self.verify_remaining -= 1;
            if self.verify_remaining == 0 {
                tracing::info!(
                    "ArcGraph: replay output VERIFIED against eager; graph output is now used \
                     as the real logits. {}",
                    self.status_line()
                );
            }
        }
    }

    /// Record a replay that did NOT match. Latches replay output off for good.
    pub fn record_verification_failure(&mut self, detail: &str) {
        self.verify_failed = true;
        tracing::error!(
            "ArcGraph: replay output DIVERGED from the eager forward ({detail}). Graph output \
             will NOT be used; every step falls back to eager. This is a correctness stop, not a \
             performance one — a graph that returns plausible-but-wrong logits is worse than no \
             graph."
        );
    }

    /// Deferred-free passes still owed before capture may begin.
    pub fn deferred_passes_remaining(&self) -> u32 {
        self.deferred_passes_remaining
    }

    pub fn tick_warmup(&mut self) -> bool {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            if self.warmup_remaining == 0 {
                tracing::info!("CUDA graph: warmup done");
            }
            true
        } else {
            false
        }
    }

    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Replay a previously captured graph. Returns the output tensor.
    ///
    /// Takes `&mut self` so a successful replay is counted — see
    /// [`status_line`](Self::status_line). Without the count there is no way to
    /// tell a runner that replayed from one that only initialised.
    pub fn replay(&mut self, batch_size: usize) -> candle_core::Result<Tensor> {
        let captured = self.graphs.get(&batch_size).ok_or_else(|| {
            candle_core::Error::Msg(format!("No graph for batch_size={batch_size}"))
        })?;
        let sync = unsafe {
            let s = cuGraphLaunch(captured.exec, self.stream);
            if s != CUDA_SUCCESS {
                candle_core::bail!("cuGraphLaunch failed: {s}");
            }
            // A fault DURING graph execution is asynchronous: the launch above
            // returns SUCCESS and only the sync reports it. Discarding this
            // return (the previous behaviour) turned an illegal access into a
            // silently poisoned context and an output tensor full of whatever
            // was in the buffer. `end_capture_and_cache` already checks its
            // first launch this way; replay must too, or every replay after the
            // first is unchecked.
            cudaStreamSynchronize(self.stream)
        };
        if sync != CUDA_SUCCESS {
            candle_core::bail!(
                "graph replay sync failed (async fault during graph execution, \
                 batch_size={batch_size}): cudaError {sync}"
            );
        }
        // 🔴 BOOBY TRAP — this is a CLONE OF THE CAPTURED OUTPUT TENSOR, i.e. a
        // handle to storage the graph OVERWRITES on its next launch. It is not
        // a snapshot. Two consequences the caller owns:
        //
        //  * every value this has ever returned aliases every other one, so
        //    holding an "old" result and comparing it to a "new" one compares a
        //    tensor with itself;
        //  * a verification path that launches the graph again — even for a
        //    diagnostic — silently rewrites the bytes it was about to grade.
        //    `normal.rs` therefore takes its verdict from the tensor BEFORE any
        //    extra launch. This nearly voided the probe that found it.
        //
        // Copy it if you need it to outlive the next launch.
        let out = captured
            .output
            .clone()
            .ok_or_else(|| candle_core::Error::Msg("captured graph has no output".into()))?;
        self.replays += 1;
        // D18/D3: the replay count is the ONE number that separates a graph
        // that serves from a graph that only records, and until now nothing
        // printed it after capture — so every report of "replays served" was an
        // inference. Emit it on a geometric-ish schedule: the first replay, the
        // first ten, then every 100. Cost is a handful of lines per run.
        if self.replays <= 10 || self.replays % 100 == 0 {
            tracing::info!("ARCGRAPH REPLAY #{} — {}", self.replays, self.status_line());
        }
        Ok(out)
    }

    /// Create a private memory pool for graph capture.
    fn create_private_pool(&self) -> candle_core::Result<CUmemoryPool> {
        let props = CUmemPoolProps {
            alloc_type: CUmemAllocationType::PINNED,
            handle_type: CUmemHandleType::NONE,
            location: CUmemLocation {
                loc_type: CUmemLocationType::DEVICE,
                id: self.device_ordinal,
            },
            win32_security_attributes: std::ptr::null_mut(),
            max_size: 0,
            usage: 0,
            reserved: [0u8; 54],
        };

        let mut pool: CUmemoryPool = std::ptr::null_mut();
        let s = unsafe { cuMemPoolCreate(&mut pool, &props) };
        if s != CUDA_SUCCESS {
            candle_core::bail!("cuMemPoolCreate failed: {s}");
        }

        // Never release memory back to OS — keep addresses stable for replay
        let mut threshold: u64 = u64::MAX;
        let s = unsafe {
            cuMemPoolSetAttribute(
                pool,
                CUmempoolAttribute::RELEASE_THRESHOLD,
                &mut threshold as *mut u64 as *mut _,
            )
        };
        if s != CUDA_SUCCESS {
            // No drain needed: this pool was created but never installed as the
            // device default (`cuDeviceSetMemPool` has not succeeded), so
            // `cuMemAllocAsync` cannot have served anything from it and the
            // alloc cache holds no pointer into it. Draining here would flush a
            // healthy cache for nothing.
            unsafe {
                cuMemPoolDestroy(pool);
            }
            candle_core::bail!("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed: {s}");
        }

        Ok(pool)
    }

    /// Begin graph capture with a private memory pool.
    /// Returns the pool and saved original pool for restoration.
    pub fn begin_capture(
        &mut self,
        batch_size: usize,
    ) -> candle_core::Result<(CUmemoryPool, CUmemoryPool)> {
        // Create private pool
        let graph_pool = self.create_private_pool()?;

        // Save original pool
        let mut original_pool: CUmemoryPool = std::ptr::null_mut();
        let s = unsafe { cuDeviceGetMemPool(&mut original_pool, self.device_ordinal) };
        if s != CUDA_SUCCESS {
            // No drain needed: this pool was created but never installed as the
            // device default (`cuDeviceSetMemPool` has not succeeded), so
            // `cuMemAllocAsync` cannot have served anything from it and the
            // alloc cache holds no pointer into it. Draining here would flush a
            // healthy cache for nothing.
            unsafe {
                cuMemPoolDestroy(graph_pool);
            }
            candle_core::bail!("cuDeviceGetMemPool failed: {s}");
        }

        // Install private pool
        let s = unsafe { cuDeviceSetMemPool(self.device_ordinal, graph_pool) };
        if s != CUDA_SUCCESS {
            // No drain needed: this pool was created but never installed as the
            // device default (`cuDeviceSetMemPool` has not succeeded), so
            // `cuMemAllocAsync` cannot have served anything from it and the
            // alloc cache holds no pointer into it. Draining here would flush a
            // healthy cache for nothing.
            unsafe {
                cuMemPoolDestroy(graph_pool);
            }
            candle_core::bail!("cuDeviceSetMemPool (install) failed: {s}");
        }

        // Sync stream before capture
        unsafe {
            cudaStreamSynchronize(self.stream);
        }

        // Begin capture
        // RELAXED tolerates cross-stream dependencies that candle's allocator
        // and helper streams create (THREAD_LOCAL rejects them with
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION). (RUN-161)
        let s = unsafe { cuStreamBeginCapture_v2(self.stream, CUstreamCaptureMode::RELAXED) };
        if s != CUDA_SUCCESS {
            // Restore original pool before bailing
            self.drain_alloc_cache();
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
            }
            unsafe {
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("cuStreamBeginCapture failed: {s}");
        }

        tracing::info!("CUDA graph: capture started for batch_size={batch_size} with private pool");
        Ok((graph_pool, original_pool))
    }

    /// End capture, restore original pool, instantiate graph, cache it.
    pub fn end_capture_and_cache(
        &mut self,
        batch_size: usize,
        output: Tensor,
        graph_pool: CUmemoryPool,
        original_pool: CUmemoryPool,
    ) -> candle_core::Result<Tensor> {
        // End capture. NOTE: the private pool stays installed as the device
        // default through instantiate + the first launch. Graph memory nodes
        // (candle's intermediate allocs recorded during capture) bind to the
        // pool active at instantiate, and the stream-ordered allocator backs
        // the capture-time virtual addresses at the first launch. Restoring the
        // original pool BEFORE that (the previous behavior) made the graph
        // allocate from the wrong pool -> the kernels' baked addresses were
        // never backed -> MMU read fault (Xid 31) on launch. Restore only AFTER
        // the first launch has materialized the graph's memory.
        let mut graph: CUgraph = std::ptr::null_mut();
        let s = unsafe { cuStreamEndCapture(self.stream, &mut graph) };
        if s != CUDA_SUCCESS {
            self.drain_alloc_cache();
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("cuStreamEndCapture failed: {s}");
        }

        // Step markers. This sequence is five driver calls with no logging
        // between them, and it has aborted the PROCESS inside that window
        // (`malloc_consolidate(): invalid chunk size`) on a capture that
        // recorded with zero allocator misses. A glibc abort leaves no CUDA
        // error to read, so without a marker per call the only thing the log
        // establishes is "somewhere in here" — which is not a located blocker.
        // These are `info!` on purpose: they cost one line per captured graph,
        // and there is at most one capture per process.
        tracing::info!("ARC capture: [1/5] cuStreamEndCapture OK, graph recorded");

        // How many host->device copies did this capture record, and did their
        // sources get retained?
        //
        // A captured `cuMemcpyHtoDAsync` stores the HOST POINTER; the graph
        // re-reads it on the first launch and on every replay. Every host-built
        // tensor inside the captured region is therefore a dangling read once
        // its transient `Vec` drops, and the driver's launch-time validation of
        // an unmapped source region is what returns 700 SYNCHRONOUSLY from
        // `cuGraphLaunch` on a context that `[1b]` has just proved clean.
        //
        // candle now retains those sources (`arc_capture_retain_host`). This
        // line is the proof it happened: a run that reports `retained=0` has a
        // fix that never fired, which is a different fact from "the capture had
        // no H2D copies" only because the counter also reports the byte total.
        // Never infer this from the absence of a fault.
        let (htod_n, htod_bytes) = candle_core::cuda::arc_capture_htod_retained();
        let dtoh_n = candle_core::cuda::arc_capture_dtoh_count();
        tracing::info!(
            "ARC capture: [1a] capture-time H2D sources retained: {htod_n} copies, \
             {htod_bytes} B; capture-time D2H copies: {dtoh_n} \
             (ARC_HTOD_TRACE=1 backtraces each one)"
        );
        if dtoh_n > 0 {
            // The destructive half of the same bug: a captured D2H writes into
            // the caller's host `Vec`, which is freed long before the graph
            // launches. candle redirects those writes to a leaked buffer so the
            // heap survives, but the forward still wanted a host value it can
            // never get from a graph -- so say so rather than let the eager/graph
            // verification failure look like a numerics problem.
            tracing::error!(
                "ARC capture: {dtoh_n} device->host copies happened INSIDE the captured \
                 region. A graph cannot serve a host readback; the sites must be moved out \
                 of the forward or made device-resident. Re-run with ARC_HTOD_TRACE=1 to \
                 name them."
            );
        }

        // Is the context ALREADY in error before the graph has ever run?
        //
        // `cuGraphLaunch` returned 700 SYNCHRONOUSLY on a capture with zero
        // allocator misses. A synchronous illegal-address from a launch call is
        // characteristic of a STICKY context error raised by earlier work, not
        // of the launch itself — CUDA reports a real graph fault asynchronously,
        // at the following sync. Nothing executes between begin_capture and
        // here (capture records, it does not run), so a non-zero result on this
        // line places the fault BEFORE capture — in the warmup forwards — and a
        // zero places it in the graph. Without this the two are indistinguishable
        // and the 700 gets blamed on whichever one is being worked on.
        let pre = unsafe { cudaStreamSynchronize(self.stream) };
        if pre == CUDA_SUCCESS {
            tracing::info!("ARC capture: [1b] context CLEAN before instantiate");
        } else {
            tracing::error!(
                "ARC capture: [1b] context ALREADY IN ERROR before instantiate (cudaError \
                 {pre}). The fault happened during warmup, not in the graph; every later \
                 CUDA call inherits it."
            );
        }

        // Instantiate (private pool still installed). RUN-161 2b:
        // AUTO_FREE_ON_LAUNCH (=1) so a graph with memory-alloc nodes can be
        // RE-launched (replayed) -- otherwise the 2nd launch fails with
        // INVALID_VALUE.
        //
        // It is NOT unconditionally harmless when the graph has no alloc nodes,
        // which is now the normal case: the capture-miss gate refuses to reach
        // this function unless every allocation was a cache hit, so a clean
        // capture records no memory nodes at all and the flag governs nothing
        // that exists. `ARC_GRAPH_AUTO_FREE=0` drops it so that can be
        // ablated against the abort above rather than assumed innocent.
        const CU_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: u64 = 1;
        let auto_free = !matches!(
            std::env::var("ARC_GRAPH_AUTO_FREE").as_deref(),
            Ok("0") | Ok("false")
        );
        let flags = if auto_free {
            CU_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
        } else {
            0
        };
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let s = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, flags) };
        tracing::info!(
            "ARC capture: [2/5] cuGraphInstantiateWithFlags(flags={flags}) returned {s}"
        );
        unsafe {
            cuGraphDestroy(graph);
        }
        tracing::info!("ARC capture: [3/5] cuGraphDestroy OK");
        if s != CUDA_SUCCESS {
            self.drain_alloc_cache();
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("cuGraphInstantiate failed: {s}");
        }

        // First launch (private pool still installed -> graph memory is
        // allocated/backed here at the capture-time addresses).
        let s = unsafe { cuGraphLaunch(exec, self.stream) };
        tracing::info!("ARC capture: [4/5] first cuGraphLaunch returned {s}");
        if s != CUDA_SUCCESS {
            self.drain_alloc_cache();
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
                cuGraphExecDestroy(exec);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("First cuGraphLaunch failed: {s}");
        }
        // Check the sync: an illegal access during graph EXECUTION surfaces
        // here (async), not at launch. Without this check it silently poisons
        // the CUDA context and the process dies later with no diagnostic.
        // cudaError: 700 = illegalAddress, 719 = launchFailure, 1 = invalidValue.
        let sync = unsafe { cudaStreamSynchronize(self.stream) };
        tracing::info!("ARC capture: [5/5] first-launch cudaStreamSynchronize returned {sync}");
        // RUN-161 diagnostic: measure the TRUE clean graph replay latency here,
        // while the captured input tensors are still alive (no stale-input
        // fault). 10 back-to-back launch+sync, report the best. Compare to the
        // eager forward time to decide if the forward is dispatch- or
        // kernel-bound (i.e. whether the graph can actually speed it up).
        if sync == CUDA_SUCCESS {
            let mut best = std::time::Duration::from_secs(3600);
            for _ in 0..10 {
                let t = std::time::Instant::now();
                let ls = unsafe { cuGraphLaunch(exec, self.stream) };
                unsafe { cudaStreamSynchronize(self.stream) };
                if ls == CUDA_SUCCESS {
                    let e = t.elapsed();
                    if e < best {
                        best = e;
                    }
                }
            }
            tracing::info!("ARC capture: CLEAN graph-launch best-of-10 = {best:?}");
        }
        // Restore the original pool now that the graph's memory is materialized.
        // Subsequent replays reuse the already-allocated graph memory, so the
        // device default pool no longer needs to be the private one.
        unsafe {
            cuDeviceSetMemPool(self.device_ordinal, original_pool);
        }
        if sync != CUDA_SUCCESS {
            self.drain_alloc_cache();
            unsafe {
                cuGraphExecDestroy(exec);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!(
                "Graph first-launch sync failed (async fault during graph execution): cudaError {sync}"
            );
        }

        let result = output.clone();
        self.graphs.insert(
            batch_size,
            CapturedGraph {
                exec,
                output: Some(output),
                pool: graph_pool,
                device: self.device.clone(),
            },
        );
        tracing::info!(
            "CUDA graph: captured + launched for batch_size={batch_size} — {}",
            self.status_line()
        );
        Ok(result)
    }

    /// Cancel an in-progress capture and restore the original pool.
    pub fn cancel_capture(&self, graph_pool: CUmemoryPool, original_pool: CUmemoryPool) {
        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            let _ = cuStreamEndCapture(self.stream, &mut graph);
            if !graph.is_null() {
                cuGraphDestroy(graph);
            }
            cuDeviceSetMemPool(self.device_ordinal, original_pool);
        }
        // Drain before the pool dies: an aborted capture may already have
        // allocated from it, and those pointers are in the cache.
        self.drain_alloc_cache();
        unsafe {
            cuMemPoolDestroy(graph_pool);
        }
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphRunner {
    fn drop(&mut self) {
        // CapturedGraph handles cleanup via its own Drop.
        //
        // The final tally goes to stderr directly, NOT through `tracing`: a
        // subscriber may already be torn down by the time the pipeline drops,
        // and a status line that silently vanishes at exit is exactly the
        // silent-success failure this line exists to prevent (D18).
        eprintln!("ARCGRAPH FINAL {}", self.status_line());
    }
}
