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
    output: Tensor,
    /// Private memory pool for this graph's allocations
    pool: CUmemoryPool,
}

#[cfg(feature = "cuda")]
impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            cuGraphExecDestroy(self.exec);
            cuMemPoolDestroy(self.pool);
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
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

    /// Number of deferred-free warmup passes, from `ARC_GRAPH_DEFERRED_PASSES`
    /// (default 1, the historical behaviour). See `deferred_passes_remaining`.
    pub fn default_deferred_passes() -> u32 {
        std::env::var("ARC_GRAPH_DEFERRED_PASSES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1)
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
                stream,
                device_ordinal: 0,
                graphs: HashMap::new(),
                enabled: false,
                warmup_remaining: 0,
                replays: 0,
                verify_remaining: 0,
                verify_failed: false,
                deferred_passes_remaining: 0,
            });
        }

        // Get device ordinal from the stream's context
        let ordinal = match device.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id as CUdevice,
            _ => 0,
        };

        tracing::info!("CUDA graph: non-null stream on device {ordinal}, capture enabled");

        Ok(Self {
            stream,
            device_ordinal: ordinal,
            graphs: HashMap::new(),
            enabled: true,
            warmup_remaining: warmup_steps,
            replays: 0,
            verify_remaining: Self::default_verify_replays(),
            verify_failed: false,
            deferred_passes_remaining: deferred_passes,
        })
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
        let out = captured.output.clone();
        self.replays += 1;
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
            unsafe {
                cuMemPoolDestroy(graph_pool);
            }
            candle_core::bail!("cuDeviceGetMemPool failed: {s}");
        }

        // Install private pool
        let s = unsafe { cuDeviceSetMemPool(self.device_ordinal, graph_pool) };
        if s != CUDA_SUCCESS {
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
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("cuStreamEndCapture failed: {s}");
        }

        // Instantiate (private pool still installed). RUN-161 2b:
        // AUTO_FREE_ON_LAUNCH (=1) so a graph with memory-alloc nodes can be
        // RE-launched (replayed) -- otherwise the 2nd launch fails with
        // INVALID_VALUE. Harmless if the graph has no alloc nodes.
        const CU_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: u64 = 1;
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let s = unsafe {
            cuGraphInstantiateWithFlags(
                &mut exec,
                graph,
                CU_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
        };
        unsafe {
            cuGraphDestroy(graph);
        }
        if s != CUDA_SUCCESS {
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
        if s != CUDA_SUCCESS {
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
                output,
                pool: graph_pool,
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
        // CapturedGraph handles cleanup via its own Drop
    }
}
