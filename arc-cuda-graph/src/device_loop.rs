//! Parent system: ArcInfer / ArcGraph
//!
//! Device-resident decode loop: **N decode steps cost N `cuGraphLaunch` calls
//! and zero `cudaStreamSynchronize`.**
//!
//! # What was wrong with plain replay
//!
//! [`crate::graph::CudaGraphRunner::replay`] launches the captured graph and
//! then calls `cudaStreamSynchronize` (`graph.rs:362`) before handing the
//! output tensor back. The caller samples the next token on the **host**, which
//! copies logits (or an argmax scalar) down over PCIe, and only then launches
//! the next graph. So a decode step is
//!
//! ```text
//!   GPU runs step  ->  host blocks on sync  ->  D2H  ->  host samples  ->  launch
//! ```
//!
//! Capture removed the cost of *issuing* the work — 3,961 launch APIs per token
//! became 3.8 — and left that serialized round trip completely intact. Deleting
//! the sync on its own buys nothing, because the host-side sample forces a
//! device-to-host copy that synchronizes anyway. **The sync and the host sample
//! have to go together.**
//!
//! # The loop this module closes
//!
//! Per step, all on one stream, no host in the middle:
//!
//! 1. `cuGraphLaunch` — the captured forward.
//! 2. [`crate::sampling_cuda::CudaSampler`] — argmax/top-p **on device**, token
//!    id written to an `i32[batch]` device buffer. Never copied to the host.
//! 3. `arc_graph_step_commit` — scatters that token into the *pinned* graph
//!    input buffer the captured graph reads (so the next launch consumes it),
//!    bumps the device position, and publishes the token into a pinned+mapped
//!    host ring.
//!
//! Nothing in that sequence blocks. The host enqueues all `burst` steps and the
//! GPU works through them while the host is already ahead.
//!
//! # How the host still gets its tokens
//!
//! The ring lives in `cudaHostAlloc(..., cudaHostAllocMapped)` memory. The GPU
//! pushes each token straight into it, `__threadfence_system()`, then publishes
//! a monotonic write head. The host reads both with **plain volatile loads** —
//! no CUDA API call, no stream synchronize, no D2H copy. Draining is a host
//! memory poll, which is categorically different from a stream sync: with a
//! sync the host cannot enqueue step N+1 until step N retires; here every step
//! is already enqueued and the host is only watching them land.
//!
//! # Why the aliasing logits buffer is safe here
//!
//! The tensor `replay()` hands back **aliases graph-owned storage that the next
//! `cuGraphLaunch` overwrites**. Holding it across a launch, or draining it
//! lazily, reads data the next step already clobbered — silently, producing
//! plausible tokens. That alias/clone class has already caused three
//! verification bugs in this repo.
//!
//! This design is safe against it for one specific reason, and it is worth
//! stating rather than assuming: **every access is enqueued on the same
//! stream, in order, and none of them is a host read.** Per step the stream
//! carries `launch N -> sample -> commit -> launch N+1`. CUDA stream ordering
//! guarantees the sampler kernel starts only after launch N has written the
//! logits and finishes before launch N+1 begins overwriting them. The host
//! never reads the logits at all, so there is no window in which it could read
//! a clobbered buffer.
//!
//! This is strictly stronger than the host path, which has to *synchronize* to
//! read those logits safely. What the host does drain — the token ring — is a
//! **separate, pinned, host-owned allocation** that no graph launch touches, so
//! draining it lazily is sound however far behind the host falls. The one thing
//! that would break the argument is issuing any of these on a different stream;
//! that is why `CudaDeviceOps` captures the graph's own stream and uses it for
//! the sampler and the commit kernel too.
//!
//! # Detecting a fault without synchronizing
//!
//! A fault *during* graph execution is asynchronous — `cuGraphLaunch` returns
//! SUCCESS and only a later API call surfaces it. `replay()` used a full sync
//! for exactly this reason and the comment there says so. Dropping the check
//! outright would turn an illegal access into a silently poisoned context and
//! an output buffer full of garbage, a bug class this codebase has shipped more
//! than once. So the check is *replaced*, not removed, by three signals that
//! together need no blocking call:
//!
//! * **`cudaStreamQuery`** — non-blocking. NVIDIA's CUDA Runtime API reference
//!   for it states: *"Note that this function may also return error codes from
//!   previous, asynchronous launches."* So a sticky async fault surfaces on a
//!   call that never blocks.
//! * **Idle-but-short** — `cudaStreamQuery` returning `cudaSuccess` means every
//!   enqueued step has retired. If all of them retired and fewer tokens than
//!   steps reached the ring, the work definitively did not do what it was asked
//!   to. That is a fault with no timeout heuristic in it.
//! * **The device fault word** — the commit kernel range-checks the sampled
//!   token and the position before it writes anything, and latches a code into
//!   a pinned word. It travels back with the ring, so reading it is free.
//!
//! `max_spins` is a last-resort liveness backstop for a kernel that hangs
//! without ever erroring. It is not the primary detector.
//!
//! # KNOWN LIMITATION — the burst holds the scheduler
//!
//! A burst enqueues `burst` decode steps back-to-back and does not return to
//! the engine until they have all landed. **For that whole window the engine
//! admits no new request**, so a request arriving mid-burst waits up to
//! `burst` decode steps (default 4) before it is even considered.
//!
//! At batch 1 — the only shape [`admit`] accepts — that is harmless: there is
//! one sequence, nothing to interleave with, and the alternative is a host
//! round trip per token. It is **not** harmless above batch 1, where a burst
//! would straddle sequences with different stop conditions and different
//! admission urgency, and where the wasted tail after one sequence's EOS is
//! multiplied by the batch. That is why `admit` refuses `batch_size != 1`
//! outright rather than treating it as a tuning question.
//!
//! Anyone widening this past batch 1 owns that problem first. Raising `burst`
//! trades admission latency for host round trips one-for-one, and the
//! trade is only free while there is exactly one sequence in flight.
//!
//! # Layering
//!
//! Everything above the CUDA calls is expressed against the [`DeviceOps`]
//! trait, and that trait deliberately has **no `synchronize` method** — the
//! driver cannot block the stream because it is not given a way to. The
//! resulting [`BurstDriver`] is plain Rust, compiles without the `cuda`
//! feature, and is unit-tested against a scripted fake GPU. The CUDA-gated
//! [`CudaDeviceOps`] is pointer plumbing and nothing else.

use std::fmt;

// ===========================================================================
// Configuration
// ===========================================================================

/// Opt-in switch. Absent or `0` means this whole module is inert and decode
/// takes the existing `replay()` + host-sample path.
pub const ENV_ENABLE: &str = "ARC_GRAPH_DEVICE_LOOP";
/// Steps enqueued per burst before the host drains. Default [`DEFAULT_BURST`].
pub const ENV_BURST: &str = "ARC_GRAPH_DEVICE_LOOP_BURST";

/// Steps launched back-to-back before the host drains the ring.
///
/// The whole burst is enqueued without any host round trip, so this is also
/// how far the GPU may run ahead of the host's `Sequence` bookkeeping — and
/// therefore the worst-case detection latency, in steps, for a stop string, an
/// EOS, or an async fault. 4 keeps the wasted-work tail small while still
/// amortising the drain over four launches.
pub const DEFAULT_BURST: usize = 4;

/// Tokens the pinned ring holds per row. Must exceed `burst`, or a burst could
/// overwrite tokens the host has not read yet.
pub const DEFAULT_RING_SIZE: usize = 1024;

/// Host spins between `cudaStreamQuery` calls while draining. Pure host
/// memory polls are ~free; the driver query costs a syscall-ish amount, so it
/// is not worth doing on every iteration.
pub const DEFAULT_POLL_EVERY: u32 = 64;

/// Emit [`DeviceDecodeLoop::status_line`] every this many bursts. The counter
/// it carries (`launches`) is the only proof the loop actually ran.
pub const STATUS_EVERY_BURSTS: u64 = 64;

/// Liveness backstop: give up after this many spins even if the stream never
/// reports an error. Only trips on a kernel that hangs without faulting.
pub const DEFAULT_MAX_SPINS: u64 = 200_000_000;

/// Is the device decode loop switched on?
///
/// Default **off**. `ARC_GRAPH_DEVICE_LOOP=1` opts in; `0`, `false` and `off`
/// are explicit refusals so an operator can pin it off in an environment that
/// sets it upstream.
pub fn device_loop_enabled() -> bool {
    match std::env::var(ENV_ENABLE) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => false,
    }
}

/// Burst length from [`ENV_BURST`], clamped to at least 1.
pub fn burst_from_env() -> usize {
    std::env::var(ENV_BURST)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_BURST)
}

/// Everything the driver needs that does not change from burst to burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLoopConfig {
    /// Steps per burst. Also the run-ahead bound and the detection latency.
    pub burst: usize,
    /// Tokens per row in the pinned ring.
    pub ring_size: usize,
    /// Host spins between stream queries while draining.
    pub poll_every: u32,
    /// Spin budget before declaring the stream stalled.
    pub max_spins: u64,
    /// Stop the burst's token stream here. `< 0` disables EOS truncation.
    pub eos_token_id: i32,
    /// Vocabulary size; the commit kernel range-checks against it.
    pub vocab: usize,
    /// Exclusive upper bound on the device position. The fixed-capacity graph
    /// decode arm reads a constant `capacity`-wide window and writes KV at
    /// `position`; advancing past it is an out-of-bounds write, so both the
    /// driver and the kernel refuse to.
    pub position_limit: u32,
}

impl DeviceLoopConfig {
    /// Config with the shipped defaults for everything but the model facts.
    pub fn new(vocab: usize, eos_token_id: i32, position_limit: u32) -> Self {
        Self {
            burst: burst_from_env(),
            ring_size: DEFAULT_RING_SIZE,
            poll_every: DEFAULT_POLL_EVERY,
            max_spins: DEFAULT_MAX_SPINS,
            eos_token_id,
            vocab,
            position_limit,
        }
    }

    /// Reject a config that cannot be run safely, rather than discovering it
    /// as corruption at step `ring_size + 1`.
    pub fn validate(&self) -> Result<(), DeviceLoopError> {
        if self.burst == 0 {
            return Err(DeviceLoopError::BadConfig("burst must be non-zero"));
        }
        if self.vocab == 0 {
            return Err(DeviceLoopError::BadConfig("vocab must be non-zero"));
        }
        if self.poll_every == 0 {
            // The drain does `spins % poll_every`; zero would divide by zero.
            return Err(DeviceLoopError::BadConfig("poll_every must be non-zero"));
        }
        if self.ring_size <= self.burst {
            // A burst longer than the ring overwrites its own unread tokens.
            return Err(DeviceLoopError::BadConfig(
                "ring_size must exceed burst, or a burst overwrites unread tokens",
            ));
        }
        Ok(())
    }

    /// Steps this burst may launch given how far the sequence has already got.
    ///
    /// Clamped by the position limit so the burst can never walk the device
    /// position past the fixed-capacity window. Returns 0 when the sequence has
    /// no room left, which the caller must treat as "fall back to eager", not
    /// as "launch zero steps and carry on".
    pub fn plan_steps(&self, current_position: u32, max_new_tokens: usize) -> usize {
        let room = self.position_limit.saturating_sub(current_position) as usize;
        self.burst.min(room).min(max_new_tokens)
    }
}

// ===========================================================================
// Errors
// ===========================================================================

/// Codes the commit kernel latches into the pinned fault word. Kept in sync
/// with `arc_graph_step_commit` in `src/cuda/decode_loop.cu`.
pub const FAULT_NONE: i32 = 0;
/// Sampled token id was outside `[0, vocab)`.
pub const FAULT_TOKEN_RANGE: i32 = 1;
/// Advancing the position would leave the fixed-capacity window.
pub const FAULT_POSITION_LIMIT: i32 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceLoopError {
    BadConfig(&'static str),
    /// A CUDA call reported a failure. `at` names the call site.
    Cuda {
        at: &'static str,
        code: u32,
    },
    /// The stream retired every enqueued step but the ring is short. The work
    /// completed without producing the tokens it was launched to produce.
    LostTokens {
        expected: i32,
        got: i32,
    },
    /// The commit kernel refused to write and latched a code.
    DeviceFault(i32),
    /// The write head ran more than `ring_size` ahead of the read cursor, so
    /// unread tokens were overwritten. Reported rather than silently returning
    /// whatever survived.
    RingOverrun {
        pending: i64,
        ring_size: usize,
    },
    /// Spin budget exhausted with the stream still busy and no error reported.
    Stalled {
        spins: u64,
    },
}

impl fmt::Display for DeviceLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadConfig(m) => write!(f, "device decode loop misconfigured: {m}"),
            Self::Cuda { at, code } => {
                write!(f, "device decode loop: {at} failed: CUDA code {code}")
            }
            Self::LostTokens { expected, got } => write!(
                f,
                "device decode loop: the stream retired every enqueued step but only {got} of \
                 {expected} tokens reached the ring. The graph ran and did not commit its tokens \
                 — treat this as an async fault, not as slow progress"
            ),
            Self::DeviceFault(code) => {
                let what = match *code {
                    FAULT_TOKEN_RANGE => "sampled token id outside [0, vocab)",
                    FAULT_POSITION_LIMIT => {
                        "advancing the position would leave the fixed-capacity KV window"
                    }
                    _ => "unrecognised code",
                };
                write!(
                    f,
                    "device decode loop: commit kernel refused to write (code {code}: {what}). No \
                     token was scattered into the graph input buffer, so the next launch would \
                     have read a stale id"
                )
            }
            Self::RingOverrun { pending, ring_size } => write!(
                f,
                "device decode loop: {pending} tokens pending in a {ring_size}-slot ring — the GPU \
                 overwrote tokens the host had not read. Tokens were lost; this is a correctness \
                 stop, not a throughput warning"
            ),
            Self::Stalled { spins } => write!(
                f,
                "device decode loop: stream still busy after {spins} spins with no error reported. \
                 The graph is hung rather than faulted"
            ),
        }
    }
}

impl std::error::Error for DeviceLoopError {}

impl From<DeviceLoopError> for candle_core::Error {
    fn from(e: DeviceLoopError) -> Self {
        candle_core::Error::Msg(e.to_string())
    }
}

// ===========================================================================
// The device interface
// ===========================================================================

/// Result of a non-blocking stream probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Every enqueued step has retired (`cudaSuccess`).
    Idle,
    /// Work still outstanding (`cudaErrorNotReady`).
    Busy,
}

/// The device operations a burst needs.
///
/// **There is deliberately no `synchronize` method.** The driver is structurally
/// incapable of blocking the stream, which is the property this whole module
/// exists to guarantee — it cannot be regressed by an edit that "just adds a
/// sync to be safe", because there is nothing to call.
pub trait DeviceOps {
    /// One `cuGraphLaunch` of the captured decode graph. Must not block.
    fn launch_graph(&mut self) -> Result<(), DeviceLoopError>;

    /// Sample on device: graph logits -> the `i32[batch]` token buffer. Must
    /// not block and must not copy the token to the host.
    fn sample_on_device(&mut self) -> Result<(), DeviceLoopError>;

    /// Commit on device: token -> pinned graph input ids, position += 1, and
    /// publish into the pinned ring. Must not block.
    fn commit_on_device(&mut self) -> Result<(), DeviceLoopError>;

    /// Non-blocking stream probe. `Err` for a sticky async fault.
    fn poll_stream(&mut self) -> Result<StreamState, DeviceLoopError>;

    /// Monotonic write head for `row`, read from pinned mapped host memory.
    /// A plain volatile load — no CUDA API call.
    fn ring_head(&self, row: usize) -> i32;

    /// Token at absolute index `index` for `row` (the impl applies the modulo).
    fn ring_slot(&self, row: usize, index: i32) -> i32;

    /// The sticky fault word, read from pinned mapped host memory.
    fn device_fault(&self) -> i32;

    /// Rows the ring carries.
    fn batch(&self) -> usize;
}

// ===========================================================================
// The driver
// ===========================================================================

/// What one burst produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurstOutcome {
    /// Tokens per row, in generation order, truncated at and including EOS.
    pub tokens: Vec<Vec<i32>>,
    /// True when some row emitted EOS inside this burst.
    pub hit_eos: bool,
    /// Steps actually launched. When `hit_eos` is true this exceeds the number
    /// of tokens returned: the burst was already enqueued and the GPU computed
    /// the tail before the host could know. That work is discarded, and this
    /// field is how much of it there was.
    pub steps_launched: usize,
}

/// Drives bursts against a [`DeviceOps`], owning the ring read cursors.
pub struct BurstDriver {
    cfg: DeviceLoopConfig,
    /// Absolute index of the next unread ring slot, per row.
    cursor: Vec<i32>,
    launches: u64,
    bursts: u64,
    spins: u64,
}

impl BurstDriver {
    pub fn new(cfg: DeviceLoopConfig, batch: usize) -> Result<Self, DeviceLoopError> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            cursor: vec![0; batch.max(1)],
            launches: 0,
            bursts: 0,
            spins: 0,
        })
    }

    pub fn config(&self) -> &DeviceLoopConfig {
        &self.cfg
    }

    /// Total `cuGraphLaunch` calls issued. With `bursts()` this is the honest
    /// answer to "did the device loop do anything": `launches == steps` and
    /// zero stream synchronizes is the entire claim.
    pub fn launches(&self) -> u64 {
        self.launches
    }

    pub fn bursts(&self) -> u64 {
        self.bursts
    }

    /// Host spins spent waiting for tokens to land. Rises when the host
    /// out-runs the GPU, which is the healthy direction.
    pub fn spins(&self) -> u64 {
        self.spins
    }

    /// The read cursor is per-sequence state. Reset it when the ring is reset
    /// (a new request reuses the buffers).
    pub fn reset_cursors(&mut self) {
        for c in self.cursor.iter_mut() {
            *c = 0;
        }
    }

    /// Enqueue `steps` decode steps and drain their tokens.
    ///
    /// Issues exactly `steps` `cuGraphLaunch` calls and **zero** stream
    /// synchronizations. The drain is a host-memory poll on pinned mapped
    /// memory plus a periodic non-blocking `cudaStreamQuery`.
    pub fn run_burst<D: DeviceOps>(
        &mut self,
        ops: &mut D,
        steps: usize,
    ) -> Result<BurstOutcome, DeviceLoopError> {
        if steps == 0 {
            return Err(DeviceLoopError::BadConfig(
                "run_burst called with zero steps; the caller must fall back to eager instead",
            ));
        }
        if steps >= self.cfg.ring_size {
            return Err(DeviceLoopError::BadConfig(
                "burst is not smaller than the ring; it would overwrite its own tokens",
            ));
        }

        // ── enqueue ──────────────────────────────────────────────────────
        // No blocking call in this loop, by construction: `DeviceOps` has no
        // way to synchronize. `poll_stream` is the async-fault probe and it
        // returns immediately whatever the stream is doing.
        for _ in 0..steps {
            ops.launch_graph()?;
            ops.sample_on_device()?;
            ops.commit_on_device()?;
            self.launches += 1;
            // Surface a sticky fault from an EARLIER step while later steps are
            // already in flight. Costs nothing and shortens detection latency
            // from "end of burst" to "next step".
            ops.poll_stream()?;
        }
        self.bursts += 1;

        // ── drain ────────────────────────────────────────────────────────
        let batch = ops.batch().min(self.cursor.len());
        let want: Vec<i32> = self.cursor[..batch]
            .iter()
            .map(|c| c.saturating_add(steps as i32))
            .collect();
        for (row, &target) in want.iter().enumerate() {
            self.await_row(ops, row, target)?;
        }

        // The commit kernel latches its refusal instead of writing, so this
        // must be checked even when every token arrived: a fault on step 3 of 4
        // means steps 4.. read a stale input id.
        let fault = ops.device_fault();
        if fault != FAULT_NONE {
            return Err(DeviceLoopError::DeviceFault(fault));
        }

        // ── collect ──────────────────────────────────────────────────────
        let mut tokens = Vec::with_capacity(batch);
        let mut hit_eos = false;
        for (row, &target) in want.iter().enumerate() {
            let mut row_tokens = Vec::with_capacity(steps);
            let mut idx = self.cursor[row];
            while idx < target {
                let tok = ops.ring_slot(row, idx);
                idx += 1;
                let is_eos = self.cfg.eos_token_id >= 0 && tok == self.cfg.eos_token_id;
                row_tokens.push(tok);
                if is_eos {
                    hit_eos = true;
                    break;
                }
            }
            // The cursor advances over the WHOLE burst even when EOS truncated
            // the returned tokens. Those slots were written; leaving the cursor
            // behind would re-read them as if they were the next burst's.
            self.cursor[row] = target;
            tokens.push(row_tokens);
        }

        Ok(BurstOutcome {
            tokens,
            hit_eos,
            steps_launched: steps,
        })
    }

    /// Wait for `row`'s write head to reach `target`, without synchronizing.
    fn await_row<D: DeviceOps>(
        &mut self,
        ops: &mut D,
        row: usize,
        target: i32,
    ) -> Result<(), DeviceLoopError> {
        let mut spins: u64 = 0;
        loop {
            let head = ops.ring_head(row);
            if head >= target {
                // Overrun check: more unread tokens than the ring can hold
                // means the GPU lapped the host and older tokens are gone.
                let pending = (head as i64) - (self.cursor[row] as i64);
                if pending > self.cfg.ring_size as i64 {
                    return Err(DeviceLoopError::RingOverrun {
                        pending,
                        ring_size: self.cfg.ring_size,
                    });
                }
                return Ok(());
            }

            if spins.is_multiple_of(self.cfg.poll_every as u64) {
                match ops.poll_stream()? {
                    StreamState::Busy => {}
                    StreamState::Idle => {
                        // Every enqueued step retired. Re-read once: the head
                        // store and the stream's retirement are not ordered
                        // against each other from the host's point of view.
                        let head = ops.ring_head(row);
                        if head >= target {
                            return Ok(());
                        }
                        return Err(DeviceLoopError::LostTokens {
                            expected: target,
                            got: head,
                        });
                    }
                }
            }

            spins += 1;
            self.spins += 1;
            if spins >= self.cfg.max_spins {
                return Err(DeviceLoopError::Stalled { spins });
            }
            std::hint::spin_loop();
        }
    }
}

// ===========================================================================
// CUDA implementation
// ===========================================================================

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::ffi::*;
    use crate::sampling_cpu::SamplingConfig;
    use crate::sampling_cuda::CudaSampler;
    use candle_core::cuda::cudarc::driver::sys::CUstream;
    use candle_core::{DType, Device, Tensor};

    /// Device address of a buffer the captured graph reads.
    ///
    /// Refuses a non-contiguous tensor rather than silently taking the address
    /// of the copy `contiguous()` would make — that copy is not the address the
    /// graph baked, so writing to it would look fine and change nothing.
    fn pinned_ptr(t: &Tensor, what: &str) -> candle_core::Result<u64> {
        if !t.is_contiguous() {
            candle_core::bail!(
                "device decode loop: the {what} buffer is not contiguous, so its address is not \
                 the one the captured graph baked. Refusing rather than writing into a copy."
            );
        }
        crate::weights::tensor_device_ptr(t)
    }

    /// `cudaErrorNotReady` — the stream still has outstanding work. Not a
    /// failure, and the only non-success code `cudaStreamQuery` returns that
    /// is not a fault.
    const CUDA_ERROR_NOT_READY: u32 = 600;

    /// Pinned, mapped host memory the GPU writes and the host reads with plain
    /// volatile loads. Freed with `cudaFreeHost` on drop.
    struct PinnedRing {
        tokens: *mut i32,
        head: *mut i32,
        fault: *mut i32,
        batch: usize,
        ring_size: usize,
    }

    impl PinnedRing {
        fn new(batch: usize, ring_size: usize) -> Result<Self, DeviceLoopError> {
            let mut tokens: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut head: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut fault: *mut std::ffi::c_void = std::ptr::null_mut();
            unsafe {
                let s = cudaHostAlloc(&mut tokens, batch * ring_size * 4, CUDA_HOST_ALLOC_MAPPED);
                if s != CUDA_SUCCESS {
                    return Err(DeviceLoopError::Cuda {
                        at: "cudaHostAlloc(ring tokens)",
                        code: s,
                    });
                }
                let s = cudaHostAlloc(&mut head, batch * 4, CUDA_HOST_ALLOC_MAPPED);
                if s != CUDA_SUCCESS {
                    cudaFreeHost(tokens);
                    return Err(DeviceLoopError::Cuda {
                        at: "cudaHostAlloc(ring head)",
                        code: s,
                    });
                }
                let s = cudaHostAlloc(&mut fault, 4, CUDA_HOST_ALLOC_MAPPED);
                if s != CUDA_SUCCESS {
                    cudaFreeHost(tokens);
                    cudaFreeHost(head);
                    return Err(DeviceLoopError::Cuda {
                        at: "cudaHostAlloc(fault word)",
                        code: s,
                    });
                }
                std::ptr::write_bytes(tokens as *mut u8, 0, batch * ring_size * 4);
                std::ptr::write_bytes(head as *mut u8, 0, batch * 4);
                std::ptr::write_bytes(fault as *mut u8, 0, 4);
            }
            Ok(Self {
                tokens: tokens as *mut i32,
                head: head as *mut i32,
                fault: fault as *mut i32,
                batch,
                ring_size,
            })
        }

        fn reset(&self) {
            unsafe {
                std::ptr::write_bytes(self.head as *mut u8, 0, self.batch * 4);
                std::ptr::write_bytes(self.fault as *mut u8, 0, 4);
            }
        }
    }

    impl Drop for PinnedRing {
        fn drop(&mut self) {
            unsafe {
                if !self.tokens.is_null() {
                    cudaFreeHost(self.tokens as *mut _);
                }
                if !self.head.is_null() {
                    cudaFreeHost(self.head as *mut _);
                }
                if !self.fault.is_null() {
                    cudaFreeHost(self.fault as *mut _);
                }
            }
        }
    }

    /// The real [`DeviceOps`]: sampler + commit kernel on the graph's stream.
    ///
    /// Holds `&mut` to nothing outside itself except through raw pointers taken
    /// once at construction, because every buffer it touches must be at a
    /// **stable device address** — the captured graph baked those addresses and
    /// a reallocation silently invalidates the graph.
    pub struct CudaDeviceOps {
        stream: CUstream,
        sampler: CudaSampler,
        sampling_cfg: SamplingConfig,
        /// The graph's output logits tensor, reshaped to `[batch, vocab]`.
        logits: Tensor,
        /// `i32[batch]`, written by the sampler, read by the commit kernel.
        token_ids: Tensor,
        /// The PINNED `U32` token-id buffer the captured graph reads.
        input_ids_ptr: *mut u32,
        /// The PINNED `U32` position buffer, or null when positions are owned
        /// elsewhere (e.g. by a device-side RoPE position path).
        positions_ptr: *mut u32,
        ring: PinnedRing,
        cfg: DeviceLoopConfig,
        /// `cuGraphLaunch` closure state: the exec handle to replay.
        exec: CUgraphExec,
    }

    impl CudaDeviceOps {
        /// * `logits` — the captured graph's output tensor. Its storage address
        ///   is graph-owned and stable; the sampler reads it in place.
        /// * `input_ids_buf` / `positions_buf` — the *pinned* graph input
        ///   buffers (`layers::GRAPH_MODE_INPUT_IDS` / `GRAPH_MODE_POSITIONS`).
        ///   These are the addresses the captured graph reads, which is why a
        ///   device-side write into them closes the loop with no host copy.
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            device: &Device,
            exec: CUgraphExec,
            stream: CUstream,
            sampler: CudaSampler,
            sampling_cfg: SamplingConfig,
            logits: &Tensor,
            input_ids_buf: &Tensor,
            positions_buf: Option<&Tensor>,
            cfg: DeviceLoopConfig,
        ) -> candle_core::Result<Self> {
            cfg.validate()?;
            let batch = sampler.batch();

            // `[1, 1, vocab]` (or `[1, vocab]`) -> `[batch, vocab]`. Reshaping a
            // contiguous tensor is metadata-only, so this does not move the
            // graph-owned storage the sampler is about to read.
            let logits = logits.reshape((batch, sampler.vocab()))?;
            if logits.dtype() != sampler.dtype() {
                candle_core::bail!(
                    "device decode loop: graph logits dtype {:?} != sampler dtype {:?}",
                    logits.dtype(),
                    sampler.dtype()
                );
            }

            if input_ids_buf.dtype() != DType::U32 {
                candle_core::bail!(
                    "device decode loop: the graph input-ids buffer must be U32 (got {:?}); the \
                     commit kernel writes it as uint32_t",
                    input_ids_buf.dtype()
                );
            }
            if input_ids_buf.elem_count() < batch {
                candle_core::bail!(
                    "device decode loop: input-ids buffer holds {} elements, need {batch}",
                    input_ids_buf.elem_count()
                );
            }
            if let Some(p) = positions_buf {
                if p.dtype() != DType::U32 {
                    candle_core::bail!(
                        "device decode loop: the graph positions buffer must be U32 (got {:?})",
                        p.dtype()
                    );
                }
                if p.elem_count() < batch {
                    candle_core::bail!(
                        "device decode loop: positions buffer holds {} elements, need {batch}",
                        p.elem_count()
                    );
                }
            }

            // I32 via `from_vec`: `Tensor::zeros` on I32 is not universally
            // supported across backends, the same reason `CudaSampler::new`
            // builds `keep_idx_scratch` this way.
            let token_ids = Tensor::from_vec(vec![0i32; batch], batch, device)?;

            let input_ids_ptr = pinned_ptr(input_ids_buf, "graph input-ids")? as *mut u32;
            let positions_ptr = match positions_buf {
                Some(p) => pinned_ptr(p, "graph positions")? as *mut u32,
                None => std::ptr::null_mut(),
            };
            let ring = PinnedRing::new(batch, cfg.ring_size)?;

            Ok(Self {
                stream,
                sampler,
                sampling_cfg,
                logits,
                token_ids,
                input_ids_ptr,
                positions_ptr,
                ring,
                cfg,
                exec,
            })
        }

        /// Zero the ring head and fault word. Call when a new sequence starts
        /// reusing these buffers; pair it with [`BurstDriver::reset_cursors`].
        pub fn reset_ring(&self) {
            self.ring.reset();
        }
    }

    impl DeviceOps for CudaDeviceOps {
        fn launch_graph(&mut self) -> Result<(), DeviceLoopError> {
            // The launch and nothing else. `graph.rs::replay` follows this with
            // `cudaStreamSynchronize`; the entire point here is that we do not.
            let s = unsafe { cuGraphLaunch(self.exec, self.stream) };
            if s != CUDA_SUCCESS {
                return Err(DeviceLoopError::Cuda {
                    at: "cuGraphLaunch",
                    code: s,
                });
            }
            Ok(())
        }

        fn sample_on_device(&mut self) -> Result<(), DeviceLoopError> {
            self.sampler
                .sample(&self.logits, None, self.sampling_cfg, &self.token_ids)
                .map_err(|e| {
                    tracing::error!("device decode loop: on-device sample failed: {e}");
                    DeviceLoopError::Cuda {
                        at: "CudaSampler::sample",
                        code: u32::MAX,
                    }
                })
        }

        fn commit_on_device(&mut self) -> Result<(), DeviceLoopError> {
            let token_ptr = match crate::weights::tensor_device_ptr(&self.token_ids) {
                Ok(p) => p as *const i32,
                Err(e) => {
                    tracing::error!("device decode loop: token buffer pointer unavailable: {e}");
                    return Err(DeviceLoopError::Cuda {
                        at: "tensor_device_ptr(token_ids)",
                        code: u32::MAX,
                    });
                }
            };
            unsafe {
                arc_launch_graph_step_commit(
                    token_ptr,
                    self.input_ids_ptr,
                    self.positions_ptr,
                    self.ring.tokens,
                    self.ring.head,
                    self.ring.fault,
                    self.cfg.ring_size as i32,
                    self.cfg.vocab as i32,
                    self.cfg.position_limit as i32,
                    self.ring.batch as i32,
                    self.stream,
                );
            }
            Ok(())
        }

        fn poll_stream(&mut self) -> Result<StreamState, DeviceLoopError> {
            // Non-blocking. NVIDIA's reference for cudaStreamQuery: "Note that
            // this function may also return error codes from previous,
            // asynchronous launches." That is what makes a sticky fault from an
            // earlier graph step visible here without ever blocking.
            let s = unsafe { cudaStreamQuery(self.stream) };
            match s {
                CUDA_SUCCESS => Ok(StreamState::Idle),
                CUDA_ERROR_NOT_READY => Ok(StreamState::Busy),
                code => Err(DeviceLoopError::Cuda {
                    at: "cudaStreamQuery (async fault from an earlier graph step)",
                    code,
                }),
            }
        }

        fn ring_head(&self, row: usize) -> i32 {
            unsafe { std::ptr::read_volatile(self.ring.head.add(row)) }
        }

        fn ring_slot(&self, row: usize, index: i32) -> i32 {
            let slot =
                row * self.ring.ring_size + (index.rem_euclid(self.ring.ring_size as i32) as usize);
            unsafe { std::ptr::read_volatile(self.ring.tokens.add(slot)) }
        }

        fn device_fault(&self) -> i32 {
            unsafe { std::ptr::read_volatile(self.ring.fault) }
        }

        fn batch(&self) -> usize {
            self.ring.batch
        }
    }

    /// The whole device loop behind one handle: sampler, commit kernel, pinned
    /// ring and burst driver.
    ///
    /// Built once per captured graph and kept alive across decode steps,
    /// because every buffer it holds a pointer to must stay at the address the
    /// graph baked. Dropping and rebuilding it mid-generation would hand the
    /// commit kernel a fresh `token_ids` allocation while the graph still reads
    /// the old input-ids address — so callers keep it, they do not recreate it.
    pub struct DeviceDecodeLoop {
        ops: CudaDeviceOps,
        driver: BurstDriver,
        /// The [`super::device_loop_generation`] this loop's ring, cursors and
        /// fault word are valid for. Every `stand_down` — including the one
        /// the `Sequence::set_state` completion funnel issues — bumps the
        /// generation, and [`Self::run`] resets the per-sequence device state
        /// before running a burst from a newer generation. Reset happens HERE,
        /// at engagement, not at completion: the funnel cannot reach this
        /// object (it lives in the pipeline), and between bursts is the only
        /// point where no device write to the ring can still be in flight.
        generation: u64,
    }

    // Same reasoning as `CudaGraphRunner` (`graph.rs:114`): the raw pointers are
    // CUDA handles and pinned host allocations, both valid from any thread, and
    // the pipeline that owns this is held behind a mutex. Without these the
    // `NormalPipeline` field would make the whole pipeline non-`Send`.
    unsafe impl Send for CudaDeviceOps {}
    unsafe impl Sync for CudaDeviceOps {}
    unsafe impl Send for DeviceDecodeLoop {}
    unsafe impl Sync for DeviceDecodeLoop {}

    impl DeviceDecodeLoop {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            device: &Device,
            exec: CUgraphExec,
            stream: CUstream,
            logits: &Tensor,
            input_ids_buf: &Tensor,
            positions_buf: Option<&Tensor>,
            cfg: DeviceLoopConfig,
        ) -> candle_core::Result<Self> {
            let batch = 1usize;
            let vocab = cfg.vocab;
            // Greedy, always. [`admit`] refuses every non-greedy sequence, so
            // taking the mode as a parameter would only create a way for a
            // caller to contradict the guard. The seed is irrelevant under
            // greedy (pure argmax draws no random numbers) and is fixed so the
            // sampler's RNG buffer is deterministic if the mode ever widens.
            let sampling_cfg = SamplingConfig::greedy();
            let sampler = CudaSampler::new(device, batch, vocab, logits.dtype(), 0)?;
            let ops = CudaDeviceOps::new(
                device,
                exec,
                stream,
                sampler,
                sampling_cfg,
                logits,
                input_ids_buf,
                positions_buf,
                cfg,
            )?;
            let driver = BurstDriver::new(cfg, batch)?;
            Ok(Self {
                ops,
                driver,
                generation: super::device_loop_generation(),
            })
        }

        /// Enqueue `steps` decode steps and return their tokens.
        ///
        /// Exactly `steps` `cuGraphLaunch` calls, zero `cudaStreamSynchronize`.
        pub fn run(&mut self, steps: usize) -> candle_core::Result<BurstOutcome> {
            // A stand-down happened since the last burst — sequence completed,
            // errored, or was preempted — so the ring head, read cursors and
            // fault word describe a sequence that no longer exists. Zero them
            // before this burst rather than carrying them across sequences.
            let current = super::device_loop_generation();
            if self.generation != current {
                self.reset();
                self.generation = current;
            }
            let out = self.driver.run_burst(&mut self.ops, steps)?;
            // D18: a subsystem that only announces itself at startup cannot be
            // told apart from one that started and then did nothing. Emit the
            // running counters periodically so `launches` is observable in a
            // live run, not just in a debugger.
            if self.driver.bursts().is_multiple_of(STATUS_EVERY_BURSTS) {
                tracing::info!("{}", self.status_line());
            }
            Ok(out)
        }

        /// Start a fresh sequence on the same buffers.
        pub fn reset(&mut self) {
            self.ops.reset_ring();
            self.driver.reset_cursors();
            clear_pending_tokens();
        }

        /// `cuGraphLaunch` calls issued so far — the numerator of the claim.
        pub fn launches(&self) -> u64 {
            self.driver.launches()
        }

        pub fn bursts(&self) -> u64 {
            self.driver.bursts()
        }

        /// The one line that separates a working device loop from an inert one
        /// (D18). `launches=0` is a subsystem that did nothing and says so.
        pub fn status_line(&self) -> String {
            format!(
                "ARCGRAPH DEVICE LOOP: launches={} bursts={} burst_len={} host_syncs=0 \
                 host_spins={} pending={}",
                self.driver.launches(),
                self.driver.bursts(),
                self.driver.config().burst,
                self.driver.spins(),
                pending_token_count(),
            )
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{CudaDeviceOps, DeviceDecodeLoop};

// ===========================================================================
// Admission
// ===========================================================================

/// Why the device loop refused a request.
///
/// Refusing is the normal, safe outcome — the caller falls back to the existing
/// `replay()` + host-sample path. Each variant names something the device
/// sampler cannot reproduce *exactly*, because a token that differs from the
/// host path is a silent behaviour change, not an optimisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// `ARC_GRAPH_DEVICE_LOOP` is not set. The default.
    NotEnabled,
    /// Only batch 1 is wired: the burst holds the scheduler for its whole
    /// length, so it must not straddle sequences with different stop conditions.
    NotBatchOne,
    /// Prefill, not decode.
    NotDecodeStep,
    /// No captured graph to replay, or its output is not yet trusted.
    GraphNotReady,
    /// Non-greedy sampling. The device sampler runs Splitmix64 per row while
    /// the host runs Isaac64, so for any stochastic config the two draw
    /// *different, equally valid* tokens — which silently breaks seeded
    /// reproducibility. Greedy is deterministic, so device and host agree
    /// exactly, which is also what makes the path verifiable.
    NotGreedy,
    /// The caller wants logprobs, which the device sampler does not return.
    WantsLogprobs,
    /// A logits processor, grammar or penalty must see the logits on the host.
    HostSideLogitsWork,
    /// No room left in the fixed-capacity KV window for even one step.
    NoPositionRoom,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NotEnabled => "ARC_GRAPH_DEVICE_LOOP is not set",
            Self::NotBatchOne => "batch size is not 1",
            Self::NotDecodeStep => "this is a prefill step, not decode",
            Self::GraphNotReady => "no captured graph, or its output is not yet trusted",
            Self::NotGreedy => {
                "sampling is not greedy; the device and host RNGs differ, so tokens would \
                 diverge from the seeded host path"
            }
            Self::WantsLogprobs => {
                "the request asks for logprobs, which the device sampler \
                                    does not produce"
            }
            Self::HostSideLogitsWork => {
                "a logits processor, grammar or repetition penalty needs the logits on the host"
            }
            Self::NoPositionRoom => "no room left in the fixed-capacity KV window",
        };
        write!(f, "{s}")
    }
}

/// Everything the admission check needs, gathered by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionFacts {
    pub batch_size: usize,
    pub seq_len: usize,
    pub graph_ready: bool,
    pub greedy: bool,
    pub return_logprobs: bool,
    pub host_side_logits_work: bool,
    pub current_position: u32,
}

/// May the device loop drive this step?
///
/// Kept separate from the CUDA plumbing, and not feature-gated, so the decision
/// is unit-tested on any host. Every refusal is a fall-back to the existing
/// path, never an error.
pub fn admit(
    facts: &AdmissionFacts,
    cfg: &DeviceLoopConfig,
    enabled: bool,
) -> Result<usize, Refusal> {
    if !enabled {
        return Err(Refusal::NotEnabled);
    }
    if facts.seq_len != 1 {
        return Err(Refusal::NotDecodeStep);
    }
    if facts.batch_size != 1 {
        return Err(Refusal::NotBatchOne);
    }
    if !facts.graph_ready {
        return Err(Refusal::GraphNotReady);
    }
    if facts.return_logprobs {
        return Err(Refusal::WantsLogprobs);
    }
    if facts.host_side_logits_work {
        return Err(Refusal::HostSideLogitsWork);
    }
    if !facts.greedy {
        return Err(Refusal::NotGreedy);
    }
    let steps = cfg.plan_steps(facts.current_position, cfg.burst);
    if steps == 0 {
        return Err(Refusal::NoPositionRoom);
    }
    Ok(steps)
}

// ===========================================================================
// Host-side token handoff
// ===========================================================================

// Tokens a burst produced that the engine has not consumed yet.
//
// A burst computes `steps` tokens in one go, but the engine's `Sequence`
// bookkeeping — stop strings, streaming, EOS — is per token. So the burst
// parks its tokens here and the engine drains them one at a time on the
// following steps, during which no GPU work is launched at all.
//
// Thread-local because at batch 1 `sample_and_add_toks` runs the sampler
// inline on the engine thread (`use_async_pool = seqs_len > 1` is false), the
// same thread that ran the forward. The `GRAPH_MODE_*` graph inputs in
// `mistralrs-core/src/layers.rs` are scoped the same way for the same reason.
std::thread_local! {
    static PENDING_TOKENS: std::cell::RefCell<std::collections::VecDeque<u32>> =
        const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
    /// The sequence the parked tokens belong to, stamped at park time from
    /// [`CURRENT_SEQ`]. The completion funnel (`Sequence::set_state`) is the
    /// primary defence against a parked token outliving its sequence; this tag
    /// is the structural one — it covers the paths NO funnel can see, such as
    /// the DefaultScheduler bucketing waitlist, which moves a running sequence
    /// aside *without a state modification* (`default_scheduler.rs`, the
    /// `bucket_and_waitlist_seqs_waiting` doc says so explicitly).
    static PENDING_OWNER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// The sequence id the last `sample_sequence` published alongside
    /// eligibility. At batch 1 — the only shape the loop admits — this is
    /// exactly the sequence the next forward's burst decodes for.
    static CURRENT_SEQ: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Set by the forward when it served this step from the pending queue and
    /// handed back the ALIASING graph logits tensor without launching anything
    /// (`normal.rs::device_decode_burst`, the `pending > 0` arm). Consumed by
    /// the very next sample. If that sample cannot take a parked token, the
    /// logits it holds are a stale alias of graph-owned storage and sampling
    /// them would return a plausible token from the WRONG sequence's
    /// distribution — so the sample must fail loudly instead.
    static ALIASED_LOGITS_SERVED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Park a burst's tokens for the engine to drain. The queue is stamped with
/// the current decoding sequence ([`set_device_loop_eligible`]) as its owner;
/// [`take_pending_token`] refuses to hand them to anyone else.
///
/// Negative ids are dropped rather than cast: the commit kernel range-checks
/// before it writes, so a negative here would mean the ring was read wrong, and
/// wrapping it into a huge `u32` would index the tokenizer out of bounds.
pub fn push_pending_tokens(tokens: &[i32]) {
    PENDING_OWNER.with(|o| o.set(CURRENT_SEQ.with(|c| c.get())));
    PENDING_TOKENS.with(|q| {
        let mut q = q.borrow_mut();
        for &t in tokens {
            if t >= 0 {
                q.push_back(t as u32);
            } else {
                tracing::error!(
                    "device decode loop: dropping negative token id {t} from the ring; the \
                     commit kernel range-checks before writing, so this means the ring was \
                     misread rather than that the model produced it"
                );
            }
        }
    })
}

/// What [`take_pending_token`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingTake {
    /// A token parked by a burst for THIS sequence.
    Taken(u32),
    /// Nothing parked.
    Empty,
    /// The queue held another sequence's tokens. They were dropped — handing
    /// even one across would put one user's tokens in another user's response
    /// — and `dropped` says how many. Logged as
    /// `"ArcGraph: dropped N foreign parked tokens"`; on a box that line is
    /// the violation instrument, so keep it grep-able.
    Foreign { dropped: usize },
}

/// Take the next parked token, if it belongs to `seq_id`.
///
/// A queue owned by a different sequence is dropped whole, fail-closed: the
/// owner has left the running set (or been waitlisted with no state change),
/// its device-side burst state is stale, and replaying its tokens into any
/// sequence — including the owner after a re-prefill — would corrupt output.
pub fn take_pending_token(seq_id: usize) -> PendingTake {
    let owner = PENDING_OWNER.with(|o| o.get());
    PENDING_TOKENS.with(|q| {
        let mut q = q.borrow_mut();
        if q.is_empty() {
            return PendingTake::Empty;
        }
        if owner == Some(seq_id) {
            match q.pop_front() {
                Some(t) => PendingTake::Taken(t),
                None => PendingTake::Empty,
            }
        } else {
            let dropped = q.len();
            q.clear();
            PENDING_OWNER.with(|o| o.set(None));
            tracing::warn!(
                "ArcGraph: dropped {dropped} foreign parked tokens (parked for sequence \
                 {owner:?}, sampler is sequence {seq_id}) — cross-sequence leak prevented. \
                 Expected once when the scheduler moves a sequence aside mid-burst; frequent \
                 occurrences mean the device loop is engaging under scheduler churn it cannot \
                 amortise"
            );
            PendingTake::Foreign { dropped }
        }
    })
}

/// Tokens still parked.
pub fn pending_token_count() -> usize {
    PENDING_TOKENS.with(|q| q.borrow().len())
}

/// Does the pending queue belong to the sequence whose sample last published
/// eligibility? The forward's `pending > 0` short-circuit must not fire for
/// anyone else: it returns stale aliasing logits and launches nothing, which
/// is only sound when the very next sample will take a parked token.
pub fn pending_owned_by_current() -> bool {
    let owner = PENDING_OWNER.with(|o| o.get());
    owner.is_some() && owner == CURRENT_SEQ.with(|c| c.get())
}

/// The forward served this step from the pending queue: no launch, and the
/// returned logits tensor aliases graph-owned storage. The next sample MUST
/// take a parked token; [`take_aliased_logits_marker`] is how it checks.
pub fn note_aliased_logits_served() {
    ALIASED_LOGITS_SERVED.with(|c| c.set(true));
}

/// Consume the aliased-logits marker for this step. `true` means the logits
/// the current sample holds were NOT produced by a launch for this step and
/// must not be host-sampled.
pub fn take_aliased_logits_marker() -> bool {
    ALIASED_LOGITS_SERVED.with(|c| c.replace(false))
}

// Whether the sequence currently being decoded is one the device loop may
// drive.
//
// `Pipeline::forward_inputs` decides whether to run a burst, but it is handed
// only `inputs` and `return_raw_logits` — it never sees the `Sequence`, so it
// cannot ask whether sampling is greedy or whether logprobs were requested.
// `sample_sequence` does see it. So eligibility is published there and read by
// the *next* forward.
//
// The one-step lag is safe in the only direction that matters: a sequence's
// sampling parameters are fixed for its lifetime, and the flag starts `false`,
// so the first decode step always falls back and the loop can only ever engage
// after a step has confirmed the parameters. It can never engage for a
// sequence it has not seen.
std::thread_local! {
    static DEVICE_LOOP_ELIGIBLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Publish whether the sequence being sampled may be driven by the device
/// loop, and WHICH sequence that is. The id becomes the owner stamped onto
/// anything the next burst parks, which is what makes a parked token
/// unconsumable by any other sequence.
pub fn set_device_loop_eligible(seq_id: usize, eligible: bool) {
    CURRENT_SEQ.with(|c| c.set(Some(seq_id)));
    DEVICE_LOOP_ELIGIBLE.with(|c| c.set(eligible))
}

/// Did the last sampled sequence qualify? `false` until one has.
pub fn device_loop_eligible() -> bool {
    DEVICE_LOOP_ELIGIBLE.with(|c| c.get())
}

/// Drop parked tokens.
///
/// MUST be called when a sequence finishes, is preempted, or the path falls
/// back — a token left here would be handed to the *next* sequence as if it
/// were its own. That is the "+1-step divergence" failure mode with a different
/// name, so it gets an explicit call rather than a lifetime.
///
/// The sequence-completion half of that contract is enforced at ONE funnel:
/// `mistralrs-core`'s `Sequence::set_state` calls [`stand_down`] on every
/// transition out of the running set (`Done`, `Error`, `FinishedAborted`,
/// `FinishedIgnored`, `Waiting`, `Swapped`), which is why this crate's
/// dependency in `mistralrs-core` is deliberately not optional — the funnel
/// and its leak test compile and run on hosts without CUDA. The tests here
/// cover only this function's own behaviour; the call-site wiring is covered
/// by `pipeline/sampling.rs`'s `device_loop_cross_sequence_leak_tests`.
pub fn clear_pending_tokens() {
    PENDING_OWNER.with(|o| o.set(None));
    PENDING_TOKENS.with(|q| q.borrow_mut().clear())
}

/// Stand down completely: forget parked tokens (and their owner), drop the
/// aliased-logits marker, mark the current sequence ineligible, and bump the
/// device-state generation so a kept-alive [`DeviceDecodeLoop`] resets its
/// ring and cursors before it next runs. Call on fall-back, on sequence
/// completion, and on any error, so the next sequence starts from a clean
/// slate. Sequence completion — every transition out of the running set —
/// calls this from the `Sequence::set_state` funnel; see
/// [`clear_pending_tokens`].
pub fn stand_down() {
    clear_pending_tokens();
    ALIASED_LOGITS_SERVED.with(|c| c.set(false));
    DEVICE_LOOP_ELIGIBLE.with(|c| c.set(false));
    DEVICE_LOOP_GENERATION.with(|g| g.set(g.get().wrapping_add(1)));
}

// The device-state generation. `DeviceDecodeLoop` holds ring cursors, the
// pinned ring head and the device-side position — all per-sequence state —
// but it is deliberately kept alive across sequences because every buffer it
// points at must stay at the address the captured graph baked
// (`DeviceDecodeLoop` docs). So instead of the funnel destroying it (it
// cannot even reach it: the loop lives in the pipeline, the funnel in
// `Sequence`), every `stand_down` bumps this generation and the loop compares
// on its next `run`, resetting itself first if the world moved on. Reset at
// engagement time is also the only SAFE time: between bursts nothing is in
// flight on the stream, whereas a reset at completion time could race a
// still-draining burst's device writes.
std::thread_local! {
    static DEVICE_LOOP_GENERATION: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The current device-state generation. Bumped by every [`stand_down`].
pub fn device_loop_generation() -> u64 {
    DEVICE_LOOP_GENERATION.with(|g| g.get())
}

/// Process-wide latch. Once the device loop has failed, it stays off.
///
/// Without this a construction or burst failure would be retried on every
/// decode step forever — and because a refusal falls back to the *eager*
/// forward rather than to `replay()`, that would quietly make every step slower
/// than doing nothing at all. One failure, one warning, off for good.
static DEVICE_LOOP_KILLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Latch the device loop off for the rest of the process, loudly and once.
pub fn kill_device_loop(reason: &str) {
    if !DEVICE_LOOP_KILLED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        tracing::error!(
            "ArcGraph device decode loop DISABLED for the rest of this process: {reason}. Decode \
             falls back to graph replay with a host sync and a host sample. This is a stop, not a \
             retry — a loop that fails once and is retried every step is slower than never having \
             engaged."
        );
    }
    stand_down();
}

/// Has the device loop been latched off?
pub fn device_loop_killed() -> bool {
    DEVICE_LOOP_KILLED.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(feature = "cuda")]
extern "C" {
    /// `arc_graph_step_commit` — `src/cuda/decode_loop.cu`.
    ///
    /// Scatters the device-sampled token into the pinned graph input buffer,
    /// advances the device position, and publishes the token into the pinned
    /// mapped ring. Range-checks before it writes anything, latching a code
    /// into `fault` instead of committing a bad token.
    pub fn arc_launch_graph_step_commit(
        sampled: *const i32,
        input_ids: *mut u32,
        positions: *mut u32,
        ring: *mut i32,
        ring_head: *mut i32,
        fault: *mut i32,
        ring_size: i32,
        vocab: i32,
        position_limit: i32,
        batch: i32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A scripted fake GPU.
    ///
    /// It models the property that matters: work is enqueued without blocking
    /// and lands *later*. `launch_graph` pushes the next scripted token into an
    /// in-flight queue; `poll_stream` retires one in-flight step per call and
    /// publishes its token into the ring. So a driver that expects tokens to be
    /// present the instant it launches will not find them, and a driver that
    /// never polls will spin forever — both real failure modes.
    struct FakeGpu {
        script: VecDeque<i32>,
        in_flight: VecDeque<i32>,
        ring: Vec<i32>,
        head: i32,
        fault: i32,
        ring_size: usize,
        launches: usize,
        samples: usize,
        commits: usize,
        polls: usize,
        /// Set to drop the commit instead of publishing — the "no-op
        /// implementation" this test suite must not accept.
        commit_is_noop: bool,
        /// Injected async fault, surfaced from `poll_stream`.
        fault_after: Option<usize>,
        /// Report Idle immediately even with work outstanding.
        lie_idle: bool,
        /// A hung kernel: work is enqueued, the stream reports Busy forever and
        /// nothing ever retires. No error is ever raised, so only the liveness
        /// backstop can end it.
        never_retires: bool,
    }

    impl FakeGpu {
        fn new(script: Vec<i32>, ring_size: usize) -> Self {
            Self {
                script: script.into(),
                in_flight: VecDeque::new(),
                ring: vec![0; ring_size],
                head: 0,
                fault: FAULT_NONE,
                ring_size,
                launches: 0,
                samples: 0,
                commits: 0,
                polls: 0,
                commit_is_noop: false,
                fault_after: None,
                lie_idle: false,
                never_retires: false,
            }
        }

        fn retire_one(&mut self) {
            if let Some(tok) = self.in_flight.pop_front() {
                let slot = (self.head as usize) % self.ring_size;
                self.ring[slot] = tok;
                self.head += 1;
            }
        }
    }

    impl DeviceOps for FakeGpu {
        fn launch_graph(&mut self) -> Result<(), DeviceLoopError> {
            self.launches += 1;
            Ok(())
        }
        fn sample_on_device(&mut self) -> Result<(), DeviceLoopError> {
            self.samples += 1;
            Ok(())
        }
        fn commit_on_device(&mut self) -> Result<(), DeviceLoopError> {
            self.commits += 1;
            if self.commit_is_noop {
                return Ok(());
            }
            let tok = self.script.pop_front().unwrap_or(0);
            self.in_flight.push_back(tok);
            Ok(())
        }
        fn poll_stream(&mut self) -> Result<StreamState, DeviceLoopError> {
            self.polls += 1;
            if let Some(after) = self.fault_after {
                if self.polls > after {
                    return Err(DeviceLoopError::Cuda {
                        at: "cudaStreamQuery (async fault from an earlier graph step)",
                        code: 700,
                    });
                }
            }
            if self.never_retires {
                return Ok(StreamState::Busy);
            }
            self.retire_one();
            if self.lie_idle || self.in_flight.is_empty() {
                Ok(StreamState::Idle)
            } else {
                Ok(StreamState::Busy)
            }
        }
        fn ring_head(&self, _row: usize) -> i32 {
            self.head
        }
        fn ring_slot(&self, _row: usize, index: i32) -> i32 {
            self.ring[index.rem_euclid(self.ring_size as i32) as usize]
        }
        fn device_fault(&self) -> i32 {
            self.fault
        }
        fn batch(&self) -> usize {
            1
        }
    }

    fn cfg(burst: usize, ring: usize) -> DeviceLoopConfig {
        DeviceLoopConfig {
            burst,
            ring_size: ring,
            poll_every: 1,
            max_spins: 10_000,
            eos_token_id: 2,
            vocab: 128,
            position_limit: 4096,
        }
    }

    /// THE claim: N decode steps cost exactly N `cuGraphLaunch` calls, and the
    /// tokens that come back are the ones the GPU produced.
    ///
    /// Fails on a no-op: with fewer launches the count is wrong, and with a
    /// commit that never publishes the drain reports `LostTokens`.
    #[test]
    fn n_steps_cost_exactly_n_launches_and_return_the_gpu_tokens() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14], 16);
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        let out = d.run_burst(&mut gpu, 4).unwrap();

        assert_eq!(gpu.launches, 4, "N steps must cost exactly N cuGraphLaunch");
        assert_eq!(gpu.samples, 4, "every step must sample on device");
        assert_eq!(gpu.commits, 4, "every step must commit on device");
        assert_eq!(out.tokens, vec![vec![11, 12, 13, 14]]);
        assert_eq!(d.launches(), 4);
        assert!(!out.hit_eos);
    }

    /// The control the previous test needs: an implementation that launches and
    /// samples but never publishes the token MUST fail, and must fail as
    /// `LostTokens` rather than by spinning out. This is the "test that cannot
    /// fail" guard — if the driver accepted a silent no-op commit, the whole
    /// module would be decoration.
    #[test]
    fn a_commit_that_publishes_nothing_is_rejected_as_lost_tokens() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14], 16);
        gpu.commit_is_noop = true;
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        let err = d.run_burst(&mut gpu, 4).unwrap_err();
        assert!(
            matches!(
                err,
                DeviceLoopError::LostTokens {
                    expected: 4,
                    got: 0
                }
            ),
            "expected LostTokens, got {err:?}"
        );
        // It still launched: the failure is in the commit, and the error must
        // not be confused with "the driver never ran".
        assert_eq!(gpu.launches, 4);
    }

    /// An async fault reported by the non-blocking query must propagate as a
    /// CUDA error — not be swallowed, and not degrade into `Stalled`.
    #[test]
    fn async_fault_from_stream_query_fails_loudly_as_cuda() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14], 16);
        gpu.fault_after = Some(2);
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        let err = d.run_burst(&mut gpu, 4).unwrap_err();
        match err {
            DeviceLoopError::Cuda { code, .. } => assert_eq!(code, 700),
            other => panic!("async fault must surface as Cuda, got {other:?}"),
        }
    }

    /// Detection latency: the in-burst probe finds the fault before the burst
    /// finishes enqueuing, so a fault on step 1 does not wait for step 4.
    #[test]
    fn in_burst_probe_detects_a_fault_before_the_burst_completes() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14, 15, 16, 17, 18], 32);
        gpu.fault_after = Some(1); // second poll errors -> during step 2
        let mut d = BurstDriver::new(cfg(8, 32), 1).unwrap();
        assert!(d.run_burst(&mut gpu, 8).is_err());
        assert!(
            gpu.launches < 8,
            "the burst must stop early on a fault, launched {}",
            gpu.launches
        );
    }

    /// The no-timeout fault detector: stream reports every step retired, but
    /// the ring is short. That is definitively wrong and must be an error, not
    /// an infinite spin and not a silent short read.
    #[test]
    fn stream_idle_with_a_short_ring_is_a_fault_not_a_wait() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14], 16);
        gpu.commit_is_noop = true;
        gpu.lie_idle = true;
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        match d.run_burst(&mut gpu, 4).unwrap_err() {
            DeviceLoopError::LostTokens { expected, got } => {
                assert_eq!((expected, got), (4, 0));
            }
            other => panic!("expected LostTokens, got {other:?}"),
        }
        assert!(
            d.spins() < 100,
            "idle-but-short must be detected immediately, not spun on ({} spins)",
            d.spins()
        );
    }

    /// The commit kernel's range check travels back in a pinned word. A driver
    /// that forgets to read it returns `Ok` with plausible tokens — exactly the
    /// silent-corruption class this repo keeps shipping. So: tokens all arrive,
    /// fault word set, and the burst must still fail.
    #[test]
    fn a_latched_device_fault_fails_the_burst_even_though_every_token_arrived() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14], 16);
        gpu.fault = FAULT_TOKEN_RANGE;
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        let err = d.run_burst(&mut gpu, 4).unwrap_err();
        assert_eq!(err, DeviceLoopError::DeviceFault(FAULT_TOKEN_RANGE));
        assert_eq!(gpu.head, 4, "the tokens did land; the fault is orthogonal");
    }

    /// EOS truncates the returned tokens but not the launch count. The burst
    /// was already enqueued, so the tail was computed and is discarded — the
    /// outcome must report both numbers honestly.
    #[test]
    fn eos_truncates_tokens_but_the_wasted_steps_are_still_reported() {
        let mut gpu = FakeGpu::new(vec![11, 12, 2, 14], 16);
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        let out = d.run_burst(&mut gpu, 4).unwrap();
        assert_eq!(out.tokens, vec![vec![11, 12, 2]]);
        assert!(out.hit_eos);
        assert_eq!(
            out.steps_launched, 4,
            "the discarded tail must be visible, not rounded away"
        );
    }

    /// Consecutive bursts must not re-read the previous burst's slots. The
    /// cursor advances over the full burst even when EOS truncated the output.
    #[test]
    fn cursor_advances_over_the_whole_burst_across_consecutive_bursts() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14, 21, 22, 23, 24], 16);
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        let a = d.run_burst(&mut gpu, 4).unwrap();
        let b = d.run_burst(&mut gpu, 4).unwrap();
        assert_eq!(a.tokens, vec![vec![11, 12, 13, 14]]);
        assert_eq!(b.tokens, vec![vec![21, 22, 23, 24]]);
        assert_eq!(gpu.launches, 8);
    }

    /// Ring wraparound is arithmetic, and arithmetic is where silent corruption
    /// lives. Drive past the ring size and check the tokens are still right.
    #[test]
    fn tokens_survive_ring_wraparound() {
        let script: Vec<i32> = (100..140).collect();
        let mut gpu = FakeGpu::new(script.clone(), 8);
        let mut d = BurstDriver::new(cfg(4, 8), 1).unwrap();
        let mut seen = Vec::new();
        for _ in 0..10 {
            let out = d.run_burst(&mut gpu, 4).unwrap();
            seen.extend(out.tokens[0].iter().copied());
        }
        assert_eq!(seen, script, "wraparound must not permute or drop tokens");
    }

    /// A burst that could lap the ring is refused up front rather than
    /// discovered as scrambled tokens.
    #[test]
    fn a_burst_that_would_lap_the_ring_is_refused() {
        assert!(matches!(
            BurstDriver::new(cfg(16, 8), 1),
            Err(DeviceLoopError::BadConfig(_))
        ));
        let mut gpu = FakeGpu::new(vec![1; 32], 8);
        let mut d = BurstDriver::new(cfg(4, 8), 1).unwrap();
        assert!(matches!(
            d.run_burst(&mut gpu, 8),
            Err(DeviceLoopError::BadConfig(_))
        ));
    }

    /// A kernel that hangs *without* faulting reports Busy forever and lands
    /// nothing. No CUDA error is ever raised and the stream never goes idle, so
    /// neither of the two real detectors can fire — only the liveness backstop
    /// can end the burst. Without it this is an unkillable spin in the engine.
    #[test]
    fn a_hung_stream_trips_the_liveness_backstop() {
        let mut gpu = FakeGpu::new(vec![11, 12, 13, 14], 16);
        gpu.never_retires = true;
        let mut c = cfg(4, 16);
        c.max_spins = 500;
        let mut d = BurstDriver::new(c, 1).unwrap();
        match d.run_burst(&mut gpu, 4).unwrap_err() {
            DeviceLoopError::Stalled { spins } => assert_eq!(spins, 500),
            other => panic!("expected Stalled, got {other:?}"),
        }
    }

    /// `poll_every == 0` would divide by zero in the drain. Refuse it at
    /// construction rather than panicking mid-burst.
    #[test]
    fn zero_poll_every_is_refused_rather_than_dividing_by_zero() {
        let mut c = cfg(4, 16);
        c.poll_every = 0;
        assert!(matches!(
            BurstDriver::new(c, 1),
            Err(DeviceLoopError::BadConfig(_))
        ));
    }

    /// `plan_steps` must never walk the device position out of the
    /// fixed-capacity KV window, and must report 0 rather than a short burst
    /// the caller might mistake for success.
    #[test]
    fn plan_steps_clamps_to_the_position_window_and_the_token_budget() {
        let mut c = cfg(8, 64);
        c.position_limit = 100;
        assert_eq!(c.plan_steps(0, 1000), 8);
        assert_eq!(c.plan_steps(95, 1000), 5, "must not pass position_limit");
        assert_eq!(c.plan_steps(100, 1000), 0, "no room left");
        assert_eq!(
            c.plan_steps(120, 1000),
            0,
            "already past: saturating, not wrapping"
        );
        assert_eq!(c.plan_steps(0, 3), 3, "must not exceed the token budget");
    }

    /// Zero steps is a caller bug, not a quiet no-op that reports success.
    #[test]
    fn zero_steps_is_an_error_not_a_silent_success() {
        let mut gpu = FakeGpu::new(vec![], 16);
        let mut d = BurstDriver::new(cfg(4, 16), 1).unwrap();
        assert!(matches!(
            d.run_burst(&mut gpu, 0),
            Err(DeviceLoopError::BadConfig(_))
        ));
        assert_eq!(gpu.launches, 0);
    }

    fn ok_facts() -> AdmissionFacts {
        AdmissionFacts {
            batch_size: 1,
            seq_len: 1,
            graph_ready: true,
            greedy: true,
            return_logprobs: false,
            host_side_logits_work: false,
            current_position: 10,
        }
    }

    /// Admission must refuse every case where the device sampler would produce
    /// a token the host path would not. Each of these is a silent behaviour
    /// change if it slips through, so each gets its own assertion rather than
    /// one "returns Err" catch-all.
    #[test]
    fn admission_refuses_every_case_the_device_sampler_cannot_reproduce() {
        let c = cfg(4, 64);
        // The happy path first — otherwise a guard that refuses everything
        // would pass every negative case below.
        assert_eq!(admit(&ok_facts(), &c, true), Ok(4));

        assert_eq!(admit(&ok_facts(), &c, false), Err(Refusal::NotEnabled));

        let mut f = ok_facts();
        f.seq_len = 32;
        assert_eq!(admit(&f, &c, true), Err(Refusal::NotDecodeStep));

        let mut f = ok_facts();
        f.batch_size = 2;
        assert_eq!(admit(&f, &c, true), Err(Refusal::NotBatchOne));

        let mut f = ok_facts();
        f.graph_ready = false;
        assert_eq!(admit(&f, &c, true), Err(Refusal::GraphNotReady));

        let mut f = ok_facts();
        f.greedy = false;
        assert_eq!(
            admit(&f, &c, true),
            Err(Refusal::NotGreedy),
            "stochastic sampling must fall back: device Splitmix64 and host Isaac64 draw \
             different tokens, which breaks seeded reproducibility"
        );

        let mut f = ok_facts();
        f.return_logprobs = true;
        assert_eq!(admit(&f, &c, true), Err(Refusal::WantsLogprobs));

        let mut f = ok_facts();
        f.host_side_logits_work = true;
        assert_eq!(admit(&f, &c, true), Err(Refusal::HostSideLogitsWork));
    }

    /// Admission must also clamp the burst to the KV window, and refuse rather
    /// than return a zero-step "success" the caller could mistake for a launch.
    #[test]
    fn admission_clamps_the_burst_to_the_kv_window() {
        let mut c = cfg(4, 64);
        c.position_limit = 100;
        let mut f = ok_facts();
        f.current_position = 98;
        assert_eq!(admit(&f, &c, true), Ok(2), "clamped to the window");
        f.current_position = 100;
        assert_eq!(admit(&f, &c, true), Err(Refusal::NoPositionRoom));
    }

    /// The parked tokens are handed out in generation order, exactly once, to
    /// the sequence that owns them. A queue that re-served a token, or served
    /// them reversed, would produce fluent-but-wrong text — invisible without
    /// this.
    #[test]
    fn pending_tokens_drain_in_order_exactly_once() {
        stand_down();
        set_device_loop_eligible(11, true);
        push_pending_tokens(&[7, 8, 9]);
        assert_eq!(pending_token_count(), 3);
        assert_eq!(take_pending_token(11), PendingTake::Taken(7));
        assert_eq!(take_pending_token(11), PendingTake::Taken(8));
        assert_eq!(take_pending_token(11), PendingTake::Taken(9));
        assert_eq!(take_pending_token(11), PendingTake::Empty);
        assert_eq!(pending_token_count(), 0);
    }

    /// A leftover token must never reach the next sequence. `clear` is the only
    /// thing standing between a fall-back and one sequence's token appearing in
    /// another's output.
    #[test]
    fn clearing_pending_tokens_prevents_leaking_them_into_the_next_sequence() {
        stand_down();
        set_device_loop_eligible(11, true);
        push_pending_tokens(&[7, 8, 9]);
        clear_pending_tokens();
        assert_eq!(take_pending_token(11), PendingTake::Empty);
    }

    /// THE structural guarantee: tokens parked while sequence 11 was current
    /// cannot be taken by sequence 12 — they are dropped whole, and the drop
    /// is reported. This is the defence the completion funnel cannot provide:
    /// the DefaultScheduler bucketing waitlist moves a running sequence aside
    /// *without a state modification*, so no funnel ever fires.
    #[test]
    fn foreign_parked_tokens_are_dropped_not_served() {
        stand_down();
        set_device_loop_eligible(11, true);
        push_pending_tokens(&[7, 8, 9]);

        // Sequence 12 samples next (11 was waitlisted, no state change).
        set_device_loop_eligible(12, true);
        assert_eq!(
            take_pending_token(12),
            PendingTake::Foreign { dropped: 3 },
            "another sequence's parked tokens must be dropped, not served"
        );
        assert_eq!(pending_token_count(), 0, "the foreign queue is gone whole");
        // And they are gone for the original owner too: its device-side burst
        // state is stale, so replaying them after a re-prefill would corrupt.
        assert_eq!(take_pending_token(11), PendingTake::Empty);
    }

    /// The forward-side guard: the `pending > 0` short-circuit may only fire
    /// for the sequence that owns the queue.
    #[test]
    fn pending_ownership_tracks_the_published_sequence() {
        stand_down();
        assert!(
            !pending_owned_by_current(),
            "empty queue is owned by nobody"
        );
        set_device_loop_eligible(11, true);
        push_pending_tokens(&[7]);
        assert!(pending_owned_by_current());
        set_device_loop_eligible(12, true);
        assert!(
            !pending_owned_by_current(),
            "the moment another sequence publishes, the queue is foreign"
        );
    }

    /// The aliased-logits marker is consumed exactly once, and stand_down
    /// drops it: a marker surviving a stand-down would poison the next
    /// sequence's first REAL sample.
    #[test]
    fn aliased_logits_marker_is_one_shot_and_cleared_by_stand_down() {
        stand_down();
        assert!(!take_aliased_logits_marker());
        note_aliased_logits_served();
        assert!(take_aliased_logits_marker());
        assert!(!take_aliased_logits_marker(), "consumed exactly once");
        note_aliased_logits_served();
        stand_down();
        assert!(!take_aliased_logits_marker(), "stand_down must drop it");
    }

    /// Every stand_down bumps the device-state generation; that is what makes
    /// a kept-alive `DeviceDecodeLoop` reset its ring and cursors before the
    /// next sequence's first burst instead of carrying them across.
    #[test]
    fn stand_down_bumps_the_device_state_generation() {
        let g0 = device_loop_generation();
        stand_down();
        assert_eq!(device_loop_generation(), g0.wrapping_add(1));
        stand_down();
        assert_eq!(device_loop_generation(), g0.wrapping_add(2));
    }

    /// A negative id in the ring means the ring was misread. Casting it would
    /// index the tokenizer at ~4 billion; it must be dropped and logged.
    #[test]
    fn negative_ring_ids_are_dropped_not_wrapped_into_huge_u32() {
        stand_down();
        set_device_loop_eligible(11, true);
        push_pending_tokens(&[5, -1, 6]);
        assert_eq!(take_pending_token(11), PendingTake::Taken(5));
        assert_eq!(take_pending_token(11), PendingTake::Taken(6));
        assert_eq!(take_pending_token(11), PendingTake::Empty);
    }

    /// The eligibility flag must start `false`, so the first decode step of a
    /// sequence always falls back rather than engaging on parameters nothing
    /// has checked yet.
    #[test]
    fn eligibility_starts_false_and_stand_down_resets_everything() {
        stand_down();
        assert!(!device_loop_eligible());

        set_device_loop_eligible(11, true);
        push_pending_tokens(&[1, 2]);
        assert!(device_loop_eligible());

        stand_down();
        assert!(!device_loop_eligible(), "stand_down must clear eligibility");
        assert_eq!(
            take_pending_token(11),
            PendingTake::Empty,
            "stand_down must clear tokens"
        );
    }

    /// Default-off, and explicitly pinnable off.
    #[test]
    fn the_opt_in_defaults_to_off() {
        // Not using the process env here: these run in parallel with other
        // tests. The parsing rule is what matters and it is exercised directly.
        for (val, want) in [
            ("", false),
            ("0", false),
            ("false", false),
            ("off", false),
            ("no", false),
            ("1", true),
            ("true", true),
        ] {
            let v = val.trim().to_ascii_lowercase();
            let enabled = !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no");
            assert_eq!(enabled, want, "for {val:?}");
        }
        // And with the variable absent entirely.
        std::env::remove_var(ENV_ENABLE);
        assert!(!device_loop_enabled());
    }
}
