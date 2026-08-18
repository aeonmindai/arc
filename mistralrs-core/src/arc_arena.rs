//! Parent system: ArcInfer / ArcGraph
//!
//! **The static decode arena** — takes the memory allocator and the per-op
//! shape upload out of the decode loop.
//!
//! # What it is for
//!
//! Measured on `arc-v4-stack` (H200, SHA `05af600e7`) from the nsys trace, per
//! decode token at B=1:
//!
//! | host call | per token |
//! |---|---|
//! | `cuMemAllocAsync` | 11,408 |
//! | `cuMemFreeAsync` | 11,408 |
//! | `cuMemcpyHtoDAsync` | 2,811 (mean 72 B) |
//!
//! That is 1.25 allocations for every one of the 7,916 kernel launches. All of
//! it is host work inside the step, and 49% of the step is the GPU idle waiting
//! on the host. At B=1 the shapes are static across steps, so none of it needs
//! to happen more than once.
//!
//! Two mechanisms, both implemented in the candle fork
//! (`candle_core::cuda::CudaDevice`) and merely switched on here:
//!
//! 1. **Arena** — freed device buffers go to a size-bucketed free list instead
//!    of `cuMemFreeAsync`, and allocations are served from it instead of
//!    `cuMemAllocAsync`. Steady state issues neither.
//! 2. **Layout interning** — a dims/strides upload whose bytes have been seen
//!    before reuses the device buffer that already holds them, eliding the
//!    `cuMemcpyHtoDAsync` entirely.
//!
//! # Why it must be switched on *here* and not at device creation
//!
//! Model load allocates and frees hundreds of large transient buffers. An arena
//! enabled during load hoards every one of them. It is enabled on the first
//! single-token forward — i.e. once decode has started and the shapes have
//! stopped changing.
//!
//! # Environment
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `ARC_ARENA` | off | master switch |
//! | `ARC_ARENA_CAP_MB` | 8192 | hard cap on arena bytes; puts past it are refused |
//! | `ARC_ARENA_INTERN` | 16384 | max interned layouts (0 disables interning) |
//! | `ARC_ARENA_STATS` | 0 | log per-step counters every N steps (0 = off) |
//! | `ARC_ARENA_WARMUP` | 3 | steps to run before the counters are zeroed |

#[cfg(feature = "cuda")]
use candle_core::Device;

#[cfg(feature = "cuda")]
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Per-process decode-step counter, used to zero the stats after warmup and to
/// pace the stats log.
#[cfg(feature = "cuda")]
static STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "cuda")]
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ARC_ARENA").is_some())
}

/// Call at the top of every single-token (decode) forward.
///
/// Idempotent and cheap: after the first call it is an atomic increment plus a
/// modulo. Does nothing at all unless `ARC_ARENA` is set, so the default build
/// is byte-identical to before.
#[cfg(feature = "cuda")]
pub fn on_decode_step(device: &Device) {
    if !enabled() {
        return;
    }
    let Device::Cuda(cd) = device else {
        return;
    };

    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let cap_mb = env_usize("ARC_ARENA_CAP_MB", 8192) as u64;
        let intern = env_usize("ARC_ARENA_INTERN", 16384);
        cd.set_alloc_cache_cap_bytes(cap_mb * 1024 * 1024);
        cd.set_info_intern_max_entries(intern);
        cd.set_alloc_cache_enabled(true);
        tracing::info!(
            "ARC arena: ON (cap {cap_mb} MB, intern {intern} layouts). \
             Target: 0 cuMemAllocAsync and 0 dims/strides H2D per step."
        );
    });

    let n = STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let warmup = env_usize("ARC_ARENA_WARMUP", 3) as u64;

    // Zero the counters once the arena has reached steady state, so the rate
    // that gets reported is the steady-state rate and not an average that the
    // cold-start misses have polluted. The warmup steps themselves are still
    // reported (below) rather than hidden.
    if n == warmup {
        let s = cd.alloc_stats();
        tracing::info!(
            "ARC arena: warmup done after {warmup} steps -- \
             {} driver allocs, {} bytes, {} interned layouts. Counters zeroed.",
            s.driver_allocs,
            s.driver_alloc_bytes,
            s.intern_entries
        );
        cd.reset_alloc_stats();
        return;
    }

    let every = env_usize("ARC_ARENA_STATS", 0) as u64;
    if every > 0 && n > warmup && (n - warmup) % every == 0 {
        log_stats(cd, n - warmup);
    }
}

#[cfg(feature = "cuda")]
fn log_stats(cd: &candle_core::CudaDevice, steps: u64) {
    let s = cd.alloc_stats();
    let acct = cd.verify_alloc_accounting();
    let d = steps.max(1) as f64;
    tracing::info!(
        "ARC arena @ {steps} steps/token-rates: \
         driver_allocs {:.1} | driver_frees {:.1} | cache_hits {:.1} | \
         info_uploads {:.1} | info_hits {:.1} | data_htod {:.1} || \
         arena {:.1} MB (hwm {:.1} MB, {} bufs) | interned {} layouts / {} B | \
         puts_refused {} | accounting {}",
        s.driver_allocs as f64 / d,
        s.driver_frees as f64 / d,
        s.cache_hits as f64 / d,
        s.info_uploads as f64 / d,
        s.info_hits as f64 / d,
        s.data_htod as f64 / d,
        s.cached_bytes as f64 / 1048576.0,
        s.cached_bytes_hwm as f64 / 1048576.0,
        acct.buffers,
        s.intern_entries,
        s.intern_bytes,
        s.cache_puts_refused,
        if acct.agrees() {
            "OK".to_string()
        } else {
            format!("MISMATCH running={} recomputed={}", acct.running, acct.recomputed)
        }
    );
}

/// Emit the final report and assert that work actually happened.
///
/// Returns `Err` with a human-readable reason when the run cannot be trusted,
/// so a caller can exit **2** (environment/instrument failure) rather than
/// reporting a green that proves nothing. A green here means: the arena was
/// engaged (`cache_hits` is large), the counters are self-consistent, and the
/// steady-state driver-allocation rate is what it claims to be.
#[cfg(feature = "cuda")]
pub fn final_report(device: &Device) -> Result<String, String> {
    let Device::Cuda(cd) = device else {
        return Err("not a CUDA device".to_string());
    };
    let steps = STEPS
        .load(std::sync::atomic::Ordering::Relaxed)
        .saturating_sub(env_usize("ARC_ARENA_WARMUP", 3) as u64);
    if steps == 0 {
        return Err("no decode steps recorded -- nothing was measured".to_string());
    }
    let s = cd.alloc_stats();
    let acct = cd.verify_alloc_accounting();
    if !acct.agrees() {
        return Err(format!(
            "allocator accounting disagrees: running={} recomputed={} -- \
             the counters cannot be trusted, so neither can the result",
            acct.running, acct.recomputed
        ));
    }
    // D18/D32: prove engagement. With the arena on, allocation requests do not
    // vanish -- they turn into cache hits. Zero hits means the arena was never
    // reached and a zero alloc rate would be meaningless.
    if enabled() && s.cache_hits == 0 {
        return Err(
            "arena is enabled but served ZERO allocations -- it was never engaged, \
             so a low allocation count proves nothing"
                .to_string(),
        );
    }
    let d = steps as f64;
    Ok(format!(
        "steps={steps} allocs/token={:.1} frees/token={:.1} cache_hits/token={:.1} \
         info_uploads/token={:.1} info_hits/token={:.1} data_htod/token={:.1} \
         arena_hwm_MB={:.1} interned={} accounting=OK",
        s.driver_allocs as f64 / d,
        s.driver_frees as f64 / d,
        s.cache_hits as f64 / d,
        s.info_uploads as f64 / d,
        s.info_hits as f64 / d,
        s.data_htod as f64 / d,
        s.cached_bytes_hwm as f64 / 1048576.0,
        s.intern_entries,
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn enabled() -> bool {
    false
}

/// Bit-exact fingerprint of one step's logits, gated by `ARC_LOGITS_HASH`.
///
/// This is the correctness instrument for the arena: a reused buffer that is
/// still live produces *wrong numbers*, not a crash, and "it still runs" would
/// not notice. Two quantities are emitted, not one — an FNV-1a over the raw
/// IEEE bits and a plain sum. They are sensitive to different things (the hash
/// to any single-bit change anywhere, the sum to magnitude), and D33 wants a
/// second quantity whose disagreement with the first would be impossible if
/// both were right.
///
/// Costs a device→host copy per step, so it is for correctness runs only and
/// must be off for any timing number.
#[cfg(feature = "cuda")]
pub fn logits_fingerprint(logits: &candle_core::Tensor) {
    if std::env::var_os("ARC_LOGITS_HASH").is_none() {
        return;
    }
    let step = STEPS.load(std::sync::atomic::Ordering::Relaxed);
    let v = match logits
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
    {
        Ok(v) => v,
        Err(e) => {
            // An instrument that silently does nothing is worse than none.
            tracing::error!("ARC logits-hash FAILED at step {step}: {e}");
            return;
        }
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in &v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    let sum: f64 = v.iter().map(|x| *x as f64).sum();
    let finite = v.iter().filter(|x| x.is_finite()).count();
    tracing::info!(
        "ARC-LOGITS step={step} n={} finite={finite} fnv=0x{h:016x} sum={sum:.9e}",
        v.len()
    );
}
