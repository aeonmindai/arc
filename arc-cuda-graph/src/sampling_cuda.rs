//! CUDA-backed sampler matching `sampling_cpu.rs` bit-for-bit (see
//! `cuda/sampling_kernel.cu` for the kernel contract and tiebreak note).
//!
//! Layout:
//! - `SamplingParams` is the POD shared with CUDA — same layout as the
//!   `arc_sampler::SamplingParams` struct in `sampling_kernel.cu`.
//! - `CudaSampler` owns a per-batch RNG state buffer and a scratch buffer
//!   for the softmax pass; both are allocated once on construction and
//!   reused across decode steps. There is no per-token host alloc.
//!
//! Usage from the engine:
//!     let mut sampler = CudaSampler::new(device, batch, vocab, dtype, seed)?;
//!     // Each step:
//!     sampler.sample(&logits, freq_counts.as_deref(), cfg, &mut token_ids)?;
//!
//! Where `logits` is a Candle CUDA tensor of shape [batch, vocab], `dtype`
//! is one of {F32, BF16, F16}, `freq_counts` is optional u32 tensor
//! [batch, vocab], and `token_ids` is an i32 tensor [batch] on the same
//! device. The kernel does its work on the device-default stream.
//!
//! ## Bit-exactness vs `sampling_cpu`
//!
//! The kernel matches `sampling_cpu::sample` whenever no two kept
//! probabilities tie. For ties, the CPU uses Rust's `sort_unstable_by`
//! (pdqsort) which leaves the order among `Ordering::Equal` elements
//! unspecified, whereas the kernel always pulls the lowest vocab index
//! first. The `gpu_simulator_matches_cpu_for_diverse_inputs` test exercises
//! random inputs to confirm the algorithms agree for the tied-free case.

use crate::sampling_cpu::{apply_penalties, apply_temperature, greedy_argmax, SamplingConfig};

#[cfg(feature = "cuda")]
use candle_core::{cuda::cudarc::driver::sys::CUstream, DType, Device, Tensor};

/// POD passed to the CUDA kernel. Field layout MUST match
/// `arc_sampler::SamplingParams` in `cuda/sampling_kernel.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub greedy: i32,
    pub eos_token_id: i32,
}

/// Historical cap on the GPU kept-token list. The kernel previously held the
/// kept (idx, p) pairs in shared memory and clamped the kept count to 256,
/// which silently truncated large-vocab samplers when `top_k <= 0` left
/// `top_p` alone to walk a diffuse distribution. The kernel now stores the
/// keep list in global memory at full-vocab capacity (see `keep_scratch` on
/// `CudaSampler`), so there is no runtime cap. The constant remains for
/// back-compat with consumers that reference it.
pub const GPU_MAX_KEEP: usize = usize::MAX;

/// Host-side simulator of the exact algorithm `cuda/sampling_kernel.cu`
/// implements. Used by the test suite to verify the GPU algorithm produces
/// the same token as `sampling_cpu::sample` for tied-free inputs without
/// requiring a CUDA device. The math is structurally identical to the
/// kernel — same Splitmix64 step, same iterative argmax with lowest-index
/// tiebreak, same CDF walk.
pub fn gpu_algorithm_simulate(
    logits: &mut [f32],
    frequency_counts: &[u32],
    cfg: SamplingConfig,
    rng_state: &mut u64,
) -> u32 {
    if cfg.greedy {
        apply_penalties(logits, frequency_counts, cfg);
        // Lowest-index tiebreak greedy argmax — matches arc_greedy_kernel.
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx as u32;
    }
    apply_penalties(logits, frequency_counts, cfg);
    apply_temperature(logits, cfg.temperature);

    // Softmax (max-shifted).
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    // Iterative argmax → keep list, with lowest-index tiebreak (mirrors GPU).
    //
    // The kernel stores the keep list in global memory at full-vocab capacity,
    // so its effective cap matches the CPU reference: `top_k` when set, else
    // the full vocab. This was previously clamped to 256 in shared memory,
    // which silently truncated large-vocab samplers (RUN-170).
    let cap = if cfg.top_k > 0 {
        (cfg.top_k as usize).min(probs.len())
    } else {
        probs.len()
    };
    // Reference GPU_MAX_KEEP to keep the constant linked from this module
    // (avoids dead-code warnings; the constant is now a historical marker
    // documenting the previous cap).
    let _ = GPU_MAX_KEEP;
    let mut keep: Vec<(usize, f32)> = Vec::with_capacity(cap);
    let mut cum = 0.0f32;
    let mut kept_sum = 0.0f32;
    while keep.len() < cap && cum < cfg.top_p {
        let mut best_val = -f32::INFINITY;
        let mut best_idx: i32 = -1;
        for (i, &v) in probs.iter().enumerate() {
            if v > best_val || (v == best_val && (best_idx == -1 || (i as i32) < best_idx)) {
                best_val = v;
                best_idx = i as i32;
            }
        }
        if best_idx < 0 || best_val <= 0.0 {
            break;
        }
        let bi = best_idx as usize;
        keep.push((bi, best_val));
        cum += best_val;
        kept_sum += best_val;
        probs[bi] = f32::NEG_INFINITY; // tombstone
    }

    if kept_sum <= 0.0 {
        return greedy_argmax(logits);
    }

    // Splitmix64 RNG step (matches kernel `splitmix_uniform`).
    *rng_state = rng_state
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xDEADBEEFC0DECAFE);
    let mixed = (*rng_state ^ (*rng_state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let u = ((mixed >> 33) as u32) as f32 / (1u64 << 31) as f32;
    let target = u * kept_sum;

    let mut acc = 0.0;
    for (idx, p) in &keep {
        acc += *p;
        if acc >= target {
            return *idx as u32;
        }
    }
    keep.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

impl From<SamplingConfig> for SamplingParams {
    fn from(c: SamplingConfig) -> Self {
        Self {
            temperature: c.temperature,
            top_p: c.top_p,
            top_k: c.top_k,
            frequency_penalty: c.frequency_penalty,
            presence_penalty: c.presence_penalty,
            greedy: if c.greedy { 1 } else { 0 },
            eos_token_id: c.eos_token_id,
        }
    }
}

#[cfg(feature = "cuda")]
extern "C" {
    pub fn arc_launch_sampler_f32(
        logits: *const f32,
        freq_counts: *const u32, // nullable
        rng_state: *mut u64,
        token_ids: *mut i32,
        probs_scratch: *mut f32,
        keep_idx_scratch: *mut i32,
        keep_p_scratch: *mut f32,
        vocab: i32,
        batch: i32,
        cfg: SamplingParams,
        stream: CUstream,
    );

    pub fn arc_launch_sampler_bf16(
        logits: *const std::ffi::c_void,
        freq_counts: *const u32, // nullable
        rng_state: *mut u64,
        token_ids: *mut i32,
        probs_scratch: *mut f32,
        keep_idx_scratch: *mut i32,
        keep_p_scratch: *mut f32,
        vocab: i32,
        batch: i32,
        cfg: SamplingParams,
        stream: CUstream,
    );

    pub fn arc_launch_sampler_f16(
        logits: *const std::ffi::c_void,
        freq_counts: *const u32, // nullable
        rng_state: *mut u64,
        token_ids: *mut i32,
        probs_scratch: *mut f32,
        keep_idx_scratch: *mut i32,
        keep_p_scratch: *mut f32,
        vocab: i32,
        batch: i32,
        cfg: SamplingParams,
        stream: CUstream,
    );
}

/// Raw device pointer for a CUDA tensor, byte-offset aware.
///
/// `as_cuda_slice::<T>()` type-checks `T` against the storage dtype, so the
/// type argument has to follow `t.dtype()`. This used to hardcode `::<u8>()`
/// and therefore failed on every F32/BF16/U32 tensor with
/// `unexpected dtype, expected: U8, got: F32`, which took out the whole fused
/// GPU sampler. `weights::tensor_device_ptr` already dispatches correctly.
#[cfg(feature = "cuda")]
fn cuda_tensor_ptr(t: &Tensor) -> candle_core::Result<usize> {
    let t = t.contiguous()?;
    Ok(crate::weights::tensor_device_ptr(&t)? as usize)
}

#[cfg(feature = "cuda")]
fn cuda_stream(device: &Device) -> candle_core::Result<CUstream> {
    let Device::Cuda(cuda_dev) = device else {
        candle_core::bail!("CudaSampler requires CUDA device");
    };
    Ok(cuda_dev.cuda_stream().cu_stream())
}

/// CUDA-resident sampler. Owns per-batch RNG state and FP32 softmax scratch.
#[cfg(feature = "cuda")]
pub struct CudaSampler {
    device: Device,
    stream: CUstream,
    batch: usize,
    vocab: usize,
    dtype: DType,
    /// u64[batch] — Splitmix64 state per row, mutated in place per call.
    rng_state: Tensor,
    /// f32[batch, vocab] — softmax scratch, reused across calls.
    probs_scratch: Tensor,
    /// i32[batch, vocab] — global-memory keep-list indices, written by the
    /// kernel's Phase 4 (one entry per kept token). Sized to full vocab so
    /// `top_k <= 0 + top_p < 1.0` cannot truncate. See RUN-170.
    keep_idx_scratch: Tensor,
    /// f32[batch, vocab] — global-memory keep-list probabilities, paired
    /// with `keep_idx_scratch` row-wise.
    keep_p_scratch: Tensor,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaSampler {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaSampler {}

#[cfg(feature = "cuda")]
impl CudaSampler {
    /// Allocate RNG state + scratch on the device. `seed` initializes each
    /// row's state to `seed ^ (row_idx * 0xA24BAED4963EE407)` so distinct
    /// batch rows draw distinct sequences while remaining seedable.
    pub fn new(
        device: &Device,
        batch: usize,
        vocab: usize,
        dtype: DType,
        seed: u64,
    ) -> candle_core::Result<Self> {
        match dtype {
            DType::F32 | DType::BF16 | DType::F16 => {}
            other => candle_core::bail!(
                "CudaSampler: unsupported logits dtype {other:?}; expected F32/BF16/F16"
            ),
        }
        let stream = cuda_stream(device)?;

        // Build initial RNG states on the host then upload (one-shot, no hot-loop cost).
        let mut state_init: Vec<u64> = Vec::with_capacity(batch);
        for i in 0..batch {
            state_init.push(seed.wrapping_add((i as u64).wrapping_mul(0xA24BAED4963EE407)));
        }
        // Candle has no u64 dtype; pack two u32 halves into U32 [2*batch] then
        // reinterpret-cast on the GPU. We store as raw bytes via a U8 tensor.
        let bytes: Vec<u8> = state_init.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rng_state = Tensor::from_vec(bytes, (batch * 8,), device)?;

        let probs_scratch = Tensor::zeros((batch, vocab), DType::F32, device)?;
        // Keep-list scratch (RUN-170): one entry per token, sized to full vocab
        // so the kernel's iterative-argmax loop can never truncate. The keep
        // arrays are only written/read by thread 0 of each block, so global
        // memory latency is fully amortized by the parallel argmax phases.
        // We allocate an I32 buffer directly via `from_vec` because Candle's
        // `Tensor::zeros` on I32 may not be supported across all backends.
        let keep_idx_scratch = Tensor::from_vec(vec![0i32; batch * vocab], (batch, vocab), device)?;
        let keep_p_scratch = Tensor::zeros((batch, vocab), DType::F32, device)?;

        Ok(Self {
            device: device.clone(),
            stream,
            batch,
            vocab,
            dtype,
            rng_state,
            probs_scratch,
            keep_idx_scratch,
            keep_p_scratch,
        })
    }

    /// Run the sampler. Logits: [batch, vocab] of the configured dtype.
    /// `freq_counts`: optional [batch, vocab] u32 token-count tensor (one per
    /// batch row); pass `None` when no penalties are needed.
    /// `token_ids`: i32 [batch] on the same device, written in place.
    pub fn sample(
        &mut self,
        logits: &Tensor,
        freq_counts: Option<&Tensor>,
        cfg: SamplingConfig,
        token_ids: &Tensor,
    ) -> candle_core::Result<()> {
        cfg.validate().map_err(candle_core::Error::msg)?;

        // Shape & dtype guards.
        let (b, v) = match logits.dims() {
            &[b, v] => (b, v),
            other => candle_core::bail!("logits must be [batch, vocab], got {other:?}"),
        };
        if b != self.batch || v != self.vocab {
            candle_core::bail!(
                "logits shape {:?} != sampler ({},{})",
                (b, v),
                self.batch,
                self.vocab
            );
        }
        if logits.dtype() != self.dtype {
            candle_core::bail!(
                "logits dtype {:?} != sampler dtype {:?}",
                logits.dtype(),
                self.dtype
            );
        }
        if token_ids.dtype() != DType::I32 || token_ids.dims() != [self.batch] {
            candle_core::bail!("token_ids must be I32 [{}]", self.batch);
        }
        if let Some(fc) = freq_counts {
            if fc.dtype() != DType::U32 || fc.dims() != [self.batch, self.vocab] {
                candle_core::bail!("freq_counts must be U32 [{},{}]", self.batch, self.vocab);
            }
        }

        let logits_ptr = cuda_tensor_ptr(logits)? as *const std::ffi::c_void;
        let token_ptr = cuda_tensor_ptr(token_ids)? as *mut i32;
        let rng_ptr = cuda_tensor_ptr(&self.rng_state)? as *mut u64;
        let probs_ptr = cuda_tensor_ptr(&self.probs_scratch)? as *mut f32;
        let keep_idx_ptr = cuda_tensor_ptr(&self.keep_idx_scratch)? as *mut i32;
        let keep_p_ptr = cuda_tensor_ptr(&self.keep_p_scratch)? as *mut f32;
        let freq_ptr = match freq_counts {
            Some(t) => cuda_tensor_ptr(t)? as *const u32,
            None => std::ptr::null(),
        };

        let params: SamplingParams = cfg.into();
        let v_i = self.vocab as i32;
        let b_i = self.batch as i32;

        unsafe {
            match self.dtype {
                DType::F32 => arc_launch_sampler_f32(
                    logits_ptr as *const f32,
                    freq_ptr,
                    rng_ptr,
                    token_ptr,
                    probs_ptr,
                    keep_idx_ptr,
                    keep_p_ptr,
                    v_i,
                    b_i,
                    params,
                    self.stream,
                ),
                DType::BF16 => arc_launch_sampler_bf16(
                    logits_ptr,
                    freq_ptr,
                    rng_ptr,
                    token_ptr,
                    probs_ptr,
                    keep_idx_ptr,
                    keep_p_ptr,
                    v_i,
                    b_i,
                    params,
                    self.stream,
                ),
                DType::F16 => arc_launch_sampler_f16(
                    logits_ptr,
                    freq_ptr,
                    rng_ptr,
                    token_ptr,
                    probs_ptr,
                    keep_idx_ptr,
                    keep_p_ptr,
                    v_i,
                    b_i,
                    params,
                    self.stream,
                ),
                _ => unreachable!("dtype guarded in new()"),
            }
        }
        Ok(())
    }

    /// Raw u64 pointer to the per-batch RNG state buffer. Useful for embedding
    /// the sampler inside a captured CUDA graph that wants direct pointer
    /// arguments.
    pub fn rng_state_ptr(&self) -> candle_core::Result<*mut u64> {
        Ok(cuda_tensor_ptr(&self.rng_state)? as *mut u64)
    }

    pub fn probs_scratch_ptr(&self) -> candle_core::Result<*mut f32> {
        Ok(cuda_tensor_ptr(&self.probs_scratch)? as *mut f32)
    }

    pub fn batch(&self) -> usize {
        self.batch
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirm SamplingParams ⇄ SamplingConfig conversion preserves all fields.
    #[test]
    fn params_conversion_roundtrips() {
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 50,
            frequency_penalty: 0.1,
            presence_penalty: 0.2,
            greedy: false,
            eos_token_id: 2,
        };
        let p: SamplingParams = cfg.into();
        assert_eq!(p.temperature, 0.7);
        assert_eq!(p.top_p, 0.95);
        assert_eq!(p.top_k, 50);
        assert_eq!(p.frequency_penalty, 0.1);
        assert_eq!(p.presence_penalty, 0.2);
        assert_eq!(p.greedy, 0);
        assert_eq!(p.eos_token_id, 2);
    }

    #[test]
    fn greedy_params_set_flag() {
        let cfg = SamplingConfig::greedy();
        let p: SamplingParams = cfg.into();
        assert_eq!(p.greedy, 1);
    }

    /// POD struct must match the C++ definition's size + alignment. The C++
    /// struct is: f32 + f32 + i32 + f32 + f32 + i32 + i32 = 28 bytes, 4-byte
    /// aligned. `#[repr(C)]` on the Rust side produces the same.
    #[test]
    fn sampling_params_layout_is_28_bytes() {
        assert_eq!(std::mem::size_of::<SamplingParams>(), 28);
        assert_eq!(std::mem::align_of::<SamplingParams>(), 4);
    }

    /// GPU simulator matches the CPU reference for greedy mode.
    /// Both use a strict `>` first-wins comparison; outputs must agree.
    #[test]
    fn gpu_simulator_greedy_matches_cpu() {
        let cfg = SamplingConfig::greedy();
        let counts = vec![0u32; 8];
        let logits = vec![0.0f32, 1.0, 2.0, -1.0, 5.0, 4.0, 3.0, 2.5];
        let mut cpu_l = logits.clone();
        let mut gpu_l = logits.clone();
        let mut cpu_st = 0u64;
        let mut gpu_st = 0u64;
        let cpu = crate::sampling_cpu::sample(&mut cpu_l, &counts, cfg, &mut cpu_st);
        let gpu = gpu_algorithm_simulate(&mut gpu_l, &counts, cfg, &mut gpu_st);
        assert_eq!(cpu, gpu);
    }

    /// GPU simulator + CPU reference agree on diverse non-tied inputs.
    /// We generate logits with random-ish floats (deterministic LCG) so no
    /// two probabilities exactly tie after softmax — the tied case is
    /// documented as a non-bit-exact path (see module docs).
    #[test]
    fn gpu_simulator_matches_cpu_for_diverse_inputs() {
        let cfg_temp_only = SamplingConfig {
            temperature: 0.8,
            top_p: 1.0,
            top_k: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: false,
            eos_token_id: -1,
        };
        let cfg_topp = SamplingConfig {
            temperature: 1.0,
            top_p: 0.9,
            top_k: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: false,
            eos_token_id: -1,
        };
        let cfg_topk = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 5,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: false,
            eos_token_id: -1,
        };
        let cfg_all = SamplingConfig {
            temperature: 0.7,
            top_p: 0.92,
            top_k: 40,
            frequency_penalty: 0.1,
            presence_penalty: 0.05,
            greedy: false,
            eos_token_id: -1,
        };

        let configs = [cfg_temp_only, cfg_topp, cfg_topk, cfg_all];

        // Deterministic logits-generator LCG.
        for vocab in [16usize, 64, 257, 1024] {
            let mut lcg: u64 = 0xDEADBEEFCAFE;
            let logits_base: Vec<f32> = (0..vocab)
                .map(|i| {
                    lcg = lcg
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let bits = (lcg >> 32) as u32;
                    // Map to [-4, 4) — diverse range, no exact ties.
                    let f = (bits as f32 / u32::MAX as f32) * 8.0 - 4.0;
                    // Stagger by i*1e-6 to guarantee distinct values.
                    f + (i as f32) * 1e-6
                })
                .collect();
            // Mix in some non-zero counts so penalty paths trigger.
            let mut counts = vec![0u32; vocab];
            for i in (0..vocab).step_by(7) {
                counts[i] = ((i as u32) % 3) + 1;
            }

            for (ci, cfg) in configs.iter().enumerate() {
                let seed = 0xCAFE_BABE_u64 ^ ((vocab as u64) << 8) ^ (ci as u64);
                let mut cpu_l = logits_base.clone();
                let mut gpu_l = logits_base.clone();
                let mut cpu_s = seed;
                let mut gpu_s = seed;
                let cpu_tok = crate::sampling_cpu::sample(&mut cpu_l, &counts, *cfg, &mut cpu_s);
                let gpu_tok = gpu_algorithm_simulate(&mut gpu_l, &counts, *cfg, &mut gpu_s);
                assert_eq!(
                    cpu_tok, gpu_tok,
                    "Disagreement at vocab={vocab} cfg_idx={ci}: cpu={cpu_tok} gpu={gpu_tok}"
                );
                assert_eq!(cpu_s, gpu_s, "RNG state diverged");
            }
        }
    }

    /// RUN-170 regression: large-vocab + `top_k <= 0` + `top_p < 1.0` must not
    /// truncate. The kernel previously held the keep list in a fixed
    /// shared-memory buffer of 256 entries, silently dropping mass once 256
    /// tokens had been accumulated. After the fix the keep list is global-
    /// memory backed and grows up to full vocab; the GPU simulator (which
    /// mirrors the kernel's algorithm) must agree with the CPU reference for
    /// vocabularies in the 128K range used by Kimi K2.6 / GLM / V4 Flash.
    #[test]
    fn gpu_simulator_matches_cpu_at_128k_vocab_top_k_disabled() {
        // 128K is realistic for current frontier OSS models; 257 is the
        // smallest vocab that exceeds the historical MAX_KEEP=256 cap, so
        // we include it as an extra cheap sanity check.
        for vocab in [257usize, 128 * 1024] {
            // Deterministic LCG-generated logits with a 1e-6 stagger so no
            // probabilities tie exactly (the kernel and CPU disagree on
            // ordering of tied probabilities — documented in module docs).
            let mut lcg: u64 = 0x0BADC0FFEEC0DE15u64.wrapping_add(vocab as u64);
            let logits_base: Vec<f32> = (0..vocab)
                .map(|i| {
                    lcg = lcg
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let bits = (lcg >> 32) as u32;
                    // Map to [-6, 6) — diffuse enough that top_p=0.95 keeps
                    // significantly more than 256 candidates at vocab=128K.
                    let f = (bits as f32 / u32::MAX as f32) * 12.0 - 6.0;
                    f + (i as f32) * 1e-6
                })
                .collect();
            let counts = vec![0u32; vocab];

            // Two configs that previously triggered silent truncation:
            //   1. top_k disabled + top_p=0.95 (the "natural" diffuse case)
            //   2. top_k disabled + top_p=1.0  (full distribution sampling)
            let configs = [
                SamplingConfig {
                    temperature: 1.0,
                    top_p: 0.95,
                    top_k: -1,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    greedy: false,
                    eos_token_id: -1,
                },
                SamplingConfig {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: -1,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    greedy: false,
                    eos_token_id: -1,
                },
            ];

            for (ci, cfg) in configs.iter().enumerate() {
                let seed = 0xFEEDFACEu64 ^ ((vocab as u64) << 16) ^ (ci as u64);
                let mut cpu_l = logits_base.clone();
                let mut gpu_l = logits_base.clone();
                let mut cpu_s = seed;
                let mut gpu_s = seed;
                let cpu_tok = crate::sampling_cpu::sample(&mut cpu_l, &counts, *cfg, &mut cpu_s);
                let gpu_tok = gpu_algorithm_simulate(&mut gpu_l, &counts, *cfg, &mut gpu_s);
                assert_eq!(
                    cpu_tok, gpu_tok,
                    "RUN-170: large-vocab disagreement at vocab={vocab} cfg_idx={ci}: \
                     cpu={cpu_tok} gpu={gpu_tok}"
                );
                assert_eq!(cpu_s, gpu_s, "RNG state diverged at vocab={vocab}");
            }
        }
    }

    /// RUN-170 regression: verify the kernel cap formula
    /// `cap = (top_k > 0 && top_k < vocab) ? top_k : vocab` actually keeps
    /// more than 256 candidates for a top_p=0.999 cut on a diffuse 4096-vocab
    /// distribution. (The 256 cap would have stopped the keep loop early; the
    /// simulator counts kept entries as a proxy for the kernel's behavior.)
    #[test]
    fn gpu_simulator_keep_set_exceeds_old_max_keep_when_warranted() {
        // Vocab large enough that we expect significantly more than 256 kept
        // tokens for a near-1 top_p on a roughly-uniform distribution.
        let vocab = 4096usize;
        // Near-uniform logits → softmax ≈ 1/vocab per token → ~95% of
        // tokens needed to cross top_p=0.95.
        let logits: Vec<f32> = (0..vocab).map(|i| (i as f32) * 1e-6).collect();
        let counts = vec![0u32; vocab];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 0.95,
            top_k: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: false,
            eos_token_id: -1,
        };
        let mut cpu_l = logits.clone();
        let mut gpu_l = logits.clone();
        let mut cpu_s = 0xC0FFEEu64;
        let mut gpu_s = 0xC0FFEEu64;
        let cpu_tok = crate::sampling_cpu::sample(&mut cpu_l, &counts, cfg, &mut cpu_s);
        let gpu_tok = gpu_algorithm_simulate(&mut gpu_l, &counts, cfg, &mut gpu_s);
        assert_eq!(
            cpu_tok, gpu_tok,
            "GPU simulator must match CPU on near-uniform diffuse top_p"
        );

        // Sanity: confirm the cap *would* select more than 256 candidates.
        // Count how many candidates the simulator-equivalent loop keeps by
        // running a manual version that exposes the kept-set size.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let mut cum = 0.0f32;
        let mut kept = 0usize;
        loop {
            let mut best_val = -f32::INFINITY;
            let mut best_idx: i32 = -1;
            for (i, &v) in probs.iter().enumerate() {
                if v > best_val || (v == best_val && (best_idx == -1 || (i as i32) < best_idx)) {
                    best_val = v;
                    best_idx = i as i32;
                }
            }
            if best_idx < 0 || best_val <= 0.0 {
                break;
            }
            probs[best_idx as usize] = f32::NEG_INFINITY;
            cum += best_val;
            kept += 1;
            if cum >= cfg.top_p || kept >= vocab {
                break;
            }
        }
        assert!(
            kept > 256,
            "RUN-170 test setup invalid: top_p=0.95 over uniform 4096-vocab \
             should keep > 256 tokens, got {kept}"
        );
    }

    /// Empty-distribution fallback: when all logits are -inf, both samplers
    /// fall back to argmax (the first index).
    #[test]
    fn gpu_simulator_handles_all_neg_inf() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 0.9,
            top_k: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: false,
            eos_token_id: -1,
        };
        let mut logits = vec![f32::NEG_INFINITY; 8];
        let counts = vec![0u32; 8];
        let mut state = 0u64;
        let tok = gpu_algorithm_simulate(&mut logits, &counts, cfg, &mut state);
        // greedy_argmax on all -inf returns 0 (no value beats -inf).
        assert_eq!(tok, 0);
    }

    /// Splitmix64 step in the simulator (and thus in the kernel) produces
    /// the same uniform draw as the CPU reference: identical state → identical u.
    #[test]
    fn splitmix_rng_step_matches_cpu_reference() {
        // Reproduce the exact lines from sampling_cpu.rs lines 156-161.
        let mut state: u64 = 0xCAFEBABE;
        let original = state;
        state = state
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xDEADBEEFC0DECAFE);
        let mixed = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        let cpu_u = ((mixed >> 33) as u32) as f32 / (1u64 << 31) as f32;

        // Re-derive via simulator path (cfg with top_p=1.0 so only one draw needed).
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: false,
            eos_token_id: -1,
        };
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let counts = vec![0u32; 4];
        let mut s = original;
        let _ = gpu_algorithm_simulate(&mut logits, &counts, cfg, &mut s);
        // After one call, state should equal the manually-computed `state` value.
        assert_eq!(s, state, "RNG state advanced differently in simulator");
        // And the draw value matches what would be derived from the same state.
        assert!((0.0..1.0).contains(&cpu_u));
    }
}
