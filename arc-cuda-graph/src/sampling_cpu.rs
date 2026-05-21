//! Portable CPU sampler — the reference implementation that the on-device
//! conditional-graph sampling kernel must match bit-for-bit.
//!
//! This module is CUDA-independent and unit-tested on any host. Its purpose is
//! to specify the exact sampling math (temperature, top-p, top-k, greedy, EOS)
//! so the autonomous decode GPU kernel (Tier B) can be validated against it.
//!
//! Determinism is essential: the GPU's on-device RNG must produce identical
//! outputs to this CPU reference given the same seed + state. The test suite
//! pins the seed and asserts bit-exact agreement.

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32, // -1 = disabled
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub greedy: bool,
    pub eos_token_id: i32,
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(format!(
                "temperature must be finite and non-negative, got {}",
                self.temperature
            ));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(format!("top_p must be in [0,1], got {}", self.top_p));
        }
        if self.top_k != -1 && self.top_k < 1 {
            return Err(format!("top_k must be -1 (disabled) or >= 1, got {}", self.top_k));
        }
        if !self.frequency_penalty.is_finite() || !self.presence_penalty.is_finite() {
            return Err("penalty values must be finite".into());
        }
        Ok(())
    }

    /// Greedy mode is "pure argmax" — bypasses temperature/top-p/top-k.
    pub fn greedy() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: true,
            eos_token_id: -1,
        }
    }
}

/// Apply frequency + presence penalties to logits in-place using the token-count
/// history. Matches HuggingFace's `RepetitionPenaltyLogitsProcessor`.
///
/// `frequency_counts[t]` = number of times token t has been generated so far.
pub fn apply_penalties(
    logits: &mut [f32],
    frequency_counts: &[u32],
    cfg: SamplingConfig,
) {
    debug_assert_eq!(logits.len(), frequency_counts.len());
    for (tok, &count) in frequency_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        logits[tok] -= cfg.frequency_penalty * count as f32;
        logits[tok] -= cfg.presence_penalty;
    }
}

/// Apply temperature in-place: logits ← logits / T.
/// Skip the divide if temperature is 1.0 (common case).
/// At temperature = 0.0, fall back to greedy by leaving logits alone; the
/// caller checks `cfg.greedy` separately.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 || temperature == 0.0 {
        return;
    }
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}

/// Greedy: return argmax of logits.
pub fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Run the full sampler given pre-computed logits.
///
/// `rng_state` is mutated by the call (Splitmix-style) so successive calls
/// produce different draws while remaining deterministic.
pub fn sample(
    logits: &mut [f32],
    frequency_counts: &[u32],
    cfg: SamplingConfig,
    rng_state: &mut u64,
) -> u32 {
    if cfg.greedy {
        apply_penalties(logits, frequency_counts, cfg);
        return greedy_argmax(logits);
    }
    apply_penalties(logits, frequency_counts, cfg);
    apply_temperature(logits, cfg.temperature);

    // Softmax → cumulative top-p → multinomial draw.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    // Sort descending and apply top-p truncation.
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep: Vec<(usize, f32)> = Vec::with_capacity(indexed.len());
    let mut cum = 0.0f32;
    for (idx, p) in indexed {
        cum += p;
        keep.push((idx, p));
        if cum >= cfg.top_p {
            break;
        }
    }
    if cfg.top_k > 0 {
        keep.truncate(cfg.top_k as usize);
    }
    // Re-normalize
    let kept_sum: f32 = keep.iter().map(|(_, p)| *p).sum();
    if kept_sum <= 0.0 {
        // Pathological — fall back to argmax.
        return greedy_argmax(logits);
    }

    // Deterministic RNG: Splitmix64 → u32 → [0, 1).
    *rng_state = rng_state
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xDEADBEEFC0DECAFE);
    let mixed = (*rng_state ^ (*rng_state >> 30))
        .wrapping_mul(0xBF58476D1CE4E5B9);
    let u = ((mixed >> 33) as u32) as f32 / (1u64 << 31) as f32;
    let target = u * kept_sum;

    let mut acc = 0.0;
    for (idx, p) in &keep {
        acc += *p;
        if acc >= target {
            return *idx as u32;
        }
    }
    // Numerical fallback (shouldn't happen with kept_sum > 0).
    keep.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Detect end-of-sequence.
pub fn is_eos(token: u32, cfg: SamplingConfig) -> bool {
    cfg.eos_token_id >= 0 && (token as i32) == cfg.eos_token_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_default_greedy_config() {
        SamplingConfig::greedy().validate().unwrap();
    }

    #[test]
    fn validate_rejects_negative_temperature() {
        let mut cfg = SamplingConfig::greedy();
        cfg.temperature = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_top_p_above_1() {
        let mut cfg = SamplingConfig::greedy();
        cfg.top_p = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_top_k_zero() {
        let mut cfg = SamplingConfig::greedy();
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn greedy_argmax_picks_largest() {
        let logits = vec![1.0, 5.0, 3.0, -2.0, 5.1];
        assert_eq!(greedy_argmax(&logits), 4);
    }

    #[test]
    fn temperature_scales_logits() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        assert_eq!(logits, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn temperature_1_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let orig = logits.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, orig);
    }

    #[test]
    fn penalties_decrease_repeated_tokens() {
        let mut logits = vec![1.0, 1.0, 1.0];
        let counts = vec![0u32, 3, 0];
        let mut cfg = SamplingConfig::greedy();
        cfg.frequency_penalty = 0.5;
        cfg.presence_penalty = 0.2;
        apply_penalties(&mut logits, &counts, cfg);
        // token 1 had count=3: logits[1] -= 0.5*3 + 0.2 = 1.7 → -0.7
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - (-0.7)).abs() < 1e-5);
        assert!((logits[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eos_detection() {
        let mut cfg = SamplingConfig::greedy();
        cfg.eos_token_id = 2;
        assert!(is_eos(2, cfg));
        assert!(!is_eos(3, cfg));
        // -1 means disabled
        cfg.eos_token_id = -1;
        assert!(!is_eos(0, cfg));
    }

    /// Sampling is deterministic for a fixed seed.
    #[test]
    fn sample_is_deterministic_for_seed() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 0.9,
            top_k: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            greedy: false,
            eos_token_id: -1,
        };
        let logits_base: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let counts = vec![0u32; 16];

        let mut state_a = 0xCAFEBABEu64;
        let mut logits_a = logits_base.clone();
        let tok_a = sample(&mut logits_a, &counts, cfg, &mut state_a);

        let mut state_b = 0xCAFEBABEu64;
        let mut logits_b = logits_base.clone();
        let tok_b = sample(&mut logits_b, &counts, cfg, &mut state_b);

        assert_eq!(tok_a, tok_b, "Same seed produced different tokens");
    }

    /// Greedy sampling always picks the argmax regardless of temperature/top_p.
    #[test]
    fn greedy_ignores_temperature_and_top_p() {
        let mut cfg = SamplingConfig::greedy();
        cfg.temperature = 0.7;
        cfg.top_p = 0.3;
        let logits_base = vec![0.0f32, 1.0, 5.0, 2.0];
        let counts = vec![0u32; 4];
        let mut state = 0;
        let mut logits = logits_base.clone();
        let tok = sample(&mut logits, &counts, cfg, &mut state);
        assert_eq!(tok, 2); // argmax
    }
}
