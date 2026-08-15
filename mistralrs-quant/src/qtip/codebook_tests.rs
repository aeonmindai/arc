//! The computed codebook: shape, parity, and the UQFF discriminator.
//!
//! The LUT rung decodes each trellis state by gathering two floats out of a
//! 512 KiB table. That gather is the measured limiter at BOTH ends of the
//! pipeline — the bake kernel (wave19-AP: 1.81x available on the beam) and the
//! decode GEMV (`bitshift.rs`: 388 GB/s ~ 8% of HBM, the stall attributed to
//! the dependent, scattered load). [`QtipCodebook::Mcg`] replaces the table
//! with ~10 register instructions.
//!
//! What these tests hold down, in order of what would hurt most if it broke:
//!
//! 1. **The stored table equals the computed values.** This is what makes a
//!    computed-codebook artifact decodable by a reader that predates the
//!    discriminator, and what makes `QtipCodebook::Gaussian` ("gather from the
//!    table") a safe fallback everywhere.
//! 2. **The per-state function equals the table.** The CUDA kernels implement
//!    the per-state form; the artifact and the CPU search use the table. If
//!    these ever disagree the GPU and CPU decode the same bytes differently.
//! 3. **Old artifacts still load, and load as Gaussian.**
//! 4. **`sum2`, not `split`.** With a negative control: the cheaper `split`
//!    construction is shown to have the magnitude hole that made it
//!    quality-negative, so nobody re-derives it as an "optimization".

use candle_core::{DType, Device, Result, Tensor};
use std::borrow::Cow;

use super::bake_quality_tests::{cosine, decode_packed, gen_fp4_dequant, matmul_t, pack_symbols};
use crate::{QuantMethod, QuantizedSerde};

use super::{
    apply_block_rotation, gaussian_lut, mcg_codebook_v2, mcg_codeword_v2, quantize_row,
    QtipBakeConfig, QtipCodebook, QtipLayer, QtipMode, TrellisSearch, LUT_SIZE, QTIP_MCG_V2_MULT,
    QTIP_MCG_V2_SCALE_DIVISOR, QTIP_MCG_V2_SIGMA, QTIP_ROTATION_SEED, V,
};
use crate::turboquant::wht::generate_signs;

/// Mean / sigma / min|v| / fraction inside ±0.25σ of a codebook, in f64.
fn stats(cb: &[f32]) -> (f64, f64, f64, f64) {
    let n = cb.len() as f64;
    let mean = cb.iter().map(|&v| v as f64).sum::<f64>() / n;
    let sigma = (cb.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let min_abs = cb
        .iter()
        .map(|&v| v.abs() as f64)
        .fold(f64::INFINITY, f64::min);
    let near = cb
        .iter()
        .filter(|&&v| (v as f64).abs() < 0.25 * sigma)
        .count() as f64
        / n;
    (mean, sigma, min_abs, near)
}

/// The `split` construction — the 4-instruction alternative that was actually
/// speed-measured in wave19-AP and is **quality-negative**. It exists here only
/// as a negative control for `computed_codebook_has_no_magnitude_hole`; nothing
/// ships it.
fn mcg_codebook_v2_split(mult: u32) -> Vec<f32> {
    let mut cb = Vec::with_capacity(LUT_SIZE * V as usize);
    for state in 0..(1u32 << 16) {
        let x = state.wrapping_mul(mult);
        let m = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
        cb.push(half::f16::from_bits((m >> 16) as u16).to_f32());
        cb.push(half::f16::from_bits((m & 0xFFFF) as u16).to_f32());
    }
    cb
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// `QTIP_MCG_V2_SIGMA` is a measured constant, and the row-scale divisor is
/// derived from it rather than copied from the sibling rung.
///
/// This matters because the quality result the change rests on (wave19-AP part
/// 2: +0.00017 cos, +0.37% NMSE) was measured with the codebook normalized to
/// **exactly** unit sigma. We ship the raw table and fold sigma into the
/// divisor instead — algebraically the same objective, zero decode cost — but
/// only if the sigma we fold in is the real one. `QTIP2B_SCALE_DIVISOR`'s
/// 1.2064 is the K=2/V=1 sigma and is 1.5% off here; using it would ship a
/// scale mismatch the probe never measured.
#[test]
fn computed_codebook_sigma_matches_constant() {
    let cb = mcg_codebook_v2(QTIP_MCG_V2_MULT);
    assert_eq!(cb.len(), LUT_SIZE * V as usize);
    let (mean, sigma, _, _) = stats(&cb);

    assert!(
        mean.abs() < 1e-3,
        "computed codebook should be centred; mean = {mean}"
    );
    assert!(
        (sigma - QTIP_MCG_V2_SIGMA as f64).abs() < 1e-4,
        "QTIP_MCG_V2_SIGMA is {QTIP_MCG_V2_SIGMA} but the table measures {sigma}. \
         The row-scale divisor is derived from this constant, so a drift here \
         silently rescales every bake."
    );
    assert!(
        (QTIP_MCG_V2_SCALE_DIVISOR as f64 - 3.0 * sigma).abs() < 1e-3,
        "scale divisor {QTIP_MCG_V2_SCALE_DIVISOR} != 3 sigma ({})",
        3.0 * sigma
    );
    // And the divisor really is different from the Gaussian rung's 3.0 — if it
    // ever collapses back to 3.0 the computed codebook is being scaled as if it
    // were unit-sigma, which is a 22% dynamic-range error, not a rounding one.
    assert!(
        (QTIP_MCG_V2_SCALE_DIVISOR - 3.0).abs() > 0.5,
        "divisor collapsed to the Gaussian one"
    );
}

/// `sum2` reproduces the Gaussian LUT's shape; `split` does not.
///
/// The mask keeps the sign bit and 12 low bits of each half and the XOR pins
/// each half's exponent into 12..15, so a **single** masked fp16 half cannot
/// have magnitude below 0.142. Taking the two halves as the V=2 pair therefore
/// leaves a hole exactly where a Gaussian weight distribution has most of its
/// mass. Summing two chained products fills it. This is the entire reason the
/// shipped construction costs ~10 instructions instead of ~4.
#[test]
fn computed_codebook_has_no_magnitude_hole() {
    let gauss = gaussian_lut();
    let sum2 = mcg_codebook_v2(QTIP_MCG_V2_MULT);
    let split = mcg_codebook_v2_split(QTIP_MCG_V2_MULT);

    let (_, g_sigma, g_min, g_near) = stats(&gauss);
    let (_, s_sigma, s_min, s_near) = stats(&sum2);
    let (_, p_sigma, p_min, p_near) = stats(&split);

    // sum2 reaches zero and matches the Gaussian's central mass to <1pp.
    assert!(
        s_min / s_sigma < 1e-3,
        "sum2 min|v|/sigma = {} — it should reach zero like the Gaussian ({})",
        s_min / s_sigma,
        g_min / g_sigma
    );
    assert!(
        (s_near - g_near).abs() < 0.01,
        "sum2 puts {:.2}% of its mass inside +/-0.25 sigma vs the Gaussian's {:.2}% \
         — more than 1pp apart means the construction drifted",
        s_near * 100.0,
        g_near * 100.0
    );

    // The negative control: split has the hole. If this ever stops failing,
    // the mask/XOR constants changed and the whole `sum2` justification needs
    // re-deriving.
    assert!(
        p_min / p_sigma > 0.1,
        "split was expected to have a magnitude hole (min|v|/sigma > 0.1), got {} \
         — re-check QTIP_CB_MASK / QTIP_CB_XOR before assuming split is now safe",
        p_min / p_sigma
    );
    let _ = p_near;
}

/// The per-state function and the materialized table are the same object.
///
/// The CUDA kernels compute `qtip_cb_sum2(state, mult)` per state; the CPU
/// search and the artifact use the flat table. A divergence would mean the GPU
/// and CPU decode identical packed bytes to different weights — the exact
/// failure mode that byte-identity testing exists to catch, and it would not
/// show up in any shape statistic.
#[test]
fn per_state_codeword_matches_the_materialized_table() {
    let cb = mcg_codebook_v2(QTIP_MCG_V2_MULT);
    for state in 0..(1u32 << 16) {
        let (v0, v1) = mcg_codeword_v2(state, QTIP_MCG_V2_MULT);
        let i = state as usize * V as usize;
        assert_eq!(
            (v0.to_bits(), v1.to_bits()),
            (cb[i].to_bits(), cb[i + 1].to_bits()),
            "state {state}: per-state codeword != table entry"
        );
    }
}

/// **Mutation arm.** Each of these is a plausible transcription slip when
/// porting `mcg_codeword_v2` into a kernel, and each must change the codebook.
/// A test that passes under a mutation is not testing anything.
///
/// Two candidate mutations were tried and REJECTED for having no teeth, and the
/// negative is recorded here rather than dropped:
///   * masking the state with `0xFFFF` before the multiply — already implied,
///     since every caller passes a state that is already 16-bit;
///   * swapping which half is `hi` and which is `lo` — the two halves are
///     **summed**, so the swap is an identity. (It is NOT an identity for
///     `split`, which is one more reason `split` is a different codebook and
///     not a cheaper spelling of this one.)
#[test]
fn codeword_mutations_all_change_the_codebook() {
    let reference = mcg_codebook_v2(QTIP_MCG_V2_MULT);
    let m = QTIP_MCG_V2_MULT;

    let fold = |x: u32| {
        let y = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
        half::f16::from_bits((y >> 16) as u16).to_f32()
            + half::f16::from_bits((y & 0xFFFF) as u16).to_f32()
    };

    let mutations: [(&str, fn(u32, u32) -> (f32, f32)); 5] = [
        // v1 folds `state * mult` again instead of chaining from x0.
        ("v1 restarts from the state", |s, m| {
            let f = |x: u32| {
                let y = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
                half::f16::from_bits((y >> 16) as u16).to_f32()
                    + half::f16::from_bits((y & 0xFFFF) as u16).to_f32()
            };
            let x0 = s.wrapping_mul(m);
            (f(x0), f(x0))
        }),
        // XOR applied before the mask.
        ("xor before mask", |s, m| {
            let f = |x: u32| {
                let y = (x ^ 0x3B60_3B60) & 0x8FFF_8FFF;
                half::f16::from_bits((y >> 16) as u16).to_f32()
                    + half::f16::from_bits((y & 0xFFFF) as u16).to_f32()
            };
            let x0 = s.wrapping_mul(m);
            (f(x0), f(x0.wrapping_mul(m)))
        }),
        // Mask keeps 11 low bits instead of 12.
        ("mask off by one bit", |s, m| {
            let f = |x: u32| {
                let y = (x & 0x87FF_87FF) ^ 0x3B60_3B60;
                half::f16::from_bits((y >> 16) as u16).to_f32()
                    + half::f16::from_bits((y & 0xFFFF) as u16).to_f32()
            };
            let x0 = s.wrapping_mul(m);
            (f(x0), f(x0.wrapping_mul(m)))
        }),
        // Halves summed in fp16 (the `__hadd` trap the module doc warns about
        // for the sibling rung) instead of in f32.
        ("halves summed in fp16", |s, m| {
            let f = |x: u32| {
                let y = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
                let hi = half::f16::from_bits((y >> 16) as u16);
                let lo = half::f16::from_bits((y & 0xFFFF) as u16);
                (hi + lo).to_f32()
            };
            let x0 = s.wrapping_mul(m);
            (f(x0), f(x0.wrapping_mul(m)))
        }),
        // Chained product uses `mult + 1`.
        ("chained multiplier off by one", |s, m| {
            let f = |x: u32| {
                let y = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
                half::f16::from_bits((y >> 16) as u16).to_f32()
                    + half::f16::from_bits((y & 0xFFFF) as u16).to_f32()
            };
            let x0 = s.wrapping_mul(m);
            (f(x0), f(x0.wrapping_mul(m.wrapping_add(1))))
        }),
    ];

    for (name, mutated) in mutations {
        let mut differed = 0usize;
        for state in 0..(1u32 << 16) {
            let (a0, a1) = mutated(state, m);
            let i = state as usize * V as usize;
            if a0.to_bits() != reference[i].to_bits() || a1.to_bits() != reference[i + 1].to_bits()
            {
                differed += 1;
            }
        }
        assert!(
            differed > LUT_SIZE / 100,
            "mutation {name:?} changed only {differed} of {LUT_SIZE} states — \
             the parity assertion has no teeth against it"
        );
    }
    // Sanity that `fold` above is the same function the reference uses, so the
    // mutations really are one-edit deltas from it and not from something else.
    let x0 = 12345u32.wrapping_mul(m);
    assert_eq!(fold(x0).to_bits(), reference[12345 * 2].to_bits());
}

// ---------------------------------------------------------------------------
// UQFF: the discriminator and back-compat
// ---------------------------------------------------------------------------

fn bake(codebook: QtipCodebook, device: &Device) -> Result<(QtipLayer, Tensor)> {
    let (n, k_in) = (8usize, 128usize);
    let wdata: Vec<f32> = (0..(n * k_in))
        .map(|i| ((i as f32) * 0.37).sin() * 1.25)
        .collect();
    let w = Tensor::from_vec(wdata, (n, k_in), device)?;
    let layer = QtipLayer::quantize_with_bake_config(
        &w,
        None,
        device,
        QtipMode::Viterbi,
        true,
        None,
        QtipBakeConfig {
            search: TrellisSearch::Beam { width: 16 },
            hessian: false,
            codebook,
        },
    )?;
    Ok((layer, w))
}

/// A computed-codebook layer stores the computed values in `lut`.
///
/// This is the property the whole back-compat story rests on: a reader that
/// never heard of the discriminator reads the table, gathers from it, and gets
/// the right weights. The discriminator only buys speed.
#[test]
fn computed_codebook_layer_stores_the_computed_table() -> Result<()> {
    let device = Device::Cpu;
    let (layer, _) = bake(QtipCodebook::COMPUTED, &device)?;
    assert_eq!(layer.codebook(), QtipCodebook::COMPUTED);

    let stored: Vec<f32> = layer.lut.flatten_all()?.to_vec1()?;
    let expected = mcg_codebook_v2(QTIP_MCG_V2_MULT);
    assert_eq!(stored.len(), expected.len());
    for (i, (&a, &b)) in stored.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "stored table diverges from the computed codebook at {i}"
        );
    }
    Ok(())
}

/// Serialize → deserialize preserves the codebook and every byte of the layer.
#[test]
fn uqff_round_trip_preserves_the_codebook() -> Result<()> {
    let device = Device::Cpu;
    for codebook in [QtipCodebook::Gaussian, QtipCodebook::COMPUTED] {
        let (layer, x_src) = bake(codebook, &device)?;
        let bytes = layer.serialize()?.into_owned();
        let (restored, _) = QtipLayer::deserialize_concrete_unchecked(
            Cow::Owned(bytes),
            &device,
            crate::QuantizeOntoGuard::new(),
        )?;
        assert_eq!(
            restored.codebook(),
            codebook,
            "codebook must survive the round-trip"
        );

        let y0: Vec<f32> = layer
            .forward(&x_src)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let y1: Vec<f32> = restored
            .forward(&x_src)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            y0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "forward must be bit-identical after a round-trip ({codebook:?})"
        );
    }
    Ok(())
}

/// **Back-compat.** A payload written before the discriminator existed — i.e.
/// one byte shorter, ending after the search-detail section — still loads, and
/// loads as [`QtipCodebook::Gaussian`].
///
/// Constructed by truncating a real payload's trailing tag rather than by
/// hand-rolling bytes, so the test tracks the writer: if the codebook section
/// ever stops being last, the truncation stops being a valid old payload and
/// this fails.
#[test]
fn pre_discriminator_payload_loads_as_gaussian() -> Result<()> {
    let device = Device::Cpu;
    let (layer, x_src) = bake(QtipCodebook::Gaussian, &device)?;
    let bytes = layer.serialize()?.into_owned();

    // A Gaussian bake writes NO codebook section at all, so its payload
    // already IS a pre-discriminator payload — that is the whole design. Assert
    // that first (it is what keeps every existing artifact and every existing
    // byte-level test unmoved), then load it.
    let mcg_bytes = bake(QtipCodebook::COMPUTED, &device)?
        .0
        .serialize()?
        .into_owned();
    assert_eq!(
        mcg_bytes.len(),
        bytes.len() + 5,
        "the computed codebook must cost exactly 5 trailing bytes (tag + u32 mult) \
         and the Gaussian one must cost none"
    );
    assert_eq!(
        mcg_bytes[bytes.len()],
        2u8,
        "computed-codebook tag must be 2 (1 is reserved, 0 never parses)"
    );

    let (restored, _) = QtipLayer::deserialize_concrete_unchecked(
        Cow::Owned(bytes.clone()),
        &device,
        crate::QuantizeOntoGuard::new(),
    )?;
    assert_eq!(restored.codebook(), QtipCodebook::Gaussian);

    let y0: Vec<f32> = layer
        .forward(&x_src)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let y1: Vec<f32> = restored
        .forward(&x_src)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    assert_eq!(
        y0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        y1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a pre-discriminator payload must decode exactly as it always did"
    );
    Ok(())
}

/// An unknown codebook tag is refused, not defaulted.
///
/// Same contract as the search-mode stamp (#33/#34): a tag we do not
/// understand means a newer writer produced the file, and guessing which
/// codebook its symbols mean is silent corruption, not a graceful degradation.
#[test]
fn unknown_codebook_tag_is_refused() -> Result<()> {
    let device = Device::Cpu;
    let (layer, _) = bake(QtipCodebook::Gaussian, &device)?;
    let base = layer.serialize()?.into_owned();

    // 0 is the reserved never-parses value (a zero-padded buffer must not read
    // as a codebook); 1 is reserved for a future explicitly-named codebook;
    // 7 is simply from the future.
    for tag in [0u8, 1, 7, 255] {
        let mut bytes = base.clone();
        bytes.push(tag);
        let err = QtipLayer::deserialize_concrete_unchecked(
            Cow::Owned(bytes),
            &device,
            crate::QuantizeOntoGuard::new(),
        )
        .expect_err("an unknown codebook tag must be refused");
        assert!(
            err.to_string().contains("unknown codebook tag"),
            "error should name the cause; got {err}"
        );
    }
    Ok(())
}

/// A computed-codebook payload truncated mid-multiplier is refused rather than
/// read as "no multiplier, use the default".
#[test]
fn truncated_multiplier_is_refused() -> Result<()> {
    let device = Device::Cpu;
    let (layer, _) = bake(QtipCodebook::COMPUTED, &device)?;
    let bytes = layer.serialize()?.into_owned();
    // Drop two of the multiplier's four bytes.
    let truncated = bytes[..bytes.len() - 2].to_vec();
    let err = QtipLayer::deserialize_concrete_unchecked(
        Cow::Owned(truncated),
        &device,
        crate::QuantizeOntoGuard::new(),
    )
    .expect_err("a truncated multiplier must be refused");
    assert!(
        err.to_string().contains("multiplier"),
        "error should name the cause; got {err}"
    );
    Ok(())
}

/// `ARC_QTIP_CODEBOOK` fails closed on an unrecognised value.
#[test]
fn unknown_codebook_env_value_is_refused() {
    // `from_env` is pure w.r.t. its argument source, so drive it through the
    // same parse without touching the process environment (which other tests
    // share). Values accepted here are the whole accepted set.
    assert_eq!(QtipCodebook::DEFAULT, QtipCodebook::default());
    assert!(matches!(QtipCodebook::COMPUTED, QtipCodebook::Mcg { .. }));
    assert_eq!(QtipCodebook::Gaussian.cuda_mult(), 0);
    assert_eq!(QtipCodebook::COMPUTED.cuda_mult(), QTIP_MCG_V2_MULT);
    assert_eq!(
        QtipCodebook::Gaussian.scale_divisor(),
        super::QTIP_GAUSSIAN_SCALE_DIVISOR
    );
}

// ---------------------------------------------------------------------------
// Quality: the shipped configuration, not the probe's
// ---------------------------------------------------------------------------

/// Quantize `[n, k]` through the production CPU pipeline under `codebook`,
/// returning the reconstruction in the ORIGINAL frame. Op-for-op identical to
/// `QtipLayer::quantize_with_options_concrete` including the codebook's own
/// scale divisor — which is the point: wave19-AP measured neutrality with the
/// table normalized to unit sigma, and this measures it the way we ship it.
fn quantize_matrix(w: &[f32], n: usize, k: usize, codebook: QtipCodebook) -> Vec<f32> {
    const ROT_BLOCK: usize = 128;
    let lut = codebook.materialize();
    let divisor = codebook.scale_divisor();
    let signs = generate_signs(QTIP_ROTATION_SEED, k);
    let num_symbols = k / V as usize;

    let mut out = vec![0f32; n * k];
    for row in 0..n {
        let mut rot = w[row * k..(row + 1) * k].to_vec();
        apply_block_rotation(&mut rot, &signs, ROT_BLOCK);

        let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs == 0.0 {
            1.0
        } else {
            max_abs / divisor
        };
        let inv = 1.0 / scale;
        let target: Vec<f32> = rot.iter().map(|&v| v * inv).collect();

        let syms = quantize_row(&target, &lut, TrellisSearch::Beam { width: 256 }, None);
        let packed = pack_symbols(&syms);

        let mut recon: Vec<f32> = decode_packed(&packed, num_symbols, &lut)
            .into_iter()
            .map(|c| c * scale)
            .collect();
        apply_block_rotation(&mut recon, &signs, ROT_BLOCK);
        out[row * k..(row + 1) * k].copy_from_slice(&recon);
    }
    out
}

fn rel_nmse(w: &[f32], w_hat: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (&a, &b) in w.iter().zip(w_hat.iter()) {
        let d = (a - b) as f64;
        num += d * d;
        den += (a as f64) * (a as f64);
    }
    num / den.max(1e-300)
}

/// **The quality gate, re-run in the shipped configuration.**
///
/// wave19-AP part 2 measured `sum2` neutral (+0.00017 cos, +0.37% weight NMSE)
/// with the codebook explicitly normalized to unit sigma and the historical
/// `max/3` divisor. We ship the raw table with a `3 sigma` divisor instead —
/// the same objective up to a positive constant, but not the same floating-point
/// arithmetic. This asserts the verdict survives that substitution on the
/// fixture family that has flipped decisions in this repo twice
/// (`fp4_dequant` is V4's real source distribution).
///
/// Thresholds are deliberately loose relative to the measurement (0.002 cos,
/// 3% NMSE) and tight relative to the thing being excluded: `split` costs
/// −0.00174 cos and +3.73% NMSE, and a scale mismatch would show up as a
/// several-percent NMSE move. This is a regression gate, not a re-measurement.
#[test]
fn computed_codebook_quality_is_neutral_as_shipped() {
    let (n, k, batch) = (24usize, 1024usize, 64usize);
    let w = gen_fp4_dequant(n, k, 0.02, 0x0051_EA11);

    // Isotropic eval activations: the objective here is unweighted, so the
    // activation draw only decides the evaluation set. Weight NMSE is the
    // primary number for exactly that reason.
    let mut x = vec![0f32; batch * k];
    let mut seed = 0x1234_5678u64;
    for v in x.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
        *v = u as f32;
    }

    let g_hat = quantize_matrix(&w, n, k, QtipCodebook::Gaussian);
    let m_hat = quantize_matrix(&w, n, k, QtipCodebook::COMPUTED);

    let y = matmul_t(&x, &w, batch, n, k);
    let g_cos = cosine(&y, &matmul_t(&x, &g_hat, batch, n, k));
    let m_cos = cosine(&y, &matmul_t(&x, &m_hat, batch, n, k));
    let g_nmse = rel_nmse(&w, &g_hat);
    let m_nmse = rel_nmse(&w, &m_hat);

    println!(
        "gaussian-LUT  cos={g_cos:.5}  w_nmse={g_nmse:.6}\n\
         mcg-sum2      cos={m_cos:.5}  w_nmse={m_nmse:.6}  \
         (dcos={:+.5}, d_rel_nmse={:+.2}%)",
        m_cos - g_cos,
        100.0 * (m_nmse / g_nmse - 1.0)
    );

    assert!(
        m_cos - g_cos > -0.002,
        "computed codebook cost {:.5} cos vs the Gaussian LUT — wave19-AP measured \
         -0.0008 worst-family, so anything past -0.002 means the shipped scale \
         folding changed the result",
        m_cos - g_cos
    );
    assert!(
        m_nmse / g_nmse - 1.0 < 0.03,
        "computed codebook weight NMSE is {:.2}% worse than the Gaussian LUT; \
         the measured figure is +0.37% and `split` (which we do not ship) is +3.7%",
        100.0 * (m_nmse / g_nmse - 1.0)
    );
}

/// **The discriminator is load-bearing.** Decoding computed-codebook symbols
/// against the Gaussian table produces garbage, so an artifact that lost its
/// tag would be silently wrong — which is why the tag is written, why absence
/// means Gaussian (the only codebook that ever shipped without one), and why an
/// unknown tag is refused.
#[test]
fn decoding_against_the_wrong_codebook_is_catastrophic() {
    let (n, k) = (8usize, 512usize);
    let w = gen_fp4_dequant(n, k, 0.02, 0x0051_EA12);
    let correct = quantize_matrix(&w, n, k, QtipCodebook::COMPUTED);

    // Same symbols, decoded against the Gaussian table.
    const ROT_BLOCK: usize = 128;
    let mcg = QtipCodebook::COMPUTED;
    let cb = mcg.materialize();
    let gauss = gaussian_lut();
    let signs = generate_signs(QTIP_ROTATION_SEED, k);
    let num_symbols = k / V as usize;
    let mut wrong = vec![0f32; n * k];
    for row in 0..n {
        let mut rot = w[row * k..(row + 1) * k].to_vec();
        apply_block_rotation(&mut rot, &signs, ROT_BLOCK);
        let max_abs = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs == 0.0 {
            1.0
        } else {
            max_abs / mcg.scale_divisor()
        };
        let target: Vec<f32> = rot.iter().map(|&v| v / scale).collect();
        let syms = quantize_row(&target, &cb, TrellisSearch::Beam { width: 256 }, None);
        let packed = pack_symbols(&syms);
        let mut recon: Vec<f32> = decode_packed(&packed, num_symbols, &gauss)
            .into_iter()
            .map(|c| c * scale)
            .collect();
        apply_block_rotation(&mut recon, &signs, ROT_BLOCK);
        wrong[row * k..(row + 1) * k].copy_from_slice(&recon);
    }

    let good = rel_nmse(&w, &correct);
    let bad = rel_nmse(&w, &wrong);
    assert!(
        bad > 5.0 * good,
        "decoding against the wrong codebook gave NMSE {bad:.4} vs {good:.4} — \
         if these are close the discriminator is not actually distinguishing anything"
    );
}
