//! DOCTRINE D4 enforcement tests — "ban greedy forever" (wave13-AG).
//!
//! Greedy escaped three times before these existed: RUN-161 shipped it as the
//! production default (PPL 58.85 vs q2k 22.50); a session-4 "fast bake" was
//! silently greedy and burned hours on a phantom regression; and the
//! `--isq qtip2b` 3-D dispatch kept selecting it long after the sibling rung
//! was fixed. Every escape was invisible for the same reason — nothing recorded
//! which search produced an artifact.
//!
//! The ban is code, not prose, and this module is the proof:
//!
//! | § | property | test |
//! |---|---|---|
//! | 1 | the production mode is Viterbi, with no env input at all | `production_expert_mode_is_viterbi_and_const` |
//! | 1 | every production door hard-errors on greedy | `greedy_is_refused_at_every_production_door` |
//! | 2 | greedy is reachable only from a `cfg(test)` fixture door | `greedy_fixture_door_exists_only_under_cfg_test` |
//! | 3 | production bakes stamp the artifact, both rungs | `production_bakes_stamp_the_artifact_on_both_rungs` |
//! | 3 | a greedy-stamped artifact is refused at load | `greedy_stamped_artifact_is_refused_at_load` |
//! | 3 | legacy (unstamped) policy: rotation ⇒ serve, no rotation ⇒ refuse | `legacy_unstamped_*` |
//! | — | rotation is an explicit per-mode policy, not a `matches!` side effect | `rotation_policy_is_pinned_per_mode` |

use candle_core::{DType, Device, Tensor};
use std::borrow::Cow;
use std::sync::{atomic::AtomicUsize, Arc};

use super::{Qtip2bLayer, QtipLayer, QtipMode, QtipRotation, QtipSearchDetail, QtipSearchStamp};
use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, UnquantLinear,
};

/// The provenance tail of a serialized QTIP payload, in ONE place.
///
/// #34 appended a stamp byte; wave13-AF appended a search-detail flags byte
/// after it (plus a `u16` little-endian width when the beam bit is set), so
/// `bytes.last()` is no longer the stamp. Every artifact in this module is
/// baked with the default exhaustive/unweighted search, so its tail is exactly
/// `[stamp, 0x00]` — and asserting the flags byte here is not incidental: it
/// pins that a default production bake claims "exhaustive, unweighted" rather
/// than leaving those fields unset.
fn provenance_tail(bytes: &[u8]) -> (Option<u8>, u8) {
    let n = bytes.len();
    assert!(n >= 2, "payload too short to carry a provenance tail");
    (Some(bytes[n - 2]), bytes[n - 1])
}

/// Strip the provenance tail, producing a byte-for-byte pre-0.3.0 payload.
fn strip_provenance(bytes: &mut Vec<u8>) {
    let n = bytes.len();
    bytes.truncate(n - 2);
}

/// Deterministic Gaussian-ish fixture (splitmix64 + Box-Muller), so no test in
/// this module depends on an RNG seed that could drift.
fn fixture(len: usize, seed: u64, sigma: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let mut z = ((i + 1) as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFF_FFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            (-2.0_f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * sigma
        })
        .collect()
}

// ---------------------------------------------------------------------------
// §1 — no production path selects greedy
// ---------------------------------------------------------------------------

/// The production mode is Viterbi, and `default_expert_mode` is a `const fn`,
/// which is a *compile-time* proof that it reads no environment variable: const
/// evaluation cannot call `std::env`. That is stronger than asserting the value
/// with `ARC_QTIP_EXPERT_GREEDY` unset — an env-based assertion would race with
/// cargo's parallel test threads anyway.
#[test]
fn production_expert_mode_is_viterbi_and_const() {
    const PRODUCTION: QtipMode = QtipMode::default_expert_mode();
    assert_eq!(PRODUCTION, QtipMode::Viterbi);
    // The ISQ thread policy must key off the same decision, or the bake picks
    // its thread count for a mode it is not running (the session-5 trap).
    assert_eq!(
        crate::IsqType::QtipBitshift2
            .isq_cpu_thread_policy(crate::IsqQuantizeBackend::Cpu)
            .0,
        crate::IsqType::Qtip2b
            .isq_cpu_thread_policy(crate::IsqQuantizeBackend::Cpu)
            .0,
        "the two QTIP rungs must resolve to one thread policy — they now share one bake mode"
    );
}

/// Both rungs' production quantize door refuses greedy, in every build.
#[test]
fn greedy_is_refused_at_every_production_door() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 1, 0.5), (8usize, 64usize), &device).unwrap();

    for (rung, err) in [
        (
            "qtip2",
            QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::Greedy).err(),
        ),
        (
            "qtip2b",
            Qtip2bLayer::quantize_with_mode(&w, None, &device, QtipMode::Greedy).err(),
        ),
    ] {
        let err = err.unwrap_or_else(|| panic!("{rung}: quantize_with_mode accepted Greedy"));
        let msg = err.to_string();
        assert!(
            msg.contains("banned in production") && msg.contains("D4"),
            "{rung}: refusal must name the doctrine it enforces, got: {msg}"
        );
    }

    // ...and the same door with the production mode succeeds, so the test above
    // is not passing because the door is simply broken.
    QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::default_expert_mode()).unwrap();
    Qtip2bLayer::quantize_with_mode(&w, None, &device, QtipMode::default_expert_mode()).unwrap();
}

/// The whole ISQ dispatch — every rung, both ranks — never hands greedy to the
/// quantizer. Guards the specific regression this branch fixes: the `Qtip2b`
/// arm of `UnquantLinear::apply_isq` used to send 3-D expert stacks to Greedy
/// while the `QtipBitshift2` arm beside it used the shared decision point.
#[test]
fn isq_dispatch_never_bakes_greedy_on_either_rung_or_rank() {
    let device = Device::Cpu;
    for isq in [IsqType::QtipBitshift2, IsqType::Qtip2b] {
        for dims in [vec![8usize, 64usize], vec![2usize, 8usize, 64usize]] {
            let len: usize = dims.iter().product();
            let w = Tensor::from_vec(fixture(len, 7, 0.5), dims.clone(), &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let unquant: Arc<UnquantLinear> = Arc::new(
                UnquantLinear::new(QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                    w, None,
                )))
                .unwrap(),
            );
            let layer = unquant
                .apply_isq(
                    Some(isq),
                    device.clone(),
                    &AtomicUsize::new(0),
                    None,
                    QuantizeOntoGuard::new(),
                )
                .unwrap_or_else(|e| panic!("{isq:?} rank-{} dispatch failed: {e}", dims.len()));

            // The artifact is the evidence: a production bake must stamp
            // `trellis` AND carry the incoherence rotation.
            let bytes = layer.serialize().unwrap().into_owned();
            let (stamp, flags) = provenance_tail(&bytes);
            assert_eq!(
                stamp,
                QtipSearchStamp::Trellis.to_wire(),
                "{isq:?} rank-{}: production bake did not stamp `trellis`",
                dims.len()
            );
            assert_eq!(
                flags,
                0x00,
                "{isq:?} rank-{}: production bake must record exhaustive + unweighted, \
                 not leave the search detail unset",
                dims.len()
            );
            // Reload through the D4 gate (it must pass) and confirm the bake
            // also turned incoherence processing on — the second half of the
            // defect, which a stamp alone would not catch.
            let rotation_block = match isq {
                IsqType::QtipBitshift2 => QtipLayer::deserialize_concrete(
                    Cow::Owned(bytes),
                    &device,
                    QuantizeOntoGuard::new(),
                )
                .unwrap_or_else(|e| panic!("{isq:?} rank-{}: reload refused: {e}", dims.len()))
                .0
                .rotation_block(),
                _ => Qtip2bLayer::deserialize_concrete(
                    Cow::Owned(bytes),
                    &device,
                    QuantizeOntoGuard::new(),
                )
                .unwrap_or_else(|e| panic!("{isq:?} rank-{}: reload refused: {e}", dims.len()))
                .0
                .rotation_block(),
            };
            assert!(
                rotation_block >= 2,
                "{isq:?} rank-{}: production bake left incoherence rotation off",
                dims.len()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §2 — greedy survives only as a directly-constructed test fixture
// ---------------------------------------------------------------------------

/// The fixture door exists, produces a real greedy layer, and is `cfg(test)` —
/// `quantize_greedy_fixture` does not compile into a release build at all, so a
/// downstream crate cannot call it even by name. The low-level
/// `quantize_with_options` door it wraps is `pub`, but it runs
/// `deny_greedy_outside_tests`, which bails whenever `cfg!(test)` is false;
/// that is why an external caller (or a release binary) cannot use it either.
#[test]
fn greedy_fixture_door_exists_only_under_cfg_test() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 2, 0.5), (8usize, 64usize), &device).unwrap();

    let greedy = QtipLayer::quantize_greedy_fixture(&w, None, &device).unwrap();
    // A greedy fixture is a real artifact — it just may never be served. This
    // module (and `quantize_greedy_fixture` itself) is `cfg(test)`-only, so the
    // door does not exist in a release build at all.
    let bytes = greedy.serialize().unwrap().into_owned();
    let (stamp, flags) = provenance_tail(&bytes);
    assert_eq!(
        stamp,
        QtipSearchStamp::Greedy.to_wire(),
        "the fixture door must stamp what it actually ran"
    );
    assert_eq!(
        flags, 0x00,
        "a greedy walk runs no trellis search, so it must claim no beam and no objective"
    );
}

// ---------------------------------------------------------------------------
// Rotation is an explicit policy, not a side effect of the search mode
// ---------------------------------------------------------------------------

/// Pins the rotation policy per mode. Rotation used to be inferred at three
/// independent call sites with `matches!(mode, QtipMode::Viterbi)`, which
/// answers "off" for any variant added later — silently, at bake time, on the
/// rung carrying a MoE model's entire weight mass. `QtipRotation::for_mode` is
/// now an exhaustive `match`, so a new `QtipMode` variant is a COMPILE error
/// until someone states its rotation policy, and this test pins the answers.
#[test]
fn rotation_policy_is_pinned_per_mode() {
    assert_eq!(QtipRotation::for_mode(QtipMode::Viterbi), QtipRotation::On);
    assert_eq!(QtipRotation::for_mode(QtipMode::Greedy), QtipRotation::Off);
    assert!(QtipRotation::On.enabled());
    assert!(!QtipRotation::Off.enabled());

    // And the policy is what the bake actually applies, on both rungs — a table
    // nobody consults is not a policy.
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 3, 0.5), (8usize, 64usize), &device).unwrap();

    let lut = QtipLayer::quantize_with_options_concrete(
        &w,
        None,
        &device,
        QtipMode::Viterbi,
        QtipRotation::for_mode(QtipMode::Viterbi).enabled(),
    )
    .unwrap();
    assert!(lut.rotation_block() >= 2, "Viterbi must bake with rotation");

    let lut_greedy = QtipLayer::quantize_with_options_concrete(
        &w,
        None,
        &device,
        QtipMode::Greedy,
        QtipRotation::for_mode(QtipMode::Greedy).enabled(),
    )
    .unwrap();
    assert_eq!(
        lut_greedy.rotation_block(),
        0,
        "the Greedy fixture policy is rotation-off; changing it changes fixture bytes"
    );

    let bitshift = Qtip2bLayer::quantize_with_options_concrete(
        &w,
        None,
        &device,
        QtipMode::Viterbi,
        QtipRotation::for_mode(QtipMode::Viterbi).enabled(),
    )
    .unwrap();
    assert!(
        bitshift.rotation_block() >= 2,
        "qtip2b Viterbi must bake with rotation"
    );
}

// ---------------------------------------------------------------------------
// §3 — the stamp travels with the artifact and is checked at load
// ---------------------------------------------------------------------------

/// UQFF carries the stamp, and this build writes 0.3.0. The MINOR bump matters:
/// `version_is_compatible` only gates major/minor, so a patch bump would let a
/// pre-ban reader accept a stamped payload and ignore the stamp.
#[test]
fn uqff_version_is_0_3_0_so_old_readers_fail_cleanly() {
    assert_eq!(
        crate::utils::UQFF_VERSION,
        3u32 << 8,
        "the search stamp changed the QTIP payload layout; the minor version must record it"
    );
}

/// Round-trip: a production bake stamps `trellis`, reloads, and keeps the stamp.
#[test]
fn production_bakes_stamp_the_artifact_on_both_rungs() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 4, 0.5), (8usize, 64usize), &device).unwrap();

    let lut = QtipLayer::quantize_with_options_concrete(
        &w,
        None,
        &device,
        QtipMode::default_expert_mode(),
        QtipRotation::for_mode(QtipMode::default_expert_mode()).enabled(),
    )
    .unwrap();
    assert_eq!(lut.search_stamp(), QtipSearchStamp::Trellis);
    let bytes = QuantizedSerde::serialize(&lut).unwrap().into_owned();
    let (restored, _) =
        QtipLayer::deserialize_ext_bias(Cow::Owned(bytes), &device, QuantizeOntoGuard::new())
            .expect("a trellis-stamped artifact must load");
    drop(restored);

    let bitshift = Qtip2bLayer::quantize_with_options_concrete(
        &w,
        None,
        &device,
        QtipMode::default_expert_mode(),
        QtipRotation::for_mode(QtipMode::default_expert_mode()).enabled(),
    )
    .unwrap();
    assert_eq!(bitshift.search_stamp(), QtipSearchStamp::Trellis);
    let bytes = QuantizedSerde::serialize(&bitshift).unwrap().into_owned();
    Qtip2bLayer::deserialize_ext_bias(Cow::Owned(bytes), &device, QuantizeOntoGuard::new())
        .expect("a trellis-stamped qtip2b artifact must load");
}

/// The teeth: a greedy-stamped artifact is refused by the serving load path on
/// both rungs, with no override.
#[test]
fn greedy_stamped_artifact_is_refused_at_load() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 5, 0.5), (8usize, 64usize), &device).unwrap();

    for (rung, bytes) in [
        (
            "qtip2",
            QuantizedSerde::serialize(
                &QtipLayer::quantize_with_options_concrete(
                    &w,
                    None,
                    &device,
                    QtipMode::Greedy,
                    false,
                )
                .unwrap(),
            )
            .unwrap()
            .into_owned(),
        ),
        (
            "qtip2b",
            QuantizedSerde::serialize(
                &Qtip2bLayer::quantize_with_options_concrete(
                    &w,
                    None,
                    &device,
                    QtipMode::Greedy,
                    false,
                )
                .unwrap(),
            )
            .unwrap()
            .into_owned(),
        ),
    ] {
        assert_eq!(
            provenance_tail(&bytes).0,
            QtipSearchStamp::Greedy.to_wire(),
            "{rung}: greedy bake must stamp greedy"
        );
        let err = match rung {
            "qtip2" => QtipLayer::deserialize_ext_bias(
                Cow::Owned(bytes),
                &device,
                QuantizeOntoGuard::new(),
            )
            .err(),
            _ => Qtip2bLayer::deserialize_ext_bias(
                Cow::Owned(bytes),
                &device,
                QuantizeOntoGuard::new(),
            )
            .err(),
        };
        let msg = err
            .unwrap_or_else(|| panic!("{rung}: a greedy-stamped artifact loaded"))
            .to_string();
        assert!(
            msg.contains("greedy") && msg.contains("no override"),
            "{rung}: refusal must be explicit and final, got: {msg}"
        );
    }
}

/// Legacy policy, refuse half: a pre-0.3.0 payload with no incoherence rotation
/// can only have come from the greedy walk under every bake policy Arc shipped,
/// so it is refused — but overridably, because it is an inference rather than a
/// recorded fact.
#[test]
fn legacy_unstamped_without_rotation_is_refused() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 6, 0.5), (8usize, 64usize), &device).unwrap();
    let greedy =
        QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Greedy, false)
            .unwrap();
    assert_eq!(greedy.rotation_block(), 0);

    // Truncate the whole provenance tail: byte-for-byte a pre-0.3.0 payload.
    let mut bytes = QuantizedSerde::serialize(&greedy).unwrap().into_owned();
    strip_provenance(&mut bytes);

    let msg = QtipLayer::deserialize_ext_bias(
        Cow::Owned(bytes.clone()),
        &device,
        QuantizeOntoGuard::new(),
    )
    .expect_err("an unstamped, unrotated artifact must not be served silently")
    .to_string();
    assert!(
        msg.contains("ARC_ALLOW_UNSTAMPED_QTIP"),
        "the refusal must name its own escape hatch, got: {msg}"
    );
}

/// Legacy policy, serve half: a pre-0.3.0 payload that carries the rotation
/// implies the trellis search under every shipped policy, so refusing it would
/// brick honest artifacts (everything baked between PR #20 and the stamp) for
/// no quality gain. It loads, with a one-time warn.
#[test]
fn legacy_unstamped_with_rotation_is_served() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 7, 0.5), (8usize, 64usize), &device).unwrap();
    let viterbi =
        QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Viterbi, true)
            .unwrap();
    assert!(viterbi.rotation_block() >= 2);

    let mut bytes = QuantizedSerde::serialize(&viterbi).unwrap().into_owned();
    strip_provenance(&mut bytes);

    let (restored, _) =
        QtipLayer::deserialize_ext_bias(Cow::Owned(bytes), &device, QuantizeOntoGuard::new())
            .expect("a rotated pre-0.3.0 artifact must still load");
    drop(restored);
}

/// A layer whose provenance we never learned must not be re-serialized as if it
/// had one — laundering `Unstamped` into `Trellis` would undo the whole point.
#[test]
fn unknown_provenance_cannot_be_laundered_into_a_stamp() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 8, 0.5), (8usize, 64usize), &device).unwrap();
    let viterbi =
        QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Viterbi, true)
            .unwrap();
    let mut bytes = QuantizedSerde::serialize(&viterbi).unwrap().into_owned();
    strip_provenance(&mut bytes); // pre-0.3.0 payload

    let (legacy, _) =
        QtipLayer::deserialize_concrete(Cow::Owned(bytes), &device, QuantizeOntoGuard::new())
            .unwrap();
    assert_eq!(legacy.search_stamp(), QtipSearchStamp::Unstamped);
    assert_eq!(
        legacy.search_detail(),
        QtipSearchDetail::Unknown,
        "an unstamped payload's search detail is unknown too — not a default claim"
    );

    let msg = QuantizedSerde::serialize(&legacy)
        .expect_err("re-serializing an unstamped layer must not invent provenance")
        .to_string();
    assert!(msg.contains("unknown search provenance"), "got: {msg}");
}

/// Wire encoding is pinned: `0` must never parse, so a zero-filled or
/// zero-padded buffer cannot read as a valid stamp.
#[test]
fn stamp_wire_encoding_is_pinned() {
    assert_eq!(QtipSearchStamp::Trellis.to_wire(), Some(1));
    assert_eq!(QtipSearchStamp::Greedy.to_wire(), Some(2));
    assert_eq!(QtipSearchStamp::Unstamped.to_wire(), None);
    assert_eq!(
        QtipSearchStamp::from_wire(1).unwrap(),
        QtipSearchStamp::Trellis
    );
    assert_eq!(
        QtipSearchStamp::from_wire(2).unwrap(),
        QtipSearchStamp::Greedy
    );
    assert!(QtipSearchStamp::from_wire(0).is_err());
    assert!(QtipSearchStamp::from_wire(3).is_err());
    assert_eq!(
        QtipSearchStamp::for_mode(QtipMode::Viterbi),
        QtipSearchStamp::Trellis
    );
    assert_eq!(
        QtipSearchStamp::for_mode(QtipMode::Greedy),
        QtipSearchStamp::Greedy
    );
}

// ---------------------------------------------------------------------------
// Search DETAIL (wave13-AF): the flags byte beside the stamp
// ---------------------------------------------------------------------------
//
// The stamp answers "trellis or greedy". It deliberately cannot answer "which
// trellis search", and a W=64 beam is a genuinely different quality point from
// the exhaustive DP (PR #29: matmul cos 0.95054 vs 0.96495). An artifact that
// cannot tell them apart is a mislabelled artifact, which is the failure class
// the stamp exists to end — so these tests hold the detail to the same standard
// the stamp is held to.
//
// What is and is not checkable, stated plainly: you cannot recover the search
// from the packed symbols, so no test can prove an artifact's width claim
// matches the search that actually ran. What the format CAN guarantee, and what
// is tested here, is that a claim is internally consistent, that it round-trips
// exactly, and that it is never invented at re-serialize time.

/// `for_bake` records the plan that ran, and normalises the two widths that are
/// not pruned searches (0, and anything at or above the state space) rather
/// than recording them as beams.
#[test]
fn search_detail_records_the_plan_that_ran() {
    use super::{TrellisSearch, LUT_SIZE};

    for w in [1usize, 64, 128, 256, LUT_SIZE - 1] {
        assert_eq!(
            QtipSearchDetail::for_bake(QtipMode::Viterbi, TrellisSearch::Beam { width: w }, false)
                .beam_width(),
            Some(w as u16),
            "a beam of width {w} must be recorded as such"
        );
    }
    // Not pruned searches: width 0 is not a search, and a beam at or above the
    // state space prunes nothing and IS the exhaustive DP.
    for w in [0usize, LUT_SIZE, LUT_SIZE + 1] {
        assert_eq!(
            QtipSearchDetail::for_bake(QtipMode::Viterbi, TrellisSearch::Beam { width: w }, false)
                .beam_width(),
            None,
            "width {w} is not a pruned beam and must not be recorded as one"
        );
    }
    assert_eq!(
        QtipSearchDetail::for_bake(QtipMode::Viterbi, TrellisSearch::Exhaustive, false),
        QtipSearchDetail::EXHAUSTIVE_MSE
    );
    // Greedy runs no trellis search, so it can claim neither a beam nor an
    // objective — even if the caller passes one.
    assert_eq!(
        QtipSearchDetail::for_bake(QtipMode::Greedy, TrellisSearch::Beam { width: 64 }, true),
        QtipSearchDetail::EXHAUSTIVE_MSE
    );
}

/// **The round-trip gate.** An exhaustive bake and a beam bake must be
/// distinguishable from the artifact alone, and the width must survive the wire
/// exactly — including the boundary widths.
#[test]
fn beam_width_round_trips_through_uqff_exactly() {
    use super::LUT_SIZE;

    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 21, 0.5), (8usize, 64usize), &device).unwrap();
    let mut layer =
        QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Viterbi, true)
            .unwrap();

    // The bake above ran the exhaustive DP; that is what it recorded.
    assert_eq!(layer.search_detail(), QtipSearchDetail::EXHAUSTIVE_MSE);
    let exhaustive_bytes = QuantizedSerde::serialize(&layer).unwrap().into_owned();

    for (width, hessian) in [
        (1u16, false),
        (64, false),
        (128, true),
        (256, false),
        ((LUT_SIZE - 1) as u16, true),
    ] {
        layer.search_detail = QtipSearchDetail::Known {
            beam_width: Some(width),
            hessian,
        };
        let bytes = QuantizedSerde::serialize(&layer).unwrap().into_owned();

        // Distinguishable from the exhaustive artifact by the bytes alone.
        assert_ne!(
            bytes, exhaustive_bytes,
            "a W={width} beam artifact must not be byte-identical to an exhaustive one"
        );

        let (restored, _) =
            QtipLayer::deserialize_concrete(Cow::Owned(bytes), &device, QuantizeOntoGuard::new())
                .unwrap();
        assert_eq!(
            restored.search_detail(),
            QtipSearchDetail::Known {
                beam_width: Some(width),
                hessian
            },
            "W={width} hessian={hessian} did not round-trip exactly"
        );
        assert_eq!(restored.search_stamp(), QtipSearchStamp::Trellis);
    }
}

/// **A real beam bake, end to end** (wave14-AK).
///
/// `beam_width_round_trips_through_uqff_exactly` above proves the *wire* half:
/// a width, once set, survives serialize/deserialize. It cannot prove the half
/// that actually matters — that asking for a beam produces an artifact which
/// says so — because it stamps a layer that was baked exhaustively. That gap
/// existed only because `ARC_QTIP_BEAM` is memoised in a `OnceLock`, so no test
/// could request a search. [`QtipLayer::quantize_with_bake_config`] closes it.
///
/// This test bakes the SAME weights three ways through the production door and
/// asserts on what comes out — the path a mislabelled artifact would have to
/// travel.
#[test]
fn real_beam_bake_at_w256_stamps_the_width_it_ran() {
    use super::{QtipBakeConfig, QtipCodebook, TrellisSearch};

    let device = Device::Cpu;
    // Deterministic Gaussian; this test is about provenance plumbing, not
    // quality, so the realistic-fixture requirement (D12) is carried by
    // `search_bench.rs` / `bake_quality_tests.rs`, which is where beam-vs-
    // exhaustive quality is actually decided.
    let w = Tensor::from_vec(fixture(8 * 64, 0xB3A3, 0.5), (8usize, 64usize), &device).unwrap();

    let bake = |search: TrellisSearch| {
        QtipLayer::quantize_with_bake_config(
            &w,
            None,
            &device,
            QtipMode::Viterbi,
            true,
            None,
            QtipBakeConfig {
                search,
                hessian: false,
                codebook: QtipCodebook::Gaussian,
            },
        )
        .expect("viterbi bake must succeed")
    };

    let exhaustive = bake(TrellisSearch::Exhaustive);
    let beam256 = bake(TrellisSearch::Beam { width: 256 });

    // 1. The artifact records the search that ran, not a default.
    assert_eq!(exhaustive.search_detail(), QtipSearchDetail::EXHAUSTIVE_MSE);
    assert_eq!(
        beam256.search_detail().beam_width(),
        Some(256),
        "a W=256 bake must stamp beam(W=256)"
    );
    assert_eq!(beam256.search_stamp(), QtipSearchStamp::Trellis);

    // 2. Distinguishable from an exhaustive bake by the bytes alone — the
    //    property that makes "which search made this checkpoint" answerable
    //    from a file rather than from a session log nobody kept.
    let ex_bytes = QuantizedSerde::serialize(&exhaustive).unwrap().into_owned();
    let beam_bytes = QuantizedSerde::serialize(&beam256).unwrap().into_owned();
    assert_ne!(
        beam_bytes, ex_bytes,
        "a real W=256 beam artifact must not be byte-identical to an exhaustive one"
    );
    assert_eq!(provenance_tail(&ex_bytes), (Some(1), 0x00));
    // A beam artifact's tail is 4 bytes, not 2: `[stamp, flags, width_le…]`.
    assert_eq!(
        &beam_bytes[beam_bytes.len() - 4..],
        &[1u8, 0x01, 0x00, 0x01],
        "tail must read: trellis stamp, beam bit set, width 256 little-endian"
    );

    // 3. The width round-trips exactly through UQFF.
    let (restored, _) =
        QtipLayer::deserialize_concrete(Cow::Owned(beam_bytes), &device, QuantizeOntoGuard::new())
            .unwrap();
    assert_eq!(
        restored.search_detail(),
        QtipSearchDetail::Known {
            beam_width: Some(256),
            hessian: false
        }
    );

    // 4. NON-DEGENERACY (D12): the width must reach the SEARCH, not just the
    //    stamp. If `bake_cfg.search` were dropped on the floor between the
    //    caller and `viterbi::quantize_row`, every assertion above would still
    //    pass — the stamp is derived from the request, so a request that never
    //    ran would stamp identically. A W=1 beam keeps one state per timestep
    //    and provably cannot find the exhaustive DP's path on this fixture, so
    //    differing packed symbols are proof the search actually ran.
    let beam1 = bake(TrellisSearch::Beam { width: 1 });
    let packed = |l: &QtipLayer| l.blocks.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    let (ex_syms, b1_syms) = (packed(&exhaustive), packed(&beam1));
    let differing = ex_syms.iter().zip(&b1_syms).filter(|(a, b)| a != b).count();
    assert!(
        differing > 0,
        "a W=1 beam must pack different symbols than the exhaustive DP over \
         {} bytes — identical output means the requested width never reached \
         the trellis search and the beam stamp is decorative",
        ex_syms.len()
    );
    assert_eq!(beam1.search_detail().beam_width(), Some(1));

    // 5. And the env-driven production door is untouched: with no
    //    `ARC_QTIP_BEAM` set in this process, the default config is exhaustive.
    assert_eq!(
        QtipBakeConfig::default().search,
        TrellisSearch::Exhaustive,
        "the default bake is the exhaustive DP; the override door must not \
         change what production picks"
    );
}

/// **A real beam bake on the 3-D EXPERT-STACK path** (wave14-AL).
///
/// The test above covers 2-D linears. This one covers the path
/// `--isq qtip2`/`qtip2b` actually takes on V4 Flash: `[E, N, K]` stacked MoE
/// experts, which are essentially the model's entire weight mass. Until
/// [`QtipLayer::quantize_3d_with_bake_config`] existed, every per-expert chunk
/// went through the memoised `QtipBakeConfig::get()`, so the one code path a
/// paid GPU session spends its money on was the one path no test had ever run
/// with a beam.
///
/// Fixture (D12 — state the distribution AND the dispersion): `gen_fp4_dequant`
/// at sigma 0.02 — Student-t(4) heavy tails snapped to the FP4 (e2m1) magnitude
/// lattice `{0, .5, 1, 1.5, 2, 3, 4, 6}` with a per-32-column block scale. That
/// is the real source chain of V4's experts (INT4-packed FP4 dequantized to
/// BF16): rows are 16-level lattice mixtures, NOT Gaussian. Gaussian fixtures
/// are exactly what hid the original 58.85 defect — greedy+no-rotation scores
/// 0.888 on Gaussian and 0.675 on this distribution.
#[test]
fn real_3d_expert_stack_beam_bake_at_w256_stamps_the_width_it_ran() {
    use super::bake_quality_tests::gen_fp4_dequant;
    use super::{QtipBakeConfig, QtipCodebook, TrellisSearch};

    let device = Device::Cpu;
    let (e, n, k) = (3usize, 4usize, 64usize);
    let w = Tensor::from_vec(
        gen_fp4_dequant(e * n, k, 0.02, 0x3D_BEA3),
        (e, n, k),
        &device,
    )
    .unwrap();

    // The production 3-D door, with the search stated instead of inherited
    // from a `OnceLock`. Rotation on, Viterbi — the production V4 recipe.
    let bake = |search: TrellisSearch| {
        let layer = QtipLayer::quantize_3d_with_bake_config(
            &w,
            &device,
            QtipMode::Viterbi,
            true,
            None,
            QtipBakeConfig {
                search,
                hessian: false,
                codebook: QtipCodebook::Gaussian,
            },
        )
        .expect("3-D viterbi bake must succeed");
        layer.serialize().unwrap().into_owned()
    };

    let ex_bytes = bake(TrellisSearch::Exhaustive);
    let beam_bytes = bake(TrellisSearch::Beam { width: 256 });

    // 1. Distinguishable from an exhaustive bake by the bytes alone. The tail
    //    is checked FIRST so a regression prints four bytes rather than two
    //    whole payload dumps.
    assert_eq!(provenance_tail(&ex_bytes), (Some(1), 0x00));
    // A beam artifact's tail is 4 bytes, not 2: `[stamp, flags, width_le..]`.
    assert_eq!(
        &beam_bytes[beam_bytes.len() - 4..],
        &[1u8, 0x01, 0x00, 0x01],
        "tail must read: trellis stamp, beam bit set, width 256 little-endian \
         — an exhaustive tail here means the requested search never reached \
         the per-expert chunks"
    );
    let differing_payload_bytes = beam_bytes
        .iter()
        .zip(&ex_bytes)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        beam_bytes.len() != ex_bytes.len() || differing_payload_bytes > 0,
        "a real W=256 expert-stack artifact must not be byte-identical to an \
         exhaustive one ({} bytes each)",
        beam_bytes.len()
    );

    // 2. The width round-trips exactly through UQFF, and the artifact is still
    //    a 3-D expert stack on the far side — a 2-D layer that happened to
    //    carry the right stamp would not prove anything about this path.
    let restore = |bytes: Vec<u8>| {
        QtipLayer::deserialize_concrete(Cow::Owned(bytes), &device, QuantizeOntoGuard::new())
            .expect("stamped trellis artifact must load")
            .0
    };
    let beam256 = restore(beam_bytes);
    assert_eq!(
        beam256.search_detail(),
        QtipSearchDetail::Known {
            beam_width: Some(256),
            hessian: false
        },
        "a W=256 expert-stack bake must stamp beam(W=256)"
    );
    assert_eq!(beam256.search_stamp(), QtipSearchStamp::Trellis);
    assert_eq!(beam256.num_experts, Some(e));
    assert_eq!(beam256.blocks.dims(), &[e, n, k / 4]);

    // 3. NON-DEGENERACY (D12): the width must reach the SEARCH, not just the
    //    stamp. The stamp is derived from the *request*, so if `bake_cfg` were
    //    dropped anywhere between this caller and `viterbi::quantize_row`,
    //    every assertion above would still pass on a bake that never ran a
    //    beam. A W=1 beam keeps one state per timestep and cannot reproduce the
    //    exhaustive DP's path on this fixture, so differing packed symbols are
    //    proof the search actually ran.
    let beam1 = restore(bake(TrellisSearch::Beam { width: 1 }));
    let exhaustive = restore(ex_bytes);
    let packed = |l: &QtipLayer| l.blocks.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    let (ex_syms, b1_syms) = (packed(&exhaustive), packed(&beam1));
    assert_eq!(ex_syms.len(), e * n * (k / 4));
    let differing = ex_syms.iter().zip(&b1_syms).filter(|(a, b)| a != b).count();
    assert!(
        differing > 0,
        "a W=1 beam must pack different symbols than the exhaustive DP over \
         {} expert-stack bytes — identical output means the requested width \
         never reached the trellis search and the beam stamp is decorative",
        ex_syms.len()
    );
    assert_eq!(beam1.search_detail().beam_width(), Some(1));

    // 4. EVERY CHUNK, not just the first. `quantize_3d_with_bake_config`
    //    quantizes experts in batches of `ARC_QTIP_EXPERT_BATCH` (default 16),
    //    and only chunk 0's provenance is copied onto the stack; the others are
    //    cross-checked. A config threaded into the first chunk alone would
    //    either trip that check or, if the check were removed, silently stamp
    //    chunk 0's search onto the whole stack. So bake a stack wider than one
    //    default batch and require the requested width to survive. W=1 keeps it
    //    cheap — the point here is reach, not quality.
    let wide_e = 17usize;
    let wide = Tensor::from_vec(
        gen_fp4_dequant(wide_e * 2, k, 0.02, 0x3D_BEA4),
        (wide_e, 2usize, k),
        &device,
    )
    .unwrap();
    let wide_layer = QtipLayer::quantize_3d_with_bake_config(
        &wide,
        &device,
        QtipMode::Viterbi,
        true,
        None,
        QtipBakeConfig {
            search: TrellisSearch::Beam { width: 1 },
            hessian: false,
            codebook: QtipCodebook::Gaussian,
        },
    )
    .expect("a multi-chunk expert stack must bake");
    let wide_restored = restore(wide_layer.serialize().unwrap().into_owned());
    assert_eq!(wide_restored.num_experts, Some(wide_e));
    assert_eq!(
        wide_restored.search_detail().beam_width(),
        Some(1),
        "a {wide_e}-expert stack spans more than one quantize chunk; the \
         requested width must reach the last chunk, not only the first"
    );

    // 5. The env-driven production door is untouched: the default config is
    //    still the exhaustive DP.
    assert_eq!(
        QtipBakeConfig::default().search,
        TrellisSearch::Exhaustive,
        "the default bake is the exhaustive DP; the 3-D override door must not \
         change what production picks"
    );
}

/// A self-contradictory or uninterpretable provenance claim is REFUSED, never
/// normalised into a plausible one. Reading a malformed claim as "exhaustive"
/// is exactly how a mislabelled artifact would get laundered into a trusted one.
#[test]
fn malformed_search_detail_claims_are_rejected_not_laundered() {
    let device = Device::Cpu;
    let w = Tensor::from_vec(fixture(8 * 64, 22, 0.5), (8usize, 64usize), &device).unwrap();
    let layer =
        QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Viterbi, true)
            .unwrap();
    let good = QuantizedSerde::serialize(&layer).unwrap().into_owned();
    // Tail is [stamp=1(trellis), flags=0x00].
    assert_eq!(provenance_tail(&good), (Some(1), 0x00));

    let load = |bytes: Vec<u8>| {
        QtipLayer::deserialize_concrete(Cow::Owned(bytes), &device, QuantizeOntoGuard::new())
    };

    // (a) Reserved bits set — written by a newer Arc whose provenance fields
    //     this build cannot interpret. Refuse rather than guess.
    for reserved in [0x04u8, 0x40, 0x80, 0xFC] {
        let mut bytes = good.clone();
        *bytes.last_mut().unwrap() = reserved;
        let msg = load(bytes)
            .expect_err("reserved provenance bits must not be ignored")
            .to_string();
        assert!(msg.contains("reserved bits"), "{msg}");
    }

    // (b) Beam bit set with a width that is not a pruned search.
    let mut bytes = good.clone();
    *bytes.last_mut().unwrap() = 0x01;
    bytes.extend_from_slice(&0u16.to_le_bytes());
    let msg = load(bytes)
        .expect_err("a beam of width 0 is not a search")
        .to_string();
    assert!(msg.contains("pruned search"), "{msg}");

    // (c) Beam bit set but the width is missing — truncated payload.
    let mut bytes = good.clone();
    *bytes.last_mut().unwrap() = 0x01;
    let msg = load(bytes)
        .expect_err("a beam claim with no width is truncated, not exhaustive")
        .to_string();
    assert!(
        msg.contains("width is") || msg.contains("truncated"),
        "{msg}"
    );

    // (d) A `greedy` stamp claiming a beam width: two halves of the artifact
    //     contradicting each other. This is the strongest form of "a stamp
    //     claiming a search the payload did not use" that is detectable at all.
    let mut bytes = good.clone();
    let n = bytes.len();
    bytes[n - 2] = 2; // greedy
    bytes[n - 1] = 0x01; // ...but claims a beam
    bytes.extend_from_slice(&256u16.to_le_bytes());
    let msg = load(bytes)
        .expect_err("greedy + beam width is self-contradictory")
        .to_string();
    assert!(msg.contains("contradicts itself"), "{msg}");
    // ...and the same contradiction is refused on the WRITE side, so it can
    // never be produced in the first place.
    let mut bad =
        QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Viterbi, true)
            .unwrap();
    bad.search = QtipSearchStamp::Greedy;
    bad.search_detail = QtipSearchDetail::Known {
        beam_width: Some(256),
        hessian: false,
    };
    let msg = QuantizedSerde::serialize(&bad)
        .expect_err("a greedy stamp must not be writable with a beam width")
        .to_string();
    assert!(msg.contains("cannot carry a beam width"), "{msg}");

    // (e) Stamp present, flags byte missing: a truncated 0.3.0 payload. It must
    //     fail closed, NOT read as an exhaustive unweighted bake. (Dropping the
    //     stamp too is a genuine pre-0.3.0 payload and stays legal — that
    //     distinction is the whole reason the flags byte is mandatory.)
    let mut bytes = good.clone();
    bytes.pop();
    let msg = load(bytes)
        .expect_err("a stamped payload with no flags byte is truncated")
        .to_string();
    assert!(msg.contains("no search-detail"), "{msg}");

    // (f) Unknown detail is never invented at re-serialize time — the same
    //     no-laundering rule the stamp already enforces.
    let mut unknown =
        QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Viterbi, true)
            .unwrap();
    unknown.search_detail = QtipSearchDetail::Unknown;
    let msg = QuantizedSerde::serialize(&unknown)
        .expect_err("unknown search detail must not be written as a specific claim")
        .to_string();
    assert!(msg.contains("never recorded"), "{msg}");
}
