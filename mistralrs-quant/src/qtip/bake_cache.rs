//! Parent system: ArcQuant / ArcBake — on-disk cache for benchmark fixtures.
//!
//! # Why this exists
//!
//! A grouped-GEMM kernel A/B on the V4-Flash MoE shapes (E=256, 2048x4096 and
//! 4096x2048) spends **~95% of its GPU time re-deriving a fixture that is
//! byte-identical every run**: `qtip_grouped_curve` goes through the
//! production quantizer door — `quantize_with_mode(Viterbi)`, because D4 bans
//! `Greedy` outside `cfg(test)` — and that Viterbi bake is minutes per shape
//! while the measurement it feeds is under a second. Measured on an H200 on
//! 2026-08-18: **~13 minutes of bake before the first timing line printed.**
//!
//! That is not a tidiness problem. We are GPU-budget-constrained, so an
//! instrument ~20x more expensive than its own measurement is the reason the
//! keystone kernel has been swept so rarely. Caching the bake makes a kernel
//! sweep cost under a minute of card.
//!
//! # What makes a fixture cache safe
//!
//! A wrong-but-plausible fixture produces a clean, confident, meaningless A/B
//! — worse than no run at all. So:
//!
//! 1. **The key is stamped INSIDE the file and verified on load.** A cache
//!    that trusts its own path is one rename (or one stale `--experts`) away
//!    from silently serving a different fixture.
//!
//! 2. **`mode` is part of the key.** D4 in cache form: a `Viterbi` bake must
//!    never be served to a caller that asked for anything else, and the
//!    payload additionally carries the [`QtipSearchStamp`] that
//!    `deserialize_concrete` enforces at load.
//! 3. **A miss is loud and falls through to a real bake. A MISMATCH is an
//!    error, never a fallthrough.** Silently re-baking over a disagreeing
//!    stamp would hide exactly the bug the stamp exists to catch, so callers
//!    are expected to treat [`BakeCacheError::Mismatch`] as an environment
//!    failure (exit 2), not as a result.
//!
//! The payload is the layer's own UQFF serialization, so the cache inherits
//! the wire format's self-describing tensor shapes and its D4 load gate
//! rather than inventing a second format that could drift from it.

use candle_core::{Device, Result};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

use super::bitshift::Qtip2bLayer;
use super::QtipMode;
use crate::{QuantizeOntoGuard, QuantizedSerde};

/// File magic + format version. Bumping the version invalidates every cached
/// fixture, which is the correct behaviour when the payload layout changes.
const BAKE_CACHE_MAGIC: &[u8; 8] = b"ARCBAKE1";

/// Identity of a baked fixture. Every field that can change the resulting
/// bytes belongs here; anything missing is a way to serve the wrong fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakeKey {
    /// Expert count (`E`), or 1 for a plain 2-D layer.
    pub experts: usize,
    /// Output rows per expert.
    pub n: usize,
    /// Input features (`K`).
    pub k: usize,
    /// Seed of the weight draw the fixture was baked from.
    pub seed: u64,
    /// Quantizer mode. Part of the key so a Viterbi bake can never be handed
    /// to a caller asking for another mode (D4 in cache form).
    pub mode: QtipMode,
}

impl BakeKey {
    /// Canonical stamp text. This exact string is written into the file and
    /// compared byte-for-byte on load — it is the identity, not the filename.
    pub fn stamp(&self) -> String {
        format!(
            "qtip2b/v1 E={} N={} K={} seed={:#018x} mode={:?}",
            self.experts, self.n, self.k, self.seed, self.mode
        )
    }

    /// Convenience filename. **Never trusted** — a file found at this path is
    /// still rejected unless its internal stamp matches.
    pub fn file_name(&self) -> String {
        format!(
            "qtip2b_E{}_N{}_K{}_s{:016x}_{:?}.bake",
            self.experts, self.n, self.k, self.seed, self.mode
        )
    }

    /// Full path of this key's fixture inside `dir`.
    pub fn path_in(&self, dir: &Path) -> PathBuf {
        dir.join(self.file_name())
    }
}

/// Why a cache load did not produce a layer.
#[derive(Debug)]
pub enum BakeCacheError {
    /// No file for this key. Callers bake and [`store`] the result.
    Miss,
    /// A file exists but does not describe this fixture. **Not** a miss:
    /// something is wrong with the cache directory and re-baking over it
    /// would hide it. Callers should exit 2.
    Mismatch { found: String, want: String },
    /// The file exists and claims to be ours but could not be read back.
    Corrupt(candle_core::Error),
}

impl std::fmt::Display for BakeCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Miss => write!(f, "bake cache MISS (no file for this key)"),
            Self::Mismatch { found, want } => write!(
                f,
                "bake cache STAMP MISMATCH — the file exists but describes a different fixture.\n\
                 file says: {found}\n\
                 asked for: {want}\n\
                 Refusing to serve it and refusing to silently re-bake over it: a fixture that is \
                 wrong but plausible yields a clean, confident, meaningless measurement. Delete \
                 the file deliberately if that is what you mean."
            ),
            Self::Corrupt(e) => write!(f, "bake cache file is unreadable: {e}"),
        }
    }
}

/// Serialize `layer` under `key` into `dir`, creating `dir` if needed.
///
/// Writes to a temporary file and renames, so an interrupted store can never
/// leave a half-written fixture that a later run would load as real.
pub fn store(dir: &Path, key: &BakeKey, layer: &dyn QuantizedSerde) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(candle_core::Error::wrap)?;
    let payload = layer.serialize()?;
    let stamp = key.stamp();
    let stamp_bytes = stamp.as_bytes();

    let mut out = Vec::with_capacity(payload.len() + stamp_bytes.len() + 16);
    out.extend_from_slice(BAKE_CACHE_MAGIC);
    out.extend_from_slice(&(stamp_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(stamp_bytes);
    out.extend_from_slice(&payload);

    let path = key.path_in(dir);
    let tmp = path.with_extension("bake.tmp");
    std::fs::write(&tmp, &out).map_err(candle_core::Error::wrap)?;
    std::fs::rename(&tmp, &path).map_err(candle_core::Error::wrap)?;
    Ok(path)
}

/// Load the fixture for `key` from `dir`.
///
/// Resolves THREE states, never two: a layer, a [`BakeCacheError::Miss`] that
/// callers fall through to a real bake, and a [`BakeCacheError::Mismatch`]
/// that callers must treat as an environment failure.
pub fn load(
    dir: &Path,
    key: &BakeKey,
    device: &Device,
) -> std::result::Result<Qtip2bLayer, BakeCacheError> {
    let path = key.path_in(dir);
    let raw = match std::fs::read(&path) {
        Ok(r) => r,
        Err(_) => return Err(BakeCacheError::Miss),
    };
    let want = key.stamp();

    // A truncated or foreign file is a MISMATCH, not a miss: something put a
    // file at our exact path, and quietly overwriting it would hide that.
    let head = BAKE_CACHE_MAGIC.len() + 4;
    if raw.len() < head || &raw[..BAKE_CACHE_MAGIC.len()] != BAKE_CACHE_MAGIC {
        return Err(BakeCacheError::Mismatch {
            found: format!(
                "<not an {} file>",
                String::from_utf8_lossy(BAKE_CACHE_MAGIC)
            ),
            want,
        });
    }
    let slen = u32::from_le_bytes(
        raw[BAKE_CACHE_MAGIC.len()..head]
            .try_into()
            .expect("4 bytes"),
    ) as usize;
    if raw.len() < head + slen {
        return Err(BakeCacheError::Mismatch {
            found: "<truncated stamp>".to_string(),
            want,
        });
    }
    let found = String::from_utf8_lossy(&raw[head..head + slen]).to_string();
    if found != want {
        return Err(BakeCacheError::Mismatch { found, want });
    }

    // The payload's own D4 load gate runs inside `deserialize_concrete`.
    let (layer, _bias) = Qtip2bLayer::deserialize_concrete(
        Cow::Borrowed(&raw[head + slen..]),
        device,
        QuantizeOntoGuard::new(),
    )
    .map_err(BakeCacheError::Corrupt)?;
    Ok(layer)
}

/// Store `layer`, load it straight back, and require the round trip to be
/// byte-identical **over the whole serialized payload** — packed blocks, row
/// scales, rotation signs, block size and MCG multiplier alike.
///
/// Whole-payload equality is strictly stronger than comparing the two weight
/// tensors: a cache that reproduced `blocks` and `row_scales` but dropped the
/// rotation signs would pass a two-tensor check and then decode every weight
/// in the wrong frame. Callers run this the first time they populate a key,
/// so no later run can depend on a fixture that was never proved to reload.
pub fn store_and_verify(
    dir: &Path,
    key: &BakeKey,
    layer: &dyn QuantizedSerde,
    device: &Device,
) -> Result<(PathBuf, usize)> {
    let path = store(dir, key, layer)?;
    let want = layer.serialize()?.into_owned();
    let back = load(dir, key, device).map_err(|e| candle_core::Error::msg(format!("{e}")))?;
    let got = back.serialize()?.into_owned();
    if got != want {
        let ndiff = want.iter().zip(got.iter()).filter(|(a, b)| a != b).count()
            + want.len().abs_diff(got.len());
        candle_core::bail!(
            "bake cache round trip is NOT byte-identical for {}: {ndiff} of {} payload bytes \
             differ (lengths {} -> {}). Refusing to leave a fixture on disk that does not \
             reproduce the bake it replaces.",
            key.stamp(),
            want.len(),
            want.len(),
            got.len()
        );
    }
    Ok((path, want.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> BakeKey {
        BakeKey {
            experts: 4,
            n: 32,
            k: 64,
            seed: 0x5EED_1006_C0DE_2B00,
            mode: QtipMode::Viterbi,
        }
    }

    /// The stamp must move when ANY keyed field moves — otherwise two
    /// different fixtures share an identity and the cache can serve the wrong
    /// one while every check passes.
    #[test]
    fn stamp_separates_every_keyed_field() {
        let base = key();
        let mut seen = std::collections::HashSet::new();
        seen.insert(base.stamp());

        let variants = [
            BakeKey {
                experts: 8,
                ..base.clone()
            },
            BakeKey {
                n: 64,
                ..base.clone()
            },
            BakeKey {
                k: 128,
                ..base.clone()
            },
            BakeKey {
                seed: 1,
                ..base.clone()
            },
            BakeKey {
                mode: QtipMode::Greedy,
                ..base.clone()
            },
        ];
        for v in &variants {
            assert!(
                seen.insert(v.stamp()),
                "stamp collision: {:?} and the base key produce the same stamp {:?} — the cache \
                 could serve one for the other",
                v,
                v.stamp()
            );
            assert_ne!(
                v.file_name(),
                base.file_name(),
                "file name collision for {v:?}"
            );
        }
        assert_eq!(seen.len(), variants.len() + 1);
    }

    /// D12: a file whose stamp disagrees must be REJECTED, and rejected as a
    /// `Mismatch` rather than a `Miss` — a miss would silently re-bake over
    /// it and hide the disagreement.
    #[test]
    fn disagreeing_stamp_is_a_mismatch_not_a_miss() {
        let dir = std::env::temp_dir().join(format!(
            "arc-bake-cache-test-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let k = key();

        // Hand-build a file at the RIGHT path carrying the WRONG stamp: this
        // is the rename / stale-fixture scenario the stamp exists to catch.
        let wrong = BakeKey {
            experts: 999,
            ..k.clone()
        }
        .stamp();
        let mut raw = Vec::new();
        raw.extend_from_slice(BAKE_CACHE_MAGIC);
        raw.extend_from_slice(&(wrong.len() as u32).to_le_bytes());
        raw.extend_from_slice(wrong.as_bytes());
        raw.extend_from_slice(b"payload that must never be parsed");
        std::fs::write(k.path_in(&dir), &raw).expect("write");

        match load(&dir, &k, &Device::Cpu) {
            Err(BakeCacheError::Mismatch { found, want }) => {
                assert_eq!(found, wrong);
                assert_eq!(want, k.stamp());
            }
            Err(other) => panic!("expected Mismatch, got {other}"),
            Ok(_) => panic!(
                "GUARD IS BLIND: a file carrying a DIFFERENT fixture's stamp was accepted, so the \
                 cache cannot tell fixtures apart and any A/B built on it proves nothing."
            ),
        }

        // ...and a genuinely absent file is a Miss, so the two states are
        // actually distinguished rather than everything being an error.
        std::fs::remove_file(k.path_in(&dir)).expect("rm");
        assert!(matches!(
            load(&dir, &k, &Device::Cpu),
            Err(BakeCacheError::Miss)
        ));

        // A file that is not ours at all is also a Mismatch, never a Miss.
        std::fs::write(k.path_in(&dir), b"totally unrelated bytes").expect("write2");
        assert!(matches!(
            load(&dir, &k, &Device::Cpu),
            Err(BakeCacheError::Mismatch { .. })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
