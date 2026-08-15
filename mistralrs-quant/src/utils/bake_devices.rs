//! Which CUDA devices a UQFF bake may spread its per-layer quantize work
//! across (wave22).
//!
//! # Why this exists
//!
//! ISQ layers are *independent*: quantizing layer 7 needs nothing produced by
//! layer 6. A bake has therefore always been embarrassingly parallel and has
//! always run on exactly one device — 373.6 s/layer x 43 layers = 4.5 h on one
//! A100-80G. Spreading the layers over the N devices already present in the box
//! turns that into ~4.5/N h for the same GPU-hours, because the expensive
//! bootstrap (checkpoint download, model load) happens once per *box*, not once
//! per device.
//!
//! # The invariant this feature is built on
//!
//! The artifact a bake produces must be **byte-identical** whether it ran on
//! one device or on N. That is only possible if the quantizer's output depends
//! on the *weights*, not on which device the kernels ran on. It already does:
//! the trellis search is deterministic and its result is packed identically
//! from any device. This module never relaxes that requirement — it just names
//! the devices. See `parallel_bake_tests` in `mistralrs-core` for the
//! round-trip that pins it.
//!
//! Crucially, nothing downstream of the quantize learns that the bake was
//! parallel: the layer list, its order, the `get_layers()` ordinals that become
//! UQFF tensor names, and the shard packing are all untouched. There is no
//! merge step, so there is no merge contract to get wrong.

use std::cell::Cell;
use std::sync::{Mutex, OnceLock};

/// Environment variable naming the CUDA ordinals a UQFF bake may use, e.g.
/// `ARC_BAKE_DEVICES=0,1,2,3`. The `--bake-devices` CLI flag sets the same
/// list through [`set_bake_device_ordinals`], which takes precedence.
pub const BAKE_DEVICES_ENV: &str = "ARC_BAKE_DEVICES";

/// Parse a bake device list such as `"0,1,2,3"`.
///
/// Pure and total so the CLI can reject a bad list before a 149 GB download
/// rather than after it. Whitespace around entries is ignored.
///
/// Duplicates are an error, not a no-op: two workers pinned to one device is
/// exactly the session-5 trap that `isq_thread_policy` exists to prevent (24
/// host threads on one device measured 4-9 min/layer against ~30 s/layer at
/// one thread), and silently deduplicating would hide a typo that halves the
/// user's expected throughput.
pub fn parse_bake_devices(spec: &str) -> Result<Vec<usize>, String> {
    let mut ordinals = Vec::new();
    for field in spec.split(',') {
        let field = field.trim();
        if field.is_empty() {
            return Err(format!(
                "empty device ordinal in `{spec}`; expected a comma-separated list like `0,1,2,3`"
            ));
        }
        let ordinal: usize = field.parse().map_err(|_| {
            format!("`{field}` is not a CUDA device ordinal; expected a list like `0,1,2,3`")
        })?;
        if ordinals.contains(&ordinal) {
            return Err(format!(
                "CUDA device {ordinal} is listed twice in `{spec}`; each device may appear once \
                 (two bake workers on one device only contend for it)"
            ));
        }
        ordinals.push(ordinal);
    }
    if ordinals.is_empty() {
        return Err(format!(
            "no device ordinals in `{spec}`; expected a list like `0,1,2,3`"
        ));
    }
    Ok(ordinals)
}

/// Programmatic override installed by `--bake-devices`.
///
/// Three states, and they are all needed:
/// * outer `None` — nothing was set programmatically, so [`BAKE_DEVICES_ENV`]
///   decides. This is the default and what every existing run gets.
/// * `Some(None)` — an explicit "one device", which *beats* the environment. An
///   inherited `ARC_BAKE_DEVICES` in a shell profile must not silently
///   reinterpret a run that asked for single-device behaviour, and tests need a
///   hermetic way to say so.
/// * `Some(Some(list))` — an explicit list, which also beats the environment.
fn override_slot() -> &'static Mutex<Option<Option<Vec<usize>>>> {
    static OVERRIDE: OnceLock<Mutex<Option<Option<Vec<usize>>>>> = OnceLock::new();
    OVERRIDE.get_or_init(|| Mutex::new(None))
}

/// Install the bake device list programmatically (used by `--bake-devices`),
/// taking precedence over [`BAKE_DEVICES_ENV`]. `None` means an explicit
/// single-device bake; see [`clear_bake_device_override`] to fall back to the
/// environment again.
pub fn set_bake_device_override(ordinals: Option<Vec<usize>>) {
    *override_slot().lock().expect("bake device lock poisoned") = Some(ordinals);
}

/// Drop the programmatic override so [`BAKE_DEVICES_ENV`] decides again.
pub fn clear_bake_device_override() {
    *override_slot().lock().expect("bake device lock poisoned") = None;
}

/// The CUDA ordinals this bake may spread layers across, or `None` for the
/// historical single-device behaviour.
///
/// A malformed [`BAKE_DEVICES_ENV`] is rejected loudly and treated as unset: a
/// typo must not silently serialize a bake the user asked to parallelize, and
/// it must not abort a 4-hour job either. (`--bake-devices` is validated up
/// front by the CLI instead, before the checkpoint download.)
pub fn bake_device_ordinals() -> Option<Vec<usize>> {
    if let Some(explicit) = override_slot()
        .lock()
        .expect("bake device lock poisoned")
        .clone()
    {
        return explicit;
    }
    let spec = std::env::var(BAKE_DEVICES_ENV).ok()?;
    match parse_bake_devices(&spec) {
        Ok(ordinals) => Some(ordinals),
        Err(e) => {
            tracing::warn!(
                "Ignoring {BAKE_DEVICES_ENV}=`{spec}`: {e}. Baking on the single mapped device."
            );
            None
        }
    }
}

thread_local! {
    /// CUDA ordinal this ISQ worker thread owns for the duration of one layer.
    static WORKER_CUDA_ORDINAL: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Pin the calling ISQ worker thread to a CUDA ordinal for the layer it is
/// about to quantize, so device-selecting fallbacks deeper in the quantizer
/// (see `expert_stack_quant_device`) route to *this* worker's device instead of
/// unconditionally to device 0.
///
/// Thread-local on purpose: it is read from inside `apply_isq`, which runs on
/// whichever rayon worker picked the layer up, and each worker owns exactly one
/// device for the whole bake.
pub fn set_worker_cuda_ordinal(ordinal: Option<usize>) {
    WORKER_CUDA_ORDINAL.with(|c| c.set(ordinal));
}

/// The CUDA ordinal pinned by [`set_worker_cuda_ordinal`], if any.
pub fn worker_cuda_ordinal() -> Option<usize> {
    WORKER_CUDA_ORDINAL.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_plain_list() {
        assert_eq!(parse_bake_devices("0,1,2,3").unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parses_a_single_device() {
        assert_eq!(parse_bake_devices("2").unwrap(), vec![2]);
    }

    #[test]
    fn tolerates_whitespace() {
        assert_eq!(parse_bake_devices(" 0 , 1 ").unwrap(), vec![0, 1]);
    }

    #[test]
    fn preserves_the_given_order() {
        // The order decides which worker index owns which device; it is the
        // user's to choose (e.g. to skip a device already busy with a serve).
        assert_eq!(parse_bake_devices("3,1").unwrap(), vec![3, 1]);
    }

    #[test]
    fn rejects_duplicates() {
        let err = parse_bake_devices("0,1,0").unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    #[test]
    fn rejects_non_numeric() {
        assert!(parse_bake_devices("0,cuda1").is_err());
    }

    #[test]
    fn rejects_empty_and_trailing_commas() {
        assert!(parse_bake_devices("").is_err());
        assert!(parse_bake_devices("0,").is_err());
        assert!(parse_bake_devices(",0").is_err());
    }

    #[test]
    fn an_explicit_override_beats_the_environment() {
        // An `ARC_BAKE_DEVICES` inherited from a shell profile must not
        // reinterpret a run that explicitly asked for one device.
        set_bake_device_override(Some(vec![2, 3]));
        assert_eq!(bake_device_ordinals(), Some(vec![2, 3]));
        set_bake_device_override(None);
        assert_eq!(bake_device_ordinals(), None);
        clear_bake_device_override();
    }

    #[test]
    fn worker_ordinal_defaults_to_unset() {
        // A fresh thread must not inherit another worker's device.
        std::thread::spawn(|| assert_eq!(worker_cuda_ordinal(), None))
            .join()
            .unwrap();
    }

    #[test]
    fn worker_ordinal_is_thread_local() {
        set_worker_cuda_ordinal(Some(7));
        assert_eq!(worker_cuda_ordinal(), Some(7));
        std::thread::spawn(|| assert_eq!(worker_cuda_ordinal(), None))
            .join()
            .unwrap();
        set_worker_cuda_ordinal(None);
        assert_eq!(worker_cuda_ordinal(), None);
    }
}
