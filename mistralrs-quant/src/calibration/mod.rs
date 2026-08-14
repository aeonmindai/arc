//! Calibration statistics: collection and the `.arccalib` artifact.
//!
//! Three separate quality levers all need the same input — per-linear-layer
//! activation statistics gathered from a calibration corpus:
//!
//! - **activation-aware trellis search** (QTIP) wants per-row weights, ideally
//!   `diag(XᵀX)` or a blockwise Hessian;
//! - **TD-MoE whitening** needs a real activation covariance (it currently
//!   whitens against the identity, which makes the whitening a no-op);
//! - **EoRA-style low-rank error correction** needs the same Hessian.
//!
//! This module produces those statistics once and writes them to a versioned,
//! self-describing artifact keyed by the *ISQ layer index*, so every consumer
//! looks the same layer up with the ordinal it already has.
//!
//! - [`collector`] — the accumulator that layers feed during a forward-only
//!   sweep.
//! - [`artifact`] — the `.arccalib` file format plus its loader API.

pub mod artifact;
pub mod collector;

pub use artifact::{
    CalibrationArtifact, CalibrationMeta, LayerCalibStats, CALIB_COLLECTOR_VERSION,
    CALIB_EXTENSION, CALIB_FORMAT_VERSION, CALIB_MAGIC,
};
pub use collector::{
    CalibAccumulator, CalibLayerData, CalibOptions, ExpertCalibData, ExpertStatus, GramBlocks,
    GramLayout, GramMode,
};
