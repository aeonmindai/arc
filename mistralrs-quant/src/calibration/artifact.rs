//! `.arccalib` — the Arc calibration-statistics artifact.
//!
//! # Layout
//!
//! ```text
//! magic        8 bytes   b"ARCCALIB"
//! version      u32 LE    [ UNSPECIFIED ][ MAJOR ][ MINOR ][ PATCH ]
//! header_len   u64 LE    byte length of the JSON header
//! header       bytes     UTF-8 JSON: { "meta": {...}, "layers": [ ... ] }
//! payload      bytes     f64 LE arrays, referenced by (offset, len) from the header
//! ```
//!
//! The header is self-describing JSON — `stats_info` (and any external tool)
//! can read the metadata and the full per-layer inventory without touching the
//! payload. Bulk float arrays live in the payload so the header stays small and
//! cheap to parse. Versioning follows the UQFF convention in
//! [`crate::utils`]: the major version must match exactly, and a minor version
//! newer than this build is rejected with a "please update" error.
//!
//! # Layer identity — the consumer contract
//!
//! Layers are keyed by **`isq_index`**: the position of the layer in the
//! model's `IsqModel::get_layers()` list. That is the exact same ordinal the
//! quantizer uses to index imatrix weights and to name UQFF tensors, so a
//! consumer holding an ISQ layer list can look up its statistics with the index
//! it already has. `artifact_name` mirrors the UQFF tensor name for the same
//! position (`"12"`, or `"mtp.3"` for the optional MTP decoder tail), and
//! `name` carries the model's symbolic name from `IsqModel::imatrix_names()`
//! when it provides one.

use std::{
    fs,
    io::{Cursor, Read},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use candle_core::Result;
use serde::{Deserialize, Serialize};

use super::collector::{
    CalibLayerData, CalibOptions, ExpertCalibData, ExpertStatus, GramBlocks, GramLayout,
};

/// File magic. 8 bytes so the header stays 8-byte aligned.
pub const CALIB_MAGIC: &[u8; 8] = b"ARCCALIB";

/// Canonical file extension.
pub const CALIB_EXTENSION: &str = "arccalib";

const CALIB_VERSION_MAJOR: u32 = 0;
const CALIB_VERSION_MINOR: u32 = 1;
const CALIB_VERSION_PATCH: u32 = 0;

// v0.1.0: initial release. diag(XᵀX), optional full/blockwise gram, optional
//         per-expert diagonals with explicit zero-token marking.

/// Format version, 4 bytes LE: `[ UNSPECIFIED ][ MAJOR ][ MINOR ][ PATCH ]`.
pub const CALIB_FORMAT_VERSION: u32 =
    (CALIB_VERSION_MAJOR << (8 * 2)) | (CALIB_VERSION_MINOR << 8) | CALIB_VERSION_PATCH;

/// Bumped whenever the *semantics* of what the collector accumulates change
/// (as opposed to the file layout). Recorded in the metadata so a consumer can
/// refuse statistics gathered by an incompatible collector.
pub const CALIB_COLLECTOR_VERSION: u32 = 1;

fn version_is_compatible(version: u32) -> Result<()> {
    let major = version >> (8 * 2);
    let minor = (version >> 8) & 0xFF;
    let patch = version & 0xFF;
    if major != CALIB_VERSION_MAJOR {
        candle_core::bail!(
            "Major version of calibration artifact ({major}) does not match this build ({CALIB_VERSION_MAJOR})"
        );
    }
    if minor > CALIB_VERSION_MINOR {
        candle_core::bail!(
            "Calibration artifact version {major}.{minor}.{patch} is newer than this build supports \
             ({CALIB_VERSION_MAJOR}.{CALIB_VERSION_MINOR}.{CALIB_VERSION_PATCH}). Please update Arc."
        );
    }
    Ok(())
}

/// Provenance and collection settings for a whole artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMeta {
    /// HuggingFace repo id or local path the statistics were collected from.
    pub model_id: String,
    /// Commit sha / revision of the model, when known.
    #[serde(default)]
    pub model_sha: Option<String>,
    /// Architecture string (`NormalLoaderType`), when known.
    #[serde(default)]
    pub arch: Option<String>,
    /// ISQ organization the layer list was taken from (`default` / `moqe`).
    pub isq_organization: String,
    /// Path of the calibration corpus.
    #[serde(default)]
    pub calibration_file: Option<String>,
    /// Number of forward chunks ("samples") swept.
    pub samples: usize,
    /// Tokens per chunk.
    pub seq_len: usize,
    /// Total calibration tokens fed to the model.
    pub total_tokens: u64,
    /// Model compute dtype during collection (e.g. `BF16`).
    pub model_dtype: String,
    /// Dtype of every float array in the payload. Always `F64` in v0.1.
    pub storage_dtype: String,
    /// See [`CALIB_COLLECTOR_VERSION`].
    pub collector_version: u32,
    /// Crate version of the build that produced the artifact.
    pub producer: String,
    /// Unix seconds at write time.
    pub created_unix: u64,
    /// The options the sweep ran with.
    pub options: CalibOptions,
    /// Number of ISQ layers in the model (including unsupported ones).
    pub isq_layer_count: usize,
}

/// Per-layer statistics, keyed to the ISQ layer list. See the module docs for
/// the naming contract.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerCalibStats {
    /// Position in `IsqModel::get_layers()`.
    pub isq_index: usize,
    /// UQFF tensor name for this position (`"12"` / `"mtp.3"`).
    pub artifact_name: String,
    /// Symbolic name from `IsqModel::imatrix_names()`, when available.
    pub name: Option<String>,
    /// Device-mapper layer number, when the model reports one.
    pub layer_num: Option<usize>,
    /// `false` when the layer's quant method does not support activation
    /// tracking (e.g. weights already loaded in a quantized format). The entry
    /// is still emitted so the layer inventory matches the ISQ layer set
    /// exactly, but carries no statistics.
    pub supported: bool,
    pub in_features: usize,
    pub tokens: u64,
    pub calls: u64,
    /// Raw `diag(XᵀX)`. Empty when `!supported`.
    pub diag: Vec<f64>,
    pub gram: Option<GramBlocks>,
    pub experts: Vec<ExpertCalibData>,
}

impl LayerCalibStats {
    /// Build an entry from collected data plus its identity in the ISQ list.
    pub fn from_data(
        isq_index: usize,
        artifact_name: String,
        name: Option<String>,
        layer_num: Option<usize>,
        data: CalibLayerData,
    ) -> Self {
        Self {
            isq_index,
            artifact_name,
            name,
            layer_num,
            supported: true,
            in_features: data.in_features,
            tokens: data.tokens,
            calls: data.calls,
            diag: data.diag,
            gram: data.gram,
            experts: data.experts,
        }
    }

    /// Placeholder for a layer whose quant method cannot track activations.
    pub fn unsupported(
        isq_index: usize,
        artifact_name: String,
        name: Option<String>,
        layer_num: Option<usize>,
    ) -> Self {
        Self {
            isq_index,
            artifact_name,
            name,
            layer_num,
            supported: false,
            in_features: 0,
            tokens: 0,
            calls: 0,
            diag: Vec::new(),
            gram: None,
            experts: Vec::new(),
        }
    }

    /// `diag(XᵀX) / tokens` — the per-input-channel second moment (a diagonal
    /// activation covariance). `None` when the layer has no usable statistics.
    pub fn normalized_diag(&self) -> Option<Vec<f64>> {
        if !self.supported || self.tokens == 0 || self.diag.is_empty() {
            return None;
        }
        let inv = 1.0 / self.tokens as f64;
        Some(self.diag.iter().map(|v| v * inv).collect())
    }

    /// Per-expert statistics for `expert`, if collected.
    pub fn expert(&self, expert: usize) -> Option<&ExpertCalibData> {
        self.experts.iter().find(|e| e.expert == expert)
    }

    /// Experts that saw no routed tokens during collection.
    pub fn zero_token_experts(&self) -> Vec<usize> {
        self.experts
            .iter()
            .filter(|e| e.status == ExpertStatus::ZeroTokens)
            .map(|e| e.expert)
            .collect()
    }
}

/// A loaded (or freshly collected) calibration artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationArtifact {
    pub meta: CalibrationMeta,
    /// Sorted by `isq_index`.
    pub layers: Vec<LayerCalibStats>,
}

impl CalibrationArtifact {
    /// Build an artifact, sorting layers into ISQ order.
    pub fn new(meta: CalibrationMeta, mut layers: Vec<LayerCalibStats>) -> Self {
        layers.sort_by_key(|l| l.isq_index);
        Self { meta, layers }
    }

    /// Look up a layer by its position in `IsqModel::get_layers()` — the
    /// primary consumer entry point.
    pub fn by_isq_index(&self, isq_index: usize) -> Option<&LayerCalibStats> {
        self.layers
            .binary_search_by_key(&isq_index, |l| l.isq_index)
            .ok()
            .map(|i| &self.layers[i])
    }

    /// Look up a layer by its UQFF tensor name (`"12"` / `"mtp.3"`).
    pub fn by_artifact_name(&self, name: &str) -> Option<&LayerCalibStats> {
        self.layers.iter().find(|l| l.artifact_name == name)
    }

    /// Look up a layer by its symbolic (`imatrix_names`) name.
    pub fn by_name(&self, name: &str) -> Option<&LayerCalibStats> {
        self.layers.iter().find(|l| l.name.as_deref() == Some(name))
    }

    /// The artifact's layer inventory in ISQ order. Compare against the
    /// quantizer's own naming to verify the integration contract.
    pub fn artifact_names(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|l| l.artifact_name.clone())
            .collect()
    }

    /// Layers that actually carry statistics.
    pub fn supported_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.supported).count()
    }

    /// Write the artifact to `path`. The extension is not enforced, but
    /// [`CALIB_EXTENSION`] is the convention.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut payload: Vec<u8> = Vec::new();
        let mut push = |data: &[f64]| -> Result<ArrayRef> {
            let offset = (payload.len() / 8) as u64;
            for v in data {
                payload.write_f64::<LittleEndian>(*v)?;
            }
            Ok(ArrayRef {
                offset,
                len: data.len() as u64,
            })
        };

        let mut descs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let diag = if layer.diag.is_empty() {
                None
            } else {
                Some(push(&layer.diag)?)
            };
            let gram = match &layer.gram {
                Some(g) => Some(GramDesc {
                    layout: g.layout,
                    data: push(&g.data)?,
                }),
                None => None,
            };
            let mut experts = Vec::with_capacity(layer.experts.len());
            for e in &layer.experts {
                let diag = match &e.diag {
                    Some(d) => Some(push(d)?),
                    None => None,
                };
                experts.push(ExpertDesc {
                    expert: e.expert,
                    tokens: e.tokens,
                    status: e.status,
                    diag,
                });
            }
            descs.push(LayerDesc {
                isq_index: layer.isq_index,
                artifact_name: layer.artifact_name.clone(),
                name: layer.name.clone(),
                layer_num: layer.layer_num,
                supported: layer.supported,
                in_features: layer.in_features,
                tokens: layer.tokens,
                calls: layer.calls,
                diag,
                gram,
                experts,
            });
        }

        let header = Header {
            meta: self.meta.clone(),
            layers: descs,
        };
        let header_bytes = serde_json::to_vec(&header)
            .map_err(|e| candle_core::Error::Msg(format!("calibration header encode: {e}")))?;

        let mut buf: Vec<u8> = Vec::with_capacity(8 + 4 + 8 + header_bytes.len() + payload.len());
        buf.extend_from_slice(CALIB_MAGIC);
        buf.write_u32::<LittleEndian>(CALIB_FORMAT_VERSION)?;
        buf.write_u64::<LittleEndian>(header_bytes.len() as u64)?;
        buf.extend_from_slice(&header_bytes);
        buf.extend_from_slice(&payload);

        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, buf)?;
        Ok(())
    }

    /// Read an artifact written by [`Self::save`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let buf = fs::read(path)?;
        Self::from_bytes(&buf)
    }

    /// Parse an artifact from an in-memory buffer.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(buf);
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        if &magic != CALIB_MAGIC {
            candle_core::bail!(
                "Not a calibration artifact: bad magic {:?} (expected {:?})",
                String::from_utf8_lossy(&magic),
                String::from_utf8_lossy(CALIB_MAGIC)
            );
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        version_is_compatible(version)?;
        let header_len = cursor.read_u64::<LittleEndian>()? as usize;
        let header_start = cursor.position() as usize;
        let header_end = header_start
            .checked_add(header_len)
            .filter(|e| *e <= buf.len())
            .context_msg("calibration artifact truncated in header")?;
        let header: Header = serde_json::from_slice(&buf[header_start..header_end])
            .map_err(|e| candle_core::Error::Msg(format!("calibration header decode: {e}")))?;

        let payload = &buf[header_end..];
        if !payload.len().is_multiple_of(8) {
            candle_core::bail!("calibration payload is not a whole number of f64 values");
        }
        let read = |r: &ArrayRef| -> Result<Vec<f64>> {
            let start = (r.offset as usize)
                .checked_mul(8)
                .context_msg("calibration array offset overflow")?;
            let end = start
                .checked_add((r.len as usize) * 8)
                .filter(|e| *e <= payload.len())
                .context_msg("calibration artifact truncated in payload")?;
            let mut out = Vec::with_capacity(r.len as usize);
            let mut c = Cursor::new(&payload[start..end]);
            for _ in 0..r.len {
                out.push(c.read_f64::<LittleEndian>()?);
            }
            Ok(out)
        };

        let mut layers = Vec::with_capacity(header.layers.len());
        for d in header.layers {
            let diag = match &d.diag {
                Some(r) => read(r)?,
                None => Vec::new(),
            };
            let gram = match &d.gram {
                Some(g) => Some(GramBlocks {
                    layout: g.layout,
                    data: read(&g.data)?,
                }),
                None => None,
            };
            let mut experts = Vec::with_capacity(d.experts.len());
            for e in &d.experts {
                let diag = match &e.diag {
                    Some(r) => Some(read(r)?),
                    None => None,
                };
                experts.push(ExpertCalibData {
                    expert: e.expert,
                    tokens: e.tokens,
                    status: e.status,
                    diag,
                });
            }
            layers.push(LayerCalibStats {
                isq_index: d.isq_index,
                artifact_name: d.artifact_name,
                name: d.name,
                layer_num: d.layer_num,
                supported: d.supported,
                in_features: d.in_features,
                tokens: d.tokens,
                calls: d.calls,
                diag,
                gram,
                experts,
            });
        }
        Ok(Self::new(header.meta, layers))
    }

    /// Human-readable summary, used by the `stats_info` utility.
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let m = &self.meta;
        let mut s = String::new();
        let _ = writeln!(s, "model_id          {}", m.model_id);
        if let Some(sha) = &m.model_sha {
            let _ = writeln!(s, "model_sha         {sha}");
        }
        if let Some(arch) = &m.arch {
            let _ = writeln!(s, "arch              {arch}");
        }
        let _ = writeln!(s, "isq_organization  {}", m.isq_organization);
        if let Some(f) = &m.calibration_file {
            let _ = writeln!(s, "calibration_file  {f}");
        }
        let _ = writeln!(
            s,
            "samples           {} x {} tok = {} tokens",
            m.samples, m.seq_len, m.total_tokens
        );
        let _ = writeln!(s, "model_dtype       {}", m.model_dtype);
        let _ = writeln!(s, "storage_dtype     {}", m.storage_dtype);
        let _ = writeln!(
            s,
            "collector         v{} (producer {})",
            m.collector_version, m.producer
        );
        let _ = writeln!(s, "options           {:?}", m.options);
        let _ = writeln!(
            s,
            "layers            {} total, {} with statistics ({} declared by model)",
            self.layers.len(),
            self.supported_layer_count(),
            m.isq_layer_count
        );
        let with_gram = self.layers.iter().filter(|l| l.gram.is_some()).count();
        let with_experts = self.layers.iter().filter(|l| !l.experts.is_empty()).count();
        let zero_tok: usize = self
            .layers
            .iter()
            .map(|l| l.zero_token_experts().len())
            .sum();
        let _ = writeln!(s, "with gram         {with_gram}");
        let _ = writeln!(
            s,
            "with per-expert   {with_experts} ({zero_tok} zero-token experts marked)"
        );
        s
    }
}

trait ContextMsg<T> {
    fn context_msg(self, msg: &str) -> Result<T>;
}

impl<T> ContextMsg<T> for Option<T> {
    fn context_msg(self, msg: &str) -> Result<T> {
        self.ok_or_else(|| candle_core::Error::Msg(msg.to_string()))
    }
}

// ---------------------------------------------------------------------------
// On-disk header types. Kept private so the public API can evolve independently
// of the serialized shape.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Header {
    meta: CalibrationMeta,
    layers: Vec<LayerDesc>,
}

/// Reference into the payload, in **f64 elements** (not bytes).
#[derive(Clone, Copy, Serialize, Deserialize)]
struct ArrayRef {
    offset: u64,
    len: u64,
}

#[derive(Serialize, Deserialize)]
struct GramDesc {
    layout: GramLayout,
    data: ArrayRef,
}

#[derive(Serialize, Deserialize)]
struct ExpertDesc {
    expert: usize,
    tokens: u64,
    status: ExpertStatus,
    #[serde(default)]
    diag: Option<ArrayRef>,
}

#[derive(Serialize, Deserialize)]
struct LayerDesc {
    isq_index: usize,
    artifact_name: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    layer_num: Option<usize>,
    supported: bool,
    in_features: usize,
    tokens: u64,
    calls: u64,
    #[serde(default)]
    diag: Option<ArrayRef>,
    #[serde(default)]
    gram: Option<GramDesc>,
    #[serde(default)]
    experts: Vec<ExpertDesc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::collector::GramMode;

    fn meta() -> CalibrationMeta {
        CalibrationMeta {
            model_id: "test/model".to_string(),
            model_sha: Some("deadbeef".to_string()),
            arch: Some("deepseek4".to_string()),
            isq_organization: "default".to_string(),
            calibration_file: Some("calibration_data/calibration_datav3_small.txt".to_string()),
            samples: 4,
            seq_len: 1024,
            total_tokens: 4096,
            model_dtype: "BF16".to_string(),
            storage_dtype: "F64".to_string(),
            collector_version: CALIB_COLLECTOR_VERSION,
            producer: "test".to_string(),
            created_unix: 1_700_000_000,
            options: CalibOptions {
                gram: GramMode::Blockwise { block: 2 },
                per_expert: true,
                min_expert_tokens: 8,
            },
            isq_layer_count: 3,
        }
    }

    fn sample_artifact() -> CalibrationArtifact {
        let l0 = LayerCalibStats {
            isq_index: 0,
            artifact_name: "0".to_string(),
            name: Some("blk.0.attn_q.weight".to_string()),
            layer_num: Some(0),
            supported: true,
            in_features: 4,
            tokens: 100,
            calls: 2,
            diag: vec![1.5, 2.5, 3.5, 4.5],
            gram: Some(GramBlocks {
                layout: GramLayout::Blockwise { dim: 4, block: 2 },
                data: vec![1.0, 0.5, 0.5, 2.0, 3.0, 0.25, 0.25, 4.0],
            }),
            experts: Vec::new(),
        };
        let l1 = LayerCalibStats {
            isq_index: 1,
            artifact_name: "1".to_string(),
            name: None,
            layer_num: Some(0),
            supported: true,
            in_features: 3,
            tokens: 60,
            calls: 2,
            diag: vec![9.0, 8.0, 7.0],
            gram: None,
            experts: vec![
                ExpertCalibData {
                    expert: 0,
                    tokens: 40,
                    status: ExpertStatus::Ok,
                    diag: Some(vec![6.0, 5.0, 4.0]),
                },
                ExpertCalibData {
                    expert: 1,
                    tokens: 20,
                    status: ExpertStatus::Insufficient,
                    diag: Some(vec![3.0, 3.0, 3.0]),
                },
                ExpertCalibData {
                    expert: 2,
                    tokens: 0,
                    status: ExpertStatus::ZeroTokens,
                    diag: None,
                },
            ],
        };
        // Trailing MTP-tail layer that the model could not track.
        let l2 = LayerCalibStats::unsupported(2, "mtp.0".to_string(), None, Some(61));
        CalibrationArtifact::new(meta(), vec![l2, l0, l1])
    }

    #[test]
    fn round_trip_preserves_everything() {
        let art = sample_artifact();
        let dir = std::env::temp_dir().join(format!("arccalib-rt-{}", std::process::id()));
        let path = dir.join(format!("stats.{CALIB_EXTENSION}"));
        art.save(&path).unwrap();
        let back = CalibrationArtifact::load(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(art, back);
    }

    #[test]
    fn lookup_by_index_name_and_artifact_name() {
        let art = sample_artifact();
        assert_eq!(art.by_isq_index(1).unwrap().tokens, 60);
        assert_eq!(art.by_artifact_name("mtp.0").unwrap().isq_index, 2);
        assert_eq!(
            art.by_name("blk.0.attn_q.weight").unwrap().isq_index,
            0,
            "symbolic lookup must resolve to the right ISQ slot"
        );
        assert!(art.by_isq_index(99).is_none());
        assert_eq!(art.artifact_names(), vec!["0", "1", "mtp.0"]);
        assert_eq!(art.supported_layer_count(), 2);
    }

    #[test]
    fn zero_token_experts_are_reported_and_carry_no_diag() {
        let art = sample_artifact();
        let l1 = art.by_isq_index(1).unwrap();
        assert_eq!(l1.zero_token_experts(), vec![2]);
        assert!(l1.expert(2).unwrap().diag.is_none());
        assert_eq!(l1.expert(1).unwrap().status, ExpertStatus::Insufficient);
    }

    #[test]
    fn normalized_diag_divides_by_tokens_and_declines_unsupported() {
        let art = sample_artifact();
        let n = art.by_isq_index(0).unwrap().normalized_diag().unwrap();
        assert!((n[0] - 0.015).abs() < 1e-12);
        assert!(art.by_isq_index(2).unwrap().normalized_diag().is_none());
    }

    #[test]
    fn gram_blocks_are_addressable_after_round_trip() {
        let art = sample_artifact();
        let bytes = {
            let dir = std::env::temp_dir().join(format!("arccalib-gram-{}", std::process::id()));
            let p = dir.join("g.arccalib");
            art.save(&p).unwrap();
            let b = std::fs::read(&p).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            b
        };
        let back = CalibrationArtifact::from_bytes(&bytes).unwrap();
        let g = back.by_isq_index(0).unwrap().gram.as_ref().unwrap();
        assert_eq!(g.layout.num_blocks(), 2);
        let (b0, w0) = g.block(0).unwrap();
        assert_eq!(w0, 2);
        assert_eq!(b0, &[1.0, 0.5, 0.5, 2.0]);
        let (b1, _) = g.block(1).unwrap();
        assert_eq!(b1, &[3.0, 0.25, 0.25, 4.0]);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = vec![0u8; 32];
        bytes[..8].copy_from_slice(b"NOTACALB");
        assert!(CalibrationArtifact::from_bytes(&bytes).is_err());
    }

    #[test]
    fn future_major_version_is_rejected() {
        let art = sample_artifact();
        let dir = std::env::temp_dir().join(format!("arccalib-ver-{}", std::process::id()));
        let p = dir.join("v.arccalib");
        art.save(&p).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        // Bump the major byte.
        let future = ((CALIB_VERSION_MAJOR + 1) << 16) | (CALIB_VERSION_MINOR << 8);
        bytes[8..12].copy_from_slice(&future.to_le_bytes());
        let err = CalibrationArtifact::from_bytes(&bytes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Major version"), "unexpected error: {err}");
    }

    #[test]
    fn truncated_payload_errors_cleanly() {
        let art = sample_artifact();
        let dir = std::env::temp_dir().join(format!("arccalib-trunc-{}", std::process::id()));
        let p = dir.join("t.arccalib");
        art.save(&p).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        bytes.truncate(bytes.len() - 16);
        assert!(CalibrationArtifact::from_bytes(&bytes).is_err());
    }

    #[test]
    fn summary_mentions_the_key_provenance_fields() {
        let s = sample_artifact().summary();
        assert!(s.contains("test/model"));
        assert!(s.contains("deadbeef"));
        assert!(s.contains("zero-token experts marked"));
    }
}
