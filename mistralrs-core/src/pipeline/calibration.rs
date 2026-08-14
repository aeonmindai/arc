//! Calibration-statistics collection over a model's ISQ layer set.
//!
//! A calibration run is a **forward-only** sweep: the model is loaded
//! unquantized, every ISQ-eligible linear is armed with a
//! [`CalibAccumulator`], a corpus is pushed through in fixed-size chunks, and
//! the accumulated `diag(XᵀX)` (plus optional gram blocks and per-expert
//! statistics) is written to a `.arccalib` artifact.
//!
//! # Why this lives next to ISQ
//!
//! The artifact is keyed by the layer's index in `IsqModel::get_layers()` and
//! named with [`isq_artifact_tensor_name`] — the very function the UQFF
//! serializer uses. Consumers (the QTIP trellis search, TD-MoE whitening) hold
//! an ISQ layer list and can therefore look statistics up with the ordinal they
//! already have, with no name-matching heuristics in between.
//!
//! # Requesting a run
//!
//! `mistralrs-cli`'s `calibrate` command registers a [`CalibrationRequest`]
//! before building the model; the normal-model loader picks it up inside the
//! existing calibration sweep. When no request is registered, nothing here
//! runs and the loader behaves exactly as before.

use std::{
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::Result;
use mistralrs_quant::{
    CalibOptions, CalibrationArtifact, CalibrationMeta, LayerCalibStats, CALIB_COLLECTOR_VERSION,
};

use crate::pipeline::isq::{isq_artifact_tensor_name, IsqModel};

/// A pending request to collect calibration statistics during the next model
/// load. Registered by the CLI, consumed once by the loader.
#[derive(Debug, Clone)]
pub struct CalibrationRequest {
    /// Where to write the `.arccalib` artifact.
    pub out: PathBuf,
    /// What to accumulate.
    pub opts: CalibOptions,
    /// Stop the sweep after this many chunks. `None` sweeps the whole corpus.
    pub max_chunks: Option<usize>,
    /// Recorded in the artifact metadata.
    pub model_id: String,
    /// Recorded in the artifact metadata.
    pub arch: Option<String>,
}

static REQUEST: OnceLock<Mutex<Option<CalibrationRequest>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<CalibrationRequest>> {
    REQUEST.get_or_init(|| Mutex::new(None))
}

/// Register a calibration request for the next model load.
pub fn set_calibration_request(request: CalibrationRequest) {
    *slot().lock().expect("calibration request poisoned") = Some(request);
}

/// Whether a calibration request is pending.
pub fn calibration_requested() -> bool {
    slot()
        .lock()
        .expect("calibration request poisoned")
        .is_some()
}

/// Read the pending request without consuming it.
pub fn peek_calibration_request() -> Option<CalibrationRequest> {
    slot().lock().expect("calibration request poisoned").clone()
}

/// Take the pending request, clearing it.
pub fn take_calibration_request() -> Option<CalibrationRequest> {
    slot().lock().expect("calibration request poisoned").take()
}

/// Outcome of arming a model for calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmReport {
    /// Total ISQ layers in the model.
    pub total: usize,
    /// Layers whose quant method accepted an accumulator.
    pub armed: usize,
}

/// Arm every ISQ layer of `model` with a calibration accumulator.
///
/// Layers whose quant method cannot track activations (already-quantized
/// formats such as FP8 or GPTQ) are skipped rather than failing the run — they
/// are still listed in the artifact, marked `supported: false`, so the layer
/// inventory keeps matching the ISQ layer set exactly.
pub fn begin_model_calibration(model: &mut dyn IsqModel, opts: &CalibOptions) -> ArmReport {
    let layers = model.get_layers().0;
    let total = layers.len();
    let mut armed = 0;
    for (layer, _) in layers {
        let Some(inner) = std::sync::Arc::get_mut(layer) else {
            continue;
        };
        if inner.begin_calibration(opts).is_ok() {
            armed += 1;
        }
    }
    ArmReport { total, armed }
}

/// Provenance fields the loader knows but this module does not.
#[derive(Debug, Clone)]
pub struct CalibrationRunInfo {
    pub model_id: String,
    pub model_sha: Option<String>,
    pub arch: Option<String>,
    pub isq_organization: String,
    pub calibration_file: Option<String>,
    pub samples: usize,
    pub seq_len: usize,
    pub total_tokens: u64,
    pub model_dtype: String,
    pub options: CalibOptions,
}

/// Harvest the accumulated statistics and build the artifact.
///
/// Walks `get_layers()` in the exact order [`IsqModel::quantize`] does, so
/// entry `i` of the artifact describes ISQ layer `i`, and `artifact_name`
/// matches the UQFF tensor name the quantizer would emit for that slot.
pub fn extract_calibration_artifact(
    model: &mut dyn IsqModel,
    info: CalibrationRunInfo,
) -> Result<CalibrationArtifact> {
    // Both borrows below need `&mut self`, so resolve everything that does not
    // depend on the layer list first.
    let mtp_tail = model.mtp_isq_tail_len();
    let names = model.imatrix_names().unwrap_or_default();

    let layers = model.get_layers().0;
    let total = layers.len();
    let main_len = total.saturating_sub(mtp_tail);

    let mut out = Vec::with_capacity(total);
    for (i, (layer, layer_num)) in layers.into_iter().enumerate() {
        let artifact_name = isq_artifact_tensor_name(i, main_len);
        let name = names.get(i).cloned().flatten();
        match layer.end_calibration() {
            Ok(data) => out.push(LayerCalibStats::from_data(
                i,
                artifact_name,
                name,
                layer_num,
                data,
            )),
            Err(_) => out.push(LayerCalibStats::unsupported(
                i,
                artifact_name,
                name,
                layer_num,
            )),
        }
    }

    let meta = CalibrationMeta {
        model_id: info.model_id,
        model_sha: info.model_sha,
        arch: info.arch,
        isq_organization: info.isq_organization,
        calibration_file: info.calibration_file,
        samples: info.samples,
        seq_len: info.seq_len,
        total_tokens: info.total_tokens,
        model_dtype: info.model_dtype,
        storage_dtype: "F64".to_string(),
        collector_version: CALIB_COLLECTOR_VERSION,
        producer: format!("mistralrs-core {}", env!("CARGO_PKG_VERSION")),
        created_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        options: info.options,
        isq_layer_count: total,
    };
    Ok(CalibrationArtifact::new(meta, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;
    use mistralrs_quant::{
        GramMode, QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear,
    };
    use std::sync::Arc;

    #[derive(Debug)]
    struct NoopMapper;

    impl crate::DeviceMapper for NoopMapper {
        fn map(&self, input: Tensor, _layer: usize) -> candle_core::Result<Tensor> {
            Ok(input)
        }
        fn set_nm_device(&self, vb: ShardedVarBuilder, _loading_isq: bool) -> ShardedVarBuilder {
            vb
        }
        fn set_device(
            &self,
            _layer: usize,
            vb: ShardedVarBuilder,
            _loading_isq: bool,
        ) -> ShardedVarBuilder {
            vb
        }
        fn device_for(&self, _layer: usize, _loading_isq: bool) -> Option<&Device> {
            None
        }
        fn get_unique_devices(&self) -> Vec<Device> {
            vec![Device::Cpu]
        }
        fn cast_nm_device(&self, x: &Tensor, _loading_isq: bool) -> candle_core::Result<Tensor> {
            Ok(x.clone())
        }
        fn get_min_dtype(&self, _: &dyn crate::TryIntoDType) -> candle_core::Result<DType> {
            Ok(DType::F32)
        }
        fn num_device_mapping_layers(&self) -> usize {
            1
        }
        fn get_comm_for(
            &self,
            _layer_idx: usize,
        ) -> candle_core::Result<Arc<mistralrs_quant::Comm>> {
            let id = mistralrs_quant::Id::new();
            Ok(Arc::new(mistralrs_quant::Comm::from_device(
                id,
                &Device::Cpu,
                0,
                1,
            )?))
        }
    }

    /// Model with `n_main` plain linears plus `n_mtp` trailing "MTP" linears,
    /// mirroring the layout `quantize()` assumes.
    struct FakeModel {
        layers: Vec<Arc<dyn QuantMethod>>,
        layer_nums: Vec<Option<usize>>,
        n_mtp: usize,
        names: Option<Vec<Option<String>>>,
        mapper: NoopMapper,
    }

    impl FakeModel {
        fn new(n_main: usize, n_mtp: usize, in_features: usize) -> Self {
            let layers = (0..n_main + n_mtp)
                .map(|_| {
                    let w = Tensor::ones((6, in_features), DType::F32, &Device::Cpu).unwrap();
                    let l = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                        Linear::new(w, None),
                    ))
                    .unwrap();
                    Arc::new(l) as Arc<dyn QuantMethod>
                })
                .collect();
            Self {
                layers,
                layer_nums: (0..n_main + n_mtp).map(Some).collect(),
                n_mtp,
                names: None,
                mapper: NoopMapper,
            }
        }
    }

    impl IsqModel for FakeModel {
        fn get_layers(
            &mut self,
        ) -> (
            Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
            &dyn crate::DeviceMapper,
        ) {
            let nums = self.layer_nums.clone();
            (self.layers.iter_mut().zip(nums).collect(), &self.mapper)
        }
        fn mtp_isq_tail_len(&mut self) -> usize {
            self.n_mtp
        }
        fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
            match &self.names {
                Some(n) => Ok(n.clone()),
                None => candle_core::bail!("no imatrix names"),
            }
        }
        fn residual_tensors(&self) -> Vec<(String, Tensor)> {
            Vec::new()
        }
    }

    fn run_info(opts: CalibOptions) -> CalibrationRunInfo {
        CalibrationRunInfo {
            model_id: "test/model".to_string(),
            model_sha: None,
            arch: Some("llama".to_string()),
            isq_organization: "default".to_string(),
            calibration_file: None,
            samples: 1,
            seq_len: 4,
            total_tokens: 4,
            model_dtype: "F32".to_string(),
            options: opts,
        }
    }

    /// THE INTEGRATION CONTRACT: the artifact's layer inventory must be exactly
    /// the ISQ layer set, in order, under the quantizer's own naming.
    #[test]
    fn artifact_layer_names_match_the_isq_layer_set_exactly() {
        let (n_main, n_mtp, d) = (5usize, 2usize, 3usize);
        let mut model = FakeModel::new(n_main, n_mtp, d);

        // Independently derive the names the quantizer would emit.
        let total = n_main + n_mtp;
        let expected: Vec<String> = (0..total)
            .map(|i| isq_artifact_tensor_name(i, n_main))
            .collect();
        assert_eq!(expected[0], "0");
        assert_eq!(expected[n_main], "mtp.0");

        let opts = CalibOptions::default();
        let report = begin_model_calibration(&mut model, &opts);
        assert_eq!(report.total, total);
        assert_eq!(report.armed, total);

        let x = Tensor::ones((4, d), DType::F32, &Device::Cpu).unwrap();
        for layer in &model.layers {
            layer.forward(&x).unwrap();
        }

        let art = extract_calibration_artifact(&mut model, run_info(opts)).unwrap();
        assert_eq!(art.artifact_names(), expected);
        assert_eq!(art.meta.isq_layer_count, total);
        assert_eq!(art.supported_layer_count(), total);
        for (i, layer) in art.layers.iter().enumerate() {
            assert_eq!(layer.isq_index, i);
            assert_eq!(layer.layer_num, Some(i));
            assert_eq!(layer.in_features, d);
            assert_eq!(layer.tokens, 4);
            // X is all-ones, so diag(XᵀX) == rows for every column.
            assert!(layer.diag.iter().all(|v| (*v - 4.0).abs() < 1e-9));
        }
        // And the index lookup a consumer would use resolves the same slot.
        assert_eq!(art.by_isq_index(n_main).unwrap().artifact_name, "mtp.0");
    }

    #[test]
    fn symbolic_names_are_carried_when_the_model_supplies_them() {
        let mut model = FakeModel::new(2, 0, 3);
        model.names = Some(vec![Some("blk.0.attn_q.weight".to_string()), None]);
        let opts = CalibOptions::default();
        begin_model_calibration(&mut model, &opts);
        let x = Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap();
        for layer in &model.layers {
            layer.forward(&x).unwrap();
        }
        let art = extract_calibration_artifact(&mut model, run_info(opts)).unwrap();
        assert_eq!(art.by_name("blk.0.attn_q.weight").unwrap().isq_index, 0);
        assert!(art.by_isq_index(1).unwrap().name.is_none());
    }

    #[test]
    fn layers_never_armed_are_listed_as_unsupported_not_dropped() {
        let mut model = FakeModel::new(3, 0, 4);
        // Arm nothing; harvesting must still produce a full inventory.
        let opts = CalibOptions::default();
        let art = extract_calibration_artifact(&mut model, run_info(opts)).unwrap();
        assert_eq!(art.layers.len(), 3, "inventory must cover every ISQ layer");
        assert_eq!(art.supported_layer_count(), 0);
        assert!(art.layers.iter().all(|l| l.diag.is_empty()));
        assert!(art.by_isq_index(1).unwrap().normalized_diag().is_none());
    }

    #[test]
    fn gram_options_propagate_through_the_arming_path() {
        let mut model = FakeModel::new(1, 0, 4);
        let opts = CalibOptions {
            gram: GramMode::Full { max_dim: 8 },
            ..Default::default()
        };
        begin_model_calibration(&mut model, &opts);
        let x = Tensor::ones((5, 4), DType::F32, &Device::Cpu).unwrap();
        model.layers[0].forward(&x).unwrap();
        let art = extract_calibration_artifact(&mut model, run_info(opts)).unwrap();
        let gram = art.by_isq_index(0).unwrap().gram.as_ref().unwrap();
        assert_eq!(gram.data.len(), 16);
        assert!(gram.data.iter().all(|v| (*v - 5.0).abs() < 1e-9));
    }

    #[test]
    fn request_registry_round_trips_and_clears() {
        assert!(!calibration_requested());
        set_calibration_request(CalibrationRequest {
            out: PathBuf::from("/tmp/x.arccalib"),
            opts: CalibOptions::default(),
            max_chunks: Some(3),
            model_id: "m".to_string(),
            arch: None,
        });
        assert!(calibration_requested());
        assert_eq!(peek_calibration_request().unwrap().max_chunks, Some(3));
        assert!(take_calibration_request().is_some());
        assert!(!calibration_requested());
    }
}
