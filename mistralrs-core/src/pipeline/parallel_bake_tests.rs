//! What a parallel UQFF bake must not change (wave22).
//!
//! A bake spread over N CUDA devices must produce **exactly** the artifact a
//! single-device bake would have produced: the same tensor names, in the same
//! order, with the same bytes. These tests pin that from both ends.
//!
//! * The CPU tests pin the *inventory contract* — that a UQFF artifact's tensor
//!   names come only from `get_layers()` positions and that each name carries
//!   that layer's own content. Every test here runs on any machine, and each
//!   assertion is paired with a mutation test proving it actually fails when a
//!   layer is dropped, duplicated, misnamed, or given another layer's bytes.
//!   Without those, "the artifacts matched" would be a claim about an assertion
//!   nobody had watched fail.
//! * [`byte_identical_across_two_cuda_devices`] runs the real thing on real
//!   silicon: it bakes the same model twice, once on one device and once spread
//!   over two, and compares the shards byte for byte.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use indicatif::MultiProgress;
use mistralrs_quant::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};

use super::isq::{IsqModel, IsqOrganization, UqffFullSer};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct NoopMapper;

impl crate::DeviceMapper for NoopMapper {
    fn map(&self, input: Tensor, _layer: usize) -> candle_core::Result<Tensor> {
        Ok(input)
    }
    fn set_nm_device(&self, vb: ShardedVarBuilder, _loading_isq: bool) -> ShardedVarBuilder {
        vb
    }
    fn set_device(&self, _layer: usize, vb: ShardedVarBuilder, _isq: bool) -> ShardedVarBuilder {
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
    fn get_comm_for(&self, _layer_idx: usize) -> candle_core::Result<Arc<mistralrs_quant::Comm>> {
        let id = mistralrs_quant::Id::new();
        Ok(Arc::new(mistralrs_quant::Comm::from_device(
            id,
            &Device::Cpu,
            0,
            1,
        )?))
    }
}

/// `n_main` plain linears plus `n_mtp` trailing "MTP" linears, matching the
/// layout `IsqModel::quantize` assumes.
///
/// **Every layer's weights are distinct**, which is what gives the mutation
/// tests teeth: with identical weights (the obvious fixture) a bake that
/// swapped two layers, or wrote layer 3 twice, would produce a byte-identical
/// artifact and every assertion here would still pass.
struct FakeModel {
    layers: Vec<Arc<dyn QuantMethod>>,
    layer_nums: Vec<Option<usize>>,
    n_mtp: usize,
    mapper: NoopMapper,
}

impl FakeModel {
    fn new(n_main: usize, n_mtp: usize, out_features: usize, in_features: usize) -> Self {
        let device = Device::Cpu;
        let layers = (0..n_main + n_mtp)
            .map(|i| {
                // Layer i is filled with a value unique to i, so its serialized
                // bytes identify it.
                let values: Vec<f32> = (0..out_features * in_features)
                    .map(|j| (i * 1_000 + j % 97) as f32 * 0.001 + i as f32)
                    .collect();
                let w = Tensor::from_vec(values, (out_features, in_features), &device).unwrap();
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
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

/// Serializes the tests against the process-global bake settings and restores
/// them afterwards, **including on panic**.
///
/// `bake_isq_to_host` and the bake device override are process-global (they
/// have to be: the ISQ pool runs a layer's quantize off the construction
/// thread). Without this, `cargo test`'s parallel harness lets one test's
/// device list decide another test's bake — which is exactly how this file
/// first failed.
struct BakeGlobals {
    _lock: std::sync::MutexGuard<'static, ()>,
    previous_host: bool,
}

impl BakeGlobals {
    /// Acquire with an explicit device setting, so no test inherits either the
    /// environment or a sibling's leftovers.
    fn acquire(devices: Option<Vec<usize>>, to_host: bool) -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous_host = mistralrs_quant::bake_isq_to_host();
        mistralrs_quant::set_bake_device_override(devices);
        mistralrs_quant::set_bake_isq_to_host(to_host);
        Self {
            _lock: lock,
            previous_host,
        }
    }

    /// Change the device list while still holding the lock (the multi-device
    /// comparisons bake twice under one guard).
    fn set_devices(&self, devices: Option<Vec<usize>>) {
        mistralrs_quant::set_bake_device_override(devices);
    }
}

impl Drop for BakeGlobals {
    fn drop(&mut self) {
        mistralrs_quant::clear_bake_device_override();
        mistralrs_quant::set_bake_isq_to_host(self.previous_host);
    }
}

/// Fresh scratch directory; stale shards from an earlier run would be read back
/// as if this run had produced them.
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "arc-parallel-bake-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    dir
}

/// Run the real `IsqModel::quantize` UQFF write path into `dir`.
fn bake(
    model: &mut FakeModel,
    dir: &Path,
    dtype: Option<mistralrs_quant::IsqType>,
    device: Device,
) {
    let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    model
        .quantize(
            dtype,
            device,
            None,
            /*silent=*/ true,
            None,
            IsqOrganization::Default,
            /*apply_quantization=*/ true,
            Some(&dir.join("model.uqff")),
            UqffFullSer {
                tokenizer: &tokenizer,
                template_filename: &None,
                modules: None,
                module_paths: None,
                generation_config: None,
                config: "{}".to_string(),
                processor_filename: &None,
                preprocessor_filename: &None,
            },
            Arc::new(MultiProgress::new()),
        )
        .expect("the UQFF bake must succeed");
}

/// Every `(name, bytes)` in the artifact, in the order the shards store them.
fn inventory(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("artifact dir must be readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "uqff"))
        .collect();
    shards.sort();
    assert!(
        !shards.is_empty(),
        "the bake wrote no UQFF shard to {dir:?}"
    );
    let mut out = Vec::new();
    for shard in shards {
        let bytes = std::fs::read(&shard).expect("shard must be readable");
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes)
            .expect("shard must be valid safetensors");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("shard must deserialize");
        // `metadata()` iteration order is not the stored order, so recover the
        // stored order from each tensor's data offset.
        let mut named: Vec<(String, usize)> = meta
            .tensors()
            .into_iter()
            .map(|(name, info)| (name, info.data_offsets.0))
            .collect();
        named.sort_by_key(|(_, offset)| *offset);
        for (name, _) in named {
            let view = st.tensor(&name).expect("named tensor must exist");
            out.push((name, view.data().to_vec()));
        }
    }
    out
}

/// The single comparison every test in this file goes through, so the mutation
/// tests below prove the *same* check that the identity tests rely on.
///
/// Returns `Err(reason)` rather than panicking so a test can assert it fails.
fn compare_inventories(
    actual: &[(String, Vec<u8>)],
    expected: &[(String, Vec<u8>)],
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "layer count differs: {} vs {}",
            actual.len(),
            expected.len()
        ));
    }
    for (i, ((a_name, a_bytes), (e_name, e_bytes))) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        if a_name != e_name {
            return Err(format!("position {i}: name `{a_name}` != `{e_name}`"));
        }
        if a_bytes != e_bytes {
            return Err(format!(
                "tensor `{a_name}`: {} bytes differ from the reference",
                a_bytes.len()
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The inventory contract (runs everywhere)
// ---------------------------------------------------------------------------

#[test]
fn artifact_names_are_layer_positions_with_an_mtp_tail() {
    let _globals = BakeGlobals::acquire(None, false);
    let dir = scratch_dir("names");
    let mut model = FakeModel::new(5, 2, 6, 8);
    bake(&mut model, &dir, None, Device::Cpu);

    let names: Vec<String> = inventory(&dir).into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        vec!["0", "1", "2", "3", "4", "mtp.0", "mtp.1"],
        "UQFF tensor names are positional `get_layers()` ordinals with an \
         `mtp.<j>` tail; a parallel bake must not perturb either"
    );
}

#[test]
fn every_layer_carries_its_own_content() {
    let _globals = BakeGlobals::acquire(None, false);
    let dir = scratch_dir("content");
    let mut model = FakeModel::new(4, 0, 6, 8);
    bake(&mut model, &dir, None, Device::Cpu);

    let got = inventory(&dir);
    // Distinct weights per layer => distinct serialized bytes per layer. If any
    // two entries collide, the mutation tests below prove nothing.
    for i in 0..got.len() {
        for j in (i + 1)..got.len() {
            assert_ne!(
                got[i].1, got[j].1,
                "fixture is degenerate: layers {i} and {j} serialize identically, \
                 so a swap or duplicate would be undetectable"
            );
        }
    }
}

#[test]
fn the_same_model_bakes_to_the_same_bytes_twice() {
    let _globals = BakeGlobals::acquire(None, false);
    // The baseline the multi-device comparison rests on: the write path is
    // deterministic, so a later mismatch is attributable to the device spread.
    let dir_a = scratch_dir("determinism-a");
    let dir_b = scratch_dir("determinism-b");
    bake(&mut FakeModel::new(5, 2, 6, 8), &dir_a, None, Device::Cpu);
    bake(&mut FakeModel::new(5, 2, 6, 8), &dir_b, None, Device::Cpu);

    compare_inventories(&inventory(&dir_a), &inventory(&dir_b))
        .expect("two bakes of the same model must be byte-identical");
}

#[test]
fn quantized_bakes_are_deterministic_too() {
    let _globals = BakeGlobals::acquire(None, false);
    // Same, through an actual quantizer rather than the passthrough, so the
    // comparison covers packed bytes and not just raw weights.
    let dir_a = scratch_dir("q-determinism-a");
    let dir_b = scratch_dir("q-determinism-b");
    let ty = Some(mistralrs_quant::IsqType::Q8_0);
    bake(&mut FakeModel::new(4, 1, 32, 64), &dir_a, ty, Device::Cpu);
    bake(&mut FakeModel::new(4, 1, 32, 64), &dir_b, ty, Device::Cpu);

    compare_inventories(&inventory(&dir_a), &inventory(&dir_b))
        .expect("two quantized bakes of the same model must be byte-identical");
}

// ---------------------------------------------------------------------------
// Mutation tests: the comparison above must FAIL on each way a parallel bake
// could go wrong. Without these, a vacuous check would look like a green suite.
// ---------------------------------------------------------------------------

/// A reference inventory to mutate.
fn reference_inventory() -> Vec<(String, Vec<u8>)> {
    let _globals = BakeGlobals::acquire(None, false);
    let dir = scratch_dir("reference");
    bake(&mut FakeModel::new(5, 2, 6, 8), &dir, None, Device::Cpu);
    let got = inventory(&dir);
    assert_eq!(got.len(), 7);
    got
}

#[test]
fn comparison_rejects_a_dropped_layer() {
    let reference = reference_inventory();
    let mut mutated = reference.clone();
    mutated.remove(3);
    let err = compare_inventories(&mutated, &reference)
        .expect_err("dropping a layer must not compare equal");
    assert!(err.contains("layer count"), "{err}");
}

#[test]
fn comparison_rejects_a_duplicated_layer() {
    let reference = reference_inventory();
    let mut mutated = reference.clone();
    // Layer 2 written twice, layer 3 lost — the shape of a work-stealing bug
    // where two workers claim the same index.
    mutated[3] = mutated[2].clone();
    let err = compare_inventories(&mutated, &reference)
        .expect_err("duplicating a layer must not compare equal");
    assert!(
        err.contains("name") || err.contains("bytes differ"),
        "{err}"
    );
}

#[test]
fn comparison_rejects_a_misnamed_layer() {
    let reference = reference_inventory();
    let mut mutated = reference.clone();
    // The MTP tail renamed to a positional ordinal: the exact regression the
    // `mtp.<j>` convention exists to prevent.
    mutated[5].0 = "5".to_string();
    let err = compare_inventories(&mutated, &reference)
        .expect_err("renaming a layer must not compare equal");
    assert!(err.contains("name"), "{err}");
}

#[test]
fn comparison_rejects_swapped_content() {
    let reference = reference_inventory();
    let mut mutated = reference.clone();
    // Names right, contents crossed: what a bake that raced two workers onto
    // one output slot would produce.
    let a = mutated[1].1.clone();
    mutated[1].1 = mutated[2].1.clone();
    mutated[2].1 = a;
    let err = compare_inventories(&mutated, &reference)
        .expect_err("swapping two layers' contents must not compare equal");
    assert!(err.contains("bytes differ"), "{err}");
}

#[test]
fn comparison_rejects_a_truncated_tensor() {
    let reference = reference_inventory();
    let mut mutated = reference.clone();
    mutated[0].1.truncate(4);
    let err = compare_inventories(&mutated, &reference)
        .expect_err("a short tensor must not compare equal");
    assert!(err.contains("bytes differ"), "{err}");
}

// ---------------------------------------------------------------------------
// Device-list plumbing (runs everywhere)
// ---------------------------------------------------------------------------

#[test]
fn a_device_list_is_ignored_outside_a_bake() {
    // `--bake-devices` must never move a layer for a run that will actually
    // execute a forward pass; only a host-materializing bake may be spread.
    let _globals = BakeGlobals::acquire(Some(vec![0, 1]), /*to_host=*/ false);
    let dir = scratch_dir("ignored");
    // Would `bail!` on a CPU device if the spread were attempted.
    bake(&mut FakeModel::new(3, 0, 6, 8), &dir, None, Device::Cpu);
    assert_eq!(inventory(&dir).len(), 3);
}

#[test]
fn a_device_list_is_rejected_on_a_non_cuda_bake() {
    // Silently ignoring it would let a user believe a 4-hour bake was spread
    // across four devices when it ran on one.
    let _globals = BakeGlobals::acquire(Some(vec![0, 1]), /*to_host=*/ true);
    let dir = scratch_dir("non-cuda");
    let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let mut model = FakeModel::new(3, 0, 6, 8);
    let err = model
        .quantize(
            None,
            Device::Cpu,
            None,
            true,
            None,
            IsqOrganization::Default,
            true,
            Some(&dir.join("model.uqff")),
            UqffFullSer {
                tokenizer: &tokenizer,
                template_filename: &None,
                modules: None,
                module_paths: None,
                generation_config: None,
                config: "{}".to_string(),
                processor_filename: &None,
                preprocessor_filename: &None,
            },
            Arc::new(MultiProgress::new()),
        )
        .expect_err("a device list on a CPU bake must be an error, not a silent no-op");
    let msg = err.to_string();
    assert!(msg.contains("ARC_BAKE_DEVICES"), "{msg}");
}

// ---------------------------------------------------------------------------
// The real thing, on real silicon
// ---------------------------------------------------------------------------

/// The invariant every device-spreading design needs, isolated from the
/// spreading itself: quantizing a layer on device 1 gives the same bytes as
/// quantizing it on device 0.
///
/// This is not implied by the two-device test — it is what *makes* the
/// two-device test's result meaningful. A 2-device bake puts layer 1 on device
/// 1 where a 1-device bake put it on device 0, so if the quantizer's output
/// were device-dependent at all, no design could produce a byte-identical
/// artifact. Testing it on its own says which half broke when it breaks.
#[cfg(feature = "cuda")]
#[test]
fn a_layer_quantizes_identically_on_either_device() {
    let (Ok(dev0), Ok(_)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
        eprintln!("skipping: fewer than two CUDA devices");
        return;
    };
    let ty = Some(mistralrs_quant::IsqType::QtipBitshift2);
    let globals = BakeGlobals::acquire(Some(vec![0]), /*to_host=*/ true);

    let on_zero = scratch_dir("only-cuda0");
    bake(
        &mut FakeModel::new(4, 0, 512, 1024),
        &on_zero,
        ty,
        dev0.clone(),
    );
    let zero_counts = mistralrs_quant::bake_device_layer_counts();

    globals.set_devices(Some(vec![1]));
    let on_one = scratch_dir("only-cuda1");
    bake(
        &mut FakeModel::new(4, 0, 512, 1024),
        &on_one,
        ty,
        dev0.clone(),
    );
    let one_counts = mistralrs_quant::bake_device_layer_counts();

    // Each leg really did run entirely on the device it names.
    assert_eq!(zero_counts.keys().copied().collect::<Vec<_>>(), vec![0]);
    assert_eq!(one_counts.keys().copied().collect::<Vec<_>>(), vec![1]);

    compare_inventories(&inventory(&on_one), &inventory(&on_zero)).expect(
        "the same layer quantized on cuda:1 must produce the same bytes as on \
         cuda:0 — every parallel-bake design depends on this",
    );
}

/// Bake the same model on one CUDA device and on two, and require the artifacts
/// to be byte-identical.
///
/// This is the claim the whole feature rests on. It is not a weaker restatement
/// of the CPU tests: it also pins that the quantizer's *output* does not depend
/// on which device ran it, which any device-spreading design needs — a 2-device
/// bake puts layer 1 on device 1 where a 1-device bake put it on device 0.
///
/// Skips (rather than fails) on a box with fewer than two CUDA devices, so the
/// single-GPU CI lanes stay green.
#[cfg(feature = "cuda")]
#[test]
fn byte_identical_across_two_cuda_devices() {
    let (Ok(dev0), Ok(dev1)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
        eprintln!("skipping: fewer than two CUDA devices");
        return;
    };
    let _ = dev1;

    // The real bake rung, and the real bake mode: quantized layers land on the
    // host, which is what makes relocating a layer's quantize legal.
    let ty = Some(mistralrs_quant::IsqType::QtipBitshift2);
    let globals = BakeGlobals::acquire(None, /*to_host=*/ true);
    let (layers, mtp, out_features, in_features) = (10, 2, 512, 1024);

    let fallbacks_before = mistralrs_quant::gpu_quantize_cpu_fallback_count();

    let one = scratch_dir("cuda-1gpu");
    let t_one = std::time::Instant::now();
    bake(
        &mut FakeModel::new(layers, mtp, out_features, in_features),
        &one,
        ty,
        dev0.clone(),
    );
    let one_elapsed = t_one.elapsed();
    let one_counts = mistralrs_quant::bake_device_layer_counts();

    let two = scratch_dir("cuda-2gpu");
    globals.set_devices(Some(vec![0, 1]));
    let t_two = std::time::Instant::now();
    bake(
        &mut FakeModel::new(layers, mtp, out_features, in_features),
        &two,
        ty,
        dev0.clone(),
    );
    let two_elapsed = t_two.elapsed();
    let two_counts = mistralrs_quant::bake_device_layer_counts();

    eprintln!(
        "1 device: {:.2}s {one_counts:?}   2 devices: {:.2}s {two_counts:?}",
        one_elapsed.as_secs_f32(),
        two_elapsed.as_secs_f32()
    );

    // Byte-identity alone would pass VACUOUSLY if the spread never engaged: a
    // bake that quietly ran on one device produces exactly the same artifact.
    // So assert the parallelism actually happened before believing the match.
    assert!(
        one_counts.is_empty(),
        "the single-device leg must not pin any worker to a device, got {one_counts:?}"
    );
    assert_eq!(
        two_counts.keys().copied().collect::<Vec<_>>(),
        vec![0, 1],
        "both CUDA devices must have quantized at least one layer"
    );
    assert_eq!(
        two_counts.values().sum::<usize>(),
        layers + mtp,
        "every layer must be quantized exactly once across the devices"
    );

    // And that the quantize really ran in GPU kernels: a silent reroute to the
    // CPU Viterbi would also be byte-identical, ~20x slower, and would make the
    // whole comparison meaningless.
    assert_eq!(
        mistralrs_quant::gpu_quantize_cpu_fallback_count(),
        fallbacks_before,
        "a QTIP bake on CUDA must not fall back to the CPU pipeline"
    );

    let single = inventory(&one);
    assert_eq!(
        single.len(),
        layers + mtp,
        "every layer must reach the artifact"
    );
    compare_inventories(&inventory(&two), &single).expect(
        "a bake spread over two CUDA devices must produce the same artifact as a \
         single-device bake, byte for byte",
    );
}

/// The same comparison at a size where the wall-clock difference is meaningful.
/// Ignored by default — it exists to be run explicitly on a multi-GPU box:
/// `cargo test -p mistralrs-core --features cuda parallel_bake_scaling -- --ignored --nocapture`
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a multi-GPU box; run explicitly for a scaling datapoint"]
fn parallel_bake_scaling() {
    let (Ok(dev0), Ok(_)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
        eprintln!("skipping: fewer than two CUDA devices");
        return;
    };
    let ty = Some(mistralrs_quant::IsqType::QtipBitshift2);
    let globals = BakeGlobals::acquire(None, /*to_host=*/ true);
    let layers = 16;
    let (out_features, in_features) = (2048, 2048);

    let one = scratch_dir("scale-1gpu");
    let t_one = std::time::Instant::now();
    bake(
        &mut FakeModel::new(layers, 0, out_features, in_features),
        &one,
        ty,
        dev0.clone(),
    );
    let one_elapsed = t_one.elapsed();

    let two = scratch_dir("scale-2gpu");
    globals.set_devices(Some(vec![0, 1]));
    let t_two = std::time::Instant::now();
    bake(
        &mut FakeModel::new(layers, 0, out_features, in_features),
        &two,
        ty,
        dev0.clone(),
    );
    let two_elapsed = t_two.elapsed();

    println!(
        "parallel bake scaling: {layers} layers of {out_features}x{in_features}\n  \
         1 device : {:.2}s\n  2 devices: {:.2}s\n  speedup  : {:.2}x",
        one_elapsed.as_secs_f32(),
        two_elapsed.as_secs_f32(),
        one_elapsed.as_secs_f32() / two_elapsed.as_secs_f32().max(f32::EPSILON)
    );

    compare_inventories(&inventory(&two), &inventory(&one))
        .expect("the scaling run must still be byte-identical");
}
