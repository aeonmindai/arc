//! wave18 — device-memory behaviour of a UQFF bake.
//!
//! A full V4-Flash bake died at layer 28 of 43 with `CUDA_ERROR_OUT_OF_MEMORY`
//! on a 140 GB H200, two hours and ~$10 in, and left a 4 KB output directory:
//! the artifact is only written after every layer is quantized, so a load-time
//! OOM is a total loss. Two mechanisms fed it, and both are checkable without a
//! GPU because both are properties of *storage*, not of any device:
//!
//! | § | property | test |
//! |---|---|---|
//! | 1 | a narrowed expert chunk still owns the whole stack's storage | `narrowed_expert_chunk_still_owns_the_whole_stack` |
//! | 1 | materialising the chunk shrinks the upload to the chunk | `materialised_chunk_owns_only_its_own_experts` |
//! | 1 | and does not move the data it selected | `materialised_chunk_holds_the_same_experts` |
//! | 1 | chunk-straddling experts bake bit-identically to solo bakes | `chunked_bake_is_bit_identical_to_per_expert_bakes` |
//! | 2 | a quantized 3-D stack retains no handle on its BF16 source | `quantized_stack_retains_no_handle_on_its_source` |
//! | 2 | the bake-to-host switch is process-global, not thread-local | `bake_to_host_flag_is_visible_from_another_thread` |
//!
//! §1 is the upload bug. `Tensor::narrow` + `reshape` produce a *view*: the
//! layout names this chunk, but the backing storage is still the entire
//! `[E, N, K]` stack. `Tensor::to_device` copies **storage** and clones the
//! layout, so the per-chunk CUDA quantize was shipping all 256 experts on every
//! call — 4.3 GiB instead of 268 MiB for V4 Flash, 48 times per layer.
//!
//! §2 is the retention. A bake never runs a forward pass; the quantized layers
//! exist only to be serialized. Leaving them on the accelerator cost the full
//! artifact in device memory (~1.6 GiB per layer, ~68 GB over 43 layers) and
//! forced the allocator to fit multi-GiB transients around a permanently
//! growing resident set.

use std::borrow::Cow;

use candle_core::{DType, Device, Storage, Tensor};

use super::bake_quality_tests::gen_fp4_dequant;
use super::{QtipBakeConfig, QtipCodebook, QtipLayer, QtipMode, TrellisSearch};
use crate::{QuantizeOntoGuard, QuantizedSerde};

/// Bake a 3-D expert stack through the production door, keeping the concrete
/// layer so the test can see what memory it holds.
fn bake_3d(stack: &Tensor) -> QtipLayer {
    QtipLayer::quantize_3d_concrete_with_bake_config(
        stack,
        &Device::Cpu,
        QtipMode::Viterbi,
        true,
        None,
        QtipBakeConfig {
            search: TrellisSearch::Beam { width: 4 },
            hessian: false,
            codebook: QtipCodebook::Gaussian,
        },
    )
    .expect("3-D bake must succeed")
}

/// Round-trip a baked layer through the artifact bytes, which is what the bake
/// actually produces. Used to check the streamed/serialized form agrees with
/// what the quantizer built.
fn round_trip(layer: &QtipLayer) -> QtipLayer {
    let bytes = layer.serialize().expect("serialize").into_owned();
    QtipLayer::deserialize_concrete(Cow::Owned(bytes), &Device::Cpu, QuantizeOntoGuard::new())
        .expect("the bake must produce a loadable artifact")
        .0
}

/// Number of elements in a tensor's *backing storage* — not its shape.
///
/// This is the number that decides how many bytes `Tensor::to_device` copies,
/// which is the whole point of §1: for a view it is the parent's element count,
/// however small the view is.
fn storage_elems(t: &Tensor) -> usize {
    let (storage, _layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cpu(cpu) => match t.dtype() {
            DType::F32 => cpu.as_slice::<f32>().unwrap().len(),
            DType::BF16 => cpu.as_slice::<half::bf16>().unwrap().len(),
            DType::U8 => cpu.as_slice::<u8>().unwrap().len(),
            other => panic!("storage_elems: unhandled dtype {other:?}"),
        },
        _ => panic!("storage_elems: expected CPU storage"),
    }
}

/// Address of the first element of a tensor's backing storage. Two tensors
/// sharing this share their allocation.
fn storage_addr(t: &Tensor) -> usize {
    let (storage, _layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cpu(cpu) => match t.dtype() {
            DType::F32 => cpu.as_slice::<f32>().unwrap().as_ptr() as usize,
            DType::BF16 => cpu.as_slice::<half::bf16>().unwrap().as_ptr() as usize,
            DType::U8 => cpu.as_slice::<u8>().unwrap().as_ptr() as usize,
            other => panic!("storage_addr: unhandled dtype {other:?}"),
        },
        _ => panic!("storage_addr: expected CPU storage"),
    }
}

fn expert_stack(e: usize, n: usize, k: usize, seed: u64) -> Tensor {
    Tensor::from_vec(
        gen_fp4_dequant(e * n, k, 0.02, seed),
        (e, n, k),
        &Device::Cpu,
    )
    .expect("fixture stack")
    .to_dtype(DType::BF16)
    .expect("bf16 cast")
}

// ---------------------------------------------------------------------------
// §1 — the per-chunk upload
// ---------------------------------------------------------------------------

/// The bug, stated as an assertion: a two-expert window of a 16-expert stack
/// still carries all sixteen experts' bytes. This is why the CUDA quantize was
/// uploading 4.3 GiB per chunk on V4 Flash instead of 268 MiB.
#[test]
fn narrowed_expert_chunk_still_owns_the_whole_stack() {
    let (e, n, k) = (16usize, 4usize, 32usize);
    let stack = expert_stack(e, n, k, 0x18_A0_01);

    let chunk = stack.narrow(0, 8, 2).unwrap();
    let rows_2d = chunk.reshape((2 * n, k)).unwrap();

    assert_eq!(rows_2d.elem_count(), 2 * n * k, "the view names 2 experts");
    assert_eq!(
        storage_elems(&rows_2d),
        e * n * k,
        "but it still owns all {e} experts' storage — `to_device` copies storage, \
         not layout, so this view uploads the whole stack"
    );
    assert_eq!(
        storage_addr(&rows_2d),
        storage_addr(&stack),
        "the view shares the parent allocation"
    );
}

/// The fix: `force_contiguous` gives the chunk its own allocation, sized to the
/// chunk. On V4 Flash this is the 16x drop from 4.3 GiB to 268 MiB per call.
#[test]
fn materialised_chunk_owns_only_its_own_experts() {
    let (e, n, k) = (16usize, 4usize, 32usize);
    let stack = expert_stack(e, n, k, 0x18_A0_02);

    let rows_2d = stack
        .narrow(0, 8, 2)
        .unwrap()
        .reshape((2 * n, k))
        .unwrap()
        .force_contiguous()
        .unwrap();

    assert_eq!(
        storage_elems(&rows_2d),
        2 * n * k,
        "the materialised chunk must own exactly its own experts"
    );
    assert_ne!(
        storage_addr(&rows_2d),
        storage_addr(&stack),
        "and a separate allocation from the stack"
    );
    // The ratio is the saving: 8x here, 16x for the production batch of 16 out
    // of 256 experts.
    assert_eq!(storage_elems(&stack) / storage_elems(&rows_2d), e / 2);
}

/// Materialising must select the window the layout named — an off-by-one in the
/// offset would silently bake the wrong experts, which no shape check catches.
#[test]
fn materialised_chunk_holds_the_same_experts() {
    let (e, n, k) = (16usize, 4usize, 32usize);
    let stack = expert_stack(e, n, k, 0x18_A0_03);

    for start in [0usize, 7, 14] {
        let view = stack
            .narrow(0, start, 2)
            .unwrap()
            .reshape((2 * n, k))
            .unwrap();
        let materialised = view.force_contiguous().unwrap();
        assert_eq!(
            materialised
                .flatten_all()
                .unwrap()
                .to_vec1::<half::bf16>()
                .unwrap(),
            view.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap(),
            "materialising the chunk at expert {start} changed its contents"
        );
    }
}

/// End-to-end guard on the same property, through the production 3-D door.
///
/// With 18 experts and the default `ARC_QTIP_EXPERT_BATCH` of 16 the stack is
/// baked as two chunks, so experts 15/16/17 straddle the boundary — exactly
/// where a bad offset would show. Each is compared against a solo bake of the
/// same expert (`e == 1` takes the single-chunk path, so it never materialises)
/// and must match bit for bit.
#[test]
fn chunked_bake_is_bit_identical_to_per_expert_bakes() {
    let (e, n, k) = (18usize, 2usize, 32usize);
    let stack = expert_stack(e, n, k, 0x18_A0_04);

    let chunked_layer = bake_3d(&stack);
    // The artifact the bake writes must carry the same blocks the quantizer
    // built — a chunk-window mistake that survived serialization would be
    // invisible in a purely in-memory comparison.
    let reloaded = round_trip(&chunked_layer);
    assert_eq!(
        reloaded
            .blocks
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        chunked_layer
            .blocks
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap(),
        "serialized artifact disagrees with the in-memory bake"
    );

    for idx in [0usize, 15, 16, 17] {
        let solo_stack = stack.narrow(0, idx, 1).unwrap().force_contiguous().unwrap();
        let solo_layer = bake_3d(&solo_stack);

        let from_chunked = chunked_layer
            .blocks
            .narrow(0, idx, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        let from_solo = solo_layer
            .blocks
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        assert_eq!(
            from_chunked, from_solo,
            "expert {idx} baked differently inside a chunked stack than on its own — \
             the per-chunk materialisation is selecting the wrong window"
        );
    }
}

// ---------------------------------------------------------------------------
// §2 — retention
// ---------------------------------------------------------------------------

/// The quantized stack must not keep the dequantized BF16 source alive.
///
/// The bake's per-layer working set is the BF16 expert stack (~4.3 GiB for V4
/// Flash); if any field of the produced layer aliased it, every layer would pin
/// one and the run would die long before the packed weights alone filled the
/// card. Checked by allocation identity and by total live bytes, both of which
/// are device-independent.
#[test]
fn quantized_stack_retains_no_handle_on_its_source() {
    let (e, n, k) = (4usize, 4usize, 32usize);
    let stack = expert_stack(e, n, k, 0x18_A0_05);
    let stack_addr = storage_addr(&stack);
    let stack_bytes = storage_elems(&stack) * 2; // BF16

    let layer = bake_3d(&stack);

    assert_ne!(
        storage_addr(&layer.blocks),
        stack_addr,
        "packed blocks must not alias the BF16 source stack"
    );
    assert_ne!(
        storage_addr(&layer.row_scales),
        stack_addr,
        "row scales must not alias the BF16 source stack"
    );

    // Total bytes the layer keeps alive: packed blocks + per-row scales + the
    // shared LUT + rotation signs. At 2 bits per weight the packed blocks are
    // 1/8 of the BF16 source, so anything near `stack_bytes` means a whole
    // dequantized copy is still being held.
    let live = storage_elems(&layer.blocks)
        + storage_elems(&layer.row_scales) * 4
        + storage_elems(&layer.lut) * 4
        + layer
            .rotation_signs
            .as_ref()
            .map(|s| storage_elems(s) * 4)
            .unwrap_or(0);
    let packed_bytes = e * n * (k / 4);
    assert_eq!(
        storage_elems(&layer.blocks),
        packed_bytes,
        "packed blocks must be exactly K/4 bytes per row (2 bits/weight)"
    );
    assert!(
        live < stack_bytes + storage_elems(&layer.lut) * 4,
        "the quantized layer holds {live} bytes on top of a {stack_bytes}-byte \
         source — a dequantized copy is being retained"
    );
}

/// The bake-to-host switch must be process-global.
///
/// `apply_immediate_isq_always` can run a layer's quantize on the immediate-ISQ
/// rayon pool rather than the construction thread. A thread-local flag would
/// read `false` there and silently leave that layer's experts on the device —
/// the exact silent-partial-fix shape that makes an OOM look like a mystery.
#[test]
fn bake_to_host_flag_is_visible_from_another_thread() {
    let previous = crate::bake_isq_to_host();
    crate::set_bake_isq_to_host(true);
    let seen = std::thread::spawn(crate::bake_isq_to_host)
        .join()
        .expect("probe thread");
    crate::set_bake_isq_to_host(previous);
    assert!(
        seen,
        "the bake-to-host switch must be readable from the ISQ pool threads"
    );
}
