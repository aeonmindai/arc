// Predictor: what does the Hadamard incoherence rotation buy on REAL V4-Flash
// FP4 experts? Loads FP4-dequantized expert weights, quantizes each with and
// without rotation, and compares the forward matmul x@W against the FP4
// reference (the quantity that actually drives model quality). (RUN-161)
//
// This used to be a Viterbi-vs-Greedy predictor. Two reasons it is not:
//   1. It could not run. `quantize_with_mode` refuses `QtipMode::Greedy` in
//      EVERY build (DOCTRINE D4 clause 1), so the greedy arm hard-errored on
//      the first expert and the whole binary was dead.
//   2. The question is settled. wave3-G / wave13-AD measured greedy+no-rotation
//      at matmul cos 0.675 against Viterbi+rotation's 0.963 on exactly this
//      fixture family. Nothing here would add to that.
// The rotation axis is the one still worth probing on real weights: D4 records
// that greedy degraded on BOTH axes at once because it silently disabled
// rotation, and the split between the two has only ever been measured on
// synthetic fixtures.
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{QtipLayer, QtipMode, QtipRotation};

fn cos(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let b = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
    let na = (&a * &a)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt() as f64;
    let nb = (&b * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt() as f64;
    dot / (na * nb + 1e-12)
}

fn main() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let tensors = candle_core::safetensors::load("/ephemeral/experts_fp4.safetensors", &dev)?;
    let t = 32usize;
    let mut names: Vec<_> = tensors.keys().cloned().collect();
    names.sort();
    let (mut sg_m, mut sv_m) = (0f64, 0f64);
    let n = names.len();
    if n == 0 {
        candle_core::bail!(
            "/ephemeral/experts_fp4.safetensors holds no tensors — the averages below would be \
             printed from an empty loop (0/0)."
        );
    }
    // Production policy for the rotated arm, asked rather than assumed.
    let rot = QtipRotation::for_mode(QtipMode::Viterbi).enabled();
    if !rot {
        candle_core::bail!(
            "QtipRotation::for_mode(Viterbi) reports rotation OFF: both arms below would be \
             identical and the comparison would be vacuous."
        );
    }
    for name in &names {
        let w_f32 = tensors[name].to_dtype(DType::F32)?;
        let (_out, inn) = w_f32.dims2()?;
        let w_bf16 = w_f32.to_dtype(DType::BF16)?;
        let x_f32 = Tensor::randn(0f32, 1f32, (t, inn), &dev)?;
        let y_ref = x_f32.matmul(&w_f32.t()?)?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;

        let lg = QtipLayer::quantize_with_options(&w_bf16, None, &dev, QtipMode::Viterbi, false)?;
        let yg = lg.forward(&x_bf16)?;
        let sg = cos(&y_ref, &yg);

        let lv = QtipLayer::quantize_with_mode(&w_bf16, None, &dev, QtipMode::Viterbi)?;
        let yv = lv.forward(&x_bf16)?;
        let sv = cos(&y_ref, &yv);

        println!(
            "{name:12}  matmul cos:  no-rot={sg:.4}  rot={sv:.4}   (delta {:+.4})",
            sv - sg
        );
        sg_m += sg;
        sv_m += sv;
    }
    println!(
        "\nAVG matmul cos over {n} experts (Viterbi both arms):  no-rot={:.4}  rot={:.4}  (delta {:+.4})",
        sg_m / n as f64,
        sv_m / n as f64,
        (sv_m - sg_m) / n as f64
    );
    Ok(())
}
