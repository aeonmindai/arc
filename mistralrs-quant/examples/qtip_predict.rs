// Predictor: does Viterbi QTIP beat Greedy on real V4-Flash FP4 experts?
// Loads FP4-dequantized expert weights, quantizes each in both modes, and
// compares the forward matmul x@W against the FP4 reference (the quantity that
// actually drives model quality). (RUN-161)
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{QtipLayer, QtipMode};

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
    for name in &names {
        let w_f32 = tensors[name].to_dtype(DType::F32)?;
        let (_out, inn) = w_f32.dims2()?;
        let w_bf16 = w_f32.to_dtype(DType::BF16)?;
        let x_f32 = Tensor::randn(0f32, 1f32, (t, inn), &dev)?;
        let y_ref = x_f32.matmul(&w_f32.t()?)?;
        let x_bf16 = x_f32.to_dtype(DType::BF16)?;

        let lg = QtipLayer::quantize_with_mode(&w_bf16, None, &dev, QtipMode::Greedy)?;
        let yg = lg.forward(&x_bf16)?;
        let sg = cos(&y_ref, &yg);

        let lv = QtipLayer::quantize_with_mode(&w_bf16, None, &dev, QtipMode::Viterbi)?;
        let yv = lv.forward(&x_bf16)?;
        let sv = cos(&y_ref, &yv);

        println!(
            "{name:12}  matmul cos:  greedy={sg:.4}  viterbi={sv:.4}   (delta {:+.4})",
            sv - sg
        );
        sg_m += sg;
        sv_m += sv;
    }
    println!(
        "\nAVG matmul cos:  greedy={:.4}  viterbi={:.4}  (delta {:+.4})",
        sg_m / n as f64,
        sv_m / n as f64,
        (sv_m - sg_m) / n as f64
    );
    Ok(())
}
