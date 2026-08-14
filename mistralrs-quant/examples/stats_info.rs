//! Dump the contents of a `.arccalib` calibration artifact.
//!
//! ```bash
//! cargo run -p mistralrs-quant --example stats_info -- stats.arccalib
//! cargo run -p mistralrs-quant --example stats_info -- stats.arccalib --layers
//! cargo run -p mistralrs-quant --example stats_info -- stats.arccalib --layer 12
//! ```

use mistralrs_quant::{CalibrationArtifact, ExpertStatus};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) if !p.starts_with('-') => p,
        _ => {
            eprintln!(
                "usage: stats_info <artifact.arccalib> [--layers] [--layer <isq_index>]\n\n\
                 \x20 --layers          list every layer\n\
                 \x20 --layer <index>   detail one layer by ISQ index"
            );
            std::process::exit(2);
        }
    };

    let mut list_layers = false;
    let mut only: Option<usize> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--layers" => list_layers = true,
            "--layer" => {
                only = Some(
                    args.next()
                        .ok_or("--layer needs an ISQ index")?
                        .parse::<usize>()?,
                )
            }
            other => return Err(format!("unknown flag `{other}`").into()),
        }
    }

    let art = CalibrationArtifact::load(&path)?;
    print!("{}", art.summary());

    if let Some(idx) = only {
        let l = art
            .by_isq_index(idx)
            .ok_or_else(|| format!("no layer with isq_index {idx}"))?;
        println!("\n=== ISQ layer {idx} ===");
        println!("artifact_name   {}", l.artifact_name);
        println!("name            {}", l.name.as_deref().unwrap_or("-"));
        println!("layer_num       {:?}", l.layer_num);
        println!("supported       {}", l.supported);
        println!("in_features     {}", l.in_features);
        println!("tokens / calls  {} / {}", l.tokens, l.calls);
        if let Some(n) = l.normalized_diag() {
            let (mut min, mut max, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
            for v in &n {
                min = min.min(*v);
                max = max.max(*v);
                sum += *v;
            }
            println!(
                "diag/tokens     min {min:.6e}  mean {:.6e}  max {max:.6e}  (dynamic range {:.1}x)",
                sum / n.len() as f64,
                if min > 0.0 { max / min } else { f64::INFINITY }
            );
        }
        match &l.gram {
            Some(g) => println!(
                "gram            {:?}, {} blocks, {} f64",
                g.layout,
                g.layout.num_blocks(),
                g.data.len()
            ),
            None => println!("gram            -"),
        }
        if !l.experts.is_empty() {
            let ok = l
                .experts
                .iter()
                .filter(|e| e.status == ExpertStatus::Ok)
                .count();
            let insufficient = l
                .experts
                .iter()
                .filter(|e| e.status == ExpertStatus::Insufficient)
                .count();
            let zero = l.zero_token_experts();
            println!(
                "experts         {} total: {ok} ok, {insufficient} insufficient, {} zero-token",
                l.experts.len(),
                zero.len()
            );
            if !zero.is_empty() {
                println!("zero-token      {zero:?}");
            }
        }
        return Ok(());
    }

    if list_layers {
        println!(
            "\n{:>5}  {:<10} {:>8} {:>10} {:>6} {:>7}  symbolic",
            "isq", "name", "in_feat", "tokens", "gram", "experts"
        );
        for l in &art.layers {
            println!(
                "{:>5}  {:<10} {:>8} {:>10} {:>6} {:>7}  {}",
                l.isq_index,
                l.artifact_name,
                l.in_features,
                l.tokens,
                if l.gram.is_some() { "yes" } else { "-" },
                if l.experts.is_empty() {
                    "-".to_string()
                } else {
                    format!(
                        "{}/{}",
                        l.experts.len() - l.zero_token_experts().len(),
                        l.experts.len()
                    )
                },
                l.name.as_deref().unwrap_or("-")
            );
        }
    }

    Ok(())
}
