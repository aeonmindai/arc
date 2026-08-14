//! Per-linear activation statistics accumulator.
//!
//! A [`CalibAccumulator`] is attached to a linear layer for the duration of a
//! forward-only calibration sweep. Every call to the layer's `forward` (or
//! `gather_forward`, for MoE expert stacks) feeds the layer's *input*
//! activations `X` (shape `[rows, in_features]` after flattening) into the
//! accumulator, which maintains:
//!
//! - `diag(XᵀX)` — always. This is the per-input-channel second moment used by
//!   activation-aware quantizers (GPTQ/AWQ-style per-column weighting, QTIP
//!   trellis search) and by TD-MoE whitening.
//! - Optionally a block-diagonal or full `XᵀX` (see [`GramMode`]) for consumers
//!   that need real cross-channel structure (Hessian-aware search, EoRA).
//! - Per-expert `diag(XᵀX)` for MoE expert stacks, keyed by the routing
//!   indices handed to `gather_forward`.
//!
//! # Numerics
//!
//! Reductions run in **f32 on the layer's device** (cheap, and the within-chunk
//! sum of at most a few thousand squares is well inside f32's range), and the
//! per-chunk results are accumulated into **f64 host** accumulators. This keeps
//! the sweep fast while avoiding the catastrophic precision loss of summing
//! tens of thousands of chunks in f32. The per-expert path accumulates entirely
//! in f64 on the host because it is index-scattered and offline anyway.
//!
//! Nothing is normalised here: the artifact stores raw sums plus the token and
//! call counts, so a consumer can pick its own normalisation (`/tokens` for a
//! covariance, raw for a Hessian `XᵀX`).

use std::sync::{Arc, RwLock};

use candle_core::{Context, DType, Result, Tensor, D};
use serde::{Deserialize, Serialize};

/// How much of `XᵀX` to accumulate beyond the diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GramMode {
    /// `diag(XᵀX)` only. Storage: `in_features` f64 per layer.
    #[default]
    DiagOnly,
    /// Full `XᵀX` when `in_features <= max_dim`, otherwise diagonal only (the
    /// layer is still emitted, just without a gram block). Storage:
    /// `in_features²` f64 — only sane for small layers.
    Full { max_dim: usize },
    /// Block-diagonal `XᵀX` with square blocks of `block` consecutive input
    /// channels (the trailing block is truncated when `in_features` is not a
    /// multiple of `block`). Storage: `≈ in_features * block` f64, i.e. linear
    /// in the layer width. This is the practical choice for 7k-wide layers.
    Blockwise { block: usize },
}

/// Options controlling what a calibration sweep collects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibOptions {
    /// Off-diagonal `XᵀX` collection policy.
    pub gram: GramMode,
    /// Collect per-expert statistics for MoE expert stacks (layers driven
    /// through `gather_forward`). Off by default: a 256-expert stack multiplies
    /// the per-layer artifact size by 256.
    pub per_expert: bool,
    /// An expert that saw fewer than this many routed rows is flagged
    /// [`ExpertStatus::Insufficient`] so consumers can fall back to the
    /// layer-global statistics instead of trusting a noisy estimate. Experts
    /// with *zero* routed rows are always flagged [`ExpertStatus::ZeroTokens`]
    /// and carry no diagonal at all.
    pub min_expert_tokens: u64,
}

impl Default for CalibOptions {
    fn default() -> Self {
        Self {
            gram: GramMode::DiagOnly,
            per_expert: false,
            min_expert_tokens: 32,
        }
    }
}

/// Quality marker for a single expert's statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertStatus {
    /// Enough routed rows; the diagonal is trustworthy.
    Ok,
    /// Some routed rows, but fewer than `min_expert_tokens`. The diagonal is
    /// still emitted (it is real data) but consumers should prefer the
    /// layer-global statistics or blend towards them.
    Insufficient,
    /// The expert saw no routed rows in this calibration set. **No diagonal is
    /// emitted** — an all-zero diagonal would silently become a degenerate
    /// (singular) covariance downstream.
    ZeroTokens,
}

/// Layout of a collected gram (`XᵀX`) block set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GramLayout {
    /// Dense row-major `[dim, dim]`.
    Full { dim: usize },
    /// Concatenated square diagonal blocks. Block `j` covers input channels
    /// `[j*block, min((j+1)*block, dim))` and is stored row-major with width
    /// `min(block, dim - j*block)`.
    Blockwise { dim: usize, block: usize },
}

impl GramLayout {
    /// Number of blocks in this layout.
    pub fn num_blocks(&self) -> usize {
        match self {
            Self::Full { .. } => 1,
            Self::Blockwise { dim, block } => dim.div_ceil((*block).max(1)),
        }
    }

    /// Width of block `j` (its side length).
    pub fn block_width(&self, j: usize) -> usize {
        match self {
            Self::Full { dim } => *dim,
            Self::Blockwise { dim, block } => {
                let start = j * block;
                (*dim).saturating_sub(start).min(*block)
            }
        }
    }

    /// Element offset of block `j` within the flat data array.
    pub fn block_offset(&self, j: usize) -> usize {
        (0..j).map(|b| self.block_width(b).pow(2)).sum()
    }

    /// Total number of f64 elements this layout occupies.
    pub fn len(&self) -> usize {
        (0..self.num_blocks())
            .map(|j| self.block_width(j).pow(2))
            .sum()
    }

    /// Whether the layout carries no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A collected gram together with its layout.
#[derive(Debug, Clone, PartialEq)]
pub struct GramBlocks {
    pub layout: GramLayout,
    /// Raw `XᵀX` sums (not normalised by token count).
    pub data: Vec<f64>,
}

impl GramBlocks {
    /// Row-major slice of block `j` and its side length.
    pub fn block(&self, j: usize) -> Option<(&[f64], usize)> {
        let w = self.layout.block_width(j);
        if w == 0 {
            return None;
        }
        let off = self.layout.block_offset(j);
        self.data.get(off..off + w * w).map(|s| (s, w))
    }
}

/// Per-expert statistics for one expert of an MoE stack.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertCalibData {
    pub expert: usize,
    /// Number of `(token, expert)` routing pairs that reached this expert.
    pub tokens: u64,
    pub status: ExpertStatus,
    /// `diag(XᵀX)` over this expert's routed rows. `None` iff
    /// [`ExpertStatus::ZeroTokens`].
    pub diag: Option<Vec<f64>>,
}

/// Everything a single layer's accumulator produced.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibLayerData {
    pub in_features: usize,
    /// Rows of `X` folded in. For an MoE stack this counts `(token, expert)`
    /// routing pairs, so it equals `sum(experts[*].tokens)`.
    pub tokens: u64,
    /// Number of forward calls observed.
    pub calls: u64,
    /// `diag(XᵀX)`, length `in_features`.
    pub diag: Vec<f64>,
    pub gram: Option<GramBlocks>,
    /// Empty unless [`CalibOptions::per_expert`] was set and the layer was
    /// driven through `gather_forward`.
    pub experts: Vec<ExpertCalibData>,
}

#[derive(Debug)]
struct Inner {
    in_features: usize,
    tokens: u64,
    calls: u64,
    diag: Vec<f64>,
    gram: Option<GramBlocks>,
    experts: Option<Vec<ExpertAccum>>,
    opts: CalibOptions,
}

#[derive(Debug)]
struct ExpertAccum {
    tokens: u64,
    diag: Vec<f64>,
}

/// Handle to a layer's calibration accumulator. Cloning shares the accumulator
/// (it lives behind an `Arc<RwLock<_>>`) so a layer can hand a clone to a
/// wrapper without splitting the statistics.
#[derive(Debug, Clone)]
pub struct CalibAccumulator(Arc<RwLock<Option<Inner>>>);

impl CalibAccumulator {
    /// Create an accumulator for a layer with `in_features` input channels.
    ///
    /// `num_experts` is `Some(e)` for a 3-D MoE expert stack of shape
    /// `[e, out_features, in_features]`; per-expert accumulation is only armed
    /// when it is `Some` *and* [`CalibOptions::per_expert`] is set.
    pub fn new(in_features: usize, num_experts: Option<usize>, opts: CalibOptions) -> Result<Self> {
        if in_features == 0 {
            candle_core::bail!("CalibAccumulator: in_features must be non-zero");
        }
        let gram = match opts.gram {
            GramMode::DiagOnly => None,
            GramMode::Full { max_dim } => {
                if in_features <= max_dim {
                    let layout = GramLayout::Full { dim: in_features };
                    Some(GramBlocks {
                        data: vec![0f64; layout.len()],
                        layout,
                    })
                } else {
                    None
                }
            }
            GramMode::Blockwise { block } => {
                if block == 0 {
                    candle_core::bail!("CalibAccumulator: gram block size must be non-zero");
                }
                let layout = GramLayout::Blockwise {
                    dim: in_features,
                    block: block.min(in_features),
                };
                Some(GramBlocks {
                    data: vec![0f64; layout.len()],
                    layout,
                })
            }
        };
        let experts = match (opts.per_expert, num_experts) {
            (true, Some(e)) if e > 0 => Some(
                (0..e)
                    .map(|_| ExpertAccum {
                        tokens: 0,
                        diag: vec![0f64; in_features],
                    })
                    .collect(),
            ),
            _ => None,
        };
        Ok(Self(Arc::new(RwLock::new(Some(Inner {
            in_features,
            tokens: 0,
            calls: 0,
            diag: vec![0f64; in_features],
            gram,
            experts,
            opts,
        })))))
    }

    /// Fold a dense forward's input activations into the accumulator.
    ///
    /// `inp` may have any rank; everything but the last dimension is treated as
    /// the row axis.
    pub fn process(&self, inp: &Tensor) -> Result<()> {
        let mut handle = self.0.write().unwrap();
        let this = handle
            .as_mut()
            .context("Calibration accumulator was already finished")?;

        let x = inp
            .reshape(((), inp.dim(D::Minus1)?))?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (rows, cols) = x.dims2()?;
        if cols != this.in_features {
            candle_core::bail!(
                "Calibration accumulator: expected {} input features, got {cols}",
                this.in_features
            );
        }
        if rows == 0 {
            return Ok(());
        }

        this.calls += 1;
        this.tokens += rows as u64;

        // f32 reduction on-device, f64 accumulation on the host.
        let chunk_diag = x.sqr()?.sum(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        for (acc, v) in this.diag.iter_mut().zip(chunk_diag) {
            *acc += v as f64;
        }

        accumulate_gram(this, &x, 1.0)?;
        Ok(())
    }

    /// Fold an MoE `gather_forward`'s input activations and routing indices.
    ///
    /// Accepted shapes mirror `QuantMethod::gather_forward`:
    /// - `inp` `[b, s, 1, 1, hidden]` with `indices` `[b, s, k]`
    /// - `inp` `[n, 1, hidden]` with `indices` `[n, k]`
    ///
    /// Each token contributes one row per selected expert, so the layer-global
    /// `diag(XᵀX)` is the multiplicity-weighted sum (`k ×` the per-token sum)
    /// and equals the sum of the per-expert diagonals.
    pub fn process_gather(&self, inp: &Tensor, indices: &Tensor) -> Result<()> {
        let mut handle = self.0.write().unwrap();
        let this = handle
            .as_mut()
            .context("Calibration accumulator was already finished")?;

        let hidden = inp.dim(D::Minus1)?;
        if hidden != this.in_features {
            candle_core::bail!(
                "Calibration accumulator: expected {} input features, got {hidden}",
                this.in_features
            );
        }
        let x = inp
            .reshape(((), hidden))?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (n_tokens, _) = x.dims2()?;
        let flat_idx = indices
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        if n_tokens == 0 || flat_idx.is_empty() {
            return Ok(());
        }
        if flat_idx.len() % n_tokens != 0 {
            candle_core::bail!(
                "Calibration accumulator: {} routing indices do not divide {n_tokens} tokens",
                flat_idx.len()
            );
        }
        let top_k = flat_idx.len() / n_tokens;

        this.calls += 1;
        this.tokens += (n_tokens * top_k) as u64;

        // Per-token squares once, then scatter by routing multiplicity. Squares
        // are computed in f32 on-device; the scatter accumulates in f64.
        let sq = x
            .sqr()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Layer-global diagonal: every token is used exactly `top_k` times.
        for t in 0..n_tokens {
            let row = &sq[t * hidden..(t + 1) * hidden];
            for (acc, v) in this.diag.iter_mut().zip(row) {
                *acc += (*v as f64) * top_k as f64;
            }
        }

        if let Some(experts) = this.experts.as_mut() {
            let n_experts = experts.len();
            for (pair, &e) in flat_idx.iter().enumerate() {
                let e = e as usize;
                if e >= n_experts {
                    candle_core::bail!(
                        "Calibration accumulator: routing index {e} out of range for {n_experts} experts"
                    );
                }
                let t = pair / top_k;
                let row = &sq[t * hidden..(t + 1) * hidden];
                let acc = &mut experts[e];
                acc.tokens += 1;
                for (a, v) in acc.diag.iter_mut().zip(row) {
                    *a += *v as f64;
                }
            }
        }

        // Gram over the multiplicity-expanded rows == `top_k * XᵀX` because
        // every token selects exactly `top_k` experts.
        accumulate_gram(this, &x, top_k as f64)?;
        Ok(())
    }

    /// Consume the accumulator and return the collected statistics. Subsequent
    /// `process*` calls fail.
    pub fn finish(&self) -> Result<CalibLayerData> {
        let mut handle = self.0.write().unwrap();
        let this = handle
            .take()
            .context("Calibration accumulator was already finished")?;

        let min_tokens = this.opts.min_expert_tokens;
        let experts = this
            .experts
            .map(|accs| {
                accs.into_iter()
                    .enumerate()
                    .map(|(expert, acc)| {
                        let (status, diag) = if acc.tokens == 0 {
                            // Never emit an all-zero diagonal: downstream it
                            // would become a singular covariance.
                            (ExpertStatus::ZeroTokens, None)
                        } else if acc.tokens < min_tokens {
                            (ExpertStatus::Insufficient, Some(acc.diag))
                        } else {
                            (ExpertStatus::Ok, Some(acc.diag))
                        };
                        ExpertCalibData {
                            expert,
                            tokens: acc.tokens,
                            status,
                            diag,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(CalibLayerData {
            in_features: this.in_features,
            tokens: this.tokens,
            calls: this.calls,
            diag: this.diag,
            gram: this.gram,
            experts,
        })
    }
}

fn accumulate_gram(this: &mut Inner, x: &Tensor, scale: f64) -> Result<()> {
    let Some(gram) = this.gram.as_mut() else {
        return Ok(());
    };
    let layout = gram.layout;
    for j in 0..layout.num_blocks() {
        let w = layout.block_width(j);
        if w == 0 {
            continue;
        }
        let start = match layout {
            GramLayout::Full { .. } => 0,
            GramLayout::Blockwise { block, .. } => j * block,
        };
        let xb = x.narrow(1, start, w)?.contiguous()?;
        let block = xb
            .t()?
            .contiguous()?
            .matmul(&xb)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let off = layout.block_offset(j);
        for (acc, v) in gram.data[off..off + w * w].iter_mut().zip(block) {
            *acc += (v as f64) * scale;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Deterministic `[rows, cols]` matrix with a wide dynamic range across
    /// columns, so a wrong reduction axis is unmistakable.
    fn make_x(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f32;
                let c = (i % cols) as f32;
                (c + 1.0) * (0.5 + 0.25 * ((r * 0.7 + c * 0.3).sin()))
            })
            .collect()
    }

    fn analytic_gram(data: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut g = vec![0f64; cols * cols];
        for r in 0..rows {
            for i in 0..cols {
                let a = data[r * cols + i] as f64;
                for j in 0..cols {
                    g[i * cols + j] += a * data[r * cols + j] as f64;
                }
            }
        }
        g
    }

    #[test]
    fn diag_matches_analytic_xtx_diagonal() {
        let (rows, cols) = (37, 11);
        let data = make_x(rows, cols);
        let x = Tensor::from_vec(data.clone(), (rows, cols), &Device::Cpu).unwrap();

        let acc = CalibAccumulator::new(cols, None, CalibOptions::default()).unwrap();
        acc.process(&x).unwrap();
        let out = acc.finish().unwrap();

        let expected = analytic_gram(&data, rows, cols);
        assert_eq!(out.tokens, rows as u64);
        assert_eq!(out.calls, 1);
        for c in 0..cols {
            let want = expected[c * cols + c];
            assert!(
                (out.diag[c] - want).abs() <= 1e-4 * want.abs().max(1.0),
                "col {c}: got {} want {want}",
                out.diag[c]
            );
        }
        assert!(out.gram.is_none());
    }

    #[test]
    fn multiple_calls_accumulate_and_rank3_input_flattens() {
        let (b, s, cols) = (3, 5, 7);
        let data = make_x(b * s, cols);
        let x = Tensor::from_vec(data.clone(), (b, s, cols), &Device::Cpu).unwrap();

        let acc = CalibAccumulator::new(cols, None, CalibOptions::default()).unwrap();
        acc.process(&x).unwrap();
        acc.process(&x).unwrap();
        let out = acc.finish().unwrap();

        let expected = analytic_gram(&data, b * s, cols);
        assert_eq!(out.tokens, (2 * b * s) as u64);
        assert_eq!(out.calls, 2);
        for c in 0..cols {
            let want = 2.0 * expected[c * cols + c];
            assert!((out.diag[c] - want).abs() <= 1e-4 * want.abs().max(1.0));
        }
    }

    #[test]
    fn full_gram_matches_analytic_xtx() {
        let (rows, cols) = (23, 9);
        let data = make_x(rows, cols);
        let x = Tensor::from_vec(data.clone(), (rows, cols), &Device::Cpu).unwrap();

        let opts = CalibOptions {
            gram: GramMode::Full { max_dim: 64 },
            ..Default::default()
        };
        let acc = CalibAccumulator::new(cols, None, opts).unwrap();
        acc.process(&x).unwrap();
        let out = acc.finish().unwrap();

        let expected = analytic_gram(&data, rows, cols);
        let gram = out.gram.expect("full gram requested");
        assert_eq!(gram.layout, GramLayout::Full { dim: cols });
        assert_eq!(gram.data.len(), cols * cols);
        for (i, (got, want)) in gram.data.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "elem {i}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn full_gram_skipped_above_max_dim() {
        let opts = CalibOptions {
            gram: GramMode::Full { max_dim: 4 },
            ..Default::default()
        };
        let acc = CalibAccumulator::new(9, None, opts).unwrap();
        let x = Tensor::from_vec(make_x(3, 9), (3, 9), &Device::Cpu).unwrap();
        acc.process(&x).unwrap();
        let out = acc.finish().unwrap();
        assert!(out.gram.is_none(), "gram must be skipped above max_dim");
        assert_eq!(out.diag.len(), 9, "diagonal is still collected");
    }

    #[test]
    fn blockwise_gram_matches_diagonal_blocks_of_xtx() {
        // 10 columns, block 4 -> widths [4, 4, 2] (ragged trailing block).
        let (rows, cols, block) = (17, 10, 4);
        let data = make_x(rows, cols);
        let x = Tensor::from_vec(data.clone(), (rows, cols), &Device::Cpu).unwrap();

        let opts = CalibOptions {
            gram: GramMode::Blockwise { block },
            ..Default::default()
        };
        let acc = CalibAccumulator::new(cols, None, opts).unwrap();
        acc.process(&x).unwrap();
        let out = acc.finish().unwrap();

        let expected = analytic_gram(&data, rows, cols);
        let gram = out.gram.expect("blockwise gram requested");
        assert_eq!(gram.layout.num_blocks(), 3);
        assert_eq!(gram.layout.block_width(2), 2);
        assert_eq!(gram.data.len(), 4 * 4 + 4 * 4 + 2 * 2);
        for j in 0..gram.layout.num_blocks() {
            let (blk, w) = gram.block(j).unwrap();
            for a in 0..w {
                for b in 0..w {
                    let want = expected[(j * block + a) * cols + (j * block + b)];
                    let got = blk[a * w + b];
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "block {j} ({a},{b}): got {got} want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn gather_per_expert_diags_sum_to_global_and_mark_unrouted() {
        let (n_tokens, hidden, n_experts, top_k) = (6, 5, 4, 2);
        let data = make_x(n_tokens, hidden);
        let x = Tensor::from_vec(data.clone(), (n_tokens, 1, hidden), &Device::Cpu).unwrap();
        // Expert 3 is never selected; expert 0 gets every token.
        let routing: Vec<u32> = vec![0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2];
        let indices = Tensor::from_vec(routing.clone(), (n_tokens, top_k), &Device::Cpu).unwrap();

        let opts = CalibOptions {
            per_expert: true,
            min_expert_tokens: 4,
            ..Default::default()
        };
        let acc = CalibAccumulator::new(hidden, Some(n_experts), opts).unwrap();
        acc.process_gather(&x, &indices).unwrap();
        let out = acc.finish().unwrap();

        assert_eq!(out.tokens, (n_tokens * top_k) as u64);
        assert_eq!(out.experts.len(), n_experts);

        // Per-expert diagonals must sum to the layer-global diagonal.
        let mut summed = vec![0f64; hidden];
        for e in &out.experts {
            if let Some(d) = &e.diag {
                for (s, v) in summed.iter_mut().zip(d) {
                    *s += v;
                }
            }
        }
        for (c, (s, g)) in summed.iter().zip(&out.diag).enumerate() {
            assert!(
                (s - g).abs() <= 1e-6 * g.abs().max(1.0),
                "col {c}: experts sum {s} vs global {g}"
            );
        }

        // Expert 0 saw 6 rows (>= min 4) -> Ok. Experts 1 and 2 saw 3 -> Insufficient.
        assert_eq!(out.experts[0].tokens, 6);
        assert_eq!(out.experts[0].status, ExpertStatus::Ok);
        assert_eq!(out.experts[1].tokens, 3);
        assert_eq!(out.experts[1].status, ExpertStatus::Insufficient);
        assert!(out.experts[1].diag.is_some());

        // Expert 3 was never routed: marked, and carries no diagonal.
        assert_eq!(out.experts[3].tokens, 0);
        assert_eq!(out.experts[3].status, ExpertStatus::ZeroTokens);
        assert!(
            out.experts[3].diag.is_none(),
            "zero-token experts must not emit a garbage diagonal"
        );
    }

    #[test]
    fn gather_without_per_expert_still_collects_global_stats() {
        let (n_tokens, hidden, top_k) = (4, 3, 2);
        let x = Tensor::from_vec(
            make_x(n_tokens, hidden),
            (n_tokens, 1, hidden),
            &Device::Cpu,
        )
        .unwrap();
        let indices = Tensor::from_vec(
            vec![0u32, 1, 1, 0, 0, 1, 1, 0],
            (n_tokens, top_k),
            &Device::Cpu,
        )
        .unwrap();
        let acc = CalibAccumulator::new(hidden, Some(2), CalibOptions::default()).unwrap();
        acc.process_gather(&x, &indices).unwrap();
        let out = acc.finish().unwrap();
        assert!(out.experts.is_empty());
        assert_eq!(out.tokens, (n_tokens * top_k) as u64);
        assert!(out.diag.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn finish_is_one_shot() {
        let acc = CalibAccumulator::new(4, None, CalibOptions::default()).unwrap();
        assert!(acc.finish().is_ok());
        assert!(acc.finish().is_err());
    }

    #[test]
    fn mismatched_in_features_is_an_error() {
        let acc = CalibAccumulator::new(4, None, CalibOptions::default()).unwrap();
        let x = Tensor::from_vec(make_x(2, 5), (2, 5), &Device::Cpu).unwrap();
        assert!(acc.process(&x).is_err());
    }
}
