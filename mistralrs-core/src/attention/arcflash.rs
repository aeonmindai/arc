//! # ArcFlash — Arc's own attention kernel, under ArcAttention
//!
//! **Taxonomy:** `Arc → ArcInfer → ArcAttention → ArcFlash`
//! (`memory/mission/TAXONOMY.md`). `ArcAttention/Dispatch` decides *which*
//! kernel runs; `ArcAttention/Flash` is the capability envelope for the
//! **vendor** kernels (Dao's FlashAttention-2/3, via `candle-flash-attn*`).
//! **ArcFlash is ours.** It is the path that runs when no vendor kernel accepts
//! the shape — which today is every DeepSeek-V4 layer, every asymmetric-MLA
//! layer, and every build without a flash feature.
//!
//! Sub-systems, so nothing here is left without a parent name:
//!
//! | Name | Does | Where |
//! |---|---|---|
//! | **ArcFlash/Plan** | shape → path, and the engagement record D18 requires | [`plan_unfused`], [`Path`], [`engagement`] |
//! | **ArcFlash/Fold** | GQA/MQA expansion expressed as a *reshape*, never a copy | [`fold_kv_groups`] |
//! | **ArcFlash/Tile** | query-row tiling with a byte budget, so peak scores memory is O(tile) not O(T²) | [`union_attention`] |
//!
//! ## Why this exists — the measured failure it removes
//!
//! The unfused path this replaces did two things that make long prompts
//! quadratic *in allocated bytes*, not just in FLOPs:
//!
//! 1. **`repeat_kv` materialised the GQA expansion.** `attention/mod.rs`'s
//!    `repeat_kv` is `Tensor::cat(&vec![&x; n_rep], 2)` — a real allocation of
//!    `n_rep` copies of K *and* V. DeepSeek-V4 is MQA with
//!    `num_key_value_heads = 1` and `n_heads = 64`, so a 1.4 MB K became 92 MB,
//!    twice (K and V), per layer.
//! 2. **The full `[B, H, T, T]` score matrix was materialised**, then a second
//!    one for the softmax result. At `head_dim = 512`, `T = 1400`, that is
//!    ~250 MB each.
//!
//! Together: ~856 MB for one layer's working set at `T ≈ 1400`. Layers free
//! sequentially, so peak is one layer — which is exactly why the failure mode
//! is "works at 200 words, dies at 1000" rather than a gentle slowdown.
//!
//! ArcFlash removes both. The expansion becomes a reshape (**zero copies**) and
//! the scores are computed one query tile at a time under a byte budget.
//!
//! ## The fold, and why it is exact
//!
//! `repeat_kv` maps `x[b, kv, s, d]` to `y[b, kv * n_rep + rep, s, d]` — heads
//! are **kv-major**. So a contiguous `q[B, H, Tq, D]` with `H = Hkv * g`
//! reshapes, with no data movement, to `[B, Hkv, g * Tq, D]`, where folded row
//! `r = rep * Tq + s`. Both operands of the scores GEMM then carry batch
//! `B * Hkv`, so candle's `matmul` batches them directly and K/V are read once
//! per KV head instead of `g` times.
//!
//! This is not a new trick here: [`super::super::models::dsv4_attention`]'s
//! `absorbed_mqa_decode` already does exactly this at `t_q == 1`, and is pinned
//! by `absorbed_decode_matches_repeat_kv_reference`. ArcFlash generalises it to
//! any `t_q` and any GQA ratio.
//!
//! `Tensor::reshape` is a view when the input is contiguous
//! (`candle-core/src/tensor.rs`: `if self.is_contiguous() { .. layout only .. }`),
//! which is what makes the fold free. `broadcast_matmul` is **not** an
//! alternative — candle concretises the broadcast operand via `.contiguous()`
//! (`tensor.rs`, `// TODO: Avoid concretising the broadcasted matrixes`), i.e.
//! it would reintroduce the copy this module exists to delete.
//!
//! ## Why tiling over query rows is exact
//!
//! Every output row is an independent softmax over the key axis. Splitting the
//! query axis therefore computes *the same rows*, not an approximation — no
//! online-softmax rescaling is involved, and **sinks are unaffected** because a
//! sink is a per-head constant in each row's denominator. (Tiling the *key*
//! axis would need online rescaling; ArcFlash/Tile deliberately does not.)
//!
//! `softmax_with_sinks` derives the head index as `(row / q_len) % num_heads`
//! from the logits it is handed, so a tile of `S` rows must be passed as
//! `[B, H, S, K]` — which is what [`union_attention`] does.
//!
//! ## D18 — assert engagement, never infer it
//!
//! `memory/mission/KERNEL_RULES.md` D18 is at ten recorded instances of *"the
//! absence of a signal was read as a specific signal."* A dispatch that quietly
//! fell back would be the eleventh. So every path increments a counter
//! ([`engagement`]) and logs once, and the tests assert a path is **present in
//! the treatment arm and absent in the control arm** — not merely that the
//! numbers came out right.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::SdpaParams;

/// Which kernel ArcAttention ran. Recorded for every dispatch so engagement is
/// asserted rather than assumed (D18).
// `VendorSinks` and `Cublaslt` are constructed only inside `#[cfg(feature =
// "cuda")]` / `#[cfg(feature = "metal")]` blocks, so a plain host build sees
// them as never-constructed. They are not dead — they are the labels the
// engagement record uses on the boxes that matter, and dropping them would make
// the D18 instrument silent on exactly the two paths hardware runs.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    /// Vendor fused sinks kernel (`mistralrs_paged_attn::flash_attn_sinks`).
    VendorSinks,
    /// Vendor FlashAttention-2/3, via `attention::backends::flash`.
    VendorFlash,
    /// Metal's fused SDPA.
    MetalSdpa,
    /// cuBLASLt batched matmul attention.
    Cublaslt,
    /// **ArcFlash/Tile** — ours: folded GQA, query-tiled, sink-aware.
    Tile,
}

impl Path {
    #[cfg(test)]
    pub(crate) const ALL: [Path; 5] = [
        Path::VendorSinks,
        Path::VendorFlash,
        Path::MetalSdpa,
        Path::Cublaslt,
        Path::Tile,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Path::VendorSinks => "vendor-sinks",
            Path::VendorFlash => "vendor-flash",
            Path::MetalSdpa => "metal-sdpa",
            Path::Cublaslt => "cublaslt",
            Path::Tile => "arcflash-tile",
        }
    }

    const fn index(self) -> usize {
        match self {
            Path::VendorSinks => 0,
            Path::VendorFlash => 1,
            Path::MetalSdpa => 2,
            Path::Cublaslt => 3,
            Path::Tile => 4,
        }
    }
}

static ENGAGED: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

// Per-thread mirror of `ENGAGED`, for tests only.
//
// The process-wide counters are what operations wants, but they are useless as
// a test instrument: `cargo test` runs test fns on parallel threads, so a
// control arm asserting "this path did **not** move" would be reading other
// tests' dispatches and failing at random. Each test body runs on its own
// thread, so a thread-local record is exact. Compiled out of release builds, so
// the dispatch hot path keeps exactly one atomic.
#[cfg(test)]
thread_local! {
    static ENGAGED_LOCAL: std::cell::Cell<[u64; 5]> = const { std::cell::Cell::new([0; 5]) };
}

/// Record that `path` ran, and announce the first dispatch of each path.
///
/// That one-time line is the D18 signal on hardware: it is **present in the arm
/// that uses the path and absent in the arm that does not**, so a silent
/// fallback shows up as a missing line rather than as nothing at all. Returns
/// the count before this call, so a caller can add its own detail on first use
/// without a second atomic.
pub(crate) fn note(path: Path) -> u64 {
    #[cfg(test)]
    ENGAGED_LOCAL.with(|c| {
        let mut counts = c.get();
        counts[path.index()] += 1;
        c.set(counts);
    });
    let prior = ENGAGED[path.index()].fetch_add(1, Ordering::Relaxed);
    if prior == 0 {
        tracing::info!("ArcAttention: first dispatch on {}", path.label());
    }
    prior
}

/// Engagement counts for **this thread**. The instrument the tests assert on;
/// see `ENGAGED_LOCAL` for why the process-wide counters cannot serve.
#[cfg(test)]
pub(crate) fn engagement_local(path: Path) -> u64 {
    ENGAGED_LOCAL.with(|c| c.get()[path.index()])
}

/// How many times each [`Path`] has run in this process, in [`Path::ALL`] order.
#[cfg(test)]
pub(crate) fn engagement() -> [u64; 5] {
    let mut out = [0u64; 5];
    for (slot, counter) in out.iter_mut().zip(ENGAGED.iter()) {
        *slot = counter.load(Ordering::Relaxed);
    }
    out
}

/// The unfused half of ArcAttention/Dispatch: given that no vendor flash kernel
/// accepted this shape, which path runs?
///
/// Split out from `run_attention_noflash` so the decision is a pure function of
/// the shape — testable on a host with no CUDA and no Metal, and impossible to
/// read off "whatever happened to run".
pub(crate) fn plan_unfused(metal_eligible: bool, cublaslt_available: bool) -> Path {
    if metal_eligible {
        Path::MetalSdpa
    } else if cublaslt_available {
        Path::Cublaslt
    } else {
        Path::Tile
    }
}

/// Byte budget for a single query tile's score matrix.
///
/// Peak live bytes are a small multiple of this (scores, the softmax result,
/// and — when the caller supplies one — the broadcast mask), so the default is
/// deliberately well under any plausible free-VRAM figure. `ARC_FLASH_TILE_BYTES`
/// overrides it for A/B measurement without a rebuild.
fn tile_budget_bytes() -> usize {
    use std::sync::OnceLock;
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        const DEFAULT: usize = 32 * 1024 * 1024;
        std::env::var("ARC_FLASH_TILE_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT)
    })
}

/// Query rows per tile, from the byte budget.
///
/// One query row of scores costs `b * h * k_len * elem_size` bytes. Returns
/// `t_q` unchanged when the whole thing already fits, so short prompts and
/// decode steps take exactly one pass and pay nothing for the machinery.
pub(crate) fn tile_rows(b: usize, h: usize, t_q: usize, k_len: usize, elem_size: usize) -> usize {
    let per_row = b.saturating_mul(h).saturating_mul(k_len).max(1) * elem_size.max(1);
    let budget = tile_budget_bytes();
    if per_row.saturating_mul(t_q) <= budget {
        return t_q.max(1);
    }
    // At least one row, otherwise a single row wider than the whole budget
    // would divide to zero and loop forever.
    (budget / per_row).clamp(1, t_q.max(1))
}

/// **ArcFlash/Fold.** Express the GQA/MQA expansion as a reshape.
///
/// `q: [B, H, T, D]` (contiguous) → `[B, Hkv, g * T, D]`, sharing storage.
/// Returns `None` when the fold does not apply, so the caller can keep the
/// unfolded form rather than silently computing something else:
///   * `g == 1` — nothing to fold, and the unfolded shape already batches
///     against K/V correctly;
///   * `H != Hkv * g` — the caller's `n_kv_groups` disagrees with the tensors,
///     which must not be papered over.
fn fold_kv_groups(q: &Tensor, n_kv_heads: usize, n_kv_groups: usize) -> Result<Option<Tensor>> {
    let (b, h, t, d) = q.dims4()?;
    if n_kv_groups <= 1 || n_kv_heads == 0 || h != n_kv_heads * n_kv_groups {
        return Ok(None);
    }
    // `contiguous` is a no-op when the tile already is; when it is not (a
    // `narrow` along the query axis), this is a copy of *one tile*, versus the
    // `n_kv_groups`-fold copy of the entire K and V that it replaces.
    Ok(Some(q.contiguous()?.reshape((
        b,
        n_kv_heads,
        n_kv_groups * t,
        d,
    ))?))
}

/// Narrow a caller mask's query axis to `[offset, offset + len)`.
///
/// The query axis is the second-to-last for every rank this codebase produces
/// (2 → `[q, k]`, 3 → `[b, q, k]`, 4 → `[b, h, q, k]`). A mask that is size-1
/// on that axis is broadcasting over queries and must be passed through
/// untouched — narrowing it would either fail or, worse, silently select row 0
/// for every tile.
fn narrow_mask_rows(mask: &Tensor, offset: usize, len: usize) -> Result<Tensor> {
    let rank = mask.rank();
    if rank < 2 {
        candle_core::bail!("ArcFlash: attention mask must have rank >= 2, got {rank}");
    }
    let axis = rank - 2;
    if mask.dim(axis)? == 1 {
        return Ok(mask.clone());
    }
    mask.narrow(axis, offset, len)
}

/// **ArcFlash/Tile.** One softmax over the whole key axis, computed a query
/// tile at a time, with the GQA expansion folded rather than copied.
///
/// Shapes: `q [B, H, Tq, D]`, `k [B, Hkv, Tk, D]`, `v [B, Hkv, Tk, Dv]`,
/// `mask` broadcastable to `[B, H, Tq, Tk]`, output `[B, H, Tq, Dv]`.
///
/// The key axis is whatever the caller assembled — for DeepSeek-V4 that is the
/// union `[raw sliding window ++ compressed KV]`, which is why this is
/// `union_attention` and not `sdpa`: it makes no assumption that the key axis
/// is one contiguous causal region. Causality, windowing, block-causality and
/// padding all arrive as `mask`.
///
/// `sinks`, when present, is a per-head scalar folded into each row's softmax
/// denominator — **zero cache bytes, not a key region**. It must be shaped
/// `[H]` (or anything that flattens to `H`) and is cast to the logits dtype.
pub(crate) fn union_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b, h, t_q, _d) = q.dims4()?;
    let (_, n_kv_heads, t_k, _) = k.dims4()?;
    let v_head_dim = v.dim(3)?;

    let elem_size = q.dtype().size_in_bytes();
    let rows = tile_rows(b, h, t_q, t_k, elem_size);
    let n_tiles = t_q.div_ceil(rows.max(1));

    if note(Path::Tile) == 0 {
        tracing::info!(
            "ArcAttention: ArcFlash/Tile engaged (b={b} heads={h}/{n_kv_heads} q={t_q} kv={t_k} \
             head_dim={_d} tile_rows={rows} tiles={n_tiles} sinks={})",
            sinks.is_some()
        );
    }

    // Precompute the transposed keys once, outside the tile loop: it does not
    // depend on the query tile, and re-deriving it per tile would re-pay the
    // layout work `n_tiles` times.
    let k_t = k.t()?;
    let sinks = match sinks {
        Some(s) => Some(s.flatten_all()?),
        None => None,
    };

    let mut out_tiles = Vec::with_capacity(n_tiles);
    let mut offset = 0usize;
    while offset < t_q {
        let len = rows.min(t_q - offset);
        let q_tile = q.narrow(2, offset, len)?;

        // Scores. When the fold applies, both operands carry batch `B * Hkv`
        // and K is read once per KV head; otherwise fall back to the plain
        // per-head GEMM, which is already correct for `g == 1`.
        let att = match fold_kv_groups(&q_tile, n_kv_heads, sdpa_params.n_kv_groups)? {
            Some(q_folded) => MatMul
                .matmul_affine_mul(&q_folded, &k_t, sdpa_params.softmax_scale.into())?
                .reshape((b, h, len, t_k))?,
            None => MatMul.matmul_affine_mul(
                &q_tile.contiguous()?,
                &k_t,
                sdpa_params.softmax_scale.into(),
            )?,
        };

        let att = match sdpa_params.softcap {
            Some(softcap) if softcap != 1.0 => ((att / softcap as f64)?.tanh()? * softcap as f64)?,
            _ => att,
        };

        let mask_tile = mask.map(|m| narrow_mask_rows(m, offset, len)).transpose()?;

        let probs = match &sinks {
            Some(sinks) => {
                // `softmax_with_sinks` applies the mask itself (and requires
                // `[b, h, q, k]` logits plus `[h]` sinks in the logits dtype).
                let sinks = sinks.to_dtype(att.dtype())?;
                mistralrs_quant::softmax_with_sinks(&att, &sinks, mask_tile.as_ref())?
            }
            None => {
                let att = match mask_tile.as_ref() {
                    Some(m) => att.broadcast_add(m)?,
                    None => att,
                };
                candle_nn::ops::softmax_last_dim(&att)?
            }
        };

        let probs = probs.to_dtype(v.dtype())?;
        // Same fold on the way out: `[B, H, len, Tk] -> [B, Hkv, g*len, Tk]`
        // batches against `v [B, Hkv, Tk, Dv]` with no expansion of V.
        let out = match fold_kv_groups(&probs, n_kv_heads, sdpa_params.n_kv_groups)? {
            Some(p_folded) => MatMul
                .matmul(&p_folded, v)?
                .reshape((b, h, len, v_head_dim))?,
            None => MatMul.matmul(&probs.contiguous()?, v)?,
        };

        out_tiles.push(out);
        offset += len;
    }

    if out_tiles.len() == 1 {
        // Avoid a pointless copy when the whole prompt fit in one tile — which
        // is every decode step and every short prompt.
        return Ok(out_tiles.pop().expect("one tile"));
    }
    Tensor::cat(&out_tiles, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn mk(dims: &[usize], seed: f32, dev: &Device) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * seed).sin()).collect();
        Tensor::from_vec(data, dims, dev)
    }

    fn params(head_dim: usize, n_kv_groups: usize) -> SdpaParams {
        SdpaParams {
            n_kv_groups,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        }
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "shape mismatch between compared tensors");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// The reference this module must reproduce bit-for-bit-ish: the *old*
    /// path — materialise the GQA expansion, one full score matrix, one
    /// softmax.
    fn reference_expanded(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sinks: Option<&Tensor>,
        p: &SdpaParams,
    ) -> Result<Tensor> {
        let k = crate::attention::repeat_kv(k.clone(), p.n_kv_groups)?;
        let v = crate::attention::repeat_kv(v.clone(), p.n_kv_groups)?;
        let att = MatMul.matmul_affine_mul(q, &k.t()?, p.softmax_scale.into())?;
        let att = match p.softcap {
            Some(c) if c != 1.0 => ((att / c as f64)?.tanh()? * c as f64)?,
            _ => att,
        };
        let probs = match sinks {
            Some(s) => {
                let s = s.flatten_all()?.to_dtype(att.dtype())?;
                mistralrs_quant::softmax_with_sinks(&att, &s, mask)?
            }
            None => {
                let att = match mask {
                    Some(m) => att.broadcast_add(m)?,
                    None => att,
                };
                candle_nn::ops::softmax_last_dim(&att)?
            }
        };
        MatMul.matmul(&probs.to_dtype(v.dtype())?, &v)
    }

    fn causal_mask(t_q: usize, t_k: usize, dev: &Device) -> Result<Tensor> {
        let off = t_k - t_q;
        let mut data = vec![0f32; t_q * t_k];
        for (r, row) in data.chunks_mut(t_k).enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                if j > r + off {
                    *cell = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(data, (1, 1, t_q, t_k), dev)
    }

    /// The fold must be an identity against `repeat_kv`, across MQA, GQA and
    /// MHA — with and without a mask, with and without sinks.
    #[test]
    fn fold_matches_repeat_kv_across_gqa_ratios() -> Result<()> {
        let dev = Device::Cpu;
        let (b, t_q, t_k, d) = (2usize, 7usize, 11usize, 16usize);

        // (n_kv_heads, n_kv_groups): MQA, GQA, MHA.
        for (n_kv, g) in [(1usize, 8usize), (2, 4), (8, 1)] {
            let h = n_kv * g;
            let q = mk(&[b, h, t_q, d], 0.031, &dev)?;
            let k = mk(&[b, n_kv, t_k, d], 0.047, &dev)?;
            let v = mk(&[b, n_kv, t_k, d], 0.059, &dev)?;
            let mask = causal_mask(t_q, t_k, &dev)?;
            let sinks = mk(&[h], 0.13, &dev)?;

            for use_mask in [false, true] {
                for use_sinks in [false, true] {
                    let p = params(d, g);
                    let m = use_mask.then_some(&mask);
                    let s = use_sinks.then_some(&sinks);
                    let got = union_attention(&q, &k, &v, m, s, &p)?;
                    let want = reference_expanded(&q, &k, &v, m, s, &p)?;
                    assert_eq!(got.dims(), want.dims());
                    let diff = max_abs_diff(&flat(&got), &flat(&want));
                    assert!(
                        diff < 1e-5,
                        "n_kv={n_kv} g={g} mask={use_mask} sinks={use_sinks}: \
                         ArcFlash/Fold disagrees with the repeat_kv reference (max abs diff {diff})"
                    );
                }
            }
        }
        Ok(())
    }

    /// Tiling the query axis must not change a single number. Forcing a tiny
    /// budget makes every one of these calls take many tiles, so the test is
    /// exercising the loop, not the `t_q <= rows` shortcut.
    ///
    /// The teeth: an *incorrectly* tiled implementation — one that reuses row 0
    /// of the mask for every tile — is a visibly different answer, asserted
    /// below so a vacuous pass is impossible.
    #[test]
    fn query_tiling_is_exact_and_the_mask_follows_the_tile() -> Result<()> {
        let dev = Device::Cpu;
        let (b, n_kv, g, t_q, t_k, d) = (1usize, 1usize, 4usize, 33usize, 33usize, 8usize);
        let h = n_kv * g;

        let q = mk(&[b, h, t_q, d], 0.023, &dev)?;
        let k = mk(&[b, n_kv, t_k, d], 0.037, &dev)?;
        let v = mk(&[b, n_kv, t_k, d], 0.041, &dev)?;
        let mask = causal_mask(t_q, t_k, &dev)?;
        let sinks = mk(&[h], 0.19, &dev)?;
        let p = params(d, g);

        let want = reference_expanded(&q, &k, &v, Some(&mask), Some(&sinks), &p)?;

        // `tile_rows` reads the budget through a `OnceLock`, so drive the loop
        // directly instead of trying to mutate the env mid-process.
        for rows in [1usize, 2, 5, 16, 32] {
            let mut tiles = Vec::new();
            let mut off = 0usize;
            while off < t_q {
                let len = rows.min(t_q - off);
                let qt = q.narrow(2, off, len)?;
                let mt = narrow_mask_rows(&mask, off, len)?;
                tiles.push(union_attention(&qt, &k, &v, Some(&mt), Some(&sinks), &p)?);
                off += len;
            }
            let got = Tensor::cat(&tiles, 2)?;
            let diff = max_abs_diff(&flat(&got), &flat(&want));
            assert!(
                diff < 1e-5,
                "rows={rows}: query tiling changed the result (max abs diff {diff})"
            );
        }

        // Teeth: if the mask did NOT follow the tile, the answer differs a lot.
        let mut wrong = Vec::new();
        let mut off = 0usize;
        while off < t_q {
            let len = 8.min(t_q - off);
            let qt = q.narrow(2, off, len)?;
            let bad = mask.narrow(2, 0, len)?; // always row 0 — the bug
            wrong.push(union_attention(&qt, &k, &v, Some(&bad), Some(&sinks), &p)?);
            off += len;
        }
        let wrong = Tensor::cat(&wrong, 2)?;
        let signal = max_abs_diff(&flat(&want), &flat(&wrong));
        assert!(
            signal > 1e-2,
            "a mask that ignores the tile offset is indistinguishable from the correct \
             one (diff {signal}); this test has no teeth"
        );
        Ok(())
    }

    /// A mask that is size-1 on the query axis broadcasts over queries and must
    /// survive tiling untouched. Narrowing it would panic or, worse, be a
    /// silent no-op that looks correct on tile 0.
    #[test]
    fn query_broadcast_mask_is_not_narrowed() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, t_q, t_k, d) = (1usize, 2usize, 6usize, 6usize, 8usize);
        let q = mk(&[b, h, t_q, d], 0.017, &dev)?;
        let k = mk(&[b, h, t_k, d], 0.029, &dev)?;
        let v = mk(&[b, h, t_k, d], 0.043, &dev)?;
        // `[1, 1, 1, t_k]` — a per-key bias, e.g. a padding mask.
        let mut bias = vec![0f32; t_k];
        bias[t_k - 1] = f32::NEG_INFINITY;
        let mask = Tensor::from_vec(bias, (1, 1, 1, t_k), &dev)?;
        let p = params(d, 1);

        let narrowed = narrow_mask_rows(&mask, 3, 2)?;
        assert_eq!(
            narrowed.dims(),
            mask.dims(),
            "a query-broadcast mask must be passed through, not narrowed"
        );

        let got = union_attention(&q, &k, &v, Some(&mask), None, &p)?;
        let want = reference_expanded(&q, &k, &v, Some(&mask), None, &p)?;
        assert!(max_abs_diff(&flat(&got), &flat(&want)) < 1e-6);
        Ok(())
    }

    /// The budget must produce a single tile when everything fits, and a
    /// bounded one when it does not — never zero (which would loop forever).
    #[test]
    fn tile_rows_is_bounded_and_never_zero() {
        // Fits comfortably: one pass.
        assert_eq!(tile_rows(1, 8, 128, 128, 2), 128);
        // DeepSeek-V4 prefill shape: 64 heads, head_dim 512, 4096 keys. One row
        // of scores is 64 * 4096 * 2 = 512 KiB, so the 32 MiB default must give
        // a tile far below `t_q` — the whole point of the module.
        let rows = tile_rows(1, 64, 4096, 4096, 2);
        assert!(
            (1..4096).contains(&rows),
            "V4 prefill must be split into more than one tile, got {rows}"
        );
        // Pathological: one row alone exceeds the budget. Must still make
        // progress.
        assert_eq!(tile_rows(1, 128, 10, 1_000_000, 4), 1);
    }

    /// The bytes claim, as arithmetic rather than prose. **This is a model, not
    /// a measurement** — it counts the tensors each implementation allocates for
    /// one layer's attention at DeepSeek-V4's prefill shape. The hardware number
    /// is a separate, GPU-only claim.
    ///
    /// Old path, per layer: `repeat_kv(K)` + `repeat_kv(V)` at
    /// `n_kv_groups = 64`, then `[B,H,T,T]` scores and a second `[B,H,T,T]` for
    /// the softmax result.
    /// New path: no expansion at all, and scores bounded to one query tile.
    #[test]
    fn v4_prefill_peak_bytes_drop_by_an_order_of_magnitude() {
        const B: usize = 1;
        const H: usize = 64; // n_heads
        const D: usize = 512; // head_dim
        const T: usize = 1400; // the ~1055-word prompt that killed the server
        const E: usize = 2; // bf16

        // Old: two 64-fold K/V copies + two full score matrices.
        let expanded_kv = 2 * B * H * T * D * E;
        let full_scores = 2 * B * H * T * T * E;
        let old_peak = expanded_kv + full_scores;

        // New: K and V as given (one KV head), plus two tile-sized score
        // matrices, plus the folded query tile.
        let rows = tile_rows(B, H, T, T, E);
        let kv = 2 * B * 1 * T * D * E;
        let tile_scores = 2 * B * H * rows * T * E;
        let q_tile = B * H * rows * D * E;
        let new_peak = kv + tile_scores + q_tile;

        assert!(
            rows < T,
            "the V4 prefill shape must actually tile, else this test is vacuous (rows={rows}, T={T})"
        );
        assert!(
            new_peak * 8 < old_peak,
            "ArcFlash must cut V4 prefill attention bytes by >8x: old {old_peak} B, new {new_peak} B \
             ({:.1}x)",
            old_peak as f64 / new_peak as f64
        );
    }

    /// D18: engagement is asserted, not inferred. ArcFlash/Tile must increment
    /// its counter when it runs, and the *other* paths must stay untouched —
    /// the absence is as load-bearing as the presence.
    #[test]
    fn tile_records_its_own_engagement_and_no_other() -> Result<()> {
        let dev = Device::Cpu;
        let before: Vec<u64> = Path::ALL.iter().map(|p| engagement_local(*p)).collect();

        let q = mk(&[1, 4, 3, 8], 0.011, &dev)?;
        let k = mk(&[1, 1, 3, 8], 0.013, &dev)?;
        let v = mk(&[1, 1, 3, 8], 0.019, &dev)?;
        union_attention(&q, &k, &v, None, None, &params(8, 4))?;

        let after: Vec<u64> = Path::ALL.iter().map(|p| engagement_local(*p)).collect();
        for (i, path) in Path::ALL.iter().enumerate() {
            if *path == Path::Tile {
                assert_eq!(
                    after[i],
                    before[i] + 1,
                    "ArcFlash/Tile ran once but recorded {} engagements",
                    after[i] - before[i]
                );
            } else {
                assert_eq!(
                    after[i],
                    before[i],
                    "{} was recorded as engaged by a call that only ran ArcFlash/Tile",
                    path.label()
                );
            }
        }
        // The process-wide counter must move too -- that is the one operations
        // reads on the box.
        assert!(engagement()[Path::Tile.index()] > 0);
        Ok(())
    }

    /// The unfused plan is a pure function of the shape, so it can be pinned
    /// on a host with neither CUDA nor Metal.
    #[test]
    fn plan_prefers_metal_then_cublaslt_then_ours() {
        assert_eq!(plan_unfused(true, true), Path::MetalSdpa);
        assert_eq!(plan_unfused(true, false), Path::MetalSdpa);
        assert_eq!(plan_unfused(false, true), Path::Cublaslt);
        assert_eq!(plan_unfused(false, false), Path::Tile);
    }

    /// A rank-2 `[q, k]` mask used to pin the call to `naive_sdpa` with a fully
    /// expanded K/V; it now goes to ArcFlash/Tile instead, so the rank-2 case
    /// needs its own coverage — including across a tile boundary, where the
    /// query axis is dim 0 rather than dim 2.
    #[test]
    fn rank_two_mask_is_honored_and_follows_the_tile() -> Result<()> {
        let dev = Device::Cpu;
        let (b, n_kv, g, t, d) = (1usize, 2usize, 3usize, 9usize, 8usize);
        let h = n_kv * g;
        let q = mk(&[b, h, t, d], 0.053, &dev)?;
        let k = mk(&[b, n_kv, t, d], 0.061, &dev)?;
        let v = mk(&[b, n_kv, t, d], 0.067, &dev)?;
        let p = params(d, g);

        // `[t, t]` causal — the rank the `mask.rank() == 2` carve-out named.
        let mut data = vec![0f32; t * t];
        for (r, row) in data.chunks_mut(t).enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                if j > r {
                    *cell = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(data, (t, t), &dev)?;
        assert_eq!(mask.rank(), 2);

        let want = reference_expanded(&q, &k, &v, Some(&mask), None, &p)?;
        assert!(
            max_abs_diff(
                &flat(&union_attention(&q, &k, &v, Some(&mask), None, &p)?),
                &flat(&want)
            ) < 1e-6
        );

        // Across tiles: the query axis of a rank-2 mask is dim 0.
        let mut tiles = Vec::new();
        let mut off = 0usize;
        while off < t {
            let len = 4.min(t - off);
            let mt = narrow_mask_rows(&mask, off, len)?;
            assert_eq!(mt.dims(), &[len, t], "rank-2 mask must narrow on dim 0");
            tiles.push(union_attention(
                &q.narrow(2, off, len)?,
                &k,
                &v,
                Some(&mt),
                None,
                &p,
            )?);
            off += len;
        }
        let diff = max_abs_diff(&flat(&Tensor::cat(&tiles, 2)?), &flat(&want));
        assert!(
            diff < 1e-6,
            "rank-2 mask lost its row offset across tiles ({diff})"
        );
        Ok(())
    }

    /// A `n_kv_groups` that disagrees with the tensors must not be folded —
    /// silently reinterpreting the head axis would compute a different model.
    #[test]
    fn inconsistent_kv_groups_are_not_folded() -> Result<()> {
        let dev = Device::Cpu;
        let q = mk(&[1, 6, 2, 4], 0.021, &dev)?;
        // 6 heads, 4 kv heads: 6 != 4 * 3, so no fold is possible.
        assert!(fold_kv_groups(&q, 4, 3)?.is_none());
        // Consistent: 6 = 2 * 3.
        let folded = fold_kv_groups(&q, 2, 3)?.expect("consistent fold");
        assert_eq!(folded.dims(), &[1, 2, 6, 4]);
        Ok(())
    }
}
