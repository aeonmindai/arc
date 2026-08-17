//! TurboQuant storage for DeepSeek-V4's fused MQA key cache.
//!
//! # Why V4 can host TurboQuant for free, and no other model can (yet)
//!
//! [`crate::models::dsv4_attention::dsv4_attention`] reads the raw key cache
//! over exactly one span: the trailing `t_q + window - 1` tokens
//! ([`crate::models::dsv4_attention::raw_keep_span`]). Every earlier raw column
//! is `-inf` on **every** query row, because V4's raw branch is a *sliding
//! window*, not dense causal attention. Distant context does not reach the
//! model through the raw cache at all — it arrives through `compressed_kv`,
//! which is built by the compressor from its own rolling history
//! ([`super::XsRollingCache`]) and never touches these bytes.
//!
//! So V4's raw K cache is already two regions with a hard boundary between
//! them, and TurboQuant is already two regions with a hard boundary between
//! them:
//!
//! | TurboQuant | V4 |
//! |---|---|
//! | `fp16_window` — recent tokens kept uncompressed | the raw sliding window |
//! | the packed/quantized region | everything older, which only the compressor reads |
//!
//! They are the *same* boundary, so this type builds them as one mechanism.
//! The consequence is the thing that makes this worth shipping: **nothing on
//! the decode path is ever dequantized.** [`V4TurboKCache::span`] over the
//! reachable window is a `narrow` of a tensor that was never compressed, and
//! the compressed region is only ever *written*.
//!
//! That is the opposite of [`arc_turbo::TurboQuantSingleCache`], whose
//! `current_data` reconstructs every compressed token on the host on every
//! call. This type has a dequantizing path too — `span` must stay total, and a
//! rollback can legitimately ask for one — but the model never takes it, and
//! `decode_never_dequantizes` pins that.
//!
//! # Cost, per token per layer, at V4-Flash's geometry
//!
//! `head_dim = 512`, `num_key_value_heads = 1`, one rotation block, BF16
//! activations, TurboQuant K at 4 bits:
//!
//! | layout | bytes |
//! |---|---|
//! | dense BF16 (512 BF16 + a 1-wide marker) | 1,026 |
//! | FP8 codes (`ARC_V4_FP8_KV=1`) | 590 |
//! | **this** (256 packed + one f32 norm) | **260** |
//!
//! against the raw window, which stays dense and is bounded by
//! `window + max(t_q - 1, margin) + EVICT_CHUNK` tokens — independent of
//! context length.
//!
//! # Why eviction is chunked
//!
//! [`mistralrs_quant::turboquant::TurboQuantLayout::quantize_into`] is a host
//! function over `&[f32]`. Compressing on every decode step would mean one
//! device→host→device round trip per layer per token — ~86 syncs a token on
//! V4's 43 layers, which is the CLAUDE.md pitfall #5 failure and would cost
//! exactly the decode speed this is supposed to be free of. Eviction therefore
//! runs only once the unreachable region has grown past
//! [`V4TurboKCache::EVICT_CHUNK`], amortising it to one round trip per layer
//! per `EVICT_CHUNK` tokens. The price is `EVICT_CHUNK` extra dense tokens of
//! retained tail.
//!
//! # The contract with the rest of the engine
//!
//! Same as [`super::XsRollingCache`]'s, and for the same reason: cache entries
//! are truncated in **token** units by the prefix cacher, the MTP verify
//! rollback and speculative rejection. This type reports `current_seq_len()`
//! in tokens, maps a token truncation onto its own two time bases, and
//! **refuses** any truncation that lands before a token whose raw row it has
//! already dropped *and* whose compressed row cannot serve — which is never,
//! because the compressed region is exact-length and addressable. What it
//! refuses instead is extension.
//!
//! Its two buffers batch like [`super::XsRollingCache`]'s: `codes` is a grown
//! capacity buffer (slack-tolerant, batches like a K half) and `tail` is live
//! content (batches like a V half). That is why the norms are carried *inside*
//! the code record rather than in a third buffer — [`super::BatchSrc`] gives a
//! cache slot exactly two tensors, and a third would not survive
//! clone-in/clone-out.

use std::sync::Once;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::turboquant::TurboQuantLayout;

use super::{NormalCache, SingleCache};

/// Extra raw tokens retained behind the sliding window so a short rollback
/// stays inside the dense region. Matches [`super::XS_TAIL_MARGIN_TOKENS`]'s
/// reasoning: MTP/speculative rejection drops at most the number of drafted
/// positions, and those paths cannot decline a rollback.
pub const V4_TURBO_TAIL_MARGIN_TOKENS: usize = 16;

/// The first token that must stay dense after a step of `t_new` tokens brought
/// the cache to `tokens_new`.
///
/// Two constraints, and the looser one wins:
///
/// * **This step's reach.** `dsv4_attention` reads the trailing
///   `t_new + window - 1` raw keys
///   ([`crate::models::dsv4_attention::raw_keep_span`]), so the floor cannot be
///   above `tokens_new - (t_new + window - 1)`.
/// * **A rollback's reach.** MTP verify and speculative rejection can truncate
///   by up to `margin` tokens and then decode again, and that decode reaches
///   back `window - 1` from the *truncated* end. Surviving a rollback of depth
///   `d` needs `margin >= d - 1`, so the floor cannot be above
///   `tokens_new - (window + margin)`.
///
/// This is a free function because it is the only arithmetic in this file that
/// can be wrong without any tensor being involved, and because
/// [`V4TurboKCache::EVICT_CHUNK`] is far larger than `margin` — chunked
/// eviction holds `base` so far below this floor that no end-to-end fixture can
/// make the margin term binding. Tested directly instead, over the regime where
/// it does bind.
pub fn retention_floor(tokens_new: usize, t_new: usize, window: usize, margin: usize) -> usize {
    let reach = window + (t_new.saturating_sub(1)).max(margin);
    tokens_new.saturating_sub(reach)
}

/// One layer's V4 keys, split at the sliding-window boundary: TurboQuant codes
/// for the tokens no query can reach, dense activations for the ones they can.
/// See the module docs.
#[derive(Debug, Clone)]
pub struct V4TurboKCache {
    /// Packed records for tokens `[0, base)`, `[B, H, base, record_bytes]` U8
    /// on seq dim 2. One record is `layout.packed_bytes` packed indices
    /// followed by `layout.num_norms()` little-endian `f32` block norms — see
    /// [`Self::RECORD_NORM_BYTES`].
    pub codes: SingleCache,
    /// Dense keys for tokens `[base, tokens)`, `[B, H, tokens - base, head_dim]`.
    /// `None` before the first token.
    pub tail: Option<Tensor>,
    /// Index of the first token still held densely in `tail`.
    pub base: usize,
    /// Total tokens seen — the length every other part of the engine means.
    pub tokens: usize,
    /// Logical key width (V4-Flash: 512).
    pub head_dim: usize,
    /// KV heads (V4 is MQA, so 1). Learned on first append; 0 until then.
    pub num_heads: usize,
    /// Batch rows. Learned on first append; 0 until then.
    pub batch_size: usize,
    /// V4's raw sliding window (`config.sliding_window`, 128).
    pub window: usize,
    /// Extra retained tokens for rollback ([`V4_TURBO_TAIL_MARGIN_TOKENS`]).
    pub margin: usize,
    /// Packed layout: block plan, signs, codebooks, byte offsets.
    pub layout: TurboQuantLayout,
}

impl V4TurboKCache {
    /// Bytes one `f32` block norm occupies inside a packed record.
    pub const RECORD_NORM_BYTES: usize = std::mem::size_of::<f32>();

    /// Tokens the unreachable region must accumulate before it is compressed.
    ///
    /// This is the amortisation knob described in the module docs: eviction
    /// costs one device→host→device round trip per layer, so it must not run
    /// per decode step. 128 makes it ~0.34 syncs per token on V4's 43 layers,
    /// at a cost of 128 extra dense tokens of tail (128 KB/layer/seq at
    /// `head_dim = 512`, BF16).
    pub const EVICT_CHUNK: usize = 128;

    /// Build a cache for one layer, or explain why the geometry is unsupported.
    ///
    /// `bits` is the TurboQuant key bit-width (see
    /// [`mistralrs_quant::turboquant::TurboQuantPreset::key_bits`]).
    pub fn try_new(
        head_dim: usize,
        bits: u32,
        seed: u64,
        window: usize,
        max_position_embeddings: usize,
    ) -> std::result::Result<Self, String> {
        let layout = TurboQuantLayout::new(head_dim, bits, seed)?;
        // `SingleCache::append` grows in `CACHE_GROW_SIZE` blocks and bails if
        // that overshoots `max_seq_len`, so round the ceiling up to a whole
        // number of blocks. The compressed region is 1:1 in tokens, so its
        // ceiling is the position ceiling itself.
        let grow = NormalCache::CACHE_GROW_SIZE;
        let max_rows = max_position_embeddings.div_ceil(grow) * grow;
        let init_rows = Self::EVICT_CHUNK.min(max_rows);
        Ok(Self {
            codes: SingleCache::new(2, max_rows, init_rows),
            tail: None,
            base: 0,
            tokens: 0,
            head_dim,
            num_heads: 0,
            batch_size: 0,
            window: window.max(1),
            margin: V4_TURBO_TAIL_MARGIN_TOKENS,
            layout,
        })
    }

    /// Bytes per compressed token per head: packed indices plus the block norms.
    pub fn record_bytes(&self) -> usize {
        self.layout.packed_bytes + Self::RECORD_NORM_BYTES * self.layout.num_norms()
    }

    /// Tokens seen — what `set_len`, the schedulers and the prefix cacher mean
    /// by "length".
    pub fn current_seq_len(&self) -> usize {
        self.tokens
    }

    /// Tokens currently held in TurboQuant form.
    ///
    /// Tests must assert this is non-zero, otherwise they are exercising a
    /// dense buffer and proving nothing about the codec.
    pub fn compressed_len(&self) -> usize {
        self.base
    }

    /// Bytes the compressed region occupies, per batch row per head.
    pub fn compressed_bytes(&self) -> usize {
        self.base * self.record_bytes()
    }

    pub fn reset(&mut self) {
        self.codes.reset();
        self.tail = None;
        self.base = 0;
        self.tokens = 0;
        self.num_heads = 0;
        self.batch_size = 0;
    }

    pub fn try_set_len(&self, len: usize) -> Result<()> {
        if len > self.tokens {
            candle_core::bail!(
                "v4 turboquant kv cache: cannot extend from {} to {len} tokens",
                self.tokens
            );
        }
        Ok(())
    }

    /// Truncate to `len` tokens, mapping the token length onto both time bases.
    ///
    /// Unlike [`super::XsRollingCache`], a truncation below `base` is *not*
    /// unrecoverable here: the compressed region is one record per token and
    /// stays addressable, so `set_len` simply moves the boundary back and lets
    /// [`Self::span`] reconstruct. It is lossy in precision, never in history.
    pub fn set_len(&mut self, len: usize) -> Result<()> {
        self.try_set_len(len)?;
        if len == self.tokens {
            return Ok(());
        }
        if len <= self.base {
            // The whole dense tail is gone; the boundary moves back onto the
            // compressed region, which keeps a record for every one of these
            // tokens.
            self.codes.set_len(len)?;
            self.base = len;
            self.tail = None;
        } else {
            let keep = len - self.base;
            self.tail = match self.tail.take() {
                Some(t) => Some(t.narrow(2, 0, keep)?.contiguous()?),
                None => candle_core::bail!(
                    "v4 turboquant kv cache: truncation to {len} needs the dense tail for tokens \
                     [{}, {len}) but it is not materialised; the cache holds {} tokens",
                    self.base,
                    self.tokens
                ),
            };
        }
        self.tokens = len;
        Ok(())
    }

    /// First token whose key is still held densely.
    pub fn dense_from(&self) -> usize {
        self.base
    }

    /// Append `k` (`[B, H, T_new, head_dim]`), compress whatever the sliding
    /// window has moved past, and return `(dense_base, dense_keys)`.
    ///
    /// `dense_keys` spans absolute tokens `[dense_base, tokens)` and is the
    /// tensor the caller hands to `dsv4_attention` as its `raw_prefix` span.
    /// It is always a superset of `raw_keep_span`'s reachable set — the
    /// retention floor below is that span's base, relaxed by the rollback
    /// margin and the eviction chunk — so `dsv4_attention`'s own
    /// `rel_base` check can only ever *pass*. If a future change made this
    /// evict too much, that check turns it into a hard error rather than a
    /// silently wrong window mask.
    pub fn append(&mut self, k: &Tensor) -> Result<(usize, Tensor)> {
        let k = k.contiguous()?;
        let (b, h, t_new, d) = k.dims4()?;
        if d != self.head_dim {
            candle_core::bail!(
                "v4 turboquant kv cache: head_dim mismatch — built for {}, got {d} (shape {:?})",
                self.head_dim,
                k.dims()
            );
        }
        if self.num_heads == 0 {
            self.batch_size = b;
            self.num_heads = h;
        } else if b != self.batch_size || h != self.num_heads {
            candle_core::bail!(
                "v4 turboquant kv cache: geometry changed — holds [B={}, H={}], got {:?}",
                self.batch_size,
                self.num_heads,
                k.dims()
            );
        }
        if t_new == 0 {
            let dense = self.dense_tail(&k)?;
            return Ok((self.base, dense));
        }

        // Grow the dense tail.
        let tail = match self.tail.take() {
            Some(t) if self.tokens > self.base => Tensor::cat(&[&t, &k], 2)?.contiguous()?,
            _ => k.clone(),
        };
        let tokens_new = self.tokens + t_new;

        let floor = retention_floor(tokens_new, t_new, self.window, self.margin);

        // Chunked eviction — see the module docs on why this is not per-step.
        let tail = if floor >= self.base + Self::EVICT_CHUNK {
            let n_evict = floor - self.base;
            let evicted = tail.narrow(2, 0, n_evict)?.contiguous()?;
            self.compress_into_codes(&evicted)?;
            self.base = floor;
            tail.narrow(2, n_evict, tokens_new - floor)?.contiguous()?
        } else {
            tail
        };

        // Materialise the code buffer even when nothing has been evicted yet:
        // `NormalCacheManager` batches `all_data` directly and unwraps it, so a
        // `None` here would panic the first time a short prompt is cloned out.
        // Same shape `SingleCache::append` would have allocated.
        if self.codes.all_data.is_none() {
            self.codes.all_data = Some(Tensor::zeros(
                (b, h, self.codes.capacity_seq_len, self.record_bytes()),
                DType::U8,
                k.device(),
            )?);
        }

        self.tokens = tokens_new;
        self.tail = Some(tail.clone());
        Ok((self.base, tail))
    }

    /// The dense tail as a tensor, synthesising an empty one when no token has
    /// arrived yet (so callers never have to special-case `None`).
    fn dense_tail(&self, like: &Tensor) -> Result<Tensor> {
        match self.tail.as_ref() {
            Some(t) => Ok(t.clone()),
            None => Tensor::zeros(
                (
                    self.batch_size.max(like.dim(0)?),
                    self.num_heads.max(like.dim(1)?),
                    0,
                    self.head_dim,
                ),
                like.dtype(),
                like.device(),
            ),
        }
    }

    /// Quantize `evicted` (`[B, H, T, head_dim]`) and append the records.
    ///
    /// One device→host→device round trip, amortised over
    /// [`Self::EVICT_CHUNK`] tokens.
    fn compress_into_codes(&mut self, evicted: &Tensor) -> Result<()> {
        let (b, h, t, _) = evicted.dims4()?;
        // The flag being ON and the codec never RUNNING look identical from
        // outside the process — that is the D18 shape, and it is exactly what
        // an on-box gate has to be able to tell apart. One line, once per
        // process, on the first eviction.
        static FIRST: Once = Once::new();
        FIRST.call_once(|| {
            tracing::info!(
                "V4 TurboQuant KV engaged: compressing {t} token(s) at head_dim {} into {} B/token                  (K{}-bit, window {}), from a dense {} B/token",
                self.head_dim,
                self.record_bytes(),
                self.layout.bits,
                self.window,
                self.head_dim * 2 + 2,
            );
        });
        let host = evicted
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let rec = self.record_bytes();
        let packed_bytes = self.layout.packed_bytes;
        let n_norms = self.layout.num_norms();
        let vectors = b * h * t;

        let mut out = vec![0u8; vectors * rec];
        let mut norms = vec![0f32; n_norms];
        for v in 0..vectors {
            let src = &host[v * self.head_dim..(v + 1) * self.head_dim];
            let dst = &mut out[v * rec..(v + 1) * rec];
            let (packed, norm_bytes) = dst.split_at_mut(packed_bytes);
            self.layout.quantize_into(src, packed, &mut norms);
            for (j, n) in norms.iter().enumerate() {
                let o = j * Self::RECORD_NORM_BYTES;
                norm_bytes[o..o + Self::RECORD_NORM_BYTES].copy_from_slice(&n.to_le_bytes());
            }
        }

        let records = Tensor::from_vec(out, (b, h, t, rec), evicted.device())?;
        self.codes.append(&records)
    }

    /// Materialise absolute tokens `[base, base + len)` at `out_dtype`.
    ///
    /// The decode path always asks for a range inside the dense tail, which
    /// makes this a `narrow` — see the module docs. A range reaching into the
    /// compressed region is reconstructed on the host; that is correct but
    /// slow, and it is why `decode_never_dequantizes` exists.
    pub fn span(&self, base: usize, len: usize, out_dtype: DType) -> Result<Tensor> {
        if base + len > self.tokens {
            candle_core::bail!(
                "v4 turboquant kv cache: span [{base}, {}) exceeds the {} cached tokens",
                base + len,
                self.tokens
            );
        }
        if len == 0 {
            let device = self
                .tail
                .as_ref()
                .map(|t| t.device().clone())
                .unwrap_or(Device::Cpu);
            return Tensor::zeros(
                (self.batch_size, self.num_heads, 0, self.head_dim),
                out_dtype,
                &device,
            );
        }

        // Wholly inside the dense tail — the decode path, and the only one that
        // runs per step.
        if base >= self.base {
            let t = self.tail.as_ref().ok_or_else(|| {
                candle_core::Error::msg(
                    "v4 turboquant kv cache: dense tail not materialised but the span asks for it",
                )
            })?;
            let off = base - self.base;
            let dense = if off == 0 && len == t.dim(2)? {
                t.clone()
            } else {
                t.narrow(2, off, len)?.contiguous()?
            };
            return dense.to_dtype(out_dtype);
        }

        // Reaches into the compressed region: reconstruct that part, then
        // concatenate whatever dense tail the range also covers.
        let comp_len = len.min(self.base - base);
        let reconstructed = self.dequant_range(base, comp_len, out_dtype)?;
        if comp_len == len {
            return Ok(reconstructed);
        }
        let dense = self.span(self.base, len - comp_len, out_dtype)?;
        Tensor::cat(&[&reconstructed, &dense], 2)?.contiguous()
    }

    /// Reconstruct `[base, base + len)` from the compressed region.
    fn dequant_range(&self, base: usize, len: usize, out_dtype: DType) -> Result<Tensor> {
        let all = self.codes.all_data.as_ref().ok_or_else(|| {
            candle_core::Error::msg("v4 turboquant kv cache: code buffer not materialised")
        })?;
        let records = all.narrow(2, base, len)?.contiguous()?;
        let (b, h, t, rec) = records.dims4()?;
        let bytes = records.flatten_all()?.to_vec1::<u8>()?;

        let packed_bytes = self.layout.packed_bytes;
        let n_norms = self.layout.num_norms();
        let vectors = b * h * t;

        let mut out = vec![0f32; vectors * self.head_dim];
        let mut norms = vec![0f32; n_norms];
        for v in 0..vectors {
            let src = &bytes[v * rec..(v + 1) * rec];
            let (packed, norm_bytes) = src.split_at(packed_bytes);
            for (j, n) in norms.iter_mut().enumerate() {
                let o = j * Self::RECORD_NORM_BYTES;
                let mut raw = [0u8; Self::RECORD_NORM_BYTES];
                raw.copy_from_slice(&norm_bytes[o..o + Self::RECORD_NORM_BYTES]);
                *n = f32::from_le_bytes(raw);
            }
            self.layout.dequantize_into(
                packed,
                &norms,
                &mut out[v * self.head_dim..(v + 1) * self.head_dim],
            );
        }

        Tensor::from_vec(out, (b, h, t, self.head_dim), records.device())?.to_dtype(out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 512;
    const WINDOW: usize = 128;

    fn cache() -> V4TurboKCache {
        V4TurboKCache::try_new(HEAD_DIM, 4, 42, WINDOW, 8192).unwrap()
    }

    /// Deterministic, outlier-bearing activations. A constant or symmetric
    /// fixture cannot disagree with a broken rotation, which is the failure
    /// mode these tests exist to catch.
    fn activations(b: usize, h: usize, t: usize, d: usize, seed: u64) -> Tensor {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let n = b * h * t * d;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) as f32) / (1u32 << 24) as f32 - 0.5;
                if i % 53 == 0 {
                    u * 9.0
                } else {
                    u
                }
            })
            .collect();
        Tensor::from_vec(data, &[b, h, t, d], &Device::Cpu).unwrap()
    }

    fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        let num: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum::<f32>().max(1e-20);
        (num / den).sqrt()
    }

    /// 🔑 The retention arithmetic, checked directly against both constraints
    /// it has to satisfy, across the regime where each one binds.
    ///
    /// This test exists because chunked eviction makes the margin term
    /// unreachable end to end: `EVICT_CHUNK` is 128 and the margin is 16, so
    /// `base` never gets close enough to the floor for the margin to matter,
    /// and three separate mutations of it passed every integration fixture.
    /// The arithmetic is defence in depth for a smaller chunk or an eager
    /// eviction, so it is tested where it can actually fail.
    #[test]
    fn retention_floor_survives_this_step_and_a_rollback() {
        for window in [1usize, 8, 128] {
            for margin in [0usize, 4, V4_TURBO_TAIL_MARGIN_TOKENS] {
                for t_new in [1usize, 2, 5, 64] {
                    for tokens_new in [t_new, t_new + 1, 200, 4096] {
                        if tokens_new < t_new {
                            continue;
                        }
                        let floor = retention_floor(tokens_new, t_new, window, margin);

                        // 1. This step's own reachable span must survive.
                        let keep = (t_new + window - 1).min(tokens_new);
                        let reach_base = tokens_new - keep;
                        assert!(
                            floor <= reach_base,
                            "w={window} m={margin} t={t_new} n={tokens_new}: floor {floor} \
                             evicts into this step's reachable span (base {reach_base})"
                        );

                        // 2. A rollback of up to `margin` tokens, then one more
                        //    decode step, must still be inside the dense region.
                        for d in 0..=margin.min(tokens_new) {
                            let rolled = tokens_new - d;
                            let after = rolled + 1;
                            let keep_after = (1 + window - 1).min(after);
                            let reach_after = after - keep_after;
                            assert!(
                                floor <= reach_after,
                                "w={window} m={margin} t={t_new} n={tokens_new}: after a \
                                 {d}-token rollback the next step reaches back to \
                                 {reach_after}, but {floor} is already compressed"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Prefill must not compress anything: at `t_q == t_k_full` every raw key
    /// is reachable by some query row, so evicting any of it would corrupt the
    /// prefill softmax. This is the invariant that decides the whole design.
    #[test]
    fn prefill_keeps_every_key_dense() {
        let mut c = cache();
        let prefill = activations(1, 1, 2048, HEAD_DIM, 7);
        let (base, dense) = c.append(&prefill).unwrap();
        assert_eq!(
            base, 0,
            "prefill evicted {base} tokens; none are unreachable"
        );
        assert_eq!(c.compressed_len(), 0);
        assert_eq!(dense.dims(), &[1, 1, 2048, HEAD_DIM]);
        // Bit-exact: nothing went through the codec.
        assert_eq!(rel_err(&prefill, &dense), 0.0);
    }

    /// 🔑 The claim the whole design rests on: over a long decode run the model
    /// never asks `span` for anything outside the dense tail, so no decode step
    /// ever pays a dequant. Asserted against the *real* reachable span
    /// (`raw_keep_span`'s formula), not against a hand-picked range.
    #[test]
    fn decode_never_dequantizes() {
        let mut c = cache();
        c.append(&activations(1, 1, 1024, HEAD_DIM, 11)).unwrap();
        for step in 0..600 {
            let tok = activations(1, 1, 1, HEAD_DIM, 3000 + step);
            let (base, _) = c.append(&tok).unwrap();
            let t_k_full = c.current_seq_len();
            // What `dsv4_attention` will actually read (t_q == 1).
            let keep = (1 + WINDOW - 1).min(t_k_full);
            let reach_base = t_k_full - keep;
            assert!(
                base <= reach_base,
                "step {step}: cache dropped raw keys to {base} but the step reaches back to \
                 {reach_base} — dsv4_attention's rel_base check would reject this"
            );
            // The span the model asks for is inside the dense region, so it is
            // a narrow, and it is bit-exact.
            let got = c.span(base, t_k_full - base, DType::F32).unwrap();
            let want_last = c.span(t_k_full - 1, 1, DType::F32).unwrap();
            assert_eq!(rel_err(&tok, &want_last), 0.0, "step {step}: newest token");
            assert_eq!(got.dim(2).unwrap(), t_k_full - base);
        }
        assert!(
            c.compressed_len() > 1024,
            "only {} tokens compressed after 600 decode steps — the codec never ran, so this \
             test proves nothing",
            c.compressed_len()
        );
    }

    /// The compressed region must actually be smaller, measured from the
    /// buffers the cache holds, against the dense layout it replaces.
    #[test]
    fn compressed_region_beats_dense_bf16() {
        let mut c = cache();
        c.append(&activations(1, 1, 4096, HEAD_DIM, 5)).unwrap();
        for step in 0..400 {
            c.append(&activations(1, 1, 1, HEAD_DIM, 8000 + step))
                .unwrap();
        }
        let n = c.compressed_len();
        assert!(n > 0, "nothing compressed");
        // V4 dense stores head_dim BF16 plus a 1-wide BF16 marker per token.
        let dense = n * (HEAD_DIM + 1) * 2;
        let got = c.compressed_bytes();
        let ratio = dense as f32 / got as f32;
        assert!(
            ratio > 3.5,
            "only {ratio:.2}x ({got} vs {dense}) — expected ~3.9x at 4-bit/512",
        );
        assert_eq!(
            c.record_bytes(),
            256 + 4,
            "512 dims at 4 bits + one f32 norm"
        );
    }

    /// MUTATION GUARD — a span reaching into the compressed region must
    /// reconstruct, not error and not silently return the dense tail. This is
    /// the path a deep rollback takes.
    #[test]
    fn span_into_the_compressed_region_reconstructs() {
        let mut c = cache();
        let prefill = activations(1, 1, 2048, HEAD_DIM, 13);
        c.append(&prefill).unwrap();
        for step in 0..400 {
            c.append(&activations(1, 1, 1, HEAD_DIM, 9000 + step))
                .unwrap();
        }
        let n = c.compressed_len();
        assert!(
            n > 300,
            "only {n} compressed; the span below would be dense"
        );

        // A range wholly inside the compressed region.
        let got = c.span(100, 200, DType::F32).unwrap();
        assert_eq!(got.dims(), &[1, 1, 200, HEAD_DIM]);
        let want = prefill.narrow(2, 100, 200).unwrap();
        let e = rel_err(&want, &got);
        assert!(e < 0.30, "compressed rel err {e}");
        assert!(e > 1e-4, "rel err {e} — nothing was quantized");

        // A range straddling the boundary must stitch both halves, and the
        // dense half must come back bit-exact.
        let straddle = c.span(n - 50, 100, DType::F32).unwrap();
        assert_eq!(straddle.dims(), &[1, 1, 100, HEAD_DIM]);
        let dense_half = straddle.narrow(2, 50, 50).unwrap();
        let want_dense = c.span(n, 50, DType::F32).unwrap();
        assert_eq!(rel_err(&want_dense, &dense_half), 0.0);
    }

    /// MUTATION GUARD — feeding the wrong head dim must be refused by name,
    /// not silently reinterpreted.
    #[test]
    fn head_dim_mismatch_is_refused() {
        let mut c = cache();
        let err = c
            .append(&activations(1, 1, 4, 128, 3))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("head_dim mismatch") && err.contains("512"),
            "unhelpful error: {err}"
        );
    }

    /// Truncation maps onto both time bases and stays readable afterwards —
    /// the prefix-cacher / MTP-rollback contract.
    #[test]
    fn truncation_maps_onto_both_time_bases() {
        let mut c = cache();
        c.append(&activations(1, 1, 2048, HEAD_DIM, 17)).unwrap();
        for step in 0..400 {
            c.append(&activations(1, 1, 1, HEAD_DIM, 6000 + step))
                .unwrap();
        }
        let n = c.compressed_len();
        assert!(n > 0);

        // Shallow rollback: inside the dense tail.
        let target = c.current_seq_len() - 8;
        c.set_len(target).unwrap();
        assert_eq!(c.current_seq_len(), target);
        assert_eq!(c.compressed_len(), n, "a shallow rollback must not evict");
        c.span(c.dense_from(), target - c.dense_from(), DType::F32)
            .unwrap();

        // Deep rollback: lands inside the compressed region, which is legal
        // here (unlike XsRollingCache) because every token still has a record.
        c.set_len(n - 10).unwrap();
        assert_eq!(c.current_seq_len(), n - 10);
        assert_eq!(c.dense_from(), n - 10);
        let got = c.span(0, n - 10, DType::F32).unwrap();
        assert_eq!(got.dims(), &[1, 1, n - 10, HEAD_DIM]);

        // Extension is the thing that is genuinely impossible.
        let err = c.set_len(c.current_seq_len() + 1).unwrap_err().to_string();
        assert!(err.contains("cannot extend"), "unhelpful error: {err}");

        c.reset();
        assert_eq!(c.current_seq_len(), 0);
        assert_eq!(c.compressed_len(), 0);
    }

    /// Batched rows must not be mixed up. Every row gets its OWN stream — a
    /// shared stream would make every row's records identical and hide any
    /// row-indexing bug entirely (the fixture trap wave63-CO hit).
    #[test]
    fn rows_keep_their_own_keys() {
        let mut c = V4TurboKCache::try_new(HEAD_DIM, 4, 42, WINDOW, 8192).unwrap();
        let b = 3;
        let prefill = activations(b, 1, 1024, HEAD_DIM, 23);
        c.append(&prefill).unwrap();
        for step in 0..400 {
            c.append(&activations(b, 1, 1, HEAD_DIM, 4000 + step))
                .unwrap();
        }
        assert!(c.compressed_len() > 0);
        let got = c.span(10, 64, DType::F32).unwrap();
        assert_eq!(got.dims(), &[b, 1, 64, HEAD_DIM]);
        let want = prefill.narrow(2, 10, 64).unwrap();
        for r in 0..b {
            let g = got.narrow(0, r, 1).unwrap();
            let w = want.narrow(0, r, 1).unwrap();
            assert!(rel_err(&w, &g) < 0.30, "row {r} reconstructed wrongly");
            // And row r must be closer to its OWN source than to any other's.
            for other in 0..b {
                if other == r {
                    continue;
                }
                let w_other = want.narrow(0, other, 1).unwrap();
                assert!(
                    rel_err(&w, &g) < rel_err(&w_other, &g),
                    "row {r} decoded closer to row {other}'s keys — rows are crossed"
                );
            }
        }
    }
}
