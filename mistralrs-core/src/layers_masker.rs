#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::ops::Add;

use candle_core::{DType, Device, Result, Tensor, WithDType, D};

use crate::pipeline::KvCache;

// https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py
pub struct CausalMasker;

// https://github.com/mokeyish/candle-ext/blob/main/src/masked_fill.rs
/// xs are on false (0), value is on true (1)
pub fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
    let on_true = Tensor::full(value, xs.shape(), xs.device())?.to_dtype(xs.dtype())?;
    let on_false = xs;
    let res = mask
        .broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)?;
    Ok(res)
}

pub struct NotACache;

pub trait PastKvLenCache {
    fn get_past_kv_len(&self) -> Result<usize>;

    /// The number of **live** past positions each sequence in the batch holds,
    /// when the dense batched cache is left-aligned and therefore ragged.
    ///
    /// `None` — the default, and what every caller in the tree returns today —
    /// means "every row is live for the whole `get_past_kv_len()`", i.e. the
    /// mask is batch-invariant and [`CausalMasker::make_causal_mask_matrix`]
    /// builds exactly the rank-2 mask it always has.
    ///
    /// `Some(lens)` means row `b`'s live run is the **suffix**
    /// `[past - lens[b], past)`; the `past - lens[b]` columns ahead of it are
    /// the zero-filled dead prefix [`crate::kv_cache::front_pad_kv_cache`]
    /// leaves behind. Those columns are not harmless: a zero K row scores logit
    /// 0 and takes real softmax weight, so they must be masked, and this is how
    /// the mask learns about them.
    fn per_seq_kv_lens(&self) -> Option<&[usize]> {
        None
    }
}

/// A left-aligned ragged batch's mask inputs: the padded width every row shares
/// and the live length each row actually holds.
///
/// Pass this as the `cache` argument to
/// [`CausalMasker::make_causal_mask_matrix`] to get a `[B, 1, t_q, k]` additive
/// mask that kills each row's dead prefix as well as the future.
pub struct RaggedKvLens<'a> {
    /// Columns every row of the batched buffer carries.
    pub padded_len: usize,
    /// `live[b]` — real past positions in row `b`, `<= padded_len`.
    pub live: &'a [usize],
}

impl PastKvLenCache for RaggedKvLens<'_> {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(self.padded_len)
    }
    fn per_seq_kv_lens(&self) -> Option<&[usize]> {
        Some(self.live)
    }
}

impl PastKvLenCache for NotACache {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(0)
    }
}

impl PastKvLenCache for Vec<KvCache> {
    fn get_past_kv_len(&self) -> Result<usize> {
        let kv_cache_1 = &self[0];
        Ok(kv_cache_1.current_seq_len())
    }
}

impl PastKvLenCache for &[usize] {
    fn get_past_kv_len(&self) -> Result<usize> {
        if self.windows(2).all(|w| w[0] == w[1]) {
            Ok(self[0])
        } else {
            Ok(0)
        }
    }
}

impl PastKvLenCache for Vec<Option<(Tensor, Tensor)>> {
    fn get_past_kv_len(&self) -> Result<usize> {
        let kv_cache_1 = &self[0];
        if kv_cache_1.is_none() {
            return Ok(0);
        }
        let k_cache_1 = &kv_cache_1.as_ref().unwrap().0;
        Ok(k_cache_1.dims()[2])
    }
}

impl CausalMasker {
    fn make_mask(&self, tgt_len: usize, past_kv_len: usize, device: &Device) -> Result<Tensor> {
        let offset = tgt_len + past_kv_len;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..offset).map(move |j| u8::from(j + tgt_len > i + offset)))
            .collect();
        Tensor::from_slice(&mask, (tgt_len, offset), device)
    }

    fn make_mask_chunked(
        &self,
        tgt_len: usize,
        past_kv_len: usize,
        chunk_size: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let offset = tgt_len + past_kv_len;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..offset).map(move |j| {
                    // For past key-value positions
                    if j < past_kv_len {
                        return 0;
                    }

                    // Adjust j to account for past_kv_len
                    let j_adj = j - past_kv_len;

                    // Calculate block position (equivalent to block_pos)
                    let i_block = i / chunk_size;
                    let j_block = j_adj / chunk_size;
                    let block_pos = (i_block as isize - j_block as isize).abs();

                    // Calculate token position (equivalent to token_pos)
                    let token_pos = j_adj as isize - i as isize;

                    // Apply mask conditions: same block and causal
                    1 - u8::from((block_pos == 0) && (token_pos <= 0))
                })
            })
            .collect();

        Tensor::from_slice(&mask, (tgt_len, offset), device)
    }

    fn make_swa_mask(
        &self,
        tgt_len: usize,
        seqlen_offset: usize,
        sliding_window: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.to_dtype(dtype)
    }

    /// Expands a mask from (bs, seq_len) to (bs, 1, tgt_len, seq_len)
    /// If tgt_len is None, use seq_len
    pub fn expand_mask(
        &self,
        mask: &Tensor,
        dtype: DType,
        tgt_len: Option<usize>,
    ) -> Result<Tensor> {
        let (bs, src_len) = mask.dims2()?;

        let expanded_mask = mask.unsqueeze(1)?.unsqueeze(1)?;
        let expanded_mask = expanded_mask
            .expand((bs, 1, tgt_len.unwrap_or(src_len), src_len))?
            .to_dtype(dtype)?;

        let inverted_mask = expanded_mask.neg()?.add(1.0f64)?;
        masked_fill(
            &inverted_mask,
            &inverted_mask.to_dtype(DType::U8)?,
            f32::MIN,
        )
    }

    pub fn calculate_past_kv_len(
        &self,
        cache: &[Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<usize> {
        let kv_cache_1 = &cache[0];
        if kv_cache_1.is_none() {
            return Ok(0);
        }
        let k_cache_1 = &kv_cache_1.as_ref().unwrap().0;
        Ok(k_cache_1.dims()[2])
    }

    /// The additive `[B, 1, t_q, past + t_q]` mask a **left-aligned ragged**
    /// batch needs: `-inf` on the future (ordinary causality) and `-inf` on
    /// each row's dead prefix `[0, past - live[b])`.
    ///
    /// Query row `i` of sequence `b` sits at absolute position
    /// `live[b] + i`, and its live keys are the columns
    /// `[past - live[b], past + i]`. Everything outside that is killed. Note
    /// the row is never fully masked — column `past + i` is the query's own
    /// position — so softmax cannot produce a NaN row.
    pub fn make_left_padded_causal_mask(
        &self,
        b_sz: usize,
        tgt_len: usize,
        past_kv_len: usize,
        live: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        if live.len() != b_sz {
            candle_core::bail!(
                "ragged mask: {} live lengths for a batch of {b_sz}",
                live.len()
            );
        }
        let k_len = past_kv_len + tgt_len;
        let mut data: Vec<f32> = Vec::with_capacity(b_sz * tgt_len * k_len);
        for &l in live {
            if l > past_kv_len {
                candle_core::bail!(
                    "ragged mask: live length {l} exceeds the padded width {past_kv_len}"
                );
            }
            let lead = past_kv_len - l;
            for i in 0..tgt_len {
                let last = past_kv_len + i;
                for j in 0..k_len {
                    data.push(if j < lead || j > last {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    });
                }
            }
        }
        Tensor::from_vec(data, (b_sz, 1, tgt_len, k_len), device)?.to_dtype(dtype)
    }

    pub fn make_causal_mask_matrix(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        dtype: DType,
        _n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        let past_kv_len = cache.get_past_kv_len()?;
        let (b_sz, tgt_len) = input_ids.dims2()?;

        // 🔑 A left-aligned ragged batch needs a mask even at `tgt_len == 1`,
        // and it cannot take the flash-attn shortcut: both of those assume the
        // mask is a pure function of causality, and the dead prefix
        // `front_pad_kv_cache` leaves is neither causal nor batch-invariant.
        // Order matters — this branch has to come before both early returns.
        if let Some(live) = cache.per_seq_kv_lens() {
            return Ok(Some(self.make_left_padded_causal_mask(
                b_sz,
                tgt_len,
                past_kv_len,
                live,
                dtype,
                input_ids.device(),
            )?));
        }

        // 🔑 The same thing, learned from the batch rather than from the cache
        // argument. `NormalCacheManager::clone_in_cache` front-aligns a ragged
        // cohort and publishes each row's dead prefix; every model reaches this
        // function already, so picking it up here is what makes ragged dense
        // decode work WITHOUT threading a new argument through all forty-odd
        // model forwards. (Threading it would not have been enough anyway —
        // `Sdpa::run_attention`'s flash branch takes no mask argument, so the
        // mask has to be routed to the bias path too; see
        // `attention::mask_must_be_applied_as_bias`.)
        //
        // `live[i] = past_kv_len - lead_pad[i]`, recovered from the cache's own
        // current length so it stays correct as the cohort grows between
        // `clone_in_cache` calls.
        if let Some(lead_pad) = crate::kv_cache::ragged_lead_pad() {
            if lead_pad.len() == b_sz && lead_pad.iter().any(|l| *l > 0) {
                if let Some(live) = lead_pad
                    .iter()
                    .map(|l| past_kv_len.checked_sub(*l))
                    .collect::<Option<Vec<usize>>>()
                {
                    return Ok(Some(self.make_left_padded_causal_mask(
                        b_sz,
                        tgt_len,
                        past_kv_len,
                        &live,
                        dtype,
                        input_ids.device(),
                    )?));
                }
            }
        }

        if tgt_len == 1 {
            return Ok(None);
        }

        // Avoid materializing large sliding-window masks when flash-attn on CUDA.
        if crate::using_flash_attn() && input_ids.device().is_cuda() {
            return Ok(Some(Tensor::zeros((1, 1), dtype, input_ids.device())?));
        }

        let mut causal_mask = self
            .make_mask(tgt_len, past_kv_len, input_ids.device())?
            .to_dtype(DType::U8)?;

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mut mask =
                causal_mask.broadcast_as((causal_mask.dims()[0], causal_mask.dims()[1]))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;
            mask
        };

        Ok(Some(causal_mask))
    }

    /// Like `make_causal_mask_matrix` but always constructs a real mask (never returns
    /// the flash-attn dummy tensor). Use when flash attention is being bypassed.
    pub fn make_causal_mask_as_attn_bias(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let past_kv_len = cache.get_past_kv_len()?;
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }

        let mut causal_mask = self
            .make_mask(tgt_len, past_kv_len, input_ids.device())?
            .to_dtype(DType::U8)?;

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mask = causal_mask.broadcast_as((causal_mask.dims()[0], causal_mask.dims()[1]))?;
            masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?
        };

        Ok(Some(causal_mask))
    }

    /// Like `make_sliding_window_causal_mask_matrix` but always constructs a real mask
    /// (never returns the flash-attn dummy tensor). Use when flash attention is being bypassed.
    pub fn make_sliding_window_causal_mask_as_attn_bias(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        sliding_window: Option<usize>,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        if sliding_window.is_none() {
            return self.make_causal_mask_as_attn_bias(input_ids, cache, dtype);
        }
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        let sliding_window = sliding_window.unwrap();

        let past_kv_len = cache
            .get_past_kv_len()?
            .min(sliding_window.saturating_sub(tgt_len));
        if tgt_len == 1 {
            return Ok(None);
        }

        Ok(Some(self.make_swa_mask(
            tgt_len,
            past_kv_len,
            sliding_window,
            input_ids.device(),
            dtype,
        )?))
    }

    pub fn make_chunked_mask_matrix(
        &self,
        input_ids: &Tensor,
        chunk_size: usize,
        cache: &dyn PastKvLenCache,
        dtype: DType,
        _n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        let past_kv_len = cache.get_past_kv_len()?;
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }

        let mut causal_mask = self
            .make_mask_chunked(tgt_len, past_kv_len, chunk_size, input_ids.device())?
            .to_dtype(DType::U8)?;

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mut mask =
                causal_mask.broadcast_as((causal_mask.dims()[0], causal_mask.dims()[1]))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;
            mask
        };

        Ok(Some(causal_mask))
    }

    pub fn make_sliding_window_causal_mask_matrix(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        sliding_window: Option<usize>,
        dtype: DType,
        n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        if sliding_window.is_none() {
            return self.make_causal_mask_matrix(input_ids, cache, dtype, n_attn_heads);
        }
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        let sliding_window = sliding_window.unwrap();

        // Avoid materializing large sliding-window masks when flash-attn on CUDA.
        if tgt_len > 1 && crate::using_flash_attn() && input_ids.device().is_cuda() {
            return Ok(Some(Tensor::zeros((1, 1), dtype, input_ids.device())?));
        }

        // Compare the past KV len to the sliding window size. If the past kv len is 0 (no prefix cache), then this will be 0.
        // Otherwise, this will be the number required such that the mask fits the size of the k/v seqlen (usually sliding window)
        let past_kv_len = cache
            .get_past_kv_len()?
            .min(sliding_window.saturating_sub(tgt_len));
        if tgt_len == 1 {
            return Ok(None);
        }

        Ok(Some(self.make_swa_mask(
            tgt_len,
            past_kv_len,
            sliding_window,
            input_ids.device(),
            dtype,
        )?))
    }

    pub fn apply_mask_one_and_zero(
        &self,
        mask: &Option<Tensor>,
        att: Tensor,
        neg_inf: &Tensor,
    ) -> Result<Tensor> {
        match mask {
            None => Ok(att),
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                mask.where_cond(
                    &neg_inf
                        .to_device(att.device())?
                        .to_dtype(att.dtype())?
                        .broadcast_as(att.dims())?,
                    &att,
                )
            }
        }
    }
}

pub struct BidirectionalMasker;

impl BidirectionalMasker {
    fn make_swa_mask(
        &self,
        tgt_len: usize,
        sliding_window: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    // https://github.com/huggingface/transformers/blob/a0bf5a82eebf88ee9f52145be427f6f1541329f6/src/transformers/models/gemma3/modeling_gemma3.py#L478
                    // A token can attend to any other token if their absolute distance is within the (exclusive) sliding window size (distance < sliding_window)."
                    if (i as isize - j as isize).unsigned_abs() >= sliding_window {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
        mask.to_dtype(dtype)
    }

    pub fn make_mask(&self, input_ids: &Tensor, dtype: DType) -> Result<Tensor> {
        let (_b_sz, tgt_len) = input_ids.dims2()?;

        // Avoid materializing large sliding-window masks when flash-attn on CUDA.
        if crate::using_flash_attn() && input_ids.device().is_cuda() {
            return Tensor::zeros((1, 1), dtype, input_ids.device());
        }

        // Do not make any -inf
        let mask = Tensor::zeros((tgt_len, tgt_len), dtype, input_ids.device())?;

        Ok(mask)
    }
    pub fn make_sliding_mask(
        &self,
        input_ids: &Tensor,
        dtype: DType,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let (_b_sz, tgt_len) = input_ids.dims2()?;

        // Avoid materializing large sliding-window masks when flash-attn on CUDA.
        if crate::using_flash_attn() && input_ids.device().is_cuda() {
            return Tensor::zeros((1, 1), dtype, input_ids.device());
        }

        let mask = self.make_swa_mask(tgt_len, sliding_window, input_ids.device(), dtype)?;

        Ok(mask)
    }
}

#[cfg(test)]
mod ragged_mask_tests {
    use super::*;

    fn mask_rows(m: &Tensor) -> Vec<Vec<f32>> {
        let (b, _, q, k) = m.dims4().unwrap();
        let flat = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (0..b * q)
            .map(|r| flat[r * k..(r + 1) * k].to_vec())
            .collect()
    }

    /// 🔑 The mask a left-aligned per-sequence KV cache needs. Three things
    /// have to hold at once, and each one is a wrong answer if it does not:
    ///
    /// * the dead prefix `front_pad_kv_cache` zero-fills is `-inf` — a zero K
    ///   row is NOT a masked row, it scores logit 0 and takes softmax weight;
    /// * the future is `-inf` — ordinary causality, which the ragged branch
    ///   must not lose;
    /// * no row is entirely `-inf` — a fully masked row is a softmax NaN.
    #[test]
    fn left_padded_mask_kills_the_dead_prefix_and_the_future_and_never_a_whole_row() {
        let device = Device::Cpu;
        // Padded width 6; sequence 0 holds 6 live positions, sequence 1 holds
        // 2, so row 1 carries a 4-wide dead prefix. Two query rows.
        let live = [6usize, 2];
        let m = CausalMasker
            .make_left_padded_causal_mask(2, 2, 6, &live, DType::F32, &device)
            .unwrap();
        assert_eq!(m.dims(), &[2, 1, 2, 8]);
        let rows = mask_rows(&m);

        // seq 0, query 0: keys 0..=6 live (own position is column 6), 7 future.
        assert!(rows[0][..7].iter().all(|x| *x == 0.0));
        assert_eq!(rows[0][7], f32::NEG_INFINITY);
        // seq 0, query 1: keys 0..=7 all live.
        assert!(rows[1].iter().all(|x| *x == 0.0));
        // seq 1, query 0: columns 0..4 are the dead prefix.
        assert!(
            rows[2][..4].iter().all(|x| *x == f32::NEG_INFINITY),
            "the dead prefix must be masked or the model attends zero-filled keys"
        );
        assert!(rows[2][4..7].iter().all(|x| *x == 0.0));
        assert_eq!(rows[2][7], f32::NEG_INFINITY);
        // seq 1, query 1.
        assert!(rows[3][..4].iter().all(|x| *x == f32::NEG_INFINITY));
        assert!(rows[3][4..8].iter().all(|x| *x == 0.0));

        for (i, row) in rows.iter().enumerate() {
            assert!(
                row.iter().any(|x| *x == 0.0),
                "row {i} is entirely masked, which is a softmax NaN"
            );
        }
    }

    /// A batch whose rows are all fully live must produce exactly the ordinary
    /// causal mask — so turning the ragged path on cannot change a uniform
    /// batch's answer.
    #[test]
    fn a_fully_live_ragged_mask_is_the_ordinary_causal_mask() {
        let device = Device::Cpu;
        let m = CausalMasker
            .make_left_padded_causal_mask(2, 3, 4, &[4, 4], DType::F32, &device)
            .unwrap();
        let rows = mask_rows(&m);
        for (r, row) in rows.iter().enumerate() {
            let last = 4 + (r % 3);
            for (j, v) in row.iter().enumerate() {
                let want = if j > last { f32::NEG_INFINITY } else { 0.0 };
                assert_eq!(*v, want, "row {r} column {j}");
            }
        }
    }

    /// 🔑 The channel is what makes ragged dense decode reach the mask at all.
    ///
    /// No model passes a `RaggedKvLens`; they pass their own cache or their
    /// `seqlen_offsets`, both of which report `per_seq_kv_lens() == None`. So a
    /// decode step (`tgt_len == 1`) would take the `return Ok(None)` shortcut
    /// and attend over each short row's zero-filled dead prefix — logit 0, real
    /// softmax weight, silently wrong. Reading `clone_in_cache`'s published
    /// `lead_pad` here is what closes that, without touching a single model.
    #[test]
    fn the_ragged_channel_produces_a_mask_where_the_cache_argument_cannot() {
        let device = Device::Cpu;
        let ids = Tensor::zeros((2, 1), DType::U32, &device).unwrap();
        // What every model actually passes on the dense path.
        let offsets: &[usize] = &[5, 5];

        // Channel unset: unchanged behaviour, no mask at tgt_len == 1.
        crate::kv_cache::set_ragged_lead_pad(None);
        assert!(CausalMasker
            .make_causal_mask_matrix(&ids, &offsets, DType::F32, 1)
            .unwrap()
            .is_none());

        // Channel set with a real dead prefix: row 1 holds 1 live position of
        // the 5 columns, so its first 4 must be killed.
        crate::kv_cache::set_ragged_lead_pad(Some(vec![0, 4]));
        let m = CausalMasker
            .make_causal_mask_matrix(&ids, &offsets, DType::F32, 1)
            .unwrap()
            .expect("a front-aligned ragged cohort must get a mask at tgt_len == 1");
        assert_eq!(m.dims(), &[2, 1, 1, 6]);
        let rows = mask_rows(&m);
        assert!(rows[0].iter().all(|x| *x == 0.0), "row 0 is fully live");
        assert!(
            rows[1][..4].iter().all(|x| *x == f32::NEG_INFINITY),
            "row 1's dead prefix must be masked or it attends zero-filled keys"
        );
        assert!(rows[1][4..].iter().all(|x| *x == 0.0));

        // An all-zero lead_pad is a uniform cohort: it must NOT divert to the
        // bias path, or every uniform decode batch would lose the flash kernel.
        crate::kv_cache::set_ragged_lead_pad(Some(vec![0, 0]));
        assert!(
            CausalMasker
                .make_causal_mask_matrix(&ids, &offsets, DType::F32, 1)
                .unwrap()
                .is_none(),
            "a cohort with no dead prefix must stay on the unmasked fast path"
        );

        crate::kv_cache::set_ragged_lead_pad(None);
    }

    /// `make_causal_mask_matrix` routes to the ragged builder BEFORE both of
    /// its early returns — the `tgt_len == 1` shortcut and the flash-attn
    /// placeholder. A decode step is `tgt_len == 1` and still has a dead prefix
    /// to mask, so returning `None` there would serve from unmasked keys.
    #[test]
    fn a_single_query_row_still_gets_its_ragged_mask() {
        let device = Device::Cpu;
        let ids = Tensor::zeros((2, 1), DType::U32, &device).unwrap();
        let ragged = RaggedKvLens {
            padded_len: 5,
            live: &[5, 1],
        };
        let m = CausalMasker
            .make_causal_mask_matrix(&ids, &ragged, DType::F32, 1)
            .unwrap()
            .expect("a ragged batch must always get a mask, even at tgt_len == 1");
        assert_eq!(m.dims(), &[2, 1, 1, 6]);
        let rows = mask_rows(&m);
        assert!(rows[0].iter().all(|x| *x == 0.0));
        assert!(rows[1][..4].iter().all(|x| *x == f32::NEG_INFINITY));
        assert!(rows[1][4..].iter().all(|x| *x == 0.0));

        // And the default (`per_seq_kv_lens() == None`) path is untouched: one
        // query row over a uniform batch still returns `None`.
        let offsets: &[usize] = &[5, 5];
        assert!(CausalMasker
            .make_causal_mask_matrix(&ids, &offsets, DType::F32, 1)
            .unwrap()
            .is_none());
    }
}
