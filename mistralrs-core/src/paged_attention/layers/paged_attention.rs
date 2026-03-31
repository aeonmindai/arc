use candle_core::{DType, Device, Result, Tensor};
#[allow(unused_imports)]
use mistralrs_paged_attn::{kv_scale_update, paged_attention, reshape_and_cache};

const KV_SCALE_UPDATE_ITERATION: i32 = 128;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::{
    attention::SdpaParams,
    layers::Sdpa,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

static PAGED_ATTN_LAYER_COUNTER: AtomicI32 = AtomicI32::new(0);

pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    kv_updated_times: AtomicI32,
    /// Layer index for looking up TurboQuant norms from the global registry.
    layer_idx: usize,
}

impl PagedAttention {
    pub fn new(head_dim: usize, device: &Device, alibi_slopes: Option<Vec<f32>>) -> Result<Self> {
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            assert_eq!(alibi_slopes.len(), head_dim);
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };

        let layer_idx = PAGED_ATTN_LAYER_COUNTER.fetch_add(1, Ordering::SeqCst) as usize;

        Ok(Self {
            alibi_slopes,
            k_scale: Some(Tensor::new(1f32, device)?),
            v_scale: Some(Tensor::new(1f32, device)?),
            kv_updated_times: AtomicI32::new(0),
            layer_idx,
        })
    }

    /// Reset the layer counter (call before model construction).
    pub fn reset_layer_counter() {
        PAGED_ATTN_LAYER_COUNTER.store(0, Ordering::SeqCst);
    }

    fn is_turboquant(&self) -> bool {
        crate::paged_attention::has_global_turbo_norms()
    }

    fn get_turbo_norms(&self) -> Option<(Tensor, Tensor)> {
        crate::paged_attention::get_global_turbo_norms(self.layer_idx)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    /// query: shape = [batch_size, seq_len, num_heads * head_size]
    /// key: shape = [batch_size, seq_len, num_kv_heads * head_size]
    /// value: shape = [batch_size, num_kv_heads * head_size]
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    /// input_metadata: metadata for paged attention.
    #[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        mut key_cache: Option<Tensor>,
        mut value_cache: Option<Tensor>,
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        // Get TurboQuant norms for this layer (if active)
        let turbo_norms = self.get_turbo_norms();

        // FP8 scale updates (only for FP8 cache, not TurboQuant)
        if !self.is_turboquant() {
            if let (Some(k_scale), Some(v_scale), Some(key_cache)) =
                (&self.k_scale, &self.v_scale, &key_cache)
            {
                if self.kv_updated_times.load(Ordering::Relaxed) < KV_SCALE_UPDATE_ITERATION
                    && key_cache.dtype() == DType::F8E4M3
                {
                    kv_scale_update(key, value, k_scale, v_scale)?;
                    self.kv_updated_times.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        let slot_mapping = input_metadata
            .slot_mappings
            .get(&query.device().location())
            .unwrap();
        let dims = slot_mapping.dims();
        let slot_mapping = if dims.len() > 1 {
            &slot_mapping.flatten(0, dims.len())?
        } else {
            slot_mapping
        };

        let use_full =
            sdpa_params.sliding_window.is_none() && input_metadata.full_block_tables.is_some();

        let block_tables = if use_full {
            input_metadata
                .full_block_tables
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        } else {
            input_metadata
                .block_tables
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        };
        let context_lens = if use_full {
            input_metadata
                .full_context_lens
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        } else {
            input_metadata
                .context_lens
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        };

        let alibi_slopes = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            Some(alibi_slopes.to_device(query.device())?)
        } else {
            None
        };

        let (batch_size, attention_heads, seq_len, head_size) = query.shape().dims4()?;
        let (_, key_value_heads, _, _) = key.shape().dims4()?;

        // === Prefix cache hit path ===
        if input_metadata.num_cached_tokens.is_some() && attention_mask.is_some() {
            // Write new tokens to cache
            if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
                let k_flat = key
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                let v_flat = value
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;

                if self.is_turboquant() {
                    #[cfg(feature = "cuda")]
                    {
                        mistralrs_paged_attn::turbo_reshape_and_cache(
                            &k_flat,
                            &v_flat,
                            key_cache.as_mut().unwrap(),
                            value_cache.as_mut().unwrap(),
                            &turbo_norms.as_ref().unwrap().0,
                            &turbo_norms.as_ref().unwrap().1,
                            slot_mapping,
                        )?;
                    }
                } else {
                    reshape_and_cache(
                        &k_flat,
                        &v_flat,
                        self.k_scale.as_ref(),
                        self.v_scale.as_ref(),
                        key_cache.as_mut().unwrap(),
                        value_cache.as_mut().unwrap(),
                        slot_mapping,
                    )?;
                }
            }

            assert!(
                alibi_slopes.is_none(),
                "alibi slopes not supported in prefix cache path"
            );

            let device = query.device();
            let cu_kv = input_metadata
                .cu_seqlens_kv
                .as_ref()
                .expect("cu_seqlens_kv required for prefix cache path")
                .get(&device.location())
                .unwrap();

            // For TurboQuant prefix cache, gather and dequantize is not yet supported.
            // Fall through to standard gather for now.
            let (k_gathered, v_gathered) = mistralrs_paged_attn::gather_kv_cache(
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                block_tables,
                cu_kv,
                query.dtype(),
            )?;

            let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;

            let prefix_flash_params = flash_params.map(|fp| {
                let max_kv = input_metadata
                    .num_cached_tokens
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(input_metadata.query_lens.as_ref().unwrap().iter())
                    .map(|(&nc, &ql)| (nc + ql) as u32)
                    .max()
                    .unwrap_or(0);
                FlashParams {
                    max_q: fp.max_q,
                    max_k: max_kv,
                    cumulative_seqlens_q: fp.cumulative_seqlens_q.clone(),
                    cumulative_seqlens_k: input_metadata.cu_seqlens_kv.as_ref().unwrap().clone(),
                    causal: fp.causal,
                }
            });

            return Sdpa.run_attention(
                query,
                &k_4d,
                &v_4d,
                attention_mask,
                prefix_flash_params.as_ref(),
                sdpa_params,
            );
        }

        #[allow(clippy::cast_possible_truncation)]
        let att = match attention_mask {
            None => None,
            Some(mask) => Some(Sdpa.run_attention(
                query,
                key,
                value,
                Some(mask),
                flash_params,
                sdpa_params,
            )?),
        };

        let (query, key, value) = if seq_len > 1 {
            let q = query
                .transpose(1, 2)?
                .reshape(((), attention_heads, head_size))?;
            let k = key
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            let v = value
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        } else {
            let q = query.reshape(((), attention_heads, head_size))?;
            let k = key.reshape(((), key_value_heads, head_size))?;
            let v = value.reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        };

        // Write K/V to paged cache
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            if self.is_turboquant() {
                #[cfg(feature = "cuda")]
                {
                    mistralrs_paged_attn::turbo_reshape_and_cache(
                        &key,
                        &value,
                        key_cache.as_mut().unwrap(),
                        value_cache.as_mut().unwrap(),
                        &turbo_norms.as_ref().unwrap().0,
                        &turbo_norms.as_ref().unwrap().1,
                        slot_mapping,
                    )?;
                }
            } else {
                reshape_and_cache(
                    &key,
                    &value,
                    self.k_scale.as_ref(),
                    self.v_scale.as_ref(),
                    key_cache.as_mut().unwrap(),
                    value_cache.as_mut().unwrap(),
                    slot_mapping,
                )?;
            }
        }

        if let Some(att) = att {
            return Ok(att);
        }

        // Decode attention
        if self.is_turboquant() {
            #[cfg(feature = "cuda")]
            {
                let res = mistralrs_paged_attn::turbo_paged_attention(
                    &query,
                    key_cache.as_ref().unwrap(),
                    value_cache.as_ref().unwrap(),
                    &turbo_norms.as_ref().unwrap().0,
                    &turbo_norms.as_ref().unwrap().1,
                    block_tables,
                    context_lens,
                    if use_full {
                        input_metadata.full_max_context_len.unwrap()
                    } else {
                        input_metadata.max_context_len.unwrap()
                    },
                    sdpa_params.softmax_scale,
                    sdpa_params.softcap.unwrap_or(1.0f32),
                    key_value_heads,
                )?;
                return Ok(res);
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("TurboQuant paged attention requires CUDA");
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        let res = paged_attention(
            &query,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            block_tables,
            context_lens,
            alibi_slopes.as_ref(),
            if use_full {
                input_metadata.full_max_context_len.unwrap()
            } else {
                input_metadata.max_context_len.unwrap()
            },
            sdpa_params.softmax_scale,
            sdpa_params.softcap.unwrap_or(1.0f32),
            sdpa_params.sinks.as_ref(),
        )?;

        Ok(res)
    }
}
