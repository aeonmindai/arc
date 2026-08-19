#![allow(clippy::cast_possible_truncation)]

use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use text_models_inputs_processor::PagedAttentionMeta;
use tokenizers::Tokenizer;

use crate::{device_map::DeviceMapper, sequence::Sequence};

#[derive(PartialEq)]
pub enum InputsProcessorType {
    Text,
    Vision,
    Embedding,
}

pub struct InputProcessorOutput {
    pub inputs: Box<dyn Any>,
    pub seq_indices: Vec<usize>,
}

/// Processor: Prepare inputs for the model (potentially preparing the images if applicable)
pub trait InputsProcessor {
    /// This should also enable matmul via f16 if prompt and the sequence length is greater than 32.
    /// Otherwise, matmul via f16 is disabled.
    ///
    /// This should return a type which can be downcasted to the proper type as used in `forward_inputs`
    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput>;

    fn get_type(&self) -> InputsProcessorType;
}

// ========================= Test models input processor

pub mod text_models_inputs_processor {
    use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

    use anyhow::Result;
    use candle_core::{DType, Device, DeviceLocation, Tensor, WithDType};
    use tokenizers::Tokenizer;

    use crate::{
        device_map::DeviceMapper,
        get_mut_arcmutex,
        paged_attention::{KVCacheManager, _PAD_SLOT_ID},
        sequence::Sequence,
    };

    use super::{InputProcessorOutput, InputsProcessor, InputsProcessorType};

    fn _make_tensor_with_pad<D: WithDType>(
        x: Vec<Vec<D>>,
        max_len: usize,
        pad: D,
        device: &Device,
    ) -> Result<Tensor> {
        let mut padded_x = Vec::new();
        for mut x_i in x {
            assert!(x_i.len() <= max_len);
            x_i.extend([pad].repeat(max_len - x_i.len()));
            let shape = (x_i.len(),);
            padded_x.push(Tensor::from_vec(x_i, shape, device)?);
        }
        Tensor::cat(&padded_x[..], 0).map_err(anyhow::Error::msg)
    }

    #[derive(Clone)]
    pub struct PagedAttentionMeta {
        pub sliding_window: Option<usize>,
        pub block_size: usize,
        pub kv_cache_manager: Arc<tokio::sync::Mutex<KVCacheManager>>,
    }

    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    pub struct PagedAttentionInputMetadata {
        /// Block tables, windowed when a global sliding_window is set.
        pub block_tables: Option<HashMap<DeviceLocation, Tensor>>,
        /// Context lens, capped by sliding_window when set.
        pub context_lens: Option<HashMap<DeviceLocation, Tensor>>,
        pub slot_mappings: HashMap<DeviceLocation, Tensor>,
        pub max_context_len: Option<usize>,
        /// Full (unwindowed) block tables, always covering the entire context.
        /// For models with per-layer sliding windows (GPT-OSS, Gemma2), layers
        /// without a sliding window should use these instead of `block_tables`.
        pub full_block_tables: Option<HashMap<DeviceLocation, Tensor>>,
        /// Full context lens (not capped by sliding_window).
        pub full_context_lens: Option<HashMap<DeviceLocation, Tensor>>,
        pub full_max_context_len: Option<usize>,
        pub is_first_prompt_chunk: bool,
        pub paged_kv_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_last_page_len: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_request_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_tile_indices: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_o_indptr: Option<HashMap<DeviceLocation, Tensor>>,
        pub paged_kv_chunk_size: Option<HashMap<DeviceLocation, Tensor>>,
        /// Number of cached tokens per sequence (from prefix cache hits).
        /// When present and > 0, gather_kv_cache + Sdpa is used during prefill
        /// instead of flash attention. The Q/K/V tensors should only contain
        /// the NEW (non-cached) tokens.
        pub num_cached_tokens: Option<Vec<usize>>,
        /// Number of new tokens per sequence (query lengths).
        pub query_lens: Option<Vec<usize>>,
        /// Cumulative query lengths [batch+1], u32 — for Sdpa varlen flash path.
        /// Precomputed to avoid Tensor::new in the forward hot path.
        pub cu_seqlens_q: Option<HashMap<DeviceLocation, Tensor>>,
        /// Cumulative KV lengths [batch+1], u32 — for gather_kv_cache and flash_attn_varlen.
        /// Each entry is sum of (cached + new) tokens.
        pub cu_seqlens_kv: Option<HashMap<DeviceLocation, Tensor>>,
    }

    impl PagedAttentionInputMetadata {
        /// Create a dummy input metadata, assuming that this will NOT be used for decoding.
        /// This is used for the case of imatrix generation.
        pub fn dummy(dev: &Device) -> candle_core::Result<Self> {
            Ok(PagedAttentionInputMetadata {
                block_tables: None,
                context_lens: None,
                max_context_len: None,
                full_block_tables: None,
                full_context_lens: None,
                full_max_context_len: None,
                slot_mappings: HashMap::from([(dev.location(), Tensor::new(&[0f32], dev)?)]),
                is_first_prompt_chunk: true,
                paged_kv_indptr: None,
                paged_kv_indices: None,
                paged_kv_last_page_len: None,
                paged_kv_request_indices: None,
                paged_kv_tile_indices: None,
                paged_kv_o_indptr: None,
                paged_kv_chunk_size: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
            })
        }
    }

    /// Flash attention sequence length metadata.
    ///
    /// `cumulative_seqlens_q/k` use **padded** lengths (each sequence is padded to
    /// `max_len` in the batch). This matches the padded Q/K tensors in the normal
    /// prefill and decode paths.
    ///
    /// For the **prefix cache path**, K/V are gathered from the paged cache into a
    /// packed (non-padded) layout via `gather_kv_cache`. The packed K/V lengths are
    /// given by `PagedAttentionInputMetadata::cu_seqlens_kv`, NOT by
    /// `cumulative_seqlens_k` here. The prefix cache attention call must build a
    /// local `FlashParams` that swaps in `cu_seqlens_kv` for K.
    #[derive(Clone, Debug)]
    pub struct FlashParams {
        pub max_q: u32,
        pub max_k: u32,
        pub cumulative_seqlens_q: HashMap<DeviceLocation, Tensor>,
        pub cumulative_seqlens_k: HashMap<DeviceLocation, Tensor>,
        pub causal: bool,
    }

    pub struct InputMetadata {
        pub input: Tensor,
        pub positions: Vec<usize>,
        pub context_lens: Vec<(usize, usize)>, // (start index, len)
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>, // For paged attention
        pub flash_meta: FlashParams,
    }

    pub struct InnerInputProcessorOutput {
        pub inputs: InputMetadata,
        pub seq_indices: Vec<usize>,
    }

    /// The per-row absolute offsets a batch needs, or `None` when it shares
    /// one — which is every batch except the fused MTP step's left-aligned
    /// ragged cohort.
    ///
    /// 🔑 Returning `None` for a batch that shares an offset is a **dispatch,
    /// not an optimisation**: `None` sends [`make_prompt_chunk`] down the
    /// pre-change scalar code verbatim, so every prompt, every decode and every
    /// batch built with per-sequence KV advance off produces the characters it
    /// always did. Split out as its own function for the reason PR #100's
    /// `resolve_ragged_rows` was: the two branches agree exactly on their
    /// common domain, so no numeric test can tell you which one ran, and a
    /// mutation that routes a uniform batch down the per-row path is invisible
    /// to everything except a test of the dispatch itself.
    pub(crate) fn resolve_row_offsets(
        per_seq: &[Option<usize>],
        shared: usize,
    ) -> Option<Vec<usize>> {
        if !per_seq.iter().any(Option::is_some) {
            return None;
        }
        Some(per_seq.iter().map(|o| o.unwrap_or(shared)).collect())
    }

    // chunk_offset_toks is the number of tokens by which the tokens are offset,
    // chunk_offset_toks / prompt_chunksize = number of batches
    //
    // row_offsets: when provided, replaces `chunk_offset_toks` per sequence —
    // the batch is a left-aligned ragged cohort with no shared absolute
    // position (see `resolve_row_offsets`). `None` is the shared-offset path
    // and is bit-for-bit what it always was.
    //
    // prefix_cache_lens: when provided, indicates how many tokens per sequence are already
    // cached in the paged KV cache. Only new (non-cached) tokens will be included in the
    // input tensor, and slot_mappings will only cover new token slots. Block tables still
    // cover the entire context so that context_attention_fwd can read cached blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn make_prompt_chunk<T: WithDType + Debug>(
        chunk_offset_toks: usize,
        toks: Vec<&[T]>,
        seq_ids: &[usize],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
        prefix_cache_lens: Option<&[usize]>,
        row_offsets: Option<&[usize]>,
    ) -> Result<InputMetadata> {
        if let Some(offsets) = row_offsets {
            if offsets.len() != seq_ids.len() {
                anyhow::bail!(
                    "inputs processor: {} per-row absolute offsets for a batch of {}",
                    offsets.len(),
                    seq_ids.len()
                );
            }
        }
        // ── The prefill cursor is applied HERE, and only here ─────────────
        // Every path that builds prompt inputs reaches this function:
        // `get_prompt_input`, and — bypassing it entirely — `normal.rs` and
        // `vision.rs`, which call this directly. `NormalPipeline` is the second
        // of those, which is what DeepSeek-V4 is.
        //
        // Applying the cursor in a CALLER is how this was written first, and it
        // made the feature inert on exactly the path that matters: a GPU run
        // reported `prompt_tokens = 2892` at chunk 512 with **zero** chunks fed.
        // One place it can be wired means it cannot be wired in only one place.
        let toks: Vec<&[T]> = match prefill_chunk_size().filter(|_| !return_raw_logits) {
            Some(c) => toks
                .iter()
                .map(|t| {
                    let (start, end) = chunk_window(t.len(), chunk_offset_toks, c);
                    if start > 0 {
                        // D32 engagement, logged once: a run where chunking did
                        // nothing and a run where chunking was never on the code
                        // path are otherwise the same numbers.
                        static CHUNKED: std::sync::Once = std::sync::Once::new();
                        CHUNKED.call_once(|| {
                            tracing::info!(
                                "ARC prefill chunk: fed tokens [{start}, {end}) of a \
                                 {}-token prompt (ARC_PREFILL_CHUNK={c}); logged once",
                                t.len()
                            );
                        });
                    }
                    &t[start..end]
                })
                .collect(),
            None => toks,
        };

        // Determine effective tokens per sequence after prefix cache trimming
        let effective_lens: Vec<usize> = toks
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let cached = prefix_cache_lens.map_or(0, |lens| lens[i]);
                seq.len().saturating_sub(cached)
            })
            .collect();
        let max_len = *effective_lens.iter().max().expect("No sequences");
        let padding_tok = T::zero();
        // Pad each sequence by the padding token to the max len.
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let flash_attn = crate::using_flash_attn();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        let mut num_cached_tokens_vec: Vec<usize> = Vec::new();
        let mut query_lens_vec: Vec<usize> = Vec::new();
        let has_any_cache_hit = prefix_cache_lens.is_some_and(|lens| lens.iter().any(|&l| l > 0));
        for (seq_idx, (seq_id, ctxt)) in seq_ids.iter().zip(&toks).enumerate() {
            let cached = prefix_cache_lens.map_or(0, |lens| lens[seq_idx]);
            // This row's own absolute start, which is the shared
            // `chunk_offset_toks` for every batch that has one.
            let chunk_offset_toks = row_offsets.map_or(chunk_offset_toks, |o| o[seq_idx]);
            let full_prompt_len = ctxt.len();
            // The new (non-cached) tokens to process
            let new_toks = &ctxt[cached..];
            let new_len = new_toks.len();

            let offset = last_n_context_len.unwrap_or_default();
            // seqlen_offset includes cached prefix so position IDs are correct
            seqlen_offsets.push(offset.1 + chunk_offset_toks + cached);

            position_ids.push(new_len + chunk_offset_toks + cached);
            let mut padded = new_toks.to_vec();
            padded.extend(std::iter::repeat_n(
                padding_tok,
                max_len.saturating_sub(padded.len()),
            ));
            // If we are returning raw logits, we want to not trim the logits at all.
            if return_raw_logits {
                if last_n_context_len.is_some() {
                    anyhow::bail!("`return_raw_logits` is incompatible with `last_n_context_len`");
                }

                context_lens.push((0, padded.len()));
            } else {
                // 🔑 `new_len`, NOT `padded.len()`. `padded.len()` is the batch
                // max, so on a ragged batch every short row would read its
                // logits out of the right-hand PADDING — same wrong row for
                // every sequence. `extract_logits` (`pipeline/mod.rs`) already
                // narrows per row, so the per-row start is honoured.
                // Uniform batches are unaffected: there `new_len ==
                // padded.len()` for every row, so this is the identity.
                let n = last_n_context_len.map(|(a, _)| a).unwrap_or(1);
                context_lens.push((new_len.saturating_sub(n), n));
            }

            if flash_attn {
                // Padded lengths — see FlashParams doc comment for prefix cache nuance.
                seqlens_q.push(padded.len() as u32);
                seqlens_k.push((padded.len() + chunk_offset_toks + cached) as u32);
            }

            seqs_tensors.push(Tensor::new(padded, device).unwrap().unsqueeze(0).unwrap());

            if has_any_cache_hit {
                num_cached_tokens_vec.push(cached);
                query_lens_vec.push(new_len);
            }

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
                let block_ids = kv_mgr.get_block_ids(*seq_id);

                if block_ids.is_none() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(new_len));
                    continue;
                }
                let table: Vec<usize> = block_ids.unwrap().to_vec();
                drop(kv_mgr);

                // Block table covers the full context (cached + new)
                let table_for_seq = table.clone();

                // Slot mappings only for new tokens (cached tokens are already in cache)
                let slot_start = cached + chunk_offset_toks;
                let slot_end = full_prompt_len + chunk_offset_toks;
                let mut slot_mapping = Vec::new();
                let mut ctxt_len = Vec::new();
                for i in slot_start..slot_end {
                    ctxt_len.push(i);

                    let block_number = if i / paged_attn_metadata.block_size >= table.len() {
                        panic!(
                            "Block table is too small (prompt)! i={} block_size={} table_len={}",
                            i,
                            paged_attn_metadata.block_size,
                            table.len()
                        );
                    } else {
                        table.get(i / paged_attn_metadata.block_size).unwrap()
                    };
                    let block_offset = i % paged_attn_metadata.block_size;
                    // Use checked arithmetic to prevent overflow
                    let slot = block_number
                        .checked_mul(paged_attn_metadata.block_size)
                        .and_then(|v| v.checked_add(block_offset))
                        .expect("Slot calculation overflowed");
                    slot_mapping.push(
                        slot.try_into()
                            .expect("Slot value too large for target integer type"),
                    );
                }
                slot_mappings.push(slot_mapping);
                paged_attn_context_lens.push(ctxt_len);
                block_tables.push(table_for_seq);
            }
        }

        let (max_q, max_k, seqlens_q_map, seqlens_k_map) = if flash_attn {
            // SAFETY: seqlens_q/k are initialized with vec![0] when flash_attn is true,
            // so they are guaranteed to be non-empty here.
            let max_q = *seqlens_q
                .iter()
                .max()
                .expect("seqlens_q should not be empty when flash_attn is enabled");
            let max_k = *seqlens_k
                .iter()
                .max()
                .expect("seqlens_k should not be empty when flash_attn is enabled");
            // Create tensors on CPU first to avoid CUDA context issues when copying
            // between different GPU devices. Each GPU has its own CUDA context, and
            // candle/cudarc doesn't properly switch contexts when doing GPU-to-GPU
            // transfers (which go through CPU). By creating on CPU first, we avoid
            // the cross-context memory access that causes CUDA_ERROR_INVALID_VALUE.
            let seqlens_q = Tensor::new(seqlens_q, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;
            let seqlens_k = Tensor::new(seqlens_k, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;

            let mut seqlens_q_map = HashMap::new();
            let mut seqlens_k_map = HashMap::new();

            let devices = mapper.unwrap().get_unique_devices();
            for device in devices {
                seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
                seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
            }
            (max_q, max_k, seqlens_q_map, seqlens_k_map)
        } else {
            (0, 0, HashMap::new(), HashMap::new())
        };

        let input = Tensor::cat(&seqs_tensors, 0).unwrap();

        let paged_attn_meta = if paged_attn_metadata.is_some() {
            // Create paged attention tensors on CPU first (see comment above about CUDA contexts)
            let max_slot_mapping_len = slot_mappings.iter().map(|x| x.len()).max().unwrap();
            let slot_mappings = _make_tensor_with_pad(
                slot_mappings,
                max_slot_mapping_len,
                _PAD_SLOT_ID,
                &Device::Cpu,
            )?;

            let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                &Device::Cpu,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens
                .iter()
                .map(|x| x.len())
                .max()
                .unwrap();

            let context_lens = _make_tensor_with_pad(
                paged_attn_context_lens
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_context_len,
                0,
                &Device::Cpu,
            )?
            .reshape(((),))?;

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(max_context_len),
                full_block_tables: None,
                full_context_lens: None,
                full_max_context_len: None,
                is_first_prompt_chunk: chunk_offset_toks == 0,
                paged_kv_indptr: None,
                paged_kv_indices: None,
                paged_kv_last_page_len: None,
                paged_kv_request_indices: None,
                paged_kv_tile_indices: None,
                paged_kv_o_indptr: None,
                paged_kv_chunk_size: None,
                num_cached_tokens: if has_any_cache_hit {
                    Some(num_cached_tokens_vec.clone())
                } else {
                    None
                },
                query_lens: if has_any_cache_hit {
                    Some(query_lens_vec.clone())
                } else {
                    None
                },
                cu_seqlens_q: if has_any_cache_hit {
                    // Cumulative query lengths for Sdpa varlen: [0, q0, q0+q1, ...]
                    let mut cu_q = vec![0u32];
                    for &ql in &query_lens_vec {
                        cu_q.push(cu_q.last().unwrap() + ql as u32);
                    }
                    let cu_q_t = Tensor::new(&cu_q[..], &Device::Cpu)?;
                    let devices = mapper.unwrap().get_unique_devices();
                    let mut map = HashMap::new();
                    for device in &devices {
                        map.insert(device.location(), cu_q_t.to_device(device)?);
                    }
                    Some(map)
                } else {
                    None
                },
                cu_seqlens_kv: if has_any_cache_hit {
                    // Cumulative KV lengths: [0, c0+q0, c0+q0+c1+q1, ...]
                    // U32 to match flash-attn varlen expectations
                    let mut cu_kv = vec![0u32];
                    for (&nc, &ql) in num_cached_tokens_vec.iter().zip(query_lens_vec.iter()) {
                        cu_kv.push(cu_kv.last().unwrap() + (nc + ql) as u32);
                    }
                    let cu_kv_t = Tensor::new(&cu_kv[..], &Device::Cpu)?;
                    let devices = mapper.unwrap().get_unique_devices();
                    let mut map = HashMap::new();
                    for device in &devices {
                        map.insert(device.location(), cu_kv_t.to_device(device)?);
                    }
                    Some(map)
                } else {
                    None
                },
            })
        } else {
            None
        };

        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
                causal: true,
            },
        })
    }

    fn make_completion_chunk<T: WithDType>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputMetadata> {
        // Pad each sequence by the padding token to the max len.
        let flash_attn = crate::using_flash_attn();
        let mut seqs_tensors = Vec::new();
        let mut seqlen_offsets = Vec::new();
        let mut context_lens = Vec::new();
        let mut position_ids = Vec::new();

        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        let mut paged_attn_context_lens = Vec::new();
        let mut full_block_tables = Vec::new();
        let mut full_paged_attn_context_lens = Vec::new();
        let mut seqlens_q = if flash_attn { vec![0] } else { Vec::new() };
        let mut seqlens_k = if flash_attn { vec![0] } else { Vec::new() };
        for (seq, ctxt) in input_seqs.iter().zip(toks) {
            let start_pos = ctxt.len().saturating_sub(1);
            let ctxt = ctxt[start_pos..].to_vec();
            seqlen_offsets.push(start_pos);
            context_lens.push((0, 1));
            position_ids.push(seq.len());

            if flash_attn {
                seqlens_q.push(ctxt.len() as u32);
                seqlens_k.push((ctxt.len() + start_pos) as u32);
            }

            // CLAUDE.md pitfall #5, still live: one `Tensor::new` on the GPU
            // device per sequence per decode step, i.e. B separate 1-element
            // H2D transfers, each of which is a host/device round trip.
            {
                let _s = arc_profiler::sync_span("input_prep.h2d_per_seq");
                seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
            }

            if let Some(paged_attn_metadata) = &mut paged_attn_metadata {
                let kv_mgr = get_mut_arcmutex!(paged_attn_metadata.kv_cache_manager);
                let table: Vec<usize> = kv_mgr
                    .get_block_ids(*seq.id())
                    .expect("Sequence must have allocated blocks for completion")
                    .to_vec();
                drop(kv_mgr);

                let block_pos = start_pos - seq.token_offset();
                let block_number = if block_pos / paged_attn_metadata.block_size >= table.len() {
                    panic!("Block table is too small (completion)! start_pos={} block_size={} table_len={}", block_pos, paged_attn_metadata.block_size, table.len());
                } else {
                    table
                        .get(block_pos / paged_attn_metadata.block_size)
                        .unwrap()
                };
                let block_offset = block_pos % paged_attn_metadata.block_size;
                // Use checked arithmetic to prevent overflow
                let slot = block_number
                    .checked_mul(paged_attn_metadata.block_size)
                    .and_then(|v| v.checked_add(block_offset))
                    .expect("Slot calculation overflowed");
                let slot = slot
                    .try_into()
                    .expect("Slot value too large for target integer type");
                slot_mappings.push(vec![slot]);

                // Always collect the full (unwindowed) block tables.
                full_block_tables.push(table.clone());
                full_paged_attn_context_lens.push(seq.len());

                if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                    let window_start = seq.len().saturating_sub(sliding_window);
                    let slide_idx = window_start / paged_attn_metadata.block_size;
                    block_tables.push(table.get(slide_idx..).unwrap().to_vec());
                } else {
                    block_tables.push(table);
                }

                let paged_attn_context_len =
                    if let Some(sliding_window) = paged_attn_metadata.sliding_window {
                        let window_start = seq.len().saturating_sub(sliding_window);
                        let block_aligned_start = (window_start / paged_attn_metadata.block_size)
                            * paged_attn_metadata.block_size;
                        seq.len() - block_aligned_start
                    } else {
                        seq.len()
                    };
                paged_attn_context_lens.push(paged_attn_context_len);
            }
        }

        let (max_q, max_k, seqlens_q_map, seqlens_k_map) = if flash_attn {
            // SAFETY: seqlens_q/k are initialized with vec![0] when flash_attn is true,
            // so they are guaranteed to be non-empty here.
            let max_q = *seqlens_q
                .iter()
                .max()
                .expect("seqlens_q should not be empty when flash_attn is enabled");
            let max_k = *seqlens_k
                .iter()
                .max()
                .expect("seqlens_k should not be empty when flash_attn is enabled");
            // Create tensors on CPU first to avoid CUDA context issues (see make_prompt_chunk)
            let seqlens_q = Tensor::new(seqlens_q, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;
            let seqlens_k = Tensor::new(seqlens_k, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(DType::U32)?;

            let mut seqlens_q_map = HashMap::new();
            let mut seqlens_k_map = HashMap::new();

            let devices = mapper.unwrap().get_unique_devices();
            for device in devices {
                seqlens_q_map.insert(device.location(), seqlens_q.to_device(&device)?);
                seqlens_k_map.insert(device.location(), seqlens_k.to_device(&device)?);
            }
            (max_q, max_k, seqlens_q_map, seqlens_k_map)
        } else {
            (0, 0, HashMap::new(), HashMap::new())
        };

        let paged_attn_meta = if let Some(paged_attn_input) = &paged_attn_metadata {
            // Create paged attention tensors on CPU first (see make_prompt_chunk for explanation)
            let slot_mappings =
                _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, &Device::Cpu)?;

            let max_block_table_len = block_tables
                .iter()
                .map(|x| x.len())
                .max()
                .expect("block_tables should not be empty when paged attention is enabled");

            let batch_size = block_tables.len();
            let mut paged_kv_indices = Vec::new();
            let mut paged_kv_indptr = Vec::with_capacity(batch_size + 1);
            let mut paged_kv_last_page_len = Vec::with_capacity(batch_size);
            paged_kv_indptr.push(0i32);
            let mut nnz_pages = 0i32;
            let block_size = paged_attn_input.block_size;
            for (table, context_len) in block_tables.iter().zip(paged_attn_context_lens.iter()) {
                let num_blocks = table.len();
                nnz_pages += num_blocks as i32;
                paged_kv_indptr.push(nnz_pages);
                paged_kv_indices.extend(table.iter().map(|x| *x as i32));
                let last_page_len = if num_blocks == 0 {
                    0usize
                } else {
                    let consumed = (num_blocks - 1) * block_size;
                    if *context_len < consumed {
                        panic!(
                            "paged kv context len underflow: context_len={} consumed={}",
                            context_len, consumed
                        );
                    }
                    *context_len - consumed
                };
                paged_kv_last_page_len.push(last_page_len as i32);
            }

            let request_indices: Vec<i32> = (0..batch_size as i32).collect();
            let kv_tile_indices = vec![0i32; batch_size];
            let o_indptr: Vec<i32> = (0..=batch_size as i32).collect();
            let kv_chunk_size = vec![block_size as i32];

            let block_tables = _make_tensor_with_pad(
                block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                max_block_table_len,
                0,
                &Device::Cpu,
            )?;
            let block_tables = block_tables.reshape(((), max_block_table_len))?;

            let max_context_len = paged_attn_context_lens.iter().max().unwrap();

            let context_lens = Tensor::from_vec(
                paged_attn_context_lens
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                (paged_attn_context_lens.len(),),
                &Device::Cpu,
            )?;

            let paged_kv_indptr =
                Tensor::from_vec(paged_kv_indptr, (batch_size + 1,), &Device::Cpu)?;
            let paged_kv_indices =
                Tensor::from_vec(paged_kv_indices, (nnz_pages as usize,), &Device::Cpu)?;
            let paged_kv_last_page_len =
                Tensor::from_vec(paged_kv_last_page_len, (batch_size,), &Device::Cpu)?;
            let request_indices = Tensor::from_vec(request_indices, (batch_size,), &Device::Cpu)?;
            let kv_tile_indices = Tensor::from_vec(kv_tile_indices, (batch_size,), &Device::Cpu)?;
            let o_indptr = Tensor::from_vec(o_indptr, (batch_size + 1,), &Device::Cpu)?;
            let kv_chunk_size = Tensor::from_vec(kv_chunk_size, (1,), &Device::Cpu)?;

            // Build full (unwindowed) block tables and context lens.
            let full_max_block_table_len =
                full_block_tables.iter().map(|x| x.len()).max().unwrap_or(0);

            let full_block_tables_tensor = _make_tensor_with_pad(
                full_block_tables
                    .iter()
                    .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                full_max_block_table_len.max(1),
                0,
                &Device::Cpu,
            )?;
            let full_block_tables_tensor =
                full_block_tables_tensor.reshape(((), full_max_block_table_len.max(1)))?;

            let full_max_context_len = full_paged_attn_context_lens
                .iter()
                .max()
                .copied()
                .unwrap_or(0);

            let full_context_lens_tensor = Tensor::from_vec(
                full_paged_attn_context_lens
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>(),
                (full_paged_attn_context_lens.len(),),
                &Device::Cpu,
            )?;

            // For device mapping, make a copy of each tensor for each device
            let devices = mapper.unwrap().get_unique_devices();
            let mut slot_mappings_map = HashMap::new();
            let mut block_tables_map = HashMap::new();
            let mut context_lens_map = HashMap::new();
            let mut full_block_tables_map = HashMap::new();
            let mut full_context_lens_map = HashMap::new();
            let mut paged_kv_indptr_map = HashMap::new();
            let mut paged_kv_indices_map = HashMap::new();
            let mut paged_kv_last_page_len_map = HashMap::new();
            let mut paged_kv_request_indices_map = HashMap::new();
            let mut paged_kv_tile_indices_map = HashMap::new();
            let mut paged_kv_o_indptr_map = HashMap::new();
            let mut paged_kv_chunk_size_map = HashMap::new();

            for device in devices {
                slot_mappings_map
                    .insert(device.location(), slot_mappings.clone().to_device(&device)?);
                block_tables_map
                    .insert(device.location(), block_tables.clone().to_device(&device)?);
                context_lens_map
                    .insert(device.location(), context_lens.clone().to_device(&device)?);
                full_block_tables_map.insert(
                    device.location(),
                    full_block_tables_tensor.clone().to_device(&device)?,
                );
                full_context_lens_map.insert(
                    device.location(),
                    full_context_lens_tensor.clone().to_device(&device)?,
                );
                paged_kv_indptr_map.insert(
                    device.location(),
                    paged_kv_indptr.clone().to_device(&device)?,
                );
                paged_kv_indices_map.insert(
                    device.location(),
                    paged_kv_indices.clone().to_device(&device)?,
                );
                paged_kv_last_page_len_map.insert(
                    device.location(),
                    paged_kv_last_page_len.clone().to_device(&device)?,
                );
                paged_kv_request_indices_map.insert(
                    device.location(),
                    request_indices.clone().to_device(&device)?,
                );
                paged_kv_tile_indices_map.insert(
                    device.location(),
                    kv_tile_indices.clone().to_device(&device)?,
                );
                paged_kv_o_indptr_map
                    .insert(device.location(), o_indptr.clone().to_device(&device)?);
                paged_kv_chunk_size_map
                    .insert(device.location(), kv_chunk_size.clone().to_device(&device)?);
            }

            Some(PagedAttentionInputMetadata {
                slot_mappings: slot_mappings_map,
                block_tables: Some(block_tables_map),
                context_lens: Some(context_lens_map),
                max_context_len: Some(*max_context_len),
                full_block_tables: Some(full_block_tables_map),
                full_context_lens: Some(full_context_lens_map),
                full_max_context_len: Some(full_max_context_len),
                is_first_prompt_chunk: false,
                paged_kv_indptr: Some(paged_kv_indptr_map),
                paged_kv_indices: Some(paged_kv_indices_map),
                paged_kv_last_page_len: Some(paged_kv_last_page_len_map),
                paged_kv_request_indices: Some(paged_kv_request_indices_map),
                paged_kv_tile_indices: Some(paged_kv_tile_indices_map),
                paged_kv_o_indptr: Some(paged_kv_o_indptr_map),
                paged_kv_chunk_size: Some(paged_kv_chunk_size_map),
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
            })
        } else {
            None
        };

        let input = {
            let _s = arc_profiler::span("input_prep.cat");
            Tensor::cat(&seqs_tensors, 0).unwrap()
        };
        Ok(InputMetadata {
            input,
            positions: seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta: FlashParams {
                max_k,
                max_q,
                cumulative_seqlens_k: seqlens_k_map,
                cumulative_seqlens_q: seqlens_q_map,
                causal: true,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Tokens of each prompt fed per engine iteration, or `None` for "all of
    /// it" — the historical behaviour, and where head-of-line blocking comes
    /// from.
    ///
    /// Read once.
    ///
    /// # 🔴 DO NOT SET THIS YET. Sizing guidance from other systems does not
    /// # transfer, and following it makes Arc's prefill several times worse.
    ///
    /// An earlier revision of this comment carried Sarathi-Serve's numbers
    /// verbatim — *"512 is the practical floor, 2048 near-free"*. Those come
    /// from a system whose per-prompt-token cost is roughly **30× lower than
    /// ours**, so they are not advice here, they are a trap. Arc's own
    /// arithmetic:
    ///
    /// **Chunking does not reduce FLOPs.** It is latency shaping only: it
    /// splits one uninterruptible prefill into `ceil(N/C)` interruptible ones
    /// so decode can run between them. Whatever a prefill step charges *per
    /// step* gets multiplied by `ceil(N/C)`.
    ///
    /// **Arc's prefill step is dominated by a per-step charge.** The QTIP MoE
    /// expert gather is **71.3% of an N=128 prefill step** (profiler and nsys
    /// agreeing to 0.2%; `memory/mission/BUDGET_V4_PREFILL.md`), and it is
    /// billed per step because each step re-reads the packed expert weights
    /// regardless of how many tokens it is serving. Chunking therefore pays it
    /// `ceil(N/C)` times. At the current measured **2.880 ms/prompt-token**
    /// (N=2048, post-PR #133 + #138), a 2048-token prompt at `C=512` is 4
    /// steps: **~4.4 s of expert gather against ~1.1 s unchunked.**
    ///
    /// **And small shapes are already worse per token, not better.** Measured
    /// ms/prompt-token by prompt length: **11.98 (N=128) · 11.48 (512) ·
    /// 11.81 (1024) · 8.23 (2048)** — i.e. N=128 costs **~1.46× more per
    /// token** than N=2048. There is no efficiency floor to fall back onto;
    /// the curve points the wrong way.
    ///
    /// ⇒ **The gate: chunking is NEGATIVE until the expert gather is fixed.**
    /// Leave this unset. It exists so the cursor plumbing is exercised and
    /// ready, not because a value is currently worth setting.
    ///
    /// # When it does become affordable
    ///
    /// Size it by *chunk wall time*, not by copying a token count. SGLang runs
    /// 8192 at ~0.0975 ms/prompt-token — 29.5× cheaper than Arc — so Arc's
    /// equal-wall-time equivalent of their 8192 is **~277 tokens**, not 8192.
    ///
    /// Two hard constraints, both verified against sglang `main`
    /// (`python/sglang/srt/server_args.py`, fetched 2026-08-19):
    ///
    /// 1. **`C` must be a multiple of the KV block size.** Otherwise every
    ///    chunk boundary lands mid-block and the trailing partial block is
    ///    never hashed, so it cannot be prefix-cached — see
    ///    `paged_attention/block_hash.rs::compute_block_hashes`, which hashes
    ///    only `tokens.len() / block_size` full blocks. SGLang enforces the
    ///    same rule outright: `assert chunked_prefill_size % page_size == 0`.
    /// 2. **Their number is a MoE parameter, not a scheduler knob.**
    ///    `chunked_prefill_size` is what sizes their MoE all-to-all dispatch
    ///    buffer — it is returned verbatim by
    ///    `_required_mori_dispatch_tokens_per_rank` ("max tokens a single rank
    ///    dispatches through MoRI in one forward") and
    ///    `_required_pplx_dispatch_tokens_per_rank`. Reading 8192 as "a good
    ///    scheduler chunk" misreads what the number is for.
    pub(crate) fn prefill_chunk_size() -> Option<usize> {
        // A test override, when one is installed on THIS thread, wins outright
        // and never touches the process environment. See `PrefillChunkGuard`
        // for why the obvious `set_var` version cannot work here.
        #[cfg(test)]
        if let Some(injected) = test_chunk_override() {
            return injected;
        }
        static C: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *C.get_or_init(|| {
            std::env::var("ARC_PREFILL_CHUNK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
        })
    }

    #[cfg(test)]
    thread_local! {
        /// `None` = no override on this thread; `Some(v)` = force `v`.
        static TEST_CHUNK: std::cell::Cell<Option<Option<usize>>> =
            const { std::cell::Cell::new(None) };
    }

    #[cfg(test)]
    fn test_chunk_override() -> Option<Option<usize>> {
        TEST_CHUNK.with(|c| c.get())
    }

    /// Injects a prefill chunk size for the duration of one test, and removes it
    /// again on drop.
    ///
    /// # Why not `std::env::set_var`
    ///
    /// The first version of the test below did exactly that, and it is unsound
    /// **twice over**:
    ///
    /// 1. **Tests share one process.** libtest runs them on parallel threads of
    ///    a single process, and the environment is process-global, so a bare
    ///    `set_var` leaks into whatever sibling happens to be running. That is
    ///    the contamination that produced the two Windows failures; passing on
    ///    Linux was scheduling luck, not correctness.
    ///
    /// 2. **A mutex would not have been enough, because of the `OnceLock`
    ///    above.** `prefill_chunk_size` caches its answer process-wide on first
    ///    call. So the paired `remove_var` never undoes anything: whichever
    ///    value was read first is latched for the rest of the binary. If any
    ///    other test reached the function first, this test sees `None` and
    ///    fails; if this test got there first, *every later test in the process*
    ///    silently runs with `chunk = 4`. Serialising the mutation only reorders
    ///    that race — it cannot fix it.
    ///
    /// So the value is injected instead of read from the environment. The
    /// override is a `thread_local`, which gives per-test isolation with no lock
    /// at all, and the reset is `Drop` rather than a trailing statement so it
    /// still happens when an assertion panics — the original `remove_var` sat
    /// *after* the `assert_eq!` and would have been skipped on failure, leaving
    /// the contamination behind precisely when things were already going wrong.
    /// `Drop` also survives `--test-threads=1`, where libtest reuses one thread
    /// and a bare thread-local set would leak to the next test.
    #[cfg(test)]
    pub(crate) struct PrefillChunkGuard(Option<Option<usize>>);

    #[cfg(test)]
    impl PrefillChunkGuard {
        pub(crate) fn set(chunk: Option<usize>) -> Self {
            let previous = TEST_CHUNK.with(|c| c.replace(Some(chunk)));
            Self(previous)
        }
    }

    #[cfg(test)]
    impl Drop for PrefillChunkGuard {
        fn drop(&mut self) {
            let previous = self.0;
            TEST_CHUNK.with(|c| c.set(previous));
        }
    }

    /// The `[start, end)` slice of a prompt of length `len` that chunk-cursor
    /// `offset` selects, given chunk size `chunk`.
    ///
    /// Clamped at both ends: a cohort shares one cursor, so a shorter prompt in
    /// a mixed batch runs out of tokens before its neighbours and must yield an
    /// empty window rather than panic on the slice.
    pub(crate) fn chunk_window(len: usize, offset: usize, chunk: usize) -> (usize, usize) {
        let start = offset.min(len);
        let end = offset.saturating_add(chunk).min(len);
        (start, end.max(start))
    }

    pub(crate) fn get_prompt_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InnerInputProcessorOutput> {
        let offset = input_seqs[0].token_offset();
        // ── Chunked prefill ───────────────────────────────────────────────
        // `offset` is this cohort's prefill cursor. With chunking on, only
        // `[offset, offset + chunk)` of each prompt is fed this step; the rest
        // is fed by later steps, and `offset` is what makes the positions,
        // RoPE offsets and `cu_seqlens` come out right for every chunk after
        // the first.
        //
        // 🔴 THE THING THAT WILL BITE WHOEVER TOUCHES THIS. On CUDA with
        // flash-attn, `CausalMasker::make_causal_mask_matrix` returns a **1x1
        // dummy tensor** — no real mask is ever built (`layers_masker.rs`,
        // "Avoid materializing large sliding-window masks"). So a chunk's
        // causality rests ENTIRELY on the flash kernel reading
        // `cumulative_seqlens_q` (chunk width) against `cumulative_seqlens_k`
        // (chunk width + offset) and aligning the causal diagonal to the
        // BOTTOM-RIGHT of that rectangle, which is FlashAttention >= 2.1
        // semantics. A top-left alignment silently lets every query in chunk
        // k>0 attend nothing but its own chunk, or attend the future — wrong
        // logits, no error, and only the second chunk onward is affected.
        //
        // `make_prompt_chunk` already builds exactly that pair, so the wiring
        // is correct by construction; what cannot be proven from here is that
        // the kernel honours it. That is what the one-shot-vs-chunked token
        // identity test exists for, and it is the acceptance gate for this
        // feature — not a nice-to-have.
        // A left-aligned ragged cohort has no shared absolute position, so each
        // row carries its own. `None` — every other batch — keeps `offset`.
        let row_offsets = resolve_row_offsets(
            &input_seqs
                .iter()
                .map(|s| s.prefill_seqlen_offset())
                .collect::<Vec<_>>(),
            offset,
        );
        // Collect prefix cache lens when paged attention is in use
        let prefix_cache_lens: Vec<usize> =
            input_seqs.iter().map(|s| s.prefix_cache_len()).collect();
        let has_paged_attn = paged_attn_metadata.is_some();
        make_prompt_chunk(
            offset,
            toks,
            &input_seqs.iter().map(|s| *s.id()).collect::<Vec<_>>(),
            device,
            last_n_context_len,
            return_raw_logits,
            paged_attn_metadata,
            mapper,
            if has_paged_attn {
                Some(&prefix_cache_lens)
            } else {
                None
            },
            row_offsets.as_deref(),
        )
        .map(|inputs| InnerInputProcessorOutput {
            inputs,
            seq_indices: (0..input_seqs.len()).collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_completion_input<T: WithDType + std::fmt::Debug>(
        toks: Vec<&[T]>,
        input_seqs: &[&mut Sequence],
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InnerInputProcessorOutput> {
        if no_kv_cache {
            return get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata,
                mapper,
            );
        }

        make_completion_chunk(toks, input_seqs, device, paged_attn_metadata, mapper).map(|inputs| {
            InnerInputProcessorOutput {
                inputs,
                seq_indices: (0..input_seqs.len()).collect(),
            }
        })
    }

    #[derive(Clone)]
    pub struct ModelInputs {
        pub input_ids: Tensor,
        /// Decode-only side channel: the same `input_ids` data on the host.
        /// Populated by `TextInputsProcessor` for completion (decode) calls so the
        /// dedicated decode path can stage tokens to GPU on its own non-blocking
        /// stream without forcing a D2H sync via `to_vec1::<u32>()`.
        pub input_ids_cpu: Option<Vec<u32>>,
        pub input_ids_full: Option<Tensor>,
        pub seqlen_offsets: Vec<usize>,
        pub seqlen_offsets_full: Option<Vec<usize>>,
        pub context_lens: Vec<(usize, usize)>,
        pub position_ids: Vec<usize>,
        pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
        pub flash_meta: FlashParams,
        pub flash_meta_full: Option<FlashParams>,
    }

    pub struct TextInputsProcessor;

    impl InputsProcessor for TextInputsProcessor {
        fn process_inputs(
            &self,
            _: Option<Arc<Tokenizer>>,
            input_seqs: &mut [&mut Sequence],
            is_prompt: bool,
            is_xlora: bool,
            device: &Device,
            no_kv_cache: bool,
            last_n_context_len: Option<(usize, usize)>,
            return_raw_logits: bool,
            _: Option<Arc<dyn Any>>,
            mut paged_attn_metadata: Option<PagedAttentionMeta>,
            mapper: Option<&dyn DeviceMapper>,
        ) -> Result<InputProcessorOutput> {
            if is_xlora && !is_prompt {
                let prompt = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let completion = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids_full,
                            positions: seqlen_offsets_full,
                            context_lens: _,
                            position_ids,
                            paged_attn_meta: _,
                            flash_meta: flash_meta_full,
                        },
                    seq_indices,
                } = prompt;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids: _,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices: _,
                } = completion;
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_cpu: None,
                    input_ids_full: Some(input_ids_full),
                    seqlen_offsets,
                    seqlen_offsets_full: Some(seqlen_offsets_full),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: Some(flash_meta_full),
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_xlora && is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids: input_ids.clone(),
                    input_ids_cpu: None,
                    input_ids_full: Some(input_ids),
                    seqlen_offsets: seqlen_offsets.clone(),
                    seqlen_offsets_full: Some(seqlen_offsets),
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta: flash_meta.clone(),
                    flash_meta_full: Some(flash_meta),
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else if is_prompt {
                let metadata = get_prompt_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_cpu: None,
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            } else {
                // Decode (completion) path: collect the last token per sequence
                // into a Vec<u32> so the dedicated decode path can stage to GPU
                // without paying for a D2H sync via to_vec1::<u32>().
                let decode_input_cpu: Vec<u32> = input_seqs
                    .iter()
                    .map(|seq| {
                        let toks = seq.get_toks();
                        *toks.last().unwrap_or(&0)
                    })
                    .collect();
                let metadata = get_completion_input(
                    input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    input_seqs,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    paged_attn_metadata.as_mut(),
                    mapper,
                )?;
                let InnerInputProcessorOutput {
                    inputs:
                        InputMetadata {
                            input: input_ids,
                            positions: seqlen_offsets,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                        },
                    seq_indices,
                } = metadata;
                let inputs: Box<dyn Any> = Box::new(ModelInputs {
                    input_ids,
                    input_ids_cpu: Some(decode_input_cpu),
                    input_ids_full: None,
                    seqlen_offsets,
                    seqlen_offsets_full: None,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                    flash_meta_full: None,
                });
                Ok(InputProcessorOutput {
                    inputs,
                    seq_indices,
                })
            }
        }

        fn get_type(&self) -> InputsProcessorType {
            InputsProcessorType::Text
        }
    }

    #[cfg(test)]
    mod ragged_prefill_tests {
        use super::*;

        /// 🔑 A right-padded ragged prefill must read each row's logits from
        /// that row's own LAST REAL TOKEN.
        ///
        /// `make_prompt_chunk` pads every row to the batch max, so a shared
        /// logit index is the PADDING for every row but the longest. Pre-fix
        /// this returns `(max_len - 1, 1)` for all three rows — the same wrong
        /// column three times. `extract_logits` (`pipeline/mod.rs`) narrows per
        /// row, so honouring the per-row start needs nothing else.
        ///
        /// Reachable in production the moment a prefill batch is ragged. The
        /// schedulers' exact-cache-length bucketing normally makes
        /// `max_len == this row's len`, but a prefix-cache hit defeats that:
        /// the hit moves the bucket key onto the matched length, so two
        /// requests behind the same system prelude with different user-message
        /// lengths share one ragged prefill — and every row but the longest
        /// then samples its first token from a pad column.
        #[test]
        fn ragged_prefill_reads_each_row_at_its_own_last_real_token() {
            let a: Vec<u32> = vec![1; 5];
            let b: Vec<u32> = vec![2; 3];
            let c: Vec<u32> = vec![3; 8];
            let toks: Vec<&[u32]> = vec![&a, &b, &c];

            let out = make_prompt_chunk::<u32>(
                0,
                toks,
                &[0, 1, 2],
                &Device::Cpu,
                None,
                false,
                None,
                None,
                None,
                // `row_offsets` — master gained this parameter after this test
                // was written. `None` is what this test means: it exercises the
                // per-row `context_lens` narrowing, not per-row RoPE placement.
                None,
            )
            .expect("CPU prompt chunk must build");

            // Teeth: the fixture must actually be ragged, or this asserts
            // nothing. A batch of equal lengths passes either way.
            assert_eq!(
                out.input.dims(),
                &[3, 8],
                "fixture must right-pad to the batch max (8), else the test is vacuous"
            );

            assert_eq!(
                out.context_lens,
                vec![(4, 1), (2, 1), (7, 1)],
                "each row must be narrowed at its own last real token \
                 (5-1, 3-1, 8-1), not at the shared padded width"
            );
        }

        /// The uniform case is the identity — the fix cannot move a batch that
        /// the old bucketing would have produced.
        #[test]
        fn uniform_prefill_is_unchanged() {
            let a: Vec<u32> = vec![1; 6];
            let b: Vec<u32> = vec![2; 6];
            let toks: Vec<&[u32]> = vec![&a, &b];
            let out = make_prompt_chunk::<u32>(
                0,
                toks,
                &[0, 1],
                &Device::Cpu,
                None,
                false,
                None,
                None,
                None,
                // `row_offsets` — see the sibling test above.
                None,
            )
            .expect("CPU prompt chunk must build");
            assert_eq!(out.context_lens, vec![(5, 1), (5, 1)]);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{make_prompt_chunk, resolve_row_offsets};
        use crate::device_map::DummyDeviceMapper;
        use candle_core::Device;

        /// 🔑 The wire itself: what the model actually receives. `positions` is
        /// `seqlen_offsets`, which is where RoPE places each row's queries and —
        /// on DeepSeek V4 — the per-row `row_q0` that masks a left-aligned
        /// cohort. A per-row vector in must produce a per-row vector out.
        #[test]
        fn per_row_offsets_reach_seqlen_offsets_and_a_shared_one_still_does_not() {
            let mapper = DummyDeviceMapper {
                nm_device: Device::Cpu,
            };
            let window: Vec<u32> = vec![7, 8, 9, 10];
            let toks: Vec<&[u32]> = vec![&window, &window, &window];
            let run = |row_offsets: Option<&[usize]>, shared: usize| -> Vec<usize> {
                make_prompt_chunk(
                    shared,
                    toks.clone(),
                    &[0, 1, 2],
                    &Device::Cpu,
                    Some((window.len(), 0)),
                    false,
                    None,
                    Some(&mapper),
                    None,
                    row_offsets,
                )
                .unwrap()
                .positions
            };

            assert_eq!(
                run(None, 40),
                vec![40, 40, 40],
                "with no per-row offsets the batch shares one, exactly as it always did"
            );
            assert_eq!(
                run(Some(&[40, 33, 37]), 40),
                vec![40, 33, 37],
                "each row's own absolute position must survive to `seqlen_offsets`"
            );
        }

        /// A per-row vector that does not describe this batch is a caller bug
        /// that would silently mis-place two rows' queries. It refuses.
        #[test]
        fn a_per_row_offset_vector_of_the_wrong_width_is_refused() {
            let mapper = DummyDeviceMapper {
                nm_device: Device::Cpu,
            };
            let window: Vec<u32> = vec![7, 8];
            let err = make_prompt_chunk(
                0,
                vec![&window, &window, &window],
                &[0, 1, 2],
                &Device::Cpu,
                Some((2, 0)),
                false,
                None,
                Some(&mapper),
                None,
                Some(&[1, 2]),
            )
            .err()
            .expect("a per-row vector that does not describe this batch must be refused")
            .to_string();
            assert!(
                err.contains("per-row absolute offsets") && err.contains("batch of 3"),
                "the refusal must name both widths; got {err}"
            );
        }

        /// 🔑 The dispatch, pinned directly. Every batch that has ever existed
        /// carries no per-row override, and must return `None` — because `None`
        /// is what runs `make_prompt_chunk`'s pre-change scalar code verbatim.
        ///
        /// ⚠️ Routing a batch that happens to agree down the per-row path is
        /// **numerically invisible** (the vector would be the shared offset
        /// repeated), so only a test of the dispatch itself can see that
        /// mutation. Same trap PR #100 documented for `resolve_ragged_rows`.
        #[test]
        fn a_batch_with_no_per_row_offset_takes_the_shared_path() {
            assert_eq!(resolve_row_offsets(&[None, None, None], 17), None);
            assert_eq!(resolve_row_offsets(&[], 17), None);
        }

        /// One row carrying its own position is enough to make the batch
        /// per-row; the rest fall back to the shared offset rather than to zero,
        /// which is what keeps a partially-overridden batch coherent.
        #[test]
        fn a_row_with_its_own_offset_makes_the_whole_batch_per_row() {
            assert_eq!(
                resolve_row_offsets(&[Some(40), Some(33), Some(37)], 17),
                Some(vec![40, 33, 37])
            );
            assert_eq!(
                resolve_row_offsets(&[Some(40), None], 17),
                Some(vec![40, 17]),
                "a row without its own offset keeps the batch's, not 0"
            );
        }
    }
}

#[cfg(test)]
mod prefill_chunk_tests {
    use super::text_models_inputs_processor::chunk_window;

    /// The windows a cursor walks must TILE the prompt: no token fed twice, none
    /// skipped, and the walk must terminate. A gap is a hole in the KV cache
    /// that nothing downstream would report — the model would simply attend
    /// over positions that were never written.
    #[test]
    fn chunk_windows_tile_the_prompt_exactly() {
        for len in [1usize, 7, 64, 255, 256, 257, 1024] {
            for chunk in [1usize, 8, 64, 256, 512] {
                let mut offset = 0usize;
                let mut seen = 0usize;
                let mut steps = 0usize;
                loop {
                    let (a, b) = chunk_window(len, offset, chunk);
                    assert_eq!(a, seen, "gap or overlap at offset {offset} (len {len})");
                    seen = b;
                    steps += 1;
                    assert!(steps <= len + 2, "walk did not terminate (len {len})");
                    if offset + chunk >= len {
                        break;
                    }
                    offset += chunk;
                }
                assert_eq!(seen, len, "walk covered {seen} of {len} tokens");
            }
        }
    }

    /// A cohort shares one cursor, so a shorter prompt runs out first. It must
    /// yield an empty window, not panic and not wrap.
    #[test]
    fn a_short_prompt_in_a_mixed_cohort_yields_an_empty_window() {
        let (a, b) = chunk_window(10, 64, 64);
        assert_eq!(
            (a, b),
            (10, 10),
            "start must clamp to len and end must not precede it"
        );
        assert!(b >= a);
    }

    /// Teeth: the tiling test must be able to fail. An off-by-one cursor — the
    /// classic way to write this — is caught.
    #[test]
    fn the_tiling_check_catches_an_off_by_one_walk() {
        let len = 100usize;
        let chunk = 32usize;
        let (a, _) = chunk_window(len, chunk - 1, chunk);
        assert_ne!(
            a, chunk,
            "advancing the cursor by chunk-1 must NOT look like a correct walk"
        );
    }
}

#[cfg(test)]
mod cursor_reaches_every_path_tests {
    use super::text_models_inputs_processor::{make_prompt_chunk, PrefillChunkGuard};
    use candle_core::Device;

    /// 🔑 THE STRUCTURAL GATE for chunked prefill.
    ///
    /// `make_prompt_chunk` is the ONE function every prompt-input path reaches:
    /// `get_prompt_input` calls it, and so do `normal.rs:962` and
    /// `vision.rs:763` — **directly, bypassing `get_prompt_input` entirely**.
    /// `NormalPipeline` is the second of those, which is what DeepSeek-V4 is,
    /// which is the model that matters.
    ///
    /// So the cursor must be honoured HERE, not in one caller. If it is applied
    /// in a caller instead, this test fails and the feature is inert on
    /// whichever path did not get patched — which is exactly what a $1.60 GPU
    /// run discovered after the code looked correct and its own unit tests were
    /// green (`prompt_tokens = 2892`, chunk 512, six chunks expected, zero
    /// observed).
    ///
    /// This is the class of defect `wave64-CP §3` named — *"a new channel wired
    /// into one of two dispatch paths"* — hit for the third time in one session,
    /// and the first time by an author who had read and quoted the warning.
    /// Documentation did not prevent it. A test that fails on CPU, with no card,
    /// does.
    #[test]
    fn a_non_zero_cursor_feeds_only_its_chunk_on_every_path() {
        // Injected, not `std::env::set_var`. The environment is process-global
        // and `prefill_chunk_size` latches its answer in a `OnceLock`, so the
        // env version contaminated sibling tests and could not be undone — see
        // `PrefillChunkGuard`. `_guard` restores the previous value on drop,
        // including on assertion panic.
        let _guard = PrefillChunkGuard::set(Some(4));
        let toks: Vec<u32> = (0..20u32).collect();
        let dev = Device::Cpu;

        // Cursor 8 with chunk 4 must feed tokens [8, 12) — four of them — not
        // the whole 20-token prompt.
        let out = make_prompt_chunk(
            8,
            vec![&toks[..]],
            &[0],
            &dev,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .expect("make_prompt_chunk must build inputs for a mid-prompt cursor");

        let width = out.input.dims().last().copied().unwrap_or(0);
        assert_eq!(
            width, 4,
            "a cursor of 8 with chunk 4 must feed a 4-token window; got {width}. \
             If this is 20, the cursor is being applied in a CALLER rather than \
             here, so every path that calls `make_prompt_chunk` directly \
             (normal.rs, vision.rs) is silently unchunked."
        );
    }

    /// The override must not outlive its guard — otherwise the fix reproduces
    /// the bug it replaces.
    ///
    /// The `set_var` version this replaces could not pass this test at all: its
    /// `remove_var` cleared the environment, but `prefill_chunk_size` had
    /// already latched the value in a `OnceLock`, so the chunk stayed at 4 for
    /// the rest of the process. A guard that cannot be shown restoring is the
    /// same class of unverified guard as one that cannot be shown failing.
    #[test]
    fn the_injected_chunk_does_not_outlive_its_guard() {
        use super::text_models_inputs_processor::prefill_chunk_size;

        // Whatever this process ambiently resolves to — normally `None`, since
        // `ARC_PREFILL_CHUNK` is unset and must stay unset (chunking is
        // measured NEGATIVE until the expert gather is fixed).
        let ambient = prefill_chunk_size();

        {
            let _guard = PrefillChunkGuard::set(Some(7));
            assert_eq!(
                prefill_chunk_size(),
                Some(7),
                "an installed guard must be what `prefill_chunk_size` returns"
            );
            {
                // Nesting must restore the OUTER override, not clear it.
                let _inner = PrefillChunkGuard::set(Some(9));
                assert_eq!(prefill_chunk_size(), Some(9));
            }
            assert_eq!(
                prefill_chunk_size(),
                Some(7),
                "dropping a nested guard must restore the enclosing override, not wipe it"
            );
        }

        assert_eq!(
            prefill_chunk_size(),
            ambient,
            "the override leaked past its guard — every later test in this \
             process now runs with a chunk size it never asked for, which is \
             exactly the contamination the `set_var` version caused"
        );
    }
}
