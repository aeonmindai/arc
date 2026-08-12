#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub mod paged_attention;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub use paged_attention::PagedAttention;

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub mod paged_attention {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
    use crate::{attention::SdpaParams, pipeline::text_models_inputs_processor::FlashParams};

    pub struct PagedAttention;

    impl PagedAttention {
        pub fn new(
            _head_dim: usize,
            _device: &Device,
            _alibi_slopes: Option<Vec<f32>>,
        ) -> Result<Self> {
            candle_core::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        #[allow(clippy::too_many_arguments)]
        #[allow(unused_variables)]
        pub fn forward(
            &self,
            _query: &Tensor,
            _key: &Tensor,
            _value: &Tensor,
            _attention_mask: Option<&Tensor>,
            _key_cache: Option<Tensor>,
            _value_cache: Option<Tensor>,
            _input_metadata: &PagedAttentionInputMetadata,
            _sdpa_params: &SdpaParams,
            _flash_params: Option<&FlashParams>,
        ) -> Result<Tensor> {
            candle_core::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        /// Stub for the V4 CSA/HCA paged dispatch (RUN-167). The real
        /// implementation lives in the CUDA / Metal builds; the stub keeps
        /// the API surface consistent so model code compiles cross-platform
        /// (call sites are gated by an `Option<PagedAttention>` that's only
        /// ever `Some` on a CUDA/Metal-enabled build).
        #[allow(clippy::too_many_arguments)]
        #[allow(unused_variables)]
        pub fn cache_write_and_gather(
            &self,
            _key: &Tensor,
            _value: &Tensor,
            _key_cache: &mut Tensor,
            _value_cache: &mut Tensor,
            _input_metadata: &PagedAttentionInputMetadata,
            _out_dtype: DType,
        ) -> Result<(Tensor, Tensor)> {
            candle_core::bail!(
                "PagedAttention::cache_write_and_gather requires the CUDA or Metal feature flags."
            );
        }
    }
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub use paged_attention::PagedAttention;
