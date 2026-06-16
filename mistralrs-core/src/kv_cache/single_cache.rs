use candle_core::{Result, Tensor};

use super::NormalCache;

#[derive(Debug, Clone)]
pub struct SingleCache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reset (as in it will not share
    // its internal state with the cloned instance).
    pub all_data: Option<Tensor>,
    pub dim: usize,
    pub current_seq_len: usize,
    pub capacity_seq_len: usize,
    pub max_seq_len: usize,
}

impl SingleCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor> {
        self.all_data.as_ref()
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.narrow(self.dim, 0, self.current_seq_len)?),
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        if len > self.capacity_seq_len {
            candle_core::bail!(
                "kv-cache: requested length ({}) exceeds current capacity ({})",
                len,
                self.capacity_seq_len
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        self.current_seq_len = len;
        Ok(())
    }

    /// CUDA-graph-capturable append (RUN-161 Step 2c).
    ///
    /// Pre-grows `all_data` to the fixed `capacity_seq_len` on first use (no
    /// realloc thereafter -> stable address for replay), then writes `src`
    /// ([B,H,1,D]) at the device-held sequence slot `position` ([B] U32) via
    /// the in-place `write_kv_inplace` kernel (launched on candle's stream, so
    /// it records into a captured graph). Returns the FULL fixed-capacity
    /// buffer `[B,H,capacity,D]`; the caller masks slots beyond the current
    /// length using the same device position (so attention shape is constant
    /// across decode steps and the graph can replay).
    ///
    /// Unlike `append`, this does NOT advance `current_seq_len`: under graph
    /// replay the host position is meaningless; the device `position` is the
    /// source of truth for both the write slot and the read mask.
    ///
    /// `read_capacity` is the FIXED window the caller attends over (V4 uses
    /// `sliding_window`). The returned `[B,H,read_capacity,D]` view is a
    /// constant-offset narrow, so the attention shape is identical every decode
    /// step -> the caching allocator hits and the graph replays. Slots beyond
    /// the current length are stale/zero and must be masked by the caller.
    pub fn append_graph(
        &mut self,
        src: &Tensor,
        position: &Tensor,
        read_capacity: usize,
    ) -> Result<Tensor> {
        if self.dim != 2 {
            candle_core::bail!(
                "append_graph requires seq dim == 2 (got {}); V4 MQA cache is [B,H,T,D]",
                self.dim
            );
        }
        let seq_len = src.dim(self.dim)?;
        if seq_len != 1 {
            candle_core::bail!("append_graph is decode-only (src seq len must be 1, got {seq_len})");
        }
        // Buffer must be at least `read_capacity` along the seq dim. Reuse the
        // existing (eager-populated) all_data so past K/V is present; only
        // allocate if absent.
        let need = read_capacity.max(self.capacity_seq_len);
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = need;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
        }
        let ad = self.all_data.as_ref().unwrap();
        // Device-slot write of the new token; `write_kv_inplace` uses the
        // buffer's real capacity from all_data's dims, so the slot is correct.
        mistralrs_quant::kvwrite::write_kv_inplace(ad, src, position)?;
        // Fixed-offset window -> constant shape for capture/replay.
        ad.narrow(self.dim, 0, read_capacity)
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
        };

        // Expand kv cache
        if self.current_seq_len + seq_len > self.capacity_seq_len {
            let diff = self.current_seq_len + seq_len - self.capacity_seq_len;
            let n_blocks_needed = diff.div_ceil(NormalCache::CACHE_GROW_SIZE);
            self.capacity_seq_len += n_blocks_needed * NormalCache::CACHE_GROW_SIZE;
            if self.capacity_seq_len > self.max_seq_len {
                candle_core::bail!(
                    "kv-cache: requested capacity ({}) above max seq len ({})",
                    self.capacity_seq_len,
                    self.max_seq_len
                )
            }
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            ad.slice_set(self.all_data.as_ref().unwrap(), self.dim, 0)?;
            self.all_data = Some(ad);
        }

        let ad = self.all_data.as_mut().unwrap();

        ad.slice_set(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}
