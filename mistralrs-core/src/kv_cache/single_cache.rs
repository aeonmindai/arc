use candle_core::{Result, Tensor};

use super::NormalCache;

/// Parent system: ArcInfer / ArcKV.
///
/// How many trailing rows a [`SingleCache::retain_window`] slot compacts down
/// to, over and above the attention window its owner asked for.
///
/// Sized like [`crate::kv_cache::XS_TAIL_MARGIN_TOKENS`] and for the same
/// reason: an MTP verify step rolls the cache *back* by up to `--mtp-depth`
/// positions (clap caps that at 8), and the forward that follows must still
/// find its whole reachable span retained. 16 covers every legal depth with
/// slack, and costs 16 rows per layer per sequence.
pub const RETAIN_WINDOW_MARGIN: usize = 16;

/// The compaction quantum for a windowed slot: how many rows of slack sit
/// above the retained window, i.e. how many appends happen between two
/// compactions.
///
/// This is the whole memory/×copy trade. A compaction rewrites `retain` rows;
/// doing it every `RETAIN_WINDOW_CHUNK` tokens amortises that to
/// `retain / chunk` extra row-writes per token. At the V4 production geometry
/// (`window = 128`, so `retain = 144`; `chunk = 128`) that is 1.12 rows copied
/// per row stored, for a resident footprint of 144–271 rows — **constant in
/// the context length**.
///
/// In bytes, at V4 Flash's 1026 B/token/layer (`head_dim = 512` BF16 K plus
/// the 1-wide V marker) over 43 layers: **6.35–11.96 MB/seq at any context**,
/// against 180.7 MB at 4k, 361.4 MB at 8k and 1,445.7 MB at 32k — 15–28×,
/// 30–57× and 121–228× respectively. At B=32 and 8k that is 11.57 GB → 0.38 GB.
/// These are arithmetic, from the geometry the guard test pins; the decode-time
/// cost of the compaction copy is NOT measured (no GPU on this change).
///
/// It is deliberately NOT [`crate::kv_cache::NormalCache::CACHE_GROW_SIZE`]
/// (512): that is the *growth* quantum for a cache that keeps everything, and
/// reusing it here would leave up to 656 rows resident — 12–29 MB/seq, i.e.
/// roughly half the saving for the same asymptotic copy cost.
pub const RETAIN_WINDOW_CHUNK: usize = 128;

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
    /// Absolute position of physical column `0` of `all_data`.
    ///
    /// `0` — every model but DeepSeek V4 under `ARC_V4_KV_WINDOW`, and V4
    /// itself with the flag off — means the buffer holds the whole sequence
    /// and every path below is character-for-character what it was.
    ///
    /// 🔑 `current_seq_len` stays the sequence's **absolute** length whatever
    /// this is. That is load-bearing: `mtp_pipeline::truncate_cache_by`,
    /// `prefix_cacher`, `draft_step_end_column`'s "slot `k` is absolute
    /// position `k`" invariant and the scheduler's bucketing all read it, and
    /// none of them know or care how much of the sequence is physically
    /// resident. The number of rows actually stored is
    /// [`Self::physical_seq_len`].
    pub first_cached: usize,
    /// Retain only the trailing `Some(rows)` of the sequence, compacting the
    /// buffer when it drifts further than [`RETAIN_WINDOW_CHUNK`] past that.
    ///
    /// `None` — the default, and every model that is not V4 — keeps the whole
    /// sequence, which is the behaviour this cache has always had.
    ///
    /// Set per append by the owner rather than at construction because
    /// `NormalCache::new_plain` is called from model constructors that do not
    /// know the attention window (see `deepseek4::apply_kv_window`).
    pub retain_window: Option<usize>,
}

impl SingleCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
            first_cached: 0,
            retain_window: None,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Absolute position of physical column 0. See [`Self::first_cached`].
    pub fn first_cached(&self) -> usize {
        self.first_cached
    }

    /// Rows physically resident in `all_data`, i.e. `current_seq_len` minus the
    /// evicted prefix. Equal to `current_seq_len` for every non-windowed slot.
    pub fn physical_seq_len(&self) -> usize {
        self.current_seq_len.saturating_sub(self.first_cached)
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
            Some(d) => Some(d.narrow(self.dim, 0, self.physical_seq_len())?),
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.first_cached = 0;
        self.all_data = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        // A windowed slot can no longer answer for positions it evicted. This
        // is a REFUSAL, not a truncation: the prefix cacher declines the match
        // and serves the request uncached (`prefix_cacher.rs`), and an MTP
        // rollback — bounded by `--mtp-depth`, which `RETAIN_WINDOW_MARGIN`
        // covers — never reaches it.
        if len < self.first_cached {
            candle_core::bail!(
                "kv-cache: requested length ({}) is below this slot's retained window, which \
                 starts at absolute position {}",
                len,
                self.first_cached
            );
        }
        if len - self.first_cached > self.capacity_seq_len {
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

    /// The `first_cached` a windowed slot should hold at absolute length `len`.
    ///
    /// A pure function of `len` on purpose. `NormalCacheManager` batches
    /// sequences into ONE dense cache described by a single template, so two
    /// sequences at the same absolute length must agree on how much of
    /// themselves is resident — otherwise the batched slot's `first_cached`
    /// would be right for `seqs[0]` and silently wrong for everyone else.
    /// Deriving it from `len` alone makes that agreement structural rather
    /// than a property of matching eviction histories.
    ///
    /// `len - target` lands in `[retain, retain + chunk)`, so the buffer never
    /// holds fewer than `retain` rows and never more than `retain + chunk`.
    pub fn window_target_first_cached(len: usize, retain: usize) -> usize {
        let over = len.saturating_sub(retain);
        over - (over % RETAIN_WINDOW_CHUNK)
    }

    /// Drop everything before absolute position `target`, rewriting `all_data`
    /// as a `retain + RETAIN_WINDOW_CHUNK`-row buffer holding the retained
    /// tail at column 0.
    ///
    /// One allocation and one copy of the retained rows, every
    /// [`RETAIN_WINDOW_CHUNK`] tokens. A `narrow` view would be free but would
    /// keep the original storage alive, which is the entire thing being
    /// bought here.
    fn evict_before(&mut self, target: usize, retain: usize) -> Result<()> {
        let drop = target - self.first_cached;
        let physical = self.physical_seq_len();
        debug_assert!(drop <= physical, "eviction target is past the live run");
        let keep = physical - drop;
        let Some(ad) = self.all_data.as_ref() else {
            // Nothing materialised: only the bookkeeping moves.
            self.first_cached = target;
            return Ok(());
        };
        let capacity = retain + RETAIN_WINDOW_CHUNK;
        let mut shape = ad.dims().to_vec();
        shape[self.dim] = capacity;
        let compacted = Tensor::zeros(shape, ad.dtype(), ad.device())?;
        if keep > 0 {
            let live = ad.narrow(self.dim, drop, keep)?.contiguous()?;
            compacted.slice_set(&live, self.dim, 0)?;
        }
        self.all_data = Some(compacted);
        self.capacity_seq_len = capacity;
        self.first_cached = target;
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
    /// Pin the graph buffer's capacity so it cannot change again for the run.
    ///
    /// # The mechanism this fixes
    ///
    /// Capture-time allocation misses were the KV cache **reallocating as it
    /// grows**. Measured on an H200: sizes 122880 / 172032 / 262144 bytes at
    /// warmups of 4 / 8 / 16, which decompose as `512 x {120,168,256} x 2`
    /// with 512 = V4's `head_dim` and 2 = BF16 — so 120/168/256 are cache
    /// capacities. Capture always lands one growth beyond whatever the warmups
    /// covered, which is why no finite warmup ever reached zero misses and why
    /// raising it merely sampled more growth points. Each miss is an unstable
    /// graph memory node; candle calls that a correctness bug, and the process
    /// died inside `cuGraphInstantiate` with no Rust-level error.
    ///
    /// # Why this pins rather than pre-grows to `max_seq_len`
    ///
    /// The obvious fix — allocate at the capacity the cache can never exceed —
    /// is a trap here. `max_seq_len` is `cfg.max_position_embeddings`, and for
    /// DeepSeek V4 that is **1,048,576**. At `head_dim` 512 in BF16 across 43
    /// layers that is ~46 GB of extra VRAM, which would OOM the very box this
    /// runs on and is simply fatal on an 80 GB card. Growth is the problem, not
    /// smallness, so the buffer is PINNED at whatever it legitimately needs on
    /// the first graph append — which happens during warmup, several steps
    /// before capture, so the size is warm in the alloc cache by then — and any
    /// later attempt to change it is refused rather than silently invalidating
    /// a captured graph.
    ///
    /// A refusal is the correct failure: the caller disables capture and falls
    /// back to eager, which is slow and right, instead of replaying a graph
    /// whose buffer moved underneath it, which is fast and wrong.
    fn pregrow_for_graph(&mut self, src: &Tensor, read_capacity: usize) -> Result<()> {
        let need = read_capacity.max(self.capacity_seq_len);
        match self.all_data.as_ref() {
            Some(ad) => {
                let have = ad.dim(self.dim)?;
                if have < need {
                    candle_core::bail!(
                        "graph KV buffer would have to grow from {have} to {need} slots \
                         mid-run. Growing it now would move the address a captured graph \
                         has already baked in, so capture must be abandoned rather than \
                         replayed against a moved buffer. Raise the window or shorten the \
                         sequence."
                    );
                }
                Ok(())
            }
            None => {
                let mut shape = src.dims().to_vec();
                shape[self.dim] = need;
                self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
                // Keep the bookkeeping honest: `try_set_len` and the cache
                // managers compare against `capacity_seq_len`, and a buffer
                // that differs from the number it advertises is a trap for the
                // next reader.
                self.capacity_seq_len = need;
                Ok(())
            }
        }
    }

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
            candle_core::bail!(
                "append_graph is decode-only (src seq len must be 1, got {seq_len})"
            );
        }
        // `position` is an ABSOLUTE sequence slot held on the device, and
        // `write_kv_inplace` indexes the buffer with it directly. A windowed
        // buffer's column 0 is absolute `first_cached`, so the two disagree by
        // exactly the evicted prefix — and the disagreement would be a silent
        // wrong-slot write, not an error.
        if self.first_cached != 0 || self.retain_window.is_some() {
            candle_core::bail!(
                "append_graph: the graph-decode path writes at an ABSOLUTE device slot and \
                 cannot address a windowed cache (first_cached {}). Unset the KV window for \
                 graph capture.",
                self.first_cached
            );
        }
        // Same preallocation correction `append` performs. Without it the
        // graph path inherits the engine's dense [B,H,cap,head_dim] buffer and
        // a 1-wide V4 marker cannot be written into it.
        self.reconcile_preallocation(src)?;

        self.pregrow_for_graph(src, read_capacity)?;
        let ad = self.all_data.as_ref().unwrap();
        // RING write. The read below is a fixed `0..read_capacity` window, so
        // writing at the ABSOLUTE position put every token from `read_capacity`
        // onward OUTSIDE the window that is read: the freshest key became
        // invisible and a stale row was attended in its place, making this arm
        // correct only while the context was shorter than `sliding_window`.
        //
        // `read_capacity` IS `sliding_window` here, so a ring of that many
        // slots holds exactly the key set the raw branch is defined to attend.
        // The slot is derived on the DEVICE from the device position — a
        // host-resolved `position % capacity` would be baked into the recorded
        // kernel and every replay would write to the capture step's slot. See
        // `layers::graph_ring_slot` for why permuting the keys is invariant
        // here and why the existing length mask is already ring-correct.
        let slot = crate::layers::graph_ring_slot(position, read_capacity)?;
        mistralrs_quant::kvwrite::write_kv_inplace(ad, src, &slot)?;
        // Fixed-offset window -> constant shape for capture/replay.
        ad.narrow(self.dim, 0, read_capacity)
    }

    /// Can `src` be written into the buffer this cache already holds?
    ///
    /// `Tensor::slice_set` requires an exact dtype match and an exact match on
    /// every dimension except the one being written along
    /// (`candle-core/src/tensor_cat.rs:254` and `:278`, in that order — dtype
    /// is checked first, which is why a dtype disagreement masks a width one).
    /// `None` (no buffer yet) always fits: [`Self::append`] allocates it from
    /// `src`.
    fn preallocation_fits(&self, src: &Tensor) -> bool {
        let Some(ad) = self.all_data.as_ref() else {
            return true;
        };
        ad.dtype() == src.dtype()
            && ad.rank() == src.rank()
            && ad
                .dims()
                .iter()
                .zip(src.dims())
                .enumerate()
                .all(|(i, (a, s))| i == self.dim || a == s)
    }

    /// Discard a preallocated buffer that `src` cannot be written into.
    ///
    /// Shared by [`Self::append`] and [`Self::append_graph`] **so the two
    /// cannot drift**. The graph path not having this was wave48-BY
    /// reappearing in a path that did not exist when wave48-BY was fixed: the
    /// engine preallocates from `ModelConfigMetadata::{k_head_dim,v_head_dim}`
    /// and `activation_dtype`, which assumes dense activation-precision K and V
    /// at the declared head width, and V4's V half is a 1-wide marker
    /// (`V4_V_MARKER_WIDTH` — V *is* K for fused MQA). Measured on an H200:
    /// `append_graph V half failed (k=[1,1,1,512], v=[1,1,1,1]) : kv-write:
    /// src [1,1,1,1] incompatible with cache [1,1,512,512]`, on the engine's
    /// own dummy run.
    ///
    /// An empty cache's buffer holds nothing, so discarding it is lossless —
    /// the pre-grow is kept, only dtype and width are corrected. A NON-empty
    /// cache is real corruption and is raised, never papered over.
    fn reconcile_preallocation(&mut self, src: &Tensor) -> Result<()> {
        if self.preallocation_fits(src) {
            return Ok(());
        }
        if self.current_seq_len != 0 {
            let ad = self.all_data.as_ref().expect("mismatch implies a buffer");
            candle_core::bail!(
                "kv-cache: cannot append {:?} {:?} into a cache already holding {} \
                 token(s) as {:?} {:?} — the layout changed mid-sequence",
                src.dtype(),
                src.dims(),
                self.current_seq_len,
                ad.dtype(),
                ad.dims()
            );
        }
        self.all_data = None;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;

        // Compact BEFORE writing, never after: the forward that follows this
        // append attends `t_q + window - 1` trailing keys
        // (`dsv4_attention::raw_keep_span`), and evicting afterwards would take
        // rows this very step still needs. Compacting first leaves `retain`
        // rows, so the post-append run is `retain + seq_len` — comfortably the
        // reachable span for any `retain >= window`.
        if let Some(retain) = self.retain_window {
            let target = Self::window_target_first_cached(self.current_seq_len, retain);
            if target > self.first_cached {
                self.evict_before(target, retain)?;
            }
        }

        // A buffer can be present before the first append: on a prompt step
        // with no prefix hit the engine issues `CacheInstruction::Reset {
        // load_preallocated_cache: true }` (`engine/mod.rs:470`) and
        // `NormalCacheManager::set_none_cache` installs the sequence's
        // preallocated tensor as `all_data` (`kv_cache/mod.rs:802`). That
        // tensor is shaped and typed from `ModelConfigMetadata::{k_head_dim,
        // v_head_dim}` and `activation_dtype` (`engine/add_request.rs:464`),
        // which assumes every model stores dense activation-precision K and V
        // of the declared head width.
        //
        // DeepSeek V4 does not, in either of its layouts: the V half is a
        // 1-wide marker (`V4_V_MARKER_WIDTH`, V *is* K for fused MQA) and,
        // under `ARC_V4_FP8_KV=1`, the K half is U8 E4M3 codes `head_dim -
        // qk_rope_head_dim` wide with `[rope ++ amax]` in the V half. Forcing
        // `src` into a buffer it does not fit is what produced `dtype mismatch
        // in slice-set, lhs: BF16, rhs: U8` on every V4 forward — including
        // the engine's own dummy run — the first time that code met a GPU
        // (wave48-BY), and `shape mismatch on dim 3, 512 <> 1` behind it.
        //
        // An empty cache's buffer holds nothing, so the honest answer is to
        // discard it and allocate from `src` at the same capacity — the
        // pre-grow is kept, only its dtype and width are corrected. A NON-empty
        // cache is a different matter: the tokens already in it were written in
        // some other layout, so a disagreement there is real corruption and is
        // raised rather than papered over.
        self.reconcile_preallocation(src)?;

        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
        };

        // Expand kv cache. Sized from the PHYSICAL run, which is
        // `current_seq_len` for every non-windowed slot.
        let physical = self.physical_seq_len();
        if physical + seq_len > self.capacity_seq_len {
            let diff = physical + seq_len - self.capacity_seq_len;
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

        ad.slice_set(src, self.dim, physical)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const CAP: usize = 8;

    /// A cache carrying a pre-grown buffer, as `set_none_cache` leaves it
    /// before the first append (`kv_cache/mod.rs:802`).
    fn preallocated(width: usize, dtype: DType) -> Result<SingleCache> {
        Ok(SingleCache {
            all_data: Some(Tensor::zeros(
                (1usize, 1usize, CAP, width),
                dtype,
                &Device::Cpu,
            )?),
            dim: 2,
            current_seq_len: 0,
            max_seq_len: 4096,
            capacity_seq_len: CAP,
            first_cached: 0,
            retain_window: None,
        })
    }

    fn row(width: usize, dtype: DType, fill: f32) -> Result<Tensor> {
        Tensor::from_vec(
            vec![fill; width],
            (1usize, 1usize, 1usize, width),
            &Device::Cpu,
        )?
        .to_dtype(dtype)
    }

    /// A preallocated buffer the source fits must be **kept**, not silently
    /// thrown away — that pre-grow is the whole point of
    /// `load_preallocated_cache`, and reallocating it every prompt would be a
    /// quiet performance regression no equality assertion would catch.
    ///
    /// Discriminated by a sentinel written past the append offset: a
    /// reallocated buffer comes back zeroed, so a test that only checked the
    /// appended row would pass either way.
    #[test]
    fn a_fitting_preallocation_is_reused() -> Result<()> {
        let cache = preallocated(4, DType::F32)?;
        let sentinel = row(4, DType::F32, 7.0)?;
        cache
            .all_data
            .as_ref()
            .unwrap()
            .slice_set(&sentinel, 2, CAP - 1)?;
        let mut cache = cache;

        cache.append(&row(4, DType::F32, 1.0)?)?;

        let ad = cache.all_data.as_ref().unwrap();
        assert_eq!(ad.dims(), &[1, 1, CAP, 4]);
        assert_eq!(cache.current_seq_len, 1);
        let tail: Vec<f32> = ad.narrow(2, CAP - 1, 1)?.flatten_all()?.to_vec1()?;
        assert_eq!(
            tail,
            vec![7.0f32; 4],
            "the preallocated buffer was reallocated even though the source fit it"
        );
        Ok(())
    }

    /// A preallocated buffer the source does NOT fit — wrong dtype (V4's U8
    /// E4M3 codes) or wrong width (V4's 1-wide V marker) — is discarded and
    /// rebuilt from the source, at the capacity that was preallocated.
    ///
    /// Before this, both cases reached `slice_set` and killed the forward:
    /// `dtype mismatch in slice-set, lhs: BF16, rhs: U8` and `shape mismatch on
    /// dim 3, 512 <> 1` respectively (wave48-BY / wave49-BZ).
    #[test]
    fn a_mismatched_preallocation_is_rebuilt_while_empty() -> Result<()> {
        // Wrong dtype.
        let mut cache = preallocated(4, DType::BF16)?;
        cache.append(&row(4, DType::U8, 3.0)?)?;
        let ad = cache.all_data.as_ref().unwrap();
        assert_eq!(ad.dtype(), DType::U8);
        assert_eq!(ad.dims(), &[1, 1, CAP, 4], "capacity must be preserved");
        assert_eq!(cache.capacity_seq_len, CAP);
        let got: Vec<u8> = ad.narrow(2, 0, 1)?.flatten_all()?.to_vec1()?;
        assert_eq!(got, vec![3u8; 4]);

        // Wrong width on a non-sequence dim.
        let mut cache = preallocated(512, DType::BF16)?;
        cache.append(&row(1, DType::BF16, 0.0)?)?;
        assert_eq!(cache.all_data.as_ref().unwrap().dims(), &[1, 1, CAP, 1]);
        assert_eq!(cache.current_seq_len, 1);
        Ok(())
    }

    /// Rebuilding is only honest while the cache is empty. Once it holds
    /// tokens, a layout disagreement means the tokens already written are in a
    /// different layout than the one now being appended — silently dropping
    /// them would corrupt the sequence, so it is refused loudly.
    #[test]
    fn a_layout_change_mid_sequence_is_refused() -> Result<()> {
        let mut cache = preallocated(4, DType::BF16)?;
        cache.append(&row(4, DType::BF16, 1.0)?)?;
        assert_eq!(cache.current_seq_len, 1);

        let err = cache.append(&row(4, DType::U8, 1.0)?).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("layout changed mid-sequence") && msg.contains("1 token(s)"),
            "expected a loud mid-sequence layout error, got: {msg}"
        );
        // And the tokens already cached are untouched.
        assert_eq!(cache.current_seq_len, 1);
        assert_eq!(cache.all_data.as_ref().unwrap().dtype(), DType::BF16);
        Ok(())
    }

    /// wave48-BY, in the CUDA-graph path. The engine preallocates the KV buffer
    /// from `ModelConfigMetadata::v_head_dim` — dense, activation-precision, at
    /// the declared head width — and installs it as `all_data` before the first
    /// append (`CacheInstruction::Reset { load_preallocated_cache: true }`).
    /// V4's V half is a 1-wide marker, so that buffer does not fit.
    ///
    /// `append` has corrected this since wave48-BY; `append_graph` was written
    /// later and did not, which is why V4 capture died on the engine's own
    /// dummy run with
    ///   `kv-write: src [1,1,1,1] incompatible with cache [1,1,512,512]`.
    ///
    /// Mutation: drop the `reconcile_preallocation` call from `append_graph`
    /// and this fails with exactly that message.
    #[test]
    fn append_graph_corrects_a_preallocated_buffer_of_the_wrong_width() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = SingleCache::new(2, 4096, 512);
        // What the engine installs: dense V at the declared head width.
        cache.all_data = Some(Tensor::zeros((1, 1, 512, 512), DType::F32, &dev)?);
        assert_eq!(cache.current_seq_len, 0, "fixture must be an EMPTY preallocation");

        // What V4 actually writes: a 1-wide marker.
        let marker = Tensor::zeros((1, 1, 1, 1), DType::F32, &dev)?;
        cache.reconcile_preallocation(&marker)?;
        assert!(
            cache.all_data.is_none(),
            "an ill-fitting EMPTY preallocation must be discarded so it can be \
             reallocated at the marker's width"
        );
        Ok(())
    }

    /// The other half of the contract: a NON-empty cache is real corruption and
    /// must be raised, never silently discarded — that would drop live tokens.
    #[test]
    fn reconcile_refuses_to_discard_a_cache_holding_tokens() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = SingleCache::new(2, 4096, 512);
        cache.all_data = Some(Tensor::zeros((1, 1, 512, 512), DType::F32, &dev)?);
        cache.current_seq_len = 7; // pretend seven tokens are already written

        let marker = Tensor::zeros((1, 1, 1, 1), DType::F32, &dev)?;
        let err = cache
            .reconcile_preallocation(&marker)
            .expect_err("a populated cache must not be silently discarded");
        let msg = err.to_string();
        assert!(msg.contains("layout changed mid-sequence"), "{msg}");
        assert!(msg.contains('7'), "the message must name how many tokens are at risk: {msg}");
        Ok(())
    }

    /// The graph buffer is PINNED on first use and never resized after.
    ///
    /// Pre-growing to `max_seq_len` was the obvious fix and is a trap: V4's
    /// `max_position_embeddings` is 1,048,576, which at head_dim 512 in BF16
    /// across 43 layers is ~46 GB of extra VRAM — it would OOM the box it runs
    /// on. Growth is the defect, not smallness.
    ///
    /// Mutation: allow the `Some(ad)` arm to reallocate instead of bailing and
    /// `growth_after_pinning_is_refused` fails.
    #[test]
    fn first_graph_append_pins_the_capacity() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = SingleCache::new(2, 1_048_576, 120);
        let src = Tensor::zeros((1, 1, 1, 8), DType::F32, &dev)?;

        cache.pregrow_for_graph(&src, 128)?;

        let cap = cache.all_data.as_ref().unwrap().dim(2)?;
        assert_eq!(
            cap, 128,
            "must pin at what the graph legitimately needs (read_capacity vs \
             current growth), NOT at max_seq_len — 1M slots here would be ~46 GB"
        );
        assert_eq!(
            cache.capacity_seq_len, 128,
            "capacity_seq_len must track the real buffer"
        );
        Ok(())
    }

    /// Once pinned, a later demand for more slots must be REFUSED, not served
    /// by reallocating — reallocating moves an address a captured graph has
    /// already baked in. Falling back to eager is slow and right; replaying
    /// against a moved buffer is fast and wrong.
    #[test]
    fn growth_after_pinning_is_refused() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = SingleCache::new(2, 1_048_576, 0);
        let src = Tensor::zeros((1, 1, 1, 8), DType::F32, &dev)?;

        cache.pregrow_for_graph(&src, 128)?;
        assert_eq!(cache.all_data.as_ref().unwrap().dim(2)?, 128);

        // A wider window now would require a bigger buffer.
        let err = cache
            .pregrow_for_graph(&src, 4096)
            .expect_err("growing a pinned graph buffer must be refused");
        let msg = err.to_string();
        assert!(msg.contains("would have to grow from 128 to 4096"), "{msg}");
        assert!(
            msg.contains("capture must be abandoned"),
            "the message must say what the caller should do: {msg}"
        );
        // And the buffer must be untouched by the failed attempt.
        assert_eq!(cache.all_data.as_ref().unwrap().dim(2)?, 128);
        Ok(())
    }

    /// A pinned buffer that is already big enough is reused silently — the
    /// steady state on every decode step after the first.
    #[test]
    fn a_sufficient_pinned_buffer_is_reused() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = SingleCache::new(2, 1_048_576, 0);
        let src = Tensor::zeros((1, 1, 1, 8), DType::F32, &dev)?;
        cache.pregrow_for_graph(&src, 128)?;
        let before = cache.all_data.as_ref().unwrap().dim(2)?;
        for _ in 0..5 {
            cache.pregrow_for_graph(&src, 128)?;
        }
        assert_eq!(cache.all_data.as_ref().unwrap().dim(2)?, before);
        Ok(())
    }

    /// THE PROPERTY THAT MATTERS IS AVAILABILITY, NOT A WELL-WORDED ERROR.
    ///
    /// When the pinned graph buffer refuses to grow, `normal.rs` drops graph
    /// mode and retries the forward eagerly. That fallback is only real if the
    /// EAGER path can serve the very demand the graph path just refused —
    /// otherwise the retry fails too and the request dies anyway, which is an
    /// availability regression traded for a memory saving, and a quiet one:
    /// it fires on a sequence longer than whatever the buffer was pinned at,
    /// i.e. on real traffic rather than in tests.
    ///
    /// So assert the fallback has somewhere to go.
    ///
    /// Mutation: give `append` the same pin (make it bail when
    /// `all_data` is Some and too small) and this fails — which is exactly the
    /// state that would make the eager retry in `normal.rs` useless.
    #[test]
    fn the_eager_path_still_serves_what_the_pinned_graph_refuses() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = SingleCache::new(2, 1_048_576, 0);
        let src = Tensor::zeros((1, 1, 1, 8), DType::F32, &dev)?;

        // Pin small, then demand more than the pin: the graph path refuses.
        cache.pregrow_for_graph(&src, 128)?;
        assert!(
            cache.pregrow_for_graph(&src, 4096).is_err(),
            "precondition: the pinned graph buffer must refuse to grow"
        );

        // The eager path must still be able to serve that sequence. It has no
        // pin — it grows — so the retry in `normal.rs` completes the request.
        for _ in 0..200 {
            cache.append(&src)?;
        }
        assert_eq!(
            cache.current_seq_len, 200,
            "the eager path must keep accepting tokens past the graph's pin, or \
             the fallback is not a fallback"
        );
        assert!(
            cache.all_data.as_ref().unwrap().dim(2)? >= 200,
            "and it must have grown the buffer to hold them"
        );
        Ok(())
    }

    // -----------------------------------------------------------------
    // Windowed store (`retain_window` / `first_cached`)
    // -----------------------------------------------------------------

    /// One row whose every element is `n`, so a readback names the absolute
    /// position it came from.
    // Row values are small integer positions (< a few thousand), well inside
    // f32's exact range; the cast is how a position is made readable as data.
    #[allow(clippy::cast_precision_loss)]
    fn numbered(n: usize) -> Result<Tensor> {
        Tensor::from_vec(vec![n as f32; 4], (1usize, 1usize, 1usize, 4), &Device::Cpu)
    }

    fn windowed(retain: usize) -> SingleCache {
        let mut sc = SingleCache::new(2, 1_000_000, 8);
        sc.retain_window = Some(retain);
        sc
    }

    /// The retained run must be the sequence's TRUE tail, at the right
    /// absolute base, at every length — including the compaction boundaries.
    ///
    /// The readback is by value (`numbered`), not by width: a cache that
    /// compacted by dropping the wrong end, or that lost a token to the
    /// realloc, would still report a plausible length.
    #[test]
    // Row values are small integer positions (< a few thousand), well inside
    // f32's exact range; the cast is how a position is made readable as data.
    #[allow(clippy::cast_precision_loss)]
    fn a_windowed_cache_retains_the_true_tail_at_the_right_base() -> Result<()> {
        let retain = 8usize;
        let mut sc = windowed(retain);
        for n in 0..(4 * RETAIN_WINDOW_CHUNK + 3) {
            sc.append(&numbered(n)?)?;

            assert_eq!(sc.current_seq_len, n + 1, "absolute length must not window");
            let expect_base = SingleCache::window_target_first_cached(n, retain);
            assert_eq!(
                sc.first_cached,
                expect_base,
                "at absolute length {} the retained base drifted",
                n + 1
            );

            let live = sc.current_data()?.expect("appended, so materialised");
            assert_eq!(live.dim(2)?, sc.physical_seq_len());
            let got: Vec<f32> = live.narrow(3, 0, 1)?.flatten_all()?.to_vec1()?;
            let want: Vec<f32> = (sc.first_cached..=n).map(|i| i as f32).collect();
            assert_eq!(got, want, "retained rows are not the sequence's own tail");
        }
        // Bounded: `retain` rows, plus at most one chunk of slack.
        assert!(
            sc.capacity_seq_len <= retain + RETAIN_WINDOW_CHUNK,
            "windowed capacity grew without bound: {}",
            sc.capacity_seq_len
        );
        Ok(())
    }

    /// The retained run never drops below the window its owner asked for —
    /// that bound is what makes `dsv4_attention`'s reachable span always
    /// resident, and violating it is a wrong answer, not a slow one.
    #[test]
    fn a_windowed_cache_never_holds_fewer_rows_than_the_window() -> Result<()> {
        let retain = 8usize;
        let mut sc = windowed(retain);
        for n in 0..(3 * RETAIN_WINDOW_CHUNK) {
            sc.append(&numbered(n)?)?;
            let physical = sc.physical_seq_len();
            assert!(
                physical >= retain.min(n + 1),
                "at length {} only {physical} row(s) resident, below the {retain}-row window",
                n + 1
            );
            assert!(
                physical <= retain + RETAIN_WINDOW_CHUNK,
                "at length {} {physical} row(s) resident, above the bound",
                n + 1
            );
        }
        Ok(())
    }

    /// A prompt longer than the window is kept WHOLE for the forward that
    /// consumes it — the prefill attends every one of its own positions — and
    /// is compacted on the next append, not this one.
    #[test]
    // Row values are small integer positions (< a few thousand), well inside
    // f32's exact range; the cast is how a position is made readable as data.
    #[allow(clippy::cast_precision_loss)]
    fn a_long_prefill_is_kept_whole_and_compacted_on_the_next_append() -> Result<()> {
        let retain = 8usize;
        let mut sc = windowed(retain);
        let prompt = Tensor::from_vec(
            (0..(3 * RETAIN_WINDOW_CHUNK * 4))
                .map(|i| (i / 4) as f32)
                .collect::<Vec<_>>(),
            (1usize, 1usize, 3 * RETAIN_WINDOW_CHUNK, 4usize),
            &Device::Cpu,
        )?;
        sc.append(&prompt)?;
        assert_eq!(sc.first_cached, 0, "the prefill's own span must stay whole");
        assert_eq!(sc.physical_seq_len(), 3 * RETAIN_WINDOW_CHUNK);

        sc.append(&numbered(3 * RETAIN_WINDOW_CHUNK)?)?;
        assert_eq!(
            sc.first_cached,
            SingleCache::window_target_first_cached(3 * RETAIN_WINDOW_CHUNK, retain)
        );
        assert!(sc.physical_seq_len() < 3 * RETAIN_WINDOW_CHUNK);
        Ok(())
    }

    /// `set_len` below the retained base is a refusal, not a silent
    /// truncation: the caller (prefix cacher) has a documented decline path,
    /// and answering with rows the slot no longer holds is unrecoverable.
    /// An MTP-sized rollback stays inside the margin and must still work.
    #[test]
    // Row values are small integer positions (< a few thousand), well inside
    // f32's exact range; the cast is how a position is made readable as data.
    #[allow(clippy::cast_precision_loss)]
    fn set_len_refuses_a_rollback_past_the_retained_base() -> Result<()> {
        let retain = 8usize;
        let mut sc = windowed(retain);
        for n in 0..(2 * RETAIN_WINDOW_CHUNK) {
            sc.append(&numbered(n)?)?;
        }
        assert!(
            sc.first_cached > 0,
            "fixture never evicted; test has no teeth"
        );

        let err = sc.set_len(sc.first_cached - 1).unwrap_err().to_string();
        assert!(
            err.contains("below this slot's retained window"),
            "expected a loud retained-window refusal, got: {err}"
        );

        // …and a rollback inside the retained run is exact.
        let back = sc.current_seq_len - 4;
        sc.set_len(back)?;
        assert_eq!(sc.current_seq_len, back);
        let live = sc.current_data()?.unwrap();
        let got: Vec<f32> = live.narrow(3, 0, 1)?.flatten_all()?.to_vec1()?;
        let want: Vec<f32> = (sc.first_cached..back).map(|i| i as f32).collect();
        assert_eq!(got, want);
        Ok(())
    }

    /// `retain_window: None` — every model but V4-under-the-flag — must be
    /// byte-identical to the pre-change cache. Pinned against a windowed cache
    /// of the same length so a mutation that windowed unconditionally fails.
    #[test]
    fn an_unwindowed_cache_is_unchanged_and_a_windowed_one_is_not() -> Result<()> {
        let mut plain = SingleCache::new(2, 4096, 8);
        let mut win = windowed(8);
        for n in 0..(2 * RETAIN_WINDOW_CHUNK) {
            plain.append(&numbered(n)?)?;
            win.append(&numbered(n)?)?;
        }
        assert_eq!(plain.first_cached, 0);
        assert_eq!(plain.physical_seq_len(), plain.current_seq_len);
        assert_eq!(
            plain.current_data()?.unwrap().dim(2)?,
            2 * RETAIN_WINDOW_CHUNK
        );
        assert_eq!(plain.current_seq_len, win.current_seq_len);
        assert!(
            win.physical_seq_len() < plain.physical_seq_len(),
            "the windowed cache retained as much as the plain one; nothing was saved"
        );
        Ok(())
    }

    /// The graph-decode append writes at an absolute device slot, so it must
    /// refuse a windowed buffer rather than write to the wrong row.
    #[test]
    fn append_graph_refuses_a_windowed_slot() -> Result<()> {
        let mut sc = windowed(8);
        sc.append(&numbered(0)?)?;
        let position = Tensor::from_vec(vec![0u32], (1usize,), &Device::Cpu)?;
        let err = sc
            .append_graph(&numbered(1)?, &position, 8)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("cannot address a windowed cache"),
            "expected a loud graph-capture refusal, got: {err}"
        );
        Ok(())
    }
}
