//! Rolling compressed state for the DeepSeek V4 compressor input (`xs`).
//!
//! # Why this exists
//!
//! V4's CSA/HCA layers feed a *compressor* (`V4Compressor::forward_from_xs`)
//! the history of the attention-block input, and use its output as a second,
//! distant-context key set inside the same softmax. The first correct
//! implementation (R3, PR #21) kept that history verbatim: one
//! `[B, T, hidden]` buffer per compressed layer, re-running the whole
//! compressor over the whole history on every decode step.
//!
//! That buffer, not the KV cache, is what caps batch size. Per token per
//! layer it is `hidden * 2 = 8192 B` against the KV's
//! `head_dim * 2 * 2 = 2048 B` — 4.0x — and 41 of V4's 43 layers carry one.
//!
//! # What the compressor actually needs
//!
//! `forward_from_xs` is a *strided local reduction*: output row `j` is a
//! softmax-weighted pool over the `ratio` tokens of group `j` (plus, when
//! `ratio == 4`, the `ratio` tokens of group `j - 1`, via
//! `overlap_transform`), followed by an RMSNorm. Row `j` is therefore final
//! the moment token `(j + 1) * ratio - 1` arrives and never changes again,
//! and the *only* raw tokens a future row can need are the last
//! `span_groups * ratio` of them.
//!
//! So the raw history is not state — it is a recompute buffer. This cache
//! keeps the state instead:
//!
//! * `comp` — the completed compressed rows, `[B, G, head_dim]` with
//!   `G = tokens / ratio`. `head_dim / ratio` bytes per token per layer
//!   instead of `hidden`: `8 * ratio` times smaller (32x at `ratio = 4`,
//!   1024x at `ratio = 128` for V4's `hidden = 4096`, `head_dim = 512`).
//! * `tail` — the raw rows for tokens `[base, tokens)`, bounded by
//!   `span_groups * ratio + margin` and independent of context length.
//!
//! # The contract with the rest of the engine
//!
//! Every cache entry in a `NormalCache` is truncated in **token** units by
//! the prefix cacher (`prefix_cacher.rs:316-321`), the MTP verify rollback
//! (`pipeline/mtp_pipeline.rs:1115-1127`) and speculative rejection
//! (`pipeline/speculative.rs:810,892`). This type therefore reports
//! `current_seq_len()` in tokens and maps a token-length truncation onto its
//! own two time bases itself — `comp` to `len / ratio` rows, `tail` to
//! `len - base` rows — and *refuses* (`try_set_len` → `Err`) any truncation
//! that would land before `base`, where the raw rows needed to resume are
//! genuinely gone. The prefix cacher already treats that refusal as "no
//! match, prefill from scratch" (it is how every rotating/sliding-window
//! cache behaves); `margin` is what keeps the short MTP/speculative
//! rollbacks — which cannot decline — inside the retained window.
//!
//! Silently accepting such a truncation would resume the compressor from a
//! gap, i.e. corrupt the distant-context branch with no error. It is the
//! failure mode this type is written to make impossible.

use candle_core::{Result, Tensor};

use super::{NormalCache, SingleCache};

/// Extra raw tokens retained behind the compressor's own requirement, so a
/// short rollback stays recoverable. MTP/speculative rejection drops at most
/// the number of drafted positions; 16 covers any configured MTP depth
/// (`--mtp-depth`) with room to spare, at a cost of `16 * hidden * 2 B` per
/// compressed layer (128 KB/layer/seq at V4's `hidden = 4096`).
pub const XS_TAIL_MARGIN_TOKENS: usize = 16;

/// Rolling compressed-history state for one V4 compressor layer, for one
/// sequence. See the module docs.
#[derive(Debug, Clone)]
pub struct XsRollingCache {
    /// Completed compressed rows `[B, G, head_dim]` on seq dim 1.
    /// `comp.current_seq_len()` counts ROWS (groups), not tokens.
    pub comp: SingleCache,
    /// Raw compressor-input rows for tokens `[base, tokens)`,
    /// `[B, tokens - base, hidden]` on seq dim 1. `None` before the first
    /// token.
    pub tail: Option<Tensor>,
    /// Index of the first token retained in `tail`.
    pub base: usize,
    /// Total tokens seen — the length every other part of the engine means.
    pub tokens: usize,
    /// Compression ratio of the owning layer (V4: 4 for CSA, 128 for HCA).
    pub ratio: usize,
    /// How many consecutive groups of raw tokens one compressed row consumes:
    /// 2 for the overlapping (`ratio == 4`) compressor, 1 otherwise.
    pub span_groups: usize,
    /// Extra retained tokens for rollback ([`XS_TAIL_MARGIN_TOKENS`]).
    pub margin: usize,
    /// Width of a compressed row. Known at load time, and needed to
    /// materialise `comp.all_data` before the first group completes so the
    /// cache managers always have a tensor to batch.
    pub head_dim: usize,
}

impl XsRollingCache {
    pub fn new(
        ratio: usize,
        span_groups: usize,
        head_dim: usize,
        max_position_embeddings: usize,
    ) -> Self {
        assert!(ratio > 0, "XsRollingCache: ratio must be > 0");
        assert!(
            span_groups >= 1,
            "XsRollingCache: span_groups must be >= 1 (one row always needs its own group)"
        );
        // Rows, not tokens. `SingleCache::append` grows in `CACHE_GROW_SIZE`
        // blocks and bails if that overshoots `max_seq_len`, so round the row
        // ceiling up to a whole number of blocks.
        let grow = NormalCache::CACHE_GROW_SIZE;
        let max_rows = max_position_embeddings.div_ceil(ratio).div_ceil(grow) * grow;
        // Small initial allocation: an HCA layer (ratio 128) needs 16 rows at
        // a 2048-token context, and paying a 512-row block for it would give
        // back a third of the win.
        let init_rows = 64.min(max_rows);
        Self {
            comp: SingleCache::new(1, max_rows, init_rows),
            tail: None,
            base: 0,
            tokens: 0,
            ratio,
            span_groups,
            margin: XS_TAIL_MARGIN_TOKENS,
            head_dim,
        }
    }

    /// Tokens seen — what `set_len` / the schedulers / the prefix cacher mean
    /// by "length".
    pub fn current_seq_len(&self) -> usize {
        self.tokens
    }

    pub fn reset(&mut self) {
        self.comp.reset();
        self.tail = None;
        self.base = 0;
        self.tokens = 0;
    }

    /// First token index whose raw row is still retained. A truncation to a
    /// length below this cannot be resumed.
    pub fn resumable_from(&self) -> usize {
        self.base
    }

    pub fn try_set_len(&self, len: usize) -> Result<()> {
        if len > self.tokens {
            candle_core::bail!(
                "xs rolling cache: cannot extend from {} to {len} tokens",
                self.tokens
            );
        }
        // Two things have to survive the rollback: the raw rows the truncated
        // tail is cut from (`len >= base`), and the raw rows the NEXT
        // compressed row will be built from — group `len/ratio + 1 -
        // span_groups` onward, which for the overlapping compressor reaches a
        // whole group further back than `len` itself. Checking only the first
        // would let `set_len` succeed and leave the gap for `advance` to hit.
        let resume_from = (len / self.ratio + 1).saturating_sub(self.span_groups) * self.ratio;
        if len < self.base || resume_from < self.base {
            candle_core::bail!(
                "xs rolling cache: rollback to {len} tokens is behind the retained raw window \
                 (resuming needs the compressor input from token {resume_from}, retained from \
                 {}, currently at {} tokens). Those rows are gone, so resuming would silently \
                 skip history; recompute the prefix instead.",
                self.base,
                self.tokens
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> Result<()> {
        self.try_set_len(len)?;
        if len == self.tokens {
            return Ok(());
        }
        // Compressed rows are token-aligned: row `j` covers tokens
        // `[j*ratio, (j+1)*ratio)`, so exactly `len / ratio` of them survive.
        self.comp.set_len(len / self.ratio)?;
        let keep = len - self.base;
        self.tail = match self.tail.take() {
            Some(t) if keep > 0 => Some(t.narrow(1, 0, keep)?.contiguous()?),
            _ => None,
        };
        self.tokens = len;
        Ok(())
    }

    /// The compressed rows `[B, G, head_dim]`, or `None` when no group has
    /// completed yet (the sliding-window branch alone covers the context).
    pub fn compressed_rows(&self) -> Result<Option<Tensor>> {
        if self.comp.current_seq_len() == 0 {
            return Ok(None);
        }
        self.comp.current_data()
    }

    /// Append `xs_new` (`[B, T_new, hidden]`), compress every group that
    /// completes, drop the raw rows no future row can need, and return the
    /// full compressed history `[B, G, head_dim]` (`None` while `G == 0`).
    ///
    /// `compress` is the layer's `V4Compressor::forward_from_xs`: it maps
    /// `[B, k*ratio, hidden]` to `[B, k, head_dim]`, and its row `i` depends
    /// only on the window's groups `i` and (when `span_groups == 2`) `i - 1`
    /// — with group `-1` of the window filled by `overlap_transform`'s pad.
    /// This function therefore always hands it `span_groups - 1` groups of
    /// real context ahead of the rows it keeps, and discards the padded
    /// leading row, EXCEPT at the true start of the sequence where the pad is
    /// the correct input.
    pub fn advance<F>(&mut self, xs_new: &Tensor, compress: F) -> Result<Option<Tensor>>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        let (b, t_new, _hidden) = xs_new.dims3()?;
        if t_new == 0 {
            return self.compressed_rows();
        }
        let xs_new = xs_new.contiguous()?;

        // `window` covers tokens [win_start, tokens + t_new).
        let (window, win_start) = match self.tail.as_ref() {
            Some(tail) if self.tokens > self.base => (Tensor::cat(&[tail, &xs_new], 1)?, self.base),
            _ => (xs_new, self.tokens),
        };
        let tokens_new = self.tokens + t_new;

        let g_done = self.comp.current_seq_len();
        let g_target = tokens_new / self.ratio;
        if g_target > g_done {
            // Rows g_done..g_target need raw tokens from group
            // `g_done + 1 - span_groups` onward.
            let need_start = (g_done + 1).saturating_sub(self.span_groups) * self.ratio;
            if need_start < win_start {
                candle_core::bail!(
                    "xs rolling cache: compressor history gap — row {g_done} needs tokens from \
                     {need_start} but the retained window starts at {win_start}. This means a \
                     truncation was applied without going through `set_len`."
                );
            }
            let off = need_start - win_start;
            let len = g_target * self.ratio - need_start;
            let slice = window.narrow(1, off, len)?.contiguous()?;
            let rows = compress(&slice)?;
            let n_rows = rows.dim(1)?;
            let take = g_target - g_done;
            if n_rows < take {
                candle_core::bail!(
                    "xs rolling cache: compressor returned {n_rows} rows for a {len}-token \
                     window (ratio {}), expected at least {take}",
                    self.ratio
                );
            }
            // Drop the leading row(s) computed against `overlap_transform`'s
            // pad rather than against real predecessor tokens.
            let rows = rows.narrow(1, n_rows - take, take)?.contiguous()?;
            self.comp.append(&rows)?;
        }
        // Materialise the row buffer even when no group has completed yet:
        // `NormalCacheManager` batches `all_data` directly and unwraps it, so
        // a `None` here would panic the first time a sub-`ratio` prompt is
        // cloned out. Same shape `SingleCache::append` would have allocated.
        if self.comp.all_data.is_none() {
            self.comp.all_data = Some(Tensor::zeros(
                (b, self.comp.capacity_seq_len, self.head_dim),
                window.dtype(),
                window.device(),
            )?);
        }

        // Retain exactly what a future row can consume — including after a
        // rollback of up to `margin` tokens.
        //
        // The subtlety: "keep the last `span_groups * ratio + margin` tokens"
        // is NOT the same rule. A rollback that crosses a group boundary
        // invalidates that group's compressed row, and rebuilding it needs the
        // group's raw tokens from its START, not `margin` of them. So the
        // retention point is the one the WORST rollback would need — i.e. the
        // one computed at `tokens - margin`, not at `tokens`. It costs
        // nothing: the tail is only extended backwards during the first
        // `margin` tokens of a group, when the current group is that much
        // shorter, so the bound stays `span_groups * ratio + margin`.
        let rollback_floor = tokens_new.saturating_sub(self.margin);
        let keep_from = ((rollback_floor / self.ratio + 1).saturating_sub(self.span_groups)
            * self.ratio)
            .max(win_start);
        let new_base = keep_from.min(tokens_new);
        self.tail = Some(
            window
                .narrow(1, new_base - win_start, tokens_new - new_base)?
                .contiguous()?,
        );
        self.base = new_base;
        self.tokens = tokens_new;
        debug_assert_eq!(b, self.tail.as_ref().unwrap().dim(0)?);

        self.compressed_rows()
    }
}
