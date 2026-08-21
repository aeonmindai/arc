use std::{
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

use crate::{
    engine::{IntervalLogger, TERMINATE_ALL_NEXT_STEP},
    paged_attention::KVCacheManager,
    sequence::{Sequence, SequenceState, StopReason},
};

use super::{Scheduler, SchedulerOutput};

pub trait FcfsBacker: Default {
    fn new() -> Self;
    fn add(&mut self, item: Sequence);
    fn into_iter(self) -> impl Iterator<Item = Sequence>;
    fn len(&self) -> usize;
    fn sort_ascending_ids(&mut self);
}

impl FcfsBacker for VecDeque<Sequence> {
    fn new() -> Self {
        Self::new()
    }
    fn add(&mut self, item: Sequence) {
        self.push_back(item)
    }
    fn into_iter(self) -> impl Iterator<Item = Sequence> {
        <Self as IntoIterator>::into_iter(self)
    }
    fn sort_ascending_ids(&mut self) {
        let slice = self.make_contiguous();
        slice.sort_by_key(|seq| *seq.id());
    }
    fn len(&self) -> usize {
        VecDeque::len(self)
    }
}

pub struct DefaultSchedulerOutput<'a> {
    pub completion: Box<[&'a mut Sequence]>,
    pub prompt: Box<[&'a mut Sequence]>,
}

/// The scheduler method controld how sequences are scheduled during each
/// step of the engine. For each scheduling step, the scheduler method is used if there
/// are not only running, only waiting sequences, or none. If is it used, then it
/// is used to allow waiting sequences to run.
#[derive(Clone)]
pub enum DefaultSchedulerMethod {
    Fixed(NonZeroUsize),
}

pub struct BucketedSeqs<Backer: FcfsBacker> {
    running: Vec<Sequence>,
    waiting: Backer,
}

pub trait BucketingManager<Backer: FcfsBacker>: Send + Sync {
    /// Bucket and waitlist running input sequences, returning the newly running sequences.
    fn bucket_and_waitlist_seqs_waiting(
        &mut self,
        running: Vec<Sequence>,
        waiting: Backer,
        discrete: bool,
    ) -> BucketedSeqs<Backer>;
}

// (cache length, (has_imgs && is_prompt), sequence offset)
// Bucket by that metric for images because if we are not a prompt, then this doesn't apply
type BucketKey = (usize, bool, usize);

/// How many waiting sequences may be admitted to **prefill** in one engine
/// iteration. `None` (the default, and the historical behaviour) means "all of
/// them", which is where head-of-line blocking comes from.
///
/// # Why this exists
///
/// A prompt step is uninterruptible: `Engine::run`'s prompt branch calls
/// `pipeline.step(.., is_prompt = true, ..)` once, holding the pipeline mutex
/// for the whole prefill, and every scheduled sequence goes straight from
/// `RunningPrompt` to `RunningCompletion`. There is no partially-prefilled
/// state — `Sequence::token_offset` was write-once (set by the prefill-prompt
/// restore builder and never advanced; there was no setter at all), so it read
/// 0 on every live path and a prompt could not yield mid-flight. Admitting K
/// prompts therefore stops decode for as long as prefilling all K takes.
///
/// (This PR adds `Sequence::set_token_offset` and advances it per chunk, so
/// the mid-flight yield now exists — but only when `ARC_PREFILL_CHUNK` is set,
/// which it should not be yet; see `prefill_chunk_size` for why. Unset, the
/// paragraph above still describes the shipped behaviour exactly.)
///
/// Measured on an H200 (2026-08-17, `qtip2b`, MTP depth 3, profiler PR #113):
/// at K=32 with 256-word prompts, **ONE prompt step took 43.2 s — 50.3% of the
/// profiled window** — and at K=128 prefill ran ~120 s while the client
/// received **zero tokens for 70 s**.
///
/// Capping admission is the *request-level* half of chunked prefill: it does
/// not split an individual prompt (that needs the token-level cursor described
/// in `get_prompt_input`), but it bounds how much prefill can accumulate in one
/// uninterruptible step, so decode gets a turn between groups. It trades
/// time-to-first-token for the last-admitted requests against decode
/// availability for everyone already running, and **both numbers have to be
/// reported** — see `arc-tools`.
///
/// Read once from `ARC_PREFILL_MAX_SEQS`; unset or `0` reproduces the previous
/// admission exactly, expression for expression.
fn prefill_admission_cap() -> Option<usize> {
    static CAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("ARC_PREFILL_MAX_SEQS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
    })
}

/// Horizon, in decode steps, over which a coalescing choice must pay for itself.
///
/// See [`select_running_bucket`]. Sized as roughly one completion: a choice that
/// repays within a few hundred steps is worth taking, one that does not is not.
const COALESCE_PAYBACK_STEPS: usize = 256;

/// Pick the bucket that runs this step.
///
/// Sequences may only share a forward pass when their cache lengths are exactly
/// equal — `NormalCacheManager::clone_in_cache`
/// (`mistralrs-core/src/kv_cache/mod.rs`) builds one dense `[B*.., H, L, D]`
/// batch cache using `seqs[0]` as the template for `current_seq_len`, and
/// `SingleCache::append` writes every sequence's new K/V at that single shared
/// offset. Two different lengths in one forward therefore write to the wrong
/// slot and attend over the wrong window, silently. So exactly one bucket runs.
///
/// The consequence is that *which* bucket runs decides whether the split ever
/// heals. Running a bucket advances it by one token, so running the **shortest**
/// bucket closes the gap to the next-shortest and the two merge into a single
/// bucket that thereafter runs together, permanently. Running the highest-
/// priority bucket does not: with two equal-sized buckets one length apart,
/// `compute_priority` (`scheduling_urgency + log2(len)`) makes them alternate
/// perfectly — each advances one token every two steps, the gap stays at one,
/// and half the admitted batch idles forever. That is the measured H200
/// steady state, `32 running, 32 waiting` at B=64
/// (`memory/mission/wave26-AX-h200-measurement.md`).
///
/// So: take the greedy highest-priority bucket, *except* when running the
/// shortest bucket would merge it into the next-shortest soon enough to pay for
/// itself. Coalescing idles `total - n_min` sequences for `gap` steps and then
/// adds `n_min` sequences to every later forward, so the override is taken iff
///
/// ```text
/// (total - n_min) * gap  <=  n_min * COALESCE_PAYBACK_STEPS
/// ```
///
/// For the measured case (32 + 32, gap 1) that is `32 <= 8192` — taken, and the
/// batch is whole on the next step. For a fresh 21-token arrival against a
/// 63-sequence cohort at length 500 it is `30177 <= 256` — refused, and the
/// cohort keeps running exactly as before.
fn select_running_bucket(
    seq_buckets: &HashMap<BucketKey, Vec<Sequence>>,
    seq_priorities: &HashMap<BucketKey, f64>,
    discrete: bool,
) -> BucketKey {
    let min = *seq_buckets
        .keys()
        .min_by_key(|(x, _, _)| *x)
        .expect("No sequence buckets.");
    if discrete {
        return min;
    }

    let greedy = seq_priorities
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(k, _)| *k)
        .unwrap_or(min);
    if greedy == min {
        return min;
    }

    // A merge only happens when the *whole* key matches, so the coalescing
    // target must agree on the image flag and token offset too.
    let Some(next_len) = seq_buckets
        .keys()
        .filter(|(len, imgs, offset)| *len > min.0 && *imgs == min.1 && *offset == min.2)
        .map(|(len, _, _)| *len)
        .min()
    else {
        return greedy;
    };

    let n_min = seq_buckets[&min].len();
    let total: usize = seq_buckets.values().map(|v| v.len()).sum();
    let gap = next_len - min.0;

    if (total - n_min).saturating_mul(gap) <= n_min.saturating_mul(COALESCE_PAYBACK_STEPS) {
        min
    } else {
        greedy
    }
}

/// The default prompt-starvation floor, in decode steps: a prompt bucket is
/// forced to run after this many consecutive pass-overs.
///
/// `4` was picked against this file's own fixture (see
/// `the_floor_gives_prompts_a_turn_without_giving_them_every_turn`): behind a
/// 47-sequence decode cohort a fresh prompt wins 0 of 24 steps with no floor,
/// and 1 in every 5 with floor=4 — a turn without taking every turn.
const DEFAULT_PREFILL_FLOOR_STEPS: usize = 4;

/// How many consecutive iterations prompt buckets may be passed over before one
/// is forced to run. `None` means the historical behaviour: prompts win a step
/// only by out-scoring every decode bucket on priority.
///
/// # Why a floor is needed as well as a cap
///
/// `ARC_PREFILL_MAX_SEQS` bounds how much prefill may enter ONE uninterruptible
/// step. It is a throttle, and on its own it cannot make prompts run — that is
/// a different failure in the opposite direction.
///
/// `select_running_bucket` takes the bucket with the highest SUMMED priority
/// (`scheduling_urgency + log2(len)` per sequence). Once a large decode cohort
/// exists, its sum dominates any prompt bucket: at 47 decoding sequences the
/// decode bucket scores ~47·log2(L) before urgency, while a fresh prompt bucket
/// starts near zero and has to accumulate urgency for hundreds of steps to
/// catch up. Prompts starve behind decode.
///
/// That is the ceiling `feat/dense-ragged-decode` hit. With the decode limiter
/// removed its cohort climbs `16 running, 112 waiting` → `47 running, 48
/// waiting` and **stops**, with 48 sequences admitted and never prefilled.
/// Capping admission cannot lift it; only guaranteeing prompts a turn can.
///
/// Read once from `ARC_PREFILL_FLOOR_STEPS`, **by value**:
/// * unset (or unparseable) → [`DEFAULT_PREFILL_FLOOR_STEPS`] — the floor is
///   ON by default, because the starvation it lifts is the default behaviour;
/// * `0` → `None`, the kill-switch: reproduces the pre-floor selection
///   exactly, key for key;
/// * any other `n` → `Some(n)`.
fn prefill_starvation_floor() -> Option<usize> {
    static FLOOR: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| floor_from(std::env::var("ARC_PREFILL_FLOOR_STEPS").ok().as_deref()))
}

/// The decision itself, separated from the process-global `OnceLock` and the
/// environment so both sides of the default can be pinned by a test.
fn floor_from(raw: Option<&str>) -> Option<usize> {
    match raw.map(str::parse::<usize>) {
        Some(Ok(0)) => None,
        Some(Ok(n)) => Some(n),
        Some(Err(_)) | None => Some(DEFAULT_PREFILL_FLOOR_STEPS),
    }
}

#[derive(Default)]
struct FixedBucketingManager {
    /// Iterations since a prompt bucket last ran. Lives on the manager because
    /// the scheduler owns it for the life of the engine; a free function cannot
    /// carry it.
    steps_since_prompt: usize,
    /// The floor, resolved ONCE at construction rather than read from a global
    /// on every selection — so a test can build a manager with a floor and
    /// exercise the arithmetic, instead of racing a process-global `OnceLock`
    /// that another test in the same binary may already have latched.
    floor: Option<usize>,
}

impl FixedBucketingManager {
    fn new() -> Self {
        Self {
            steps_since_prompt: 0,
            floor: prefill_starvation_floor(),
        }
    }
}

impl<Backer: FcfsBacker> BucketingManager<Backer> for FixedBucketingManager {
    /// Move the sequences into buckets, and run the ones with the shortest lengths.
    /// The others are moved to the waiting list (retaining high priority due to start time),
    /// without a state modification.
    fn bucket_and_waitlist_seqs_waiting(
        &mut self,
        running: Vec<Sequence>,
        mut waiting: Backer,
        discrete: bool,
    ) -> BucketedSeqs<Backer> {
        // Now, get the sequences with the smallest sequence lengths, and allow them to catch up.
        let mut seq_buckets: HashMap<BucketKey, Vec<Sequence>> = HashMap::new();
        let mut seq_priorities: HashMap<BucketKey, f64> = HashMap::new();
        let ragged_decode = crate::kv_cache::ragged_decode_supported();
        for seq in running {
            // The cache length, not the token count — see
            // `Sequence::cache_bucket_len`. Identical partition for every
            // non-speculative path; the one that keeps a batched MTP cohort
            // whole when its accept lengths differ.
            //
            // 🔑 Pinned to 0 for DECODE when this pipeline's cache can carry
            // per-sequence lengths, so a cohort that differs only in length
            // forms a single bucket and nothing is preempted.
            // `clone_in_cache` then front-aligns it and `CausalMasker` masks
            // each row's dead prefix. PROMPTS keep the exact-length key: a
            // prefill batch is right-padded and its `cu_seqlens` are built from
            // those padded widths, which is a different mechanism.
            let len = if ragged_decode && !seq.is_prompt() {
                0
            } else {
                seq.cache_bucket_len()
            };
            match seq_buckets.get_mut(&(
                len,
                seq.images().is_some() && seq.is_prompt(),
                seq.token_offset(),
            )) {
                Some(bucket) => {
                    if !discrete {
                        *seq_priorities
                            .get_mut(&(
                                len,
                                seq.images().is_some() && seq.is_prompt(),
                                seq.token_offset(),
                            ))
                            .unwrap() += seq.compute_priority();
                    }
                    bucket.push(seq);
                }
                None => {
                    if !discrete {
                        seq_priorities.insert(
                            (
                                len,
                                seq.images().is_some() && seq.is_prompt(),
                                seq.token_offset(),
                            ),
                            seq.compute_priority(),
                        );
                    }
                    seq_buckets.insert(
                        (
                            len,
                            seq.images().is_some() && seq.is_prompt(),
                            seq.token_offset(),
                        ),
                        vec![seq],
                    );
                }
            }
        }
        // Engagement instrumentation, paired with the `ArcKV` line in
        // `clone_in_cache`. That one reports the capability the model published;
        // this one reports whether the scheduler ACTED on it. A ragged decode
        // cohort must collapse to exactly ONE bucket — several buckets while
        // `ragged_decode` is true means the gate resolved but the bucket key did
        // not, which is a different defect from the flag never taking at all.
        //
        // Logged on transition for the same reason as the `ArcKV` line: this is
        // once per scheduling decision. Oscillation is itself signal, so a
        // flapping bucket count is deliberately allowed to emit.
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static LAST: AtomicU64 = AtomicU64::new(u64::MAX);
            static DECISIONS: AtomicU64 = AtomicU64::new(0);
            let n = DECISIONS.fetch_add(1, Ordering::Relaxed);
            let buckets = seq_buckets.len() as u64;
            let now = (buckets << 1) | u64::from(ragged_decode);
            if LAST.swap(now, Ordering::Relaxed) != now {
                tracing::info!(
                    ragged_decode,
                    buckets,
                    seqs = seq_buckets.values().map(Vec::len).sum::<usize>(),
                    decisions = n,
                    "ArcSched: decode bucketing"
                );
            }
        }
        let running = if seq_buckets.len() <= 1 {
            // Full steam ahead or have everything
            seq_buckets
                .into_iter()
                .flat_map(|(_, x)| x)
                .map(|s| s.reset_urgency())
                .collect::<Vec<_>>()
        } else {
            // Set the winning bucket to be the running ones, and the rest to be waiting (but their
            // states are not changed!). See `select_running_bucket` for why the choice matters.
            let mut len = select_running_bucket(&seq_buckets, &seq_priorities, discrete);

            // Anti-starvation floor. Applied AFTER the ordinary choice and only
            // when that choice was not already a prompt bucket, so with the
            // floor unset — or before it expires — the selection is the
            // pre-change one, key for key.
            if let Some(floor) = self.floor.filter(|_| !discrete) {
                let is_prompt_bucket = |k: &BucketKey| {
                    seq_buckets
                        .get(k)
                        .and_then(|v| v.first())
                        .is_some_and(Sequence::is_prompt)
                };
                if is_prompt_bucket(&len) {
                    self.steps_since_prompt = 0;
                } else if self.steps_since_prompt >= floor {
                    // Take the SHORTEST waiting prompt bucket: it is the one
                    // that costs least to clear and the one most likely to
                    // merge with the running cohort afterwards.
                    if let Some(k) = seq_buckets
                        .keys()
                        .filter(|k| is_prompt_bucket(k))
                        .min_by_key(|(l, _, _)| *l)
                        .copied()
                    {
                        len = k;
                        self.steps_since_prompt = 0;
                        // 🔑 ENGAGEMENT, logged once. A run where the floor
                        // never fires and a run where the floor is not on the
                        // code path at all produce the SAME numbers, and this
                        // chain has already published one set of those: five
                        // cells measured with PagedAttention live, so
                        // `DefaultScheduler` — this whole file — never
                        // executed. "No effect" must be distinguishable from
                        // "never ran", by a positive signal rather than by the
                        // absence of a negative one.
                        static FIRED: std::sync::Once = std::sync::Once::new();
                        FIRED.call_once(|| {
                            tracing::info!(
                                "ARC prefill floor: forced a prompt bucket after {} passes \
                                 (ARC_PREFILL_FLOOR_STEPS={floor}); logged once",
                                floor
                            );
                        });
                    } else {
                        self.steps_since_prompt += 1;
                    }
                } else {
                    self.steps_since_prompt += 1;
                }
            }
            let highest_priority_seqs = seq_buckets
                .remove(&len)
                .unwrap()
                .into_iter()
                .map(|s| s.reset_urgency())
                .collect();
            for (_, seqs) in seq_buckets {
                for seq in seqs {
                    waiting.add(seq.add_urgency());
                }
            }
            // Know min_seqs.len < running.len() <= max
            highest_priority_seqs
        };
        BucketedSeqs { running, waiting }
    }
}

pub struct DefaultScheduler<Backer: FcfsBacker> {
    waiting: Backer,
    running: Vec<Sequence>,
    method: DefaultSchedulerMethod,
    bucketing_manager: Box<dyn BucketingManager<Backer>>,
}

impl<Backer: FcfsBacker> DefaultScheduler<Backer> {
    pub fn new(method: DefaultSchedulerMethod) -> Self {
        let bucketing_manager: Box<dyn BucketingManager<_>> = match method {
            DefaultSchedulerMethod::Fixed(_) => Box::new(FixedBucketingManager::new()),
        };
        Self {
            running: Vec::new(),
            waiting: Backer::new(),
            method,
            bucketing_manager,
        }
    }

    /// Move the sequences into buckets, and run the ones with the shortest lengths.
    /// The others are moved to the waiting list (retaining high priority due to start time),
    /// without a state modification.
    fn bucket_and_waitlist_seqs(&mut self, running: Vec<Sequence>) -> Vec<Sequence> {
        let waiting = std::mem::take(&mut self.waiting);
        let BucketedSeqs { running, waiting } = self
            .bucketing_manager
            .bucket_and_waitlist_seqs_waiting(running, waiting, true);
        self.waiting = waiting;
        running
    }

    /// Schedule all sequences based on their state and the available space.
    pub fn schedule(&mut self, logger: &IntervalLogger) -> DefaultSchedulerOutput<'_> {
        // Filter out all done sequences
        let running = std::mem::take(&mut self.running);
        let mut waiting = std::mem::take(&mut self.waiting);

        // REAP ABANDONED SEQUENCES — must run before the `is_running` filter,
        // because an abandoned sequence IS still running: nothing has marked it
        // done. See `Sequence::client_is_gone` for why the engine cannot learn
        // this by failing to send.
        //
        // Left unreaped, one abandoned request costs three things: a scheduler
        // slot, its KV block until `max_tokens`, and — because
        // `bucket_and_waitlist_seqs_waiting` partitions on `cache_bucket_len` —
        // a whole BUCKET pinned at a length nothing real needs, which degrades
        // the batch shape for every live request.
        //
        // Only `running` is swept. An abandoned sequence sitting in `waiting`
        // holds no KV, and is promoted to `running` by this same call, so it is
        // reaped on the next pass — one scheduling cycle later, at no risk to
        // the waiting list's ordering.
        let abandoned: Vec<usize> = running
            .iter()
            .filter(|seq| seq.is_running() && seq.client_is_gone())
            .map(|seq| *seq.id())
            .collect();
        if !abandoned.is_empty() {
            for seq in running
                .iter()
                .filter(|s| s.is_running() && s.client_is_gone())
            {
                seq.set_state(SequenceState::Done(StopReason::Canceled));
            }
            tracing::warn!(
                "Reaped {} sequence(s) whose client had already disconnected: ids {abandoned:?}. \
                 Their scheduler slots and KV are released now rather than at max_tokens. \
                 {} sequence(s) remain running.",
                abandoned.len(),
                running.len() - abandoned.len(),
            );
        }

        let mut running = running
            .into_iter()
            .filter(|seq| seq.is_running())
            .collect::<Vec<_>>();

        match (waiting.len(), running.len()) {
            (0, 0) => {
                self.running = running;
                logger.set_num_running(self.running.len());
                logger.set_num_waiting(self.waiting.len());
                return DefaultSchedulerOutput {
                    prompt: vec![].into(),
                    completion: vec![].into(),
                };
            }
            (_, 0) => {
                // Cold start: nothing is decoding, so nothing is starved by a
                // large prefill — but the cap still applies, because the group
                // admitted here is exactly the group that will be decoding when
                // the next group arrives.
                let cap = prefill_admission_cap().unwrap_or(usize::MAX);
                let mut held = Backer::new();
                for (i, seq) in waiting.into_iter().enumerate() {
                    if i < cap {
                        seq.set_state(SequenceState::RunningPrompt);
                        self.running.push(seq);
                    } else {
                        held.add(seq);
                    }
                }
                self.waiting = held;
                let running = std::mem::take(&mut self.running);
                self.running = self.bucket_and_waitlist_seqs(running);
                logger.set_num_running(self.running.len());
                logger.set_num_waiting(self.waiting.len());
                return DefaultSchedulerOutput {
                    prompt: self.running.iter_mut().collect::<Vec<_>>().into(),
                    completion: vec![].into(),
                };
            }
            (0, _) => {
                self.running = self.bucket_and_waitlist_seqs(running);
                if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
                    self.running
                        .iter_mut()
                        .for_each(|seq| seq.set_state(SequenceState::Done(StopReason::Canceled)));
                    TERMINATE_ALL_NEXT_STEP.store(false, Ordering::SeqCst);
                }
                logger.set_num_running(self.running.len());
                logger.set_num_waiting(self.waiting.len());
                return DefaultSchedulerOutput {
                    prompt: vec![].into(),
                    completion: self.running.iter_mut().collect::<Vec<_>>().into(),
                };
            }
            _ => {}
        }

        // Sort the waiting seqs
        waiting.sort_ascending_ids();

        // If the waiting sequence will fit, add it. Otherwise remove it
        let mut new_waiting = Backer::new();
        let cap = prefill_admission_cap().unwrap_or(usize::MAX);
        let mut admitted_to_prefill = 0usize;
        for seq in waiting.into_iter() {
            // A sequence that is already running its prompt is not *newly*
            // admitted and must not be counted against the cap, or a cohort
            // mid-prefill would be re-queued behind itself.
            let is_new_prefill = seq.is_waiting();
            let capped = is_new_prefill && admitted_to_prefill >= cap;
            if self.sequence_fits(&running, &seq) && !capped {
                if is_new_prefill {
                    seq.set_state(SequenceState::RunningPrompt);
                    admitted_to_prefill += 1;
                }
                running.push(seq);
            } else {
                if capped {
                    // Same reason as the floor: a cap that held nothing back
                    // and a cap that was never on the code path are the same
                    // number otherwise.
                    static HELD: std::sync::Once = std::sync::Once::new();
                    HELD.call_once(|| {
                        tracing::info!(
                            "ARC prefill cap: held a waiting sequence back at {} admitted \
                             this iteration (ARC_PREFILL_MAX_SEQS); logged once",
                            admitted_to_prefill
                        );
                    });
                }
                new_waiting.add(seq);
            }
        }

        let BucketedSeqs {
            running,
            waiting: new_waiting,
        } = self
            .bucketing_manager
            .bucket_and_waitlist_seqs_waiting(running, new_waiting, false);

        self.running = running;
        self.waiting = new_waiting;

        logger.set_num_running(self.running.len());
        logger.set_num_waiting(self.waiting.len());

        let mut completion = Vec::new();
        let mut prompt = Vec::new();
        for seq in &mut self.running {
            if seq.is_completion() {
                completion.push(seq);
            } else {
                prompt.push(seq);
            }
        }

        DefaultSchedulerOutput {
            completion: completion.into(),
            prompt: prompt.into(),
        }
    }

    fn sequence_fits(&self, running: &[Sequence], _seq: &Sequence) -> bool {
        match &self.method {
            DefaultSchedulerMethod::Fixed(n) => (running.len() + 1) <= (*n).into(),
        }
    }
}

impl Scheduler for DefaultScheduler<VecDeque<Sequence>> {
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_> {
        SchedulerOutput::DefaultScheduler {
            output: self.schedule(logger),
        }
    }
    fn waiting_len(&self) -> usize {
        self.waiting.len()
    }
    fn running_len(&self) -> usize {
        self.running.len()
    }
    fn add_seq(&mut self, seq: Sequence) {
        if seq.is_running() {
            // prefill case
            self.running.push(seq);
        } else {
            self.waiting.add(seq);
        }
    }
    fn block_size(&self) -> Option<usize> {
        None
    }
    fn free_finished_sequence_groups(&mut self) {
        // Remove finished sequences
        self.running.retain(|seq| !seq.is_finished_paged_attn());
    }
    fn get_finished_recurrent_indices(&self) -> Vec<usize> {
        self.running
            .iter()
            .filter(|seq| seq.is_finished_paged_attn())
            .filter_map(|seq| seq.recurrent_state_idx())
            .collect()
    }
    fn kv_cache_manager(&self) -> Option<Arc<tokio::sync::Mutex<KVCacheManager>>> {
        None
    }
    fn set_prefix_caching_enabled(&mut self, _enabled: bool) {
        // DefaultScheduler doesn't use PagedAttention prefix caching
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sampler::{Logprobs, Sampler},
        sequence::{SeqStepType, SequenceGroup, SequenceRecognizer},
    };
    use std::cell::RefCell;
    use std::time::Duration;

    thread_local! {
        /// Keeps every fixture sequence's `Receiver` alive for the duration of
        /// the test.
        ///
        /// 🔑 This is load-bearing, and its absence was a silent fixture bug.
        /// `seq_of_len` used to write `let (dummy_sender, _rx) = channel(1);`,
        /// and `_rx` is a real binding that drops when the helper returns — so
        /// **every sequence the suite built already had a closed responder**,
        /// i.e. modelled a client that had already disconnected. Nothing
        /// noticed, because nothing looked. The moment `schedule()` learned to
        /// reap abandoned sequences, `scheduler_runs_the_whole_admitted_batch`
        /// cancelled all 64 of its own fixtures.
        static LIVE_CLIENTS: RefCell<Vec<tokio::sync::mpsc::Receiver<crate::response::Response>>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Minimal running-completion sequence of exactly `n_toks` tokens, with a
    /// **live** client attached (see `LIVE_CLIENTS`).
    /// Mirrors `sequence::tests::dummy_seq` / `pipeline::amoe::new_dummy_seq`:
    /// no model, no engine — the scheduler only reads `len()`, `is_running()`,
    /// `images()`, `token_offset()` and the urgency counter.
    fn seq_of_len(id: usize, n_toks: usize) -> Sequence {
        let (dummy_sender, rx) = tokio::sync::mpsc::channel(1);
        LIVE_CLIENTS.with(|k| k.borrow_mut().push(rx));
        let dummy_sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            0.0,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();
        let group = Arc::new(std::sync::Mutex::new(SequenceGroup::new(
            1, false, false, None,
        )));
        let seq = Sequence::new_waiting(
            vec![1u32; n_toks],
            String::new(),
            id,
            0,
            1,
            dummy_sender,
            dummy_sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        );
        seq.set_state(SequenceState::RunningCompletion);
        seq
    }

    /// A sequence the scheduler has admitted but not yet prefilled.
    fn prompt_of_len(id: usize, n_toks: usize) -> Sequence {
        let seq = seq_of_len(id, n_toks);
        seq.set_state(SequenceState::RunningPrompt);
        seq
    }

    /// How many of `steps` iterations gave a prompt bucket the step, with the
    /// given floor. One decode bucket of 47 at length 500 against one prompt.
    fn prompt_steps_won(floor: Option<usize>, steps: usize) -> usize {
        let mut mgr = FixedBucketingManager {
            steps_since_prompt: 0,
            floor,
        };
        let mut won = 0usize;
        for _ in 0..steps {
            let mut running: Vec<Sequence> = (0..47).map(|i| seq_of_len(i, 500)).collect();
            running.push(prompt_of_len(47, 300));
            let out = <FixedBucketingManager as BucketingManager<VecDeque<Sequence>>>::
                bucket_and_waitlist_seqs_waiting(&mut mgr, running, VecDeque::new(), false);
            if out.running.iter().any(|s| s.is_prompt()) {
                won += 1;
            }
        }
        won
    }

    /// 🔑 The ceiling `feat/dense-ragged-decode` hit, reproduced in the
    /// scheduler alone and on CPU: with a large decode cohort running, a prompt
    /// bucket never wins a step, so admitted-but-unprefilled sequences stay
    /// unprefilled. That is why the cohort stops at `47 running, 48 waiting`.
    ///
    /// `select_running_bucket` takes the highest SUMMED priority, so 47
    /// sequences at length 500 outscore one fresh prompt by ~47x before urgency
    /// is counted at all.
    #[test]
    fn a_prompt_starves_behind_a_large_decode_cohort() {
        let won = prompt_steps_won(None, 24);
        assert_eq!(
            won, 0,
            "with no floor a prompt must never win a step behind a 47-sequence cohort; \
             it won {won}/24, so this fixture is not reproducing the ceiling"
        );
    }

    /// And the floor lifts exactly that, at the rate asked for.
    ///
    /// Teeth: the assertion is an equality on the count, not `> 0` — a floor
    /// that fired every step would also pass `> 0` while destroying decode.
    #[test]
    fn the_floor_gives_prompts_a_turn_without_giving_them_every_turn() {
        let steps = 24;
        let won = prompt_steps_won(Some(4), steps);
        // Fires on the 5th iteration and every 5th after: floor+1 cadence.
        let expected = steps / 5;
        assert_eq!(
            won, expected,
            "floor=4 must yield one prompt step in every 5, got {won}/{steps}"
        );
        assert!(
            won < steps,
            "a floor that took every step would starve decode instead"
        );
    }

    /// The kill-switch (`ARC_PREFILL_FLOOR_STEPS=0` → `floor: None`) is the
    /// previous selection, key for key — asserted against the same fixture
    /// rather than argued. Note "unset" no longer means this: unset is the
    /// default floor of 4 (see `floor_from`); `0` is how today's behaviour is
    /// restored.
    #[test]
    fn the_floor_disabled_reproduces_the_previous_selection() {
        assert_eq!(prompt_steps_won(None, 12), 0);
    }

    /// The env mapping behind the default, both sides pinned:
    /// unset → the default floor (ON), `0` → the kill-switch (OFF, historical
    /// selection), a number → itself, garbage → the default rather than a
    /// silent disable.
    ///
    /// Mutation check (run 2026-08-21): revert `floor_from`'s `None` arm to the
    /// pre-change `.filter(|n| *n > 0)` shape (unset → `None`) and this fails
    /// on the first assertion with `left: None, right: Some(4)`.
    #[test]
    fn the_floor_defaults_on_and_zero_is_the_kill_switch() {
        assert_eq!(floor_from(None), Some(DEFAULT_PREFILL_FLOOR_STEPS));
        assert_eq!(floor_from(Some("0")), None, "0 must cleanly disable");
        assert_eq!(floor_from(Some("7")), Some(7));
        assert_eq!(
            floor_from(Some("not-a-number")),
            Some(DEFAULT_PREFILL_FLOOR_STEPS),
            "garbage must fall back to the default, not silently disable the floor"
        );
        assert_eq!(DEFAULT_PREFILL_FLOOR_STEPS, 4);
    }

    /// Advance a scheduled sequence by one decoded token, as the engine would.
    fn decode_one(seq: &mut Sequence) {
        seq.add_token(
            Logprobs {
                token: 7,
                logprob: 0.0,
                bytes: Some("a".to_string()),
                top_logprobs: None,
            },
            b"a".to_vec(),
            &None,
        );
    }

    fn bucket_of(len: usize, n: usize, id_base: usize) -> (BucketKey, Vec<Sequence>) {
        let seqs = (0..n).map(|i| seq_of_len(id_base + i, len)).collect();
        ((len, false, 0), seqs)
    }

    /// 🔑 The dense equivalent of the paged regression: eight sequences at
    /// eight DIFFERENT cache lengths must all decode in one step once the
    /// pipeline's cache is known to carry per-sequence lengths.
    ///
    /// Before the `ragged_decode_supported` gate this asserts FALSE by
    /// construction — `select_running_bucket` runs exactly one bucket, and the
    /// coalescence override cannot rescue a spread cohort: for B=8 at 8
    /// distinct lengths with gap ~64, `(8-1)*64 = 448 > 1*256`, so it is
    /// refused and the split is permanent. That is the measured
    /// `16 running, 80 waiting` shape at B=128.
    ///
    /// The flag is restored at the end so this cannot leak into other tests in
    /// the same binary — it is process-global by design (the scheduler never
    /// sees the model).
    #[test]
    fn decode_runs_the_whole_ragged_cohort_when_the_cache_can_carry_it() {
        use crate::kv_cache::ragged_decode_test_override as flag;
        const N: usize = 8;
        let lens = [50usize, 114, 179, 243, 307, 371, 436, 500];
        let logger = IntervalLogger::new(Duration::from_secs(3600), None);

        let width_with = |on: bool| {
            flag::with(on, || {
                let mut sched: DefaultScheduler<VecDeque<Sequence>> = DefaultScheduler::new(
                    DefaultSchedulerMethod::Fixed(NonZeroUsize::new(128).unwrap()),
                );
                for (i, l) in lens.iter().enumerate() {
                    sched.add_seq(seq_of_len(i, *l));
                }
                DefaultScheduler::schedule(&mut sched, &logger)
                    .completion
                    .into_vec()
                    .len()
            })
        };

        let off_width = width_with(false);
        let on_width = width_with(true);

        // Teeth: if the OFF arm already ran the whole cohort, the fixture is not
        // ragged and the ON arm proves nothing.
        assert!(
            off_width < N,
            "fixture is not ragged: the OFF arm already scheduled {off_width}/{N}, so this \
             test cannot detect the change"
        );
        assert_eq!(
            on_width, N,
            "a ragged decode cohort must run whole; got {on_width}/{N}"
        );
    }

    /// 🔑 The cohort-can-only-shrink defect, and its fix, in one fixture: a
    /// formed decode cohort at length 500 plus ONE newcomer at length 100.
    ///
    /// With exact-length bucketing the newcomer can never join: coalescing is
    /// refused (`(9-1)*400 = 3200 > 1*256`), greedy priority keeps the cohort
    /// (`8*log2(500) >> 1*log2(100)`), and running the cohort *grows* the gap
    /// by one every step — so the newcomer idles forever and cohorts only
    /// shrink. Every published B=256 number under this regime came from a
    /// synthetic uniform burst, because a real arrival process can never refill
    /// a cohort.
    ///
    /// With ragged decode on, every decode sequence keys to one bucket and the
    /// newcomer is admitted into the running set the same step.
    ///
    /// Mutation check (run 2026-08-21): pin `len` to `seq.cache_bucket_len()`
    /// (reverting the `ragged_decode` arm at the top of
    /// `bucket_and_waitlist_seqs_waiting`) and the ON arm fails with
    /// `on: 8/9` — the newcomer is waitlisted again. The OFF arm is asserted
    /// in the same run, so a fixture that stopped being ragged also fails.
    #[test]
    fn a_formed_cohort_admits_a_newcomer_when_ragged_decode_is_on() {
        use crate::kv_cache::ragged_decode_test_override as flag;
        let logger = IntervalLogger::new(Duration::from_secs(3600), None);

        let running_ids_with = |on: bool| -> Vec<usize> {
            flag::with(on, || {
                let mut sched: DefaultScheduler<VecDeque<Sequence>> = DefaultScheduler::new(
                    DefaultSchedulerMethod::Fixed(NonZeroUsize::new(128).unwrap()),
                );
                // The formed cohort: eight decoding sequences at length 500.
                for i in 0..8 {
                    sched.add_seq(seq_of_len(i, 500));
                }
                // The newcomer: decoding at length 100 (freshly prefilled and
                // far behind the cohort — the shape coalescing refuses).
                sched.add_seq(seq_of_len(8, 100));
                // A waiting request, so selection runs the non-discrete
                // arrival path (the discrete no-waiting path is min-length
                // catch-up, a different regime with its own test).
                let waiter = seq_of_len(9, 300);
                waiter.set_state(SequenceState::Waiting);
                sched.add_seq(waiter);
                let out = DefaultScheduler::schedule(&mut sched, &logger);
                out.completion.into_vec().iter().map(|s| *s.id()).collect()
            })
        };

        let off = running_ids_with(false);
        assert!(
            !off.contains(&8) && off.len() == 8,
            "fixture guard: with exact-length bucketing the newcomer must be \
             waitlisted behind the cohort, got {off:?} — otherwise the ON arm \
             proves nothing"
        );

        let on = running_ids_with(true);
        assert_eq!(
            on.len(),
            9,
            "ragged decode must run the cohort AND the newcomer together, got {on:?}"
        );
        assert!(
            on.contains(&8),
            "the newcomer must be in the running set, got {on:?}"
        );
    }

    /// A PROMPT batch keeps the exact-length key even with the flag on. Prefill
    /// is right-padded and its `cu_seqlens` are built from those padded widths
    /// — a different mechanism from the decode one, and not covered by the
    /// front-alignment this flag enables.
    #[test]
    fn prompts_keep_their_exact_length_buckets() {
        use crate::kv_cache::ragged_decode_test_override as flag;
        let logger = IntervalLogger::new(Duration::from_secs(3600), None);

        let width = flag::with(true, || {
            let mut sched: DefaultScheduler<VecDeque<Sequence>> = DefaultScheduler::new(
                DefaultSchedulerMethod::Fixed(NonZeroUsize::new(128).unwrap()),
            );
            for (i, l) in [37usize, 91, 155].iter().enumerate() {
                // Waiting, not running: `schedule` promotes the waiting queue to
                // `RunningPrompt` itself, and `add_seq` routes on `is_running()`.
                let seq = seq_of_len(i, *l);
                seq.set_state(SequenceState::Waiting);
                sched.add_seq(seq);
            }
            DefaultScheduler::schedule(&mut sched, &logger)
                .prompt
                .into_vec()
                .len()
        });

        assert_eq!(
            width, 1,
            "three prompts of different lengths must still be bucketed apart; got {width}"
        );
    }

    /// THE regression test.
    ///
    /// 64 admitted sequences, ample KV budget (`Fixed(128)`), split across two
    /// cache lengths two tokens apart — the shape the H200 measured as
    /// `32 running, 32 waiting` at B=64
    /// (`memory/mission/wave26-AX-h200-measurement.md`). All 64 must end up in
    /// a single running set, and stay there.
    ///
    /// Before the `select_running_bucket` coalescence override this asserts
    /// FALSE forever: `compute_priority` = `urgency + log2(len)` makes the two
    /// equal-sized buckets alternate perfectly, each advancing one token every
    /// two steps, so the gap oscillates 1, 2, 1, 2, ... and never reaches 0.
    #[test]
    fn scheduler_runs_the_whole_admitted_batch() {
        const N: usize = 64;
        const HALF: usize = N / 2;

        let mut sched: DefaultScheduler<VecDeque<Sequence>> = DefaultScheduler::new(
            DefaultSchedulerMethod::Fixed(NonZeroUsize::new(128).unwrap()),
        );
        for i in 0..HALF {
            sched.add_seq(seq_of_len(i, 100));
        }
        for i in 0..HALF {
            sched.add_seq(seq_of_len(HALF + i, 102));
        }
        assert_eq!(Scheduler::running_len(&sched), N, "all N admitted");

        let logger = IntervalLogger::new(Duration::from_secs(3600), None);

        // Generous budget: coalescing two buckets two tokens apart needs 2 steps.
        let mut converged_at = None;
        for step in 0..16 {
            {
                let out = DefaultScheduler::schedule(&mut sched, &logger);
                let mut scheduled: Vec<&mut Sequence> = out
                    .completion
                    .into_vec()
                    .into_iter()
                    .chain(out.prompt.into_vec())
                    .collect();
                for seq in scheduled.iter_mut() {
                    decode_one(seq);
                }
            }
            if converged_at.is_none() && Scheduler::running_len(&sched) == N {
                converged_at = Some(step);
            }
            if let Some(first) = converged_at {
                assert_eq!(
                    Scheduler::running_len(&sched),
                    N,
                    "step {step}: batch split again after coalescing at step {first}"
                );
                assert_eq!(Scheduler::waiting_len(&sched), 0, "step {step}: waitlisted");
            }
        }

        assert!(
            converged_at.is_some(),
            "scheduler never ran all {N} admitted sequences in one step: \
             running={}, waiting={} — half the fleet batch is idling",
            Scheduler::running_len(&sched),
            Scheduler::waiting_len(&sched),
        );
    }

    /// The coalescence override is taken when it repays quickly: two equal
    /// buckets one token apart, greedy priority favours the longer one, and we
    /// must override to the shorter so the two merge on the next step.
    #[test]
    fn select_running_bucket_coalesces_adjacent_equal_buckets() {
        let (short_key, short) = bucket_of(101, 32, 0);
        let (long_key, long) = bucket_of(102, 32, 32);
        let seq_buckets = HashMap::from([(short_key, short), (long_key, long)]);
        // Longer bucket has been running, shorter has been waitlisted a step.
        let seq_priorities = HashMap::from([
            (short_key, 32.0 * (101f64).log2()),
            (long_key, 32.0 * (1.0 + (102f64).log2())),
        ]);

        assert!(
            seq_priorities[&long_key] > seq_priorities[&short_key],
            "precondition: greedy priority picks the longer bucket"
        );
        assert_eq!(
            select_running_bucket(&seq_buckets, &seq_priorities, false),
            short_key,
            "must override to the shorter bucket so the two coalesce"
        );
    }

    /// ...and refused when it does not: a single fresh 21-token arrival against
    /// a 63-sequence cohort at length 500 would idle the cohort for 479 steps.
    /// Anti-starvation behaviour of the greedy rule must be preserved.
    #[test]
    fn select_running_bucket_refuses_expensive_coalescing() {
        let (newcomer_key, newcomer) = bucket_of(21, 1, 0);
        let (cohort_key, cohort) = bucket_of(500, 63, 1);
        let seq_buckets = HashMap::from([(newcomer_key, newcomer), (cohort_key, cohort)]);
        let seq_priorities = HashMap::from([
            (newcomer_key, 1.0 * (21f64).log2()),
            (cohort_key, 63.0 * (500f64).log2()),
        ]);

        assert_eq!(
            select_running_bucket(&seq_buckets, &seq_priorities, false),
            cohort_key,
            "must keep the cohort running rather than idle 63 sequences for 479 steps"
        );
    }

    /// `discrete` (no waiting sequences) keeps its documented min-length
    /// catch-up behaviour untouched.
    #[test]
    fn select_running_bucket_discrete_is_min_length() {
        let (short_key, short) = bucket_of(101, 1, 0);
        let (long_key, long) = bucket_of(500, 63, 1);
        let seq_buckets = HashMap::from([(short_key, short), (long_key, long)]);
        assert_eq!(
            select_running_bucket(&seq_buckets, &HashMap::new(), true),
            short_key
        );
    }

    /// A sequence whose client is gone but which nothing has marked done —
    /// exactly the state observed on the box: `N running` against a single live
    /// connection, zero disconnect warnings, persisting into the next run.
    ///
    /// The sequence is deliberately left `RunningCompletion`: an abandoned
    /// sequence IS still running, which is why the pre-existing
    /// `filter(|seq| seq.is_running())` never removed it.
    fn abandoned_seq_of_len(id: usize, n_toks: usize) -> Sequence {
        let seq = seq_of_len(id, n_toks);
        // Drop this sequence's receiver, and only this one's.
        LIVE_CLIENTS.with(|k| {
            k.borrow_mut().pop();
        });
        assert!(
            seq.client_is_gone(),
            "fixture guard: the abandoned sequence must actually have a closed \
             responder, or this test proves nothing"
        );
        assert!(
            seq.is_running(),
            "fixture guard: it must still be RUNNING — that is the whole defect"
        );
        seq
    }

    /// Mutation check (run 2026-08-18): delete the reap block in `schedule()`
    /// and this reports `3 running` — the phantom rows survive, exactly as on
    /// master.
    #[test]
    fn a_sequence_whose_client_left_is_reaped_and_a_live_one_is_not() {
        let mut sched: DefaultScheduler<VecDeque<Sequence>> = DefaultScheduler::new(
            DefaultSchedulerMethod::Fixed(NonZeroUsize::new(128).unwrap()),
        );
        sched.add_seq(seq_of_len(0, 100));
        sched.add_seq(abandoned_seq_of_len(1, 100));
        sched.add_seq(seq_of_len(2, 100));
        assert_eq!(
            Scheduler::running_len(&sched),
            3,
            "all three are admitted before any scheduling pass"
        );

        let logger = IntervalLogger::new(Duration::from_secs(3600), None);
        {
            let _ = DefaultScheduler::schedule(&mut sched, &logger);
        }

        let survivors: Vec<usize> = sched.running.iter().map(|s| *s.id()).collect();
        assert_eq!(
            Scheduler::running_len(&sched),
            2,
            "the abandoned sequence must be gone; running = {survivors:?}"
        );
        assert_eq!(
            survivors,
            vec![0, 2],
            "and it must be the ABANDONED one that went, not an arbitrary row"
        );
    }

    /// The interaction with length bucketing, which is why this is not merely
    /// untidy: `bucket_and_waitlist_seqs_waiting` partitions on
    /// `cache_bucket_len`, so a phantom sequence at a stale length keeps a
    /// bucket alive that nothing real needs, and the running set is one bucket.
    ///
    /// Here two live sequences sit at length 100 and one abandoned sequence at
    /// 500. Unreaped, the scheduler sees two buckets and must waitlist one of
    /// them; reaped, there is a single bucket and both live rows run.
    ///
    /// Mutation check (run 2026-08-18): delete the reap and this fails with
    /// `1 live sequence(s) running` — the phantom wins the bucket.
    #[test]
    fn a_phantom_sequence_no_longer_pins_a_bucket() {
        let mut sched: DefaultScheduler<VecDeque<Sequence>> = DefaultScheduler::new(
            DefaultSchedulerMethod::Fixed(NonZeroUsize::new(128).unwrap()),
        );
        sched.add_seq(seq_of_len(0, 100));
        sched.add_seq(seq_of_len(1, 100));
        sched.add_seq(abandoned_seq_of_len(2, 500));

        let logger = IntervalLogger::new(Duration::from_secs(3600), None);
        {
            let _ = DefaultScheduler::schedule(&mut sched, &logger);
        }

        let running: Vec<usize> = sched.running.iter().map(|s| *s.id()).collect();
        assert_eq!(
            running.len(),
            2,
            "both live sequences must run together: {running:?} — a phantom at a \
             stale length must not split the batch into two buckets"
        );
        assert!(
            !running.contains(&2),
            "the phantom must not be running: {running:?}"
        );
        assert_eq!(
            Scheduler::waiting_len(&sched),
            0,
            "and nothing real should have been waitlisted behind it"
        );
    }
}
