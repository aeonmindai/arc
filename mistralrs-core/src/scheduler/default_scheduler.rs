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

struct FixedBucketingManager;

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
        for seq in running {
            // The cache length, not the token count — see
            // `Sequence::cache_bucket_len`. Identical partition for every
            // non-speculative path; the one that keeps a batched MTP cohort
            // whole when its accept lengths differ.
            let len = seq.cache_bucket_len();
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
            let len = select_running_bucket(&seq_buckets, &seq_priorities, discrete);
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
            DefaultSchedulerMethod::Fixed(_) => Box::new(FixedBucketingManager),
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
                for seq in waiting.into_iter() {
                    seq.set_state(SequenceState::RunningPrompt);
                    self.running.push(seq);
                }
                self.waiting = Backer::new();
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
        for seq in waiting.into_iter() {
            if self.sequence_fits(&running, &seq) {
                if seq.is_waiting() {
                    seq.set_state(SequenceState::RunningPrompt);
                }
                running.push(seq);
            } else {
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
    use std::time::Duration;

    /// Minimal running-completion sequence of exactly `n_toks` tokens.
    /// Mirrors `sequence::tests::dummy_seq` / `pipeline::amoe::new_dummy_seq`:
    /// no model, no engine — the scheduler only reads `len()`, `is_running()`,
    /// `images()`, `token_offset()` and the urgency counter.
    fn seq_of_len(id: usize, n_toks: usize) -> Sequence {
        let (dummy_sender, _rx) = tokio::sync::mpsc::channel(1);
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
}
