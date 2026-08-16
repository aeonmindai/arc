#!/usr/bin/env python3
"""DOCTRINE D12 mutation proof for the qtip2b (K=2/V=1) beam-search tests.

    python3 arc-tools/quality/mutate_qtip2b_beam.py     # from the repo root

Each entry perturbs ONE mechanism of `mistralrs-quant/src/qtip/bitshift.rs`'s
beam search and names the test that MUST fail because of it. A mutation that
leaves its test green means that mechanism is untested — which is the failure
this repo has hit repeatedly (7+ tests found passing with unreachable
assertions). Every mutation is reverted before the next one runs, and the
original file is restored on any exit path.

The tie-break entry is why `beam_2b_unpruned_matches_exhaustive_bit_for_bit`
carries a constant-codebook fixture: with a realistic codebook, flipping the
dedup comparison from `<` to `<=` SURVIVES on both gaussian and fp4_dequant
rows, because neither produces exact equal-cost predecessor pairs often enough.

CPU-only and free — no GPU required. It exercises the Rust beam, which the
CUDA kernel is pinned to bit-for-bit by `cuda_beam_2b_matches_cpu_beam_bit_for_bit`.
"""
import subprocess
import sys

P = 'mistralrs-quant/src/qtip/bitshift.rs'

MUTATIONS = [
    (
        'frontier update / tie-break: dedup keeps the LAST equal-cost '
        'predecessor (<=) instead of the first (<)',
        '''                } else {
                    let c = &mut cands[existing as usize];
                    if entry.cost < c.cost {''',
        '''                } else {
                    let c = &mut cands[existing as usize];
                    if entry.cost <= c.cost {''',
        'beam_2b_unpruned_matches_exhaustive_bit_for_bit',
    ),
    (
        'frontier update / merge: dedup keeps the WORST predecessor cost',
        '''                } else {
                    let c = &mut cands[existing as usize];
                    if entry.cost < c.cost {''',
        '''                } else {
                    let c = &mut cands[existing as usize];
                    if entry.cost > c.cost {''',
        'beam_2b_unpruned_matches_exhaustive_bit_for_bit',
    ),
    (
        'frontier update / backpointer: the merge updates the cost but forgets '
        'to move the parent with it',
        '''                    if entry.cost < c.cost {
                        c.cost = entry.cost;
                        c.parent = pi as u16;
                    }''',
        '''                    if entry.cost < c.cost {
                        c.cost = entry.cost;
                    }''',
        'beam_2b_unpruned_matches_exhaustive_bit_for_bit',
    ),
    (
        "group geometry: the K=4 rung's 12-bit group mask copied instead of "
        "the K=2 rung's 14-bit one",
        '''            let base = ((entry.state as u32) << K2B) & STATE_MASK_2B;''',
        '''            let base = ((entry.state as u32) & 0xFFF) << K2B;''',
        'beam_2b_unpruned_matches_exhaustive_bit_for_bit',
    ),
    (
        'group geometry: successors enumerated over the K=4 alphabet (16) '
        'instead of the K=2 alphabet (4)',
        '''            for sym in 0..ALPHABET_2B as u32 {
                let succ = (base | sym) as u16;''',
        '''            for sym in 0..(ALPHABET_2B as u32 * 4) {
                let succ = (base | sym) as u16;''',
        'beam_2b_unpruned_matches_exhaustive_bit_for_bit',
    ),
    (
        'selection: pruning keeps the WIDEST-cost survivors (Reverse) instead '
        'of the cheapest',
        '''        beam.select_nth_unstable_by(width - 1, |a, b| {
            a.cost.total_cmp(&b.cost).then(a.state.cmp(&b.state))
        });''',
        '''        beam.select_nth_unstable_by(width - 1, |a, b| {
            b.cost.total_cmp(&a.cost).then(a.state.cmp(&b.state))
        });''',
        'beam_2b_quality_delta_vs_exhaustive_is_bounded_and_reported',
    ),
    (
        'width plumbing: the requested width is silently narrowed to the CUDA '
        "kernel's 256-slot maximum",
        '''        TrellisSearch::Beam { width } => beam_quantize_row_2b(target_row, codebook, width),''',
        '''        TrellisSearch::Beam { width } => beam_quantize_row_2b(target_row, codebook, width.min(256)),''',
        'quantize_row_2b_never_substitutes_a_beam_width',
    ),
    (
        'width plumbing: the search argument is dropped and every bake runs '
        'exhaustive',
        '''                    QtipMode::Viterbi => quantize_row_2b(&scaled_target, &codebook, search),''',
        '''                    QtipMode::Viterbi => quantize_row_2b(&scaled_target, &codebook, TrellisSearch::Exhaustive),''',
        'beam_2b_stamps_its_width_into_the_artifact',
    ),
    (
        'width plumbing (3-D): the expert chunks are quantized exhaustively '
        'while the assembled stack still claims the beam',
        '''            let layer = Self::quantize_with_options_concrete_search(
                &rows_2d,
                None,
                &quant_device,
                mode,
                use_rotation,
                search,
            )?;''',
        '''            let layer = Self::quantize_with_options_concrete_search(
                &rows_2d,
                None,
                &quant_device,
                mode,
                use_rotation,
                TrellisSearch::Exhaustive,
            )?;''',
        'beam_2b_stamps_its_width_on_a_3d_expert_stack',
    ),
    (
        'stamp: a beam bake records itself as exhaustive',
        '''            search_detail: QtipSearchDetail::for_bake(mode, search, false),
        })
    }

    /// GPU fast path for `quantize_with_options_concrete`.''',
        '''            search_detail: QtipSearchDetail::EXHAUSTIVE_MSE,
        })
    }

    /// GPU fast path for `quantize_with_options_concrete`.''',
        'beam_2b_stamps_its_width_into_the_artifact',
    ),
]


def run(test):
    r = subprocess.run(
        ['cargo', 'test', '-p', 'mistralrs-quant', '--lib', '--', test],
        capture_output=True, text=True)
    return r.returncode == 0


def main():
    src0 = open(P).read()
    ok = True
    try:
        for name, old, new, test in MUTATIONS:
            s = open(P).read()
            if old not in s:
                print(f'SKIP  (anchor not found) :: {name}')
                ok = False
                continue
            open(P, 'w').write(s.replace(old, new, 1))
            survived = run(test)
            open(P, 'w').write(src0)
            if survived:
                print(f'SURVIVED  {test} still passes :: {name}')
                ok = False
            else:
                print(f'killed by {test:52s} :: {name}')
    finally:
        open(P, 'w').write(src0)
    print('ALL MUTATIONS KILLED' if ok else 'SOME MUTATIONS SURVIVED')
    return 0 if ok else 1


sys.exit(main())
