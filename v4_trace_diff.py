#!/usr/bin/env python3
"""RUN-161 decode-bug localizer.

Diffs the per-op tensors dumped by `V4_TRACE` between a PREFILL (known-correct
reference) run and a DECODE run. Both runs target the SAME logical position N,
so every op should match within quant noise. The FIRST op whose cosine drops
below ~1.0 is exactly where the decode path corrupts the hidden state — i.e.
the bug is in the op that PRODUCED that tensor. Everything upstream is innocent
by construction; everything downstream is just inheriting the corruption.

How to produce the two dirs (greedy/temp=0, same model, same prompt P of len N):

  # reference: prefill the FULL sequence 0..N in one forward, no decode step.
  # Use a unique leading nonce so no prefix-cache hit. max_tokens=1.
  V4_TRACE=/tmp/tr_prefill V4_TRACE_LAYER=0 <serve/probe ...>

  # decode: prefill 0..N-1, then take exactly ONE decode forward (position N).
  # max_tokens=2 (token1 from prefill, token2 from the first decode forward).
  V4_TRACE=/tmp/tr_decode  V4_TRACE_LAYER=0 <serve/probe ...>

Then:  python v4_trace_diff.py /tmp/tr_prefill /tmp/tr_decode

Alignment: prefill tensors have T=N+1 (take the LAST row = position N); decode
tensors have T=1 (take row 0). Cache tensors (40/41) are full-length N+1 in both
and are compared row-by-row to split a cache-storage bug (old rows differ) from
a new-token position bug (only the last row differs).
"""
import sys
import os
import glob
import numpy as np

GOOD = 0.999  # cosine at/above this = "matches" (quant noise floor)


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def maxabs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def seq_axis(x):
    # 4D [B,H,T,D] -> 2 ; 3D [B,T,D] -> 1
    return 2 if x.ndim == 4 else (1 if x.ndim == 3 else None)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    pdir, ddir = sys.argv[1], sys.argv[2]

    for d, name in [(pdir, "prefill"), (ddir, "decode")]:
        for m in sorted(glob.glob(os.path.join(d, "*_meta.txt"))):
            txt = open(m).read().strip().replace("\n", "  ")
            print(f"[{name:7}] {os.path.basename(m)}: {txt}")
    print()

    ptags = {os.path.basename(p) for p in glob.glob(os.path.join(pdir, "*.npy"))}
    dtags = {os.path.basename(p) for p in glob.glob(os.path.join(ddir, "*.npy"))}
    tags = sorted(ptags & dtags)
    only = (ptags ^ dtags)
    if only:
        print(f"(only in one dir, skipped: {sorted(only)})\n")

    print(f"{'op':<24}{'cosine':>11}{'maxabsdiff':>14}   shapes (prefill / decode)")
    print("-" * 90)
    first_bad = None
    for t in tags:
        try:
            P = np.load(os.path.join(pdir, t))
            D = np.load(os.path.join(ddir, t))
        except Exception as e:
            print(f"{t:<24}{'UNREADABLE':>11}{'':>14}   {e}")
            continue
        is_cache = ("k_cached" in t) or ("v_cached" in t)
        note = ""
        if P.shape == D.shape:
            c = cos(P, D)
            md = maxabs(P, D)
            if is_cache:
                ax = seq_axis(P)
                if ax is not None:
                    Pm = np.moveaxis(P, ax, 0)
                    Dm = np.moveaxis(D, ax, 0)
                    rc = [cos(Pm[i], Dm[i]) for i in range(Pm.shape[0])]
                    bad = [i for i, v in enumerate(rc) if not (v > GOOD)]
                    tail = "..." if len(bad) > 10 else ""
                    note = f"  rows_diff={bad[:10]}{tail} of {Pm.shape[0]}"
        else:
            axp, axd = seq_axis(P), seq_axis(D)
            if axp is None or axd is None:
                print(f"{t:<24}{'SKIP':>11}{'':>14}   {P.shape} / {D.shape}  (rank)")
                continue
            pi = np.take(P, P.shape[axp] - 1, axis=axp)  # prefill position N
            di = np.take(D, 0, axis=axd)                 # decode position 0
            if pi.shape != di.shape:
                print(f"{t:<24}{'SHAPE?':>11}{'':>14}   {P.shape} / {D.shape}  -> {pi.shape}/{di.shape}")
                continue
            c = cos(pi, di)
            md = maxabs(pi, di)
            note = "  (rowN vs row0)"
        flag = "" if (c > GOOD) else "   <<< DIVERGES"
        print(f"{t:<24}{c:>11.6f}{md:>14.4e}   {P.shape} / {D.shape}{note}{flag}")
        if first_bad is None and not (c > GOOD):
            first_bad = t
    print("-" * 90)
    if first_bad:
        print(f">>> FIRST DIVERGENCE: {first_bad}")
        print(">>> The bug is in the op that PRODUCED this tensor. Map the tag:")
        print("    00_xs_in/00_meta = layer input + seqlen_offsets (compare the offset values!)")
        print("    10_q_normed      = Q proj + per-head RMS-norm")
        print("    20/21_q/k_rope   = apply_rope_inplace  (RoPE @ seqlen_offsets)")
        print("    30_k_actquant    = act_quant_kv_nope")
        print("    40/41_k/v_cached = KvCache.append read-back (rows_diff tells cache-vs-newtoken)")
        print("    55_sdpa_bhtd     = attention kernel output (cuBLASLt / naive_sdpa)")
        print("    60_after_invtail = forward_inverse_tail  (inverse RoPE @ seqlen_offsets)")
        print("    70_o_out         = o_proj output")
    else:
        print(">>> No divergence at this layer (all cos>0.999). Re-run with a higher")
        print("    V4_TRACE_LAYER, or the bug is in MoE/router/residual, not attention.")


if __name__ == "__main__":
    main()
