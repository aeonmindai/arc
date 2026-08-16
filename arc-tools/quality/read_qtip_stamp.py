#!/usr/bin/env python3
"""Read the QtipSearchStamp + QtipSearchDetail back out of a qtip2b UQFF.

Why this exists: the `qtip2b` rung emits NO `log_bake_header` line, so the bake
log cannot say which trellis search produced the artifact. The artifact can:
`Qtip2bLayer::serialize` (mistralrs-quant/src/qtip/bitshift.rs) appends, after
the last tensor,

    [stamp:u8][flags:u8]  (+ [beam_width:u16 LE] when flags & 0x01)

    stamp  1 = Trellis (viterbi), 2 = Greedy  (QtipSearchStamp::to_wire,
           qtip/mod.rs:1134) -- 0 is reserved and invalid.
    flags  bit0 FLAG_BEAM, bit1 FLAG_HESSIAN, rest reserved-must-be-zero
           (QtipSearchDetail::to_wire, qtip/mod.rs:1263).

and the head of every payload is

    [version:u32 LE = 0x00030000 -> 768][quant_type:u8 = 10 (Qtip2b)][has_bias:u8]
    [in_features:u32 LE][mcg_mult:u32 LE]

so the same read proves the RUNG as well as the SEARCH.

NOTE the wave41-BS reader read "the last two bytes". That is correct only for an
exhaustive bake. A BEAM bake writes two more bytes (the width), so the last two
bytes of a beam payload are the width's, not the stamp's. This reader decodes
the tail properly and refuses to guess.

Exit 0 = every layer is qtip2b + trellis + the expected beam width.
Exit 1 = any layer disagrees (greedy, exhaustive, wrong width, wrong rung).
"""
import argparse
import collections
import glob
import json
import os
import struct
import sys

UQFF_VERSION = (0 << 16) | (3 << 8) | 0  # 768
QUANT_TYPE_QTIP2B = 10
QUANT_TYPE_UNQUANT = 1  # QuantizedSerdeType::Unquant (lib.rs:1250)
QUANT_TYPE_QTIP_LUT = 8  # QuantizedSerdeType::Qtip -- the OTHER rung
STAMP = {1: "trellis", 2: "GREEDY"}
FLAG_BEAM = 0x01
FLAG_HESSIAN = 0x02
FLAG_RESERVED = ~(FLAG_BEAM | FLAG_HESSIAN) & 0xFF


def payloads(path):
    """Yield (name, bytes) for every tensor in a safetensors file, lazily."""
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len))
        base = 8 + hdr_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            # We only need the head and the tail; never read a whole shard.
            fh.seek(base + start)
            head = fh.read(min(14, end - start))
            fh.seek(base + max(start, end - 8))
            tail = fh.read(min(8, end - start))
            yield name, head, tail, end - start


def decode(head, tail, size):
    if size < 16:
        return {"err": f"payload too short ({size} B)"}
    (version,) = struct.unpack("<I", head[0:4])
    qtype = head[4]
    out = {"version": version, "qtype": qtype}
    if version != UQFF_VERSION:
        out["err"] = f"version {version} != {UQFF_VERSION}"
        return out
    if qtype == QUANT_TYPE_UNQUANT:
        # A layer ISQ declined to quantize, serialized as UnquantLinear. It
        # carries no trellis and therefore no search stamp -- there is nothing
        # here to verify and its absence is not a defect. Counted, not failed.
        out["skip"] = "unquant"
        return out
    if qtype != QUANT_TYPE_QTIP2B:
        # Anything else in a qtip2b artifact is wrong, and the LUT rung (8) is
        # the one that would be quietly plausible, so it is named explicitly.
        which = " (the qtip2 LUT rung)" if qtype == QUANT_TYPE_QTIP_LUT else ""
        out["err"] = f"quant type {qtype}{which} != Qtip2b({QUANT_TYPE_QTIP2B})"
        return out
    # Tail: try the 4-byte (beam) form first, then the 2-byte (exhaustive) form.
    # The two are told apart by FLAG_BEAM, which is the ONLY thing that decides
    # whether a width follows -- so decode from the flags, do not pattern-match.
    for taillen in (4, 2):
        if len(tail) < taillen:
            continue
        cand = tail[-taillen:]
        stamp, flags = cand[0], cand[1]
        if stamp not in STAMP:
            continue
        if flags & FLAG_RESERVED:
            continue
        if bool(flags & FLAG_BEAM) != (taillen == 4):
            continue
        width = struct.unpack("<H", cand[2:4])[0] if taillen == 4 else None
        out.update(
            stamp=STAMP[stamp],
            beam=bool(flags & FLAG_BEAM),
            hessian=bool(flags & FLAG_HESSIAN),
            width=width,
        )
        return out
    out["err"] = f"tail {tail[-4:].hex()} decodes to no valid (stamp, flags[, width])"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--expect-width", type=int, default=256)
    ap.add_argument("--glob", default="*.uqff")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.folder, a.glob)))
    if not files:
        print(f"STAMP_FAIL no {a.glob} in {a.folder}")
        return 1

    tally = collections.Counter()
    bad = []
    total = 0
    skipped = 0
    qtip = 0
    for f in files:
        for name, head, tail, size in payloads(f):
            total += 1
            d = decode(head, tail, size)
            if "skip" in d:
                skipped += 1
                tally[f"(skipped) {d['skip']}"] += 1
                continue
            if "err" in d:
                tally[f"ERR:{d['err']}"] += 1
                if len(bad) < 8:
                    bad.append((os.path.basename(f), name, d["err"]))
                continue
            qtip += 1
            key = f"{d['stamp']}/{'beam(W=%d)' % d['width'] if d['beam'] else 'exhaustive'}/{'hessian' if d['hessian'] else 'mse'}"
            tally[key] += 1
            if d["stamp"] != "trellis" or not d["beam"] or d["width"] != a.expect_width:
                if len(bad) < 8:
                    bad.append((os.path.basename(f), name, key))

    print(f"STAMP_SCAN files={len(files)} payloads={total} qtip2b={qtip} unquant={skipped}")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {v:6d}  {k}")
    for f, n, why in bad:
        print(f"  BAD {f} :: {n} :: {why}")

    want = f"trellis/beam(W={a.expect_width})/mse"
    if qtip == 0:
        print("STAMP_FAIL no qtip2b payloads at all — nothing was verified")
        return 1
    if tally.get(want, 0) == qtip and not bad:
        print(f"STAMP_OK all {qtip} qtip2b layers == {want} ({skipped} unquant skipped)")
        return 0
    print(f"STAMP_FAIL expected all {qtip} qtip2b layers == {want}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
