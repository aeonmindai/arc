#!/usr/bin/env python3
"""D12 smoke test for read_stamp.py — both arms, no artifact required.

A reader that only ever prints OK is worthless. This plants four payloads whose
answers are known and asserts the reader separates them.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
READER = os.path.join(HERE, "read_qtip_stamp.py")
UQFF_VERSION = 768
QTIP2B = 10


def payload(stamp, flags, width=None, qtype=QTIP2B, version=UQFF_VERSION):
    b = bytearray()
    b += struct.pack("<I", version)
    b.append(qtype)
    b.append(0)                       # has_bias
    b += struct.pack("<I", 4096)      # in_features
    b += struct.pack("<I", 0xCAF6A435)  # mcg_mult
    b += b"\x00" * 64                 # stand-in for the tensor payloads
    b.append(stamp)
    b.append(flags)
    if width is not None:
        b += struct.pack("<H", width)
    return bytes(b)


def write_st(path, tensors):
    header, off = {}, 0
    for name, blob in tensors:
        header[name] = {"dtype": "U8", "shape": [len(blob)], "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    hb = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for _n, blob in tensors:
            fh.write(blob)


def run(folder, width=256):
    p = subprocess.run(
        [sys.executable, READER, "--folder", folder, "--expect-width", str(width)],
        capture_output=True, text=True,
    )
    return p.returncode, p.stdout


UNQ = payload(0, 0, qtype=1)  # UnquantLinear: no trellis, no stamp to read

CASES = [
    ("beam W=256 (the artifact we want)", [payload(1, 0x01, 256)] * 3, 0, "STAMP_OK"),
    ("GREEDY stamp — D4 banned",          [payload(2, 0x00)] * 3,      1, "GREEDY"),
    ("exhaustive (no beam)",              [payload(1, 0x00)] * 3,      1, "exhaustive"),
    ("one greedy layer hidden in 3",      [payload(1, 0x01, 256), payload(2, 0x00), payload(1, 0x01, 256)], 1, "GREEDY"),
    ("beam at the WRONG width (64)",      [payload(1, 0x01, 64)] * 3,  1, "beam(W=64)"),
    ("wrong rung (qtip LUT, type 8)",     [payload(1, 0x01, 256, qtype=8)] * 3, 1, "the qtip2 LUT rung"),
    # The real artifact mixes Qtip2b stacks with UnquantLinear passthroughs.
    ("unquant passthroughs are skipped, not failed",
     [payload(1, 0x01, 256), UNQ, payload(1, 0x01, 256), UNQ], 0, "unquant=2"),
    # ...and skipping them must not become a way to hide a bad layer.
    ("greedy layer hidden among unquants",
     [UNQ, payload(2, 0x00), UNQ, payload(1, 0x01, 256)], 1, "GREEDY"),
    # An artifact with nothing to verify must not pass by vacuity.
    ("ALL unquant — nothing verified, must not pass", [UNQ] * 3, 1, "nothing was verified"),
]

fails = 0
for name, blobs, want_rc, want_txt in CASES:
    with tempfile.TemporaryDirectory() as d:
        write_st(os.path.join(d, "t-0.uqff"), [(f"l{i}", b) for i, b in enumerate(blobs)])
        rc, out = run(d)
    ok = (rc == want_rc) and (want_txt in out)
    print(f"{'PASS' if ok else 'FAIL'}  rc={rc} (want {want_rc})  {name}")
    if not ok:
        fails += 1
        print(out)
print("ALL ARMS PASS" if not fails else f"{fails} ARM(S) FAILED")
sys.exit(1 if fails else 0)
