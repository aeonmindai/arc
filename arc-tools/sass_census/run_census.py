#!/usr/bin/env python3
"""QTIP trellis geometry SASS census -- compile, count, and price.

Parent system: ArcQuant / QTIP (measurement harness)

Runs nvcc -cubin (no GPU required), disassembles with cuobjdump, counts SASS
instructions per kernel, and turns the counts into inst/weight by DIFFERENCING
two unroll depths so the prologue and epilogue cancel exactly.

Nothing here estimates. Every number printed is derived from a compiled
instruction count, or is labelled ARITHMETIC (occupancy, budgets) because it is
a closed-form function of compiled inputs.
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "qtip_sass_census.cu")

# --- sm_90 (H100/H200) machine model ---------------------------------------
# Used only for occupancy arithmetic; all are architectural constants.
SMEM_PER_SM = 228 * 1024   # bytes usable by CUDA on Hopper
REGS_PER_SM = 65536
MAX_WARPS_PER_SM = 64
MAX_BLOCKS_PER_SM = 32
REG_ALLOC_GRAN = 8         # registers are allocated in units of 8 per thread
STATIC_SMEM_LIMIT = 48 * 1024

# Register-blocking factor compiled into every census kernel. Each of the ROWS
# rows runs its own independent trellis chain over the shared activations, so a
# group decodes WEIGHTS_PER_GROUP * ROWS weights, not WEIGHTS_PER_GROUP.
ROWS = 2

# NG ladder and the observed hard ceiling. Past ~4096 instructions the unroller
# clamps and the count stops tracking NG, silently breaking the differential.
STEP = 4          # NG spacing (4 -> 8 -> 12)
SATURATION = 4000  # refuse to report any count at or above this

# An instruction line looks like `/*0a10*/   IMAD R5, R2, R3, R4 ;`
# An encoding line looks like     `/* 0x000fe200078e00ff */`  -- note the space.
INSN_RE = re.compile(r"^\s*/\*[0-9a-f]{4}\*/\s+(\S.*?);")
FUNC_RE = re.compile(r"^\s*(?:\.section\s+\.text\.|Function : )(\S+)")

# ptxas -v lines.
PTXAS_ENTRY_RE = re.compile(r"Compiling entry function '(\S+)' for '(\S+)'")
PTXAS_USED_RE = re.compile(
    r"Used (\d+) registers"
    r"(?:.*?(\d+) bytes smem)?"
)
PTXAS_SPILL_RE = re.compile(r"(\d+) bytes spill stores,\s*(\d+) bytes spill loads")


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def compile_cubin(arch, out, extra=""):
    """nvcc -cubin: device-only codegen. No host compiler link, no GPU, no driver."""
    cmd = (
        f"nvcc -cubin -std=c++17 -arch={arch} -lineinfo -Xptxas -v "
        f"{extra} -o {out} {SRC}"
    )
    r = run(cmd)
    return cmd, r


def parse_ptxas(stderr):
    """-> {kernel: {'regs':int,'smem':int,'spill_st':int,'spill_ld':int}}"""
    info, cur = {}, None
    for line in stderr.splitlines():
        m = PTXAS_ENTRY_RE.search(line)
        if m:
            cur = m.group(1)
            info.setdefault(cur, {"regs": 0, "smem": 0, "spill_st": 0, "spill_ld": 0})
            continue
        if cur is None:
            continue
        m = PTXAS_SPILL_RE.search(line)
        if m:
            info[cur]["spill_st"] = int(m.group(1))
            info[cur]["spill_ld"] = int(m.group(2))
        if "Used" in line and "registers" in line:
            mr = re.search(r"Used (\d+) registers", line)
            if mr:
                info[cur]["regs"] = int(mr.group(1))
            ms = re.search(r"(\d+) bytes smem", line)
            if ms:
                info[cur]["smem"] = int(ms.group(1))
            cur = None
    return info


def parse_sass(text):
    """-> {kernel: instruction_count}. Counts only real instruction lines."""
    counts, cur = {}, None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            cur = m.group(1).strip(":")
            counts.setdefault(cur, 0)
            continue
        if cur and INSN_RE.match(line):
            counts[cur] += 1
    return counts


def occupancy(regs, smem, threads=256):
    """ARITHMETIC from compiled regs/smem. Returns (blocks, warps, pct, limiter)."""
    warps_per_block = threads // 32
    lim = []
    regs_per_thread = ((regs + REG_ALLOC_GRAN - 1) // REG_ALLOC_GRAN) * REG_ALLOC_GRAN
    by_reg = REGS_PER_SM // max(1, regs_per_thread * threads)
    lim.append((by_reg, "registers"))
    by_smem = (SMEM_PER_SM // smem) if smem > 0 else MAX_BLOCKS_PER_SM
    lim.append((by_smem, "shared"))
    lim.append((MAX_WARPS_PER_SM // warps_per_block, "warp slots"))
    lim.append((MAX_BLOCKS_PER_SM, "block slots"))
    blocks, limiter = min(lim, key=lambda t: t[0])
    warps = blocks * warps_per_block
    return blocks, warps, 100.0 * warps / MAX_WARPS_PER_SM, limiter


# Geometry table: name -> (K, V, L, prefix, weights_per_group, label)
# WIN_REPLAY ladder -- the mode the published 5.375 / 4.375 control was taken in.
GEOMS = [
    ("g1", 4, 2, 16, 8, "K4/V2/L16 computed (SHIPPED)"),
    ("g2", 4, 2, 13, 8, "K4/V2/L13 bf16 LUT"),
    ("g3", 8, 4, 12, 8, "K8/V4/L12 LUT  <-- CONTROL"),
    ("g4", 9, 4, 12, 8, "K9/V4/L12 LUT   <-- DECISION"),
    ("g5", 10, 4, 12, 8, "K10/V4/L12 LUT  <-- TREND"),
]

# WIN_SEQ ladder -- contiguous per-thread slice, trellis state carried across
# groups, warm-up once in the prologue. This is the shape the SERVING kernel
# (mistralrs-quant/kernels/qtip/qtip_gemv_k8v4l12.cu) actually has, and it pays
# HALF the extractions per weight that WIN_REPLAY does.
SEQ_GEOMS = [
    ("q3", 8, 4, 12, 8, "K8/V4/L12 seq   <-- CONTROL"),
    ("q4", 9, 4, 12, 8, "K9/V4/L12 seq   <-- DECISION"),
    ("q5", 10, 4, 12, 8, "K10/V4/L12 seq  <-- TREND"),
]

# Lever variants: prefix -> label. Each has _ng4 and _ng12 only.
LEVERS = [
    ("g1h", "g1", "SHIPPED + row-scale hoist (lever 2)"),
    ("g1w", "g1", "SHIPPED + PRMT window (lever 3)"),
    ("g1b", "g1", "SHIPPED + BREV window (lever 3, route A)"),
    ("g3h", "g3", "K8/V4/L12 + row-scale hoist"),
    ("g3w", "g3", "K8/V4/L12 + PRMT window"),
    ("g3wh", "g3", "K8/V4/L12 + BOTH levers"),
    ("g4h", "g4", "K9  replay + hoist [B2 2-byte]"),
    ("g4hc", "g4", "K9  replay + hoist [B2C tail-clamped]"),
    ("g4hf", "g4", "K9  replay + hoist [FUNNEL u32]"),
    ("g4hs", "g4", "K9  replay + hoist [SPLIT bit-plane]"),
    ("g5h", "g5", "K10 replay + hoist [B2 2-byte]"),
    ("g5hc", "g5", "K10 replay + hoist [B2C tail-clamped]"),
    ("g5h3", "g5", "K10 replay + hoist [B3 generic 3-byte]"),
    ("g5hf", "g5", "K10 replay + hoist [FUNNEL u32]"),
    ("q3h", "q3", "K8  seq + hoist  <-- SERVING CONTROL"),
    ("q4h", "q4", "K9  seq + hoist [B2 2-byte]"),
    ("q4hc", "q4", "K9  seq + hoist [B2C tail-clamped]"),
    ("q4hf", "q4", "K9  seq + hoist [FUNNEL u32]"),
    ("q4hs", "q4", "K9  seq + hoist [SPLIT bit-plane]"),
    ("q5h", "q5", "K10 seq + hoist [B2 2-byte]"),
    ("q5hc", "q5", "K10 seq + hoist [B2C tail-clamped]"),
    ("q5hf", "q5", "K10 seq + hoist [FUNNEL u32]"),
]

# The one format decision still open on the serving side: pad the row stride by
# MAX_BYTES-1 and read unclamped, or leave it unpadded and clamp every byte
# index. These pairs are (K, padded route, clamped route).
PAD_VS_CLAMP = [
    (9, "k9b2", "k9b2c"),
    (10, "k10b2", "k10b2c"),
]

# Extraction isolation micro-census: prefix -> (K, label). Each has _ns8/16/24.
# These kernels contain NOTHING but the symbol extract plus an XOR, and the XOR
# costs one instruction in every variant, so it cancels in the K delta.
EXTRACTS = [
    ("k4", 4, "K=4 nibble (shipped rung)"),
    ("k8", 8, "K=8 single byte  <-- ALIGNED CONTROL"),
    ("k9b2", 9, "K=9 2-byte, padded stride"),
    ("k9b2c", 9, "K=9 2-byte, tail-clamped"),
    ("k9fun", 9, "K=9 u32 pair + funnel shift"),
    ("k9split", 9, "K=9 byte plane + 1-bit plane"),
    ("k10b2", 10, "K=10 2-byte, padded stride"),
    ("k10b2c", 10, "K=10 2-byte, tail-clamped"),
    ("k10b3", 10, "K=10 3-byte generic read"),
    ("k10fun", 10, "K=10 u32 pair + funnel shift"),
    ("k10split", 10, "K=10 byte plane + 2-bit plane"),
]

# Symbols per group per row, and weights per group per row, for the V=4/L=12
# ladder. WARM == GROUP_SYMS == 2 is static_asserted in the .cu, so these are
# the same for K=8, 9 and 10 -- which is what makes the K delta a pure
# alignment delta rather than a change in how much work is done.
EXTRACTS_PER_WEIGHT = {
    "replay": (2 + 2) / 8.0,   # WARM + GROUP_SYMS extractions per 8 weights
    "seq": 2 / 8.0,            # GROUP_SYMS only; warm-up is prologue
}
NS_STEP = 8  # NS ladder spacing (8 -> 16 -> 24)

# Budget band, from the brief. bits/weight = K/V; the budget scales with bpw.
BUDGET_LO_AT_2BPW = 1.13   # 70% issue efficiency (BUDGET_V4_B1.md:571)
BUDGET_HI_AT_2BPW = 1.41   # 87.3% of sm_90 peak -- optimistic end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_90")
    args = ap.parse_args()

    cubin = "/tmp/qtip_census.cubin"
    cmd, r = compile_cubin(args.arch, cubin)
    print("=" * 78)
    print("NVCC COMMAND")
    print("=" * 78)
    print(cmd)
    ver = run("nvcc --version").stdout.strip().splitlines()
    print("\n".join(ver[-2:]) if ver else "")
    if r.returncode != 0:
        print("\nCOMPILE FAILED\n")
        print(r.stdout)
        print(r.stderr)
        return 1

    ptx = parse_ptxas(r.stderr)
    sass_txt = run(f"cuobjdump -sass {cubin}").stdout
    counts = parse_sass(sass_txt)
    if not counts:
        print("FATAL: cuobjdump produced no kernels; parser or build is wrong")
        print(sass_txt[:2000])
        return 1

    def c(name):
        return counts.get(f"census_{name}")

    print()
    print("=" * 78)
    print("RAW SASS INSTRUCTION COUNTS PER THREAD  (MEASURED)")
    print("=" * 78)
    for k in sorted(counts):
        p = ptx.get(k, {})
        print(f"  {k:<24} {counts[k]:>6} inst   regs={p.get('regs','?'):>4} "
              f"smem={p.get('smem',0):>6}B  spill={p.get('spill_st',0)}/"
              f"{p.get('spill_ld',0)}B")

    # ---- differential inst/weight -----------------------------------------
    print()
    print("=" * 78)
    print("DIFFERENTIAL inst/weight  (MEASURED; prologue+epilogue cancelled)")
    print("=" * 78)
    print(f"{'geometry':<32}{'bpw':>5}{'budget':>13}{'inst/wt':>10}"
          f"{'linearity':>11}{'verdict':>10}")
    print("-" * 78)

    results = {}

    def geom_table(table):
        for pre, K, V, L, wpg, label in table:
            n4, n8, n12 = c(f"{pre}_ng4"), c(f"{pre}_ng8"), c(f"{pre}_ng12")
            if None in (n4, n8, n12):
                print(f"  {label}: MISSING KERNELS")
                continue
            # WEIGHTS per group = weights_per_group * ROWS. ROWS is real work:
            # each of the ROWS rows decodes its own independent trellis chain
            # over the same activations. Omitting it understates the denominator
            # and inflates inst/weight by exactly ROWS (this bug made the first
            # run 2x too high).
            wpg_eff = wpg * ROWS
            d1 = (n8 - n4) / (STEP * wpg_eff)    # first half-difference
            d2 = (n12 - n8) / (STEP * wpg_eff)   # second half-difference
            ipw = (n12 - n4) / (2 * STEP * wpg_eff)
            lin = abs(d2 - d1) / max(d1, 1e-9) * 100.0
            bpw = K / V
            blo = BUDGET_LO_AT_2BPW * bpw / 2.0
            bhi = BUDGET_HI_AT_2BPW * bpw / 2.0
            sat = max(n4, n8, n12) >= SATURATION
            verdict = "SATURATED" if sat else ("OK" if lin < 1.0 else "NOT STEADY")
            results[pre] = dict(ipw=ipw, bpw=bpw, blo=blo, bhi=bhi, lin=lin,
                                label=label, over=ipw / bhi,
                                ok=(not sat and lin < 1.0))
            print(f"{label:<32}{bpw:>5.2f}{blo:>6.2f}-{bhi:<6.2f}{ipw:>10.3f}"
                  f"{lin:>10.2f}%{verdict:>11}")

    print("--- WIN_REPLAY (state re-seeded per group; the published control) ---")
    geom_table(GEOMS)
    print()
    print("--- WIN_SEQ (contiguous slice, state carried; the SERVING shape) ---")
    print(f"{'geometry':<32}{'bpw':>5}{'budget':>13}{'inst/wt':>10}"
          f"{'linearity':>11}{'verdict':>10}")
    print("-" * 78)
    geom_table(SEQ_GEOMS)

    # ---- levers ------------------------------------------------------------
    print()
    print("=" * 78)
    print("LEVERS  (MEASURED, same differential; delta vs its own baseline)")
    print("=" * 78)
    levers = {}
    for pre, base, label in LEVERS:
        n4, n12 = c(f"{pre}_ng4"), c(f"{pre}_ng12")
        if None in (n4, n12) or base not in results:
            continue
        wpg_eff = 8 * ROWS
        ipw = (n12 - n4) / (2 * STEP * wpg_eff)
        sat = max(n4, n12) >= SATURATION
        delta = ipw - results[base]["ipw"]
        levers[pre] = dict(ipw=ipw, sat=sat, label=label)
        flag = "  <-- SATURATED, DISCARD" if sat else ""
        print(f"{label:<44}{ipw:>9.3f} inst/wt   {delta:+7.3f} vs {base}{flag}")

    # ---- extraction isolation ---------------------------------------------
    # The number the K=9 decision turns on, with everything else stripped out.
    print()
    print("=" * 78)
    print("EXTRACTION COST, ISOLATED  (MEASURED; no table, no FMA, no shared mem)")
    print("=" * 78)
    print("Differencing NS=8 -> NS=24 over a body that is nothing but the symbol")
    print("extract plus one XOR. The XOR costs 1 inst in EVERY variant, so it")
    print("cancels in the delta-vs-K8 column, which is the alignment penalty.")
    print()
    print(f"{'extract route':<34}{'inst/sym':>10}{'vs K=8':>9}{'lin':>8}"
          f"{'regs':>6}   replay   seq")
    print("-" * 78)

    def ext(name):
        return counts.get(f"extract_{name}")

    # Two passes: every row's delta is taken against K=8, so the whole table has
    # to be measured before any of it can be printed.
    ext_res = {}
    missing = []
    for pre, K, label in EXTRACTS:
        n8_, n16_, n24_ = ext(f"{pre}_ns8"), ext(f"{pre}_ns16"), ext(f"{pre}_ns24")
        if None in (n8_, n16_, n24_):
            missing.append(label)
            continue
        d1 = (n16_ - n8_) / NS_STEP
        d2 = (n24_ - n16_) / NS_STEP
        ips = (n24_ - n8_) / (2 * NS_STEP)
        lin = abs(d2 - d1) / max(abs(d1), 1e-9) * 100.0
        sat = max(n8_, n16_, n24_) >= SATURATION
        ext_res[pre] = dict(ips=ips, lin=lin, sat=sat, K=K, label=label)

    if "k8" not in ext_res:
        print("  FATAL: the K=8 aligned control is missing; no delta is reportable")
    for pre, K, label in EXTRACTS:
        if pre not in ext_res:
            print(f"  {label}: MISSING KERNELS")
            continue
        e = ext_res[pre]
        regs = ptx.get(f"extract_{pre}_ns24", {}).get("regs", "?")
        d = e["ips"] - ext_res["k8"]["ips"] if "k8" in ext_res else float("nan")
        # A per-symbol penalty D lands on inst/weight scaled by how many
        # extractions each weight costs -- 0.5 in WIN_REPLAY, 0.25 in WIN_SEQ.
        pr = d * EXTRACTS_PER_WEIGHT["replay"]
        ps = d * EXTRACTS_PER_WEIGHT["seq"]
        flag = ("  <-- SATURATED" if e["sat"]
                else ("" if e["lin"] < 1.0 else "  <-- NOT STEADY"))
        print(f"{label:<34}{e['ips']:>10.3f}{d:>+9.3f}{e['lin']:>7.2f}%{str(regs):>6}"
              f"{pr:>+9.3f}{ps:>+7.3f}{flag}")

    print()
    print("The last two columns are the PREDICTED inst/weight penalty this route")
    print("adds, from the isolated per-symbol delta alone. The full-kernel tables")
    print("above measured the same quantity independently; they are cross-checked")
    print("below. Two instruments agreeing is the evidence, not either one alone.")

    # ---- the format decision ----------------------------------------------
    # Isolated in its own section because it is the one question blocking the
    # serving-side format: pad the row stride by MAX_BYTES-1 and read
    # unclamped, or leave it unpadded and clamp. The clamp is pure overhead --
    # both routes return the identical symbol (verified bit-exact on the host).
    print()
    print("=" * 78)
    print("FORMAT DECISION: pad the row stride, or clamp every byte index?")
    print("=" * 78)
    print("MAX_BYTES = ceil((8 - gcd(K,8) + K)/8): K=8 -> 1, K=9 -> 2, K=10 -> 2.")
    print("So padding costs MAX_BYTES-1 = 1 byte per ROW. What it buys:")
    print()
    print(f"{'':<10}{'padded':>10}{'clamped':>10}{'clamp cost':>12}"
          f"{'  -> inst/weight at 0.25 / 0.375 / 0.50 extractions/wt'}")
    print("-" * 78)
    for K, padded, clamped in PAD_VS_CLAMP:
        if padded not in ext_res or clamped not in ext_res:
            print(f"K={K:<8} MISSING ({padded} or {clamped} did not compile)")
            continue
        p_, c_ = ext_res[padded]["ips"], ext_res[clamped]["ips"]
        d = c_ - p_
        print(f"K={K:<8}{p_:>10.3f}{c_:>10.3f}{d:>+12.3f}"
              f"      {d*0.25:+.3f} / {d*0.375:+.3f} / {d*0.50:+.3f}")
    print()
    print("The three rightmost figures are the SAME clamp cost expressed as")
    print("inst/weight at three amortisations: 0.25 = long contiguous slices,")
    print("0.375 = a short-slice kernel paying its warm-up per row, 0.50 = the")
    print("per-group re-seed. Multiply the clamp cost by YOUR shape's measured")
    print("extractions/weight rather than quoting any one of them.")

    # ---- instrument cross-check -------------------------------------------
    print()
    print("=" * 78)
    print("CROSS-CHECK: isolated extract delta  vs  full-kernel delta")
    print("=" * 78)
    print(f"{'comparison':<40}{'predicted':>11}{'observed':>10}{'agree':>10}")
    print("-" * 78)
    checks = [
        ("K9 B2 vs K8, replay+hoist", "k9b2", "g4h", "g3h", "replay"),
        ("K9 B2C vs K8, replay+hoist", "k9b2c", "g4hc", "g3h", "replay"),
        ("K9 FUN vs K8, replay+hoist", "k9fun", "g4hf", "g3h", "replay"),
        ("K9 SPLIT vs K8, replay+hoist", "k9split", "g4hs", "g3h", "replay"),
        ("K10 B2 vs K8, replay+hoist", "k10b2", "g5h", "g3h", "replay"),
        ("K9 B2 vs K8, seq+hoist", "k9b2", "q4h", "q3h", "seq"),
        ("K10 B2C vs K8, replay+hoist", "k10b2c", "g5hc", "g3h", "replay"),
        ("K9 B2C vs K8, seq+hoist", "k9b2c", "q4hc", "q3h", "seq"),
        ("K10 B2C vs K8, seq+hoist", "k10b2c", "q5hc", "q3h", "seq"),
        ("K9 FUN vs K8, seq+hoist", "k9fun", "q4hf", "q3h", "seq"),
        ("K9 SPLIT vs K8, seq+hoist", "k9split", "q4hs", "q3h", "seq"),
        ("K10 B2 vs K8, seq+hoist", "k10b2", "q5h", "q3h", "seq"),
        ("K10 FUN vs K8, seq+hoist", "k10fun", "q5hf", "q3h", "seq"),
    ]
    for label, e, lv, ctrl, mode in checks:
        if e not in ext_res or lv not in levers or ctrl not in levers:
            print(f"{label:<40}{'MISSING':>11}")
            continue
        pred = (ext_res[e]["ips"] - ext_res["k8"]["ips"]) * EXTRACTS_PER_WEIGHT[mode]
        obs = levers[lv]["ipw"] - levers[ctrl]["ipw"]
        ok = "OK" if abs(pred - obs) <= 0.15 else "DISAGREE"
        print(f"{label:<40}{pred:>+11.3f}{obs:>+10.3f}{ok:>10}")

    # ---- shared memory + occupancy ----------------------------------------
    print()
    print("=" * 78)
    print("SHARED MEMORY + OCCUPANCY  (smem/regs MEASURED; occupancy ARITHMETIC)")
    print("=" * 78)
    print(f"{'kernel':<26}{'smem B':>9}{'>48K?':>7}{'regs':>6}{'spill':>7}"
          f"{'blk/SM':>8}{'warps':>7}{'occ%':>7}  limiter")
    print("-" * 78)
    for pre, K, V, L, wpg, label in GEOMS + SEQ_GEOMS:
        k = f"census_{pre}_ng12"
        p = ptx.get(k)
        if not p:
            continue
        smem, regs = p["smem"], p["regs"]
        b, w, pct, limiter = occupancy(regs, smem)
        over = "YES" if smem > STATIC_SMEM_LIMIT else "no"
        spill = p["spill_st"] + p["spill_ld"]
        print(f"{pre + ' ' + label[:20]:<26}{smem:>9}{over:>7}{regs:>6}"
              f"{spill:>7}{b:>8}{w:>7}{pct:>6.1f}%  {limiter}")
    # The hoisted variants are what would actually ship, so their register and
    # spill figures are the ones that matter for occupancy.
    print()
    print("  (hoisted variants -- the shipping configuration)")
    for pre in ("g3h", "g4h", "g4hf", "g5h", "q3h", "q4h", "q4hf", "q5h"):
        p = ptx.get(f"census_{pre}_ng12")
        if not p:
            continue
        smem, regs = p["smem"], p["regs"]
        b, w, pct, limiter = occupancy(regs, smem)
        over = "YES" if smem > STATIC_SMEM_LIMIT else "no"
        spill = p["spill_st"] + p["spill_ld"]
        print(f"{pre:<26}{smem:>9}{over:>7}{regs:>6}{spill:>7}{b:>8}{w:>7}"
              f"{pct:>6.1f}%  {limiter}")

    print()
    print("NOTE: static shared limit without cudaFuncSetAttribute is "
          f"{STATIC_SMEM_LIMIT} B.")
    print("      No QTIP kernel calls cudaFuncSetAttribute today; Marlin does")
    print("      (marlin_kernel.cuh:1339), so the mechanism exists in-tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
