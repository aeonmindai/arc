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
GEOMS = [
    ("g1", 4, 2, 16, 8, "K4/V2/L16 computed (SHIPPED)"),
    ("g2", 4, 2, 13, 8, "K4/V2/L13 bf16 LUT"),
    ("g3", 8, 4, 12, 8, "K8/V4/L12 bf16 LUT"),
]

# Lever variants: prefix -> label. Each has _ng8 and _ng24 only.
LEVERS = [
    ("g1h", "g1", "SHIPPED + row-scale hoist (lever 2)"),
    ("g1w", "g1", "SHIPPED + PRMT window (lever 3)"),
    ("g1b", "g1", "SHIPPED + BREV window (lever 3, route A)"),
    ("g3h", "g3", "K8/V4/L12 + row-scale hoist"),
    ("g3w", "g3", "K8/V4/L12 + PRMT window"),
    ("g3wh", "g3", "K8/V4/L12 + BOTH levers"),
]

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
    for pre, K, V, L, wpg, label in GEOMS:
        n8, n16, n24 = c(f"{pre}_ng8"), c(f"{pre}_ng16"), c(f"{pre}_ng24")
        if None in (n8, n16, n24):
            print(f"  {label}: MISSING KERNELS")
            continue
        d1 = (n16 - n8) / (8 * wpg)     # first half-difference
        d2 = (n24 - n16) / (8 * wpg)    # second half-difference
        ipw = (n24 - n8) / (16 * wpg)
        lin = abs(d2 - d1) / max(d1, 1e-9) * 100.0
        bpw = K / V
        blo = BUDGET_LO_AT_2BPW * bpw / 2.0
        bhi = BUDGET_HI_AT_2BPW * bpw / 2.0
        verdict = "OK" if lin < 1.0 else "NOT STEADY"
        results[pre] = dict(ipw=ipw, bpw=bpw, blo=blo, bhi=bhi, lin=lin,
                            label=label, over=ipw / bhi)
        print(f"{label:<32}{bpw:>5.2f}{blo:>6.2f}-{bhi:<6.2f}{ipw:>10.3f}"
              f"{lin:>10.2f}%{verdict:>10}")

    # ---- levers ------------------------------------------------------------
    print()
    print("=" * 78)
    print("LEVERS  (MEASURED, same differential; delta vs its own baseline)")
    print("=" * 78)
    for pre, base, label in LEVERS:
        n8, n24 = c(f"{pre}_ng8"), c(f"{pre}_ng24")
        if None in (n8, n24) or base not in results:
            continue
        wpg = 8
        ipw = (n24 - n8) / (16 * wpg)
        delta = ipw - results[base]["ipw"]
        print(f"{label:<44}{ipw:>9.3f} inst/wt   {delta:+7.3f} vs {base}")

    # ---- shared memory + occupancy ----------------------------------------
    print()
    print("=" * 78)
    print("SHARED MEMORY + OCCUPANCY  (smem/regs MEASURED; occupancy ARITHMETIC)")
    print("=" * 78)
    print(f"{'kernel':<26}{'smem B':>9}{'>48K?':>7}{'regs':>6}{'spill':>7}"
          f"{'blk/SM':>8}{'warps':>7}{'occ%':>7}  limiter")
    print("-" * 78)
    for pre, K, V, L, wpg, label in GEOMS:
        k = f"census_{pre}_ng24"
        p = ptx.get(k)
        if not p:
            continue
        smem, regs = p["smem"], p["regs"]
        b, w, pct, limiter = occupancy(regs, smem)
        over = "YES" if smem > STATIC_SMEM_LIMIT else "no"
        spill = p["spill_st"] + p["spill_ld"]
        print(f"{pre + ' ' + label[:20]:<26}{smem:>9}{over:>7}{regs:>6}"
              f"{spill:>7}{b:>8}{w:>7}{pct:>6.1f}%  {limiter}")

    print()
    print("NOTE: static shared limit without cudaFuncSetAttribute is "
          f"{STATIC_SMEM_LIMIT} B.")
    print("      No QTIP kernel calls cudaFuncSetAttribute today; Marlin does")
    print("      (marlin_kernel.cuh:1339), so the mechanism exists in-tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
