#!/usr/bin/env python3
"""FlashAttention-4 feasibility + benchmark probe for Arc.

WHY THIS EXISTS
---------------
FA4 is written in CuTeDSL and JIT-compiled at runtime through ``cute.compile()``.
It exposes **no C or C++ API**, so a Rust engine cannot link against it the way
``candle-flash-attn`` (FA2) and ``candle-flash-attn-v3`` (FA3) link against
their C++ kernels. Whether FA4 can be called from Arc at all therefore hinges on
one empirical question:

    Can the compiled kernel be recovered as a cubin/PTX and launched from the
    CUDA driver API, without a Python interpreter in the serving path?

Nothing in the FA4 docs answers that, and it cannot be settled by reading —
CuTeDSL's compile cache is an implementation detail that changes between
releases. This script settles it **on the box**, by trying every plausible route
and reporting which ones actually produced bytes.

It also answers the two other questions that gate the work: which head_dims FA4
accepts on this GPU, and how FA4 compares to FA3/FA2 at the shapes Arc serves.

RUNS ON: any CUDA box with a Hopper or Blackwell GPU.
NEEDS:   torch. Optionally flash-attn (FA2/FA3) and flash-attn-4 (FA4); missing
         backends are reported as such rather than crashing the run.

    # MANDATORY on a fresh runcrate image: it ships pip 22.0.2 with no numpy,
    # and flash-attn's metadata generation then dies with
    #   TypeError: canonicalize_version() got an unexpected keyword argument
    #              'strip_trailing_zero'
    python3 -m pip install -q --upgrade pip setuptools wheel
    python3 -m pip install -q numpy

    pip install flash-attn-4          # provides flash_attn.cute
    python3 arc-tools/fa4/fa4_probe.py --json /tmp/fa4_probe.json

Every section is independent: a failure in one is recorded and the probe
continues, so a single run yields the complete picture even when FA4 is broken
on this arch. Nothing here writes into the repo.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import platform
import sys
import time
import traceback
from typing import Any, Callable

# --------------------------------------------------------------------------
# Result accumulator
# --------------------------------------------------------------------------

RESULT: dict[str, Any] = {
    "schema": "arc.fa4_probe/1",
    "env": {},
    "backends": {},
    "head_dims": {},
    "jit": {},
    "aot": {},
    "bench": [],
    "errors": [],
}


def record_error(where: str, exc: BaseException) -> None:
    RESULT["errors"].append(
        {"where": where, "type": type(exc).__name__, "msg": str(exc)[:600]}
    )
    print(f"  !! {where}: {type(exc).__name__}: {str(exc)[:300]}")


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def guarded(where: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn``; record and swallow any exception. Never aborts the probe."""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 - deliberately total
        record_error(where, exc)
        return None


# --------------------------------------------------------------------------
# 1. Environment
# --------------------------------------------------------------------------


def probe_env() -> None:
    section("1. ENVIRONMENT")
    env = RESULT["env"]
    env["python"] = sys.version.split()[0]
    env["platform"] = platform.platform()

    try:
        import torch
    except ImportError as exc:
        record_error("import torch", exc)
        print("  torch missing — cannot continue meaningfully.")
        return

    env["torch"] = torch.__version__
    env["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        print("  NO CUDA DEVICE. Benchmarks and AOT probes will be skipped.")
        return

    props = torch.cuda.get_device_properties(0)
    env["gpu_name"] = props.name
    env["sm"] = f"{props.major}{props.minor}"
    env["torch_cuda"] = torch.version.cuda
    env["total_mem_gb"] = round(props.total_memory / 1e9, 1)
    print(f"  torch {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"  GPU: {props.name}  sm_{props.major}{props.minor}  "
          f"{env['total_mem_gb']} GB")
    if props.major == 9:
        print("  -> Hopper. FA3 is the mature path here; FA4's stated target is "
              "Hopper AND Blackwell.")
    elif props.major >= 10:
        print("  -> Blackwell. FA4's primary optimization target.")


# --------------------------------------------------------------------------
# 2. Which backends actually import and run
# --------------------------------------------------------------------------


def probe_backends() -> None:
    section("2. BACKEND AVAILABILITY")
    b = RESULT["backends"]

    # FA2 / FA3 live in the same `flash_attn` distribution.
    for name, modpath, attr in [
        ("fa2", "flash_attn", "flash_attn_func"),
        ("fa3", "flash_attn_interface", "flash_attn_func"),
        ("fa4", "flash_attn.cute", "flash_attn_func"),
    ]:
        entry: dict[str, Any] = {"available": False}
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, attr, None)
            entry["available"] = fn is not None
            entry["module"] = modpath
            entry["version"] = getattr(mod, "__version__", None)
            entry["file"] = getattr(mod, "__file__", None)
            if fn is None:
                entry["note"] = f"module imported but has no {attr}"
        except BaseException as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        b[name] = entry
        status = "OK" if entry["available"] else "MISSING"
        print(f"  {name.upper():4s} {status:8s} {modpath}"
              + (f"  ({entry.get('error','')})" if entry.get("error") else ""))

    # The DSL itself — this is what would have to emit a cubin.
    for modpath in ("cutlass", "cutlass.cute"):
        try:
            mod = importlib.import_module(modpath)
            b[modpath] = {
                "available": True,
                "version": getattr(mod, "__version__", None),
                "file": getattr(mod, "__file__", None),
            }
            print(f"  DSL  OK       {modpath} "
                  f"v{getattr(mod, '__version__', '?')}")
        except BaseException as exc:  # noqa: BLE001
            b[modpath] = {"available": False,
                          "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"  DSL  MISSING  {modpath}")


# --------------------------------------------------------------------------
# 3. Shapes. V4's real numbers are read from config when available.
# --------------------------------------------------------------------------

# DeepSeek-V4-Flash: head_dim 512 (448 nope + 64 rope), 64 Q heads, 1 KV head
# (fused wkv MQA). Recorded for completeness — it is OUTSIDE every FA
# generation's envelope (FA2/FA3 cap at 256), which is precisely why V4 cannot
# use any of them. See mistralrs-core/src/attention/backends/sinks.rs:78.
V4_SHAPE = dict(name="v4-decode(UNSUPPORTED by all FA)", b=1, sq=1, sk=2048,
                hq=64, hkv=1, d=512)

# The shapes that actually matter for FA4 in Arc: the head_dim<=256 fleet.
FLEET_SHAPES = [
    # (name, batch, seqlen_q, seqlen_k, heads_q, heads_kv, head_dim)
    dict(name="llama-8b-decode", b=32, sq=1, sk=2048, hq=32, hkv=8, d=128),
    dict(name="llama-8b-prefill", b=1, sq=2048, sk=2048, hq=32, hkv=8, d=128),
    dict(name="llama-70b-decode", b=32, sq=1, sk=4096, hq=64, hkv=8, d=128),
    dict(name="llama-70b-prefill", b=1, sq=4096, sk=4096, hq=64, hkv=8, d=128),
    dict(name="qwen-d64-decode", b=32, sq=1, sk=2048, hq=32, hkv=4, d=64),
    dict(name="gemma-d256-prefill", b=1, sq=2048, sk=2048, hq=16, hkv=8, d=256),
]

CANDIDATE_HEAD_DIMS = [32, 64, 96, 128, 192, 256, 512, 576]


def _make_qkv(shape: dict, dtype, layout: str = "bshd"):
    import torch

    dev = "cuda"
    q = torch.randn(shape["b"], shape["sq"], shape["hq"], shape["d"],
                    device=dev, dtype=dtype)
    k = torch.randn(shape["b"], shape["sk"], shape["hkv"], shape["d"],
                    device=dev, dtype=dtype)
    v = torch.randn(shape["b"], shape["sk"], shape["hkv"], shape["d"],
                    device=dev, dtype=dtype)
    return q, k, v


# --------------------------------------------------------------------------
# 4. head_dim envelope — what does FA4 actually accept on THIS gpu?
# --------------------------------------------------------------------------


def probe_head_dims() -> None:
    section("3. FA4 head_dim ENVELOPE (empirical, this GPU)")
    if not RESULT["backends"].get("fa4", {}).get("available"):
        print("  FA4 not importable — skipping.")
        RESULT["head_dims"]["skipped"] = "fa4 unavailable"
        return
    try:
        import torch
        from flash_attn.cute import flash_attn_func as fa4
    except BaseException as exc:  # noqa: BLE001
        record_error("fa4 import for head_dim probe", exc)
        return

    for d in CANDIDATE_HEAD_DIMS:
        shape = dict(name=f"d{d}", b=1, sq=64, sk=64, hq=8, hkv=8, d=d)
        entry: dict[str, Any] = {}

        def run(shape=shape):
            q, k, v = _make_qkv(shape, torch.bfloat16)
            out = fa4(q, k, v, causal=True)
            torch.cuda.synchronize()
            return tuple(out.shape) if hasattr(out, "shape") else None

        t0 = time.perf_counter()
        try:
            entry["ok"] = True
            entry["out_shape"] = run()
            entry["wall_s"] = round(time.perf_counter() - t0, 3)
            print(f"  head_dim {d:4d}  ACCEPTED   ({entry['wall_s']}s incl JIT)")
        except BaseException as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"  head_dim {d:4d}  rejected   {str(exc)[:110]}")
        RESULT["head_dims"][str(d)] = entry

    accepted = [d for d, e in RESULT["head_dims"].items()
                if isinstance(e, dict) and e.get("ok")]
    print(f"\n  ACCEPTED head_dims: {accepted}")
    if "512" not in accepted:
        print("  NOTE: head_dim 512 rejected => FA4 cannot serve DeepSeek-V4 "
              "attention, consistent with FA2/FA3.")


# --------------------------------------------------------------------------
# 5. JIT cost — the number that sets the AOT cost model
# --------------------------------------------------------------------------


def probe_jit_cost() -> None:
    section("4. FA4 JIT COMPILE COST (cute.compile first call)")
    if not RESULT["backends"].get("fa4", {}).get("available"):
        print("  FA4 not importable — skipping.")
        return
    try:
        import torch
        from flash_attn.cute import flash_attn_func as fa4
    except BaseException as exc:  # noqa: BLE001
        record_error("fa4 import for jit probe", exc)
        return

    # A config unlikely to be warm from the head_dim sweep (different seqlen and
    # head counts change the config key on most CuTeDSL kernels).
    shape = dict(name="jit", b=2, sq=333, sk=333, hq=16, hkv=4, d=128)

    def once(causal: bool, tag: str) -> None:
        q, k, v = _make_qkv(shape, torch.bfloat16)
        t0 = time.perf_counter()
        fa4(q, k, v, causal=causal)
        torch.cuda.synchronize()
        cold = time.perf_counter() - t0

        t1 = time.perf_counter()
        for _ in range(5):
            fa4(q, k, v, causal=causal)
        torch.cuda.synchronize()
        warm = (time.perf_counter() - t1) / 5

        RESULT["jit"][tag] = {"cold_s": round(cold, 3),
                              "warm_s": round(warm, 6),
                              "ratio": round(cold / max(warm, 1e-9), 1)}
        print(f"  {tag:16s} cold {cold:8.3f}s   warm {warm * 1e6:9.1f}us   "
              f"={cold / max(warm, 1e-9):,.0f}x")

    guarded("jit causal", lambda: once(True, "causal=True"))
    guarded("jit non-causal", lambda: once(False, "causal=False"))
    print("\n  Cold time is per (config key) and is paid on EVERY process start "
          "unless a persistent cache or an AOT artifact exists. That is the "
          "number the AOT path has to beat.")


# --------------------------------------------------------------------------
# 6. THE DECISIVE SECTION — can we get a cubin/PTX out?
# --------------------------------------------------------------------------

CACHE_ENV_CANDIDATES = [
    "CUTE_DSL_KEEP",             # current form: CUTE_DSL_KEEP=cubin
    "CUTE_DSL_KEEP_CUBIN",       # deprecated predecessor
    "CUTE_DSL_DUMP_DIR",
    "CUTE_DSL_CACHE_DIR",
    "CUTE_DSL_FILE_CACHING_CAPACITY",
    "CUTE_DSL_DISABLE_FILE_CACHING",
    "CUTE_DSL_LOG_LEVEL",
    "CUTE_DSL_ARCH",
]


def probe_aot() -> None:
    """Establish which of the three documented cubin routes works on this box.

    Route 1  `compiled.__cubin__`         -> bytes, in-process
    Route 2  `CUTE_DSL_KEEP=cubin`        -> {DUMP_DIR}/{fn}.{arch}.cubin on disk
    Route 3  `export_to_c()`/`dump_to_object()` -> .o + .h with a C ABI

    Route 3 is what FA4 ships in production and is the one that gets us a Rust
    binding with **no Python at runtime and no serve-time JIT**. Route 2 is the
    leanest to automate. Route 1 is the cheapest to verify.
    """
    section("5. AOT / CUBIN EXTRACTION  <-- THE DECISIVE QUESTION")
    aot = RESULT["aot"]

    aot["env_seen"] = {k: os.environ.get(k) for k in CACHE_ENV_CANDIDATES
                       if os.environ.get(k) is not None}
    print(f"  CuTeDSL env vars set: {aot['env_seen'] or 'none'}")

    # FA4 pins nvidia-cutlass-dsl>=4.6.2; the API surface moves between minors,
    # so record exactly what is installed alongside every result below.
    def dsl_version() -> None:
        import cutlass
        aot["dsl_version"] = getattr(cutlass, "__version__", None)
        print(f"  nvidia-cutlass-dsl: {aot['dsl_version']}")

    guarded("dsl version", dsl_version)

    # --- Route 1: documented `__cubin__` / `__ptx__` / `__sass__` / `__mlir__`
    #     on the object returned by cute.compile (CuTeDSL guides/debugging.rst).
    def route1() -> None:
        import cutlass.cute as cute

        r: dict[str, Any] = {"route": "compiled.__cubin__"}
        # A trivial kernel is enough to establish the attribute contract; we do
        # not need FA4 itself to answer "does this API exist and return bytes".
        try:
            @cute.jit
            def _noop():
                pass

            compiled = cute.compile(_noop)
            for attr in ("__cubin__", "__ptx__", "__sass__", "__mlir__"):
                val = getattr(compiled, attr, None)
                if val is None:
                    r[attr] = None
                    continue
                if isinstance(val, (bytes, bytearray)):
                    r[attr] = {"nbytes": len(val),
                               "elf": bytes(val[:4]) == b"\x7fELF"}
                else:
                    r[attr] = {"type": type(val).__name__,
                               "repr": str(val)[:200]}
            # `.artifacts.CUBIN` is contested in the sources — bytes in one
            # reading, a path in another. Resolve it empirically.
            arts = getattr(compiled, "artifacts", None)
            if arts is not None:
                cub = getattr(arts, "CUBIN", None)
                r["artifacts.CUBIN"] = {
                    "type": type(cub).__name__,
                    "is_bytes": isinstance(cub, (bytes, bytearray)),
                    "repr": str(cub)[:200],
                }
            r["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            r["ok"] = False
            r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        aot["route1"] = r
        print(f"  ROUTE 1 (__cubin__): {'OK' if r.get('ok') else 'FAILED'}  "
              f"{ {k: v for k, v in r.items() if k.startswith('__')} }")

    guarded("route1", route1)

    # --- Route 2: CUTE_DSL_KEEP=cubin writes {DUMP_DIR}/{fn}.{arch}.cubin.
    #     Must run in a CHILD process: the env var is read at DSL import.
    def route2() -> None:
        import subprocess
        import tempfile

        dump = tempfile.mkdtemp(prefix="fa4_cubin_")
        env = dict(os.environ, CUTE_DSL_KEEP="cubin", CUTE_DSL_DUMP_DIR=dump)
        code = (
            "import torch\n"
            "from flash_attn.cute import flash_attn_func as f\n"
            "q=torch.randn(1,128,8,128,device='cuda',dtype=torch.bfloat16)\n"
            "f(q,q,q,causal=True); torch.cuda.synchronize()\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, timeout=1800)
        files = [f for f in glob.glob(os.path.join(dump, "**", "*"),
                                      recursive=True) if os.path.isfile(f)]
        cubins = [f for f in files if f.endswith((".cubin", ".ptx", ".sass"))]
        aot["route2"] = {
            "route": "CUTE_DSL_KEEP=cubin",
            "dump_dir": dump,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-800:],
            "n_files": len(files),
            "cubins": [{"path": f, "nbytes": os.path.getsize(f)}
                       for f in cubins[:20]],
        }
        print(f"  ROUTE 2 (CUTE_DSL_KEEP=cubin): rc={proc.returncode}, "
              f"{len(cubins)} cubin/ptx files in {dump}")
        for c in cubins[:10]:
            print(f"      {os.path.basename(c)}  {os.path.getsize(c)} bytes")

    guarded("route2", route2)

    # --- Route 3: export_to_c / dump_to_object -> linkable .o with a C ABI.
    #     THIS is the one that yields a Rust binding with no Python at runtime.
    def route3() -> None:
        import cutlass.cute as cute

        r: dict[str, Any] = {"route": "export_to_c / dump_to_object"}
        try:
            @cute.jit
            def _noop():
                pass

            compiled = cute.compile(_noop)
            r["has_export_to_c"] = hasattr(compiled, "export_to_c")
            r["has_dump_to_object"] = hasattr(compiled, "dump_to_object")
            r["exportish_attrs"] = [a for a in dir(compiled)
                                    if any(t in a.lower()
                                           for t in ("export", "object",
                                                     "dump", "aot"))]
            if hasattr(compiled, "dump_to_object"):
                obj = compiled.dump_to_object("arc_fa4_probe")
                r["dump_to_object"] = {
                    "type": type(obj).__name__,
                    "nbytes": len(obj) if isinstance(obj, (bytes, bytearray))
                    else None,
                }
            r["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            r["ok"] = False
            r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        aot["route3"] = r
        print(f"  ROUTE 3 (export_to_c/.o): export_to_c="
              f"{r.get('has_export_to_c')} dump_to_object="
              f"{r.get('has_dump_to_object')} "
              f"attrs={r.get('exportish_attrs')}")

    guarded("route3", route3)

    # --- Runtime .so symbols. NVIDIA's AOT docs name CuteDSLRT_Module_Load,
    #     which does NOT exist in the shipped wheel; the real entry point is
    #     CuteDSLRT_Module_Create_From_Bytes. Bind against the wheel, not docs.
    def runtime_syms() -> None:
        import subprocess

        libs = glob.glob(os.path.join(
            os.path.dirname(importlib.import_module("cutlass").__file__),
            "**", "libcute_dsl_runtime*.so"), recursive=True)
        libs += glob.glob("/usr/**/libcute_dsl_runtime*.so", recursive=True)
        out: dict[str, Any] = {"libs": libs[:5], "symbols": {}}
        for lib in libs[:2]:
            try:
                syms = subprocess.run(["nm", "-D", "--defined-only", lib],
                                      capture_output=True, text=True,
                                      timeout=60).stdout
                out["symbols"][lib] = sorted(
                    {ln.split()[-1] for ln in syms.splitlines()
                     if "CuteDSLRT" in ln or "tvm_ffi" in ln.lower()}
                )[:60]
            except BaseException as exc:  # noqa: BLE001
                out["symbols"][lib] = f"nm failed: {exc}"
        aot["runtime_symbols"] = out
        print(f"  Runtime lib(s): {libs[:2] or 'not found'}")
        for lib, syms in out["symbols"].items():
            print(f"    {os.path.basename(lib)}: {len(syms) if isinstance(syms, list) else syms} "
                  f"CuteDSLRT/tvm_ffi symbols")
            if isinstance(syms, list):
                for s in syms[:15]:
                    print(f"      {s}")

    guarded("runtime symbols", runtime_syms)

    # --- Verdict, written by the data rather than by us.
    r1 = (aot.get("route1") or {})
    r2 = (aot.get("route2") or {})
    r3 = (aot.get("route3") or {})
    cubin_in_proc = isinstance(r1.get("__cubin__"), dict) and \
        (r1["__cubin__"].get("nbytes") or 0) > 0
    cubin_on_disk = len(r2.get("cubins") or []) > 0
    c_abi = bool(r3.get("has_export_to_c") or r3.get("has_dump_to_object"))

    if c_abi:
        conclusion = "USE_ROUTE3_EXPORT_TO_C__no_python_at_runtime"
    elif cubin_on_disk or cubin_in_proc:
        conclusion = "USE_CUBIN_AOT__cuModuleLoadData_from_rust"
    else:
        conclusion = "NO_CUBIN_ACCESS_ON_THIS_VERSION__investigate"

    aot["verdict"] = {
        "cubin_reachable_in_process": cubin_in_proc,
        "cubin_on_disk": cubin_on_disk,
        "c_abi_export_available": c_abi,
        "conclusion": conclusion,
    }
    print(f"\n  VERDICT: {conclusion}")


# --------------------------------------------------------------------------
# 7. Benchmark: FA2 vs FA3 vs FA4 at fleet shapes
# --------------------------------------------------------------------------


def _bench_one(fn: Callable, q, k, v, causal: bool, iters: int = 30) -> float:
    """Median seconds per call, CUDA-event timed, after warmup."""
    import torch

    for _ in range(5):
        fn(q, k, v, causal=causal)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(q, k, v, causal=causal)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    return times[len(times) // 2]


def probe_bench(shapes: list[dict]) -> None:
    section("6. BENCHMARK  FA2 vs FA3 vs FA4  (fleet shapes, head_dim<=256)")
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("  no CUDA — skipping")
        return

    backends: dict[str, Callable] = {}
    if RESULT["backends"].get("fa2", {}).get("available"):
        from flash_attn import flash_attn_func as _fa2
        backends["fa2"] = lambda q, k, v, causal: _fa2(q, k, v, causal=causal)
    if RESULT["backends"].get("fa3", {}).get("available"):
        from flash_attn_interface import flash_attn_func as _fa3
        backends["fa3"] = lambda q, k, v, causal: _fa3(q, k, v, causal=causal)
    if RESULT["backends"].get("fa4", {}).get("available"):
        from flash_attn.cute import flash_attn_func as _fa4
        backends["fa4"] = lambda q, k, v, causal: _fa4(q, k, v, causal=causal)

    if not backends:
        print("  no backends available — skipping")
        return
    print(f"  backends: {list(backends)}\n")
    hdr = f"  {'shape':28s} " + " ".join(f"{b:>12s}" for b in backends)
    print(hdr + f" {'fa4/fa3':>10s}")
    print("  " + "-" * (len(hdr) + 10))

    for shape in shapes:
        row: dict[str, Any] = {"shape": shape}
        cells = []
        for name, fn in backends.items():
            try:
                q, k, v = _make_qkv(shape, torch.bfloat16)
                sec = _bench_one(fn, q, k, v, causal=True)
                row[name] = {"median_s": sec, "median_us": round(sec * 1e6, 1)}
                cells.append(f"{sec * 1e6:10.1f}us")
            except BaseException as exc:  # noqa: BLE001
                row[name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                cells.append(f"{'ERR':>12s}")
        # The pre-registered decision ratio.
        try:
            ratio = row["fa3"]["median_s"] / row["fa4"]["median_s"]
            row["fa4_over_fa3_speedup"] = round(ratio, 3)
            ratio_s = f"{ratio:10.3f}"
        except BaseException:  # noqa: BLE001
            ratio_s = f"{'n/a':>10s}"
        print(f"  {shape['name']:28s} " + " ".join(cells) + f" {ratio_s}")
        RESULT["bench"].append(row)

    ratios = [r["fa4_over_fa3_speedup"] for r in RESULT["bench"]
              if "fa4_over_fa3_speedup" in r]
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        RESULT["bench_summary"] = {"fa4_over_fa3_median": med,
                                   "n_cells": len(ratios)}
        print(f"\n  MEDIAN FA4/FA3 SPEEDUP: {med:.3f}x over {len(ratios)} cells")
        print("  Pre-registered decision threshold was 1.15x on cost/benefit "
              "grounds (recorded before measuring; the build proceeded on an "
              "explicit product call regardless).")


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="/tmp/fa4_probe.json",
                    help="where to write the machine-readable result")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="two benchmark shapes instead of six")
    args = ap.parse_args()

    print("Arc FA4 probe — feasibility, envelope, JIT cost, AOT route, speed.")

    probe_env()
    probe_backends()
    probe_head_dims()
    probe_jit_cost()
    probe_aot()
    if not args.skip_bench:
        shapes = FLEET_SHAPES[:2] if args.quick else FLEET_SHAPES
        guarded("benchmark", lambda: probe_bench(shapes))

    RESULT["v4_shape_for_reference"] = V4_SHAPE

    with open(args.json, "w") as fh:
        json.dump(RESULT, fh, indent=2, default=str)

    section("SUMMARY")
    print(f"  JSON: {args.json}")
    print(f"  AOT verdict: "
          f"{RESULT.get('aot', {}).get('verdict', {}).get('conclusion', 'n/a')}")
    accepted = [d for d, e in RESULT["head_dims"].items()
                if isinstance(e, dict) and e.get("ok")]
    print(f"  FA4 head_dims accepted: {accepted or 'none/unavailable'}")
    if "bench_summary" in RESULT:
        print(f"  FA4/FA3 median speedup: "
              f"{RESULT['bench_summary']['fa4_over_fa3_median']}x")
    print(f"  errors recorded: {len(RESULT['errors'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        # argparse --help and our own sys.exit must pass straight through;
        # catching BaseException here would swallow them.
        raise
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        with open("/tmp/fa4_probe.partial.json", "w") as fh:
            json.dump(RESULT, fh, indent=2, default=str)
        print("\nPartial result: /tmp/fa4_probe.partial.json")
        sys.exit(1)
