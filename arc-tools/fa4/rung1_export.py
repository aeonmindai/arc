#!/usr/bin/env python3
"""RUNG 1 STAGE A/B — produce a CuTeDSL AOT object and describe its ABI.

THE GATE
--------
The whole FA4->MLA bet rests on one question:

    Does `export_to_c()` emit a linkable `.o` exposing `__tvm_ffi_<name>`
    symbols that Rust can call, with NO Python at runtime?

If the answer is no, hand-written FlashMLA-per-arch is the only road and the
strategy has to be revisited on evidence. So this stage assumes NOTHING about
the API surface: it tries every plausible spelling, records exactly what each
one did, and dumps the generated header verbatim — the header IS the ABI
contract, and reading it beats guessing at it.

Output: /tmp/arc_fa4_rung1/manifest.json, consumed by rung1_link_test.sh,
which does stage C (link from Rust, run, prove no libpython).

Run:
    python3 arc-tools/fa4/rung1_export.py
    bash    arc-tools/fa4/rung1_link_test.sh

Prereqs on a fresh runcrate image (pip 22.0.2 has no numpy and breaks
flash-attn metadata generation):
    python3 -m pip install -q --upgrade pip setuptools wheel
    python3 -m pip install -q numpy
    python3 -m pip install -q flash-attn-4      # pulls nvidia-cutlass-dsl
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import traceback
from typing import Any

OUT = "/tmp/arc_fa4_rung1"
MANIFEST = os.path.join(OUT, "manifest.json")

M: dict[str, Any] = {
    "schema": "arc.fa4_rung1/1",
    "stage": "A/B (export + ABI description)",
    "attempts": [],
    "artifacts": {},
    "symbols": {},
    "header": None,
    "runtime_libs": [],
    "verdict": None,
    "errors": [],
}


def err(where: str, exc: BaseException) -> None:
    M["errors"].append({"where": where, "type": type(exc).__name__,
                        "msg": str(exc)[:800],
                        "tb": traceback.format_exc()[-1500:]})
    print(f"  !! {where}: {type(exc).__name__}: {str(exc)[:300]}")


def sect(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# A trivial CuTeDSL kernel. Deliberately minimal: rung 1 tests the TOOLCHAIN,
# not the kernel. A no-op that still goes through the full compile pipeline is
# the most robust thing to export, and any DSL-syntax drift shows up here
# loudly instead of being confused with an export failure.
# ---------------------------------------------------------------------------


def build_compiled():
    import cutlass  # noqa: F401
    import cutlass.cute as cute

    @cute.jit
    def arc_probe_noop():
        # cute.printf is the DSL's device-side print; using it keeps the body
        # from being optimised away entirely, so the compile produces real code.
        cute.printf("arc")

    compiled = cute.compile(arc_probe_noop)
    print(f"  compiled object: {type(compiled).__name__}")
    return compiled


# ---------------------------------------------------------------------------
# Stage A — try every plausible export spelling, record what each one did.
# ---------------------------------------------------------------------------


def try_export(compiled) -> None:
    sect("STAGE A — export to a linkable object")

    os.makedirs(OUT, exist_ok=True)
    name = "arc_fa4_probe"

    exportish = [a for a in dir(compiled)
                 if any(t in a.lower()
                        for t in ("export", "object", "dump", "aot", "cubin",
                                  "ptx", "save", "serial"))]
    M["exportish_attrs"] = exportish
    print(f"  export-shaped attributes on the compiled object:\n    {exportish}")

    # Each candidate: (label, callable). Signatures differ across DSL minors,
    # so every one is tried in several shapes and the outcome recorded.
    candidates = [
        ("export_to_c(dir, name)",
         lambda: compiled.export_to_c(OUT, name)),
        ("export_to_c(name)",
         lambda: compiled.export_to_c(name)),
        ("export_to_c(dir)",
         lambda: compiled.export_to_c(OUT)),
        ("export_to_c()",
         lambda: compiled.export_to_c()),
        ("dump_to_object(name)",
         lambda: compiled.dump_to_object(name)),
        ("dump_to_object()",
         lambda: compiled.dump_to_object()),
    ]

    for label, fn in candidates:
        attr = label.split("(")[0]
        if not hasattr(compiled, attr):
            M["attempts"].append({"call": label, "ok": False,
                                  "error": "attribute not present"})
            continue
        before = set(glob.glob(os.path.join(OUT, "**", "*"), recursive=True))
        try:
            ret = fn()
            after = set(glob.glob(os.path.join(OUT, "**", "*"), recursive=True))
            new = sorted(after - before)
            rec = {
                "call": label,
                "ok": True,
                "return_type": type(ret).__name__,
                "return_nbytes": len(ret) if isinstance(ret, (bytes, bytearray))
                else None,
                "return_repr": str(ret)[:300],
                "new_files": [{"path": p, "nbytes": os.path.getsize(p)}
                              for p in new if os.path.isfile(p)],
            }
            M["attempts"].append(rec)
            nb = rec["return_nbytes"]
            nb_s = " ({} bytes)".format(nb) if nb else ""
            print("  OK   {} -> {}{}, {} new file(s)".format(
                label, rec["return_type"], nb_s, len(rec["new_files"])))
            for f in rec["new_files"]:
                print(f"         {f['path']}  {f['nbytes']} bytes")
            # If a call produced bytes but no file, persist it ourselves so the
            # link stage has something on disk to work with.
            if isinstance(ret, (bytes, bytearray)) and not rec["new_files"]:
                p = os.path.join(OUT, f"{name}.o")
                with open(p, "wb") as fh:
                    fh.write(ret)
                rec["written_from_bytes"] = {"path": p, "nbytes": len(ret)}
                print(f"         (wrote returned bytes to {p})")
        except BaseException as exc:  # noqa: BLE001
            M["attempts"].append({"call": label, "ok": False,
                                  "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
            print(f"  fail {label}: {type(exc).__name__}: {str(exc)[:180]}")

    # Collect whatever landed.
    objs = sorted(glob.glob(os.path.join(OUT, "**", "*.o"), recursive=True))
    hdrs = sorted(glob.glob(os.path.join(OUT, "**", "*.h"), recursive=True))
    others = [p for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True)
              if os.path.isfile(p) and not p.endswith((".o", ".h", ".json"))]
    M["artifacts"] = {
        "objects": [{"path": p, "nbytes": os.path.getsize(p)} for p in objs],
        "headers": [{"path": p, "nbytes": os.path.getsize(p)} for p in hdrs],
        "other": [{"path": p, "nbytes": os.path.getsize(p)} for p in others],
    }
    print(f"\n  objects: {objs or 'NONE'}")
    print(f"  headers: {hdrs or 'NONE'}")


# ---------------------------------------------------------------------------
# Stage B — describe the ABI: header text + symbol table + runtime libs.
# ---------------------------------------------------------------------------


def describe_abi() -> None:
    sect("STAGE B — ABI description (header is the contract)")

    # --- header verbatim. This is what tells us the exact call signature.
    for h in M["artifacts"].get("headers") or []:
        try:
            txt = open(h["path"]).read()
            M["header"] = {"path": h["path"], "text": txt}
            print(f"  --- {h['path']} ---")
            print("\n".join("    " + ln for ln in txt.splitlines()[:120]))
        except BaseException as exc:  # noqa: BLE001
            err(f"read header {h['path']}", exc)
        break
    if not M["artifacts"].get("headers"):
        print("  no header emitted")

    # --- symbol table of the object. The gate symbol is __tvm_ffi_<name>.
    for o in M["artifacts"].get("objects") or []:
        p = o["path"]
        try:
            out = subprocess.run(["nm", "-g", "--defined-only", p],
                                 capture_output=True, text=True, timeout=120)
            lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
            syms = [ln.split()[-1] for ln in lines]
            tvm = [s for s in syms if "tvm_ffi" in s.lower()]
            M["symbols"][p] = {"all": syms[:200], "tvm_ffi": tvm}
            print(f"\n  {p}: {len(syms)} defined global symbols, "
                  f"{len(tvm)} tvm_ffi")
            for s in tvm[:20]:
                print(f"      TVM_FFI  {s}")
            for s in [s for s in syms if s not in tvm][:20]:
                print(f"      sym      {s}")
            if out.returncode != 0:
                print(f"      (nm stderr: {out.stderr[:200]})")
        except BaseException as exc:  # noqa: BLE001
            err(f"nm {p}", exc)

    # --- the CuTeDSL runtime .so we will have to link against.
    #     NVIDIA's AOT docs name CuteDSLRT_Module_Load, which does NOT exist in
    #     the shipped wheel; the real entry point is
    #     CuteDSLRT_Module_Create_From_Bytes. Confirm against the wheel here.
    try:
        import cutlass
        root = os.path.dirname(cutlass.__file__)
        libs = sorted(set(
            glob.glob(os.path.join(root, "**", "libcute_dsl_runtime*.so"),
                      recursive=True)
            + glob.glob(os.path.join(root, "..", "**",
                                     "libcute_dsl_runtime*.so"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*tvm_ffi*.so"),
                        recursive=True)
        ))
        for lib in libs[:4]:
            entry: dict[str, Any] = {"path": lib,
                                     "nbytes": os.path.getsize(lib)}
            try:
                out = subprocess.run(["nm", "-D", "--defined-only", lib],
                                     capture_output=True, text=True, timeout=120)
                syms = [ln.split()[-1] for ln in out.stdout.splitlines()
                        if ln.strip()]
                entry["CuteDSLRT"] = sorted(
                    {s for s in syms if "CuteDSLRT" in s})[:60]
                entry["has_Module_Load"] = any(
                    s == "CuteDSLRT_Module_Load" for s in syms)
                entry["has_Create_From_Bytes"] = any(
                    "Create_From_Bytes" in s for s in syms)
            except BaseException as exc:  # noqa: BLE001
                entry["nm_error"] = str(exc)[:200]
            M["runtime_libs"].append(entry)
            print(f"\n  runtime lib: {lib}")
            print(f"    CuteDSLRT_Module_Load present:        "
                  f"{entry.get('has_Module_Load')}  "
                  f"(NVIDIA's docs claim this one)")
            print(f"    *_Create_From_Bytes present:          "
                  f"{entry.get('has_Create_From_Bytes')}")
            for s in (entry.get("CuteDSLRT") or [])[:25]:
                print(f"      {s}")
        if not libs:
            print("  no libcute_dsl_runtime*.so found under the cutlass package")
    except BaseException as exc:  # noqa: BLE001
        err("runtime lib discovery", exc)


# ---------------------------------------------------------------------------


def verdict() -> None:
    sect("RUNG 1 STAGE A/B VERDICT")
    objs = M["artifacts"].get("objects") or []
    tvm_syms = [s for v in M["symbols"].values() for s in v.get("tvm_ffi", [])]

    if objs and tvm_syms:
        v = "OBJECT_AND_TVM_FFI_SYMBOLS_PRESENT__proceed_to_link_test"
    elif objs:
        v = "OBJECT_BUT_NO_TVM_FFI_SYMBOL__inspect_symbol_list"
    else:
        v = "NO_OBJECT_EMITTED__gate_fails_here"

    M["verdict"] = {
        "code": v,
        "n_objects": len(objs),
        "tvm_ffi_symbols": tvm_syms,
        "gate_symbol": tvm_syms[0] if tvm_syms else None,
    }
    print(f"  {v}")
    print(f"  objects: {len(objs)}   tvm_ffi symbols: {tvm_syms or 'none'}")
    if v.startswith("NO_OBJECT"):
        print("\n  => STOP AND REPORT. export_to_c() did not produce a linkable")
        print("     object on this DSL version. FlashMLA-per-arch becomes the")
        print("     only road and the strategy needs revisiting on this evidence.")
    else:
        print("\n  => Next: bash arc-tools/fa4/rung1_link_test.sh")


def main() -> int:
    print("Arc FA4 RUNG 1 — export_to_c() gate, stage A/B")
    os.makedirs(OUT, exist_ok=True)

    # Record versions with every result; the DSL surface moves between minors.
    try:
        import cutlass
        M["dsl_version"] = getattr(cutlass, "__version__", None)
        print(f"  nvidia-cutlass-dsl: {M['dsl_version']}")
    except BaseException as exc:  # noqa: BLE001
        err("import cutlass", exc)
        M["verdict"] = {"code": "CUTLASS_DSL_NOT_INSTALLED"}
        with open(MANIFEST, "w") as fh:
            json.dump(M, fh, indent=2, default=str)
        print(f"\n  cutlass DSL not installed. Manifest: {MANIFEST}")
        return 2

    M["nvcc"] = shutil.which("nvcc")
    M["rustc"] = shutil.which("rustc")

    compiled = None
    try:
        compiled = build_compiled()
    except BaseException as exc:  # noqa: BLE001
        err("build/compile trivial cute.jit kernel", exc)

    if compiled is not None:
        try:
            try_export(compiled)
        except BaseException as exc:  # noqa: BLE001
            err("stage A", exc)
        try:
            describe_abi()
        except BaseException as exc:  # noqa: BLE001
            err("stage B", exc)

    verdict()
    with open(MANIFEST, "w") as fh:
        json.dump(M, fh, indent=2, default=str)
    print(f"\n  manifest: {MANIFEST}")
    print(f"  errors recorded: {len(M['errors'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        os.makedirs(OUT, exist_ok=True)
        with open(MANIFEST, "w") as fh:
            json.dump(M, fh, indent=2, default=str)
        print(f"\nPartial manifest: {MANIFEST}")
        sys.exit(1)
