#!/usr/bin/env python3
"""RUNG 1 STAGE A/B — produce a CuTeDSL AOT object and describe its ABI.

THE GATE
--------
The whole FA4->MLA bet rests on one question:

    Does the AOT export emit a linkable `.o` with a C-callable entry point
    that Rust can reach, with NO Python at runtime?

If the answer is no, hand-written FlashMLA-per-arch is the only road and the
strategy has to be revisited on evidence.

WHAT v1 GOT WRONG, AND THE RULE IT PRODUCED
-------------------------------------------
v1 asserted the entry symbol would be named `__tvm_ffi_<name>` — a name taken
from NVIDIA's documentation — and reported GATE_FAILS on an object that in fact
carried perfectly good C symbols under MLIR's own convention:

    arc_fa4_probe_cutlass_arc_probe_noop
    arc_fa4_probe__mlir_ciface_cutlass_arc_probe_noop
    arc_fa4_probe_args_spec

`_mlir_ciface_` is the standard MLIR wrapper emitted precisely so C/C++/Rust
can call in — it IS the linkable C ABI. That was a false negative on the single
most consequential question in the project.

Same lesson as `CuteDSLRT_Module_Load`, which NVIDIA's AOT docs name and the
shipped wheel does not define: THE ARTIFACT IS THE AUTHORITY, NOT THE
DOCUMENTATION. So this stage now enumerates `nm -g --defined-only`, keeps the
FULL untruncated table, classifies by ELF symbol type, and RANKS candidates by
pattern instead of testing for one predicted name. A future DSL rename degrades
to a worse ranking, never to a false failure.

It also introspects `inspect.signature` before calling the export functions
rather than brute-forcing argument shapes (every v1 `export_to_c` attempt died
on `missing 1 required positional argument: 'file_name'`), and — since no `.h`
was emitted — it extracts the bytes of `<prefix>_args_spec`, which is the
actual calling convention rung 2 needs.

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

    # Read the ACTUAL signatures rather than guessing at them. v1 of this probe
    # brute-forced argument shapes and every export_to_c call died on
    # `missing 1 required positional argument: 'file_name'`; introspecting first
    # turns that into a correct call.
    import inspect

    sigs: dict[str, Any] = {}
    for a in exportish:
        try:
            fn = getattr(compiled, a)
            if callable(fn):
                sigs[a] = str(inspect.signature(fn))
                doc = (inspect.getdoc(fn) or "").strip().splitlines()
                sigs[a + "__doc"] = " ".join(doc[:4])[:400]
        except BaseException as exc:  # noqa: BLE001
            sigs[a] = f"<unavailable: {type(exc).__name__}>"
    M["export_signatures"] = sigs
    print("  signatures:")
    for a in exportish:
        if a in sigs:
            print(f"    {a}{sigs[a]}")
            if sigs.get(a + "__doc"):
                print(f"        {sigs[a + '__doc'][:200]}")

    def by_signature(attr: str):
        """Call `attr` filling its parameters by NAME from what we know."""
        fn = getattr(compiled, attr)
        params = inspect.signature(fn).parameters
        kwargs = {}
        for pname, p in params.items():
            if pname in ("self",) or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            low = pname.lower()
            if "file" in low or "name" in low:
                kwargs[pname] = name
            elif "dir" in low or "path" in low or "folder" in low:
                kwargs[pname] = OUT
            elif p.default is inspect.Parameter.empty:
                # A required parameter we cannot infer — surface it loudly.
                kwargs[pname] = name
        return fn(**kwargs)

    # Signature-driven calls first, then the brute-force shapes as a fallback so
    # a surprising signature still gets covered.
    candidates: list[tuple[str, Any]] = []
    for attr in ("export_to_c", "dump_to_object"):
        if hasattr(compiled, attr):
            candidates.append((f"{attr}(<by signature>)",
                               (lambda a=attr: by_signature(a))))
    candidates += [
        ("export_to_c(file_name=name)",
         lambda: compiled.export_to_c(file_name=name)),
        ("export_to_c(OUT, file_name=name)",
         lambda: compiled.export_to_c(OUT, file_name=name)),
        ("export_to_c(dir, name)",
         lambda: compiled.export_to_c(OUT, name)),
        ("export_to_c(name)",
         lambda: compiled.export_to_c(name)),
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


def extract_data_symbol(obj: str, sym: str) -> dict[str, Any] | None:
    """Dump the bytes a DATA symbol points at.

    With no header emitted, `<prefix>_args_spec` IS the calling convention —
    it is the thing rung 2 has to read. `objdump -t` gives the section, offset
    and size; `objcopy --dump-section` gives the section bytes; we slice.
    """
    try:
        t = subprocess.run(["objdump", "-t", obj], capture_output=True,
                           text=True, timeout=120).stdout
        section = offset = size = None
        for ln in t.splitlines():
            if ln.split()[-1:] != [sym]:
                continue
            parts = ln.split()
            # objdump -t: <value> <flags...> <section> <size> <name>
            try:
                offset = int(parts[0], 16)
                size = int(parts[-2], 16)
                section = parts[-3]
            except (ValueError, IndexError):
                return {"error": f"unparsed objdump line: {ln[:200]}"}
            break
        if section is None:
            return {"error": "symbol not found in objdump -t"}
        if not size:
            # Size 0 usually means a NUL-terminated string; grab a window and
            # cut at the terminator below.
            size = 4096

        raw_path = f"{obj}.{sym}.section.bin"
        rc = subprocess.run(
            ["objcopy", "--dump-section", f"{section}={raw_path}", obj],
            capture_output=True, text=True, timeout=120)
        if rc.returncode != 0 or not os.path.exists(raw_path):
            return {"error": f"objcopy failed: {rc.stderr[:200]}",
                    "section": section, "offset": offset, "size": size}

        with open(raw_path, "rb") as fh:
            blob = fh.read()[offset:offset + size]
        if b"\x00" in blob:
            blob = blob.split(b"\x00", 1)[0] or blob
        printable = sum(1 for b in blob if 32 <= b < 127 or b in (9, 10, 13))
        out: dict[str, Any] = {"section": section, "offset": offset,
                               "nbytes": len(blob)}
        if blob and printable / len(blob) > 0.85:
            out["text"] = blob.decode("utf-8", "replace")
        else:
            out["hex"] = blob[:512].hex()
        return out
    except BaseException as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


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

    # --- symbol table.
    #
    # v1 of this probe looked ONLY for `__tvm_ffi_*` — a name taken from
    # NVIDIA's docs — and reported GATE_FAILS on an object that in fact carried
    # perfectly good C symbols under MLIR's own convention
    # (`<prefix>_<kernel>` and `<prefix>__mlir_ciface_<kernel>`). Same lesson as
    # CuteDSLRT_Module_Load, which the docs name and the wheel does not define:
    # THE ARTIFACT IS THE AUTHORITY, NOT THE DOCUMENTATION.
    #
    # So: enumerate everything, keep the FULL list, classify by ELF symbol type,
    # and RANK candidates by pattern rather than testing for one hardcoded name.
    # A future DSL rename then degrades to a worse ranking, not a false failure.
    for o in M["artifacts"].get("objects") or []:
        p = o["path"]
        try:
            out = subprocess.run(["nm", "-g", "--defined-only", p],
                                 capture_output=True, text=True, timeout=120)
            raw = out.stdout
            # Persist the untruncated table — inspectors upstream have clipped it.
            dump = p + ".symbols.txt"
            with open(dump, "w") as fh:
                fh.write(raw)

            entries = []
            for ln in raw.splitlines():
                parts = ln.split()
                if len(parts) >= 2:
                    entries.append({"type": parts[-2], "name": parts[-1]})
            text = [e["name"] for e in entries if e["type"].upper() == "T"]
            data = [e for e in entries if e["type"].upper() != "T"]

            def score(s: str) -> int:
                # Normalise away the platform's leading-underscore convention
                # (Mach-O prepends one, ELF does not) so the same rule ranks
                # correctly on both.
                bare = s.lstrip("_")
                lead = len(s) - len(bare)
                v = 0
                # The MLIR C-interface wrapper exists precisely so C/C++/Rust
                # can call in. It is the intended external entry point.
                if "_mlir_ciface_" in s:
                    v += 100
                if "tvm_ffi" in s.lower():
                    v += 60
                # `_mlir_`-prefixed duplicates are the internally mangled forms
                # of the same function; prefer the clean exported spelling.
                # Checked on the de-underscored name so `_mlir_x` and `__mlir_x`
                # are both caught.
                if bare.startswith("mlir_"):
                    v -= 40
                # Fewer leading underscores = the more public spelling. This is
                # also the tie-break that keeps the mangled twin from winning.
                v -= 5 * lead
                return v

            ranked = sorted(text, key=lambda s: (-score(s), s))
            M["symbols"][p] = {
                "full_table_path": dump,
                "all": [e["name"] for e in entries],
                "by_type": entries,
                "text": text,
                "ranked_candidates": [{"name": s, "score": score(s)}
                                      for s in ranked],
                "tvm_ffi": [s for s in text if "tvm_ffi" in s.lower()],
                "mlir_ciface": [s for s in text if "_mlir_ciface_" in s],
            }
            print(f"\n  {p}: {len(entries)} defined global symbols "
                  f"({len(text)} TEXT). Full table -> {dump}")
            for e in entries:
                print(f"      [{e['type']}] {e['name']}")
            print("    ranked callable candidates:")
            for s in ranked[:6]:
                print(f"      score {score(s):4d}  {s}")
            if out.returncode != 0:
                print(f"      (nm stderr: {out.stderr[:200]})")

            # --- the non-TEXT symbols are the ABI contract when no .h is
            #     emitted: `*_args_spec` and `*_function_name` describe how to
            #     call the kernel. Extract their bytes.
            for e in data:
                nm_ = e["name"]
                if not any(t in nm_ for t in ("args_spec", "function_name",
                                              "arg_spec", "signature")):
                    continue
                blob = extract_data_symbol(p, nm_)
                if blob is not None:
                    M["symbols"][p].setdefault("data_blobs", {})[nm_] = blob
                    print(f"\n    --- {nm_} ({blob.get('nbytes')} bytes,"
                          f" section {blob.get('section')}) ---")
                    if blob.get("text"):
                        print("      " + blob["text"][:1200].replace(
                            "\n", "\n      "))
                    elif blob.get("hex"):
                        print("      hex: " + blob["hex"][:300])
                    elif blob.get("error"):
                        print(f"      (extraction failed: {blob['error']})")

            # Belt-and-braces: `strings` on the object recovers args_spec /
            # function_name content even when the objdump symbol-table parse
            # fails (its output format is not portable). The contract is what
            # matters; how we recover it does not.
            try:
                st = subprocess.run(["strings", "-a", "-n", "6", p],
                                    capture_output=True, text=True, timeout=120)
                found = [ln for ln in st.stdout.splitlines() if ln.strip()]
                M["symbols"][p]["strings"] = found[:200]
                print("\n    strings(1) in the object (args_spec / "
                      "function_name content lives here):")
                for ln in found[:40]:
                    print(f"      {ln[:200]}")
            except BaseException as exc:  # noqa: BLE001
                err(f"strings {p}", exc)
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

    # Rank across every object. The gate is "is there a callable C symbol",
    # NOT "is there a symbol with the name the docs predicted".
    ranked: list[dict[str, Any]] = []
    for v_ in M["symbols"].values():
        ranked += v_.get("ranked_candidates") or []
    ranked.sort(key=lambda c: -c["score"])
    gate_symbol = ranked[0]["name"] if ranked else None

    # Which call actually produced the object? That decides the rung-2 plan.
    produced_by = None
    for a in M["attempts"]:
        if a.get("ok") and (a.get("new_files") or a.get("written_from_bytes")):
            produced_by = a["call"]
            break

    if objs and gate_symbol:
        v = "OBJECT_WITH_CALLABLE_C_SYMBOLS__proceed_to_link_test"
    elif objs:
        v = "OBJECT_BUT_NO_TEXT_SYMBOL__inspect_symbol_list"
    else:
        v = "NO_OBJECT_EMITTED__gate_fails_here"

    M["verdict"] = {
        "code": v,
        "n_objects": len(objs),
        "gate_symbol": gate_symbol,
        "candidates": ranked[:10],
        "produced_by": produced_by,
    }
    print(f"  {v}")
    print(f"  objects: {len(objs)}   produced by: {produced_by or 'unknown'}")
    print(f"  gate symbol: {gate_symbol or 'none'}")
    for c in ranked[:5]:
        print(f"      score {c['score']:4d}  {c['name']}")
    if v.startswith("NO_OBJECT"):
        print("\n  => STOP AND REPORT. No linkable object on this DSL version.")
        print("     FlashMLA-per-arch becomes the only road and the substrate")
        print("     decision needs revisiting on this evidence.")
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
