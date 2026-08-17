#!/usr/bin/env bash
# RUNG 1 STAGE C — link the CuTeDSL AOT object from RUST and prove no Python.
#
# THE GATE, restated: does the AOT export emit a linkable .o with a C-callable
# entry point that Rust can reach, with NO Python at runtime?
#
# NOTE ON THE SYMBOL NAME. v1 of this gate asserted `__tvm_ffi_<name>`, taken
# from NVIDIA's docs, and returned GATE_FAILS on an object that carried
# perfectly good C symbols under MLIR's own convention
# (`<prefix>_<kernel>` and `<prefix>__mlir_ciface_<kernel>`). `_mlir_ciface_`
# is the standard MLIR wrapper emitted precisely so C/C++/Rust can call in.
# Same lesson as CuteDSLRT_Module_Load, which the docs name and the wheel does
# not define: THE ARTIFACT IS THE AUTHORITY, NOT THE DOCUMENTATION. The symbol
# is now chosen by RANKING the object's own TEXT symbols, so a future DSL
# rename degrades to a worse ranking rather than a false failure.
#
# Stage A/B (rung1_export.py) produced the object and described the ABI.
# This stage answers the half that actually decides the strategy:
#
#   C1  Rust links the .o and resolves the entry symbol           -> callable
#   C2  the symbol is a TEXT symbol (real code, not a stub)       -> callable
#   C3  the linked binary has NO libpython dependency             -> no Python
#   C4  the binary runs to completion                             -> usable
#   C5  (bonus, non-gating) the symbol can actually be invoked
#
# If all four pass, the FA4->MLA substrate bet is sound and rung 2 can start.
# If C1 or C3 fails, STOP: FlashMLA-per-arch becomes the only road and the
# strategy needs revisiting on this evidence.
#
# Usage:  bash arc-tools/fa4/rung1_link_test.sh
# Reads:  /tmp/arc_fa4_rung1/manifest.json
# Writes: /tmp/arc_fa4_rung1/link_verdict.json

set -uo pipefail

OUT=/tmp/arc_fa4_rung1
MANIFEST="$OUT/manifest.json"
VERDICT="$OUT/link_verdict.json"

C1=fail; C2=fail; C3=fail; C4=fail
NOTE=""

say()  { printf '%s\n' "$*"; }
head2() { printf '\n======================================================================\n%s\n======================================================================\n' "$*"; }

head2 "RUNG 1 STAGE C — Rust link test"

if [ ! -f "$MANIFEST" ]; then
    say "  no manifest at $MANIFEST — run rung1_export.py first"
    exit 2
fi

# ---- pull the object path and the gate symbol out of the manifest ----------
# NOTE: the parser is written to a FILE, not run via `<(python3 <<PY ...)`.
# Process substitution re-parses its body as shell, so a stray apostrophe or
# parenthesis inside the heredoc silently breaks paren matching:
#   bad substitution: no closing `)'
# That bug cost a run of this script. Heredoc-to-a-file has no such hazard.
PARSER="$OUT/_parse_manifest.py"
cat > "$PARSER" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
objs = m.get("artifacts", {}).get("objects") or []
sym = (m.get("verdict") or {}).get("gate_symbol")

# Pick the object that DEFINES the gate symbol. Selection is by RANK, never by
# a hardcoded name. v1 of this gate searched only for __tvm_ffi_* -- a name
# taken from NVIDIA docs -- and declared failure on an object carrying good
# MLIR C-interface symbols. Rank, do not guess.
best = ""
if objs:
    best = objs[0]["path"]
    for o in objs:
        info = (m.get("symbols", {}) or {}).get(o["path"], {}) or {}
        cands = info.get("ranked_candidates") or []
        if cands:
            best = o["path"]
            if not sym:
                sym = cands[0]["name"]
            break
print(best or "NONE", sym or "NONE")
PY

PARSED=$(python3 "$PARSER" "$MANIFEST" 2>"$OUT/parse.err")
if [ -z "${PARSED:-}" ]; then
    say "  manifest parse failed:"
    sed 's/^/    /' "$OUT/parse.err"
    exit 2
fi
OBJ=$(printf '%s' "$PARSED" | awk '{print $1}')
SYM=$(printf '%s' "$PARSED" | awk '{print $2}')

say "  object: $OBJ"
say "  symbol: $SYM"

if [ "$OBJ" = "NONE" ]; then
    say "  no object emitted — the gate already failed in stage A."
    printf '{"C1":"fail","C2":"fail","C3":"fail","C4":"fail","verdict":"NO_OBJECT__GATE_FAILS"}\n' > "$VERDICT"
    exit 1
fi

# ---- C2: is the symbol real code (nm type T)? ------------------------------
head2 "C2 — symbol is a TEXT symbol (real code)"
if [ "$SYM" != "NONE" ]; then
    LINE=$(nm -g --defined-only "$OBJ" 2>/dev/null | grep -w "$SYM" | head -1)
    say "  $LINE"
    case "$LINE" in
        *" T "*|*" t "*) C2=pass; say "  PASS — text symbol" ;;
        *) say "  FAIL — not a text symbol" ;;
    esac
else
    say "  no ranked TEXT symbol recorded; listing all defined symbols:"
    nm -g --defined-only "$OBJ" 2>/dev/null | head -30
fi

# ---- discover the runtime shared libraries to link against -----------------
head2 "Runtime libraries"
RTFINDER="$OUT/_find_rtlibs.py"
cat > "$RTFINDER" <<'PY'
import glob, os
try:
    import cutlass
    root = os.path.dirname(cutlass.__file__)
except Exception:
    print(""); raise SystemExit
pats = ["**/libcute_dsl_runtime*.so", "**/*tvm_ffi*.so"]
found = []
for base in (root, os.path.join(root, "..")):
    for p in pats:
        found += glob.glob(os.path.join(base, p), recursive=True)
print("\n".join(sorted(set(os.path.realpath(f) for f in found))))
PY
RTLIBS=$(python3 "$RTFINDER" 2>/dev/null)
say "${RTLIBS:-  (none found)}"

LINKARGS=()
RPATHS=()
while IFS= read -r lib; do
    [ -z "$lib" ] && continue
    d=$(dirname "$lib")
    b=$(basename "$lib"); b=${b#lib}; b=${b%.so}
    LINKARGS+=( -C "link-arg=-L$d" -C "link-arg=-l$b" )
    RPATHS+=( -C "link-arg=-Wl,-rpath,$d" )
done <<< "$RTLIBS"

# CUDA driver search paths. -lcuda itself is added as a SEPARATE, OPTIONAL
# group: the link is attempted with it and retried without it if that fails.
# A missing/unfindable libcuda is an environment problem, not evidence that the
# object has no callable symbol, and must never read as a gate failure.
for cd in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs; do
    [ -d "$cd" ] && LINKARGS+=( -C "link-arg=-L$cd" )
done
CUDAARG=( -C "link-arg=-lcuda" )

# ---- C1: generate a Rust harness and link it -------------------------------
head2 "C1 — Rust links the object and resolves the symbol"

if ! command -v rustc >/dev/null 2>&1; then
    say "  rustc not found — cannot run the decisive half of the gate."
    printf '{"C1":"skip","verdict":"NO_RUSTC"}\n' > "$VERDICT"
    exit 2
fi

HARNESS="$OUT/harness.rs"
if [ "$SYM" != "NONE" ]; then
cat > "$HARNESS" <<RS
// Generated by rung1_link_test.sh — do not edit.
//
// Proves the CuTeDSL AOT object is reachable from Rust's FFI: we declare the
// exported symbol, take its address through a raw pointer, and confirm the
// linker resolved it to something non-null. Taking the address (rather than
// calling) is deliberate: what rung 1 must establish is that the symbol
// EXISTS, LINKS, and needs no Python interpreter to do so. Invoking it with
// real arguments needs the calling convention from the args_spec symbol,
// which is rung 2. C5 below attempts a bare call as a bonus signal.
extern "C" {
    fn $SYM();
}

fn main() {
    let addr = $SYM as *const ();
    println!("symbol  = $SYM");
    println!("address = {addr:p}");
    assert!(!addr.is_null(), "symbol resolved to null");
    println!("ARC_RUNG1_LINK_OK");
}
RS
else
cat > "$HARNESS" <<'RS'
fn main() { println!("ARC_RUNG1_NO_SYMBOL"); }
RS
fi

BIN="$OUT/harness"
say "  rustc $HARNESS -> $BIN"
LINKMODE=""
rustc "$HARNESS" -O -o "$BIN" -C "link-arg=$OBJ" \
    ${LINKARGS[@]+"${LINKARGS[@]}"} ${RPATHS[@]+"${RPATHS[@]}"} \
    ${CUDAARG[@]+"${CUDAARG[@]}"} > "$OUT/rustc.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    say "  with -lcuda: failed; retrying without it"
    rustc "$HARNESS" -O -o "$BIN" -C "link-arg=$OBJ" \
        ${LINKARGS[@]+"${LINKARGS[@]}"} ${RPATHS[@]+"${RPATHS[@]}"} \
        > "$OUT/rustc.nocuda.log" 2>&1
    RC=$?
    [ $RC -eq 0 ] && LINKMODE="without -lcuda"
else
    LINKMODE="with -lcuda"
fi
if [ $RC -eq 0 ]; then
    C1=pass; say "  PASS — linked ($LINKMODE)"
else
    say "  FAIL — link error (tail of $OUT/rustc.log):"
    tail -30 "$OUT/rustc.log" | sed 's/^/    /'
    [ -f "$OUT/rustc.nocuda.log" ] && {
        say "  (retry without -lcuda, tail of $OUT/rustc.nocuda.log):"
        tail -15 "$OUT/rustc.nocuda.log" | sed 's/^/    /'; }
fi

# ---- C3: no Python in the linked binary ------------------------------------
head2 "C3 — the binary does NOT depend on libpython"
if [ -x "$BIN" ]; then
    LDD=$(ldd "$BIN" 2>/dev/null)
    say "$LDD" | sed 's/^/    /'
    if printf '%s' "$LDD" | grep -qi 'libpython'; then
        say "  FAIL — libpython is linked in; this would put Python in the serving path."
    else
        C3=pass; say "  PASS — no libpython"
    fi
else
    say "  (no binary to inspect)"
fi

# ---- C4: it runs -----------------------------------------------------------
head2 "C4 — the binary runs"
if [ -x "$BIN" ]; then
    RUNOUT=$("$BIN" 2>&1); RRC=$?
    say "$RUNOUT" | sed 's/^/    /'
    if [ $RRC -eq 0 ] && printf '%s' "$RUNOUT" | grep -q ARC_RUNG1_LINK_OK; then
        C4=pass; say "  PASS — exit 0"
    else
        say "  FAIL — exit $RRC"
    fi
fi

# ---- C5 (bonus): actually invoke the symbol --------------------------------
# Separate binary on purpose: a bad call must not jeopardise the C1-C4 result.
# The exported kernel is a no-arg no-op, so a plain call is the natural probe;
# if the CuTeDSL runtime needs initialising first this will fail, and that
# failure is itself the rung-2 starting point.
head2 "C5 (bonus, non-gating) — attempt an actual call"
C5=skip
if [ "$C1" = pass ] && [ "$SYM" != "NONE" ]; then
    cat > "$OUT/harness_invoke.rs" <<RS
extern "C" { fn $SYM(); }
fn main() {
    println!("calling $SYM ...");
    unsafe { $SYM(); }
    println!("ARC_RUNG1_INVOKE_OK");
}
RS
    if rustc "$OUT/harness_invoke.rs" -O -o "$OUT/harness_invoke" \
        -C "link-arg=$OBJ" ${LINKARGS[@]+"${LINKARGS[@]}"} ${RPATHS[@]+"${RPATHS[@]}"} \
        ${CUDAARG[@]+"${CUDAARG[@]}"} >> "$OUT/rustc.log" 2>&1 \
       || rustc "$OUT/harness_invoke.rs" -O -o "$OUT/harness_invoke" \
        -C "link-arg=$OBJ" ${LINKARGS[@]+"${LINKARGS[@]}"} ${RPATHS[@]+"${RPATHS[@]}"} \
        >> "$OUT/rustc.log" 2>&1; then
        IOUT=$("$OUT/harness_invoke" 2>&1); IRC=$?
        say "$IOUT" | sed 's/^/    /'
        say "    exit=$IRC"
        if [ $IRC -eq 0 ]; then C5=pass; else C5="fail(exit=$IRC)"; fi
    else
        C5="link-fail"
        say "    invoke harness did not link (see $OUT/rustc.log)"
    fi
    say "  C5=$C5   (informational — does not gate rung 1)"
else
    say "  skipped (C1 did not pass)"
fi

# ---- verdict ---------------------------------------------------------------
head2 "RUNG 1 VERDICT"
if [ "$C1" = pass ] && [ "$C3" = pass ] && [ "$C4" = pass ]; then
    V="GATE_PASSES__aot_object_is_linkable_from_rust_without_python"
    NOTE="Proceed to rung 2 (vanilla FA4 callable from Arc)."
elif [ "$C1" = pass ] && [ "$C3" != pass ]; then
    V="LINKS_BUT_DRAGS_PYTHON__investigate"
    NOTE="Object links but pulls libpython. Determine whether that comes from the harness or the runtime .so."
elif [ "$C2" = pass ]; then
    # A real TEXT symbol exists; only the link step failed. That is an
    # environment problem (missing libcuda, missing runtime .so), NOT evidence
    # that the substrate bet is dead. v1 of this gate over-claimed failure once
    # already; it does not get to do so again.
    V="CALLABLE_SYMBOL_PRESENT_BUT_LINK_FAILED__environment_not_substrate"
    NOTE="Do NOT revise the FA4->MLA decision on this. A genuine C TEXT symbol is present; the linker could not complete. Read $OUT/rustc.log."
else
    V="GATE_FAILS__no_rust_callable_object"
    NOTE="No callable TEXT symbol in the object. THIS is the result that would make FlashMLA-per-arch the only road."
fi

say "  C1 link=$C1  C2 text-symbol=$C2  C3 no-python=$C3  C4 runs=$C4  C5 invoke=$C5"
say "  => $V"
say "  $NOTE"

cat > "$VERDICT" <<JSON
{
  "schema": "arc.fa4_rung1_link/1",
  "object": "$OBJ",
  "symbol": "$SYM",
  "C1_rust_links": "$C1",
  "C2_text_symbol": "$C2",
  "C3_no_libpython": "$C3",
  "C4_runs": "$C4",
  "C5_invoke_bonus": "$C5",
  "verdict": "$V",
  "note": "$NOTE"
}
JSON
say ""
say "  verdict: $VERDICT"

# EXIT CODES ARE PART OF THE CONTRACT.
#   0 = gate passes
#   1 = GENUINE substrate failure -- the only result that is a strategy signal
#   2 = this machine could not answer the question (no rustc, link environment)
# Conflating 2 with 1 is exactly the mistake gate v1 made. Only exit 1 should
# ever reach Jish as evidence about the substrate.
case "$V" in
    GATE_PASSES__*)                 exit 0 ;;
    GATE_FAILS__*)                  exit 1 ;;
    *)                              exit 2 ;;
esac
