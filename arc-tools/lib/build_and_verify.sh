# shellcheck shell=bash
# arc-tools/lib/build_and_verify.sh — build a binary and PROVE you are about to
# measure it. Source this; do not execute it.
#
# ---------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------
# On 2026-08-17 a capture-probe run produced a complete, plausible set of
# numbers from a binary that its own build step had not produced. The idiom,
# copied from `measure_v4_prefill_curve.sh:241-244`, was:
#
#     cargo build --release --features "$FEATURES" -p arc-cli
#     BIN="$REPO/target/release/mistralrs"
#     [ -x "$BIN" ] || fail
#
# Those are two different binaries. `arc-cli`'s bin is named **`arc`**
# (`arc-cli/Cargo.toml`), while **`mistralrs`** is produced by `mistralrs-cli`,
# a package `arc-cli` does not depend on. So the build succeeded, the existence
# check passed against an hour-old binary left by another chain, and the run
# measured code that was never compiled. The server even logged
# `git revision: ab42c4508` while the checkout was `7fbdfcfdb` — the evidence
# was in the log and nothing looked at it.
#
# This failure mode is worse than the ordinary kind because it is SILENT and
# BIASED TOWARD "NO EFFECT": a stale binary yields a clean, flat, internally
# consistent result that reads exactly like an honest negative. You cannot
# distinguish "the change did nothing" from "the change was never in the
# binary" after the fact. The only defence is to refuse before measuring.
#
# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------
# 1. Build the package that actually produces the binary you will run.
# 2. Take the binary's path from CARGO'S OWN artifact stream, never from a
#    hand-written `target/release/<guess>`.
# 3. Refuse to proceed unless the binary contains a marker string that only the
#    code under test carries.  (`arc_build_and_verify`)
# 4. Once the process is running, assert IT reports the commit you built.
#    (`arc_assert_running_revision`)
#
# Existence is not freshness. `[ -x "$BIN" ]` answers a question nobody asked.
#
# ---------------------------------------------------------------------------
# WHY 3 AND 4 ARE DIFFERENT GUARANTEES — the subtle part
# ---------------------------------------------------------------------------
# It is tempting to treat the marker check (3) as sufficient. It is not.
#
#   * A stale binary MISSING the new markers fails (3) and the run is correctly
#     voided.
#   * A stale binary that is a DIFFERENT RECENT BUILD already containing those
#     markers PASSES (3) — while every number came from the wrong commit.
#
# So: **engagement proves the feature is present; only the revision check
# proves it is the build you meant.** Conflating them is how a run can be
# "verified" and still measure someone else's commit.
#
# There is a second reason (4) matters more than it looks. Scripts often lean
# on an invariant instead — "I built into a fresh target dir, so nothing stale
# can exist." That is an invariant, not an assertion: it holds until the next
# person reuses a target dir to save 50 minutes of build time, and then it
# fails silently. (4) validates the RUNNING PROCESS rather than the artifact,
# so it survives that shortcut.
#
# `mistralrs-core/build.rs:115-131` bakes `git rev-parse HEAD` of the build
# directory into `MISTRALRS_GIT_REVISION`, and `mistralrs-core/src/lib.rs:657`
# logs it as `git revision: <sha>`. Note build.rs falls back to the literal
# string `unknown` when git fails — treated here as a FAILURE, never a pass.
#
# Worth knowing: build.rs's `rerun-if-changed=.git/HEAD` does not resolve in a
# git WORKTREE (there `.git` is a file, not a directory), so the baked revision
# can go stale across a worktree checkout without cargo noticing. That is
# precisely the case (4) catches and (3) cannot.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#     source "$REPO/arc-tools/lib/build_and_verify.sh"
#     arc_build_and_verify \
#         --package mistralrs-cli \
#         --bin      mistralrs \
#         --features "cuda flash-attn" \
#         --marker   "ARCGRAPH STATUS" \        # optional but strongly advised
#         --log      "$LOGDIR/build.log"
#     # on success: $ARC_VERIFIED_BIN is the cargo-reported path
#
# Returns 0 on success, 1 on any failure, with a diagnosis on stderr. Callers
# decide how to report (most Arc scripts call their own `env_fail`).

arc_build_and_verify() {
    local package="" binname="" features="" marker="" log="/dev/null" repo="$PWD"
    while [ $# -gt 0 ]; do
        case "$1" in
            --package)  package="$2"; shift 2 ;;
            --bin)      binname="$2"; shift 2 ;;
            --features) features="$2"; shift 2 ;;
            --marker)   marker="$2";  shift 2 ;;
            --log)      log="$2";     shift 2 ;;
            --repo)     repo="$2";    shift 2 ;;
            *) echo "arc_build_and_verify: unknown argument '$1'" >&2; return 1 ;;
        esac
    done

    if [ -z "$package" ] || [ -z "$binname" ]; then
        echo "arc_build_and_verify: --package and --bin are both required" >&2
        return 1
    fi

    # cudnn is banned outright: -62% decode on V4 (CLAUDE.md). Asserted rather
    # than merely omitted, because a stray CARGO_BUILD_* or .cargo/config can
    # reintroduce it without the caller noticing.
    case "$features" in
        *cudnn*)
            echo "arc_build_and_verify: cudnn is in the feature set and is banned (-62% decode on V4)" >&2
            return 1 ;;
    esac

    local artifacts
    artifacts="$(mktemp)" || return 1

    local -a cmd=(cargo build --release -p "$package" --message-format=json-render-diagnostics)
    [ -n "$features" ] && cmd+=(--features "$features")

    if ! "${cmd[@]}" >"$artifacts" 2>>"$log"; then
        echo "arc_build_and_verify: BUILD FAILED for package '$package'. First errors:" >&2
        grep -m 20 -A 6 -E "^error" "$log" >&2 2>/dev/null
        rm -f "$artifacts"
        return 1
    fi

    # Cargo emits one compiler-artifact message per built target. The one we
    # want is the executable whose target name matches --bin. Asking cargo is
    # the entire point: a guessed path is how this bug happened.
    local resolved
    resolved="$(python3 - "$artifacts" "$binname" <<'PY'
import json, sys
found = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    if (msg.get("reason") == "compiler-artifact"
            and msg.get("executable")
            and msg.get("target", {}).get("name") == sys.argv[2]):
        found = msg["executable"]
print(found or "")
PY
)"

    rm -f "$artifacts"

    if [ -z "$resolved" ]; then
        echo "arc_build_and_verify: package '$package' produced no executable named '$binname'." >&2
        echo "  This is the stale-binary trap. Check the [[bin]] name in ${package}/Cargo.toml —" >&2
        echo "  arc-cli builds 'arc', mistralrs-cli builds 'mistralrs'. Do NOT fall back to a" >&2
        echo "  guessed target/release path; that is what made the bug invisible." >&2
        return 1
    fi
    if [ ! -x "$resolved" ]; then
        echo "arc_build_and_verify: cargo reported '$resolved' but it is not executable" >&2
        return 1
    fi
    if [ -n "$marker" ] && ! grep -qa -- "$marker" "$resolved"; then
        echo "arc_build_and_verify: FRESHNESS CHECK FAILED." >&2
        echo "  '$resolved' does not contain the marker '$marker', so it is not the code under" >&2
        echo "  test. Refusing to measure. If the marker is simply wrong, fix --marker; do not" >&2
        echo "  drop it — a stale binary fails in the direction of 'no effect'." >&2
        return 1
    fi

    ARC_VERIFIED_BIN="$resolved"
    export ARC_VERIFIED_BIN

    # Record the ref we built, so `arc_assert_running_revision` can hold the
    # running process to it. `git rev-parse HEAD` here mirrors exactly what
    # mistralrs-core/build.rs bakes in.
    ARC_VERIFIED_REV="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo "")"
    export ARC_VERIFIED_REV
    if [ -z "$ARC_VERIFIED_REV" ]; then
        echo "arc_build_and_verify: WARNING — could not read git HEAD of '$repo';" >&2
        echo "  arc_assert_running_revision will be unable to check provenance." >&2
    fi
    return 0
}

# Assert that the RUNNING process reports the commit we built.
#
#     arc_assert_running_revision --log "$SERVERLOG" [--timeout-s 600]
#
# Call it after the server is healthy (the line is emitted once the model is
# loaded). Returns 0 only if a `git revision: <sha>` line appears and matches
# $ARC_VERIFIED_REV. Absent, `unknown`, or mismatched are all failures — a
# missing line means the assertion did not run, which is not the same as passing.
arc_assert_running_revision() {
    local log="" timeout_s=600
    while [ $# -gt 0 ]; do
        case "$1" in
            --log)       log="$2";       shift 2 ;;
            --timeout-s) timeout_s="$2"; shift 2 ;;
            *) echo "arc_assert_running_revision: unknown argument '$1'" >&2; return 1 ;;
        esac
    done
    [ -n "$log" ] || { echo "arc_assert_running_revision: --log is required" >&2; return 1; }

    if [ -z "${ARC_VERIFIED_REV:-}" ]; then
        echo "arc_assert_running_revision: no ARC_VERIFIED_REV recorded — call" >&2
        echo "  arc_build_and_verify first. Refusing to pass an unchecked provenance." >&2
        return 1
    fi

    local waited=0 line=""
    while [ "$waited" -lt "$timeout_s" ]; do
        line="$(grep -ao 'git revision: [A-Za-z0-9]\{4,40\}' "$log" 2>/dev/null | tail -1)"
        [ -n "$line" ] && break
        sleep 2
        waited=$((waited + 2))
    done

    if [ -z "$line" ]; then
        echo "arc_assert_running_revision: no 'git revision:' line in $log after ${timeout_s}s." >&2
        echo "  The assertion did not run. That is a failure, not a pass." >&2
        return 1
    fi

    local running="${line##git revision: }"
    if [ "$running" = "unknown" ]; then
        echo "arc_assert_running_revision: the process reports 'git revision: unknown'" >&2
        echo "  (build.rs could not run git in the build dir). Provenance unprovable." >&2
        return 1
    fi
    # build.rs bakes the full sha; compare on the shorter of the two so an
    # abbreviated log format still matches exactly rather than loosely.
    local n=${#running}
    if [ "${ARC_VERIFIED_REV:0:$n}" != "$running" ]; then
        echo "arc_assert_running_revision: PROVENANCE MISMATCH." >&2
        echo "  built:   $ARC_VERIFIED_REV" >&2
        echo "  running: $running" >&2
        echo "  The process under measurement is NOT the commit that was built." >&2
        return 1
    fi
    ARC_RUNNING_REV="$running"
    export ARC_RUNNING_REV
    return 0
}
