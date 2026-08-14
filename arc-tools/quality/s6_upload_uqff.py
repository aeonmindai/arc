#!/usr/bin/env python3
"""Push a baked UQFF checkpoint to the Hub, then VERIFY every shard landed.

Why this exists: FACTS.md's bake-time correction says it plainly — "FIX THE
COST, NOT THE MYSTERY: bake ONCE and reuse the UQFF instead of re-baking every
session." Every session so far has re-baked (~$5-30 each) because no artifact
was ever published. This script is the step that ends that, so it runs
IMMEDIATELY after the bake, before any measurement, so that a session which
dies later still leaves the artifact behind.

API notes (verified against the huggingface_hub docs, 2026-08-14, NOT memory):
  * `HfApi.upload_folder` is the current API. It streams in several commits and
    RESUMES interrupted uploads on re-run, so a dropped 68 GB upload is retried,
    not restarted.
  * `upload_large_folder` / `hf upload-large-folder` are DEPRECATED and slated
    for removal — do not reach for them.
  * `HF_XET_HIGH_PERFORMANCE=1` turns on hf_xet's high-performance mode (uses
    available bandwidth + cores). `HF_HUB_ENABLE_HF_TRANSFER` is deprecated and
    no longer in use.
  * The token is read from `HF_TOKEN` in the environment
    (`_get_token_from_environment`, constants.py). We pass it explicitly anyway
    so an inherited stale token in the box's HF cache cannot win.

TOKEN DISCIPLINE (a session-5 lesson: a token was shared in plaintext and had
to be rotated): the token arrives via a 0600 file, is never an argv, is never
printed, and is scrubbed from any exception text before it is raised. The
caller deletes the file before teardown.

Exit codes: 0 upload verified · 1 upload or verification failed · 2 bad usage.
"""
import argparse
import json
import os
import sys


def _fail(msg):
    print(f"UPLOAD_FAIL {msg}", flush=True)
    return 1


def _scrub(text, secret):
    """Never let a token reach stdout/stderr through an exception string."""
    if secret and secret in text:
        return text.replace(secret, "<HF_TOKEN>")
    return text


def local_inventory(folder, patterns):
    """(relpath -> size) for every file matching one of `patterns`."""
    import fnmatch

    out = {}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if not any(fnmatch.fnmatch(name, p) for p in patterns):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, folder)
            out[rel.replace(os.sep, "/")] = os.path.getsize(full)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", required=True, help="local UQFF directory")
    ap.add_argument("--repo-id", required=True, help="e.g. aeonmind/DeepSeek-V4-Flash-UQFF-qtip2")
    ap.add_argument("--token-file", required=True, help="0600 file holding the HF token")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--path-in-repo", default=None,
                    help="subdirectory in the repo (default: repo root)")
    ap.add_argument("--patterns", default="*.uqff,*.json,*.txt,*.safetensors",
                    help="comma-separated allow_patterns")
    ap.add_argument("--commit-message", default="Arc session-6 qtip2 bake")
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory + auth check only; uploads nothing")
    args = ap.parse_args(argv)

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    if not os.path.isdir(args.folder):
        return _fail(f"folder {args.folder} does not exist")
    inv = local_inventory(args.folder, patterns)
    if not inv:
        return _fail(f"no files matching {patterns} under {args.folder}")
    total = sum(inv.values())
    shards = [f for f in inv if f.endswith(".uqff")]
    if not shards:
        return _fail(f"no *.uqff shard under {args.folder} — refusing to publish a bake-less folder")
    print(f"UPLOAD_PLAN files={len(inv)} shards={len(shards)} bytes={total}", flush=True)

    try:
        with open(args.token_file) as fh:
            token = fh.read().strip()
    except OSError as e:
        return _fail(f"cannot read token file: {type(e).__name__}")
    if not token:
        return _fail("token file is empty")

    # Belt and braces: make the token visible to the library the documented way
    # too, but never let it reach a log line.
    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return _fail("huggingface_hub not installed (pip install -U huggingface_hub)")

    api = HfApi(token=token)
    try:
        who = api.whoami()
        print(f"UPLOAD_AUTH ok as {who.get('name', '?')}", flush=True)
    except Exception as e:  # noqa: BLE001
        return _fail("whoami failed: " + _scrub(f"{type(e).__name__}: {e}", token)[:180])

    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        return _fail("create_repo failed: " + _scrub(f"{type(e).__name__}: {e}", token)[:180])

    if args.dry_run:
        print("UPLOAD_DRYRUN ok (nothing sent)", flush=True)
        return 0

    try:
        api.upload_folder(
            folder_path=args.folder,
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=args.path_in_repo,
            allow_patterns=patterns,
            commit_message=args.commit_message,
        )
    except Exception as e:  # noqa: BLE001
        return _fail("upload_folder failed: " + _scrub(f"{type(e).__name__}: {e}", token)[:300])

    # THE TEETH: an upload that silently drops shards is worse than no upload,
    # because the next session would trust it. List the repo back and diff.
    try:
        remote = set(api.list_repo_files(repo_id=args.repo_id, repo_type="model"))
    except Exception as e:  # noqa: BLE001
        return _fail("list_repo_files failed: " + _scrub(f"{type(e).__name__}: {e}", token)[:180])

    prefix = (args.path_in_repo.rstrip("/") + "/") if args.path_in_repo else ""
    missing = sorted(f for f in inv if (prefix + f) not in remote)
    if missing:
        return _fail(f"{len(missing)} file(s) missing on the hub after upload: {missing[:5]}")

    print(json.dumps({
        "repo_id": args.repo_id,
        "files": len(inv),
        "shards": len(shards),
        "bytes": total,
        "private": args.private,
    }), flush=True)
    print(f"UPLOAD_OK repo={args.repo_id} files={len(inv)} shards={len(shards)} bytes={total}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
