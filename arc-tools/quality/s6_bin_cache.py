#!/usr/bin/env python3
"""Cache the built `mistralrs` binary on the Hub so a re-entry skips the build.

WHY. A GPU re-entry costs ~70 min before any work happens: boot ~10, the
149 GB weight download ~35, the cargo build ~25. Two of those three are
avoidable. The weights are not needed at all once the UQFF is published (the
68 GB UQFF replaces them), and the build is not needed when the tree has not
changed. That turns a ~70 min / ~$5.74 re-entry into ~26 min / ~$2.13, which
is what makes "tear down and try again" viable at any granularity.

WHERE, AND WHY THERE. `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` under the
`arc-bin/` prefix — the same private repo as the checkpoint, not a sibling.
Reasons: (1) one repo means one token scope and one place to look; (2) the
checkpoint and the binary that produced and can read it are versioned
together, so a consumer who pulls the UQFF gets the exact binary that
understands its stamp/flags byte; (3) two repos would drift, and the failure
mode of drift here is a binary that silently cannot read the artifact beside
it. The cost is a non-model file in a model repo, namespaced clearly and
private.

HONESTY ABOUT WHAT THIS DOES *NOT* BUY. The commonest re-entry after a real
failure is "our Rust was wrong", and that changes the tree — so the manifest
records the arc commit and a different commit invalidates the cache
automatically. The cache helps the other loops: a bad rental, a harness bug, a
measurement-only re-run. It also does NOT cover `cargo run --example`
(the gemv sweep and stats_info), which still need a source build; a re-entry
that includes those steps still pays for one.

VALIDATION. A cached binary is only trusted when the manifest's driver
version, CUDA toolkit version, compute capability and glibc version all match
this box, AND the file's sha256 matches. Those four are the things that make a
Linux binary refuse to run or run wrongly on a different rental. Even then the
binary is smoke-tested (`--version`, plus `ldd` reporting no missing objects)
before use, and the definitive proof is its first real use — the driver falls
back to a source build if that fails.

Exit codes: 0 ok · 1 failed / cache miss / mismatch · 2 bad usage.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

MANIFEST_NAME = "manifest.json"


def _out(msg):
    print(msg, flush=True)


def _fail(msg):
    _out(f"BINCACHE_FAIL {msg}")
    return 1


def _scrub(text, secret):
    return text.replace(secret, "<HF_TOKEN>") if secret and secret in text else text


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def driver_version():
    r = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return r.stdout.strip().splitlines()[0].strip() if r and r.returncode == 0 and r.stdout.strip() else ""


def compute_cap():
    r = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    return r.stdout.strip().splitlines()[0].strip() if r and r.returncode == 0 and r.stdout.strip() else ""


def toolkit_version():
    r = _run(["nvcc", "--version"])
    if not r or r.returncode != 0:
        return ""
    for line in r.stdout.splitlines():
        if "release" in line:
            return line.split("release", 1)[1].split(",")[0].strip()
    return ""


def glibc_version():
    try:
        return ".".join(platform.libc_ver()[1].split(".")[:2])
    except Exception:  # noqa: BLE001
        return ""


def arc_commit(arc_dir):
    r = _run(["git", "-C", arc_dir, "rev-parse", "HEAD"])
    return r.stdout.strip() if r and r.returncode == 0 else ""


def box_fingerprint(arc_dir):
    return {
        "arc_commit": arc_commit(arc_dir),
        "driver_version": driver_version(),
        "toolkit_version": toolkit_version(),
        "compute_cap": compute_cap(),
        "glibc": glibc_version(),
    }


# The fields that must match for a cached binary to be trusted. `arc_commit`
# is in here deliberately: a code fix is the commonest reason we re-enter at
# all, and a stale binary for a fixed tree is worse than no cache.
MATCH_KEYS = ("arc_commit", "driver_version", "toolkit_version", "compute_cap", "glibc")


def _api(token):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return HfApi(token=token)


def read_token(path):
    with open(path) as fh:
        return fh.read().strip()


def cmd_push(args, token):
    api = _api(token)
    if api is None:
        return _fail("huggingface_hub not installed")
    if not os.path.isfile(args.binary):
        return _fail(f"{args.binary} is not a file")

    fp = box_fingerprint(args.arc)
    manifest = dict(fp)
    manifest.update({
        "binary": os.path.basename(args.binary),
        "sha256": sha256(args.binary),
        "bytes": os.path.getsize(args.binary),
        "features": args.features,
        "built_utc": args.built_utc,
        "note": "cargo-run examples (qtip_gemv_tune, stats_info) are NOT covered "
                "by this cache and still need a source build",
    })
    stage = args.stage or os.path.join(os.path.dirname(args.binary) or ".", ".bincache")
    os.makedirs(stage, exist_ok=True)
    dst = os.path.join(stage, os.path.basename(args.binary))
    if os.path.abspath(dst) != os.path.abspath(args.binary):
        with open(args.binary, "rb") as src, open(dst, "wb") as out:
            for chunk in iter(lambda: src.read(1 << 22), b""):
                out.write(chunk)
    with open(os.path.join(stage, MANIFEST_NAME), "w") as fh:
        json.dump(manifest, fh, indent=2)

    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=stage,
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=args.prefix,
            commit_message=f"Arc binary cache: {manifest['arc_commit'][:12]} on driver {fp['driver_version']}",
        )
    except Exception as e:  # noqa: BLE001
        return _fail("upload failed: " + _scrub(f"{type(e).__name__}: {e}", token)[:200])
    _out(f"BINCACHE_PUSHED repo={args.repo_id} path={args.prefix} "
         f"commit={manifest['arc_commit'][:12]} bytes={manifest['bytes']}")
    return 0


def cmd_pull(args, token):
    api = _api(token)
    if api is None:
        return _fail("huggingface_hub not installed")
    from huggingface_hub import hf_hub_download

    fp = box_fingerprint(args.arc)
    try:
        mpath = hf_hub_download(repo_id=args.repo_id, repo_type="model",
                                filename=f"{args.prefix}/{MANIFEST_NAME}", token=token)
    except Exception as e:  # noqa: BLE001
        _out("BINCACHE_MISS no manifest: " + _scrub(f"{type(e).__name__}", token)[:80])
        return 1
    with open(mpath) as fh:
        manifest = json.load(fh)

    mismatch = [k for k in MATCH_KEYS if (manifest.get(k) or "") != (fp.get(k) or "")]
    if mismatch:
        for k in mismatch:
            _out(f"BINCACHE_MISMATCH {k}: cached={manifest.get(k)!r} box={fp.get(k)!r}")
        _out("BINCACHE_MISS fingerprint mismatch — building from source")
        return 1

    try:
        bpath = hf_hub_download(repo_id=args.repo_id, repo_type="model",
                                filename=f"{args.prefix}/{manifest['binary']}", token=token)
    except Exception as e:  # noqa: BLE001
        return _fail("binary download failed: " + _scrub(f"{type(e).__name__}", token)[:120])

    got = sha256(bpath)
    if got != manifest.get("sha256"):
        return _fail(f"sha256 mismatch: manifest={manifest.get('sha256')[:16]} got={got[:16]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(bpath, "rb") as src, open(args.out, "wb") as out:
        for chunk in iter(lambda: src.read(1 << 22), b""):
            out.write(chunk)
    os.chmod(args.out, 0o755)
    _out(f"BINCACHE_HIT {args.out} commit={manifest['arc_commit'][:12]} "
         f"driver={manifest['driver_version']} toolkit={manifest['toolkit_version']} "
         f"cap={manifest['compute_cap']} glibc={manifest['glibc']} bytes={manifest['bytes']}")
    return 0


def cmd_fingerprint(args, _token):
    _out(json.dumps(box_fingerprint(args.arc), indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "pull", "fingerprint"])
    ap.add_argument("--repo-id", default="aeonmind/DeepSeek-V4-Flash-UQFF-qtip2")
    ap.add_argument("--prefix", default="arc-bin")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--arc", default=os.environ.get("ARC", "."))
    ap.add_argument("--binary", default=None, help="push: path to target/release/mistralrs")
    ap.add_argument("--out", default=None, help="pull: where to write the binary")
    ap.add_argument("--stage", default=None, help="push: staging dir (default alongside the binary)")
    ap.add_argument("--features", default="cuda flash-attn")
    ap.add_argument("--built-utc", default="")
    args = ap.parse_args(argv)

    if args.action == "fingerprint":
        return cmd_fingerprint(args, None)
    if not args.token_file:
        return _fail("--token-file is required for push/pull")
    try:
        token = read_token(args.token_file)
    except OSError as e:
        return _fail(f"cannot read token file: {type(e).__name__}")
    if not token:
        return _fail("token file is empty")

    if args.action == "push":
        if not args.binary:
            return _fail("--binary is required for push")
        return cmd_push(args, token)
    if not args.out:
        return _fail("--out is required for pull")
    return cmd_pull(args, token)


if __name__ == "__main__":
    sys.exit(main())
