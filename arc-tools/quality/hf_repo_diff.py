#!/usr/bin/env python3
"""INDEPENDENT diff: what is on the Hub vs what is on this disk.

s6_upload_uqff.py verifies the upload against its OWN plan, not against the
source directory -- it once printed `UPLOAD_OK files=13` while silently
dropping a 14th file. This script never looks at that plan. It walks the local
folder, asks the HF API (`?blobs=true`, which returns per-file `size`) what the
repo actually contains, and prints the symmetric difference plus every
size mismatch.

Exit 0 only when every local file is present on the Hub at the same byte size.
"""
import argparse
import json
import os
import sys
import urllib.request

SKIP_DIRS = {".cache", ".git"}


def local_inventory(folder):
    out = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, folder).replace(os.sep, "/")
            out[rel] = os.path.getsize(full)
    return out


def hub_inventory(repo_id, token):
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    out = {}
    for s in data.get("siblings", []):
        out[s["rfilename"]] = s.get("size")
    return out, data.get("sha"), data.get("lastModified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--token-file", required=True)
    a = ap.parse_args()

    with open(a.token_file) as fh:
        token = fh.read().strip()

    loc = local_inventory(a.folder)
    hub, sha, mtime = hub_inventory(a.repo_id, token)

    print(f"DIFF repo={a.repo_id} sha={sha} lastModified={mtime}")
    print(f"DIFF local_files={len(loc)} local_bytes={sum(loc.values())}")
    hub_bytes = sum(v for v in hub.values() if v)
    print(f"DIFF hub_files={len(hub)} hub_bytes={hub_bytes}")

    missing = sorted(set(loc) - set(hub))
    extra = sorted(set(hub) - set(loc))
    mismatch = sorted(
        f"{k} local={loc[k]} hub={hub[k]}"
        for k in set(loc) & set(hub)
        if hub[k] is not None and hub[k] != loc[k]
    )

    print("DIFF missing_on_hub: " + (", ".join(missing) if missing else "NONE"))
    print("DIFF only_on_hub:    " + (", ".join(extra) if extra else "NONE"))
    print("DIFF size_mismatch:  " + ("; ".join(mismatch) if mismatch else "NONE"))
    for k in sorted(hub):
        print(f"  HUB {hub[k]:>14} {k}")

    if missing or mismatch:
        print("DIFF_FAIL")
        return 1
    print("DIFF_OK every local file present on the Hub at the same size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
