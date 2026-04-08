"""Arc inference — Qwen3-32B with CUDA graphs on B200"""

import modal

MODEL = "Qwen/Qwen3-32B"
PORT = 8000
MINUTES = 60

arc_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "ca-certificates", "git", "build-essential", "pkg-config", "libssl-dev")
    # nsight-compute (ncu) for hardware-counter profiling of GEMV/attention kernels.
    # Comes from the CUDA toolkit; install the standalone package so we get a recent
    # ncu that supports Blackwell (sm_100).
    .apt_install("nsight-compute-2025.1.0")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/opt/nvidia/nsight-compute/2025.1.0:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_COMPUTE_CAP": "100",
        "RUSTFLAGS": "-C target-cpu=x86-64",
    })
    .run_commands(
        "git clone https://github.com/aeonmindai/arc.git /tmp/arc",
        "cd /tmp/arc && git checkout 1c536d799 && git log --oneline -1 && echo BUILD_V81",
        "cd /tmp/arc && cargo build --release -p mistralrs-cli --features 'cuda flash-attn'",
        # Build the standalone GEMV bench binary
        "cd /tmp/arc && cargo build --release -p arc-cuda-graph --features cuda --bin gemv_bench",
        "cp /tmp/arc/target/release/mistralrs /usr/local/bin/mistralrs",
        "cp /tmp/arc/target/release/gemv_bench /usr/local/bin/gemv_bench",
        "rm -rf /tmp/arc /root/.cargo/registry",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

hf_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("arc-inference")


@app.function(
    image=arc_image,
    gpu="B200",
    scaledown_window=1 * MINUTES,
    timeout=3 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_vol},
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=3 * MINUTES)
def serve():
    import subprocess
    cmd = f"mistralrs serve --port {PORT} --model-id {MODEL} --pa-cache-type turboquant"
    print(cmd)
    subprocess.Popen(cmd, shell=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware-counter profiling via nsight-compute (ncu)
# ─────────────────────────────────────────────────────────────────────────────
profile_vol = modal.Volume.from_name("arc-profile-reports", create_if_missing=True)


@app.function(
    image=arc_image,
    gpu="B200",
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_vol,
        "/profile": profile_vol,
    },
)
def profile_kernels(
    kernel_regex: str = "gemv_bf16",
    launch_skip: int = 2000,
    launch_count: int = 64,
    metric_set: str = "full",
    max_tokens: int = 200,
):
    """
    Run mistralrs serve under nsight-compute, drive a single decode request,
    and capture per-kernel hardware counters for the specified kernel regex.

    Reports are written to /profile/ (Modal Volume) so they can be downloaded
    after the function returns.

    Args:
        kernel_regex: regex matched against kernel names to profile
                      (e.g. "gemv_bf16", "tq_attn", "rmsnorm_qk_pair")
        launch_skip: skip this many kernel launches before starting capture
                     (lets the model warm up + cuda graph capture finish)
        launch_count: number of matching launches to capture
        metric_set: ncu --set value: full | detailed | basic | roofline
        max_tokens: completion length to drive enough kernel launches
    """
    import subprocess
    import time
    import os
    import urllib.request
    import json

    os.makedirs("/profile", exist_ok=True)
    timestamp = int(time.time())
    report_base = f"/profile/profile_{kernel_regex.replace('/', '_')}_{timestamp}"

    # Build the ncu command. ncu launches mistralrs as a child and profiles the
    # kernel launches matching the regex. --target-processes all so we catch
    # the spawned mistralrs binary.
    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name", f"regex:{kernel_regex}",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--set", metric_set,
        "--export", report_base,
        "--force-overwrite",
        "--print-summary", "per-kernel",
        "--print-units", "base",
        "mistralrs", "serve",
        "--port", str(PORT),
        "--model-id", MODEL,
        "--pa-cache-type", "turboquant",
    ]
    print("Launching:", " ".join(ncu_cmd), flush=True)

    # ncu will keep mistralrs serve alive in the foreground and exit only when
    # serve exits. We need to send a request to drive decode, then kill serve.
    proc = subprocess.Popen(
        ncu_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    # Wait for the server port to come up
    deadline = time.time() + 5 * MINUTES
    server_up = False
    while time.time() < deadline:
        if proc.poll() is not None:
            print("ncu exited before server came up", flush=True)
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1).read()
            server_up = True
            break
        except Exception:
            time.sleep(2)
    if not server_up:
        out = proc.stdout.read() if proc.stdout else ""
        return {"ok": False, "stage": "server-start", "log_tail": out[-4000:]}

    print("Server up. Sending decode request...", flush=True)
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": "Explain quantum computing in detail with many examples."}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10 * MINUTES).read()
        print("Decode request done.", flush=True)
    except Exception as e:
        print(f"Decode request error (may be ok if ncu finished early): {e}", flush=True)

    # Give ncu a moment to flush the report, then terminate the server.
    time.sleep(5)
    subprocess.run(["pkill", "-TERM", "mistralrs"], check=False)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()

    log_tail = ""
    if proc.stdout:
        log_tail = proc.stdout.read()
    print(log_tail, flush=True)

    # List captured report files
    files = sorted(os.listdir("/profile"))
    profile_vol.commit()
    return {
        "ok": True,
        "report_base": report_base,
        "files": files,
        "log_tail": log_tail[-8000:],
    }


@app.function(
    image=arc_image,
    timeout=5 * MINUTES,
    volumes={"/profile": profile_vol},
)
def list_reports():
    """List all profile reports captured in the volume."""
    import os
    profile_vol.reload()
    files = []
    for name in sorted(os.listdir("/profile")):
        path = f"/profile/{name}"
        try:
            size = os.path.getsize(path)
            files.append({"name": name, "size": size})
        except Exception:
            pass
    return files


@app.function(
    image=arc_image,
    timeout=5 * MINUTES,
    volumes={"/profile": profile_vol},
)
def fetch_report(filename: str) -> bytes:
    """Read a profile report from the volume and return it."""
    profile_vol.reload()
    with open(f"/profile/{filename}", "rb") as f:
        return f.read()


@app.function(
    image=arc_image,
    gpu="B200",
    timeout=5 * MINUTES,
    volumes={"/profile": profile_vol},
)
def summarize_report(filename: str, page: str = "details", kernel_regex: str = ".*") -> str:
    """
    Run `ncu --import file.ncu-rep --page details` and return the textual summary.
    Useful for getting bandwidth/stall data without downloading the binary report.
    """
    import subprocess
    profile_vol.reload()
    cmd = [
        "ncu",
        "--import", f"/profile/{filename}",
        "--page", page,
        "--kernel-name", f"regex:{kernel_regex}",
        "--print-units", "base",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * MINUTES)
    return (out.stdout or "") + ("\n=== STDERR ===\n" + out.stderr if out.stderr else "")


@app.local_entrypoint()
def profile(
    kernel_regex: str = "gemv_bf16",
    launch_skip: int = 2000,
    launch_count: int = 64,
    metric_set: str = "full",
    max_tokens: int = 200,
):
    """Local entrypoint: kick off a profiling run and print the result."""
    result = profile_kernels.remote(
        kernel_regex=kernel_regex,
        launch_skip=launch_skip,
        launch_count=launch_count,
        metric_set=metric_set,
        max_tokens=max_tokens,
    )
    print("=== RESULT ===")
    print({k: v for k, v in result.items() if k != "log_tail"})
    print("=== LOG TAIL ===")
    print(result.get("log_tail", "")[-4000:])


@app.local_entrypoint()
def summarize(filename: str, kernel_regex: str = ".*", page: str = "details"):
    """Local entrypoint: print the textual summary of a captured report."""
    text = summarize_report.remote(filename=filename, kernel_regex=kernel_regex, page=page)
    print(text)


@app.local_entrypoint()
def list_profiles():
    """List captured profile files in the volume."""
    files = list_reports.remote()
    for f in files:
        print(f"{f['size']:>12}  {f['name']}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone GEMV benchmark — runs in seconds, no model load
# ─────────────────────────────────────────────────────────────────────────────


@app.function(image=arc_image, gpu="B200", timeout=10 * MINUTES)
def run_gemv_bench() -> str:
    """Run the standalone GEMV bench binary and return its stdout."""
    import subprocess
    out = subprocess.run(
        ["gemv_bench"],
        capture_output=True,
        text=True,
        timeout=8 * MINUTES,
    )
    return (out.stdout or "") + ("\n=== STDERR ===\n" + out.stderr if out.stderr else "")


@app.local_entrypoint()
def bench():
    """Local entrypoint: run the GEMV bench and print results."""
    text = run_gemv_bench.remote()
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# nsys timeline tracing attempt — uses CUDA Activity API which may pass gVisor
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=arc_image,
    gpu="B200",
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_vol,
        "/profile": profile_vol,
    },
)
def run_nsys_trace(max_tokens: int = 100):
    """
    Try to capture an nsys timeline of one decode request.
    Uses CUDA Activity API (different ioctl path than ncu HW counters).
    """
    import subprocess
    import time
    import os
    import urllib.request
    import json

    os.makedirs("/profile", exist_ok=True)
    timestamp = int(time.time())
    out_base = f"/profile/nsys_{timestamp}"

    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--gpu-metrics-device=none",
        "--output", out_base,
        "--force-overwrite=true",
        "--stop-on-exit=true",
        "mistralrs", "serve",
        "--port", str(PORT),
        "--model-id", MODEL,
        "--pa-cache-type", "turboquant",
    ]
    print("Launching:", " ".join(nsys_cmd), flush=True)
    proc = subprocess.Popen(
        nsys_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    deadline = time.time() + 5 * MINUTES
    server_up = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1).read()
            server_up = True
            break
        except Exception:
            time.sleep(2)
    if not server_up:
        out = proc.stdout.read() if proc.stdout else ""
        return {"ok": False, "stage": "server-start", "log_tail": out[-4000:]}

    print("Server up. Sending decode request...", flush=True)
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": "Explain quantum computing in detail."}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5 * MINUTES).read()
    except Exception as e:
        print(f"Decode error: {e}", flush=True)

    time.sleep(3)
    subprocess.run(["pkill", "-TERM", "mistralrs"], check=False)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    out = proc.stdout.read() if proc.stdout else ""
    print(out, flush=True)
    profile_vol.commit()

    files = sorted(os.listdir("/profile"))
    return {"ok": True, "files": files, "log_tail": out[-4000:]}


@app.local_entrypoint()
def nsys(max_tokens: int = 100):
    """Local entrypoint: run nsys trace."""
    result = run_nsys_trace.remote(max_tokens=max_tokens)
    print({k: v for k, v in result.items() if k != "log_tail"})
    print("=== LOG TAIL ===")
    print(result.get("log_tail", "")[-4000:])
