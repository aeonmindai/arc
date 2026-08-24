#!/usr/bin/env bash
# =============================================================================
# v4_rebake_driver.sh — turnkey, RESUMABLE V=4 re-bake driver for
# DeepSeek-V4-Flash.               Parent system: ArcQuant/ArcBake + ArcLab/Ops
#
# Companion doc: arc-tools/V4_REBAKE_RUNBOOK.md  (read it BEFORE renting).
# Quality gates:  arc-tools/v4_rebake_quality_gate.sh   (run after this passes)
# After-numbers:  arc-tools/v4_rebake_measure.sh        (run after the gate)
#
#   file_upload  -> /root/v4_rebake_driver.sh
#   (nohup bash /root/v4_rebake_driver.sh > /root/logs/rebake.log 2>&1 < /dev/null &)
#
# NEVER paste this into ssh_execute as a heredoc (ssh_execute mangles heredocs;
# session-1 lesson). Upload the file, then `bash` it.
#
# D10b VERSION RULE: a revision of this script gets a NEW filename
# (v4_rebake_driver_v2.sh, ...). Never overwrite a script that may be running —
# bash reads scripts lazily and an overwritten running script executes a splice
# of two versions.
#
# GEOMETRY IS PARAMETERIZED because the capability audit may adjust names.
# Everything the audit could change arrives as env, nothing is baked in:
#
#   RB_ISQ_RUNG      ISQ rung name handed to --isq        (default qtip2b)
#   RB_BAKE_W        trellis beam width -> ARC_QTIP_BEAM  (default 256)
#   RB_ARCH          -a architecture                      (default deepseekv4)
#   RB_BASE_MODEL    --uqff-base-model                    (default deepseek-ai/DeepSeek-V4-Flash)
#   RB_HF_REPO       --uqff-repo-id (README metadata ONLY — this driver never
#                    uploads; see the runbook's token section)
#   RB_BAKE_ENV      extra KEY=VAL pairs (space-separated) exported into the
#                    bake process, e.g. "ARC_QTIP_GEOMETRY=... ARC_QTIP_CODEBOOK=..."
#                    for whatever V=4 geometry knobs the audit lands on
#   RB_BAKE_DEVICES  --bake-devices list, e.g. "0,1"      (default: unset)
#   RB_MODEL_DIR     local source checkpoint (~149 GB)    (default $WORKROOT/models/DeepSeek-V4-Flash)
#   RB_UQFF_DIR      bake output dir                      (default $RB_MODEL_DIR/uqff-rebake)
#
# Verified CLI shape (mistralrs-cli/src/args/quantize.rs, this ref):
#   mistralrs quantize text -m <dir> -a <arch> --isq <rung> -o <dir>/ \
#       --uqff-base-model <id> --uqff-repo-id <id> [--bake-devices 0,1]
#
# DOCTRINE HONORED HERE
#   D4   greedy is banned. The driver refuses to launch if any greedy switch is
#        in the environment, and verifies the ARTIFACT search stamp after the
#        bake (qtip2b stamps every layer; qtip2 prints a bake header). A greedy
#        result quarantines the artifact and aborts.
#   D4b  W=256 or no bake. RB_BAKE_W below 256 is refused; there is no width
#        ladder and no degraded fallback rung.
#   D10b every status-feed line carries STAMP=<epoch> so staleness is provable.
#   D12  fixture gates run on realistic fp4_dequant/heavy-tailed fixtures — see
#        the quality gate script, step A.
#   D14  this driver only runs on a GPU box; there is no CPU fallback path.
#
# RESUME MODEL (honest statement of what the pipeline supports):
#   The bake pipeline streams UQFF shards to disk as they fill
#   (mistralrs-core/src/pipeline/isq.rs UqffShardWriter) but the quantize pass
#   itself has NO mid-model checkpoint: a killed bake cannot continue from
#   layer k. What this driver checkpoints instead:
#     * every step (preflight/download/build/parity/bake/manifest) writes a
#       done-file; a re-run skips completed steps, so a died box never repeats
#       the 149 GB download or the ~25 min build;
#     * a COMPLETE artifact (stamp-verified) skips the bake entirely;
#     * PARTIAL shards are detected, moved aside for forensics, and the bake
#       restarts — that restart is the floor the pipeline imposes, and the
#       driver says so out loud rather than pretending to resume.
#   Resume:  just re-run the script. Force a step: RB_FORCE_STEPS=bake,...
#
# ABORT POLICY (D4b loop): the instant an ABORT-IF trips, the driver harvests
# everything into $RB_STATUS_DIR/v4_rebake_results.tgz, writes
# results/rebake_failure.json with expected-vs-got, and exits 10. Then: pull
# the tarball, DELETE the box, diagnose free. Never debug on a paid box.
#
# Exit codes: 0 = bake complete + manifest + tar
#            10 = an ABORT-IF tripped (failure json + tarball are ready)
#             2 = invocation/environment error before any work
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------- paths
RB_ROOT=${RB_ROOT:-/root}
if [ -z "${WORKROOT:-}" ]; then
  WORKROOT=$RB_ROOT
  for m in /ephemeral /workspace /mnt /root; do
    [ -d "$m" ] && WORKROOT=$m && break
  done
fi
RB_ARC=${RB_ARC:-$WORKROOT/work/arc}
RB_MODEL_DIR=${RB_MODEL_DIR:-$WORKROOT/models/DeepSeek-V4-Flash}
RB_UQFF_DIR=${RB_UQFF_DIR:-$RB_MODEL_DIR/uqff-rebake}
RB_LOG_DIR=${RB_LOG_DIR:-$RB_ROOT/logs}
RB_STATUS_DIR=${RB_STATUS_DIR:-/srv/arcstatus}
RB_RESULTS=${RB_RESULTS:-$RB_ARC/arc-tools/quality/results}
RB_CKPT_DIR=${RB_CKPT_DIR:-$WORKROOT/rebake_ckpt}
RB_STATUS_FILE=${RB_STATUS_FILE:-$RB_STATUS_DIR/v4_rebake_status.txt}
Q=${Q:-$RB_ARC/arc-tools/quality}

# ---------------------------------------------------------------- geometry
RB_ISQ_RUNG=${RB_ISQ_RUNG:-qtip2b}
RB_BAKE_W=${RB_BAKE_W:-256}
RB_ARCH=${RB_ARCH:-deepseekv4}
RB_BASE_MODEL=${RB_BASE_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
RB_HF_REPO=${RB_HF_REPO:-aeonmind/DeepSeek-V4-Flash-UQFF-qtip2}
RB_BAKE_ENV=${RB_BAKE_ENV:-}
RB_BAKE_DEVICES=${RB_BAKE_DEVICES:-}

# ---------------------------------------------------------------- policy
RB_RATE_HR=${RB_RATE_HR:-1.49}              # bake leg default: A100-80G $/hr
RB_MIN_MODEL_GB=${RB_MIN_MODEL_GB:-140}     # source checkpoint is ~149 GB
RB_DL_WIRE_S=${RB_DL_WIRE_S:-5400}          # ~20 min at ~183 MB/s; wire at 90 min
RB_BAKE_WIRE_H=${RB_BAKE_WIRE_H:-6.5}       # 44 layers x 370-376 s/layer (A100) = ~4.6 h; H200 ~2.9 h
RB_PACE_T1_S=${RB_PACE_T1_S:-1500}          # first pace look: 25 min in
RB_PACE_T1_MIN_LAYERS=${RB_PACE_T1_MIN_LAYERS:-2}
RB_LAYER_REGEX=${RB_LAYER_REGEX:-Detected INT4}
RB_STATUS_EVERY=${RB_STATUS_EVERY:-30}
RB_EXPECT_MIN_GB=${RB_EXPECT_MIN_GB:-40}    # artifact sanity band (qtip2 bake was ~68 GB)
RB_EXPECT_MAX_GB=${RB_EXPECT_MAX_GB:-120}
RB_TAR_ARTIFACT=${RB_TAR_ARTIFACT:-0}       # 1 = also tar the UQFF dir for scp harvest
RB_FORCE_STEPS=${RB_FORCE_STEPS:-}
RB_DRYRUN=${RB_DRYRUN:-0}

START_EPOCH=${RB_START_EPOCH:-$(date +%s)}
export RB_START_EPOCH=$START_EPOCH
BAKE_PID=""
POLLER_PID=""
FAIL_LOG="rebake.log"

mkdir -p "$RB_LOG_DIR" "$RB_STATUS_DIR" "$RB_RESULTS" "$RB_CKPT_DIR" 2>/dev/null || {
  echo "FATAL: cannot create working directories" >&2; exit 2; }

# ---------------------------------------------------------------- helpers
mark() { echo ":::::: $* ::::::"; date -u +%H:%M:%S; }
nap()  { if [ "$RB_DRYRUN" = "1" ]; then sleep 0.2; else sleep "$1"; fi; }
elapsed_h() { awk -v s="$START_EPOCH" -v n="$(date +%s)" 'BEGIN{printf "%.3f", (n-s)/3600.0}'; }
past() { awk -v e="$(elapsed_h)" -v w="$1" 'BEGIN{exit !(e > w)}'; }
budget_line() { awk -v e="$(elapsed_h)" -v r="$RB_RATE_HR" 'BEGIN{printf "cum_h=%.2f spend_usd=%.2f\n", e, e*r}'; }
jstr() { printf '%s' "$1" | tr '\n\t' '  ' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# D10b: EVERY status-feed line carries STAMP=<epoch>. A feed that cannot be
# aged cannot be trusted.
status_line() { printf 'STAMP=%s %s\n' "$(date +%s)" "$*" >> "$RB_STATUS_FILE"; }

# Step checkpointing: a done-file per step, stamped. Re-runs skip done steps
# unless the step is named in RB_FORCE_STEPS.
step_done() {
  local s=$1
  case ",$RB_FORCE_STEPS," in *",$s,"*) return 1;; esac
  [ -f "$RB_CKPT_DIR/$s.done" ]
}
mark_done() {
  printf 'STAMP=%s step=%s %s\n' "$(date +%s)" "$1" "${2:-}" > "$RB_CKPT_DIR/$1.done"
  status_line "STEP_DONE $1 ${2:-}"
}

kill_pid() {  # PID kill only. Never a pattern (a pkill -f once killed the SSH caller).
  local pid=$1 grace=${2:-8}
  [ -z "$pid" ] && return 0
  kill "$pid" 2>/dev/null
  nap "$grace"
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  return 0
}

# THE tar-and-publish path, shared by success and failure: the tarball is the
# deliverable of a FAILED session too. A box that dies with nothing pulled
# makes the next attempt blind and we pay twice for the same ignorance.
harvest() {
  local why=${1:-done}
  mark "HARVEST ($why)"
  kill_pid "${BAKE_PID:-}" 4; BAKE_PID=""
  kill_pid "${POLLER_PID:-}" 2; POLLER_PID=""
  cp "$RB_LOG_DIR"/*.log "$RB_RESULTS/" 2>/dev/null
  cp "$RB_STATUS_FILE" "$RB_RESULTS/" 2>/dev/null
  cp "$RB_STATUS_DIR"/box_health*.json "$RB_RESULTS/" 2>/dev/null
  ( cd "$(dirname "$RB_RESULTS")" && tar czf "$RB_ROOT/v4_rebake_results.tgz" "$(basename "$RB_RESULTS")/" ) 2>/dev/null
  cp "$RB_ROOT/v4_rebake_results.tgz" "$RB_STATUS_DIR/" 2>/dev/null
  if [ -s "$RB_STATUS_DIR/v4_rebake_results.tgz" ]; then
    echo "RESULTS_TGZ: $RB_STATUS_DIR/v4_rebake_results.tgz ($(wc -c < "$RB_STATUS_DIR/v4_rebake_results.tgz" | tr -d ' ') bytes)"
  else
    echo "FAIL: could not write the results tarball — pull $RB_LOG_DIR/ file by file before deleting"
  fi
  if [ "$RB_TAR_ARTIFACT" = "1" ] && [ -d "$RB_UQFF_DIR" ]; then
    ( cd "$(dirname "$RB_UQFF_DIR")" && tar cf "$RB_ROOT/v4_rebake_uqff.tar" "$(basename "$RB_UQFF_DIR")/" ) 2>/dev/null \
      && echo "ARTIFACT_TAR: $RB_ROOT/v4_rebake_uqff.tar ($(wc -c < "$RB_ROOT/v4_rebake_uqff.tar" | tr -d ' ') bytes) — scp it BEFORE deleting the box"
  fi
  budget_line
}

#   abort <STEP_TAG> <assertion-id> <expected> <got> <message>
# All five are mandatory: a failure report that does not say expected-vs-got is
# a report someone has to rent a box to reproduce.
abort() {
  local step=$1 assertion=$2 expected=$3 got=$4 msg=$5
  echo "ABORT_$step $msg"
  echo "FAILED_ASSERTION $assertion  expected=[$expected]  got=[$got]"
  status_line "ABORT step=$step assertion=$assertion"
  {
    printf '{\n  "step": "%s",\n  "assertion": "%s",\n' "$step" "$assertion"
    printf '  "expected": "%s",\n  "got": "%s",\n' "$(jstr "$expected")" "$(jstr "$got")"
    printf '  "message": "%s",\n' "$(jstr "$msg")"
    printf '  "logs_in_tarball": ["results/%s"],\n' "$(basename "${FAIL_LOG:-rebake.log}")"
    printf '  "geometry": {"isq": "%s", "beam_w": %s, "arch": "%s", "bake_env": "%s"},\n' \
      "$RB_ISQ_RUNG" "$RB_BAKE_W" "$RB_ARCH" "$(jstr "$RB_BAKE_ENV")"
    printf '  "elapsed_h": %s,\n' "$(elapsed_h)"
    printf '  "gpu": "%s",\n' "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    printf '  "arc_commit": "%s",\n' "$(git -C "$RB_ARC" rev-parse HEAD 2>/dev/null)"
    printf '  "utc": "%s"\n}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$RB_RESULTS/rebake_failure.json" 2>/dev/null
  cp "$RB_RESULTS/rebake_failure.json" "$RB_STATUS_DIR/" 2>/dev/null
  harvest "FAILED at $step"
  echo "DELETE-AND-DIAGNOSE (D4b): pull $RB_STATUS_DIR/v4_rebake_results.tgz, DELETE the box, fix on CPU where compute is free, rent again. Never debug here."
  exit 10
}

# Background progress poller: one stamped line every RB_STATUS_EVERY seconds
# while the bake runs. Layer counting is regex-parameterized because the
# per-layer line differs by rung; shards + bytes are rung-independent.
progress_poller() {
  local log=$1 pid=$2
  while kill -0 "$pid" 2>/dev/null; do
    local layers shards bytes
    layers=$(grep -cE "$RB_LAYER_REGEX" "$log" 2>/dev/null || true)
    shards=$(ls "$RB_UQFF_DIR"/*.uqff 2>/dev/null | wc -l | tr -d ' ')
    bytes=$(du -sb "$RB_UQFF_DIR" 2>/dev/null | cut -f1)
    status_line "BAKE_PROGRESS layers=${layers:-0} shards=${shards:-0} bytes=${bytes:-0} log_bytes=$(wc -c < "$log" 2>/dev/null | tr -d ' ') pid=$pid"
    nap "$RB_STATUS_EVERY"
  done
  status_line "BAKE_PROCESS_EXITED pid=$pid"
}

# Artifact completeness: for qtip2b, read_qtip_stamp.py decodes rung + search +
# width out of every layer payload (exit 0 only if all agree). For other rungs
# there is no in-artifact stamp reader here; a prior bake log header is the
# only provenance, so absence of one means NOT complete.
artifact_complete() {
  local shards
  shards=$(ls "$RB_UQFF_DIR"/*.uqff 2>/dev/null | wc -l | tr -d ' ')
  [ "${shards:-0}" -ge 1 ] || return 1
  if [ "$RB_ISQ_RUNG" = "qtip2b" ]; then
    python3 "$Q/read_qtip_stamp.py" --folder "$RB_UQFF_DIR" --expect-width "$RB_BAKE_W" \
      > "$RB_LOG_DIR/stamp_precheck.log" 2>&1
    return $?
  fi
  grep -q "search=viterbi-beam(W=$RB_BAKE_W)" "$RB_LOG_DIR/bake.log" 2>/dev/null
}

# =============================================================================
mark "V4 REBAKE DRIVER START rung=$RB_ISQ_RUNG W=$RB_BAKE_W arch=$RB_ARCH arc=$RB_ARC"
echo "model=$RB_MODEL_DIR out=$RB_UQFF_DIR bake_env=[$RB_BAKE_ENV]"
status_line "DRIVER_START rung=$RB_ISQ_RUNG W=$RB_BAKE_W arch=$RB_ARCH"
budget_line

# ---------------------------------------------------------------- hard refusals
# D4b: W=256 or no bake. There is no width ladder and no degraded rung — the
# width is stamped into the artifact forever.
awk -v w="$RB_BAKE_W" 'BEGIN{exit !(w+0 >= 256)}' \
  || abort GEOMETRY "RB_BAKE_W >= 256 (D4b: W=256 or no bake)" ">= 256" "$RB_BAKE_W" "a narrower beam is a permanently inferior artifact (-0.004 matmul cos at 128, -0.014 at 64). If W=256 does not hold on this box, the answer is a different box, not a worse artifact."

# D4: greedy is banned. Refuse to launch if any switch that resolves the search
# to greedy is in the effective environment (session env or RB_BAKE_ENV).
for g in ARC_QTIP_EXPERT_GREEDY ARC_ALLOW_UNSTAMPED_QTIP; do
  if env | grep -q "^$g=" || printf '%s' "$RB_BAKE_ENV" | grep -q "$g="; then
    abort GREEDY_ENV "no greedy/unstamped switch in the effective env" "$g unset" "$g present" "D4: greedy is banned forever, structurally. Unset $g in the launching shell / RB_BAKE_ENV and re-run."
  fi
done
# Banned measurement gates must not leak into a bake either (s6 list).
if env | grep -qE '^(ARC_QUANT_ATTENTION|ARC_QTIP_REFINE_SCALES|ARC_V4_WINDOW_ONLY|ARC_V4_STANDARD_DENSE|ARC_V4_NO_ABSORBED_DECODE|ARC_FORCE_NAIVE_SDPA|ARC_TIME_DECODE|ARC_QTIP_HESSIAN)='; then
  env | grep -E '^(ARC_QUANT_ATTENTION|ARC_QTIP_REFINE_SCALES|ARC_V4_WINDOW_ONLY|ARC_V4_STANDARD_DENSE|ARC_V4_NO_ABSORBED_DECODE|ARC_FORCE_NAIVE_SDPA|ARC_TIME_DECODE|ARC_QTIP_HESSIAN)='
  abort ENV_DIRTY "banned gate exported" "none exported" "listed above" "a banned gate is in the session environment; unset it in the launching shell and re-run."
fi
echo "ENV_CLEAN"

# =============================================================================
# R0 — PREFLIGHT, before the 149 GB download.
# The silent-PTX trap: a driver that refuses a too-new PTX ISA fails in
# cudaGetLastError(), which nothing checks — the box produces plausible wrong
# numbers, not a crash. `source preflight || handler` NEVER FIRES: the
# preflight's trailing `[ ... ] || exit 1` makes `source` return 0 on a FAILING
# box. Check the _ARC_PF_FAILED flag directly, and require it to EXIST — an
# unset flag means the checks never ran.
# =============================================================================
if step_done preflight; then
  echo "SKIP preflight (checkpointed: $(cat "$RB_CKPT_DIR/preflight.done"))"
else
  mark "R0 preflight"
  FAIL_LOG=health.log
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "COMPUTE_CAP=$CC"
  awk -v c="${CC:-0}" 'BEGIN{exit !(c+0 >= 8.0)}' \
    || abort BOX_COMPUTE_CAP "nvidia-smi compute_cap" ">= 8.0" "${CC:-<no answer>}" "below 8.0 mistralrs-quant/build.rs does not set has_qtip_kernels, every CUDA parity test passes VACUOUSLY, and the bake falls to the CPU crawl. Rent an A100/H100/H200-class box."

  PF="/root/arc-tools/gpu_box_preflight.sh"
  [ -f "$PF" ] || PF="$RB_ARC/arc-tools/gpu_box_preflight.sh"
  PTX_RECEIPT="${PTX_RECEIPT:-$RB_LOG_DIR/PTX_GATE_PASSED}"
  if [ -f "$PF" ]; then
    # shellcheck disable=SC1090
    source "$PF" || true
    [ -n "${_ARC_PF_FAILED+x}" ] \
      || abort PREFLIGHT_VACUOUS "_ARC_PF_FAILED is set after sourcing" "flag set (0 or 1)" "unset" "sourcing $PF did not set _ARC_PF_FAILED — its checks never ran. Refusing to bake behind a gate that never executed."
    [ "$_ARC_PF_FAILED" = "0" ] \
      || abort PREFLIGHT_FAILED "_ARC_PF_FAILED" "0" "$_ARC_PF_FAILED" "gpu_box_preflight.sh FAILED; its FAIL[<CHECK>] line above names the reason. Most likely cuda-compat: apt-get install the matching cuda-compat package and put /usr/local/cuda/compat first on LD_LIBRARY_PATH — then DELETE this box anyway and rent from an image whose toolkit matches (policy)."
    echo "PTX_GATE=gpu_box_preflight ($PF)"
  elif [ -f "$PTX_RECEIPT" ]; then
    echo "PTX_GATE=receipt ($(cat "$PTX_RECEIPT"))"
  else
    abort PTX_GATE_MISSING "one PTX-JIT gate ran" "gpu_box_preflight.sh sourced OR $PTX_RECEIPT receipt" "neither exists" "NO PTX GATE RAN. gpu_box_preflight.sh is absent on this ref and no bootstrap receipt exists (arcgraph_box_bootstrap.sh writes one). An ungated box reports a BETTER number, not a worse one — refusing to bake ungated."
  fi

  if [ -f "$Q/box_health_gate.sh" ]; then
    bash "$Q/box_health_gate.sh" --burn-secs 60 --json "$RB_STATUS_DIR/box_health.json" \
      2>&1 | tee "$RB_LOG_DIR/health.log"
    rc=${PIPESTATUS[0]}
    [ "$rc" = "0" ] \
      || abort BOX_HEALTH "box_health_gate.sh exit code" "0 (all gates PASS)" "$rc — $(grep -m1 '^FAIL ' "$RB_LOG_DIR/health.log" || echo see results/health.log)" "the rental is bad (s5a precedent: 99% util at 132 W = transfer-starved, 3 min/layer with a 30 s/layer binary). DELETE it and rent a different box, preferably a different region; box_health.json says which offline."
  else
    echo "WARN: $Q/box_health_gate.sh missing on this ref — PTX gate ran, burn gate did not"
  fi
  mark_done preflight "cc=$CC"
fi

# =============================================================================
# R1 — SOURCE CHECKPOINT (149 GB), in parallel with the build.
# NO persistent volumes exist: assume a fresh download may be needed (~20 min
# at ~183 MB/s). The download runs in the background while cargo builds.
# =============================================================================
DL_PID=""
model_present() {
  [ -f "$RB_MODEL_DIR/config.json" ] || return 1
  [ -f "$RB_MODEL_DIR/model.safetensors.index.json" ] || return 1
  local gb
  gb=$(du -sb "$RB_MODEL_DIR" 2>/dev/null | cut -f1)
  awk -v b="${gb:-0}" -v m="$RB_MIN_MODEL_GB" 'BEGIN{exit !(b > m*1e9)}'
}
if step_done download && model_present; then
  echo "SKIP download (checkpointed + present: $(du -sh "$RB_MODEL_DIR" 2>/dev/null | cut -f1))"
elif model_present; then
  echo "MODEL_PRESENT $(du -sh "$RB_MODEL_DIR" 2>/dev/null | cut -f1)"
  mark_done download "already-present"
else
  mark "R1 download (background, parallel with build)"
  HF_CLI=""
  command -v hf >/dev/null 2>&1 && HF_CLI="hf"
  [ -z "$HF_CLI" ] && command -v huggingface-cli >/dev/null 2>&1 && HF_CLI="huggingface-cli"
  [ -n "$HF_CLI" ] || abort DL_TOOL "an HF download CLI exists" "hf or huggingface-cli on PATH" "neither" "pip install -U huggingface_hub, then re-run."
  mkdir -p "$RB_MODEL_DIR"
  nohup "$HF_CLI" download "$RB_BASE_MODEL" --local-dir "$RB_MODEL_DIR" \
    > "$RB_LOG_DIR/dl.log" 2>&1 < /dev/null &
  DL_PID=$!
  echo "DL_PID=$DL_PID (~149 GB, ~20 min at ~183 MB/s)"
  status_line "DOWNLOAD_STARTED pid=$DL_PID repo=$RB_BASE_MODEL"
fi

# =============================================================================
# R2 — BUILD: cuda + flash-attn, NEVER cudnn (-62% decode on V4, session 4).
# arc_build_and_verify takes the binary path from cargo's own artifact stream
# (the stale-binary trap: arc-cli builds `arc`, mistralrs-cli builds
# `mistralrs`; a guessed target/release path once measured an hour-old binary).
# =============================================================================
if step_done build && [ -x "$(cat "$RB_CKPT_DIR/build.bin" 2>/dev/null)" ]; then
  BIN=$(cat "$RB_CKPT_DIR/build.bin")
  echo "SKIP build (checkpointed: $BIN)"
else
  mark "R2 build"
  FAIL_LOG=build.log
  [ -f "$RB_ARC/arc-tools/lib/build_and_verify.sh" ] \
    || abort BUILD_LIB "arc-tools/lib/build_and_verify.sh exists" "present" "missing" "this ref lacks the shared build verifier; check out a ref that has it (it is what prevents measuring a binary the build did not produce)."
  # shellcheck disable=SC1091
  source "$RB_ARC/arc-tools/lib/build_and_verify.sh"
  ( cd "$RB_ARC" && arc_build_and_verify \
      --package mistralrs-cli --bin mistralrs \
      --features "cuda flash-attn" \
      --repo "$RB_ARC" \
      --log "$RB_LOG_DIR/build.log" \
      && printf '%s\n' "$ARC_VERIFIED_BIN" > "$RB_CKPT_DIR/build.bin" )
  rc=$?
  [ "$rc" = "0" ] && BIN=$(cat "$RB_CKPT_DIR/build.bin" 2>/dev/null)
  [ "$rc" = "0" ] && [ -x "${BIN:-/nonexistent}" ] \
    || abort BUILD "arc_build_and_verify(mistralrs-cli -> mistralrs, cuda flash-attn)" "exit 0 + executable path from cargo's artifact stream" "exit $rc" "the build failed or produced no `mistralrs` executable. First errors are in results/build.log; reproduce on CPU for free."
  mark_done build "$BIN"
fi
echo "BIN=$BIN commit=$(git -C "$RB_ARC" rev-parse --short HEAD 2>/dev/null)"

# --- join the download ---
if [ -n "$DL_PID" ]; then
  waited=0
  while kill -0 "$DL_PID" 2>/dev/null; do
    nap 30; waited=$((waited + 30))
    [ "$waited" -ge "$RB_DL_WIRE_S" ] && { kill_pid "$DL_PID"; break; }
    [ "$RB_DRYRUN" = "1" ] && break
  done
fi
model_present \
  || abort MODEL_PRESENT "source checkpoint at $RB_MODEL_DIR" "config.json + model.safetensors.index.json + > ${RB_MIN_MODEL_GB} GB" "$(du -sh "$RB_MODEL_DIR" 2>/dev/null | cut -f1 || echo absent)" "the 149 GB download did not complete inside ${RB_DL_WIRE_S}s (results/dl.log has the tail). If the box's ingress is the bottleneck, that is a bad rental — D4b loop."
step_done download || mark_done download "$(du -sh "$RB_MODEL_DIR" 2>/dev/null | cut -f1)"

# =============================================================================
# R3 — CUDA BEAM PARITY GATE. No pass, no bake. The tests return Ok(())
# SILENTLY when kernels are not compiled in and print 'CUDA not available;
# skipping' when no device answers — both vacuous passes, both aborted on.
# =============================================================================
if step_done parity; then
  echo "SKIP parity (checkpointed)"
else
  mark "R3 beam-parity"
  FAIL_LOG=beam_parity.log
  ( cd "$RB_ARC" && cargo test -p mistralrs-quant --release --features cuda cuda_ -- --nocapture ) \
    > "$RB_LOG_DIR/beam_parity.log" 2>&1
  rc=$?
  cp "$RB_LOG_DIR/beam_parity.log" "$RB_RESULTS/beam_parity.txt" 2>/dev/null
  grep -E "^test result:" "$RB_LOG_DIR/beam_parity.log" | sed 's/^/PARITY /' || true
  grep -q "CUDA not available; skipping" "$RB_LOG_DIR/beam_parity.log" \
    && abort PARITY_VACUOUS "parity tests actually ran on a device" "no 'CUDA not available; skipping'" "the test binary printed it" "the parity tests SKIPPED — their passes mean nothing and a bake on this box is unproven. results/beam_parity.txt"
  [ "$rc" = "0" ] \
    || abort BEAM_PARITY "CUDA beam == CPU beam byte-identical through W=$RB_BAKE_W" "cargo test exit 0" "exit $rc: $(grep -m1 -E '^test .* FAILED' "$RB_LOG_DIR/beam_parity.log" || echo see results/beam_parity.txt)" "the CUDA search does not reproduce the CPU search on this silicon. W=$RB_BAKE_W or no bake — no degraded rung, no exhaustive fallback. Diagnose on CPU: cargo test -p mistralrs-quant qtip::search_bench."
  mark_done parity "exit0"
fi

# =============================================================================
# R4 — THE BAKE.
# Complete artifact -> skip. Partial artifact -> quarantine + restart (the
# pipeline has no mid-model checkpoint; see header). Then launch, verify the
# search provenance, pace-gate, wire.
# =============================================================================
if step_done bake && artifact_complete; then
  echo "SKIP bake (checkpointed + artifact stamp-verified)"
elif artifact_complete; then
  echo "BAKE_ALREADY_COMPLETE (stamp-verified at W=$RB_BAKE_W) — skipping"
  mark_done bake "pre-existing artifact verified"
else
  mark "R4 bake rung=$RB_ISQ_RUNG W=$RB_BAKE_W"
  FAIL_LOG=bake.log
  SHARDS=$(ls "$RB_UQFF_DIR"/*.uqff 2>/dev/null | wc -l | tr -d ' ')
  if [ "${SHARDS:-0}" -ge 1 ]; then
    PARTIAL="$RB_UQFF_DIR.partial.$(date +%s)"
    mv "$RB_UQFF_DIR" "$PARTIAL"
    echo "PARTIAL_ARTIFACT_DETECTED shards=$SHARDS moved_to=$PARTIAL"
    status_line "PARTIAL_RESTART shards=$SHARDS kept=$PARTIAL (pipeline has no mid-model resume; restarting the bake)"
  fi
  mkdir -p "$RB_UQFF_DIR"

  BAKE_ARGS=(quantize text -m "$RB_MODEL_DIR" -a "$RB_ARCH" --isq "$RB_ISQ_RUNG" \
    -o "$RB_UQFF_DIR/" --uqff-base-model "$RB_BASE_MODEL" --uqff-repo-id "$RB_HF_REPO")
  [ -n "$RB_BAKE_DEVICES" ] && BAKE_ARGS+=(--bake-devices "$RB_BAKE_DEVICES")
  # RB_BAKE_ENV is a space-separated KEY=VAL list — geometry knobs from the
  # audit ride here. Word-splitting is the intended behavior.
  # shellcheck disable=SC2086
  nohup env ARC_QTIP_BEAM="$RB_BAKE_W" MISTRALRS_ISQ_SINGLETHREAD=1 $RB_BAKE_ENV \
    "$BIN" "${BAKE_ARGS[@]}" > "$RB_LOG_DIR/bake.log" 2>&1 < /dev/null &
  BAKE_PID=$!
  BAKE_T0=$(date +%s)
  echo "BAKE_PID=$BAKE_PID"
  status_line "BAKE_STARTED pid=$BAKE_PID rung=$RB_ISQ_RUNG W=$RB_BAKE_W env=[$RB_BAKE_ENV]"
  progress_poller "$RB_LOG_DIR/bake.log" "$BAKE_PID" &
  POLLER_PID=$!

  # Search-provenance gate. The qtip2 rung prints a bake header within seconds;
  # gate on it immediately. The qtip2b rung emits NO header (its provenance is
  # the per-layer stamp, verified at R5) — for it, only the greedy line and the
  # CPU-fallback line are gateable this early.
  if [ "$RB_ISQ_RUNG" = "qtip2" ]; then
    HDR=""
    HDR_DEADLINE=180
    [ "$RB_DRYRUN" = "1" ] && HDR_DEADLINE=10
    while [ $(( $(date +%s) - BAKE_T0 )) -lt "$HDR_DEADLINE" ]; do
      HDR=$(grep -m1 "QTIP bake \[" "$RB_LOG_DIR/bake.log" 2>/dev/null)
      [ -n "$HDR" ] && break
      kill -0 "$BAKE_PID" 2>/dev/null || { sleep 1; HDR=$(grep -m1 "QTIP bake \[" "$RB_LOG_DIR/bake.log" 2>/dev/null); break; }
      sleep 2
    done
    [ -n "$HDR" ] || { kill_pid "$BAKE_PID"; abort BAKE_HEADER "bake log contains 'QTIP bake ['" "header within ${HDR_DEADLINE}s" "absent" "this binary cannot prove which search ran (pre-PR#33 GPU path returned before log_bake_header). Refusing an unprovenanced artifact (D4). results/bake.log"; }
    echo "BAKE_HEADER: $HDR"
    case "$HDR" in
      *"search=viterbi-beam(W=$RB_BAKE_W)"*) echo "BAKE_SEARCH_OK W=$RB_BAKE_W" ;;
      *"search=greedy"*) kill_pid "$BAKE_PID"; abort BAKE_GREEDY "bake header search field" "search=viterbi-beam(W=$RB_BAKE_W)" "$HDR" "the bake selected GREEDY — a CODE regression in the D4 ban (QtipRotation::for_mode / greedy_ban_tests.rs), not a box problem. results/bake.log" ;;
      *) kill_pid "$BAKE_PID"; abort BAKE_SEARCH "bake header search field" "search=viterbi-beam(W=$RB_BAKE_W)" "$HDR" "the header names a different search than ARC_QTIP_BEAM=$RB_BAKE_W requested (translate is legal, substitute is not — qtip/mod.rs cuda_search_plan). results/bake.log" ;;
    esac
  else
    echo "BAKE_HEADER: n/a for rung=$RB_ISQ_RUNG — provenance is the per-layer stamp, verified after the bake (R5)"
  fi

  # CPU-fallback gate (the ~11 min/layer crawl). grep -c exits 1 on zero
  # matches and pipefail would misread it — count into a variable.
  nap 60
  FB=$(grep -c "QTIP GPU quantize fallback" "$RB_LOG_DIR/bake.log" 2>/dev/null || true)
  if [ "${FB:-0}" -gt 0 ]; then
    grep -m1 "QTIP GPU quantize fallback" "$RB_LOG_DIR/bake.log"
    kill_pid "$BAKE_PID"
    abort BAKE_FALLBACK "QTIP GPU quantize fallback count" "0" "$FB" "the bake fell back to the CPU pipeline (~11 min/layer). The warn line printed above names the exact condition and IS the deliverable. results/bake.log"
  fi

  # Pace gate at RB_PACE_T1_S, calibrated for 370-376 s/layer (A100 W=256
  # historical; H200 241 s/layer). Not the beam-kernel 42-85 s/layer pace —
  # wrong rung, wrong gate.
  if [ "$RB_DRYRUN" != "1" ]; then
    while [ $(( $(date +%s) - BAKE_T0 )) -lt "$RB_PACE_T1_S" ]; do
      kill -0 "$BAKE_PID" 2>/dev/null || break
      sleep 15
    done
    L1=$(grep -cE "$RB_LAYER_REGEX" "$RB_LOG_DIR/bake.log" 2>/dev/null || true)
    echo "PACE[t=${RB_PACE_T1_S}s]=${L1:-0} layers"
    if kill -0 "$BAKE_PID" 2>/dev/null && [ "${L1:-0}" -lt "$RB_PACE_T1_MIN_LAYERS" ]; then
      kill_pid "$BAKE_PID"
      abort BAKE_PACE "layer lines ('$RB_LAYER_REGEX') at t=${RB_PACE_T1_S}s" ">= $RB_PACE_T1_MIN_LAYERS (370-376 s/layer A100 basis)" "${L1:-0}" "bake pace too slow: either the GPU path is not engaged (results/bake.log) or the box is transfer-starved like s5a (99% util at 132 W). Compare results/box_health.json power against its floor offline."
    fi
  fi

  # Run to completion under the anti-hang wire.
  while kill -0 "$BAKE_PID" 2>/dev/null; do
    nap 30
    if past "$RB_BAKE_WIRE_H"; then
      kill_pid "$BAKE_PID"
      abort BAKE_TRIPWIRE "bake completes before the anti-hang wire" "done by ${RB_BAKE_WIRE_H}h cumulative" "still running at $(elapsed_h)h" "the bake overran its wire; it is hung or the box is starved. Count layers/min from results/bake.log offline."
    fi
    [ "$RB_DRYRUN" = "1" ] && break
  done
  wait "$BAKE_PID" 2>/dev/null
  BAKE_RC=$?
  BAKE_PID=""
  kill_pid "${POLLER_PID:-}" 2; POLLER_PID=""

  BYTES=$(du -sb "$RB_UQFF_DIR" 2>/dev/null | cut -f1)
  SHARDS=$(ls "$RB_UQFF_DIR"/*.uqff 2>/dev/null | wc -l | tr -d ' ')
  echo "BAKE_EXIT rc=$BAKE_RC bytes=${BYTES:-0} shards=${SHARDS:-0} layers=$(grep -cE "$RB_LAYER_REGEX" "$RB_LOG_DIR/bake.log" 2>/dev/null || true)"
  status_line "BAKE_EXIT rc=$BAKE_RC bytes=${BYTES:-0} shards=${SHARDS:-0}"
  if [ "$RB_DRYRUN" != "1" ]; then
    [ "$BAKE_RC" = "0" ] || abort BAKE_EXIT "quantize exit code" "0" "$BAKE_RC" "the bake process failed; last lines of results/bake.log say where."
    [ "${SHARDS:-0}" -ge 1 ] || abort BAKE_SIZE "*.uqff shards in $RB_UQFF_DIR" ">= 1" "0" "the bake exited 0 without writing a shard. results/bake.log"
    awk -v b="${BYTES:-0}" -v lo="$RB_EXPECT_MIN_GB" -v hi="$RB_EXPECT_MAX_GB" 'BEGIN{exit !(b > lo*1e9 && b < hi*1e9)}' \
      || abort BAKE_SIZE "uqff directory size" "${RB_EXPECT_MIN_GB}-${RB_EXPECT_MAX_GB} GB" "$(awk -v b="${BYTES:-0}" 'BEGIN{printf "%.1f GB", b/1e9}') / ${SHARDS:-0} shards" "the artifact is the wrong size — do not ship it. results/bake.log"
  fi
  mark_done bake "rc=$BAKE_RC shards=${SHARDS:-0} bytes=${BYTES:-0}"
fi

# =============================================================================
# R5 — SEARCH-STAMP VERIFICATION (D4 §3: the artifact must be self-describing).
# qtip2b: every layer payload carries [stamp][flags][width]; the reader exits 0
# only if every layer is the right rung + trellis + the expected width. A
# greedy or wrong-width artifact is QUARANTINED, never shipped.
# =============================================================================
if step_done stampverify; then
  echo "SKIP stampverify (checkpointed)"
else
  mark "R5 stamp verify"
  FAIL_LOG=stamp.log
  if [ "$RB_ISQ_RUNG" = "qtip2b" ]; then
    python3 "$Q/read_qtip_stamp.py" --folder "$RB_UQFF_DIR" --expect-width "$RB_BAKE_W" \
      2>&1 | tee "$RB_LOG_DIR/stamp.log"
    rc=${PIPESTATUS[0]}
    cp "$RB_LOG_DIR/stamp.log" "$RB_RESULTS/stamp_verify.txt" 2>/dev/null
    if [ "$rc" != "0" ]; then
      QDIR="$RB_UQFF_DIR.QUARANTINE.$(date +%s)"
      mv "$RB_UQFF_DIR" "$QDIR" 2>/dev/null
      abort STAMP "every layer stamps rung=qtip2b search=trellis width=$RB_BAKE_W" "read_qtip_stamp.py exit 0" "exit $rc (results/stamp_verify.txt names the disagreeing layers)" "the artifact does not prove the search we paid for — a GREEDY or wrong-width layer is in it. Quarantined at $QDIR; do NOT ship or upload it (D4)."
    fi
    echo "STAMP_OK every layer trellis W=$RB_BAKE_W"
  else
    grep -q "search=viterbi-beam(W=$RB_BAKE_W)" "$RB_LOG_DIR/bake.log" 2>/dev/null \
      || abort STAMP "bake header names viterbi-beam(W=$RB_BAKE_W)" "header present in bake.log" "absent" "for rung=$RB_ISQ_RUNG the header is the only provenance and it is missing — treat the artifact as unproven (D4)."
    echo "STAMP_OK via bake header (rung=$RB_ISQ_RUNG)"
  fi
  mark_done stampverify "rung=$RB_ISQ_RUNG W=$RB_BAKE_W"
fi

# =============================================================================
# R6 — SHA256 MANIFEST. Every shard hashed; geometry, commit, GPU and the
# stamp verdict recorded next to the artifact AND in results/.
# =============================================================================
if step_done manifest; then
  echo "SKIP manifest (checkpointed)"
else
  mark "R6 manifest"
  MANIFEST="$RB_UQFF_DIR/arc_rebake_manifest.json"
  SHA_TOOL="sha256sum"
  command -v sha256sum >/dev/null 2>&1 || SHA_TOOL="shasum -a 256"
  {
    printf '{\n'
    printf '  "produced_by": "arc-tools/v4_rebake_driver.sh",\n'
    printf '  "arc_commit": "%s",\n' "$(git -C "$RB_ARC" rev-parse HEAD 2>/dev/null)"
    printf '  "base_model": "%s",\n' "$RB_BASE_MODEL"
    printf '  "isq": "%s",\n  "beam_width": %s,\n  "arch": "%s",\n' "$RB_ISQ_RUNG" "$RB_BAKE_W" "$RB_ARCH"
    printf '  "bake_env": "%s",\n' "$(jstr "$RB_BAKE_ENV")"
    printf '  "gpu": "%s",\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    printf '  "stamp_verified": true,\n'
    printf '  "utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "shards": {\n'
    first=1
    for f in "$RB_UQFF_DIR"/*.uqff; do
      [ -f "$f" ] || continue
      h=$($SHA_TOOL "$f" | awk '{print $1}')
      [ "$first" = "1" ] || printf ',\n'
      printf '    "%s": {"sha256": "%s", "bytes": %s}' \
        "$(basename "$f")" "$h" "$(wc -c < "$f" | tr -d ' ')"
      first=0
    done
    printf '\n  }\n}\n'
  } > "$MANIFEST"
  [ -s "$MANIFEST" ] || abort MANIFEST "manifest written" "non-empty $MANIFEST" "empty/absent" "sha256 hashing failed — check disk state before trusting the artifact."
  cp "$MANIFEST" "$RB_RESULTS/arc_rebake_manifest.json" 2>/dev/null
  echo "MANIFEST: $MANIFEST"
  status_line "MANIFEST_WRITTEN $(grep -c sha256 "$MANIFEST") shards hashed"
  mark_done manifest "$MANIFEST"
fi

# =============================================================================
# R7 — HARVEST + DONE. The results tarball always exists before anyone deletes
# the box; the UQFF itself is harvested per the runbook (scp the artifact tar
# if RB_TAR_ARTIFACT=1, or run the quality gate + measure legs first).
# =============================================================================
mark "R7 harvest"
harvest "rebake complete"
status_line "DRIVER_DONE rung=$RB_ISQ_RUNG W=$RB_BAKE_W"
echo "REBAKE_COMPLETE — next: bash arc-tools/v4_rebake_quality_gate.sh (see V4_REBAKE_RUNBOOK.md)"
echo "Do NOT delete the box before the artifact is off it (harvest-before-delete)."
exit 0
