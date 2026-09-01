#!/usr/bin/env bash
# start.sh - one-command boot of the GLM-5.3-Flash EXL3 3.5bpw mixed serve.
#
#   ./start.sh                 # defaults: port 8012, 1M ctx, graphs, tools
#   PORT=8013 ./start.sh       # any env below overrides
#   ./start.sh --no-wait       # launch without waiting for /health
#
# Requirements: 2x RTX PRO 6000 (SM120), docker, the artifact at MODEL below,
# image glm53-exl3:dflash2-mixed (see Dockerfile). Boot is ~13 min: 120 shards
# + b12x compile + CUDA graph capture. Health polling prints progress.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- serve configuration (env-overridable) --------------------------------
MODEL="${MODEL:-$HERE/artifact-glm53-k35-mixed}"
IMAGE="${IMAGE:-glm53-exl3:dflash2-mixed}"
CACHE="${CACHE:-nvfp4_ds_mla}"          # calibrated NVFP4 MLA KV (KLD-passing)
DCP="${DCP:-1}"                         # DCP=2 wedges A2A on 2 discrete cards
PROFILE="${PROFILE:-long500k}"          # single-request long-context profile
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
VLLM_EXL3_PREFILL_BLOCK_M="${VLLM_EXL3_PREFILL_BLOCK_M:-64}"  # MUST be 64 or 32
GLM53_CACHE_PATH="${GLM53_CACHE_PATH:-$HOME/.cache/glm53-exl3-k4}"
PORT="${PORT:-8012}"
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-EXL3-3.5bpw}"
NAME="${NAME:-glm53-k35-serve}"
WAIT="${WAIT:-1}"

for arg in "$@"; do
  case "$arg" in
    --no-wait) WAIT=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# --- preflight -------------------------------------------------------------
if [ ! -f "$MODEL/config.json" ]; then
  echo "ERROR: artifact not found at $MODEL (set MODEL=...)" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image $IMAGE missing (build from the repo Dockerfile or set IMAGE=...)" >&2
  exit 1
fi
free_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "${free_mib:-0}" -gt 2000 ]; then
  echo "WARNING: GPUs already hold ${free_mib} MiB; another serve may be running." >&2
  echo "         Run ./stop.sh first for a clean boot." >&2
fi

# --- launch ----------------------------------------------------------------
# serve-k35.sh derives its container name from CACHE/DCP; NAME pins ours.
export MODEL IMAGE CACHE DCP PROFILE MAX_MODEL_LEN VLLM_EXL3_PREFILL_BLOCK_M
export GLM53_CACHE_PATH PORT SERVED_NAME
mkdir -p "$GLM53_CACHE_PATH"
LOG="${LOG:-$HERE/serve-$(date +%Y%m%d-%H%M%S).log}"

echo "booting $SERVED_NAME on :$PORT (ctx $MAX_MODEL_LEN, log $LOG)"
echo "expected: ~13 min (shards + b12x compile + CUDA graphs)"
( cd "$HERE/serve" && exec ./serve-k35.sh ) >"$LOG" 2>&1 &
LAUNCHER=$!

# --- health wait -----------------------------------------------------------
if [ "$WAIT" = "1" ]; then
  start=$(date +%s)
  while true; do
    if ! kill -0 "$LAUNCHER" 2>/dev/null && ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "ERROR: launcher exited before serving; last log lines:" >&2
      tail -20 "$LOG" >&2
      exit 1
    fi
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      break
    fi
    if grep -qE "Traceback|CUDA out of memory|error while" "$LOG" 2>/dev/null; then
      echo "ERROR: boot failed; last log lines:" >&2
      grep -B2 -A12 -m1 "Traceback\|CUDA out of memory\|error while" "$LOG" >&2
      exit 1
    fi
    elapsed=$(( $(date +%s) - start ))
    printf "\r  waiting for /health... %dm%02ds" $((elapsed / 60)) $((elapsed % 60))
    sleep 10
  done
  echo
  echo "READY: http://localhost:$PORT/v1  model: $SERVED_NAME"
  echo "      tools enabled (glm45 parser); verify: curl -s localhost:$PORT/v1/models"
fi
