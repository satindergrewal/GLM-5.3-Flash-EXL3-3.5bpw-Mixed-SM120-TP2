#!/usr/bin/env bash
# stop.sh - clean shutdown of the 3.5bpw serve, including the two failure
# modes plain `docker rm` misses on this stack:
#   1. serve-k35.sh derives the container name from CACHE/DCP, so a
#      hardcoded name can silently match nothing.
#   2. the vLLM process tree can survive `docker rm -f` and keep holding
#      ~90GB per GPU; the survivors must be swept and VRAM verified at 0.
set -uo pipefail

IMAGE="${IMAGE:-glm53-exl3:dflash2-mixed}"
PORT="${PORT:-8012}"

echo "[1/3] stopping containers from image $IMAGE ..."
mapfile -t CIDS < <(docker ps -aq --filter "ancestor=$IMAGE")
if [ "${#CIDS[@]}" -gt 0 ]; then
  docker rm -f "${CIDS[@]}" 2>/dev/null || true
else
  echo "      no containers from $IMAGE; sweeping by port $PORT pattern too"
  docker rm -f "$(docker ps -aq --filter publish="$PORT")" 2>/dev/null || true
fi

echo "[2/3] sweeping surviving engine processes (docker rm can miss the tree) ..."
PIDS=$(pgrep -f "vllm serve /model" || true)
if [ -n "${PIDS:-}" ]; then
  echo "      killing: $PIDS"
  kill -TERM $PIDS 2>/dev/null || true
  sleep 5
  PIDS2=$(pgrep -f "vllm serve /model" || true)
  [ -n "${PIDS2:-}" ] && kill -KILL $PIDS2 2>/dev/null || true
fi

echo "[3/3] verifying VRAM is actually free ..."
sleep 3
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$used" -gt 2000 ]; then
  echo "WARNING: GPUs still hold ${used} MiB. Check: nvidia-smi"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi
echo "stopped clean; ${used} MiB in use."
