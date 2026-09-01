#!/usr/bin/env bash
# status.sh - one-glance state of the 3.5bpw serve.
set -uo pipefail

IMAGE="${IMAGE:-glm53-exl3:dflash2-mixed}"
PORT="${PORT:-8012}"

echo "== container =="
docker ps -a --filter "ancestor=$IMAGE" --format "{{.Names}}  {{.Status}}  {{.Ports}}" || true

echo "== endpoint (:$PORT) =="
if curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null; then
  echo "health: OK"
  curl -sf -m 5 "http://localhost:$PORT/v1/models" | python3 -c '
import json, sys
data = json.load(sys.stdin)["data"]
print("models:", ", ".join(m["id"] for m in data))
' 2>/dev/null || true
else
  echo "health: DOWN"
fi

echo "== GPUs =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

LOG=$(ls -t "$(dirname "$0")"/serve-*.log 2>/dev/null | head -1)
if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
  echo "== last boot log: $LOG =="
  grep -E "Application startup complete|Capturing CUDA graph \(FULL\).*100%|GPU KV cache size" "$LOG" | tail -3 || true
  tail -1 "$LOG" | cut -c1-160
fi
