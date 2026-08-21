#!/usr/bin/env bash
# Emit a machine-readable hardware fingerprint. Every benchmark artifact must
# embed this — results are meaningless without the topology they ran on.
# Usage: ./detect-hardware.sh > ../results/<host>-hardware.json
set -euo pipefail

q() { nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//;s/"/\\"/g'; }

GPU_NAME="$(q name)"
GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
VRAM="$(q memory.total)"
DRIVER="$(q driver_version)"
CUDA="$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
[ -n "$CUDA" ] || CUDA="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3)"

# P2P matters enormously for tensor split. Absent NVLink on consumer 5090s,
# this is frequently unavailable — which is a REASON NOT to use Profile D.
P2P_READ="$(nvidia-smi topo -p2p r 2>/dev/null | grep -c 'OK' || echo 0)"
P2P_SUPPORTED=false
[ "${P2P_READ:-0}" -gt 0 ] && P2P_SUPPORTED=true

cat <<JSON
{
  "gpu": "${GPU_NAME:-unknown}",
  "gpu_count": ${GPU_COUNT:-0},
  "vram_per_gpu": "${VRAM:-unknown}",
  "driver": "${DRIVER:-unknown}",
  "cuda": "${CUDA:-unknown}",
  "p2p_supported": ${P2P_SUPPORTED},
  "host": "$(hostname)",
  "kernel": "$(uname -r)",
  "cpu": "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//' || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)",
  "captured_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
