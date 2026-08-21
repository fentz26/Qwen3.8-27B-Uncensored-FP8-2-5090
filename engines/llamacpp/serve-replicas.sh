#!/usr/bin/env bash
# Profiles B and E — N independent replicas, one per GPU. No NCCL, no TP, no
# inter-GPU sync. This is the leading candidate for AGGREGATE throughput on
# multi-GPU boxes because the 27B Q4 target fits on a single 32GB card.
#
# Reported throughput here is AGGREGATE across independent requests. It is NOT
# single-request speed. Never report "one request at N tok/s" from this mode.
#
# Usage: ./serve-replicas.sh [profile] [gpu-list] [base-port]
#   ./serve-replicas.sh 2x5090-replicas 0,1  9000   -> :9000 (GPU0), :9001 (GPU1)
#   ./serve-replicas.sh 4x5090-replicas 0,1,2,3 9000
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

load_profile "${1:-2x5090-replicas}"
IFS=',' read -ra GPU_LIST <<< "${2:-${GPUS:-0,1}}"
BASE_PORT="${3:-${BASE_PORT:-9000}}"
preflight

LOG_DIR="${LOG_DIR:-/tmp/qwen-replicas}"
mkdir -p "$LOG_DIR"
mapfile -d '' SPEC < <(spec_args)

PIDS=()
for i in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$i]}"
  port=$(( BASE_PORT + i ))
  echo "Starting replica $i: GPU $gpu -> :$port"
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$LLAMA_SERVER" \
    -m "$MODEL" \
    ${SPEC[@]+"${SPEC[@]}"} \
    -ngl "${NGL:-all}" \
    -fa "${FLASH_ATTN:-on}" \
    -c "${CTX:-131072}" \
    -ctk "${KV_K:-q4_0}" -ctv "${KV_V:-q4_0}" \
    -b "${BATCH:-2048}" -ub "${UBATCH:-512}" \
    -np "${PARALLEL:-1}" \
    --metrics --host "${HOST:-0.0.0.0}" --port "$port" \
    ${EXTRA_ARGS:-} > "$LOG_DIR/replica-gpu${gpu}-${port}.log" 2>&1 &
  PIDS+=($!)
done

echo "Replica PIDs: ${PIDS[*]}"
echo "Logs: $LOG_DIR"
echo "Point router/router.py at: $(for i in "${!GPU_LIST[@]}"; do printf 'http://127.0.0.1:%s ' $((BASE_PORT+i)); done)"
trap 'echo "stopping..."; kill "${PIDS[@]}" 2>/dev/null' INT TERM
wait
