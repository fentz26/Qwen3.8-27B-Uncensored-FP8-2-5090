#!/usr/bin/env bash
# Profile A — one RTX 5090, one full model + DFlash2 draft.
# Usage: ./serve-single.sh [profile] [gpu-index] [port]
#   ./serve-single.sh 1x5090-fast 0 9000
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

load_profile "${1:-1x5090-fast}"
GPU="${2:-${GPU:-0}}"
PORT="${3:-${PORT:-9000}}"
preflight

mapfile -d '' SPEC < <(spec_args)

set -x
CUDA_VISIBLE_DEVICES="$GPU" \
"$LLAMA_SERVER" \
  -m "$MODEL" \
  ${SPEC[@]+"${SPEC[@]}"} \
  -ngl "${NGL:-all}" \
  -fa "${FLASH_ATTN:-on}" \
  -c "${CTX:-262144}" \
  -ctk "${KV_K:-q4_0}" \
  -ctv "${KV_V:-q4_0}" \
  -b "${BATCH:-2048}" \
  -ub "${UBATCH:-512}" \
  -np "${PARALLEL:-1}" \
  --metrics \
  --host "${HOST:-0.0.0.0}" --port "$PORT" \
  ${EXTRA_ARGS:-}
