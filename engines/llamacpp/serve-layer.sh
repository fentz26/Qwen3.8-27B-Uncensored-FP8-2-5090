#!/usr/bin/env bash
# Profile C — 2x RTX 5090, single model, pipeline/layer split.
#
# Layer split has far lower interconnect demand than tensor split and keeps
# quantized KV working. But the 27B Q4 target already fits on ONE 32GB card,
# so splitting may add overhead without improving decode. Measure against
# Profile A before recommending it — see docs/multi-gpu.md.
#
# TENSOR_SPLIT bias: if the DFlash draft is pinned to GPU0, GPU0 also carries
# draft weights + draft KV + feature-injection buffers, so an even 1,1 target
# split is NOT an even runtime load. Sweep 1,1 / 0.9,1.1 / 0.8,1.2.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

load_profile "${1:-2x5090-layer}"
PORT="${2:-${PORT:-9000}}"
preflight

mapfile -d '' SPEC < <(spec_args)

set -x
CUDA_VISIBLE_DEVICES="${GPUS:-0,1}" \
"$LLAMA_SERVER" \
  -m "$MODEL" \
  ${SPEC[@]+"${SPEC[@]}"} \
  ${DRAFT_MODEL:+--spec-draft-device "${SPEC_DRAFT_DEVICE:-CUDA0}"} \
  ${DRAFT_MODEL:+--spec-draft-ngl "${SPEC_DRAFT_NGL:-all}"} \
  -ngl "${NGL:-all}" \
  -sm layer \
  -ts "${TENSOR_SPLIT:-1,1}" \
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
