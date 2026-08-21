#!/usr/bin/env bash
# Profile D — 2x RTX 5090, single model, EXPERIMENTAL tensor split.
#
# Constraints (upstream, verify at runtime — these change):
#   * Flash Attention is REQUIRED.
#   * Quantized KV is NOT supported -> must use f16/bf16/f32 KV. That costs a
#     lot of the context budget this project cares about.
#   * NCCL should be built in (WITH_NCCL=1 ./build.sh).
#   * Auto-fit does not work in this mode.
#
# Consumer RTX 5090 has no NVLink, and CUDA P2P is frequently unavailable.
# On the dual-5090 node originally tested for this repo, P2P reported
# UNAVAILABLE, which makes cross-GPU reduction cost dominate. Run
# scripts/topology.sh first. Do NOT force GGML_CUDA_P2P if it reports
# unsupported.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

load_profile "${1:-2x5090-tensor-experimental}"
PORT="${2:-${PORT:-9000}}"
preflight

case "${KV_K:-f16}" in q*|iq*) echo "ERROR: tensor split does not support quantized KV (KV_K=${KV_K})." >&2; exit 1;; esac
case "${KV_V:-f16}" in q*|iq*) echo "ERROR: tensor split does not support quantized KV (KV_V=${KV_V})." >&2; exit 1;; esac

echo "--- P2P status (tensor split is topology-sensitive) ---"
nvidia-smi topo -p2p r 2>/dev/null || echo "(p2p query unavailable)"

mapfile -d '' SPEC < <(spec_args)

set -x
CUDA_VISIBLE_DEVICES="${GPUS:-0,1}" \
"$LLAMA_SERVER" \
  -m "$MODEL" \
  ${SPEC[@]+"${SPEC[@]}"} \
  -ngl "${NGL:-all}" \
  -sm tensor \
  -ts "${TENSOR_SPLIT:-1,1}" \
  -fa on \
  -c "${CTX:-262144}" \
  -ctk "${KV_K:-f16}" \
  -ctv "${KV_V:-f16}" \
  -b "${BATCH:-2048}" \
  -ub "${UBATCH:-512}" \
  -np "${PARALLEL:-1}" \
  --metrics \
  --host "${HOST:-0.0.0.0}" --port "$PORT" \
  ${EXTRA_ARGS:-}
