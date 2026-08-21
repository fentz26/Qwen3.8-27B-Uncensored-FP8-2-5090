#!/usr/bin/env bash
# Human-readable topology + an explicit profile recommendation.
#
# Rule (Section 25): never enable P2P or recommend tensor split based on GPU
# COUNT alone. Tensor split is only worth benchmarking with working P2P.
set -euo pipefail

echo "=============== GPUs ==============="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || { echo "nvidia-smi unavailable"; exit 1; }

echo; echo "=============== Interconnect matrix ==============="
nvidia-smi topo -m 2>/dev/null || echo "(unavailable)"

echo; echo "=============== P2P capability ==============="
for mode in r w p; do
  echo "--- topo -p2p $mode ---"
  nvidia-smi topo -p2p "$mode" 2>/dev/null || echo "(unavailable)"
done

GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
P2P_OK="$(nvidia-smi topo -p2p r 2>/dev/null | grep -c 'OK' || echo 0)"

echo; echo "=============== Recommended benchmark order ==============="
case "$GPU_COUNT" in
  1) echo "1 GPU -> profiles/1x5090-fast.env (then -quality, then -256k)";;
  2)
    if [ "${P2P_OK:-0}" -gt 0 ]; then
      echo "2 GPUs WITH P2P:"
      echo "  1. 2x5090-replicas.env        (aggregate throughput)"
      echo "  2. 2x5090-layer.env           (single-request)"
      echo "  3. 2x5090-tensor-experimental.env  (P2P present, worth measuring)"
    else
      echo "2 GPUs, NO working P2P:"
      echo "  1. 2x5090-replicas.env  (strongly favoured)"
      echo "  2. 2x5090-layer.env"
      echo "  3. tensor split: NOT RECOMMENDED without P2P — expect reduction cost"
      echo "     to dominate. Benchmark only to document the negative result."
    fi;;
  4) echo "4 GPUs -> 4x5090-replicas.env first; 4x5090-2x2.env only if one card"
     echo "         cannot hold your required quant/KV/context combination.";;
  *) echo "$GPU_COUNT GPUs — no preset ordering; start with independent replicas.";;
esac
echo
echo "Always compare multi-GPU results against the single-GPU baseline."
echo "A multi-GPU profile that loses to 1 GPU is a valid, publishable result."
