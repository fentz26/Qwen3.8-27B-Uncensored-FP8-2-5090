#!/bin/bash
# Final, validated vLLM launch command for Qwen3.8-27B-Uncensored-FP8 on
# 2x RTX 5090. Run on the host that has the GPUs (after model download, see
# README.md). Exact flags are load-bearing — see README.md's "vLLM flags"
# table and "Numbers" section before changing any of them.
#
# `xxhash` must be installed first: /venv/main/bin/pip install xxhash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

nohup /venv/main/bin/vllm serve /workspace/models/Qwen3.8-27B-Uncensored-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --dtype auto \
  --served-model-name Qwen3.8-27B-Uncensored-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.96 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 --port 8000 \
  > /tmp/vllm_serve.log 2>&1 &

echo "SERVE_PID:$!"

# NOT used in the end (deliberately dropped, see README "MTP" section):
#   --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'
# The checkpoint does ship real MTP weights and the method works (+56-60%
# decode speed, verified byte-identical output under greedy decoding), but
# combining it with --kv-cache-dtype fp8 regresses throughput below either
# alone (49 tok/s vs 76 tok/s fp8-only / 88-91 tok/s MTP-only-bf16), so pick
# one. This repo's default keeps fp8 + prefix caching.
