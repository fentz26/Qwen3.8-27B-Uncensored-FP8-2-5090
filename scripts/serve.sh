#!/bin/bash
# vLLM launch command for Qwen3.8-27B-Uncensored-FP8 on 2x RTX 5090 (the target hardware).
# Run on the instance itself (after model download, see README.md).
set -euo pipefail

export HF_HOME=/workspace/.hf_home

nohup /venv/main/bin/vllm serve /workspace/models/Qwen3.8-27B-Uncensored-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --dtype auto \
  --served-model-name Qwen3.8-27B-Uncensored-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.96 \
  --host 0.0.0.0 --port 8000 \
  > /tmp/vllm_serve.log 2>&1 &

echo "SERVE_PID:$!"
