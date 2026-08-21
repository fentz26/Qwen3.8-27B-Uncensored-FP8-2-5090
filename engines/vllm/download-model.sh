#!/bin/bash
# Downloads the model. Model is gated: the HF account behind $HF_TOKEN must
# click "Agree and access repository" at
# https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8 before this
# succeeds.
#
# If huggingface.co isn't reachable from your network, set
# HF_ENDPOINT=https://hf-mirror.com (a common mirror) before running this.
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

/venv/main/bin/hf auth login --token "${HF_TOKEN:?set HF_TOKEN in the environment, do not hardcode it here}"

nohup /venv/main/bin/hf download orcarouter/Qwen3.8-27B-Uncensored-FP8 \
  --local-dir /workspace/models/Qwen3.8-27B-Uncensored-FP8 \
  > /tmp/model_dl.log 2>&1 &

echo "DL_PID:$!"
