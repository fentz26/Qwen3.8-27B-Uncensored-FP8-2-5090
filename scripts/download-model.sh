#!/bin/bash
# Download the model on the instance. huggingface.co was blocked on this
# instance's network (China route); hf-mirror.com worked as a fallback.
# Model is gated: the HF account behind $HF_TOKEN must click "Agree and
# access repository" at https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8
# before this succeeds.
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/workspace/.hf_home

/venv/main/bin/hf auth login --token "${HF_TOKEN:?set HF_TOKEN in the environment, do not hardcode it here}"

nohup /venv/main/bin/hf download orcarouter/Qwen3.8-27B-Uncensored-FP8 \
  --local-dir /workspace/models/Qwen3.8-27B-Uncensored-FP8 \
  > /tmp/model_dl.log 2>&1 &

echo "DL_PID:$!"
