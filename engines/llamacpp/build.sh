#!/usr/bin/env bash
# Build llama.cpp for RTX 5090 (Blackwell, SM120) with DFlash2 speculative decoding.
#
# IMPORTANT — read this before building master.
#
# llama.cpp master ALREADY supports DFlash v1: `--spec-type draft-dflash` is a
# valid flag there (alongside draft-simple / draft-mtp / draft-dspark), and
# --spec-draft-n-max / --spec-draft-device / --spec-draft-ngl all exist.
#
# What master does NOT have is **DFlash2** — the local-convolution + candidate
# selector variant used by the DFlash2 checkpoints this repo downloads. That is
# an UNMERGED pull request (verified OPEN 2026-08-21):
#   https://github.com/ggml-org/llama.cpp/pull/27342  (branch `dflash2`)
# It adds new tensors (attn_conv_base/proj, ffn_conv_base/proj,
# selector_predecessor/successor/hidden).
#
# Practical consequence: on master the FLAG PARSES, so failure appears at model
# load or as degraded/incorrect drafting — not as a clean "unknown argument"
# error. Build the PR branch for DFlash2 checkpoints.
#
# Re-check before building: `gh pr view 27342 --repo ggml-org/llama.cpp --json state`
# If it has merged, set LLAMACPP_REF to a master commit instead and record that
# in your benchmark artifact.
set -euo pipefail

LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
# Pin explicitly. Never benchmark "latest" without recording the SHA.
LLAMACPP_PR="${LLAMACPP_PR:-27342}"
LLAMACPP_REF="${LLAMACPP_REF:-pr-${LLAMACPP_PR}}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

if [ ! -d "$LLAMACPP_DIR/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMACPP_DIR"
fi
cd "$LLAMACPP_DIR"

if [ "$LLAMACPP_REF" = "pr-${LLAMACPP_PR}" ]; then
  git fetch origin "pull/${LLAMACPP_PR}/head:pr-${LLAMACPP_PR}" --force
fi
git switch "$LLAMACPP_REF" 2>/dev/null || git checkout "$LLAMACPP_REF"

BUILD_SHA="$(git rev-parse HEAD)"
echo "llama.cpp build SHA: ${BUILD_SHA}"

# SM120 = Blackwell consumer (RTX 5090). Building only this arch keeps compile
# time down; add more archs if you share the binary across machines.
CMAKE_ARGS=(
  -B build -G Ninja
  -DGGML_CUDA=ON
  -DGGML_CUDA_GRAPHS=ON
  -DGGML_CUDA_FA=ON
  -DCMAKE_CUDA_ARCHITECTURES=120
  -DCMAKE_BUILD_TYPE=Release
)
# NCCL is only needed for --split-mode tensor (Profile D). It is pointless
# without working P2P — see docs/multi-gpu.md.
if [ "${WITH_NCCL:-0}" = "1" ]; then
  CMAKE_ARGS+=( -DGGML_CUDA_NCCL=ON )
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build build -j "$JOBS"

# Record what was actually built, for benchmark provenance.
mkdir -p "${BUILD_INFO_DIR:-$LLAMACPP_DIR}"
cat > "${BUILD_INFO_DIR:-$LLAMACPP_DIR}/build-info.json" <<JSON
{
  "engine": "llama.cpp",
  "engine_commit": "${BUILD_SHA}",
  "engine_ref": "${LLAMACPP_REF}",
  "dflash_pr": ${LLAMACPP_PR},
  "cuda_architectures": "120",
  "nccl": ${WITH_NCCL:-0},
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
echo "Wrote build-info.json — attach this to every benchmark result."
