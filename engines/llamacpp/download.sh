#!/usr/bin/env bash
# Download GGUF target + DFlash2 draft models.
#
# TRACK A (reference / performance ceiling): stock Qwen3.8 GGUF.
# TRACK B (this repo's actual uncensored target): see docs/model-lineage.md
#   before substituting one for the other. They are NOT interchangeable and
#   a stock GGUF must never be labelled "uncensored".
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
TRACK="${TRACK:-A}"
QUANT="${QUANT:-UD-Q4_K_XL}"
DRAFT_QUANT="${DRAFT_QUANT:-Q4_K_M}"

mkdir -p "$MODEL_DIR"
command -v hf >/dev/null || { echo "install: pip install -U huggingface_hub"; exit 1; }

case "$TRACK" in
  A)
    # Verified present 2026-08-20: Qwen3.8-27B-UD-Q4_K_XL.gguf is 17,559,178,144 B
    TARGET_REPO="unsloth/Qwen3.8-27B-GGUF"
    TARGET_FILE="Qwen3.8-27B-${QUANT}.gguf"
    ;;
  B)
    # orcarouter publishes its own GGUF of the abliterated model — no FP8
    # requantization needed. NOTE: this repo is GATED; accept its terms on the
    # model page with the account behind $HF_TOKEN first.
    TARGET_REPO="orcarouter/Qwen3.8-27B-Uncensored-GGUF"
    TARGET_FILE="${TARGET_FILE:?set TARGET_FILE — run: hf download --repo-type model $TARGET_REPO --  to list, quant names differ from Track A}"
    ;;
  *) echo "TRACK must be A or B"; exit 1 ;;
esac

echo "Downloading target: ${TARGET_REPO} :: ${TARGET_FILE}"
hf download "$TARGET_REPO" "$TARGET_FILE" --local-dir "$MODEL_DIR"

# DFlash2 draft. Canonical: incoai/. z-lab/ is an identical mirror.
# The draft is trained against STOCK Qwen3.8. Cross-target acceptance on an
# abliterated target is an OPEN QUESTION — see docs/dflash.md.
DRAFT_REPO="${DRAFT_REPO:-incoai/Qwen3.8-27B-DFlash2-GGUF}"
DRAFT_FILE="Qwen3.8-27B-DFlash2-${DRAFT_QUANT}.gguf"
echo "Downloading draft: ${DRAFT_REPO} :: ${DRAFT_FILE}"
hf download "$DRAFT_REPO" "$DRAFT_FILE" --local-dir "$MODEL_DIR"

echo
echo "Recording SHA256 for benchmark provenance (may take a minute)..."
for f in "$MODEL_DIR/$TARGET_FILE" "$MODEL_DIR/$DRAFT_FILE"; do
  [ -f "$f" ] && echo "$(sha256sum "$f" 2>/dev/null || shasum -a 256 "$f")"
done
