#!/usr/bin/env bash
# Shared helpers. Sourced by the serve-*.sh scripts.
set -euo pipefail

LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMACPP_DIR/build/bin/llama-server}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

load_profile() {
  local p="${1:?usage: load_profile <profile-name-or-path>}"
  local f="$p"
  [ -f "$f" ] || f="$REPO_ROOT/profiles/${p}"
  [ -f "$f" ] || f="$REPO_ROOT/profiles/${p}.env"
  [ -f "$f" ] || { echo "profile not found: $p" >&2; exit 1; }
  # shellcheck disable=SC1090
  set -a; . "$f"; set +a
  echo "Loaded profile: $f"
}

preflight() {
  [ -x "$LLAMA_SERVER" ] || {
    echo "llama-server not found at $LLAMA_SERVER — run engines/llamacpp/build.sh" >&2; exit 1; }
  [ -f "${MODEL:?MODEL unset}" ] || { echo "MODEL not found: $MODEL" >&2; exit 1; }
  if [ -n "${DRAFT_MODEL:-}" ] && [ ! -f "$DRAFT_MODEL" ]; then
    echo "DRAFT_MODEL not found: $DRAFT_MODEL" >&2; exit 1
  fi
  # DFlash lives in an unmerged PR; fail loudly rather than mid-benchmark.
  if [ -n "${DRAFT_MODEL:-}" ] && ! "$LLAMA_SERVER" --help 2>&1 | grep -q -- '--spec-type'; then
    echo "ERROR: this llama-server has no --spec-type flag." >&2
    echo "DFlash2 requires PR #27342 (unmerged). See engines/llamacpp/build.sh." >&2
    exit 1
  fi
}

# Build the speculation args only when a draft is configured.
spec_args() {
  [ -n "${DRAFT_MODEL:-}" ] || return 0
  printf '%s\0' -md "$DRAFT_MODEL" \
    --spec-type "${SPEC_TYPE:-draft-dflash}" \
    --spec-draft-n-max "${SPEC_N_MAX:-7}"
}
