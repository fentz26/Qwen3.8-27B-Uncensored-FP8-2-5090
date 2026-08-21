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
  # NOTE: --spec-type EXISTS on llama.cpp master (DFlash v1), so its presence
  # does NOT prove DFlash2 support. DFlash2 is PR #27342 and adds new tensors;
  # on master a DFlash2 checkpoint fails at model load, not at flag parse.
  # This check therefore catches only the crude case (a binary with no
  # speculative support at all) and warns otherwise. It cannot fully verify
  # DFlash2 — record your build SHA and confirm the draft actually loads.
  if [ -n "${DRAFT_MODEL:-}" ]; then
    if ! "$LLAMA_SERVER" --help 2>&1 | grep -q -- '--spec-type'; then
      echo "ERROR: this llama-server has no --spec-type flag at all." >&2
      echo "Build via engines/llamacpp/build.sh." >&2
      exit 1
    fi
    if [ "${SPEC_TYPE:-draft-dflash}" = "draft-dflash" ] && [ "${SKIP_DFLASH2_WARN:-0}" != "1" ]; then
      echo "NOTE: using a DFlash2 draft requires llama.cpp PR #27342." >&2
      echo "      --spec-type alone does not prove DFlash2 support (master has DFlash v1)." >&2
      echo "      If the draft fails to load, that is the cause. SKIP_DFLASH2_WARN=1 to silence." >&2
    fi
  fi
}

# Build the speculation args only when a draft is configured.
spec_args() {
  [ -n "${DRAFT_MODEL:-}" ] || return 0
  printf '%s\0' -md "$DRAFT_MODEL" \
    --spec-type "${SPEC_TYPE:-draft-dflash}" \
    --spec-draft-n-max "${SPEC_N_MAX:-7}"
}
