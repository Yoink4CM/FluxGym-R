#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

if [[ ! -f env/bin/activate ]]; then
  echo "error: env/ not found. Run ./install.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source env/bin/activate

BACKEND="cpu"
if [[ -f env/.fluxgym-gpu-backend ]]; then
  BACKEND="$(tr -d '[:space:]' < env/.fluxgym-gpu-backend)"
fi

export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"

case "$BACKEND" in
  rocm)
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
    if [[ -f env/.fluxgym-rocm-index ]]; then
      echo "ROCm PyTorch index: $(tr -d '[:space:]' < env/.fluxgym-rocm-index)"
    fi
    ;;
  cuda)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    ;;
esac

exec python app.py
