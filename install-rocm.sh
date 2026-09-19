#!/usr/bin/env bash
# Compatibility wrapper: force ROCm install.
# Usage:
#   ./install-rocm.sh           # auto-detect ROCm version (default PyTorch rocm7.2)
#   ./install-rocm.sh 10        # PyTorch nightly rocm10.0 wheels
#   ./install-rocm.sh 7.2       # stable rocm7.2 wheels
set -euo pipefail
cd "$(dirname "$0")"
export FORCE_GPU=rocm

if [[ $# -ge 1 ]]; then
  case "$1" in
    10|10.*|rocm10|rocm10.*)
      export ROCM_VERSION=10
      shift
      ;;
    7.2|7.2.*|rocm7.2|rocm7.2.*)
      export ROCM_VERSION=7.2
      shift
      ;;
    7|7.0|7.1|7.0.*|7.1.*)
      export ROCM_VERSION=7.2
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [10|7.2]"
      exit 0
      ;;
    *)
      echo "error: unknown ROCm version '$1' (use 10 or 7.2)" >&2
      exit 1
      ;;
  esac
fi

exec ./install.sh "$@"
