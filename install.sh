#!/usr/bin/env bash
# FluxGym installer: auto-detects AMD (ROCm) vs NVIDIA (CUDA) and installs matching PyTorch first.
# Override with: FORCE_GPU=rocm|cuda|cpu ./install.sh
# Optional: CUDA_TORCH_INDEX=https://download.pytorch.org/whl/cu126 ./install.sh
# Optional: ROCM_VERSION=10|7.2 or ROCM_TORCH_INDEX=https://download.pytorch.org/whl/... ./install.sh
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-env}"
DEFAULT_ROCM72_TORCH_INDEX="${DEFAULT_ROCM72_TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
DEFAULT_ROCM10_TORCH_INDEX="${DEFAULT_ROCM10_TORCH_INDEX:-https://download.pytorch.org/whl/nightly/rocm10.0}"
CPU_TORCH_INDEX="${CPU_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
# Used only when CUDA_TORCH_INDEX is unset and detection cannot run
DEFAULT_CUDA_TORCH_INDEX="${DEFAULT_CUDA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found. On Ubuntu 24.04 install: sudo apt install python3 python3-venv python3-pip git" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1; then
  echo "error: python venv module is missing. On Ubuntu 24.04 install: sudo apt install python3-venv" >&2
  exit 1
fi

detect_gpu() {
  local forced="${FORCE_GPU:-}"
  if [[ -n "$forced" ]]; then
    case "$forced" in
      rocm|cuda|cpu) echo "$forced"; return ;;
      amd) echo rocm; return ;;
      nvidia) echo cuda; return ;;
      *)
        echo "error: FORCE_GPU must be rocm, cuda, cpu (or amd/nvidia)" >&2
        exit 1
        ;;
    esac
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo cuda
    return
  fi

  if [[ -e /dev/kfd ]] || command -v rocm-smi >/dev/null 2>&1; then
    echo rocm
    return
  fi

  if command -v lspci >/dev/null 2>&1; then
    local pci
    pci="$(lspci 2>/dev/null || true)"
    if echo "$pci" | grep -qiE '(VGA|3D|Display).*NVIDIA|NVIDIA.*(VGA|3D|Controller)'; then
      echo cuda
      return
    fi
    if echo "$pci" | grep -qi NVIDIA; then
      echo cuda
      return
    fi
    if echo "$pci" | grep -qiE '(VGA|3D|Display).*(AMD|ATI)|AMD/ATI'; then
      echo rocm
      return
    fi
    if echo "$pci" | grep -qiE 'Advanced Micro Devices|AMD/ATI'; then
      echo rocm
      return
    fi
  fi

  echo cpu
}

# Pick the newest PyTorch CUDA wheel the installed NVIDIA driver can run.
# RTX 50 / Blackwell needs cu128. Older drivers fall back to cu126 or cu124.
select_cuda_torch_index() {
  if [[ -n "${CUDA_TORCH_INDEX:-}" ]]; then
    echo "$CUDA_TORCH_INDEX"
    return
  fi

  local gpu_name="" driver_cuda="" major=0 minor=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr -d '\r' || true)"
    driver_cuda="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | awk '{print $3}' || true)"
  fi

  if [[ -n "$driver_cuda" ]]; then
    major="${driver_cuda%%.*}"
    minor="${driver_cuda#*.}"
    minor="${minor%%.*}"
  fi

  # Blackwell / RTX 50-series: require cu128
  if echo "$gpu_name" | grep -qiE 'RTX[[:space:]]*50[0-9]{2}|Blackwell'; then
    if [[ "$major" -gt 12 ]] || { [[ "$major" -eq 12 ]] && [[ "$minor" -ge 8 ]]; }; then
      echo "https://download.pytorch.org/whl/cu128"
      return
    fi
    echo "warning: GPU '$gpu_name' typically needs CUDA 12.8+ driver support (detected driver CUDA ${driver_cuda:-unknown}). Installing cu128 anyway." >&2
    echo "https://download.pytorch.org/whl/cu128"
    return
  fi

  if [[ "$major" -gt 12 ]] || { [[ "$major" -eq 12 ]] && [[ "$minor" -ge 8 ]]; }; then
    echo "https://download.pytorch.org/whl/cu128"
    return
  fi
  if [[ "$major" -eq 12 ]] && [[ "$minor" -ge 6 ]]; then
    echo "https://download.pytorch.org/whl/cu126"
    return
  fi
  if [[ "$major" -eq 12 ]] && [[ "$minor" -ge 4 ]]; then
    echo "https://download.pytorch.org/whl/cu124"
    return
  fi
  if [[ "$major" -eq 12 ]] && [[ "$minor" -ge 1 ]]; then
    echo "https://download.pytorch.org/whl/cu121"
    return
  fi
  if [[ "$major" -eq 11 ]]; then
    echo "https://download.pytorch.org/whl/cu118"
    return
  fi

  echo "warning: could not detect NVIDIA driver CUDA version; defaulting to cu128. Override with CUDA_TORCH_INDEX=..." >&2
  echo "$DEFAULT_CUDA_TORCH_INDEX"
}

# Return dotted ROCm version string (e.g. 7.2.0 or 10.0.0), or empty if unknown.
detect_rocm_version() {
  local ver=""
  if [[ -f /opt/rocm/.info/version ]]; then
    ver="$(tr -d '[:space:]' < /opt/rocm/.info/version || true)"
  fi
  if [[ -z "$ver" ]] && [[ -f /opt/rocm/.info/version-dev ]]; then
    ver="$(tr -d '[:space:]' < /opt/rocm/.info/version-dev || true)"
  fi
  if [[ -z "$ver" ]] && command -v rocminfo >/dev/null 2>&1; then
    ver="$(rocminfo 2>/dev/null | grep -oE 'ROCm[[:space:]]+[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' || true)"
  fi
  if [[ -z "$ver" ]] && command -v rocm-smi >/dev/null 2>&1; then
    ver="$(rocm-smi --showdriverversion 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
  fi
  echo "$ver"
}

# Pick PyTorch ROCm wheel index: stable 7.2 by default, nightly rocm10.0 for ROCm 10+.
select_rocm_torch_index() {
  if [[ -n "${ROCM_TORCH_INDEX:-}" ]]; then
    echo "$ROCM_TORCH_INDEX"
    return
  fi

  local forced="${ROCM_VERSION:-}"
  local detected=""
  local major=0

  if [[ -n "$forced" ]]; then
    case "$forced" in
      10|10.*|rocm10|rocm10.*)
        echo "$DEFAULT_ROCM10_TORCH_INDEX"
        return
        ;;
      7.2|7.2.*|rocm7.2|rocm7.2.*)
        echo "$DEFAULT_ROCM72_TORCH_INDEX"
        return
        ;;
      7|7.0|7.1|7.0.*|7.1.*)
        echo "warning: ROCM_VERSION=$forced mapped to stable rocm7.2 PyTorch wheels." >&2
        echo "$DEFAULT_ROCM72_TORCH_INDEX"
        return
        ;;
      *)
        echo "error: ROCM_VERSION must be 10, 7.2 (or similar). Got: $forced" >&2
        exit 1
        ;;
    esac
  fi

  detected="$(detect_rocm_version)"
  if [[ -n "$detected" ]]; then
    major="${detected%%.*}"
    echo "Detected ROCm version: $detected" >&2
    if [[ "$major" =~ ^[0-9]+$ ]] && [[ "$major" -ge 10 ]]; then
      echo "Using PyTorch nightly ROCm 10 wheels (stable rocm10.0 index not available yet)." >&2
      echo "$DEFAULT_ROCM10_TORCH_INDEX"
      return
    fi
  else
    echo "warning: could not detect ROCm version; defaulting to stable rocm7.2. Override with ROCM_VERSION=10 or ROCM_TORCH_INDEX=..." >&2
  fi

  echo "$DEFAULT_ROCM72_TORCH_INDEX"
}

GPU_BACKEND="$(detect_gpu)"
echo "Detected GPU backend: $GPU_BACKEND"

case "$GPU_BACKEND" in
  rocm)
    TORCH_INDEX="$(select_rocm_torch_index)"
    echo "Selected ROCm PyTorch index: $TORCH_INDEX"
    ;;
  cuda)
    TORCH_INDEX="$(select_cuda_torch_index)"
    echo "Selected CUDA PyTorch index: $TORCH_INDEX"
    ;;
  cpu)
    TORCH_INDEX="$CPU_TORCH_INDEX"
    ;;
esac

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel setuptools

echo "Installing PyTorch first from $TORCH_INDEX (avoids wrong CUDA/ROCm wheel later)"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

if [[ ! -d sd-scripts ]]; then
  echo "sd-scripts not found; cloning kohya-ss/sd-scripts (sd3 branch)"
  git clone -b sd3 https://github.com/kohya-ss/sd-scripts
fi

echo "Pinning opencv-python binary wheel (avoids Python 3.12 source build backtrack)"
pip install "opencv-python==4.10.0.84" --only-binary=:all:

echo "Installing sd-scripts requirements"
(
  cd sd-scripts
  pip install -r requirements.txt --prefer-binary
)

echo "Installing FluxGym requirements"
pip install -r requirements.txt --prefer-binary
pip install -U bitsandbytes hf-xet

# Re-assert opencv pin in case a dependency tried to change it
pip install "opencv-python==4.10.0.84" --only-binary=:all:

install_torch() {
  pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" --force-reinstall
}

verify_torch() {
  GPU_BACKEND="$GPU_BACKEND" python - <<'PY'
import os
import torch

backend = os.environ["GPU_BACKEND"]
version = torch.__version__.lower()
available = torch.cuda.is_available()
print(f"torch {torch.__version__}")
print(f"cuda_available {available}")
if available:
    props = torch.cuda.get_device_properties(0)
    print(f"device {torch.cuda.get_device_name(0)}")
    print(f"vram_gb {round(props.total_memory / 1024**3, 1)}")

ok = False
if backend == "rocm":
    ok = "rocm" in version and available
elif backend == "cuda":
    ok = available and "rocm" not in version
elif backend == "cpu":
    ok = True
else:
    ok = False

if not ok:
    raise SystemExit(1)
PY
}

if ! verify_torch; then
  echo "PyTorch check failed for backend=$GPU_BACKEND; reinstalling from $TORCH_INDEX"
  install_torch
  verify_torch
fi

# Remember backend for app-launch.sh
mkdir -p "$VENV_DIR"
echo "$GPU_BACKEND" > "$VENV_DIR/.fluxgym-gpu-backend"
if [[ "$GPU_BACKEND" == "rocm" ]]; then
  echo "$TORCH_INDEX" > "$VENV_DIR/.fluxgym-rocm-index"
fi

echo
echo "Install complete (backend=$GPU_BACKEND). Launch with: ./app-launch.sh"
if [[ "$GPU_BACKEND" == "rocm" ]]; then
  echo "On AMD, torch.cuda.is_available() is expected to be True (ROCm uses the CUDA API)."
  echo "Used PyTorch index: $TORCH_INDEX"
  echo "Override anytime with ROCM_VERSION=10 ./install.sh or ROCM_TORCH_INDEX=https://download.pytorch.org/whl/... ./install.sh"
elif [[ "$GPU_BACKEND" == "cuda" ]]; then
  echo "On NVIDIA, torch.cuda.is_available() should be True."
  echo "Used PyTorch index: $TORCH_INDEX"
  echo "Override anytime with CUDA_TORCH_INDEX=https://download.pytorch.org/whl/cuXXX ./install.sh"
fi
