"""ROCm GPU telemetry for the FluxGym-R HUD."""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GpuStats:
    name: str
    watts: Optional[float]
    mem_used_gb: Optional[float]
    mem_total_gb: Optional[float]
    available: bool = True

    @property
    def mem_pct(self) -> float:
        if not self.mem_used_gb or not self.mem_total_gb or self.mem_total_gb <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.mem_used_gb / self.mem_total_gb) * 100.0))


def _run_rocm_smi() -> str:
    binary = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    result = subprocess.run(
        [binary, "--showproductname", "--showmeminfo", "vram", "--showpower"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return (result.stdout or "") + "\n" + (result.stderr or "")


def _parse_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _device_name(gfx: Optional[str]) -> str:
    name = "GPU"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if name.strip().lower() in {"amd radeon graphics", "radeon graphics"} and gfx:
        return f"{name} · {gfx}"
    if gfx and gfx.lower() not in name.lower():
        return f"{name} · {gfx}" if name == "GPU" else name
    return name


def get_gpu_stats() -> GpuStats:
    try:
        text = _run_rocm_smi()
    except Exception:
        return GpuStats(name="GPU · unavailable", watts=None, mem_used_gb=None, mem_total_gb=None, available=False)

    if not text.strip():
        return GpuStats(name="GPU · unavailable", watts=None, mem_used_gb=None, mem_total_gb=None, available=False)

    gfx_match = re.search(r"GFX Version:\s*(gfx\w+)", text, re.IGNORECASE)
    gfx = gfx_match.group(1) if gfx_match else None

    watts = _parse_float(r"Average Graphics Package Power \(W\):\s*([0-9.]+)", text)
    used_b = _parse_float(r"VRAM Total Used Memory \(B\):\s*([0-9.]+)", text)
    total_b = _parse_float(r"VRAM Total Memory \(B\):\s*([0-9.]+)", text)

    mem_used = round(used_b / (1024**3), 1) if used_b is not None else None
    mem_total = round(total_b / (1024**3), 1) if total_b is not None else None

    return GpuStats(
        name=_device_name(gfx),
        watts=watts,
        mem_used_gb=mem_used,
        mem_total_gb=mem_total,
        available=True,
    )


def render_gpu_hud(stats: Optional[GpuStats] = None) -> str:
    """Inline GPU strip for the training status card (not a floating popup)."""
    stats = stats or get_gpu_stats()
    if not stats.available:
        return """
<div class="train-gpu unavailable">
  <div class="train-gpu-name">GPU · unavailable</div>
</div>
"""
    watts = f"{int(round(stats.watts))} W" if stats.watts is not None else "— W"
    if stats.mem_used_gb is not None and stats.mem_total_gb is not None:
        mem = f"{stats.mem_used_gb:.1f} / {stats.mem_total_gb:.1f} GB"
    else:
        mem = "— / — GB"
    pct = stats.mem_pct
    return f"""
<div class="train-gpu">
  <div class="train-gpu-name">{stats.name}</div>
  <div class="train-gpu-row">
    <span class="train-gpu-watts">{watts}</span>
    <div class="train-gpu-mem-wrap">
      <span class="train-gpu-mem">{mem}</span>
      <div class="train-gpu-bar"><div class="train-gpu-bar-fill" style="width:{pct:.1f}%"></div></div>
    </div>
  </div>
</div>
"""


def poll_gpu_hud() -> str:
    return render_gpu_hud()
