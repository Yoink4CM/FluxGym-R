"""Training progress parsing and status card rendering for FluxGym-R."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional


TQDM_STEPS_RE = re.compile(
    r"steps:\s*.*?(\d+)\s*/\s*(\d+)\s*\["
    r"(?:([\d:]+))?"
    r"(?:<([\d:]+))?",
    re.IGNORECASE,
)
AVR_LOSS_RE = re.compile(r"avr_loss\s*=\s*([0-9.eE+-]+)")
EPOCH_RE = re.compile(r"(?:^|\s)epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
TOTAL_STEPS_RE = re.compile(
    r"(?:total optimization steps|指定エポックまでのステップ数|steps for \d+ epochs is)[^\d]*(\d+)",
    re.IGNORECASE,
)
SAMPLE_RE = re.compile(r"generating sample images", re.IGNORECASE)
SAVING_RE = re.compile(r"\bsaving\b|\bsave (?:model|state|checkpoint)\b", re.IGNORECASE)
FP8_START_RE = re.compile(r"Cast FLUX model to fp8", re.IGNORECASE)
FP8_PROGRESS_RE = re.compile(r"fp8_cast:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)

# Prefer specific, actionable failure lines over generic traceback noise.
ERROR_LINE_RE = re.compile(
    r"(?:"
    r"^(?:[a-zA-Z_][\w.]*\.)*(?:Error|Exception|Fatal|panic)\s*:\s*.+"
    r"|^(?:RuntimeError|ValueError|TypeError|OSError|FileNotFoundError|KeyError|"
    r"AssertionError|ImportError|ModuleNotFoundError|OutOfMemoryError)\b.+"
    r"|CUDA out of memory|out of memory|OOM"
    r"|HIP error|ROCm|MIOpen"
    r"|No such file or directory"
    r"|Permission denied"
    r"|killed|SIGKILL|SIGSEGV|Segmentation fault"
    r"|Traceback \(most recent call last\):"
    r")",
    re.IGNORECASE,
)
TRACEBACK_HEADER_RE = re.compile(r"^Traceback \(most recent call last\):", re.IGNORECASE)
EXCEPTION_TAIL_RE = re.compile(
    r"^((?:[a-zA-Z_][\w.]*)?(?:Error|Exception|Fatal)):\s*(.+)$"
)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def format_bytes(n: Optional[float]) -> str:
    if n is None or n < 0:
        return "—"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return "—"


def _parse_tqdm_time(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    token = token.strip().rstrip("]")
    parts = token.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return float(parts[0])
    return None


@dataclass
class TrainProgress:
    phase: str = "Idle"
    mode: str = "idle"  # idle | download | fp8 | training
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    max_epochs: int = 0
    loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    started_at: float = field(default_factory=time.time)
    finished: bool = False
    success: bool = False
    debug_lines: List[str] = field(default_factory=list)
    detail: str = ""
    task_current: float = 0.0
    task_total: float = 0.0
    task_unit: str = ""
    indeterminate: bool = False
    _step_times: List[float] = field(default_factory=list)
    _last_step_wall: Optional[float] = None
    _phase_started_at: float = field(default_factory=time.time)

    def append_debug(self, line: str, limit: int = 200) -> None:
        line = line.rstrip("\n")
        if not line.strip():
            return
        self.debug_lines.append(line)
        if len(self.debug_lines) > limit:
            self.debug_lines = self.debug_lines[-limit:]

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    @property
    def debug_text(self) -> str:
        return "\n".join(self.debug_lines)

    @property
    def percent(self) -> float:
        if self.indeterminate:
            return -1.0
        if self.mode == "training":
            total = self.total_steps or 0
            step = self.step or 0
            if total <= 0:
                return 0.0
            return max(0.0, min(100.0, (step / total) * 100.0))
        if self.task_total and self.task_total > 0:
            return max(0.0, min(100.0, (self.task_current / self.task_total) * 100.0))
        return 0.0

    def set_totals(self, total_steps: int = 0, max_epochs: int = 0) -> None:
        if total_steps:
            self.total_steps = int(total_steps)
        if max_epochs:
            self.max_epochs = int(max_epochs)

    def begin_download(self, label: str = "Downloading models") -> None:
        self.mode = "download"
        self.phase = label
        self.detail = ""
        self.task_current = 0.0
        self.task_total = 0.0
        self.task_unit = "bytes"
        self.indeterminate = False
        self.eta_seconds = None
        self._phase_started_at = time.time()

    def update_download(
        self,
        *,
        file_label: str,
        current: float,
        total: float,
        file_index: int = 0,
        file_count: int = 1,
    ) -> None:
        self.mode = "download"
        self.phase = "Downloading models"
        self.detail = file_label
        if file_count > 1 and total > 0:
            overall_total = float(file_count)
            overall_current = float(file_index) + min(1.0, current / total)
            self.task_current = overall_current
            self.task_total = overall_total
            self.task_unit = "files"
        else:
            self.task_current = float(current)
            self.task_total = float(total) if total else 0.0
            self.task_unit = "bytes"
        self.indeterminate = self.task_total <= 0
        elapsed = max(0.1, time.time() - self._phase_started_at)
        if self.task_total > 0 and self.task_current > 0:
            rate = self.task_current / elapsed
            remaining = max(0.0, self.task_total - self.task_current)
            self.eta_seconds = remaining / rate if rate > 0 else None

    def begin_fp8(self) -> None:
        self.mode = "fp8"
        self.phase = "Casting FLUX to FP8"
        self.detail = "Converting bf16/fp16 weights → FP8"
        self.task_current = 0.0
        self.task_total = 0.0
        self.task_unit = "tensors"
        self.indeterminate = True
        self.eta_seconds = None
        self._phase_started_at = time.time()

    def update_fp8(self, current: int, total: int) -> None:
        self.mode = "fp8"
        self.phase = "Casting FLUX to FP8"
        self.detail = "Converting bf16/fp16 weights → FP8"
        self.task_current = float(current)
        self.task_total = float(total)
        self.task_unit = "tensors"
        self.indeterminate = total <= 0
        elapsed = max(0.1, time.time() - self._phase_started_at)
        if total > 0 and current > 0:
            rate = current / elapsed
            remaining = max(0, total - current)
            self.eta_seconds = remaining / rate if rate > 0 else None
        if total > 0 and current >= total:
            self.detail = "FP8 cast complete"
            self.indeterminate = False

    def begin_training(self) -> None:
        self.mode = "training"
        self.phase = "Training"
        self.detail = ""
        self.indeterminate = False
        self.task_unit = "steps"
        self._phase_started_at = time.time()

    def mark_failed(self, reason: str, *, phase: str = "Failed") -> None:
        self.finished = True
        self.success = False
        self.phase = phase
        self.indeterminate = False
        self.eta_seconds = 0
        cleaned = " ".join((reason or "").split())
        if not cleaned:
            cleaned = extract_failure_summary(self.debug_lines) or "Unknown error — check the debug log."
        self.detail = cleaned
        self.append_debug(f"FAILURE: {cleaned}")

    def _update_epoch_from_step(self) -> None:
        if self.max_epochs and self.total_steps and self.step > 0:
            steps_per_epoch = max(1, self.total_steps // self.max_epochs)
            self.epoch = min(self.max_epochs, max(1, (self.step + steps_per_epoch - 1) // steps_per_epoch))

    def _record_step_timing(self) -> None:
        now = time.time()
        if self._last_step_wall is not None:
            delta = now - self._last_step_wall
            if 0.01 < delta < 600:
                self._step_times.append(delta)
                if len(self._step_times) > 50:
                    self._step_times = self._step_times[-50:]
                if self.eta_seconds is None and self.total_steps and self.step < self.total_steps:
                    avg = sum(self._step_times) / len(self._step_times)
                    self.eta_seconds = avg * (self.total_steps - self.step)
        self._last_step_wall = now

    def ingest_line(self, line: str) -> None:
        self.append_debug(line)
        stripped = line.strip()
        if not stripped:
            return

        if FP8_START_RE.search(stripped):
            self.begin_fp8()
            return

        fp8_match = FP8_PROGRESS_RE.search(stripped)
        if fp8_match:
            self.update_fp8(int(fp8_match.group(1)), int(fp8_match.group(2)))
            return

        if self.mode == "fp8" and (
            "Loaded fp8" in stripped
            or "enable block swap" in stripped.lower()
            or "preparing accelerator" in stripped.lower()
            or TQDM_STEPS_RE.search(stripped)
        ):
            if self.task_total > 0:
                self.update_fp8(int(self.task_total), int(self.task_total))
            self.begin_training()

        if SAMPLE_RE.search(stripped):
            self.mode = "training"
            self.phase = "Generating sample image"
            self.indeterminate = False
            return
        if SAVING_RE.search(stripped) and "state dict" not in stripped.lower():
            self.mode = "training"
            self.phase = "Saving checkpoint"
            self.indeterminate = False
            return

        total_match = TOTAL_STEPS_RE.search(stripped)
        if total_match:
            self.total_steps = int(total_match.group(1))
            if self.mode in {"idle", "fp8", "download"}:
                self.begin_training()

        epoch_match = EPOCH_RE.search(stripped)
        if epoch_match:
            self.epoch = int(epoch_match.group(1))
            self.max_epochs = int(epoch_match.group(2))
            if self.phase not in {"Generating sample image", "Saving checkpoint"}:
                self.begin_training()

        tqdm_match = TQDM_STEPS_RE.search(stripped)
        if tqdm_match:
            if self.mode != "training":
                self.begin_training()
            self.step = int(tqdm_match.group(1))
            self.total_steps = int(tqdm_match.group(2))
            self.task_current = float(self.step)
            self.task_total = float(self.total_steps)
            eta = _parse_tqdm_time(tqdm_match.group(4))
            if eta is not None:
                self.eta_seconds = eta
            self._record_step_timing()
            self._update_epoch_from_step()
            if self.phase not in {"Generating sample image", "Saving checkpoint"}:
                self.phase = "Training"

        loss_match = AVR_LOSS_RE.search(stripped)
        if loss_match:
            try:
                self.loss = float(loss_match.group(1))
            except ValueError:
                pass
            if self.phase == "Generating sample image":
                self.phase = "Training"


def extract_failure_summary(debug_lines: List[str], *, lookback: int = 80) -> str:
    """Pull a human-readable failure reason from recent training output."""
    if not debug_lines:
        return ""

    window = debug_lines[-lookback:]
    # Prefer the last exception message after a traceback.
    last_exc = ""
    traceback_idx = None
    for i, line in enumerate(window):
        stripped = line.strip()
        if TRACEBACK_HEADER_RE.match(stripped):
            traceback_idx = i
    if traceback_idx is not None:
        for line in reversed(window[traceback_idx + 1 :]):
            stripped = line.strip()
            if not stripped:
                continue
            m = EXCEPTION_TAIL_RE.match(stripped)
            if m:
                last_exc = f"{m.group(1)}: {m.group(2).strip()}"
                break
            if ERROR_LINE_RE.search(stripped) and not stripped.startswith("File "):
                last_exc = stripped
                break
            # Bare final traceback line (e.g. "RuntimeError: ...")
            if ":" in stripped and not stripped.startswith("File "):
                last_exc = stripped
                break

    if last_exc:
        return last_exc[:500]

    # Otherwise take the most recent matching error-ish line.
    for line in reversed(window):
        stripped = line.strip()
        if not stripped or stripped.startswith("FAILURE:"):
            continue
        if ERROR_LINE_RE.search(stripped):
            return stripped[:500]

    # Fallback: last non-empty, non-tqdm progress line.
    for line in reversed(window):
        stripped = line.strip()
        if not stripped or stripped.startswith("FAILURE:"):
            continue
        if TQDM_STEPS_RE.search(stripped) or FP8_PROGRESS_RE.search(stripped):
            continue
        if stripped.startswith("|") or "it/s" in stripped:
            continue
        return stripped[:500]
    return ""


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_status_card(progress: TrainProgress) -> str:
    from gpu_stats import render_gpu_hud

    pct = progress.percent
    indeterminate = progress.indeterminate or pct < 0
    fill_style = ""
    bar_class = "train-progress-bar"
    fill_class = "train-progress-fill"
    if indeterminate:
        bar_class += " is-indeterminate"
        fill_class += " is-indeterminate"
        pct_label = "…"
    else:
        fill_style = f'style="width:{pct:.1f}%"'
        pct_label = f"{pct:.1f}%"

    if progress.mode == "download":
        if progress.task_unit == "files" and progress.task_total:
            primary_value = f"{progress.task_current:.2f} / {progress.task_total:.0f} files"
        elif progress.task_total:
            primary_value = f"{format_bytes(progress.task_current)} / {format_bytes(progress.task_total)}"
        else:
            primary_value = format_bytes(progress.task_current)
        primary_label = "Progress"
        secondary_label, secondary_value = "File", (progress.detail or "—")
        tertiary_label = "Elapsed"
        tertiary_value = format_duration(progress.elapsed_seconds)
        quaternary_label = "Remaining"
        quaternary_value = format_duration(progress.eta_seconds if not progress.finished else 0)
        extra = ""
    elif progress.mode == "fp8":
        primary_label = "Tensors"
        primary_value = (
            f"{int(progress.task_current)} / {int(progress.task_total)}"
            if progress.task_total
            else "…"
        )
        secondary_label, secondary_value = "Task", (progress.detail or "FP8 cast")
        tertiary_label = "Elapsed"
        tertiary_value = format_duration(progress.elapsed_seconds)
        quaternary_label = "Remaining"
        quaternary_value = format_duration(progress.eta_seconds if not progress.finished else 0)
        extra = ""
    else:
        epoch_label = (
            f"{progress.epoch} / {progress.max_epochs}"
            if progress.max_epochs
            else f"{progress.epoch or '—'}"
        )
        step_label = (
            f"{progress.step} / {progress.total_steps}"
            if progress.total_steps
            else f"{progress.step or '—'}"
        )
        primary_label, primary_value = "Step", step_label
        secondary_label, secondary_value = "Epoch", epoch_label
        tertiary_label = "Elapsed"
        tertiary_value = format_duration(progress.elapsed_seconds)
        quaternary_label = "Remaining"
        quaternary_value = format_duration(progress.eta_seconds if not progress.finished else 0)
        extra = (
            f"<div class='stat'><div class='stat-label'>Loss</div>"
            f"<div class='stat-value subtle'>{progress.loss:.4f}</div></div>"
            if progress.loss is not None
            else ""
        )

    done_class = (
        "done" if progress.finished and progress.success else ("failed" if progress.finished else "")
    )
    if progress.finished and not progress.success and progress.detail:
        detail_html = (
            f"<div class='train-error'>"
            f"<div class='train-error-label'>What broke</div>"
            f"<div class='train-error-message'>{_html_escape(progress.detail)}</div>"
            f"<div class='train-error-hint'>Full output is in Debug log below.</div>"
            f"</div>"
        )
    elif progress.detail and progress.mode in {"download", "fp8"}:
        detail_html = f"<div class='train-detail'>{_html_escape(progress.detail)}</div>"
    else:
        detail_html = ""

    gpu_html = render_gpu_hud()

    return f"""
<div class="train-status-card {done_class} mode-{progress.mode}">
  <div class="phase-chip">{_html_escape(progress.phase)}</div>
  {detail_html}
  <div class="train-metrics">
    <div class="stat">
      <div class="stat-label">{primary_label}</div>
      <div class="stat-value">{primary_value}</div>
    </div>
    <div class="stat">
      <div class="stat-label">{secondary_label}</div>
      <div class="stat-value {'subtle' if progress.mode != 'training' else ''}">{secondary_value}</div>
    </div>
    <div class="stat">
      <div class="stat-label">{tertiary_label}</div>
      <div class="stat-value">{tertiary_value}</div>
    </div>
    <div class="stat">
      <div class="stat-label">{quaternary_label}</div>
      <div class="stat-value">{quaternary_value}</div>
    </div>
    {extra}
  </div>
  <div class="{bar_class}"><div class="{fill_class}" {fill_style}></div></div>
  <div class="train-progress-pct">{pct_label}</div>
  {gpu_html}
</div>
"""
