# FluxGym-R

Dead simple web UI for training FLUX LoRA with **LOW VRAM (12GB / 16GB / 20GB / 32GB)** support. This fork targets **AMD ROCm** first (Ubuntu 24.04), and also installs cleanly on **NVIDIA CUDA**.

- **Frontend:** Gradio WebUI (originally based on the [AI-Toolkit](https://github.com/ostris/ai-toolkit) UI by [@multimodalart](https://x.com/multimodalart)), redesigned as a 4-step wizard for FluxGym-R
- **Backend:** Training powered by [Kohya sd-scripts](https://github.com/kohya-ss/sd-scripts) (`sd3` branch)

FluxGym-R exposes Kohya launch flags through an **Advanced options** accordion (hidden by default). While training, a status card shows phase, step/epoch, elapsed time, ETA, loss, download/FP8 progress when relevant, and live GPU telemetry (name, watts, VRAM). Raw logs stay in a collapsed **Debug log** accordion. You can stop a run from the UI.

---

# Features

| Area | What you get |
|---|---|
| Setup wizard | Upload → LoRA settings → Caption → Review & train |
| Captioning | **JoyCaption** (default) or **Florence-2**; trigger word enforced in every caption |
| Captions on upload | Pair `.txt` files with images (`img0.png` + `img0.txt`) |
| VRAM profiles | `32G Quality`, `32G Safe`, `20G`, `16G`, `12G` |
| Samples | Optional sample prompts every N steps (Kohya sample flags supported) |
| Training UI | Status card, stop button, sample gallery, debug log |
| Publish | Upload trained LoRAs to Hugging Face |
| Download | Grab epoch checkpoints / final LoRA from a training run |
| Delete | Remove selected project `outputs/` and `datasets/` folders |
| Models | Auto-download selected base + CLIP/T5/VAE; extend via `models.yaml` |
| Install | `./install.sh` auto-detects ROCm vs CUDA; `./install-rocm.sh` forces AMD |
| Clean | `./clean.sh` wipes `env/`, weights, datasets, outputs, and `sd-scripts/` |

---

# What is this?

1. A simple UI for training Flux LoRAs without living in the terminal.
2. [AI-Toolkit](https://github.com/ostris/ai-toolkit) is great, but its Gradio UI path was aimed at high VRAM.
3. [Kohya sd-scripts](https://github.com/kohya-ss/sd-scripts) are flexible for FLUX, but CLI-heavy.
4. FluxGym combined AI-Toolkit-style simplicity with Kohya underneath for 12–32GB VRAM.
5. **FluxGym-R** continues that for AMD GPUs on ROCm, with a clearer training view and AMD-first install scripts (CUDA still supported).

---

# Supported Models

1. **flux-dev**
2. **bdsqlsz/flux1-dev2pro-single** ([background](https://medium.com/@zhiwangshi28/why-flux-lora-so-hard-to-train-and-how-to-overcome-it-a0c70bc59eaf))
3. **flux-schnell** (often lower quality; fine for experiments)

Models download automatically when you start training with one selected. Add more by editing [`models.yaml`](models.yaml).

---

# Install

Native install targets **Ubuntu 24.04 LTS** (Python 3.12). The installer auto-detects **AMD (ROCm)** vs **NVIDIA (CUDA)** and installs matching PyTorch **before** other requirements so pip does not pull the wrong wheel.

**AMD ROCm wheel selection** (when `ROCM_TORCH_INDEX` is unset):

| ROCm stack | PyTorch index | Notes |
|---|---|---|
| 10.x (or `ROCM_VERSION=10`) | `nightly/rocm10.0` | Nightly until a stable `rocm10.0` index ships |
| 7.2 (default) | `rocm7.2` | Stable; used when ROCm is below 10 or version is unknown |
| Forced 7.x via `ROCM_VERSION=7.2` | `rocm7.2` | Explicit pin |

**NVIDIA CUDA wheel selection** (when `CUDA_TORCH_INDEX` is unset):

| Driver reports CUDA | PyTorch index | Notes |
|---|---|---|
| 12.8+ | `cu128` | Default for modern GPUs; required for RTX 50 / Blackwell |
| 12.6–12.7 | `cu126` | |
| 12.4–12.5 | `cu124` | |
| 12.1–12.3 | `cu121` | |
| 11.x | `cu118` | Older stacks |

System packages (if needed):

```bash
sudo apt install python3 python3-venv python3-pip git
```

From the repo root:

```bash
chmod +x install.sh install-rocm.sh app-launch.sh clean.sh
./install.sh
```

Optional overrides:

```bash
# Force a backend
FORCE_GPU=rocm ./install.sh
FORCE_GPU=cuda ./install.sh

# Same as FORCE_GPU=rocm (auto-detect 7.2 vs 10)
./install-rocm.sh

# Force ROCm 10 PyTorch nightly wheels
ROCM_VERSION=10 ./install.sh
./install-rocm.sh 10

# Force stable ROCm 7.2 wheels
./install-rocm.sh 7.2

# Pin a specific wheel index
ROCM_TORCH_INDEX=https://download.pytorch.org/whl/nightly/rocm10.0 ./install.sh
CUDA_TORCH_INDEX=https://download.pytorch.org/whl/cu126 ./install.sh
```

The installer creates `env/`, pins `opencv-python==4.10.0.84` (avoids a Python 3.12 source-build failure), clones `sd-scripts` on the `sd3` branch if missing, installs requirements, and writes `env/.fluxgym-gpu-backend` (plus `env/.fluxgym-rocm-index` on AMD) for `app-launch.sh`.

# Start

```bash
./app-launch.sh
```

Open the UI at `http://localhost:7860` (binds to `0.0.0.0` by default via `GRADIO_SERVER_NAME`).

On AMD and NVIDIA, training uses `torch.cuda` (`torch.cuda.is_available()` should be `True` when a GPU backend was installed).

## Install via Docker

The included Dockerfiles / `docker-compose.yml` are **NVIDIA-oriented**. For AMD ROCm, prefer the native `./install.sh` path above.

```bash
git clone https://github.com/Yoink4CM/FluxGym-R
cd FluxGym-R
git clone -b sd3 https://github.com/kohya-ss/sd-scripts
```

Set `PUID` / `PGID` if your user is not `1000` (`id` on Linux). Then:

```bash
docker compose up -d --build
```

Open http://localhost:7860

Use `Dockerfile.cuda12.4` in `docker-compose.yml` if you need that CUDA driver line.

---

# Usage

1. **Upload images** — at least 2 images (about 4–30 is ideal). Optional matching `.txt` caption files.
2. **LoRA settings** — name, trigger word/sentence, base model, VRAM profile, repeats/epochs, optional sample prompts, resize resolution.
3. **Caption images** — run JoyCaption or Florence-2, or edit captions by hand. The trigger must appear in every caption.
4. **Review & train** — check the generated script/config, open Advanced options if needed, then start. Use **Stop training** to abort.

Outputs land under `outputs/<lora_name>/`; datasets under `datasets/<lora_name>/`.

---

# Configuration

## Sample Images

Sample generation is off until you set both:

1. **Sample Image Prompts** — one prompt per line.
2. **Sample Image Every N Steps** — e.g. expected steps 960 and interval 100 → samples at 100, 200, …, 900 for each prompt.

### Advanced sample flags

Kohya [sample syntax](https://github.com/kohya-ss/sd-scripts?tab=readme-ov-file#sample-image-generation-during-training) works in each prompt line:

```text
hrld person is riding a bike --d 42
hrld person is a body builder --d 42
```

Useful flags:

- `--n` — negative prompt (until the next option)
- `--w` / `--h` — width / height
- `--d` — seed
- `--l` — CFG scale
- `--s` — sampling steps

Attention weighting such as `( )` and `[ ]` also works.

## Publishing to Hugging Face

1. Create a token at https://huggingface.co/settings/tokens
2. On the **Publish** tab, paste it and click **Login** (saved locally as `HF_TOKEN`)
3. Pick a trained LoRA, set visibility/name, upload

## Download checkpoints

On the **Download** tab, select a training run and download intermediate epoch saves or the final LoRA.

## Delete projects

On the **Delete** tab, select one or more projects and confirm to permanently remove their `outputs/` and `datasets/` folders.

## Advanced options

The Advanced accordion is built from Kohya sd-scripts launch flags, so FluxGym-R can drive the full script surface. It stays collapsed by default.

## Clean install / reclaim disk

```bash
./clean.sh          # interactive confirm
./clean.sh --yes    # non-interactive
```

Removes `env/`, downloaded model weights under `models/`, `datasets/`, `outputs/`, `sd-scripts/`, `__pycache__/`, and `HF_TOKEN`. Hugging Face hub cache under `~/.cache/huggingface` is left alone. Run `./install.sh` again afterward.

---

# License

See [LICENSE](LICENSE).
