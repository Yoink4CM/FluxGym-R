import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
import subprocess
import threading
import time
import signal
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from huggingface_hub import hf_hub_download, HfApi
from library import flux_train_utils, huggingface_util
from argparse import Namespace
import train_network
import toml
import re
from train_progress import TrainProgress, render_status_card, extract_failure_summary
from captioning import (
    CAPTION_MODELS,
    ensure_trigger_in_caption,
    get_backend,
    resolve_device_dtype,
)

MAX_IMAGES = 150

_train_process = None
_train_stop_requested = False
_train_lock = threading.Lock()


def _kill_train_process(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def request_stop_training():
    global _train_stop_requested
    with _train_lock:
        _train_stop_requested = True
        proc = _train_process
    if proc is not None:
        _kill_train_process(proc)
        gr.Info("Stopping training...")
        return gr.update(interactive=False, value="Stopping...")
    gr.Warning("No active training process to stop")
    return gr.update()


with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def readme(base_model, lora_name, instance_prompt, sample_prompts):

    # model license
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    # tags
    tags = [ "text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym", "fluxgym-r" ]

    # widgets
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # Filename Schema: [name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # Sort by numeric index
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"no samples")
    dtype = "torch.bfloat16"
    # Construct the README content
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A Flux LoRA trained on a local computer with [FluxGym-R](https://github.com/Yoink4CM/FluxGym-R)

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

"""
    return readme_content

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

"""
hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


"""
hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None:
            if "name" in account:
                with open("HF_TOKEN", "w") as file:
                    file.write(hf_token)
                global current_account
                current_account = account_hf()
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        print(f"incorrect hf_token")
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"Uploading to Huggingface. Please Stand by...", duration=None)
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    print(f"upload_hf args={args}")
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[Upload Complete] https://huggingface.co/{repo_id}", duration=None)

def load_captioning(uploaded_files, concept_sentence):
    paths = [_file_path(f) for f in (uploaded_files or [])]
    uploaded_images = [p for p in paths if p and not p.lower().endswith(".txt")]
    txt_files = [p for p in paths if p.lower().endswith(".txt")]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= len(uploaded_images)

        updates.append(gr.update(visible=visible))

        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        corresponding_caption = False
        if image_value:
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()

        if visible and corresponding_caption:
            text_value = ensure_trigger_in_caption(corresponding_caption, concept_sentence)
        elif visible and concept_sentence:
            text_value = concept_sentence
        else:
            text_value = None
        updates.append(gr.update(value=text_value, visible=visible))

    return updates

def hide_captioning():
    updates = []
    for _ in range(1, MAX_IMAGES + 1):
        updates.append(gr.update(visible=False))
        updates.append(gr.update(value=None, visible=False))
        updates.append(gr.update(value=None, visible=False))
    return updates

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_only = []
    for image in (images or []):
        path = _file_path(image)
        if not path or path.lower().endswith(".txt"):
            continue
        image_only.append(path)

    for index, image in enumerate(image_only):
        new_image_path = shutil.copy(image, destination_folder)
        resize_image(new_image_path, new_image_path, size)

        original_caption = inputs[index + 1] if index + 1 < len(inputs) else ""
        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        print(f"image_path={new_image_path}, caption_path = {caption_path}, original_caption={original_caption}")
        with open(caption_path, 'w') as file:
            file.write(original_caption or "")

    print(f"destination_folder {destination_folder}")
    return destination_folder


def run_captioning(images, concept_sentence, caption_model, *captions):
    print(f"run_captioning model={caption_model}")
    print(f"concept sentence {concept_sentence}")
    device, torch_dtype = resolve_device_dtype()
    print(f"device={device} dtype={torch_dtype}")

    image_paths = [
        path for path in (_file_path(p) for p in (images or []))
        if path and not path.lower().endswith(".txt")
    ]
    backend = get_backend(caption_model or "JoyCaption")
    captions = list(captions)
    try:
        backend.load(device, torch_dtype)
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            caption_text = backend.caption(image)
            print(f"caption_text = {caption_text}, concept_sentence={concept_sentence}")
            captions[i] = ensure_trigger_in_caption(caption_text, concept_sentence)
            yield captions
    finally:
        backend.unload()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(base_model, progress_cb=None):
    """Download FLUX assets. progress_cb(file_label, current, total, file_index, file_count) optional."""
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    if base_model == "flux-dev" or base_model == "flux-schnell":
        unet_folder = "models/unet"
    else:
        unet_folder = f"models/unet/{repo}"

    jobs = []
    unet_path = os.path.join(unet_folder, model_file)
    if not os.path.exists(unet_path):
        jobs.append(("Base model (UNET)", repo, unet_folder, model_file))

    vae_folder = "models/vae"
    vae_path = os.path.join(vae_folder, "ae.sft")
    if not os.path.exists(vae_path):
        jobs.append(("VAE (ae.sft)", "cocktailpeanut/xulf-dev", vae_folder, "ae.sft"))

    clip_folder = "models/clip"
    clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
    if not os.path.exists(clip_l_path):
        jobs.append(("CLIP-L", "comfyanonymous/flux_text_encoders", clip_folder, "clip_l.safetensors"))

    t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    if not os.path.exists(t5xxl_path):
        jobs.append(("T5-XXL", "comfyanonymous/flux_text_encoders", clip_folder, "t5xxl_fp16.safetensors"))

    if not jobs:
        if progress_cb:
            progress_cb("All models present", 1, 1, 0, 1)
        return

    # hf_transfer is fast but often skips tqdm callbacks — disable for metered downloads.
    prev_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    from huggingface_hub.utils import tqdm as hub_tqdm
    import sys

    file_count = len(jobs)
    state = {"file_index": 0, "label": "", "last_emit": 0.0}
    tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]

    class MeteredTqdm(hub_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._emit(force=True)

        def update(self, n=1):
            result = super().update(n)
            self._emit()
            return result

        def _emit(self, force: bool = False):
            if not progress_cb:
                return
            now = time.time()
            if not force and now - state["last_emit"] < 0.2:
                return
            state["last_emit"] = now
            total = float(self.total or 0)
            current = float(self.n or 0)
            progress_cb(state["label"], current, total, state["file_index"], file_count)

    original_tqdm = tqdm_mod.tqdm
    tqdm_mod.tqdm = MeteredTqdm
    try:
        for idx, (label, repo_id, local_dir, filename) in enumerate(jobs):
            state["file_index"] = idx
            state["label"] = f"{label}: {filename}"
            os.makedirs(local_dir, exist_ok=True)
            print(f"download {label} -> {filename}")
            if progress_cb:
                progress_cb(state["label"], 0, 0, idx, file_count)
            hf_hub_download(repo_id=repo_id, local_dir=local_dir, filename=filename)
            if progress_cb:
                # Mark file complete even if total was unknown.
                progress_cb(state["label"], 1, 1, idx, file_count)
    finally:
        tqdm_mod.tqdm = original_tqdm
        if prev_hf_transfer is None:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        else:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = prev_hf_transfer


def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""
def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    base_model,
    output_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    network_alpha,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components
):

    print(
        f"gen_sh: network_dim:{network_dim}, network_alpha:{network_alpha}, "
        f"learning_rate={learning_rate}, max_train_epochs={max_train_epochs}, "
        f"save_every_n_epochs={save_every_n_epochs}, timestep_sampling={timestep_sampling}, "
        f"guidance_scale={guidance_scale}, vram={vram}, sample_prompts={sample_prompts}, "
        f"sample_every_n_steps={sample_every_n_steps}"
    )

    output_dir = resolve_path(f"outputs/{output_name}")
    sample_prompts_path = resolve_path(f"outputs/{output_name}/sample_prompts.txt")

    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    ############# Sample args ########################
    sample = ""
    if len(sample_prompts) > 0 and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={sample_prompts_path} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""


    ############# Optimizer / VRAM profile args ########################
#    if vram == "8G":
#        optimizer = f"""--optimizer_type adafactor {line_break}
#    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
#        --split_mode {line_break}
#        --network_args "train_blocks=single" {line_break}
#        --lr_scheduler constant_with_warmup {line_break}
#        --max_grad_norm 0.0 {line_break}"""
    if vram == "16G":
        # 16G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
        # --fp8_base casts *base* FLUX weights only; LoRA grads stay bf16 via mixed_precision.
        mem_flags = f"--fp8_base {line_break}"
    elif vram == "12G":
        # 12G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
        mem_flags = f"--fp8_base {line_break}"
    elif vram == "32G Quality":
        # bf16 base + light block swap. UNet-only LoRA reduces CLIP-driven color oversaturation.
        optimizer = f"--optimizer_type adamw8bit {line_break}"
        mem_flags = (
            f"--blocks_to_swap 10 {line_break}"
            f"--network_train_unet_only {line_break}"
        )
    elif vram == "32G Safe":
        # Same recipe as Quality, with fp8 base + more swap for OOM headroom on ~32GB.
        optimizer = f"--optimizer_type adamw8bit {line_break}"
        mem_flags = (
            f"--fp8_base {line_break}"
            f"--blocks_to_swap 14 {line_break}"
            f"--network_train_unet_only {line_break}"
        )
    else:
        # 20G VRAM (and any legacy "32G" value)
        optimizer = f"--optimizer_type adamw8bit {line_break}"
        mem_flags = f"--fp8_base {line_break}"

    #######################################################
    model_config = models[base_model]
    model_file = model_config["file"]
    repo = model_config["repo"]
    if base_model == "flux-dev" or base_model == "flux-schnell":
        model_folder = "models/unet"
    else:
        model_folder = f"models/unet/{repo}"
    model_path = os.path.join(model_folder, model_file)
    pretrained_model_path = resolve_path(model_path)

    clip_path = resolve_path("models/clip/clip_l.safetensors")
    t5_path = resolve_path("models/clip/t5xxl_fp16.safetensors")
    ae_path = resolve_path("models/vae/ae.sft")
    # Keep mixed_precision/save_precision on bf16 so LoRA grads are never FP8.
    sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  --network_alpha {network_alpha} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  {mem_flags}--highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path(f"outputs/{output_name}/dataset.toml")} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
   


    ############# Advanced args ########################
    global advanced_component_ids
    global original_advanced_component_values
   
    # check dirty
    print(f"original_advanced_component_values = {original_advanced_component_values}")
    advanced_flags = []
    for i, current_value in enumerate(advanced_components):
#        print(f"compare {advanced_component_ids[i]}: old={original_advanced_component_values[i]}, new={current_value}")
        if original_advanced_component_values[i] != current_value:
            # dirty
            if current_value == True:
                # Boolean
                advanced_flags.append(advanced_component_ids[i])
            else:
                # string
                advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")

    if len(advanced_flags) > 0:
        advanced_flags_str = f" {line_break}\n  ".join(advanced_flags)
        sh = sh + "\n  " + advanced_flags_str

    return sh

def gen_toml(
  dataset_folder,
  resolution,
  class_tokens,
  num_repeats
):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        print(f"max_train_epochs={max_train_epochs} num_images={num_images}, num_repeats={num_repeats}, total_steps={total_steps}")
        return gr.update(value = total_steps)
    except:
        print("")

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []


_EPOCH_CKPT_RE = re.compile(r"^(?P<base>.+)-(?P<epoch>\d+)\.safetensors$", re.IGNORECASE)


def list_checkpoints(run_dir: str):
    """List LoRA .safetensors in a run folder with labels (Epoch N / Final)."""
    if not run_dir or not os.path.isdir(run_dir):
        return []
    rows = []
    for name in os.listdir(run_dir):
        if not name.lower().endswith(".safetensors"):
            continue
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        match = _EPOCH_CKPT_RE.match(name)
        if match:
            epoch = int(match.group("epoch"))
            label = f"Epoch {epoch}"
            sort_key = (0, epoch)
        else:
            label = "Final"
            sort_key = (1, 0)
        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            size_bytes = 0
        rows.append(
            {
                "path": path,
                "name": name,
                "label": label,
                "size_bytes": size_bytes,
                "sort_key": sort_key,
            }
        )
    rows.sort(key=lambda r: r["sort_key"])
    return rows


def format_checkpoint_summary(rows) -> str:
    if not rows:
        return "_No `.safetensors` checkpoints found for this run._"
    lines = ["| Checkpoint | File | Size |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| **{row['label']}** | `{row['name']}` | {_fmt_size(row['size_bytes'])} |")
    return "\n".join(lines)


def _fmt_size(n: float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return "—"


def get_download_run_choices():
    folders = get_loras()
    return [os.path.basename(p) for p in folders]


def refresh_download_runs(selected=None):
    choices = get_download_run_choices()
    if not choices:
        return gr.update(choices=[], value=None), "_No training outputs yet._", None
    value = selected if selected in choices else choices[0]
    summary, files = load_download_checkpoints(value)
    return gr.update(choices=choices, value=value), summary, files


def load_download_checkpoints(run_name):
    if not run_name:
        return "_No training outputs yet._", None
    run_dir = resolve_path_without_quotes(os.path.join("outputs", str(run_name)))
    rows = list_checkpoints(run_dir)
    if not rows:
        return f"_No `.safetensors` checkpoints in `{run_name}`._", None
    summary = format_checkpoint_summary(rows)
    files = [row["path"] for row in rows]
    return summary, files


def _dir_size(path: str) -> int:
    total = 0
    if not path or not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _safe_project_slug(slug: str) -> str:
    slug = (slug or "").strip()
    if (
        not slug
        or slug in {".", ".."}
        or "/" in slug
        or "\\" in slug
        or os.path.isabs(slug)
        or slug.startswith(".")
    ):
        raise gr.Error(f"Invalid project name: {slug!r}")
    return slug


def _project_paths(slug: str):
    slug = _safe_project_slug(slug)
    outputs_root = os.path.realpath(resolve_path_without_quotes("outputs"))
    datasets_root = os.path.realpath(resolve_path_without_quotes("datasets"))
    out_path = os.path.realpath(os.path.join(outputs_root, slug))
    ds_path = os.path.realpath(os.path.join(datasets_root, slug))
    if not out_path.startswith(outputs_root + os.sep) and out_path != outputs_root:
        raise gr.Error(f"Refusing to touch path outside outputs/: {slug}")
    if not ds_path.startswith(datasets_root + os.sep) and ds_path != datasets_root:
        raise gr.Error(f"Refusing to touch path outside datasets/: {slug}")
    return out_path, ds_path


def list_projects():
    """Union of project slugs under outputs/ and datasets/, newest first."""
    names = set()
    mtimes = {}
    for root_name in ("outputs", "datasets"):
        root = resolve_path_without_quotes(root_name)
        if not os.path.isdir(root):
            continue
        for item in os.listdir(root):
            if item.startswith("."):
                continue
            path = os.path.join(root, item)
            if not os.path.isdir(path) or item == "sample":
                continue
            names.add(item)
            try:
                mtime = os.path.getctime(path)
            except OSError:
                mtime = 0
            mtimes[item] = max(mtimes.get(item, 0), mtime)
    return sorted(names, key=lambda n: mtimes.get(n, 0), reverse=True)


def project_size(slug: str) -> int:
    out_path, ds_path = _project_paths(slug)
    return _dir_size(out_path) + _dir_size(ds_path)


def delete_project_choices():
    choices = []
    for slug in list_projects():
        try:
            size = _fmt_size(project_size(slug))
        except gr.Error:
            continue
        choices.append((f"{slug} ({size})", slug))
    return choices


def refresh_delete_projects():
    choices = delete_project_choices()
    if not choices:
        return gr.update(choices=[], value=[]), "_No projects to delete._"
    return gr.update(choices=choices, value=[]), "_Select one or more projects, confirm, then delete._"


def delete_projects(selected, confirmed):
    if not confirmed:
        raise gr.Error("Check the confirmation box before deleting.")
    selected = selected or []
    if not selected:
        raise gr.Error("Select at least one project to delete.")

    removed = []
    freed = 0
    for slug in selected:
        out_path, ds_path = _project_paths(slug)
        size = _dir_size(out_path) + _dir_size(ds_path)
        parts = []
        for path, label in ((out_path, "outputs"), (ds_path, "datasets")):
            if os.path.isdir(path):
                shutil.rmtree(path)
                parts.append(label)
        if parts:
            freed += size
            removed.append(f"- `{slug}` ({', '.join(parts)}) — {_fmt_size(size)}")

    if not removed:
        status = "_Nothing was deleted (paths missing)._"
    else:
        status = (
            f"**Deleted {len(removed)} project(s)** · freed {_fmt_size(freed)}\n\n"
            + "\n".join(removed)
        )
    choices_update, _hint = refresh_delete_projects()
    return choices_update, status


def _training_outputs(progress: TrainProgress, show_back: bool = False, show_stop: bool = False):
    stop_update = (
        gr.update(visible=True, interactive=True, value="Stop training")
        if show_stop
        else gr.update(visible=False, interactive=False)
    )
    return (
        gr.update(visible=False),  # setup_panel
        gr.update(visible=True),   # training_panel
        render_status_card(progress),
        progress.debug_text,
        gr.update(visible=show_back),
        stop_update,
        5,  # wizard_step
        wizard_indicator_html(5),
        gr.update(visible=False),  # wizard_nav
    )


WIZARD_STEP_LABELS = ["Upload", "Settings", "Caption", "Review", "Train"]


def wizard_indicator_html(step: int) -> str:
    parts = []
    for i, label in enumerate(WIZARD_STEP_LABELS, start=1):
        if i < step:
            cls = "done"
        elif i == step:
            cls = "active"
        else:
            cls = ""
        parts.append(
            f'<div class="wizard-step {cls}">'
            f'<span class="wizard-num">{i}</span>'
            f'<span class="wizard-label">{label}</span>'
            f"</div>"
        )
        if i < len(WIZARD_STEP_LABELS):
            parts.append('<div class="wizard-sep"></div>')
    indicator = f'<div class="wizard-indicator">{"".join(parts)}</div>'
    return (
        "<nav>"
        "<img id='logo' src='/file=icon.png' width='80' height='80' alt='FluxGym-R'>"
        "<div class='brand-lockup'>"
        "<div class='brand-title'>FluxGym<span>-R</span></div>"
        "<div class='brand-sub'>ROCm edition</div>"
        "</div>"
        "<div class='flexible'></div>"
        f"{indicator}"
        "<div class='nav-links'>"
        "<a href='https://fluxgym.org/' class='nav-ext-link' target='_blank' rel='noopener noreferrer'>Website</a>"
        "<a href='https://github.com/Yoink4CM/FluxGym-R' class='nav-ext-link' target='_blank' rel='noopener noreferrer'>GitHub</a>"
        "</div>"
        "</nav>"
    )


def _file_path(item) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("name") or item.get("path") or "")
    name = getattr(item, "name", None)
    return str(name) if name else str(item)


def _count_uploaded_images(images) -> int:
    if not images:
        return 0
    count = 0
    for item in images:
        path = _file_path(item)
        if path and not path.lower().endswith(".txt"):
            count += 1
    return count


def show_wizard_step(step: int):
    step = max(1, min(4, int(step)))
    return (
        step,
        gr.update(visible=True),   # setup_panel
        gr.update(visible=False),  # training_panel
        wizard_indicator_html(step),
        gr.update(visible=(step == 1)),
        gr.update(visible=(step == 2)),
        gr.update(visible=(step == 3)),
        gr.update(visible=(step == 4)),
        gr.update(visible=True),   # wizard_nav
        gr.update(interactive=(step > 1)),  # wizard_back
        gr.update(visible=(step < 4)),      # wizard_next
    )


def wizard_back(step: int):
    return show_wizard_step(max(1, int(step) - 1))


def wizard_next(step, images, lora_name, concept_sentence, *captions):
    step = int(step or 1)
    if step == 1:
        if _count_uploaded_images(images) < 2:
            raise gr.Error("Upload at least 2 images before continuing.")
        return show_wizard_step(2) + tuple(gr.update() for _ in captions)

    if step == 2:
        if not (lora_name or "").strip():
            raise gr.Error("Enter a LoRA name before continuing.")
        if not (concept_sentence or "").strip():
            raise gr.Error("Enter a trigger word/sentence before continuing.")
        return show_wizard_step(3) + tuple(gr.update() for _ in captions)

    if step == 3:
        n = _count_uploaded_images(images)
        if n < 2:
            raise gr.Error("Upload at least 2 images before continuing.")
        fixed = []
        for i in range(n):
            raw = captions[i] if i < len(captions) else ""
            if not (raw or "").strip():
                raise gr.Error(
                    f"Caption {i + 1} is empty. Add AI captions or write captions for every image."
                )
            fixed.append(ensure_trigger_in_caption(raw, concept_sentence))
        caption_updates = [
            gr.update(value=fixed[i]) if i < len(fixed) else gr.update()
            for i in range(len(captions))
        ]
        return show_wizard_step(4) + tuple(caption_updates)

    return show_wizard_step(step) + tuple(gr.update() for _ in captions)


def back_to_setup():
    idle = TrainProgress(phase="Idle")
    step_updates = show_wizard_step(4)
    return (
        step_updates[1],  # setup_panel
        step_updates[2],  # training_panel
        render_status_card(idle),
        "",
        gr.update(visible=False),  # back_btn
        gr.update(visible=False, interactive=False, value="Stop training"),  # stop_btn
        step_updates[0],  # wizard_step
        step_updates[3],  # indicator
        step_updates[4],  # step1
        step_updates[5],  # step2
        step_updates[6],  # step3
        step_updates[7],  # step4
        step_updates[8],  # wizard_nav
        step_updates[9],  # wizard_back
        step_updates[10], # wizard_next
    )


def start_training(
    base_model,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
    max_train_epochs,
    total_steps,
):
    global _train_process, _train_stop_requested
    with _train_lock:
        _train_stop_requested = False
        _train_process = None

    progress = TrainProgress(phase="Preparing")
    try:
        progress.set_totals(
            total_steps=int(total_steps or 0),
            max_epochs=int(max_train_epochs or 0),
        )
    except (TypeError, ValueError):
        pass

    yield _training_outputs(progress, show_stop=True)

    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    progress.begin_download()
    progress.append_debug(f"Downloading base model: {base_model}")
    yield _training_outputs(progress, show_stop=True)

    download_state = {"error": None, "finished": False}

    def on_download_progress(file_label, current, total, file_index, file_count):
        progress.update_download(
            file_label=file_label,
            current=current,
            total=total,
            file_index=file_index,
            file_count=file_count,
        )

    def _download_worker():
        try:
            download(base_model, progress_cb=on_download_progress)
        except Exception as exc:
            download_state["error"] = exc
        finally:
            download_state["finished"] = True

    download_thread = threading.Thread(target=_download_worker, daemon=True)
    download_thread.start()
    while download_thread.is_alive():
        if _train_stop_requested:
            progress.mark_failed("Stopped by user", phase="Stopped")
            progress.append_debug("Stop requested during model download")
            yield _training_outputs(progress, show_back=True, show_stop=False)
            return
        yield _training_outputs(progress, show_stop=True)
        time.sleep(0.25)
    download_thread.join()
    if _train_stop_requested:
        progress.mark_failed("Stopped by user", phase="Stopped")
        yield _training_outputs(progress, show_back=True, show_stop=False)
        return
    if download_state["error"] is not None:
        err = download_state["error"]
        progress.mark_failed(f"{type(err).__name__}: {err}", phase="Download failed")
        yield _training_outputs(progress, show_back=True, show_stop=False)
        return
    progress.append_debug("Model download check complete")
    yield _training_outputs(progress, show_stop=True)

    file_type = "bat" if sys.platform == "win32" else "sh"
    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, "w", encoding="utf-8") as file:
        file.write(train_script)
    progress.append_debug(f"Generated train script at {sh_filename}")

    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, "w", encoding="utf-8") as file:
        file.write(train_config)
    progress.append_debug("Generated dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "w", encoding="utf-8") as file:
        file.write(sample_prompts)
    progress.append_debug("Generated sample_prompts.txt")

    if _train_stop_requested:
        progress.mark_failed("Stopped by user", phase="Stopped")
        yield _training_outputs(progress, show_back=True, show_stop=False)
        return

    command = sh_filepath if sys.platform == "win32" else f'bash "{sh_filepath}"'
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["LOG_LEVEL"] = "DEBUG"
    cwd = os.path.dirname(os.path.abspath(__file__))

    progress.phase = "Starting training"
    progress.append_debug(f"Running: {command}")
    yield _training_outputs(progress, show_stop=True)
    gr.Info("Started training")

    popen_kwargs = dict(
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    with _train_lock:
        _train_process = process

    last_yield = 0.0
    runner_error = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            if _train_stop_requested:
                break
            # tqdm often rewrites the same line with \r
            for chunk in raw_line.replace("\r", "\n").split("\n"):
                if not chunk:
                    continue
                progress.ingest_line(chunk)
            now = time.time()
            # Snappier UI updates during download / FP8 cast
            interval = 0.2 if progress.mode in {"download", "fp8"} else 0.4
            if now - last_yield >= interval:
                yield _training_outputs(progress, show_stop=True)
                last_yield = now
        if _train_stop_requested and process.poll() is None:
            _kill_train_process(process)
        return_code = process.wait()
    except Exception as exc:
        runner_error = exc
        progress.append_debug(f"Training runner error: {exc}")
        return_code = 1
        if process.poll() is None:
            _kill_train_process(process)
    finally:
        with _train_lock:
            if _train_process is process:
                _train_process = None

    # Generate Readme on success
    if _train_stop_requested:
        progress.mark_failed("Stopped by user", phase="Stopped")
        progress.append_debug("Training stopped by user")
        gr.Info("Training stopped")
    elif return_code == 0:
        try:
            config = toml.loads(train_config)
            concept_sentence = config["datasets"][0]["subsets"][0]["class_tokens"]
            with open(sample_prompts_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
            md = readme(base_model, lora_name, concept_sentence, prompts)
            readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(md)
            progress.phase = "Complete"
            progress.finished = True
            progress.success = True
            progress.eta_seconds = 0
            progress.append_debug("Training complete.")
            gr.Info("Training Complete. Check the outputs folder for the LoRA files.", duration=None)
        except Exception as exc:
            progress.phase = "Complete (readme error)"
            progress.finished = True
            progress.success = True
            progress.detail = f"Training finished, but README generation failed: {exc}"
            progress.append_debug(f"README generation failed: {exc}")
    else:
        if runner_error is not None:
            reason = f"{type(runner_error).__name__}: {runner_error}"
        else:
            reason = extract_failure_summary(progress.debug_lines)
            if not reason:
                reason = f"Training process exited with code {return_code}"
            else:
                reason = f"{reason} (exit code {return_code})"
        progress.mark_failed(reason)
        gr.Warning(reason[:200])

    yield _training_outputs(progress, show_back=True, show_stop=False)


def update(
    base_model,
    lora_name,
    resolution,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    network_alpha,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components,
):
    output_name = slugify(lora_name)
    dataset_folder = str(f"datasets/{output_name}")
    sh = gen_sh(
        base_model,
        output_name,
        resolution,
        seed,
        workers,
        learning_rate,
        network_dim,
        network_alpha,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    )
    toml = gen_toml(
        dataset_folder,
        resolution,
        class_tokens,
        num_repeats
    )
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

"""
demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, hf_account])
"""
def loaded():
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    if current_account != None:
        return gr.update(value=current_account["token"]), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

def refresh_publish_tab():
    loras = get_loras()
    return gr.Dropdown(label="Trained LoRAs", choices=loras)

def init_advanced():
    # if basic_args
    basic_args = {
        'pretrained_model_name_or_path',
        'clip_l',
        't5xxl',
        'ae',
        'cache_latents_to_disk',
        'save_model_as',
        'sdpa',
        'persistent_data_loader_workers',
        'max_data_loader_n_workers',
        'seed',
        'gradient_checkpointing',
        'mixed_precision',
        'save_precision',
        'network_module',
        'network_dim',
        'network_alpha',
        'learning_rate',
        'cache_text_encoder_outputs',
        'cache_text_encoder_outputs_to_disk',
        'fp8_base',
        'highvram',
        'full_fp16',
        'full_bf16',
        'max_train_epochs',
        'save_every_n_epochs',
        'dataset_config',
        'output_dir',
        'output_name',
        'timestep_sampling',
        'discrete_flow_shift',
        'model_prediction_type',
        'guidance_scale',
        'loss_type',
        'optimizer_type',
        'optimizer_args',
        'lr_scheduler',
        'sample_prompts',
        'sample_every_n_steps',
        'max_grad_norm',
        'split_mode',
        'network_args'
    }

    # generate a UI config
    # if not in basic_args, create a simple form
    parser = train_network.setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)
    args_info = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip the default help argument
            # if the dest is included in basic_args
            args_info[action.dest] = {
                "action": action.option_strings,  # Option strings like '--use_8bit_adam'
                "type": action.type,              # Type of the argument
                "help": action.help,              # Help message
                "default": action.default,        # Default value, if any
                "required": action.required       # Whether the argument is required
            }
    temp = []
    for key in args_info:
        temp.append({ 'key': key, 'action': args_info[key] })
    temp.sort(key=lambda x: x['key'])
    advanced_component_ids = []
    advanced_components = []
    for item in temp:
        key = item['key']
        action = item['action']
        if key in basic_args:
            print("")
        else:
            action_type = str(action['type'])
            component = None
            with gr.Column(min_width=300):
                if action_type == "None":
                    # radio
                    component = gr.Checkbox()
    #            elif action_type == "<class 'str'>":
    #                component = gr.Textbox()
    #            elif action_type == "<class 'int'>":
    #                component = gr.Number(precision=0)
    #            elif action_type == "<class 'float'>":
    #                component = gr.Number()
    #            elif "int_or_float" in action_type:
    #                component = gr.Number()
                else:
                    component = gr.Textbox(value="")
                if component != None:
                    component.interactive = True
                    component.elem_id = action['action'][0]
                    component.label = component.elem_id
                    component.elem_classes = ["advanced"]
                if action['help'] != None:
                    component.info = action['help']
            advanced_components.append(component)
            advanced_component_ids.append(component.elem_id)
    return advanced_components, advanced_component_ids


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fff1f0",
        c100="#ffd7d4",
        c200="#ffb0a9",
        c300="#ff7f73",
        c400="#ff4d3a",
        c500="#ff2a1f",
        c600="#e01810",
        c700="#b8120d",
        c800="#8f100e",
        c900="#6e100f",
        c950="#3b0605",
    ),
    secondary_hue="zinc",
    neutral_hue="zinc",
    text_size=gr.themes.Size(lg="16px", md="14px", sm="13px", xl="18px", xs="12px", xxl="22px", xxs="10px"),
    font=[gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    # Force dark palette on both light/dark Gradio CSS paths (Soft defaults leave
    # light fills that clash with white body text — e.g. selected tabs).
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    body_text_color="#ffffff",
    body_text_color_dark="#ffffff",
    body_text_color_subdued="#a3a3a3",
    body_text_color_subdued_dark="#a3a3a3",
    background_fill_primary="#0a0a0a",
    background_fill_primary_dark="#0a0a0a",
    background_fill_secondary="#141414",
    background_fill_secondary_dark="#141414",
    border_color_primary="#2a2a2a",
    border_color_primary_dark="#2a2a2a",
    border_color_accent="#ff2a1f",
    border_color_accent_dark="#ff2a1f",
    color_accent="#ff2a1f",
    color_accent_soft="rgba(255,42,31,0.14)",
    color_accent_soft_dark="rgba(255,42,31,0.14)",
    link_text_color="#ff4d3a",
    link_text_color_dark="#ff4d3a",
    link_text_color_hover="#ff7f73",
    link_text_color_hover_dark="#ff7f73",
    link_text_color_active="#ff2a1f",
    link_text_color_active_dark="#ff2a1f",
    link_text_color_visited="#ff7f73",
    link_text_color_visited_dark="#ff7f73",
    block_background_fill="#141414",
    block_background_fill_dark="#141414",
    block_border_width="1px",
    block_border_width_dark="1px",
    block_border_color="#2a2a2a",
    block_border_color_dark="#2a2a2a",
    block_shadow="none",
    block_shadow_dark="none",
    block_radius="12px",
    block_label_background_fill="#1f1f1f",
    block_label_background_fill_dark="#1f1f1f",
    block_label_border_color="#2a2a2a",
    block_label_border_color_dark="#2a2a2a",
    block_label_text_color="#ffffff",
    block_label_text_color_dark="#ffffff",
    block_title_background_fill="#1f1f1f",
    block_title_background_fill_dark="#1f1f1f",
    block_title_text_color="#ffffff",
    block_title_text_color_dark="#ffffff",
    block_info_text_color="#a3a3a3",
    block_info_text_color_dark="#a3a3a3",
    panel_background_fill="#141414",
    panel_background_fill_dark="#141414",
    panel_border_color="#2a2a2a",
    panel_border_color_dark="#2a2a2a",
    accordion_text_color="#ffffff",
    accordion_text_color_dark="#ffffff",
    button_primary_background_fill="#ff2a1f",
    button_primary_background_fill_dark="#ff2a1f",
    button_primary_background_fill_hover="#ff4d3a",
    button_primary_background_fill_hover_dark="#ff4d3a",
    button_primary_border_color="#ff2a1f",
    button_primary_border_color_dark="#ff2a1f",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="#1f1f1f",
    button_secondary_background_fill_dark="#1f1f1f",
    button_secondary_background_fill_hover="#2a2a2a",
    button_secondary_background_fill_hover_dark="#2a2a2a",
    button_secondary_border_color="#2a2a2a",
    button_secondary_border_color_dark="#2a2a2a",
    button_secondary_text_color="#ffffff",
    button_secondary_text_color_dark="#ffffff",
    button_border_width="0px",
    button_border_width_dark="0px",
    checkbox_background_color="#0f0f0f",
    checkbox_background_color_dark="#0f0f0f",
    checkbox_background_color_selected="#ff2a1f",
    checkbox_background_color_selected_dark="#ff2a1f",
    checkbox_border_color="#2a2a2a",
    checkbox_border_color_dark="#2a2a2a",
    checkbox_border_color_selected="#ff2a1f",
    checkbox_border_color_selected_dark="#ff2a1f",
    checkbox_label_background_fill="#1f1f1f",
    checkbox_label_background_fill_dark="#1f1f1f",
    checkbox_label_background_fill_hover="#2a2a2a",
    checkbox_label_background_fill_hover_dark="#2a2a2a",
    checkbox_label_background_fill_selected="#ff2a1f",
    checkbox_label_background_fill_selected_dark="#ff2a1f",
    checkbox_label_border_color="#2a2a2a",
    checkbox_label_border_color_dark="#2a2a2a",
    checkbox_label_text_color="#ffffff",
    checkbox_label_text_color_dark="#ffffff",
    checkbox_label_text_color_selected="white",
    checkbox_label_text_color_selected_dark="white",
    code_background_fill="#0f0f0f",
    code_background_fill_dark="#0f0f0f",
    input_background_fill="#0f0f0f",
    input_background_fill_dark="#0f0f0f",
    input_background_fill_focus="#141414",
    input_background_fill_focus_dark="#141414",
    input_border_color="#2a2a2a",
    input_border_color_dark="#2a2a2a",
    input_border_color_focus="#ff2a1f",
    input_border_color_focus_dark="#ff2a1f",
    input_placeholder_color="#737373",
    input_placeholder_color_dark="#737373",
    input_shadow="none",
    input_shadow_dark="none",
    input_shadow_focus="none",
    input_shadow_focus_dark="none",
    table_even_background_fill="#141414",
    table_even_background_fill_dark="#141414",
    table_odd_background_fill="#1a1a1a",
    table_odd_background_fill_dark="#1a1a1a",
    table_border_color="#2a2a2a",
    table_border_color_dark="#2a2a2a",
    table_row_focus="rgba(255,42,31,0.14)",
    table_row_focus_dark="rgba(255,42,31,0.14)",
    table_text_color="#ffffff",
    table_text_color_dark="#ffffff",
    error_background_fill="#1a0a0c",
    error_background_fill_dark="#1a0a0c",
    error_text_color="#fecdd3",
    error_text_color_dark="#fecdd3",
    error_border_color="#fb7185",
    error_border_color_dark="#fb7185",
    error_icon_color="#fb7185",
    error_icon_color_dark="#fb7185",
    stat_background_fill="#1f1f1f",
    stat_background_fill_dark="#1f1f1f",
    shadow_drop="none",
    shadow_drop_lg="none",
)

css = """
:root {
  --fg-accent: #ff2a1f;
  --fg-accent-hover: #ff4d3a;
  --fg-accent-soft: rgba(255,42,31,0.14);
  --fg-ink: #ffffff;
  --fg-muted: #a3a3a3;
  --fg-border: #2a2a2a;
  --fg-surface: #141414;
  --fg-bg: #0a0a0a;
  --fg-ok: #34d399;
  --fg-bad: #fb7185;
  /* Override Soft light defaults that clash with white text */
  --background-fill-primary: #0a0a0a !important;
  --background-fill-secondary: #141414 !important;
  --body-background-fill: #0a0a0a !important;
  --body-text-color: #ffffff !important;
  --body-text-color-subdued: #a3a3a3 !important;
  --border-color-primary: #2a2a2a !important;
  --block-background-fill: #141414 !important;
  --block-label-background-fill: #1f1f1f !important;
  --block-label-text-color: #ffffff !important;
  --block-title-text-color: #ffffff !important;
  --input-background-fill: #0f0f0f !important;
  --input-border-color: #2a2a2a !important;
  --checkbox-label-background-fill: #1f1f1f !important;
  --checkbox-label-background-fill-selected: #ff2a1f !important;
  --checkbox-label-text-color: #ffffff !important;
  --checkbox-label-text-color-selected: #ffffff !important;
  --color-accent: #ff2a1f !important;
  --color-accent-soft: rgba(255,42,31,0.14) !important;
  --table-even-background-fill: #141414 !important;
  --table-odd-background-fill: #1a1a1a !important;
  --panel-background-fill: #141414 !important;
  --code-background-fill: #0f0f0f !important;
}
.dark {
  --background-fill-primary: #0a0a0a !important;
  --background-fill-secondary: #141414 !important;
  --body-background-fill: #0a0a0a !important;
  --body-text-color: #ffffff !important;
  --body-text-color-subdued: #a3a3a3 !important;
  --border-color-primary: #2a2a2a !important;
  --block-background-fill: #141414 !important;
  --block-label-background-fill: #1f1f1f !important;
  --block-label-text-color: #ffffff !important;
  --checkbox-label-background-fill: #1f1f1f !important;
  --checkbox-label-background-fill-selected: #ff2a1f !important;
  --checkbox-label-text-color: #ffffff !important;
  --checkbox-label-text-color-selected: #ffffff !important;
  --color-accent-soft: rgba(255,42,31,0.14) !important;
}
@keyframes rotate {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
@keyframes pulse-soft {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.65; }
}
.gradio-container {
  background: var(--fg-bg) !important;
  min-height: 100vh;
  color: var(--fg-ink) !important;
  font-family: "Manrope", ui-sans-serif, system-ui, sans-serif !important;
}
#advanced_options .advanced:nth-child(even) { background: #1a1a1a !important; }
h1 {
  font-family: "Manrope", ui-sans-serif, system-ui, sans-serif !important;
  font-weight: 600 !important;
  font-size: 24px !important;
  letter-spacing: -0.02em !important;
  color: var(--fg-ink) !important;
}
h3 { margin-top: 0; color: var(--fg-ink) !important; }
.tabitem { border: 0px; }
.group_padding {}
/* Selected tabs: Soft uses --background-fill-primary (was white) + body text (white). */
.tab-nav button,
button[role="tab"] {
  color: var(--fg-muted) !important;
  background: transparent !important;
  border-color: transparent !important;
}
.tab-nav button:hover,
button[role="tab"]:hover {
  color: var(--fg-ink) !important;
}
.tab-nav button.selected,
button[role="tab"].selected,
button[role="tab"][aria-selected="true"] {
  background: var(--fg-accent) !important;
  color: #ffffff !important;
  border-color: var(--fg-accent) !important;
}
.tab-nav .bar {
  background: var(--fg-accent) !important;
}
/* Soft leftover light fills that fight white text */
.prose, .markdown, .prose p, .markdown p,
.prose li, .markdown li, .prose span, .markdown span {
  color: var(--fg-ink) !important;
}
.prose a, .markdown a {
  color: var(--fg-accent-hover) !important;
}
.prose a:hover, .markdown a:hover {
  color: #ff7f73 !important;
}
.block .label-wrap, .block .label-wrap span {
  color: var(--fg-ink) !important;
}
.block .info, .block span.info, .form .info {
  color: var(--fg-muted) !important;
}
input, textarea, select,
.wrap input, .wrap textarea,
[data-testid="textbox"] textarea,
.scroll-hide textarea {
  color: var(--fg-ink) !important;
  background-color: #0f0f0f !important;
}
/* Dropdown / options menus */
ul.options, [role="listbox"] {
  background: #141414 !important;
  border-color: var(--fg-border) !important;
  color: var(--fg-ink) !important;
}
ul.options li, [role="option"] {
  color: var(--fg-ink) !important;
  background: transparent !important;
}
ul.options li:hover, [role="option"]:hover,
ul.options li.selected, [role="option"][aria-selected="true"] {
  background: var(--fg-accent) !important;
  color: #ffffff !important;
}
/* File upload empty states */
.upload-container, .empty, .or {
  color: var(--fg-muted) !important;
}
/* Toast: keep readable on both light Soft toasts and dark surfaces */
.toast-wrap .toast-body {
  background: #1f1f1f !important;
  border: 1px solid var(--fg-border) !important;
}
.toast-title, .toast-text, .toast-icon, .toast-close {
  color: #ffffff !important;
  font-size: 14px;
}
nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
  padding: 10px 20px; box-sizing: border-box;
  display: flex; align-items: center; gap: 16px;
  background: rgba(10,10,10,0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--fg-border);
}
nav img { height: 36px; width: 36px; border-radius: 10px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.brand-lockup { display: flex; flex-direction: column; line-height: 1.15; gap: 2px; }
.brand-title {
  font-family: "Manrope", ui-sans-serif, system-ui, sans-serif;
  font-weight: 700;
  font-size: 18px;
  letter-spacing: -0.02em;
  color: var(--fg-ink);
}
.brand-title span { color: var(--fg-accent); }
.brand-sub {
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--fg-muted);
}
.flexible { flex-grow: 1; }
.wizard-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.wizard-step {
  display: flex;
  align-items: center;
  gap: 6px;
  opacity: 0.45;
  color: var(--fg-muted);
}
.wizard-step.active { opacity: 1; color: var(--fg-ink); }
.wizard-step.done { opacity: 0.85; color: var(--fg-ink); }
.wizard-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 999px;
  border: 1px solid var(--fg-border);
  font-size: 11px;
  font-weight: 700;
}
.wizard-step.active .wizard-num {
  background: var(--fg-accent);
  border-color: var(--fg-accent);
  color: white;
}
.wizard-step.done .wizard-num {
  background: rgba(255,42,31,0.25);
  border-color: var(--fg-accent);
  color: white;
}
.wizard-label {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.wizard-sep {
  width: 18px;
  height: 1px;
  background: var(--fg-border);
}
.nav-links {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-left: 14px;
  flex-shrink: 0;
}
.nav-ext-link {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--fg-muted);
  text-decoration: none;
  white-space: nowrap;
}
.nav-ext-link:hover {
  color: var(--fg-accent);
}
.wizard-nav {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--fg-border);
}
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-body { border: none !important; }
.tabs { margin-top: 56px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: 600 !important; color: #e5e5e5 !important; }
#start_training {
  background: var(--fg-accent) !important;
  color: white !important;
  border: none !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}
#start_training:hover { background: var(--fg-accent-hover) !important; }
#start_training.clicked { background: #525252 !important; color: white !important; }
#wizard_next {
  background: var(--fg-accent) !important;
  color: white !important;
  border: none !important;
}
#wizard_next:hover { background: var(--fg-accent-hover) !important; }
#delete_selected {
  background: var(--fg-accent) !important;
  color: white !important;
  border: none !important;
  font-weight: 600 !important;
}
#delete_selected:hover { background: var(--fg-accent-hover) !important; }
#stop_training {
  background: #3f3f46 !important;
  color: white !important;
  border: 1px solid #fb7185 !important;
  font-weight: 600 !important;
}
#stop_training:hover { background: #52525b !important; }

.train-status-card {
  max-width: 820px;
  margin: 24px auto 12px;
  padding: 28px 32px;
  border-radius: 16px;
  background: var(--fg-surface);
  border: 1px solid var(--fg-border);
  box-shadow: none;
  text-align: center;
}
.train-status-card.done { border-color: rgba(52,211,153,0.45); }
.train-status-card.failed { border-color: rgba(251,113,133,0.5); }
.phase-chip {
  display: inline-block;
  padding: 5px 12px;
  border-radius: 8px;
  background: var(--fg-accent-soft);
  color: var(--fg-accent);
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 18px;
  animation: pulse-soft 2.4s ease-in-out infinite;
}
.train-status-card.done .phase-chip {
  background: rgba(52,211,153,0.14);
  color: var(--fg-ok);
  animation: none;
}
.train-status-card.failed .phase-chip {
  background: rgba(251,113,133,0.14);
  color: var(--fg-bad);
  animation: none;
}
.train-error {
  text-align: left;
  margin: 0 auto 20px;
  max-width: 640px;
  padding: 14px 16px;
  border-radius: 10px;
  background: rgba(251,113,133,0.1);
  border: 1px solid rgba(251,113,133,0.35);
}
.train-error-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--fg-bad);
  margin-bottom: 6px;
}
.train-error-message {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 13px;
  font-weight: 500;
  line-height: 1.45;
  color: #fecdd3;
  white-space: pre-wrap;
  word-break: break-word;
}
.train-error-hint {
  margin-top: 8px;
  font-size: 12px;
  font-weight: 500;
  color: #fda4af;
  opacity: 0.9;
}
.train-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}
.stat-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--fg-muted);
  margin-bottom: 6px;
}
.stat-value {
  font-family: "Manrope", ui-sans-serif, system-ui, sans-serif;
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--fg-ink);
  font-variant-numeric: tabular-nums;
}
.stat-value.subtle { font-size: 20px; color: #d4d4d4; }
.train-detail {
  margin: 4px 0 14px;
  font-size: 13px;
  font-weight: 500;
  color: var(--fg-muted);
  word-break: break-all;
}
.train-progress-bar {
  height: 10px;
  border-radius: 999px;
  background: #2a2a2a;
  overflow: hidden;
  position: relative;
}
.train-progress-fill {
  height: 100%;
  border-radius: 999px;
  background: var(--fg-accent);
  transition: width 0.35s ease;
}
.train-progress-fill.is-indeterminate,
.train-progress-bar.is-indeterminate .train-progress-fill {
  width: 40% !important;
  animation: progress-indeterminate 1.4s ease-in-out infinite;
}
@keyframes progress-indeterminate {
  0% { transform: translateX(-30%); }
  100% { transform: translateX(280%); }
}
.train-progress-pct {
  margin-top: 10px;
  font-family: "Manrope", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--fg-ink);
  font-variant-numeric: tabular-nums;
}
.train-gpu {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--fg-border);
  text-align: left;
}
.train-gpu.unavailable { opacity: 0.7; }
.train-gpu-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--fg-ink);
  margin-bottom: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.train-gpu-row {
  display: flex;
  align-items: center;
  gap: 16px;
}
.train-gpu-watts {
  flex-shrink: 0;
  font-size: 13px;
  font-weight: 700;
  color: var(--fg-accent);
  font-variant-numeric: tabular-nums;
  min-width: 4.5em;
}
.train-gpu-mem-wrap {
  flex: 1;
  min-width: 0;
}
.train-gpu-mem {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: var(--fg-muted);
  font-variant-numeric: tabular-nums;
  margin-bottom: 4px;
}
.train-gpu-bar {
  height: 6px;
  border-radius: 999px;
  background: #2a2a2a;
  overflow: hidden;
}
.train-gpu-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: var(--fg-accent);
}
.train-status-card.mode-download .phase-chip,
.train-status-card.mode-fp8 .phase-chip {
  animation: pulse-soft 1.6s ease-in-out infinite;
}
#training_panel { min-height: 60vh; }
#debug_log textarea {
  font-family: "JetBrains Mono", ui-monospace, monospace !important;
  font-size: 12px !important;
}
"""

js = """
function() {
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');

    if (!document.getElementById('fluxgym-r-fonts')) {
      const link = document.createElement('link');
      link.id = 'fluxgym-r-fonts';
      link.rel = 'stylesheet';
      link.href = 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap';
      document.head.appendChild(link);
    }

    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        const refresh = document.querySelector("#refresh");
        if (refresh) refresh.click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    const startBtn = document.querySelector("#start_training");
    if (startBtn) {
      startBtn.addEventListener("click", (e) => {
        e.target.classList.add("clicked");
        e.target.innerHTML = "Training...";
        const logo = document.querySelector("#logo");
        if (logo) logo.classList.add("rotate");
      });
    }

    if (window.fluxgymLogoWatch) {
      window.clearInterval(window.fluxgymLogoWatch);
    }
    window.fluxgymOpenedFailedLog = false;
    window.fluxgymLogoWatch = window.setInterval(function() {
      const card = document.querySelector(".train-status-card");
      const logo = document.querySelector("#logo");
      if (!card) return;
      if (logo && (card.classList.contains("done") || card.classList.contains("failed"))) {
        logo.classList.remove("rotate");
      }
      if (card.classList.contains("failed") && !window.fluxgymOpenedFailedLog) {
        window.fluxgymOpenedFailedLog = true;
        const wraps = document.querySelectorAll("#training_panel .label-wrap");
        for (const wrap of wraps) {
          const label = (wrap.textContent || "").toLowerCase();
          if (label.includes("debug") && !wrap.classList.contains("open")) {
            wrap.click();
            break;
          }
        }
      }
      if (!card.classList.contains("failed")) {
        window.fluxgymOpenedFailedLog = false;
      }
    }, 1000);
}"""

current_account = account_hf()
print(f"current_account={current_account}")

with gr.Blocks(elem_id="app", title="FluxGym-R", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            wizard_step = gr.State(1)
            wizard_indicator = gr.HTML(wizard_indicator_html(1), elem_id="wizard_indicator")

            with gr.Column(visible=True, elem_id="setup_panel") as setup_panel:
                model_names = list(models.keys())
                print(f"model_names={model_names}")

                with gr.Column(visible=True, elem_id="wizard_step1") as step1:
                    gr.Markdown(
                        """# 1. Upload images
<p style="margin-top:0">Add at least 2 training images (4–30 is ideal).</p>
""",
                        elem_classes="group_padding",
                    )
                    images = gr.File(
                        file_types=["image", ".txt"],
                        label="Upload your images",
                        file_count="multiple",
                        interactive=True,
                        visible=True,
                        scale=1,
                    )

                with gr.Column(visible=False, elem_id="wizard_step2") as step2:
                    gr.Markdown(
                        """# 2. LoRA settings
<p style="margin-top:0">Name your LoRA and set the trigger word used in captions.</p>
""",
                        elem_classes="group_padding",
                    )
                    lora_name = gr.Textbox(
                        label="The name of your LoRA",
                        info="This has to be a unique name",
                        placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True,
                    )
                    base_model = gr.Dropdown(
                        label="Base model (edit the models.yaml file to add more to this list)",
                        choices=model_names,
                        value=model_names[0],
                    )
                    vram = gr.Radio(
                        ["32G Quality", "32G Safe", "20G", "16G", "12G"],
                        value="32G Quality",
                        label="VRAM",
                        info="32G Quality: bf16 base, best fidelity/color. 32G Safe: fp8 + more swap, OOM-resistant.",
                        interactive=True,
                    )
                    num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
                    max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
                    total_steps = gr.Number(0, interactive=False, label="Expected training steps")
                    sample_prompts = gr.Textbox(
                        "", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True
                    )
                    sample_every_n_steps = gr.Number(
                        0, precision=0, label="Sample Image Every N Steps", interactive=True
                    )
                    resolution = gr.Number(value=512, precision=0, label="Resize dataset images")

                with gr.Column(visible=False, elem_id="wizard_step3") as step3:
                    gr.Markdown(
                        """# 3. Caption images
<p style="margin-top:0">Generate or edit captions. The trigger word is required in every caption.</p>
""",
                        elem_classes="group_padding",
                    )
                    with gr.Group(visible=True) as captioning_area:
                        caption_model = gr.Dropdown(
                            label="Caption model",
                            choices=CAPTION_MODELS,
                            value="JoyCaption",
                            interactive=True,
                        )
                        do_captioning = gr.Button("Add AI captions", variant="primary")
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"Caption {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])

                with gr.Column(visible=False, elem_id="wizard_step4") as step4:
                    gr.Markdown(
                        """# 4. Review & train
<p style="margin-top:0">Check the generated script, tweak advanced options, then start training.</p>
""",
                        elem_classes="group_padding",
                    )
                    refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                    start = gr.Button("Start training", visible=True, elem_id="start_training", variant="primary")
                    train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
                    with gr.Accordion("Advanced options", elem_id="advanced_options", open=False):
                        with gr.Row():
                            with gr.Column(min_width=300):
                                seed = gr.Number(label="--seed", info="Seed", value=42, interactive=True)
                            with gr.Column(min_width=300):
                                workers = gr.Number(
                                    label="--max_data_loader_n_workers",
                                    info="Number of Workers",
                                    value=2,
                                    interactive=True,
                                )
                            with gr.Column(min_width=300):
                                learning_rate = gr.Textbox(
                                    label="--learning_rate",
                                    info="R9700 / ROCm default (was 8e-4). Try 5e-5 if still oversaturated.",
                                    value="8e-5",
                                    interactive=True,
                                )
                            with gr.Column(min_width=300):
                                save_every_n_epochs = gr.Number(
                                    label="--save_every_n_epochs",
                                    info="Save every N epochs",
                                    value=4,
                                    interactive=True,
                                )
                            with gr.Column(min_width=300):
                                guidance_scale = gr.Number(
                                    label="--guidance_scale", info="Guidance Scale", value=1.0, interactive=True
                                )
                            with gr.Column(min_width=300):
                                timestep_sampling = gr.Textbox(
                                    label="--timestep_sampling",
                                    info="Timestep Sampling",
                                    value="shift",
                                    interactive=True,
                                )
                            with gr.Column(min_width=300):
                                network_dim = gr.Number(
                                    label="--network_dim",
                                    info="LoRA Rank",
                                    value=4,
                                    minimum=4,
                                    maximum=128,
                                    step=4,
                                    interactive=True,
                                )
                            with gr.Column(min_width=300):
                                network_alpha = gr.Number(
                                    label="--network_alpha",
                                    info="Keep low (1 or ~dim/2). High alpha amplifies LoRA and ROCm math noise.",
                                    value=1,
                                    minimum=1,
                                    maximum=128,
                                    step=1,
                                    interactive=True,
                                )
                            advanced_components, advanced_component_ids = init_advanced()

                with gr.Row(elem_classes="wizard-nav", elem_id="wizard_nav_row") as wizard_nav:
                    wizard_back_btn = gr.Button("Back", interactive=False, elem_id="wizard_back")
                    wizard_next_btn = gr.Button("Next", variant="primary", elem_id="wizard_next")

            with gr.Column(visible=False, elem_id="training_panel") as training_panel:
                train_status = gr.HTML(
                    render_status_card(TrainProgress(phase="Preparing")), elem_id="train_status"
                )
                with gr.Row():
                    stop_btn = gr.Button(
                        "Stop training",
                        visible=False,
                        variant="stop",
                        elem_id="stop_training",
                    )
                    back_btn = gr.Button("Back to setup", visible=False, elem_id="back_to_setup")
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="Samples", every=10, columns=6)
                with gr.Accordion("Debug log", open=False):
                    debug_log = gr.Textbox(
                        label="Raw training output",
                        lines=12,
                        max_lines=20,
                        interactive=False,
                        elem_id="debug_log",
                    )

        with gr.TabItem("Publish") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token", type="password")
            hf_login = gr.Button("Login")
            hf_logout = gr.Button("Logout")
            with gr.Row() as row:
                gr.Markdown("**LoRA**")
                gr.Markdown("**Upload**")
            loras = get_loras()
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="Account", interactive=False)
                        repo_name = gr.Textbox(label="Repository Name")
                    repo_visibility = gr.Radio(
                        choices=["public", "private"],
                        value="public",
                        label="Repository visibility",
                    )
                    upload_button = gr.Button("Upload to HuggingFace")
                    upload_button.click(
                        fn=upload_hf,
                        inputs=[
                            base_model,
                            lora_rows,
                            repo_owner,
                            repo_name,
                            repo_visibility,
                            hf_token,
                        ],
                    )
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])

        with gr.TabItem("Download") as download_tab:
            gr.Markdown(
                """# Download checkpoints
Select a training run to download intermediate epoch saves or the final LoRA.
"""
            )
            _download_choices = get_download_run_choices()
            _download_default = _download_choices[0] if _download_choices else None
            _download_summary, _download_files = load_download_checkpoints(_download_default)
            with gr.Row():
                download_run = gr.Dropdown(
                    label="Training run",
                    choices=_download_choices,
                    value=_download_default,
                    interactive=True,
                    scale=4,
                )
                download_refresh = gr.Button("Refresh", scale=1)
            download_summary = gr.Markdown(_download_summary)
            download_files = gr.File(
                label="Checkpoints",
                file_count="multiple",
                interactive=False,
                value=_download_files,
            )

        with gr.TabItem("Delete") as delete_tab:
            gr.Markdown(
                """# Delete projects
Select one or more projects to permanently remove their **outputs** and **datasets** folders.
"""
            )
            _delete_choices = delete_project_choices()
            delete_projects_box = gr.CheckboxGroup(
                label="Projects",
                choices=_delete_choices,
                value=[],
                interactive=True,
            )
            delete_confirm = gr.Checkbox(
                label="I understand this permanently deletes the selected projects",
                value=False,
            )
            with gr.Row():
                delete_refresh = gr.Button("Refresh", scale=1)
                delete_btn = gr.Button(
                    "Delete selected",
                    variant="primary",
                    elem_id="delete_selected",
                    scale=2,
                )
            delete_status = gr.Markdown(
                "_Select one or more projects, confirm, then delete._"
                if _delete_choices
                else "_No projects to delete._"
            )

    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])

    download_tab.select(
        fn=refresh_download_runs,
        inputs=[download_run],
        outputs=[download_run, download_summary, download_files],
    )
    download_refresh.click(
        fn=refresh_download_runs,
        inputs=[download_run],
        outputs=[download_run, download_summary, download_files],
    )
    download_run.change(
        fn=load_download_checkpoints,
        inputs=[download_run],
        outputs=[download_summary, download_files],
    )

    delete_tab.select(
        fn=refresh_delete_projects,
        outputs=[delete_projects_box, delete_status],
    )
    delete_refresh.click(
        fn=refresh_delete_projects,
        outputs=[delete_projects_box, delete_status],
    )
    delete_btn.click(
        fn=delete_projects,
        inputs=[delete_projects_box, delete_confirm],
        outputs=[delete_projects_box, delete_status],
    ).then(
        fn=lambda: gr.update(value=False),
        outputs=[delete_confirm],
    )

    dataset_folder = gr.State()

    wizard_outputs = [
        wizard_step,
        setup_panel,
        training_panel,
        wizard_indicator,
        step1,
        step2,
        step3,
        step4,
        wizard_nav,
        wizard_back_btn,
        wizard_next_btn,
    ]

    listeners = [
        base_model,
        lora_name,
        resolution,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        network_alpha,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    ]
    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]

    images.upload(load_captioning, inputs=[images, concept_sentence], outputs=output_components)
    images.delete(load_captioning, inputs=[images, concept_sentence], outputs=output_components)
    images.clear(hide_captioning, outputs=output_components)

    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps],
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps],
    )
    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps],
    )
    images.delete(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps],
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps],
    )
    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)

    wizard_back_btn.click(fn=wizard_back, inputs=[wizard_step], outputs=wizard_outputs)
    wizard_next_btn.click(
        fn=wizard_next,
        inputs=[wizard_step, images, lora_name, concept_sentence] + caption_list,
        outputs=wizard_outputs + caption_list,
    )

    def enforce_captions_for_train(concept_sentence, images, *captions):
        n = _count_uploaded_images(images)
        fixed = []
        for i, cap in enumerate(captions):
            if i < n:
                text = ensure_trigger_in_caption(cap, concept_sentence)
                if not text.strip():
                    raise gr.Error(f"Caption {i + 1} is empty. Go back and caption every image.")
                fixed.append(text)
            else:
                fixed.append(cap)
        return fixed

    train_event_outputs = [
        setup_panel,
        training_panel,
        train_status,
        debug_log,
        back_btn,
        stop_btn,
        wizard_step,
        wizard_indicator,
        wizard_nav,
    ]

    start.click(
        fn=enforce_captions_for_train,
        inputs=[concept_sentence, images] + caption_list,
        outputs=caption_list,
    ).then(
        fn=create_dataset,
        inputs=[dataset_folder, resolution, images] + caption_list,
        outputs=dataset_folder,
    ).then(
        fn=start_training,
        inputs=[
            base_model,
            lora_name,
            train_script,
            train_config,
            sample_prompts,
            max_train_epochs,
            total_steps,
        ],
        outputs=train_event_outputs,
    )
    stop_btn.click(fn=request_stop_training, outputs=[stop_btn])
    back_btn.click(
        fn=back_to_setup,
        outputs=[
            setup_panel,
            training_panel,
            train_status,
            debug_log,
            back_btn,
            stop_btn,
            wizard_step,
            wizard_indicator,
            step1,
            step2,
            step3,
            step4,
            wizard_nav,
            wizard_back_btn,
            wizard_next_btn,
        ],
        js="() => { const logo = document.querySelector('#logo'); if (logo) logo.classList.remove('rotate'); const start = document.querySelector('#start_training'); if (start) { start.classList.remove('clicked'); start.innerHTML = 'Start training'; } }",
    )
    do_captioning.click(
        fn=run_captioning,
        inputs=[images, concept_sentence, caption_model] + caption_list,
        outputs=caption_list,
    )
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner])
    refresh.click(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])
if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(debug=True, show_error=True, allowed_paths=[cwd])
