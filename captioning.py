"""Image captioning backends for FluxGym-R (Florence-2 and JoyCaption)."""
from __future__ import annotations

import gc
from typing import Optional, Protocol

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration

FLORENCE_MODEL = "multimodalart/Florence-2-large-no-flash-attn"
JOYCAPTION_MODEL = "fancyfeast/llama-joycaption-beta-one-hf-llava"

CAPTION_MODELS = ["JoyCaption", "Florence-2"]


def ensure_trigger_in_caption(caption: Optional[str], trigger: Optional[str]) -> str:
    """Ensure the trigger word/sentence appears in the caption (prepend if missing)."""
    text = (caption or "").strip()
    trig = (trigger or "").strip()
    if not trig:
        return text
    if not text:
        return trig
    if trig.lower() in text.lower():
        return text
    return f"{trig} {text}"


class CaptionBackend(Protocol):
    def load(self, device: str, torch_dtype: torch.dtype) -> None: ...
    def caption(self, image: Image.Image) -> str: ...
    def unload(self) -> None: ...


def _empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Florence2Backend:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.torch_dtype = torch.float32

    def load(self, device: str, torch_dtype: torch.dtype) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)

    def caption(self, image: Image.Image) -> str:
        assert self.model is not None and self.processor is not None
        prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        return parsed["<DETAILED_CAPTION>"].replace("The image shows ", "")

    def unload(self) -> None:
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        _empty_cache()


class JoyCaptionBackend:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.torch_dtype = torch.float32

    def load(self, device: str, torch_dtype: torch.dtype) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        self.processor = AutoProcessor.from_pretrained(JOYCAPTION_MODEL)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            JOYCAPTION_MODEL,
            torch_dtype=torch_dtype,
        ).to(device)
        self.model.eval()

    def caption(self, image: Image.Image) -> str:
        assert self.model is not None and self.processor is not None
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {
                "role": "user",
                "content": "Write a long descriptive caption for this image in a formal tone.",
            },
        ]
        convo_string = self.processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch_dtype)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
            )[0]

        generate_ids = generate_ids[inputs["input_ids"].shape[1] :]
        caption = self.processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return caption.strip()

    def unload(self) -> None:
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        _empty_cache()


def get_backend(caption_model: str) -> CaptionBackend:
    if caption_model == "JoyCaption" or not caption_model:
        return JoyCaptionBackend()
    return Florence2Backend()


def resolve_device_dtype() -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, torch_dtype
