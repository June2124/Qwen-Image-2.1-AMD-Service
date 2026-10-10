"""Build pinned ComfyUI API workflows for Qwen-Image-2.1."""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Tuple


MODEL_NAME = "qwen-image-2.1"
DIFFUSION_MODEL = "qwen_image_2.1_bf16.safetensors"
TEXT_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
VAE_MODEL = "qwen_image_2.1_vae_bf16.safetensors"
IMAGE_SIZE_MULTIPLE = 32
MAX_CANVAS_SIDE = 4096

SIZE_PRESETS: Dict[str, Tuple[int, int]] = {
    "640x640": (640, 640),
    "640x352": (640, 352),
    "352x640": (352, 640),
    "768x768": (768, 768),
    "960x544": (960, 544),
    "544x960": (544, 960),
}


class WorkflowValidationError(ValueError):
    """The requested workflow cannot be represented safely."""


def parse_size(value: str) -> Tuple[int, int]:
    key = str(value or "").strip().lower()
    match = re.fullmatch(r"(\d+)x(\d+)", key)
    if match is None:
        raise WorkflowValidationError(
            "size must use WIDTHxHEIGHT with 32-pixel aligned dimensions"
        )
    width, height = (int(match.group(1)), int(match.group(2)))
    if (
        width < IMAGE_SIZE_MULTIPLE
        or height < IMAGE_SIZE_MULTIPLE
        or width > MAX_CANVAS_SIDE
        or height > MAX_CANVAS_SIDE
        or width % IMAGE_SIZE_MULTIPLE
        or height % IMAGE_SIZE_MULTIPLE
    ):
        raise WorkflowValidationError(
            "size dimensions must be 32-pixel multiples between 32 and {}".format(
                MAX_CANVAS_SIDE
            )
        )
    return width, height


def canvas_workflow_size(width: int, height: int) -> Tuple[int, int]:
    """Pad canvas dimensions up to Qwen's alignment without resizing it."""
    if width <= 0 or height <= 0:
        raise WorkflowValidationError("canvas dimensions must be positive")
    aligned_width = int(math.ceil(width / IMAGE_SIZE_MULTIPLE)) * IMAGE_SIZE_MULTIPLE
    aligned_height = int(math.ceil(height / IMAGE_SIZE_MULTIPLE)) * IMAGE_SIZE_MULTIPLE
    if aligned_width > MAX_CANVAS_SIDE or aligned_height > MAX_CANVAS_SIDE:
        raise WorkflowValidationError(
            "canvas dimensions must not exceed {} pixels per side".format(
                MAX_CANVAS_SIDE
            )
        )
    return aligned_width, aligned_height


def reference_resolution(width: int, height: int) -> int:
    """Return a 32-aligned square pixel budget for conditioning images."""
    resolution = int(round(math.sqrt(width * height) / 32.0) * 32)
    return max(256, min(1024, resolution))


def build_workflow(
    *,
    mode: str,
    prompt: str,
    negative_prompt: str,
    size: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    image_names: Iterable[str] = (),
    filename_prefix: str = "qwen-image-2.1",
    diffusion_model: str = DIFFUSION_MODEL,
    text_encoder: str = TEXT_ENCODER,
    vae_model: str = VAE_MODEL,
    cache_device: str = "auto",
    cache_dtype: str = "default",
) -> dict:
    """Create an API-format workflow using only built-in ComfyUI nodes."""
    if mode not in ("text-to-image", "image-to-image"):
        raise WorkflowValidationError("unsupported generation mode")
    image_names = list(image_names)
    if mode == "text-to-image" and image_names:
        raise WorkflowValidationError("text-to-image does not accept input images")
    if mode == "image-to-image" and not image_names:
        raise WorkflowValidationError("image-to-image requires an editing canvas")
    if len(image_names) > 10:
        raise WorkflowValidationError("at most 10 input images are supported")
    if not str(prompt or "").strip():
        raise WorkflowValidationError("prompt cannot be empty")

    width, height = parse_size(size)
    encode_inputs = {
        "clip": ["3", 0],
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "vae": ["4", 0],
        # With resolution=0, each edit input keeps its staged dimensions. The
        # first image therefore defines the canvas latent without stretching,
        # while material references can remain independently downscaled.
        "resolution": 0 if mode == "image-to-image" else reference_resolution(width, height),
    }
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "QwenImage21Cache",
            "inputs": {
                "model": ["1", 0],
                "device": cache_device,
                "dtype": cache_dtype,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_model},
        },
        "5": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": encode_inputs,
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": seed,
                "steps": steps,
                "cfg": guidance_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["5", 2] if mode == "image-to-image" else ["6", 0],
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["4", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix},
        },
    }

    for index, image_name in enumerate(image_names, start=1):
        node_id = str(20 + index)
        workflow[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }
        encode_inputs["images.image_{}".format(index)] = [node_id, 0]
    return workflow


__all__ = [
    "MODEL_NAME",
    "DIFFUSION_MODEL",
    "TEXT_ENCODER",
    "VAE_MODEL",
    "IMAGE_SIZE_MULTIPLE",
    "MAX_CANVAS_SIDE",
    "SIZE_PRESETS",
    "WorkflowValidationError",
    "parse_size",
    "canvas_workflow_size",
    "reference_resolution",
    "build_workflow",
]
