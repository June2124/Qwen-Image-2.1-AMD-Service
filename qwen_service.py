"""FastAPI facade for the official ComfyUI Qwen-Image-2.1 workflow."""

from __future__ import annotations

import base64
import binascii
import hmac
import io
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Body, Depends, FastAPI, Header, HTTPException
from PIL import Image, UnidentifiedImageError

from comfy_manager import (
    ComfyProcessManager,
    ComfyServiceError,
    ComfyServiceTimeout,
)
from comfy_workflow import (
    IMAGE_SIZE_MULTIPLE,
    MODEL_NAME,
    WorkflowValidationError,
    canvas_workflow_size,
    parse_size,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen-image-service")

API_KEY = os.getenv("QWEN_API_KEY", "").strip()
TEXT_TO_IMAGE_SIZE = "640x640"
DEFAULT_STEPS = int(os.getenv("QWEN_DEFAULT_STEPS", "25"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("QWEN_DEFAULT_GUIDANCE_SCALE", "1.0"))
MAX_IMAGE_BYTES = int(os.getenv("QWEN_MAX_IMAGE_BYTES", str(30 * 1024 * 1024)))
MAX_TOTAL_IMAGE_BYTES = int(
    os.getenv("QWEN_MAX_TOTAL_IMAGE_BYTES", str(150 * 1024 * 1024))
)

MANAGER = ComfyProcessManager()


def _authorize(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    scheme, _, token = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or isinstance(value, float):
        raise HTTPException(status_code=400, detail="{} must be an integer".format(name))
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="{} must be an integer".format(name)) from exc
    if result < minimum or result > maximum:
        raise HTTPException(
            status_code=400,
            detail="{} must be between {} and {}".format(name, minimum, maximum),
        )
    return result


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail="{} must be a number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="{} must be a number".format(name)) from exc
    if result < minimum or result > maximum:
        raise HTTPException(
            status_code=400,
            detail="{} must be between {} and {}".format(name, minimum, maximum),
        )
    return result


def _decode_image(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail="{} must be a Base64 image".format(name))
    encoded = value.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower() or not header.lower().startswith("data:image/"):
            raise HTTPException(status_code=400, detail="{} data URL is invalid".format(name))
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 16:
        raise HTTPException(status_code=413, detail="{} is too large".format(name))
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="{} Base64 is invalid".format(name)) from exc
    if not raw:
        raise HTTPException(status_code=400, detail="{} is empty".format(name))
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="{} is too large".format(name))
    return raw


def _image_values(input_data: dict, mode: str) -> tuple[List[bytes], int]:
    explicit = "canvas_image" in input_data or "reference_images" in input_data
    legacy = "image" in input_data or "images" in input_data
    if explicit and legacy:
        raise HTTPException(
            status_code=400,
            detail="do not mix canvas/reference fields with legacy image fields",
        )

    canvas = input_data.get("canvas_image") if explicit else None
    references = input_data.get("reference_images", []) if explicit else []
    if explicit:
        if references is None:
            references = []
        elif isinstance(references, str):
            references = [references]
        elif not isinstance(references, list):
            raise HTTPException(status_code=400, detail="reference_images must be an array")
    elif "images" in input_data:
        values = input_data.get("images")
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            raise HTTPException(status_code=400, detail="images must be an array")
        canvas = values[0] if values else None
        references = values[1:]
    elif "image" in input_data:
        canvas = input_data.get("image")

    if mode == "text-to-image":
        if canvas is not None or references:
            raise HTTPException(status_code=400, detail="text-to-image does not accept input images")
        return [], 0
    if canvas is None:
        raise HTTPException(status_code=400, detail="image-to-image requires canvas_image")
    if len(references) > 9:
        raise HTTPException(status_code=400, detail="at most 9 reference_images are supported")

    decoded = [_decode_image(canvas, "canvas_image")]
    decoded.extend(
        _decode_image(value, "reference_images[{}]".format(index))
        for index, value in enumerate(references)
    )
    if sum(len(value) for value in decoded) > MAX_TOTAL_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="combined input images are too large")
    return decoded, len(references)


def _image_size(raw: bytes, name: str) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError
            return image.width, image.height
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(status_code=400, detail="{} is invalid".format(name)) from exc


def _payload(mode: str, payload: Dict[str, Any]) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    model = str(payload.get("model") or MODEL_NAME).strip().lower()
    if model != MODEL_NAME:
        raise HTTPException(status_code=400, detail="unsupported model")
    input_data = payload.get("input")
    parameters = payload.get("parameters") or {}
    if not isinstance(input_data, dict):
        raise HTTPException(status_code=400, detail="input must be an object")
    if not isinstance(parameters, dict):
        raise HTTPException(status_code=400, detail="parameters must be an object")

    prompt = input_data.get("prompt")
    negative_prompt = input_data.get("negative_prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt cannot be empty")
    if len(prompt) > 20000:
        raise HTTPException(status_code=400, detail="prompt is too long")
    if not isinstance(negative_prompt, str):
        raise HTTPException(status_code=400, detail="negative_prompt must be a string")
    if len(negative_prompt) > 20000:
        raise HTTPException(status_code=400, detail="negative_prompt is too long")

    steps = _integer(parameters.get("steps", DEFAULT_STEPS), "steps", 1, 60)
    guidance_value = parameters.get(
        "guidance_scale", parameters.get("cfg", DEFAULT_GUIDANCE_SCALE)
    )
    guidance_scale = _number(guidance_value, "guidance_scale", 1.0, 10.0)
    seed_value = parameters.get("seed")
    seed = (
        secrets.randbelow(0x7FFFFFFFFFFFFFFF)
        if seed_value is None or seed_value == ""
        else _integer(seed_value, "seed", 0, 0xFFFFFFFFFFFFFFFF)
    )
    close_model = parameters.get("close_model", False)
    if not isinstance(close_model, bool):
        raise HTTPException(status_code=400, detail="close_model must be a boolean")
    output_format = str(parameters.get("format", "png")).strip().lower()
    if output_format != "png":
        raise HTTPException(status_code=400, detail="only PNG output is supported")

    images, reference_count = _image_values(input_data, mode)
    try:
        if mode == "image-to-image":
            width, height = _image_size(images[0], "canvas_image")
            workflow_width, workflow_height = canvas_workflow_size(width, height)
        else:
            width, height = parse_size(TEXT_TO_IMAGE_SIZE)
            workflow_width, workflow_height = width, height
    except WorkflowValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "mode": mode,
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "size": "{}x{}".format(width, height),
        "workflow_size": "{}x{}".format(workflow_width, workflow_height),
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "close_model": close_model,
        "images": images,
        "reference_count": reference_count,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    MANAGER.start_reaper()
    logger.info("Qwen Image service ready; ComfyUI will start lazily")
    try:
        yield
    finally:
        MANAGER.shutdown()


app = FastAPI(title="Qwen-Image-2.1 AMD Service", version="1.1.0", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "service": "Qwen-Image-2.1 AMD Service",
        "model": MODEL_NAME,
        "endpoints": [
            "/v1/qwen-image/text-to-image",
            "/v1/qwen-image/image-to-image",
        ],
    }


@app.get("/health")
def health():
    status = MANAGER.status()
    return {
        "status": "ok",
        "ready": all(status["models"].values()),
        "model": MODEL_NAME,
        "text_to_image_size": TEXT_TO_IMAGE_SIZE,
        "image_to_image_size": "original_canvas",
        "image_alignment": IMAGE_SIZE_MULTIPLE,
        "max_input_images": 10,
        "input_roles": {"canvas_image": 1, "reference_images": 9},
        **status,
    }


def _generate(mode: str, payload: Dict[str, Any]) -> dict:
    values = _payload(mode, payload)
    request_id = "qwen-{}".format(uuid.uuid4().hex)
    started = time.time()
    try:
        raw = MANAGER.generate(
            mode=values["mode"],
            prompt=values["prompt"],
            negative_prompt=values["negative_prompt"],
            size=values["workflow_size"],
            steps=values["steps"],
            guidance_scale=values["guidance_scale"],
            seed=values["seed"],
            image_blobs=values["images"],
            close_model=values["close_model"],
            output_size=(values["width"], values["height"])
            if values["mode"] == "image-to-image"
            else None,
        )
    except ComfyServiceTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (ComfyServiceError, WorkflowValidationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    encoded = base64.b64encode(raw).decode("ascii")
    data_url = "data:image/png;base64," + encoded
    return {
        "id": request_id,
        "created": int(time.time()),
        "model": MODEL_NAME,
        "images": [data_url],
        "data": [{"image_base64": data_url}],
        "parameters": {
            "size": values["size"],
            "width": values["width"],
            "height": values["height"],
            "steps": values["steps"],
            "guidance_scale": values["guidance_scale"],
            "seed": values["seed"],
            "workflow_size": values["workflow_size"],
        },
        "input_layout": {
            "canvas_count": 1 if values["mode"] == "image-to-image" else 0,
            "reference_count": values["reference_count"],
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }


@app.post("/v1/qwen-image/text-to-image", dependencies=[Depends(_authorize)])
def text_to_image(payload: Dict[str, Any] = Body(...)):
    return _generate("text-to-image", payload)


@app.post("/v1/qwen-image/image-to-image", dependencies=[Depends(_authorize)])
def image_to_image(payload: Dict[str, Any] = Body(...)):
    return _generate("image-to-image", payload)


@app.post("/v1/qwen-image/unload", dependencies=[Depends(_authorize)])
def unload():
    status = MANAGER.status()
    if status["busy"]:
        raise HTTPException(status_code=409, detail="inference is running")
    stopped = MANAGER.stop(force=False, reason="manual unload")
    return {"success": True, "stopped": stopped, **MANAGER.status()}


__all__ = ["app"]
