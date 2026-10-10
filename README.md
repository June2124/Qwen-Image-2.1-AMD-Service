# Qwen-Image-2.1 AMD FastAPI Service

This service exposes the official ComfyUI split-weight Qwen-Image-2.1 workflow
through a small authenticated FastAPI API. It is configured for an AMD Radeon
AI PRO R9700 (`gfx1201`) and a single 32 GB GPU.

## Input roles

Image editing deliberately separates the two roles that a flat image array
would otherwise hide:

- `canvas_image`: exactly one required editing canvas. It is always sent to
  Qwen as `<image1>`.
- `reference_images`: zero to nine optional material/style/object references.
  They are sent as `<image2>` through `<image10>` in list order.

The edit prompt can refer to those image numbers explicitly. Text-to-image
accepts no input images.

Output dimensions are role-specific and are not caller-configurable:

- Text-to-image always generates `640x640`.
- Image-to-image preserves the editing canvas dimensions exactly. Internally,
  the service edge-pads the right and bottom to a 32-pixel multiple for Qwen,
  then crops the generated PNG back to the original canvas dimensions.
- Reference materials are independently downscaled to a maximum side of 640
  pixels by default. They never determine or stretch the output canvas.

## Runtime design

- FastAPI stays resident but does not load model weights.
- The first inference lazily starts a private ComfyUI child process.
- Requests are serialized; the GPU never runs two generations concurrently.
- The ComfyUI process is terminated after the configured idle period, releasing
  its RAM and VRAM. `parameters.close_model=true` releases it immediately after
  the response image has been collected.
- A failed or timed-out inference also terminates the child process so the next
  request starts from a clean runtime.

The pinned ComfyUI revision is
`1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8`. The Docker build applies the
unmerged upstream ComfyUI PR 16603 patch from
`patches/comfyui-pr-16603-qwen-vl-linear.patch`. It replaces a Qwen-VL patch
embedding Conv3D with the equivalent linear operation to avoid the very slow
AMD `slow_conv3d` path on `gfx1201`; model state-dict keys and math are kept.

## Models

The deployment uses:

- `diffusion_models/qwen_image_2.1_bf16.safetensors`
- `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- `vae/qwen_image_2.1_vae_bf16.safetensors`

The BF16 diffusion model is intentional. The INT8 ConvRot diffusion checkpoint
has produced invalid/black results on comparable `gfx1201` ROCm environments.
The text encoder remains INT8 to fit the complete workflow in 32 GB VRAM with
ComfyUI offload.

On this `gfx1201` host the service disables the `comfy-kitchen` HIP backend and
uses its eager fallback. ComfyUI asynchronous offload and pinned memory are also
disabled because that combination caused a GPU page fault immediately after the
INT8 text encoder loaded. Standard model offload remains enabled.

Run the resumable, size-verified ModelScope download:

```bash
./download-models.sh
```

## Build and run

Prepare the pinned ComfyUI source, build the image, and start the container:

```bash
export QWEN_API_KEY='replace-me'
./run-service-amd.sh
```

`scripts/prepare-comfyui.sh` accepts `COMFYUI_ARCHIVE=/path/to/archive.tar.gz`
for hosts that cannot reach GitHub. The container listens on host port `18036`
by default. Important environment variables include:

| Variable | Default |
| --- | --- |
| `QWEN_IDLE_UNLOAD_SECONDS` | `60` |
| `QWEN_INFERENCE_TIMEOUT` | `1800` |
| `COMFY_STARTUP_TIMEOUT` | `300` |
| `COMFY_RESERVE_VRAM_GB` | `1.0` |
| `QWEN_CACHE_DEVICE` | `auto` |
| `QWEN_CACHE_DTYPE` | `default` |
| `QWEN_REFERENCE_MAX_SIDE` | `640` |
| `COMFY_KITCHEN_DISABLE_HIP` | `1` |

Do not enable the experimental `int8` Qwen cache on this AMD deployment.
Multi-reference editing reproduced corrupted purple/green fragment images with
that cache dtype; the normal cache path produced valid images with the same
inputs.

## API

Endpoints:

- `POST /v1/qwen-image/text-to-image`
- `POST /v1/qwen-image/image-to-image`
- `POST /v1/qwen-image/unload`
- `GET /health`

Example edit request:

```json
{
  "model": "qwen-image-2.1",
  "input": {
    "prompt": "Keep <image1> unchanged except place the bin from <image2> on the floor.",
    "negative_prompt": "floating objects, stickers, distorted perspective",
    "canvas_image": "data:image/png;base64,...",
    "reference_images": ["data:image/png;base64,..."]
  },
  "parameters": {
    "steps": 40,
    "guidance_scale": 2.0,
    "seed": 1234
  }
}
```

Legacy `size`, `width`, and `height` request parameters are ignored. The sampler
is fixed to Euler/simple, matching the official Qwen workflow. CFG `1.0` is the
official path; negative prompts only influence the result when CFG is raised
above `1.0`.

Run the workflow unit tests with:

```bash
python -m unittest discover -s tests -v
```

## License notice

Qwen-Image-2.1 is distributed under the Qwen Research License. Its published
terms restrict use to non-commercial research unless a separate license is
obtained. ComfyUI and each bundled dependency keep their own licenses. Review
all applicable terms before production or commercial use.
