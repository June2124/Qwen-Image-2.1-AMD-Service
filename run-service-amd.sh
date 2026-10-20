#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${QWEN_IMAGE:-qwen-image-2.1:amd-turbo-20261019}"
CONTAINER="${QWEN_CONTAINER:-qwen-image-2.1-amd}"
MODEL_ROOT="${MODEL_ROOT:-/home/jetlinks/models/Qwen-Image-2.1-ComfyUI}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${ROOT}/runtime}"
PORT="${QWEN_PORT:-18036}"
API_KEY="${QWEN_API_KEY:?set QWEN_API_KEY}"

required=(
  "${MODEL_ROOT}/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors"
  "${MODEL_ROOT}/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
  "${MODEL_ROOT}/vae/qwen_image_2.1_vae_bf16.safetensors"
)
for path in "${required[@]}"; do
  [[ -s "${path}" ]] || { echo "missing model: ${path}" >&2; exit 1; }
done

if [[ "${BUILD_IMAGE:-1}" == "1" ]]; then
  "${ROOT}/scripts/prepare-comfyui.sh"
  docker build -f "${ROOT}/Dockerfile.amd" -t "${IMAGE}" "${ROOT}"
fi

mkdir -p "${RUNTIME_ROOT}/input" "${RUNTIME_ROOT}/output" "${RUNTIME_ROOT}/user"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d \
  --name "${CONTAINER}" \
  --restart unless-stopped \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -p "${PORT}:8801" \
  -e QWEN_API_KEY="${API_KEY}" \
  -e QWEN_IDLE_UNLOAD_SECONDS="${QWEN_IDLE_UNLOAD_SECONDS:-60}" \
  -e QWEN_INFERENCE_TIMEOUT="${QWEN_INFERENCE_TIMEOUT:-1800}" \
  -e COMFY_STARTUP_TIMEOUT="${COMFY_STARTUP_TIMEOUT:-300}" \
  -e COMFY_RESERVE_VRAM_GB="${COMFY_RESERVE_VRAM_GB:-1.0}" \
  -e QWEN_CACHE_DEVICE="${QWEN_CACHE_DEVICE:-auto}" \
  -e QWEN_CACHE_DTYPE="${QWEN_CACHE_DTYPE:-default}" \
  -e QWEN_REFERENCE_MAX_SIDE="${QWEN_REFERENCE_MAX_SIDE:-640}" \
  -e COMFY_KITCHEN_DISABLE_HIP="${COMFY_KITCHEN_DISABLE_HIP:-1}" \
  -e PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  -v "${MODEL_ROOT}/diffusion_models:/opt/ComfyUI/models/diffusion_models:ro" \
  -v "${MODEL_ROOT}/text_encoders:/opt/ComfyUI/models/text_encoders:ro" \
  -v "${MODEL_ROOT}/vae:/opt/ComfyUI/models/vae:ro" \
  -v "${RUNTIME_ROOT}:/data" \
  --entrypoint python \
  "${IMAGE}" \
  -m uvicorn qwen_service:app --host 0.0.0.0 --port 8801 --workers 1

echo "Started ${CONTAINER} on http://0.0.0.0:${PORT}"
