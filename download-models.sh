#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/home/jetlinks/models/Qwen-Image-2.1-ComfyUI}"
mkdir -p "${MODEL_ROOT}/diffusion_models" "${MODEL_ROOT}/text_encoders" "${MODEL_ROOT}/vae"

download() {
  local url="$1" target="$2" expected="$3" current=0 actual
  [[ -f "${target}" ]] && current="$(stat -c %s "${target}")"
  if [[ "${current}" == "${expected}" ]]; then
    echo "READY ${target} ${current}"
    return
  fi
  echo "START ${target} current=${current} expected=${expected}"
  curl --fail --location --retry 20 --retry-delay 5 --continue-at - \
    --output "${target}" "${url}"
  actual="$(stat -c %s "${target}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SIZE_MISMATCH ${target} expected=${expected} actual=${actual}" >&2
    exit 1
  fi
  echo "DONE ${target} ${actual}"
}

download \
  'https://modelscope.cn/models/Comfy-Org/Qwen-Image-2.1/resolve/master/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors' \
  "${MODEL_ROOT}/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors" 14230280616
download \
  'https://modelscope.cn/models/Comfy-Org/Qwen-Image-2.1/resolve/master/text_encoders/qwen3vl_8b_int8_convrot.safetensors' \
  "${MODEL_ROOT}/text_encoders/qwen3vl_8b_int8_convrot.safetensors" 9350798360
download \
  'https://modelscope.cn/models/Comfy-Org/Qwen-Image-2.1/resolve/master/vae/qwen_image_2.1_vae_bf16.safetensors' \
  "${MODEL_ROOT}/vae/qwen_image_2.1_vae_bf16.safetensors" 675509688
