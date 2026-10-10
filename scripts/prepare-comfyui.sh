#!/usr/bin/env bash
set -euo pipefail

REVISION="${COMFYUI_REVISION:-1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_ROOT="${ROOT}/vendor"
TARGET="${VENDOR_ROOT}/ComfyUI"
ARCHIVE="${COMFYUI_ARCHIVE:-}"

if [[ -f "${TARGET}/main.py" && "$(cat "${TARGET}/.codex-revision" 2>/dev/null || true)" == "${REVISION}" ]]; then
  echo "ComfyUI ${REVISION} is already prepared"
  exit 0
fi

mkdir -p "${VENDOR_ROOT}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

if [[ -n "${ARCHIVE}" ]]; then
  cp "${ARCHIVE}" "${TEMP_DIR}/comfyui.tar.gz"
else
  curl --fail --location --retry 10 --retry-delay 5 \
    --output "${TEMP_DIR}/comfyui.tar.gz" \
    "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/${REVISION}"
fi

tar -xzf "${TEMP_DIR}/comfyui.tar.gz" -C "${TEMP_DIR}"
SOURCE="$(find "${TEMP_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'ComfyUI-*' | head -1)"
test -n "${SOURCE}"
rm -rf "${TARGET}"
mv "${SOURCE}" "${TARGET}"
printf '%s\n' "${REVISION}" > "${TARGET}/.codex-revision"
echo "Prepared ComfyUI ${REVISION} at ${TARGET}"
