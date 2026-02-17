#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/ComfyUI"
  exit 1
fi

COMFYUI_ROOT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_NODE_DIR="${REPO_ROOT}/custom_nodes/comfyui_gemini_sam_points"
DST_NODE_DIR="${COMFYUI_ROOT}/custom_nodes/comfyui_gemini_sam_points"

if [[ ! -d "${COMFYUI_ROOT}" ]]; then
  echo "ComfyUI root not found: ${COMFYUI_ROOT}"
  exit 1
fi

mkdir -p "${COMFYUI_ROOT}/custom_nodes"
rm -rf "${DST_NODE_DIR}"
cp -R "${SRC_NODE_DIR}" "${DST_NODE_DIR}"
