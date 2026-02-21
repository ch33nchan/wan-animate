#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/ComfyUI"
  exit 1
fi

COMFY_DIR="$1"
NODE_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/comfyui/custom_nodes/comfy_gemini_track_selector"
NODE_DST_DIR="${COMFY_DIR}/custom_nodes/comfy_gemini_track_selector"

if [[ ! -d "$COMFY_DIR" ]]; then
  echo "ComfyUI path not found: $COMFY_DIR"
  exit 1
fi

mkdir -p "${COMFY_DIR}/custom_nodes"
rm -rf "$NODE_DST_DIR"
cp -R "$NODE_SRC_DIR" "$NODE_DST_DIR"

if [[ -d "${COMFY_DIR}/venv" ]]; then
  # shellcheck disable=SC1091
  source "${COMFY_DIR}/venv/bin/activate"
  pip3 install google-generativeai
fi

echo "Installed node at: $NODE_DST_DIR"
echo "Set GEMINI_API_KEY before launching ComfyUI."
