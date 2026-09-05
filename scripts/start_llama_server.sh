#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-}"
LLAMA_CONTEXT="${LLAMA_CONTEXT:-4096}"
LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-999}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"

if [[ -z "$LLAMA_MODEL_PATH" ]]; then
  echo "Set LLAMA_MODEL_PATH to a local .gguf file before starting the server." >&2
  exit 2
fi
if ! command -v "$LLAMA_SERVER_BIN" >/dev/null 2>&1; then
  echo "Cannot find $LLAMA_SERVER_BIN. Build llama.cpp or set LLAMA_SERVER_BIN to its absolute path." >&2
  exit 2
fi
if [[ ! -f "$LLAMA_MODEL_PATH" ]]; then
  echo "Model file does not exist: $LLAMA_MODEL_PATH" >&2
  exit 2
fi

echo "Starting llama.cpp on http://${LLAMA_HOST}:${LLAMA_PORT}"
echo "model=${LLAMA_MODEL_PATH} context=${LLAMA_CONTEXT} gpu_layers=${LLAMA_GPU_LAYERS}"
exec "$LLAMA_SERVER_BIN" \
  -m "$LLAMA_MODEL_PATH" \
  -c "$LLAMA_CONTEXT" \
  -ngl "$LLAMA_GPU_LAYERS" \
  --host "$LLAMA_HOST" \
  --port "$LLAMA_PORT"
