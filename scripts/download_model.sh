#!/usr/bin/env bash
set -euo pipefail

LLAMA_CLI_BIN="${LLAMA_CLI_BIN:-llama-cli}"
MODEL_REPO="${1:-ggml-org/Qwen3.5-0.8B-GGUF}"

if ! command -v "$LLAMA_CLI_BIN" >/dev/null 2>&1; then
  echo "Cannot find $LLAMA_CLI_BIN. Build llama.cpp first." >&2
  exit 2
fi

echo "This downloads a model from Hugging Face and runs a one-token smoke test."
echo "Model repository: $MODEL_REPO"
"$LLAMA_CLI_BIN" -hf "$MODEL_REPO" -n 1 -p "Reply with OK."
