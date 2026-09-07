#!/usr/bin/env bash
set -euo pipefail

LLAMA_CLI_BIN="${LLAMA_CLI_BIN:-llama-cli}"
MODEL_REPO="${1:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
MODEL_QUANT="${2:-Q4_K_M}"

if ! command -v "$LLAMA_CLI_BIN" >/dev/null 2>&1; then
  echo "Cannot find $LLAMA_CLI_BIN. Build llama.cpp first." >&2
  exit 2
fi

echo "This downloads a model from Hugging Face and runs a one-token smoke test."
echo "Model repository: $MODEL_REPO"
echo "Quantization: $MODEL_QUANT"
"$LLAMA_CLI_BIN" -hf "$MODEL_REPO:$MODEL_QUANT" -n 1 -p "Reply with OK."
