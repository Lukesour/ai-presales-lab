# Colab compact QLoRA run

- Source bundle: `ai-presales-lab-results/20260909-023322`
- Repository commit: `095109616c99fe665d296eaab0eebe1b6bd5818b`
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Hardware: Tesla T4, CUDA 12.8
- Dataset profile: `compact`
- Held-out test split: 18 synthetic examples

The JSON reports are copied from the Colab bundle without output previews. The
adapter weights, checkpoints and optimizer states are intentionally excluded
from Git because they are large artifacts; the original bundle remains in
Google Drive/Downloads for reproducibility.

The compact adapter results are JSON parse `18/18`, compact schema `14/18`
and policy pass `18/18`. These are model-facing structured-output metrics,
not production accuracy or a full `SolutionResponse` business-quality score.
