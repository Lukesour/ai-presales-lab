# Measured Results

Only commit small JSON reports produced by the scripts. Do not commit model files, API keys, raw customer data or logs containing prompts.

Suggested labels:

- `offline-evaluation.json`: deterministic contract and safety baseline
- `q4-metal-c1.json`: Q4 model on native Apple Silicon with concurrency 1
- `q8-metal-c1.json`: Q8 model on native Apple Silicon with concurrency 1
- `q4-metal-c4.json`: Q4 model on native Apple Silicon with concurrency 4

Each report should be accompanied by the exact model file, llama.cpp commit, hardware, context length and launch flags in the README or experiment notes.
