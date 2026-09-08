# RAG、Prompt 与 LoRA/QLoRA 实验

## 什么时候不该微调

设备手册、产品版本、价格、库存、接口和安全条款会更新，且需要引用来源；这类事实放进可版本化知识库，用 RAG 和权限过滤解决。微调不会自动带来最新事实，也不替代权限、审计和检索。

微调适合相对稳定的行为：结构化格式、需求分类、风险标签、工具参数风格、固定输出语气和长尾格式遵循。即使做微调，也保留 RAG、schema 校验和人工审核。

## 数据集设计

生成入口：

```bash
make build-finetune-dataset
make dataset-check
```

当前数据来自 24 条合成客户案例，每个案例生成三个不同任务表述，结果为 72 条 ShareGPT-style 对话：

- `train.jsonl`：42 条
- `dev.jsonl`：12 条
- `test.jsonl`：18 条

变体按源案例分组，不能把同一个案例的变体拆到不同 split，避免数据泄漏。`manifest.json` 保存文件 hash、生成器、prompt 版本和统计信息。输出中的 `run_id`、`trace_id`、运行时延迟和模型名会被剔除，避免把一次运行的偶然字段教给模型。

合成数据只用于工程演示；真实项目应使用获得授权且脱敏的历史售前问答，并由业务专家抽检事实、引用和风险标签。

## QLoRA 配置

默认实验配置在 [`configs/finetune/trl_qlora.json`](../configs/finetune/trl_qlora.json)：

- 基座：`Qwen/Qwen2.5-0.5B-Instruct`
- 4-bit NF4、double quantization
- LoRA rank 16、alpha 32、dropout 0.05
- Colab Tesla T4 配置固定使用 `float16`；其他 GPU 需根据硬件能力单独记录和验证 `bfloat16`。
- 固定 seed 42、最大长度 2048、3 epochs
- 对 conversational dataset 只对 assistant response 计算 loss

实现参考：[TRL SFTTrainer 的 conversational dataset 与 assistant-only loss](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md)、[PEFT LoRA/QLoRA target modules](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md)、[LLaMA Factory 的 QLoRA/SFT 示例](https://github.com/hiyouga/LlamaFactory/blob/main/examples/README.md)。
- 使用 train/dev/test，训练后在 test 上单独评估
- test 评估使用独立的 evaluation-only `SFTTrainer`，先复用与 train/dev 相同的 prompt/completion、chat template、tokenizer、截断和 completion-only loss 预处理，再调用 `evaluate()`；不把原始 `prompt`/`completion` 列直接交给底层 `Trainer`
- base/adapter 生成评估使用同一模型、同一 test split、确定性解码参数和显式单卡放置；adapter 加载前校验 `adapter_config.json` 与 `adapter_model.safetensors/bin`，并在报告中记录真实 device/dtype/runtime
- 在正式训练前先执行 one-step smoke test，验证 dtype、量化、LoRA、优化器和 Trainer 组合；完整训练失败时保留 stdout/stderr，而不是只报告子进程返回码

本机没有 CUDA 时只运行 dry-run；不能把未训练的本地结果描述为 adapter 效果：

```bash
make finetune-dry-run
```

Colab CUDA 流程见 [`docs/colab-qlora-runbook.md`](colab-qlora-runbook.md)。

推荐直接打开 [`notebooks/qlora_colab.ipynb`](../notebooks/qlora_colab.ipynb)。该 notebook 会检查 GPU、安装 `finetune-colab` extra、将 checkpoint 写入 Drive、完成 dry-run、训练、base/adapter held-out 对比和结果打包。发生 OOM 时将 notebook 中的 `LOW_MEMORY` 改为 `True`，或使用 [`configs/finetune/trl_qlora_colab_lowmem.json`](../configs/finetune/trl_qlora_colab_lowmem.json)。

## 训练后比较

训练不是交付终点。至少生成下面四组对比：

有可用模型/adapter 后，可运行：

```bash
PYTHONPATH=src python scripts/evaluate_finetuned_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter .runtime/models/qwen2.5-0.5b-presales-lora \
  --output data/results/qlora-test.json
```

脚本只统计 JSON 解析、schema 通过、策略检查和无证据保守输出，不把字符串相似度当成业务质量。

| 对比 | 目的 |
|---|---|
| base vs adapter | 看结构化输出、风险分类和工具参数是否改善 |
| RAG off vs RAG on | 看事实和引用是否真正由知识库支持 |
| 正常集 vs 对抗集 | 看拒答、注入和越权是否回归 |
| 本地量化 vs 远程基线 | 看质量、时延、内存和成本的整体 trade-off |

只有当 adapter 在保留集上有稳定收益、对抗集不退化、输出仍可解析且部署指标在目标硬件上通过时，才进入下一阶段。

## 交付路线

1. 先交付 Prompt + RAG baseline。
2. 用固定黄金集定位是检索、提示词、schema 还是模型能力问题。
3. 只有稳定格式/分类问题仍有收益时才训练 LoRA/QLoRA。
4. 记录 base model、dataset manifest、训练 config、adapter hash 和评测报告。
5. vLLM 可在受控场景服务 LoRA adapter；动态加载必须有可信来源和隔离策略。若转 GGUF/llama.cpp，先在目标任务集上复测，不把转换当作质量保持证明。

服务路径参考：[vLLM LoRA adapter serving](https://github.com/vllm-project/vllm/blob/main/docs/features/lora.md) 和 [vLLM OpenAI-compatible server](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/quickstart.md)。
