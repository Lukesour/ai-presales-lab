# Upstream Attribution

本作品集使用以下开源项目作为运行时或参考实现：

| 项目 | 用途 | 上游仓库 | 个人实现边界 |
|---|---|---|---|
| Dify | 知识库、工作流和应用 API | https://github.com/langgenius/dify | 工作流设计、资料、输出契约、API 适配、演示和评测 |
| llama.cpp | 本地 GGUF 推理和 OpenAI-compatible 服务 | https://github.com/ggml-org/llama.cpp | 启动脚本、客户端、并发基准、指标解释和选型报告 |
| LangGraph | 可选的状态图、持久化和人工审核适配 | https://github.com/langchain-ai/langgraph | 节点边界、状态契约、离线 fallback 和适配器；不声称拥有上游运行时 |
| TRL / PEFT | 可选 SFT、LoRA/QLoRA 训练组件 | https://github.com/huggingface/trl / https://github.com/huggingface/peft | 数据契约、配置、训练脚本、评测协议和结果分析 |
| LLaMA Factory | 可选的 SFT/LoRA 对照训练路径 | https://github.com/hiyouga/LLaMA-Factory | 配置、数据映射、复现手册和对照实验 |
| vLLM | 生产 GPU/LoRA 服务选型参考 | https://github.com/vllm-project/vllm | 部署决策、接口边界和压测指标定义，不提交上游源码 |
| Promptfoo | 可选红队与回归评测 | https://github.com/promptfoo/promptfoo | 测试 fixture、配置和本地安全策略 |
| Arize Phoenix | 生产可观测性候选 | https://github.com/Arize-ai/phoenix | 仅作为 OpenTelemetry/LLM trace 选型参考，当前本地实现为 JSONL |

上游版本、commit、模型文件和模型许可证应在实际复现时记录。Dify、Dify 依赖、模型和各训练/推理组件的许可证可能不同，公开仓库前应逐项复核。不要移除上游署名，也不要提交上游源码或大模型文件到本仓库。
