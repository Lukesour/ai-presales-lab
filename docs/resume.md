# Resume Draft

以下数字必须替换为实际 `data/results/` 评测结果，不要直接照抄示例数字。

## Project One

**企业 AI 解决方案售前助手｜Dify、RAG、Prompt Workflow、Gradio**

- 面向企业 AI 产品选型场景，梳理行业、业务场景、数据类型、部署边界、并发、时延与合规等售前需求字段，设计需求到方案的交付流程
- 基于 Dify 搭建知识库与可视化工作流，接入产品能力、部署和安全资料，输出带来源引用的推荐架构、实施步骤、风险和待确认问题
- 通过 Dify App API 与 Gradio 完成外部演示集成，增加无证据拒答、高风险需求提示和服务端密钥管理
- 构建 24 条合成黄金问题集，评估需求覆盖、证据引用、无答案识别和结构化输出质量，结果以实测报告为准

## Project Two

**本地大模型推理服务与性能评测｜llama.cpp、GGUF、量化、CUDA/Metal**

- 基于 llama.cpp 构建 CUDA/Metal 加速的 GGUF 本地推理服务，提供 OpenAI-compatible API，支持上层应用切换本地模型后端
- 编写 Python 客户端和并发基准脚本，在 Colab GPU/Apple Silicon 上对 Q4/Q5/Q8 量化、上下文长度和并发数进行对比测试，记录 TTFT、p95、聚合吞吐、CPU RSS、GPU VRAM 和结构化输出通过率
- 记录首 Token 延迟、生成速度、p95 延迟、内存占用和结构化输出通过率，形成云端 API 与本地部署的选型依据
- 针对数据不能出域、模型质量、运维成本和容量承诺等约束，输出私有化 PoC 到生产部署的风险与验证清单

## Disclosure

面试中明确说明：Dify 和 llama.cpp 的基础能力来自上游开源项目，个人工作集中在业务抽象、资料整理、配置、适配器、评测、风险规则、部署脚本和结果分析。
