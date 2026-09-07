# Architecture and Design Decisions

## Why one reusable portfolio with layered paths

项目以一个制造业设备运维旗舰案例为主线，分成三个可替换层：

- 应用交付：Dify 或本地 Agent 如何把客户需求、知识检索和方案输出串起来。
- Agent 编排：显式状态、工具白名单、风险门、人工审核和 checkpoint 如何让交付可控。
- 基础设施：同一个模型如何在本地运行，如何测量硬件、量化、上下文和并发对体验的影响。

这些路径共享输入案例、证据资料和结果契约，边界不同：Dify/Agent 负责应用流程与知识使用，llama.cpp/vLLM 负责模型服务与性能，不把基础设施指标伪装成业务准确率。

## Data flow

```text
CustomerBrief
    -> intake and missing-field detection
    -> bounded knowledge tools
    -> architecture + four-phase POC
    -> RAG / Prompt / LoRA / serving decision
    -> evidence, privacy and commitment checks
    -> human review checkpoint or final response
```

本地离线引擎使用透明的字符 n-gram 检索，用于无 API Key 的测试和基线。Dify 版本使用其知识库完成正式检索。这样可以在不支付模型费用的情况下先验证数据契约和风险规则。

## Risk boundaries

- 无证据不输出具体产品承诺
- 私有化、数据出域、合规和容量要求进入风险或待确认问题
- 个人演示机器的性能不外推为生产容量
- API Key 只在服务端环境变量中读取
- 文档中的提示词注入内容不拥有更高优先级
- Agent 不拥有任意代码执行、写库、发邮件和生产修改工具
- trace 只记录必要元数据，并在落盘前脱敏

## Fallback strategy

核心包不强制安装 LangGraph、Dify、TRL 或 GPU 依赖：

- 无外部服务：离线检索、规则引擎、SQLite checkpoint 和本地 API 仍能演示完整契约。
- 有 Dify：替换应用编排层，复用知识资料、提示词和 schema。
- 有 LangGraph：使用可选适配器和持久化 checkpointer 接入 interrupt/resume。
- 有 CUDA：运行 QLoRA 和 llama.cpp/vLLM 真实实验；无 CUDA 时只报告 dry-run 和已运行的 CPU/Apple 结果。
