# 评测方案与当前证据

## 评测分层

| 层级 | 问题 | 当前入口 |
|---|---|---|
| 契约 | JSON 是否可解析、字段是否完整 | `pytest`、`validate_solution_dict` |
| Agent | 节点是否完成、POC/模型策略是否存在、审核门是否触发 | `make agent-eval` |
| 检索 | 是否有来源；无召回是否保守 | `data/evaluation/cases.jsonl` |
| 安全 | 注入、无依据承诺、敏感数据 | `make security-check` / `scripts/run_security_checks.py` |
| 微调 | 数据格式、重复、case-level split、hash | `make dataset-check` |
| 推理 | TTFT、p95、吞吐、RSS/VRAM、结构化 JSON | Colab notebook + `benchmark_llama.py` |

## 运行命令

```bash
make test
make lint
make eval
make agent-eval
PYTHONPATH=src python scripts/run_security_checks.py
make build-finetune-dataset
make dataset-check
make finetune-dry-run
```

## 当前离线证据

在本机依赖环境中，最近一次可复现结果为：

- Python 单元测试：18 passed。
- Agent 24 条案例：24/24 完成；schema、POC、model strategy、证据/保守无证据规则和审核门均通过。
- 红队策略用例：12/12 通过。
- 微调数据：72 条对话，按 24 个源案例做 case-level split；train 42、dev 12、test 18，每个源案例最多 3 个变体。

这些数字证明的是离线契约和规则，不是模型业务准确率，也不是客户生产容量。真正的 LLM 质量报告必须另存 base/adapter、模型 revision、评测提示词、人工评分规则和完整原始输出。

## 质量门槛建议

上线前至少设置以下门槛，具体阈值由客户黄金集和风险等级决定：

1. 结构化输出解析率 100% 或失败自动进入人工/重试路径。
2. 高风险审核召回率不能以“模型感觉正确”为准，应由规则集和人工抽检共同验证。
3. 无证据时不得出现产品能力、价格、SLA、认证、准确率或容量承诺。
4. adapter 相对 base 的收益必须在保留集和对抗集上同时观察，不能只看训练 loss。
5. 性能报告必须包含失败请求，不得删除慢请求或失败请求后再计算 p95。
