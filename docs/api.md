# 本地 Agent API

这是一个用于作品集演示的最小 HTTP 服务，不是未加固的生产网关。它展示结构化 Agent、checkpoint、人审恢复和 OpenAI-compatible 适配。

## 启动

```bash
PYTHONPATH=src python scripts/serve_agent.py --host 127.0.0.1 --port 8090
```

## 健康检查

```bash
curl -s http://127.0.0.1:8090/health
curl -s http://127.0.0.1:8090/ready
```

## 启动一个线程

```bash
curl -s http://127.0.0.1:8090/v1/runs \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "thread_id": "demo:manufacturing-001",
  "brief": {
    "case_id": "case-001",
    "industry": "制造业",
    "use_case": "设备运维知识助手",
    "data_types": ["维修手册 PDF", "历史工单"],
    "deployment": "私有化",
    "concurrency": "峰值 5",
    "latency_requirement": "完整答案 10 秒内",
    "compliance": ["数据不能出域"],
    "raw_request": "请给出可引用、可审计的设备故障问答 POC。"
  }
}
JSON
```

如果风险门触发，返回 `status=pending_review`。审核恢复：

```bash
curl -s http://127.0.0.1:8090/v1/runs/demo:manufacturing-001/review \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approve"}'
```

查看 checkpoint：

```bash
curl -s http://127.0.0.1:8090/v1/runs/demo:manufacturing-001
```

## OpenAI-compatible facade

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"presales-agent-offline","messages":[{"role":"user","content":"请分析制造业设备运维 AI POC。"}]}'
```

响应的 `choices[0].message.content` 是 JSON 字符串，内部遵循 `dify/output_schema.json`。生产替换时要加认证、超时、限流、CORS/网络边界、统一错误码和请求大小限制。
