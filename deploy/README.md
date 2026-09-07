# Agent 容器化演示

这是本地作品集的最小部署样例，展示非 root、只读 root filesystem、checkpoint 独立 volume、healthcheck 和 capability drop。它没有替代企业生产网关、认证、密钥管理、租户隔离或可观测性平台。

从仓库根目录运行：

```bash
docker compose -f deploy/compose.agent.yaml up --build
curl http://127.0.0.1:8090/health
```

停止并保留 checkpoint：

```bash
docker compose -f deploy/compose.agent.yaml down
```

生产化还需要在平台侧补充 TLS/身份认证、网络策略、资源限制、备份与恢复、日志脱敏、镜像扫描、依赖锁定、数据库迁移和高风险工具审批。
