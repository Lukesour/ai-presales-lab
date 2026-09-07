# Agent 安全回归

安全用例覆盖提示词注入、间接注入、越权工具、过度承诺、敏感数据、无限循环和无证据生成。

本地无依赖检查：

```bash
PYTHONPATH=src python scripts/run_security_checks.py
```

需要更完整的红队测试时，可在隔离环境使用 [Promptfoo](https://github.com/promptfoo/promptfoo) 和 `promptfooconfig.yaml`。Promptfoo 配置、测试 fixture 和 provider 都应被当成可执行代码处理；不要把真实客户数据、生产密钥或未审查的远程配置放入测试。

Agent 的工具权限采用最小化原则：本项目只提供检索、计算、比较和校验工具，不提供删除、写库、发邮件等副作用工具。任何未来新增的副作用工具都必须增加参数校验、人工审核和审计记录。
