# Dify 售前助手复现

本目录保存 Dify 应用层的业务配置说明和输出契约，不包含 Dify 上游源码。

## 上游项目

- 仓库：[langgenius/dify](https://github.com/langgenius/dify)
- 建议使用 Dify 的稳定 release，而不是 `main`
- 首次启动时记录 release tag、commit、模型提供商和知识库版本
- Dify 的源代码、镜像和前端展示须遵守上游许可证；不要移除上游署名

## 启动方式

在单独目录按 Dify 官方 Docker Compose 文档启动服务：

```bash
git clone https://github.com/langgenius/dify.git
cd dify/docker
cp .env.example .env
docker compose up -d
```

首次访问 `http://localhost/install` 完成初始化。你当前的 Apple Silicon 环境先执行 `scripts/check_dify.sh` 做架构 smoke test；如果某个镜像不兼容，使用 Dify 官方文档提供的源码或远程实例，不要修改本作品集的接口契约。

## 知识库资料

将仓库根目录 `data/knowledge/*.md` 上传到一个名为 `ai-presales-catalog` 的 Dify 知识库。建议按文档类型分别上传，并保留文件名与版本号。

推荐配置：

- 小段落优先，标题和列表不拆散
- 打开检索测试，检查是否能召回部署、安全和推理资料
- 为无召回结果设置固定回复：资料不足，请补充客户约束或产品资料
- 不在系统提示词中写死价格、SLA、认证或准确率
- 开启结构化输出能力时，遵循 `output_schema.json`；`workflow_contract.json` 是一份示例响应

## 工作流节点

工作流按以下顺序配置：

1. `需求输入`：接收 `customer_brief` 和原始客户描述
2. `需求结构化`：提取行业、场景、数据类型、部署、并发、时延和合规
3. `知识检索`：检索产品能力、部署和安全资料
4. `方案生成`：只基于检索上下文生成方案
5. `证据与风险检查`：没有来源的事实改成待确认项
6. `条件分支`：高风险或低置信度进入追问，否则输出方案
7. `结构化响应`：返回与 `output_schema.json` 对齐的 JSON

Dify App API 的密钥只能由 `DifyClient` 在服务端调用，不能放进 Gradio 前端代码或浏览器请求中。
