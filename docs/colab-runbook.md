# Colab 实测运行手册

## 为什么把推理压测放到 Colab

本项目把两个运行环境分开：

- Dify 是需要数据库、向量库、插件守护进程和文件存储的应用平台，适合在本机 Docker 或稳定云主机上保留状态，用于录屏和应用演示。
- llama.cpp 的目标是一次性验证“同一模型、不同量化、不同并发”的硬件表现，适合放到临时的 Colab GPU runtime 中执行；它不需要公开端口，也不需要把客户数据上传到云端。

Colab 的 GPU 型号、可用时长和配额会动态变化，因此每份报告都必须记录本次 runtime 的 GPU、驱动、llama.cpp commit、模型 revision 和 SHA-256。Colab 结果是该 GPU/runtime 的实验结果，不应被表述为所有 GPU 或生产集群的 SLA。

## 已实现的入口

打开 [`notebooks/llama_cpp_colab_benchmark.ipynb`](../notebooks/llama_cpp_colab_benchmark.ipynb)，只需修改第一段代码中的：

```python
PROJECT_REPO_URL = 'https://github.com/<你的用户名>/ai-presales-lab.git'
```

然后在 Colab 中选择 GPU runtime，执行全部单元格。Notebook 会自动：

1. 固定并 checkout llama.cpp commit `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`。
2. 用 `GGML_CUDA=ON`、`GGML_NATIVE=OFF` 和 Release 构建 `llama-server`。
3. 从 Qwen 官方 Hugging Face 仓库 revision `2ed9be962c95f7625f4963ff51ed472e4538187a` 下载 `Q4_K_M` 与 `Q8_0`。
4. 校验 GGUF 文件头和 SHA-256。
5. 每个量化版本先预热，再执行 10 个请求的单并发和 4 并发测试。
6. 将原始报告保存到 `data/results/colab/`，并生成 `q4-q8-colab-summary.json`。

## 对比协议

Q4/Q8 只能在以下条件相同的情况下比较：

| 条件 | 固定值 |
|---|---|
| 模型家族 | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| 模型 revision | `2ed9be962c95f7625f4963ff51ed472e4538187a` |
| llama.cpp | `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae` |
| 上下文长度 | 2048 |
| GPU offload | `--n-gpu-layers 999` |
| 采样 | temperature 0、seed 42 |
| 最大输出 | 256 tokens |
| 请求数 | 每个设置 10 次 |
| 并发 | 1 和 4 |
| 服务并行槽 | 4 |
| 预热 | 每个设置开始前 1 次，排除在统计外 |
| API | `llama-server` 的 OpenAI-compatible `/v1/chat/completions` |

指标定义：

- `latency_ms`：单请求从客户端发出到完整响应收到的端到端时间。
- `ttft_ms`：流式响应中从请求发出到收到第一个非空 token 的时间。
- `throughput_tokens_per_second`：该批次成功请求生成 token 总数除以整批 wall time；并发测试不能把单请求 tokens/s 简单相加。
- `server_memory_mib`：llama-server 进程的 CPU RSS，包含模型映射和运行时内存。
- `gpu_memory_mib`：按 llama-server PID 从 `nvidia-smi` 读取的 GPU 显存；没有进程级数据时报告为空，而不是猜测。
- `structured_json_pass_rate`：输出能否被标准 JSON 解析的比例；它不是业务准确率。

## Colab 运行后的检查清单

不要只截图 notebook 最后一张表。保留以下证据：

- Colab runtime 的 GPU 型号和 `nvidia-smi` 输出。
- `llama-server --version` 和 `git rev-parse HEAD`。
- 两个模型文件名、文件大小、revision、SHA-256。
- Q4/Q8 各自的原始 JSON 和汇总 JSON。
- 运行日期、上下文长度、并发度和请求数。
- 失败请求及错误信息；如果失败，不要把它从分母中删除后继续宣传结果。

建议把 `data/results/colab/` 中的 JSON 下载回本地仓库后再提交。模型文件、Colab cache、API Key 和 notebook 输出中的隐私数据不要提交。

## 何时不适合用 Colab 结果做结论

- 需要长期服务、稳定 IP、持久化知识库或企业网络连通性时，Colab 不是部署目标。
- 需要估算客户生产容量时，应在客户目标 GPU、目标上下文、真实提示词和真实并发模型上复测。
- 免费 Colab 的 GPU 类型会变化；不同 GPU 的数值不能直接拼成一个“通用性能指标”。
- 0.5B 模型只用于展示推理链路和量化差异，不代表企业生产质量。质量结论需要另建业务黄金集和人工/模型评审协议。

## 官方依据

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)：GPU availability、配额和 runtime 限制是动态的。
- [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)：CUDA 构建、`GGML_NATIVE=OFF` 和 GPU 架构说明。
- [llama.cpp server guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)：`llama-server`、OpenAI-compatible 接口和启动方式。
- [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)：使用固定 revision 下载模型文件。
- [Qwen2.5-0.5B-Instruct-GGUF model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/2ed9be962c95f7625f4963ff51ed472e4538187a)：模型许可、文件和固定 revision。
