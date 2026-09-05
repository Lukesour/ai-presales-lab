# Architecture and Design Decisions

## Why two projects

项目一关注应用交付：客户提出业务需求后，系统如何检索资料并生成可解释的解决方案。

项目二关注基础设施：同一个模型如何在本地运行，如何测量硬件、量化、上下文和并发对体验的影响。

两者共享输入案例和展示语言，但边界不同：Dify 负责应用流程与知识使用，llama.cpp 负责模型服务与性能，不把基础设施指标伪装成业务准确率。

## Data flow

```text
CustomerBrief
    -> requirement extraction
    -> knowledge retrieval
    -> evidence-aware solution
    -> risk / clarification gate
    -> Dify API or Gradio response
```

本地离线引擎使用透明的字符 n-gram 检索，用于无 API Key 的测试和基线。Dify 版本使用其知识库完成正式检索。这样可以在不支付模型费用的情况下先验证数据契约和风险规则。

## Risk boundaries

- 无证据不输出具体产品承诺
- 私有化、数据出域、合规和容量要求进入风险或待确认问题
- 个人演示机器的性能不外推为生产容量
- API Key 只在服务端环境变量中读取
- 文档中的提示词注入内容不拥有更高优先级
