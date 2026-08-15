# SearchEngine AI Lab

面向 C++ 后端面试资料的本地半结构化知识库 RAG 后端原型。

这个仓库用于把传统检索、向量召回、RAG 回答层和后端接口串成一个小而完整的学习型工程闭环。当前重点不是做通用企业知识库平台，而是围绕 C++ 后端面试资料，展示 Markdown 资料解析、检索诊断、LLM 回答、接口缓存和降级边界。

## 当前能力

- 从 Markdown 八股资料中解析标题、正文、代码块和行号信息，生成 JSONL chunks。
- 基于 JSONL chunks 跑通 keyword / vector / hybrid 三种 TopK 召回。
- 支持 fake embedding 和 Zhipu `embedding-3`，方便在无 key 环境下先验证链路。
- 对召回结果输出 diagnostics，包括 query tokens、核心词、缺失词、Top1 分数和可能的 knowledge gap。
- 基于 GLM 构造 RAG prompt，返回回答、sources、diagnostics、usage 和耗时拆分。
- 提供 FastAPI 接口 `/rag/query` 和兼容入口 `/agent/query`。
- 在接口层提供同参 LRU 缓存，用于减少重复 query 的检索和生成开销。
- 当没有 API key、禁用 GLM、检索为空或疑似知识库缺口时，返回可解释的降级结果。

## 核心链路

```text
八股文.md
  -> Markdown ingestion
  -> data/bagu_chunks.jsonl
  -> keyword / vector / hybrid retrieval
  -> diagnostics / knowledge gap
  -> RAG prompt
  -> GLM answer or retrieval-summary fallback
  -> FastAPI /rag/query
  -> in-process LRU response cache
```

## 目录说明

| 文件 | 作用 |
|---|---|
| `ai17_markdown_ingestion.py` | 将 `八股文.md` 解析成带 metadata 的 JSONL chunks |
| `ai18_jsonl_retrieval.py` | JSONL 检索小闭环，包含 keyword、vector、hybrid、embedding 和 diagnostics |
| `ai19_retrieval_eval.py` | 基于固定 eval set 检查召回结果和 knowledge gap 判断 |
| `ai20_glm_rag_answer.py` | GLM RAG 回答层，负责 prompt、sources、usage、降级和耗时拆分 |
| `ai21_bagu_rag_api.py` | 当前八股 RAG FastAPI 接口，提供 `/health`、`/rag/query`、`/agent/query` |
| `data/bagu_chunks.jsonl` | 由 Markdown ingestion 生成的知识库 chunks |
| `data/bagu_retrieval_eval_v0.jsonl` | 检索评估样例 |
| `Dockerfile` | 早期 `ai14_agent_api.py` demo 的 Docker 配置，尚未更新到 `ai21_bagu_rag_api.py` |

早期 `ai07` 到 `ai14` 文件保留了从 chunk、向量检索、RAG prompt、Agent tool 到 FastAPI demo 的学习过程。当前可展示主线以 `ai17` 到 `ai21` 为准。

## 环境准备

建议使用 Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果只运行 fake embedding 或禁用 GLM 的检索摘要路径，可以不配置 API key。

如果需要真实 embedding 或 GLM 回答，需要在本机环境中配置：

```powershell
$env:ZAI_API_KEY="your-api-key"
$env:ZAI_GLM_MODEL="glm-4-flash"
```

`ZAI_API_KEY` 不要写入代码，不要提交 `.env`。仓库的 `.gitignore` 已忽略 `.env`、`.env.*`、虚拟环境、日志和 embedding 缓存文件。

## 生成知识库 chunks

当前仓库已经包含 `data/bagu_chunks.jsonl`。如果修改了 `八股文.md`，可以重新生成：

```powershell
python ai17_markdown_ingestion.py
```

## 检索验证

不依赖 API key 的检索验证：

```powershell
python ai18_jsonl_retrieval.py "epoll 水平触发和边缘触发有什么区别" --mode hybrid --embedding-provider fake --top-k 3
```

固定 eval set 回归：

```powershell
python ai19_retrieval_eval.py --mode hybrid --embedding-provider fake --top-k 3
```

使用 Zhipu embedding 时会调用外部 API，并可能产生费用：

```powershell
python ai19_retrieval_eval.py --mode hybrid --embedding-provider zhipu --top-k 3
```

首次真实 embedding 会构建本地缓存，缓存文件按 `.gitignore` 规则不提交。

## RAG 回答层

禁用 GLM，只返回检索摘要：

```powershell
python ai20_glm_rag_answer.py "智能指针 shared_ptr 和 unique_ptr 有什么区别" --no-glm
```

启用 GLM：

```powershell
python ai20_glm_rag_answer.py "智能指针 shared_ptr 和 unique_ptr 有什么区别" --embedding-provider zhipu
```

默认 GLM 模型为 `glm-4-flash`，可以通过 `ZAI_GLM_MODEL` 或 `--glm-model` 覆盖。

## 启动 API

```powershell
uvicorn ai21_bagu_rag_api:app --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

示例请求：

```powershell
curl -X POST http://127.0.0.1:8000/rag/query `
  -H "Content-Type: application/json" `
  -d '{
    "query": "epoll 水平触发和边缘触发有什么区别",
    "top_k": 3,
    "mode": "hybrid",
    "embedding_provider": "auto",
    "use_glm": true,
    "temperature": 0.2,
    "max_tokens": 800
  }'
```

请求字段：

| 字段 | 说明 |
|---|---|
| `query` | 用户问题，1 到 300 个字符 |
| `top_k` | 召回条数，范围 1 到 8，默认 3 |
| `mode` | `keyword`、`vector` 或 `hybrid`，默认 `hybrid` |
| `embedding_provider` | `auto`、`fake` 或 `zhipu`；`auto` 会在存在 `ZAI_API_KEY` 时选择 `zhipu`，否则选择 `fake` |
| `use_glm` | 是否调用 GLM；为 `false` 时只返回检索摘要 |
| `glm_model` | 可选，覆盖默认 GLM 模型 |
| `temperature` | 生成温度，默认 0.2 |
| `max_tokens` | 最大生成 token，默认 800 |

主要响应字段：

| 字段 | 说明 |
|---|---|
| `answer` | GLM 回答或降级后的检索摘要 |
| `status` | `ok`、`retrieval_empty`、`knowledge_gap`、`retrieval_summary`、`llm_unavailable`、`llm_error` 或 `retrieval_error` |
| `degraded` | 是否走了降级路径 |
| `fallback_reason` | 降级原因 |
| `sources` | 召回来源，包含 chunk id、标题路径、行号、类型和分数 |
| `diagnostics` | 检索诊断信息 |
| `usage` | GLM 返回的 token 用量；降级时为空 |
| `latency_ms` | 总耗时 |
| `retrieval_latency_ms` | 检索耗时 |
| `generation_latency_ms` | 生成耗时 |
| `cache_hit` | 是否命中接口层缓存 |
| `cached_response` | 是否直接返回缓存响应 |

同参数第二次请求会命中接口层 LRU 缓存，`cache_hit=true`，用于展示重复 query 的成本优化。

## 可以尝试的问题

- `智能指针 shared_ptr 和 unique_ptr 有什么区别`
- `epoll 水平触发和边缘触发有什么区别`
- `线程池任务队列满了应该怎么处理`
- `TCP 四次挥手和 TIME_WAIT 有什么关系`
- `Redis 缓存穿透、击穿、雪崩分别是什么`

这些问题覆盖 C++、Linux 网络、并发和缓存，和 C++ 后端面试主线比较贴近。

## Docker 状态

当前虚拟机环境已安装 Docker，但仓库中的 `Dockerfile` 仍指向早期 `ai14_agent_api.py` demo：

```dockerfile
CMD ["uvicorn", "ai14_agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

因此，Docker 目前只能作为早期 Agent API demo 的运行方式，尚不能代表最新的 `ai21_bagu_rag_api.py` 八股 RAG 接口。后续如果要把 Docker 作为展示路径，需要先更新 `Dockerfile`，复制 `ai17` 到 `ai21`、`data/` 和 `八股文.md`，并将启动命令切换为：

```dockerfile
CMD ["uvicorn", "ai21_bagu_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

在完成这一步之前，本项目默认运行方式仍是本地 Python + FastAPI。

## 成本与安全边界

- `fake` embedding 用于本地链路验证，不代表真实语义向量质量。
- `zhipu` embedding 和 GLM 回答会调用外部 API，可能产生费用。
- API key 只通过环境变量或本机 `.env` 管理，不能提交到仓库。
- 接口层 LRU 缓存只缓存同参数请求，不是语义缓存，也不是分布式缓存。
- 当前没有做用户认证、权限隔离、限流、审计日志和多租户管理。
- 当前知识库样本较小，主要用于面试资料 RAG 原型展示。

## 项目不足

- Markdown chunk 策略仍偏规则，复杂表格、图片和跨段语义没有做深层理解。
- 固定 eval set 规模较小，`failures=0` 只能说明当前样例通过，不代表真实问法全部可靠。
- fake embedding 只能用于流程验证，真实召回质量需要用 Zhipu embedding 或本地 embedding 模型验证。
- knowledge gap 依赖启发式规则，可能误判相近概念或冷门问法。
- GLM 回答依赖外部 API，可用性、耗时和成本都受外部服务影响。
- Docker 尚未更新到最新 RAG API，不应作为当前主展示路径。

## 面试表达

这个项目可以这样讲：

> 我做了一个面向 C++ 后端面试资料的本地知识库 RAG 后端原型。它先把 Markdown 八股资料解析成带标题路径、行号和 chunk 类型的 JSONL，然后支持关键词、向量和混合检索。检索阶段会输出核心词匹配、缺失词、Top1 分数和 knowledge gap 判断，避免在资料不匹配时强行让模型回答。回答层把 TopK 来源拼成 RAG prompt 调 GLM，如果没有 key、检索为空或疑似知识库缺口，就降级成检索摘要。最后用 FastAPI 暴露 `/rag/query`，并在接口层做同参 LRU 缓存，减少重复请求成本。

边界也要说清楚：

> 这个项目是学习型工程原型，不是生产级知识库平台。它的价值在于把文档解析、检索召回、诊断、LLM 回答、接口封装、缓存和降级串成闭环；不足是数据规模、评估集、权限隔离、限流和 Docker 部署还没有做到生产级。
