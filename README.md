# Tech RAG Service

面向 C++ 后端面试资料的版本化 RAG Core；它是学习型、可验证的工程原型，不是生产级知识库或通用 Agent 平台。

> 说明：本 README 前半部分描述当前 `rag_core/` 主线；后半部分的 `ai17` 至 `ai21` 内容保留为早期学习原型，不能与当前 RAG Core 的能力或部署方式混用。

## 当前 AI-25 RAG Core

当前主线以 `rag_core/` 和 `scripts/` 为准，真实语料为根目录 `八股文.md`。当前 active revision 含 337 个 parent chunks 与 961 个 child chunks。

```text
八股文.md
  -> Markdown parent/child chunking
  -> PostgreSQL: revision + chunks + embedding cache
  -> OpenSearch: child BM25 index
  -> embedding-3 -> Qdrant: child vectors
  -> 校验成功后切换 active_revision_id

查询 -> BM25 + Dense 并行召回 -> RRF -> parent context window
     -> evidence gate -> GLM answer + citation / no_knowledge
```

核心事实：所有 OpenSearch/Qdrant 查询都强制带 `revision_id` filter；外部索引和校验完成前不会切换 active revision。RRF 只融合名次，不直接混加 BM25 与 cosine 原始分数。

### 当前入口

| 路径 | 用途 |
|---|---|
| `rag_core/ingestion/` | Markdown 解析、token 切分、parent/child chunks |
| `rag_core/indexing/` | revision build 与发布屏障 |
| `rag_core/stores/` | PostgreSQL、OpenSearch、Qdrant 适配器 |
| `rag_core/retrieval/` | 并行召回、RRF、parent window、evidence gate |
| `rag_core/generation/` | token 预算、GLM、citation |
| `migrations/` | PostgreSQL schema 初始化与演进 |
| `scripts/run_real_bagu_core.py` | 新建并发布 revision；会写索引并可能调用 API |
| `scripts/query_active_bagu_core.py` | 查询当前 active revision，不重新导入文档 |
| `scripts/evaluate_ai25_retrieval.py` | 检索与 evidence gate 回归 |

### 首次准备

1. 启动本机 PostgreSQL 数据库 `tech_rag`，并执行一次 `migrations/0001_rag_core_initial.sql`、`0002_rag_core_chunks.sql`、`0003_embedding_cache.sql`。
2. 启动 Qdrant 和 OpenSearch：

```powershell
docker start tech-rag-qdrant
docker compose -f docker-compose.opensearch.yml up -d
```

3. 准备 `.env`（Git 忽略）中的 `RAG_TEST_DATABASE_DSN` 与 `ZAI_API_KEY`。密钥不得提交。

### 导入、查询与验证

```powershell
cd D:\code\Python\ai

# 仅在首次导入或八股文.md 更新后运行：创建并发布新 revision
.\.venv\Scripts\python.exe -m scripts.run_real_bagu_core

# 日常查询当前 active revision：不重建索引
.\.venv\Scripts\python.exe -m scripts.query_active_bagu_core "epoll 边缘触发时为什么必须读到 EAGAIN？"

# 离线单元测试
.\.venv\Scripts\python.exe -m unittest discover -s tests\unit -v

# revision-bound 检索与 evidence gate 回归
.\.venv\Scripts\python.exe -m scripts.build_ai25_eval
.\.venv\Scripts\python.exe -m scripts.evaluate_ai25_retrieval
```

真实 embedding/GLM 会将查询或检索上下文发送至配置的智谱 API，可能产生费用。评测集用于回归与发现失败边界，不用于反复调固定题目分数。

### 已知边界

- 首版 evidence gate 会在部分证据不足时返回 `no_knowledge`，但不是可靠防幻觉保证。
- 已验证的 MongoDB 缺口题曾被误放行，并携带无关的 MySQL citation；因此不能声称 citation 总能支撑回答。
- 当前没有多租户、认证、限流、审计、生产部署编排或高并发承诺。
- 根目录 `Dockerfile` 仍用于早期 `ai21_bagu_rag_api.py` 原型，不是当前 AI-25 RAG Core 的端到端容器化部署。

## 早期 ai17–ai21 学习原型

这个仓库用于把传统检索、向量召回、RAG 回答层和后端接口串成一个小而完整的学习型工程闭环。当前重点不是做通用企业知识库平台，而是围绕 C++ 后端面试资料，展示 Markdown 资料解析、检索诊断、LLM 回答、接口缓存和降级边界。

## 当前能力

- 从 Markdown 八股资料中解析标题、正文、代码块和行号信息，生成 JSONL chunks。
- 支持按真实 tokenizer 的 token 预算切分正文，并保留可配置的 token overlap。
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
| `ai22_token_chunking.py` | 用 `tiktoken` 按 token 上限切分 Markdown 正文，可配置 overlap，并保持 ai17 的 JSONL schema |
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

## Token 预算切分（AI22）

当需要让 chunk 尺寸更贴近模型上下文而不是字符数时，可以生成一份独立的 token-budget 版本：

```powershell
python ai22_token_chunking.py --max-tokens 450 --overlap-tokens 60
```

默认输出为 `data/bagu_chunks_token.jsonl`，可通过 `--input`、`--output` 和 `--encoding` 覆盖。正文优先按句子边界切分，单句过长时才按 token 强制切开；表格和 fenced code block 会整体保留，以避免破坏结构，因此它们可能超过正文的 token 上限。

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

当前 `Dockerfile` 已切换到最新的 `ai21_bagu_rag_api.py` 八股 RAG 接口，会复制 `ai17` 到 `ai21`、`data/` 和 `八股文.md`，并通过 Uvicorn 暴露 `8000` 端口：

```dockerfile
CMD ["uvicorn", "ai21_bagu_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

在虚拟机或 Linux 环境中构建：

```bash
docker build -t searchengine-ai-api .
```

`Dockerfile` 默认使用 `docker.m.daocloud.io/python:3.11-slim`，用于适配当前虚拟机网络环境。如果你的环境可以直接访问 Docker Hub，可以切回官方镜像：

```bash
docker build --build-arg PYTHON_IMAGE=python:3.11-slim -t searchengine-ai-api .
```

无 key 环境下可以先验证健康检查和检索摘要路径：

```bash
docker run --rm -p 8000:8000 --name searchengine-ai-api searchengine-ai-api

curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"epoll 水平触发和边缘触发有什么区别","top_k":3,"mode":"hybrid","embedding_provider":"fake","use_glm":false}'
```

如果要在 Docker 中调用真实 embedding 或 GLM，需要运行容器时显式注入环境变量：

```bash
docker run --rm -p 8000:8000 \
  -e ZAI_API_KEY="$ZAI_API_KEY" \
  -e ZAI_GLM_MODEL="glm-4-flash" \
  --name searchengine-ai-api \
  searchengine-ai-api
```

不要把 `.env` 或 API key 打进镜像；`.dockerignore` 已排除 `.env`、`.env.*`、日志和 embedding 缓存。

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
- Docker 已能启动当前 RAG API，但还没有加入健康检查、非 root 用户、生产级进程管理和多环境配置。
