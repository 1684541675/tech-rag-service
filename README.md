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

### AI-26：受控面试知识库 Agent

AI-26 在既有 revision-bound RAG Core 上增加单 Agent 编排；它不替换检索或 evidence gate，也不让模型获得任意文件、SQL 或写入能力。

```text
HTTP /agent/run (X-Request-ID)
  -> classify
  -> retrieve_evidence (active revision + hybrid retrieval + evidence gate)
  -> [证据不足] no_knowledge
  -> inspect_source (仅本 run 签发的 opaque ref_id)
  -> [问答] GLM answer + heading/line citation
  -> [掌握度更新] proposal -> LangGraph interrupt
                          -> approve: JSONL append
                          -> reject : zero write

每个 HTTP 请求 -> 脱敏结构化 trace（请求 ID、状态、错误码、任务哈希摘要）
```

真实宿主机验证已覆盖：证据不足问题短路为 `no_knowledge`；epoll ET 问题完成真实 embedding、混合检索、GLM 生成和 citation 返回。请求级 trace 不记录原始提问、完整上下文或 API key。

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
| `scripts/serve_ai26_agent.py` | AI-26 受控 Agent API；提供 `/health`、`/agent/run` 和审批恢复接口 |

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

- evidence gate 在固定回归集与本轮 MongoDB 缺口题中可短路为 `no_knowledge`，但不能保证所有开放问法都不会幻觉，citation 也不等同于答案正确性。
- 待审批运行和 run-scoped 引用只保存在单服务进程内；服务重启或多进程部署前，未审批运行不能恢复。
- 当前没有多租户、认证、限流、持久化审计、生产部署编排或高并发承诺。
- 根目录 `Dockerfile` 已验证 AI-26 API 的 `/health` 启动；真实检索仍依赖外部 PostgreSQL、OpenSearch 与 Qdrant，未声明为端到端单容器部署。

### AI-26 Agent 工程化边界

`/agent/run` 与 `/agent/{thread_id}/approval` 支持 `X-Request-ID`；服务会生成只含任务长度与哈希摘要的结构化 trace，避免把原始提问、密钥或完整上下文写入日志。GLM 生成调用配置有限次网络重试：仅重试网络异常、429 与 5xx，其他 HTTP 错误立即失败；超时或失败会受控降级，不能伪装成模型回答。

```powershell
docker build -t tech-rag-agent .
docker run --rm -p 8000:8000 tech-rag-agent
curl http://127.0.0.1:8000/health
```

该 smoke test 只验证 Agent API 进程可启动；执行真实 `/agent/run` 前仍需提供外部检索基础设施和 `ZAI_API_KEY`（如需生成）。

## Docker 状态

当前 `Dockerfile` 启动 `scripts.serve_ai26_agent:app`，通过 Uvicorn 暴露 `8000` 端口：

```dockerfile
CMD ["uvicorn", "scripts.serve_ai26_agent:app", "--host", "0.0.0.0", "--port", "8000"]
```

在虚拟机或 Linux 环境中构建：

```bash
docker build -t tech-rag-agent .
```

`Dockerfile` 默认使用 `docker.m.daocloud.io/python:3.11-slim`，用于适配当前虚拟机网络环境。如果你的环境可以直接访问 Docker Hub，可以切回官方镜像：

```bash
docker build --build-arg PYTHON_IMAGE=python:3.11-slim -t tech-rag-agent .
```

无 key 环境下可以先验证健康检查：

```bash
docker run --rm -p 8000:8000 --name tech-rag-agent tech-rag-agent

curl http://127.0.0.1:8000/health
```

容器内 `localhost` 指向容器自身，因此当前 Docker 镜像只用于 API 进程 smoke test。真实检索演示推荐在宿主机执行 `scripts/serve_ai26_agent.py`，使其连接本机 PostgreSQL、OpenSearch 与 Qdrant；如需将其做成端到端容器部署，应先把这些连接地址配置化并进行独立验证。

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
