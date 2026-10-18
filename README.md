# Enterprise Agentic RAG

这是 `all-in-rag` 工作树中的增量产品子项目。旧 `code/`、`data/`、`docs/`、`models/` 和原虚拟环境继续作为 baseline；新服务从本目录独立安装、测试和运行。

当前实现包含 FastAPI 文档/查询接口、严格契约、离线 Fake、文档 Loader、版本化 Ingestion、PostgreSQL 元数据/ACL、Milvus Dense + 原生 BM25 + RRF 混合检索、受限 Rerank、LangGraph 有界检索图、Grounded Answer 与引用验证。Redis durable checkpoint、生产身份提供方、生产任务队列和部署组合仍属于后续阶段，不能把离线 API 测试解释为生产部署已完成。

## WSL 快速开始

在 VS Code Remote - WSL 中打开旧仓库根目录，然后执行：

```bash
cd enterprise-agentic-rag
uv sync --extra dev
uv run ruff check .
uv run mypy app
uv run pytest tests/unit -q
uv run uvicorn app.main:app --reload
```

另一个 WSL 终端验证：

```bash
curl --fail --silent http://127.0.0.1:8000/health/live
```

预期响应：

```json
{"status":"ok"}
```

## 运行模式

- `APP_MODE=fake`：默认离线模式，不调用外部服务，供单测和早期演示使用。
- `APP_MODE=local`：仅允许后续阶段显式配置的本地 adapter；不会因为存在密钥而自动调用真实 API。
- `APP_MODE=real`：真实 provider 模式。当前至少要求 `LLM_API_KEY`，缺失时应用启动失败；后续阶段会逐项增加依赖 readiness 检查。

从模板创建本地配置：

```bash
cp .env.example .env
```

不要把真实密钥提交到 Git。应用设置可在测试中显式注入，模块导入阶段不会创建数据库、模型或网络连接。

## Fake Mode 与 Real Mode 的边界

Fake Mode 证明契约、解析、分块、版本切换和授权规则能够离线运行，不代表 PostgreSQL 集成、Milvus、Agent、LLM、缓存或部署已经完成。Real Mode 不会静默降级为 Fake Mode。

## 文档与查询 API

阶段 15 提供 `POST /v1/documents`、`GET/DELETE /v1/documents/{document_id}`、`POST /v1/query` 和 `POST /v1/query/stream`。所有 `/v1/*` 请求先由认证中间件验证 Bearer token，再构造服务端 `Principal`；API 请求不接受 `tenant_id`。文档上传使用 multipart，校验大小、扩展名/MIME、workspace、ACL、scope 和幂等键，成功后返回 `202` 与 pending job。生产任务队列和生产身份验证器必须通过 `ApiServices` 显式装配；未装配时接口返回 `503`，不会退化为匿名或 Fake 权限。

SSE 事件顺序为 `trace`、`status`、零到多个 `delta`、`citations`、`done`；失败路径以 `error` 结束且不发送 `done`。当前 LangGraph 只产出最终答案，所以 `delta` 是最终答案的确定性分段，`status.streaming_mode` 为 `simulated_final_answer_segments` 且 `token_level=false`，并非模型 token-level streaming。

离线 HTTP 契约测试不会监听端口或访问网络：

```bash
uv run pytest tests/integration/test_api.py -q
```

## PostgreSQL 与开发认证

复制 `.env.example` 为 `.env` 并替换其中的本地数据库密码，然后启动 PostgreSQL、等待健康检查并运行真实集成测试：

```bash
docker compose up -d postgres
docker compose ps postgres
uv run pytest tests/integration/test_postgres.py -m integration -q
```

容器只绑定 WSL 本机的 `127.0.0.1:5432`，数据保存在命名卷中。Python 使用 `psycopg[binary]` 自带的 `libpq` 运行时，不要求在 WSL 中通过 `sudo apt` 安装系统 `libpq`。停止服务使用 `docker compose stop postgres`；不要随意使用 `down -v`，它会删除本地数据库卷。

PostgreSQL migration、权限边界和集成测试说明见 [`docs/security.md`](docs/security.md)。没有设置 `TEST_DATABASE_URL` 时真实 PostgreSQL 测试会明确跳过，不能把跳过解释为真实数据库通过。

## Milvus

启动官方版本锁定的 Milvus 2.5.27，并初始化默认 collection：

```bash
docker compose up -d --wait milvus
uv run python -m scripts.milvus_init
uv run pytest tests/integration/test_milvus.py -m integration -q
```

Milvus 仅绑定 `127.0.0.1:19530`；etcd、MinIO 和 9091 端口不暴露到宿主机。版本选择、schema 和服务端 ACL 过滤设计见 [`docs/adr/001-milvus.md`](docs/adr/001-milvus.md)。原生 BM25、Jieba 中文分词、两路相同授权过滤和应用层 RRF 设计见 [`docs/adr/002-milvus-hybrid.md`](docs/adr/002-milvus-hybrid.md)。

阶段 08 的默认 collection 是 `enterprise_rag_hybrid_chunks`。Stage 07 的 `enterprise_rag_chunks` 不会被自动删除；Milvus 无法原地补加 BM25 function，因此真实数据需要重新 ingestion 后再切换。混合检索默认 dense top 30、lexical top 30、RRF `k=60`、融合 top 20，可通过 `RETRIEVAL_*` 环境变量配置。

## Redis cache、限流与 checkpoint

本地 Redis 8.6.7 只绑定 `127.0.0.1:6380`，避免覆盖机器上已有的 6379 服务。Redis 8 自带官方 LangGraph saver 所需的 Search/JSON 能力：

```bash
docker compose up -d --wait redis
uv run pytest tests/integration/test_checkpoint.py -m integration -q
```

retrieval cache 只保存已授权 chunk，不保存生成答案；key 包含 tenant、workspace、权限 hash、index/ACL 版本、query、top-k 与检索配置 hash。checkpoint thread key 绑定验证后的 Principal、workspace、session、groups 和 scopes，不能用客户端裸 `session_id` 跨租户恢复。官方 Redis Search checkpoint 只支持逻辑数据库 0，因此测试也使用 `/0` 并依赖严格前缀隔离。详细决策见 [`docs/adr/003-redis-checkpoint.md`](docs/adr/003-redis-checkpoint.md)。

## Rerank、Parent Context 与引用

阶段 09 默认把融合 top20 rerank 到 top8，模型调用超时或失败时保留原融合顺序并显式标记降级。`LocalCrossEncoderReranker` 只接收注入的本地同步模型并在线程中批量评分；项目不会在导入或测试时自动下载模型。离线测试使用确定性的 `FakeReranker`。

Parent 扩展会把可信 `AuthFilter` 传给存储层，并在返回后再次核对 tenant、workspace、document/version/index、visibility 和 group ACL。上下文默认限制 1200 个确定性估算 token 和 8 个 parent；生产模型可注入自己的 tokenizer。引用只从最终授权 child 的真实 `source_uri`、页码、chunk ID 和文档版本构造，缺失或版本不一致会失败。

分别运行离线单测与真实混合集成：

```bash
uv run pytest tests/unit -q
uv run pytest tests/integration/test_milvus_hybrid.py -m integration -q
```

`DEV_AUTH_ENABLED` 只允许用于 local、loopback、非 real 模式，并要求至少 32 字符的开发密钥。它是签名的演示身份边界，不是企业 SSO。

## 来源与许可

迁移和复用决策见 [`docs/adr/000-baseline.md`](docs/adr/000-baseline.md)。父仓库 README 声明 CC BY-NC-SA 4.0；复制或改造旧代码时必须保留来源和相应许可义务。
