# Enterprise Agentic RAG

这是 `all-in-rag` 工作树中的增量产品子项目。旧 `code/`、`data/`、`docs/`、`models/` 和原虚拟环境继续作为 baseline；新服务从本目录独立安装、测试和运行。

当前实现包含 FastAPI 文档/查询接口、严格契约、离线 Fake、文档 Loader、版本化 Ingestion、PostgreSQL 元数据/ACL、Milvus Dense + 原生 BM25 + RRF 混合检索、受限 Rerank、LangGraph 有界检索图、Grounded Answer 与引用验证、Redis cache/checkpoint、可观测性、离线评测以及可构建的 Compose Demo。生产身份提供方和生产任务队列仍需部署方接入，不能把 Fake Demo 解释为生产环境。

## Fake Quickstart（不调用 DeepSeek）

完整 Demo 使用非 root API 镜像，并启动 PostgreSQL、Redis、Milvus、etcd 与 MinIO。先创建仅用于本机的 Compose 环境文件并替换占位值：

```bash
cp deploy/.env.compose.example deploy/.env.compose
docker compose --env-file deploy/.env.compose -f deploy/docker-compose.yml up -d --build --wait
uv run python -m scripts.seed_demo
uv run python -m scripts.smoke
```

浏览器打开 `http://127.0.0.1:8000/`。Fake Demo token 是 `demo-local-token`，只用于 `APP_ENV=local`、`APP_MODE=fake` 的原创虚构数据演示；它不是生产认证。页面展示答案、citations、trace_id、重试次数和完成状态。

停止容器但保留数据卷：

```bash
docker compose --env-file deploy/.env.compose -f deploy/docker-compose.yml stop
```

不要随意执行 `down -v`，该命令会删除 PostgreSQL、Redis、Milvus、etcd 与 MinIO 的命名卷。

资源上限写在 `deploy/docker-compose.yml`：API 768 MiB、PostgreSQL 512 MiB、Redis 256 MiB、etcd/MinIO 各 512 MiB、Milvus 4 GiB。宿主端口仅将 API 的 8000 映射到 WSL 本机的 `127.0.0.1`，后端服务留在 Compose 网络内。在 VS Code 的 WSL 窗口中，可通过 **Ports** 面板转发 8000 后从 Windows 浏览器访问；不要为了方便把固定 token Demo 暴露到公网或局域网。

## Real Backend 与真实付费模型

Compose Quickstart 故意运行 Fake QueryService，但仍真实启动后端容器并进行健康检查。真实 PostgreSQL/Milvus/Redis adapter 的集成测试使用本页后续章节中的 `TEST_*` 变量单独验证，不会被 Fake 替代。CI 的手动 Milvus job 额外合并 `deploy/docker-compose.integration.yml`，只为测试发布 `127.0.0.1:19530`；日常 Demo 不发布后端端口。

DeepSeek 只在显式 `APP_MODE=real`、配置 `LLM_API_KEY`、装配真实 QueryService 且操作者确认费用后调用。仓库、镜像、Compose 和 CI 均不包含密钥，CI 默认绝不调用付费模型；阶段 19 的离线指标不能当作 DeepSeek 效果。

## CI

`.github/workflows/ci.yml` 默认运行锁文件检查所依赖的安装、Ruff、mypy、unit/security/regression，以及真实启动的离线 API seed+smoke。Milvus 真实集成是独立的手动 job，会启动官方 Milvus stack 并运行两项真实测试；它不会以 memory fake 代替向量库。

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
