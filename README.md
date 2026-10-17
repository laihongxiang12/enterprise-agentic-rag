# Enterprise Agentic RAG

这是 `all-in-rag` 工作树中的增量产品子项目。旧 `code/`、`data/`、`docs/`、`models/` 和原虚拟环境继续作为 baseline；新服务从本目录独立安装、测试和运行。

当前实现包含 FastAPI 基础、严格契约、离线 Fake、文档 Loader、版本化 Ingestion、PostgreSQL 元数据/ACL，以及 Milvus Dense + 原生 BM25 + RRF 混合检索。Redis、Agent 图和真实 LLM 尚未完成，不能把健康检查或当前检索测试解释为完整 RAG 已可用。

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

分别运行离线单测与真实混合集成：

```bash
uv run pytest tests/unit -q
uv run pytest tests/integration/test_milvus_hybrid.py -m integration -q
```

`DEV_AUTH_ENABLED` 只允许用于 local、loopback、非 real 模式，并要求至少 32 字符的开发密钥。它是签名的演示身份边界，不是企业 SSO。

## 来源与许可

迁移和复用决策见 [`docs/adr/000-baseline.md`](docs/adr/000-baseline.md)。父仓库 README 声明 CC BY-NC-SA 4.0；复制或改造旧代码时必须保留来源和相应许可义务。
