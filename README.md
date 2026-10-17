# Enterprise Agentic RAG

这是 `all-in-rag` 工作树中的增量产品子项目。旧 `code/`、`data/`、`docs/`、`models/` 和原虚拟环境继续作为 baseline；新服务从本目录独立安装、测试和运行。

当前阶段只提供 FastAPI 应用工厂、配置边界、统一错误结构和 `/health/live`。尚未连接 Milvus、PostgreSQL、Redis 或真实 LLM，不能把健康检查成功解释为完整 RAG 已可用。

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

本阶段的 Fake Mode 仅证明 HTTP/config 契约能够离线运行，不代表 Milvus、ACL、检索、Agent、LLM、缓存或部署已经完成。Real Mode 也不会静默降级为 Fake Mode。

## 来源与许可

迁移和复用决策见 [`docs/adr/000-baseline.md`](docs/adr/000-baseline.md)。父仓库 README 声明 CC BY-NC-SA 4.0；复制或改造旧代码时必须保留来源和相应许可义务。
