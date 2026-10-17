# ADR-000：WSL 现有 all-in-rag 基线审计

- 状态：Accepted（阶段 00 审计完成；不代表所有旧能力已满足产品验收）
- 日期：2026-10-17
- 工作区：`/home/laiho/projects/all-in-rag`
- 基线提交：`cec956e80ae28d6c0ca84c9d39dddcf61ad633fd`
- 当前分支：`main`

## 1. 审计范围与结论

本阶段只读检查了本地 Git、README、`code/`、`data/`、`models/`、现有 `.venv`、部署文件与安全可运行的 smoke；没有调用收费 LLM、下载模型、重建索引、删除集合或修改旧业务源码。

现有工程是教学型 RAG 基线，不是可直接上线的多租户服务。C8 已具备 Markdown Parent–Child、FAISS、BM25、手写 RRF、查询改写和生成流水线；C9 已具备中文 jieba BM25、Milvus、图检索和三路 RRF；C7 有 LangGraph 检索—评估—改写循环。这些行为可作为回归基线，但缺少可信身份、tenant/workspace/ACL 检索前过滤、版本化、稳定引用、异步边界、可靠错误模型和自动化测试，因此采取“保留旧代码作 baseline，在新产品子目录包装或独立实现安全接口”的路线。

## 2. Git、环境与数据现状

| 项目 | 实际结果 | 决定 |
|---|---|---|
| Git | `main` @ `cec956e...`；存在 8 个已修改文件和 `WSL部署说明.md`、`deployment/` 等未跟踪内容 | 全部视为用户资产，不覆盖、不 reset/clean、不自动提交 |
| Remote | `origin=https://github.com/datawhalechina/all-in-rag.git`（fetch/push） | 与用户要求的个人 GitHub 账号不一致；阶段 00 不修改，首次 push 前单独确认个人仓库 URL |
| Git identity | `laihongxiang12` / `893596099@qq.com` | 与用户所述“之前的账号”表面一致，但不能替代 remote/认证核验 |
| Python | `.venv/bin/python` = Python 3.12.7；该 venv 无 `pip` 模块 | 保留旧环境；产品子项目阶段 01 使用独立 `uv` 环境 |
| 关键包 | torch 2.6.0、langchain 0.3.26、langgraph 0.3.34、pymilvus 2.5.11、neo4j 6.4.0、pydantic 2.14.1、openai 1.109.1 | 旧版本仅作基线；产品依赖须重新锁定并做兼容测试 |
| 本地模型 | `models/bge-small-zh-v1.5`，约 96 MB，模型 metadata 标注 MIT | 可用于离线 embedding 回归；记录模型 SHA/维度后再接正式索引 |
| 数据 | C1 Markdown、C2 PDF/TXT、C3 PDF/Excel/图片、C4 JSON/TXT、C9 Neo4j CSV/Cypher | 保留；它们不是 Northstar Tech 产品评测集，不可冒充最终 gold 数据 |
| 密钥 | 根目录 `.env` 存在；仅审计变量名，未显示值 | 不复制到产品代码、日志或文档；产品使用 `.env.example` |

用户已有改动包括将 C8/C9 LLM 适配到 DeepSeek、调整模型环境变量及 Neo4j 端口/内存配置。这些改动未被覆盖。

## 3. 旧模块 → 产品模块复用映射

| 原文件和函数/入口 | 可运行状态与证据 | 决定 | 产品落点 | 许可/风险与保留回归用例 |
|---|---|---|---|---|
| `code/C2/01_unstructured_example.py`：Unstructured PDF `partition` | 仅脚本级示例；读取固定 PDF；未验证加密/空文件/MIME/资源限制 | REIMPLEMENT（参考行为） | `app/ingestion/loaders.py` | 旧脚本无安全边界；保留 C2 PDF 作解析对照，新增原创 fixture |
| `code/C2/02_character_splitter.py`、`03_recursive_character_splitter.py`、`04_semantic_chunker.py` | 固定 TXT 的字符/递归/语义分块示例；语义分块可能加载模型 | REFERENCE / WRAP（仅稳定、无副作用部分） | `app/ingestion/chunking.py` | 不直接复制教学脚本；回归中文分隔符和 overlap 行为 |
| `code/C8/rag_modules/data_preparation.py`：`load_documents`、`chunk_documents`、`get_parent_documents` | 已实现 Markdown 标题分块、父子映射和父文档回填；parent ID 基于相对路径稳定；child ID 使用 UUID，不稳定；无版本/ACL/page | REFACTOR/WRAP | `app/ingestion/chunking.py`、`pipeline.py`、`app/retrieval/context.py` | 优先保留标题感知行为；改为内容/位置派生的稳定 chunk ID，并补 tenant/version/ACL/context budget |
| `code/C3/02_langchain_faiss.py`、`code/C8/rag_modules/index_construction.py` | FAISS 构建/保存/加载可见；自带 smoke 的 LangChain 内存检索通过（50 chunks、top2） | WRAP（baseline only） | `app/retrieval` 的 local adapter | `allow_dangerous_deserialization=True` 不进入不可信生产路径；保留 Dense-only 消融 |
| `code/C8/rag_modules/retrieval_optimization.py`：`hybrid_search`、`_rrf_rerank` | 纯内存 RRF 回归通过：重合文档排名第一，分数符合 `1/(60+rank)`；BM25 使用默认 tokenizer | WRAP/REFACTOR | `app/retrieval/hybrid.py` | RRF 算法可复用；当前按正文 MD5 去重会合并同内容不同权限文档；`metadata_filtered_search` 是检索后过滤，严禁用于授权 |
| `code/C9/rag_modules/hybrid_retrieval.py`：`_tokenize_chinese`、`bm25_search`、`_rrf_merge`、`hybrid_search` | 源码确认 jieba+stopwords BM25、三路 RRF、父文档回填；未运行完整 C9 | WRAP（算法回归）/REFACTOR | `app/retrieval/lexical.py`、`hybrid.py`、`context.py` | C9 RRF 的 source 内去重设计可参考；没有 tenant/ACL 字段或 pre-filter，不能直接上线 |
| `code/C4/01_hybrid_search.py`、`01_hybrid_search_v2.py` | 使用 Milvus `RRFRanker(k=60)`；脚本导入即连接、建集合并最终 drop collection，还可能下载模型 | BLOCKED（禁止阶段 00 运行）/REFERENCE | `app/stores/milvus.py`、`app/retrieval/hybrid.py` | 破坏性副作用明显；仅参考 SDK 调用形态，正式 adapter 必须无 import-time I/O |
| `code/C9/rag_modules/milvus_index_construction.py` | 存在 collection/schema/upsert/search/delete 能力；当前 Milvus 19530 未监听，自带 smoke 在连接处失败 | BLOCKED（real）/REFACTOR | `app/stores/milvus.py` | schema 未含 tenant/workspace/ACL/version；必须服务端 pre-filter，禁止先全库搜索再过滤 |
| `code/C4/07_rerank_and_refine.py`：`ColBERTReranker` | import/运行会加载 `bert-base-uncased` 并调用 DeepSeek 压缩；未离线验证，且所谓 ColBERT 是简化 MaxSim | BLOCKED / REIMPLEMENT | `app/retrieval/reranker.py` | 不能宣称正式 Cross-Encoder；建立协议、批量/超时/fake 和锁定模型后重测 |
| `code/C8/rag_modules/generation_integration.py`：`query_rewrite`、`query_router`、生成/stream | 本地改为 OpenAI-compatible DeepSeek；需真实 key，阶段 00 未调用 | WRAP 行为 / REIMPLEMENT 边界 | `app/llm/providers/*`、`app/agents/nodes/*` | 同步调用、自由文本路由、无 usage/timeout/schema/citation；只保留提示行为作对照 |
| `code/C7/agentic_rag/agent.py`：`AgenticRAG` | LangGraph analyze→retrieve→critique→refine→retrieve→generate；未调用付费模型 | REFACTOR（仅图行为参考） | `app/agents/state.py`、`graph.py`、`nodes/` | 无 policy/tenant/citation；JSON 失败默认“充分”不安全；可建议 web；达到 max_iter 的边直接 END，可能无最终答案 |
| `code/C7/agentic_rag/tools.py` | vector/web 工具示例；包含外部 Web 能力 | REIMPLEMENT | `app/tools/policy.py`、`registry.py` | P0 不注册 Web；所有工具必须先经 PolicyEngine |
| `code/C9/rag_modules/intelligent_query_router.py`、`graph_rag_retrieval.py` | 有传统/图/组合路由和图遍历；依赖 LLM、Neo4j 和业务菜谱图 | REFERENCE（P2） | 暂不进入 P0；后续 ADR 决定 | GraphRAG 是 P2，不能抢占授权 RAG 主路径 |
| `code/C8/C9 generation` | 能基于检索上下文生成答案；未实现结构化 citation 验证 | REIMPLEMENT 边界 | `app/agents/nodes/generate.py`、`verify.py` | 生成内容不能直接当 citation；引用必须从授权 evidence 构造 |

## 4. 实际执行证据

### 4.1 通过

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python deployment/smoke_test.py`：GPU `NVIDIA GeForce RTX 4060 Laptop GPU` 可用；LangChain 本地 embedding/内存检索通过（50 chunks）；LlamaIndex 在禁用真实 LLM 后本地检索通过。
- Neo4j 只读连接：容器 `neo4j-db` healthy，`MATCH (n) RETURN count(n)` 得到 5841。
- C8 `_rrf_rerank` 纯内存回归：输入 Dense `[年假制度, 报销制度]` 与 BM25 `[报销制度, 安全制度]`，输出 `报销制度 > 年假制度 > 安全制度`，重合项得分 `0.03252247`。
- `git diff --check`：通过，无 whitespace error。

### 4.2 BLOCKED / NOT_VERIFIED

- 同一次 smoke 在 `localhost:19530` 连接 Milvus 超时；`docker ps` 仅显示健康的 Neo4j，19530 未监听。因此 Milvus real baseline 为 **BLOCKED**，不能沿用旧日志宣称当前已通过。
- C8/C9 端到端问答会调用真实 DeepSeek；本阶段按约束未产生收费调用，标 **NOT_VERIFIED**。
- C4 rerank/hybrid 脚本会下载模型、创建/删除集合或调用 LLM，本阶段不运行，标 **BLOCKED**。
- 仓库没有正式 pytest 测试套件；`deployment/smoke_test.py` 是唯一发现的 smoke，不能等同产品 CI。

## 5. 安全与工程缺口

1. `code/C7`、`C8`、`C9` 未发现 `tenant_id`、`workspace_id`、ACL、document/index version 或可信 `Principal` 契约。
2. C8 `metadata_filtered_search` 明确先取候选后过滤，不可用于权限；新实现必须在存储/检索端 pre-filter。
3. 旧 citation 只是生成提示里的文档编号，没有验证 document/chunk/version/source 的机制。
4. 多处 import/构造阶段创建模型、网络或数据库连接，且主要为同步 API；产品模块必须依赖注入、async、timeout、semaphore 和有限重试。
5. C4 示例会删除 Milvus collection；旧脚本只能隔离运行，不能作为 CI 默认命令。
6. C9 Compose 含演示用固定 Neo4j 凭据；不能进入产品 Real Mode。

## 6. 许可与来源决策

本地 README 明确声明 CC BY-NC-SA 4.0；仓库根未发现独立 `LICENSE` 文件。后续复用旧代码必须保留 Datawhale/all-in-rag 来源、署名和相同方式共享/非商业约束，不将衍生代码宣称为完全独立原创或任意商用许可。`models/bge-small-zh-v1.5` 的本地 metadata 标注 MIT，但模型和仓库代码许可应分别记录。无法确认具体文件授权边界时，只保留行为回归并独立实现产品接口。

## 7. 决策

- 原 `code/`、`docs/`、`data/`、`models/`、`.venv` 和部署资产保持原位，作为可复现 baseline。
- 新实现仅进入 `enterprise-agentic-rag/`，阶段 01 起使用独立 `uv` 项目。
- 复用顺序：验证行为 → 窄接口适配 → 小步重构 → 必要时独立实现。
- 任何旧模块进入产品前都必须补 tenant/ACL pre-filter、稳定 ID、版本、引用、错误和离线测试；未满足者不得标为生产可用。
- 首次 Git push 前，必须由用户确认个人 GitHub 仓库 URL，再调整/新增 remote；本阶段不改变 `origin`。
