# 07. RAG 技术详解 & Agent 串联 Demo

> 一篇从原理到工程的 RAG 笔记，覆盖核心流程、切分 / 检索 / 重排 / 评估，
> 以及若干高级模式，并在文末给出 **RAG + SQLite + Agent** 的可运行 Demo。

## 目录

- [0. 什么是 RAG](#0-什么是-rag)
- [1. 为什么需要 RAG](#1-为什么需要-rag)
- [2. RAG 核心流程](#2-rag-核心流程)
- [3. 关键组件详解](#3-关键组件详解)
  - [3.1 Loader](#31-loader)
  - [3.2 Splitter](#32-splitter)
  - [3.3 Embedding](#33-embedding)
  - [3.4 Vector Store / Index](#34-vector-store--index)
  - [3.5 Retriever](#35-retriever)
  - [3.6 Reranker](#36-reranker)
  - [3.7 Generator + 引用](#37-generator--引用)
- [4. 切分策略](#4-切分策略)
- [5. 检索策略进阶](#5-检索策略进阶)
- [6. 重排 (Reranking)](#6-重排-reranking)
- [7. 高级 RAG 模式](#7-高级-rag-模式)
  - [7.1 Corrective RAG (CRAG)](#71-corrective-rag-crag)
  - [7.2 Self-RAG](#72-self-rag)
  - [7.3 GraphRAG](#73-graphrag)
  - [7.4 Agentic RAG](#74-agentic-rag)
- [8. 评估与监控](#8-评估与监控)
- [9. Demo：RAG + SQLite + Agent 串联](#9-demorg--sqlite--agent-串联)
  - [9.1 场景与目标](#91-场景与目标)
  - [9.2 完整可运行代码](#92-完整可运行代码)
  - [9.3 三个典型问题演示](#93-三个典型问题演示)
  - [9.4 你能从这个 Demo 学到](#94-你能从这个-demo-学到)
- [10. 局限与最佳实践](#10-局限与最佳实践)
- [11. 参考资源](#11-参考资源)

---

## 0. 什么是 RAG

**RAG (Retrieval-Augmented Generation)**：在生成阶段让模型先"检索"外部知识源，
再"基于检索到的内容"回答。

用一句话把 RAG 的工作流程定义清楚：

> **Question + Retriever(KB) → Context → LLM(Prompt) → Answer (with citations)**

它把"模型参数中的静态世界知识"换成"可即时检索、运行时可控、可追溯"的外部知识源。

---

## 1. 为什么需要 RAG

| 痛点 | RAG 的应对 |
| --- | --- |
| 模型知识陈旧 | 把最新文档放进知识库 |
| 私有知识（公司政策、代码库、产品手册） | 在私有库上检索 |
| 幻觉 | 让模型引用原文，依据不够时拒答 |
| 不能给出可追溯出处 | 检索结果本身就带 source / page |
| 长上下文贵、漏读 | 只把"最相关"的 K 段送入上下文 |
| 监管合规（医疗/金融） | 用审计日志 + 受控文档 |

和长上下文、fine-tuning 的对比：

| 维度 | 长上下文 | Fine-tune | **RAG** |
| --- | --- | --- | --- |
| 实时更新 | 否（窗口有限） | 否（重训） | **是** |
| 可解释 / 可追溯 | 弱 | 弱 | **强** |
| 私有数据门槛 | 极高 token 成本 | 中（数据 + 算力） | **低**（只需检索） |
| 可控制拒答 | 弱 | 弱 | **强** |
| 适合规模 | 数百份短文档 | 大量 SFT 数据 | **中到大规模、多源、变动快** |

---

## 2. RAG 核心流程

```
┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Documents │ -> │  Splitter   │ -> │   Embedding  │ -> │ Vector Index │
└────────────┘    └─────────────┘    └──────────────┘    └──────┬───────┘
                                                                │ (离线建索引)
                                                                ▼
User Query ──► Retriever ──► Top-k Chunks ──► LLM(Prompt) ──► Final Answer
                                       ▲
                                   Reranker (可选)
```

离线一次：在文档集合上做"切分 → 嵌入 → 建索引"。
在线每次：拿到问题 → 检索 →（重排）→ 拼接成 Prompt → 让模型回答。

---

## 3. 关键组件详解

### 3.1 Loader

把外部格式（PDF、HTML、Markdown、Notion、Confluence、Slack、Code…）装载成 `Document(page_content, metadata)`。
- 元数据是 RAG 的"血管"：日期、来源、作者、章节路径，都能直接过滤。

### 3.2 Splitter

把长文切成大小合适、语义完整的片段（chunk）。
- 经验值：中文 200–500 字 / 英文 200–800 token，相邻 chunk 重叠 10–20%。

### 3.3 Embedding

把文本编码为向量，使语义相近的文本距离近。
- 双语 / 通用：`BAAI/bge-m3`, `intfloat/e5-large-v2`, `BAAI/bge-small-zh-v1.5`, OpenAI `text-embedding-3-*`。
- 选型三件事：MTEB 排名、语言、维度与存储成本。

### 3.4 Vector Store / Index

- **本地 / 轻量**：FAISS、Chroma。
- **生产**：pgvector、Pinecone、Weaviate、Milvus、Qdrant、Elasticsearch dense_vector。
- 支持 ANN（最近邻近似）、metadata filter、混合检索（hybrid）。

### 3.5 Retriever

```python
# 纯向量
retriever = vs.as_retriever(search_kwargs={"k": 4})

# 多路混合（向量 + BM25）
from langchain.retrievers import EnsembleRetriever
ret = EnsembleRetriever(retrievers=[vs_ret, bm25_ret], weights=[0.7, 0.3])

# 多查询改写
from langchain.retrievers.multi_query import MultiQueryRetriever
ret = MultiQueryRetriever.from_llm(retriever=vs_ret, llm=llm)

# 自查询（带 metadata 过滤）
from langchain.retrievers.self_query.base import SelfQueryRetriever
```

### 3.6 Reranker

两阶段：粗排（向量取 Top-50）+ 精排（cross-encoder 重排到 Top-5）。
- 代价：多一次重排模型推理。
- 收益：精度飞跃，特别是条款、表格、长尾问题。

### 3.7 Generator + 引用

结构化 prompt：

```text
系统：只根据下面"上下文"作答。如果上下文不含答案，请回答"信息不足"。
上下文：
  [1] {doc1}
  [2] {doc2}
  ...
问题：{question}
```

并让模型同时给出引用编号，便于前端可点击跳转。

---

## 4. 切分策略

| 场景 | 策略 | 备注 |
| --- | --- | --- |
| 普通散文 | `RecursiveCharacterTextSplitter` | 默认即可 |
| Markdown | `MarkdownTextSplitter` / 自定义按标题 | 保留结构 |
| 代码 | Language-aware splitter | 保留函数边界 |
| 长表格 | 行级切分 + 列名 repeat | 防表格语义被腰斩 |
| 长 PDF | 段落 + 页眉页脚 + 表格单独处理 | 用 PyMuPDF / Unstructured |
| 切片重叠 | 10–20% | 减少上下文割裂 |

更细的进阶：基于语义距离的"SemanticChunker"，基于上下文的"Contextual Retrieval"（Anthropic 提出），会先对 chunk 生成一段说明文字再嵌入。

---

## 5. 检索策略进阶

| 策略 | 思路 | 何时用 |
| --- | --- | --- |
| Dense | 向量相似度 | 长尾 / 语义 / 多语言 |
| Sparse (BM25) | 关键词匹配 | 缩写号 / SKU / 专业词 |
| Hybrid | 加权融合 | 默认推荐 |
| Multi-Query | LLM 改写出多个 query 再合并 | 用户问题模糊 |
| HyDE | LLM 假设答案→拿假设答去检索 | 0-shot 长尾 |
| Step-back | 先答"上位概念"再回答原问 | 推理型问题 |
| Self-Query | 提取 metadata 过滤条件 | 时间 / 来源 / 类目等结构化筛选 |
| RAG-Fusion | 多 query + Reciprocal Rank Fusion | 检索词不确定 |

实现示例（Multi-Query + Hybrid + RRF）：

```python
# 三路：向量 + BM25 + 改写后向量 → RRF
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

bm25 = BM25Retriever.from_documents(splits)
vs   = Chroma.from_documents(splits, embeddings).as_retriever(search_kwargs={"k":8})
mq   = MultiQueryRetriever.from_llm(retriever=vs, llm=llm)

hybrid = EnsembleRetriever(retrievers=[mq, bm25, vs], weights=[0.5, 0.2, 0.3])
docs = hybrid.invoke("用户原问题")
```

---

## 6. 重排 (Reranking)

常见做法：

- 向量检索 → Top-50
- Cross-encoder rerank（`BAAI/bge-reranker-large`, `Cohere Rerank 3`, LLM-based reranker）→ Top-5
- 截断送入 LLM

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")
pairs    = [(q, d.page_content) for d in candidates]
scores   = reranker.predict(pairs)
top      = [d for _, d in sorted(zip(candidates, scores), key=lambda x: -x[1])][:5]
```

经验值：rerank 后命中率 +10%–30%，token 成本基本不变（反而因为少送一堆弱相关而省 token）。

---

## 7. 高级 RAG 模式

### 7.1 Corrective RAG (CRAG)

评估检索质量 → 决定"采纳 / 用 web search 兜底 / 重写 query 再检"。

```python
def crag_step(query, docs):
    if grader(query, docs) == "correct":
        return docs
    elif grader(query, docs) == "incorrect":
        return web_search(query)
    else:                              # ambiguous
        return merge(web_search(query), docs)
```

### 7.2 Self-RAG

模型自反思"需不需要检索 / 检索够不够 / 生成是否被支持"，
用特殊 token 决定行为。

### 7.3 GraphRAG (微软)

先用 LLM 把文档建成实体-关系图，检索时既检索实体又检索社区摘要。
适合"全局性问题"，如"公司过去一年最重要的失败案例是哪三个？"。

### 7.4 Agentic RAG

把"检索"做成 Agent 的一个 Tool，让 Agent 自由选择：
- 是否需要检索？
- 用哪个数据源？
- 是否需要先计划再检索？
- 是否需要多次检索 / 多跳推理？

这正是下文 Demo 的实现思路。

---

## 8. 评估与监控

| 维度 | 指标 |
| --- | --- |
| 检索 | Recall@k, MRR, nDCG@k |
| 端到端 | EM、F1、Faithfulness、Answer Relevance（LLM-judge） |
| 拒答 | 当上下文不足，模型是否愿意说"不知道" |
| 引用 | Citation Recall / Precision |
| 工程 | P95 延迟、token 用量、缓存命中率 |

工具：`ragas`、`TruLens`、`DeepEval`、`LangSmith`、自建评测集。

---

## 9. Demo：RAG + SQLite + Agent 串联

### 9.1 场景与目标

我们要做一个"业务问答机器人"：

- **私域文档知识**（公司政策 / RAG 概念）→ 用 RAG 检索
- **结构化业务数据**（订单、产品）→ 用 Text-to-SQL 查 SQLite
- **数学计算**（总额、加权等）→ 用计算器

把三种能力都做成 **Tool**，挂在同一个 LangGraph Agent 上，
由 LLM 决定何时调用哪个。这正是 **Agentic RAG** 的典型形态。

### 9.2 完整可运行代码

```python
# -*- coding: utf-8 -*-
"""
RAG + SQLite + Agent —— 端到端可运行 demo
依赖：pip install -U langchain langgraph langchain-anthropic chromadb \
                    sentence-transformers langchain-community pydantic
"""

import os, re, sqlite3, tempfile, pathlib
from typing import Annotated, TypedDict
from operator import add

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# =============== 1) 大模型（请把 key 放到环境变量） =================
from langchain_anthropic import ChatAnthropic
os.environ.setdefault("ANTHROPIC_API_KEY", "<YOUR_KEY>")
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

# =============== 2) 文档知识：自洽的小型私域语料 ===================
docs_text = """
公司退货政策：购买 30 天内可全额退款；30-90 天按 80% 退款；超过 90 天不可退款。
电子产品质量保修 1 年；非人为损坏免费维修或换货。
免运费门槛：订单金额满 99 元包邮；否则收取 10 元运费。
常用联系方式：客服电话 4000-000-000，工作时间 9:00-18:00。

什么是 RAG：RAG（Retrieval-Augmented Generation）是一种把外部知识库
接入大模型的技术，让模型在回答时可以引用专有或最新的信息，从而减少幻觉。
它的工作流是：先根据问题检索相关片段，再把这些片段作为上下文交给 LLM 生成答案。
"""

workdir = pathlib.Path(tempfile.mkdtemp(prefix="rag_demo_"))
(p := workdir / "policy.txt").write_text(docs_text, encoding="utf-8")

# =============== 3) 切分 + 嵌入 + 向量库 ==========================
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
raw_docs = TextLoader(str(p), encoding="utf-8").load()
splits   = splitter.split_documents(raw_docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vdb        = Chroma.from_documents(splits, embeddings,
                                   collection_name="policy",
                                   persist_directory=str(workdir/"chromadb"))
retriever  = vdb.as_retriever(search_kwargs={"k": 3})

# =============== 4) Tool #1: 文档检索（RAG） =======================
@tool
def search_policy(question: str) -> str:
    """检索公司内部文档（退货政策、产品质量、运输条款、RAG 概念等）。
    输入：自然语言问题。
    输出：基于检索到的上下文的回答，并附上引用片段。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你只能根据下面【上下文】作答，不要编造。如果上下文中找不到答案，"
         "请直说『信息不足』。回答里请用 [1][2] 这样的编号引用上下文。\n\n"
         "【上下文】\n{context}"),
        ("human", "{question}")
    ])
    def join(docs): return "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    chain = (
        {"context": retriever | RunnableLambda(join),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain.invoke(question)

# =============== 5) SQLite 业务库 + Tool #2: Text-to-SQL ==========
DB_SCHEMA = """
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT, category TEXT,
    price REAL, stock INTEGER
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    product_id INTEGER, qty INTEGER,
    status TEXT, created_at DATE,
    FOREIGN KEY(product_id) REFERENCES products(id)
);
"""

conn = sqlite3.connect(":memory:")
conn.executescript(DB_SCHEMA)
conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
    (1, "无线鼠标 X1",   "外设",     89.0,  120),
    (2, "机械键盘 K2",   "外设",    399.0,   35),
    (3, "4K 显示器 D27", "显示器", 1299.0,   12),
    (4, "USB-C 集线器 H7","配件",   159.0,  200),
])
conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
    (1, 1, 2, "已完成", "2026-07-12"),
    (2, 2, 1, "已发货", "2026-08-01"),
    (3, 3, 1, "已完成", "2026-07-25"),
    (4, 4, 5, "已完成", "2026-08-04"),
    (5, 1, 1, "退款中", "2026-06-30"),
])
conn.commit()

@tool
def query_database(question: str) -> str:
    """对本地 SQLite 业务库 (products / orders) 进行自然语言查询。
    工具内部先把问题转成 SQL，再安全执行并返回结果摘要。
    数据库模式：
    """ + DB_SCHEMA

    sql_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是 SQLite 专家。schema 如下：\n" + DB_SCHEMA +
         "\n只输出可执行的 SQL（一句；不加分号；不要解释）。"),
        ("human", "{question}")
    ])
    sql = (sql_prompt | llm | StrOutputParser()).invoke({"question": question})
    sql = re.sub(r"^```[a-z]*\n?|```$", "", sql, flags=re.M).strip()
    # 只允许 SELECT，防止破坏性语句
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return f"拒绝执行非查询语句：{sql}"
    try:
        rows = conn.execute(sql).fetchall()
    except Exception as e:
        return f"SQL 执行失败：{e}\nSQL：{sql}"
    if not rows:
        return f"SQL: {sql}\n结果：无记录"
    return f"SQL: {sql}\n结果({len(rows)} 行)：{rows}"

# =============== 6) Tool #3: 计算器 =================================
@tool
def calculator(expression: str) -> str:
    """计算数学表达式（Python 表达式语法，例如 '89*5 + 399'）。"""
    if not re.match(r"^[\d\s+\-*/().,%]+$", expression):
        return "表达式包含非法字符，请仅使用数字与运算符。"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败：{e}"

tools = [search_policy, query_database, calculator]

# =============== 7) LangGraph Agent ================================
class State(TypedDict):
    messages: Annotated[list, add]

def agent_node(state: State) -> State:
    bound = llm.bind_tools(tools)
    reply = bound.invoke(state["messages"])
    return {"messages": [reply]}

builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition,
                              {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())
CFG   = {"configurable": {"thread_id": "demo-thread"}}

def chat(q: str):
    print(f"\n> 用户：{q}")
    out = graph.invoke({"messages": [HumanMessage(content=q)]}, config=CFG)
    print("  助手：", out["messages"][-1].content)
    for m in out["messages"][1:]:
        kind = type(m).__name__
        if kind == "AIMessage":
            print(f"     · [LLM] {m.content or '(tool call)'}")
            for tc in (m.tool_calls or []):
                print(f"       └─ 调用工具：{tc['name']}({tc['args']})")
        elif kind == "ToolMessage":
            preview = (m.content[:80] + "…") if len(m.content) > 80 else m.content
            print(f"     · [Tool/{m.name}] {preview}")
```

### 9.3 三个典型问题演示

```python
chat("公司的退货政策是什么？30 天和 90 天分别是怎样？")
# 预期：Agent 调用 search_policy，回答引用 [1][2]。

chat("2026 年 6 月之后有哪些已完成的订单？金额各是多少？")
# 预期：Agent 生成 SELECT ... WHERE status='已完成' AND created_at > '2026-06-01'，
#      返回订单 + 关联产品与金额。

chat("最贵与最便宜的两件商品加起来多少钱？")
# 预期：Agent 用 query_database 找 max/min 价，
#      再用 calculator('price1+price2') 算总和。
```

### 9.4 你能从这个 Demo 学到

1. **RAG 作为 Tool**：把"检索 → 拼 prompt → 生成"封到 `search_policy`，让 LLM 像调普通函数一样调度。
2. **Text-to-SQL 作为 Tool**：把自然语言问题转 SQL 并安全执行（白名单 + try/except）。
3. **多源融合**：同一个 Agent 既能读文档（向量检索），也能查业务库（结构化查询），还能算（数值）。
4. **状态与持久化**：`MemorySaver` 让多轮对话保留上下文；进一步可替换为 `PostgresSaver` 做生产级持久化。
5. **可观测**：`ToolNode` + `messages` 流式回放，每一步的 LLM ↔ Tool 切换都能在 UI 看到。
6. **安全边界**：禁止非查询 SQL（`select|with` 校验）、禁止 calc 的非数字字符。

> 把这个 Demo 视为"模板"：换掉 `vdb`（Chroma → Pinecone）、换掉数据库（SQLite → Postgres）、
> 换掉模型（Anthropic → OpenAI）、加 Reranker / 增加权限审核，就可以成为生产级智能体。

---

## 10. 局限与最佳实践

- **检索不到不是模型错**：先用更全的混合检索 + 重排；问题往往是 chunk / 切分不对。
- **长文档 + 多跳**：先计划（Plan-and-Retrieve）再检索，效果比单跳好。
- **时新性**：把"实时数据源"也作为 Tool，而不是靠 PDF 再嵌。
- **校验**：重要业务（如医疗、合规）必须有人审 / 规则审。
- **指标先行**：上线前先建评测集（≥200 条 gold Q-A），每周回归。
- **缓存**：对稳定答案开语义缓存，省 token 省时延。
- **权限**：知识库按角色分索引；Tool 限制高风险动作。

---

## 11. 参考资源

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020.
- Anthropic, *Introducing Contextual Retrieval*, 2024.
- Microsoft, *GraphRAG: Unifying LLM and Knowledge Graph*, 2024.
- LangChain / LlamaIndex / RAGAS / TruLens 官方文档。
- Cohere Rerank、BGE-reranker、Splade、FalkorDB 等检索 / 重排生态。

---

> 与正文一致性：本文档为 `docs/07-rag.md`，是 README 中"AI Skill"系列笔记的第三篇专题。
