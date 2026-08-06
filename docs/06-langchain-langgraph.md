# 06. LangChain 与 LangGraph 原理详解

> 一篇深入到原理层面的工程参考，覆盖 LangChain 的核心抽象 / LCEL 与可运行协议，
> 以及 LangGraph 的状态机模型、持久化、人机协同与多 Agent 编排。

## 目录

- [0. 写在前面：从"模型调用"到"应用层 Agent"](#0-写在前面从模型调用到应用层-agent)
- [1. LangChain 总览](#1-langchain-总览)
  - [1.1 设计哲学：组合优于继承](#11-设计哲学组合优于继承)
  - [1.2 核心抽象：Runnable 协议](#12-核心抽象runnable-协议)
  - [1.3 关键组件](#13-关键组件)
  - [1.4 LCEL：LangChain Expression Language](#14-lcellangchain-expression-language)
  - [1.5 工作原理：流式 / 批 / 异步 / 重试](#15-工作原理流式--批--异步--重试)
  - [1.6 集成生态](#16-集成生态)
- [2. LangGraph 总览](#2-langgraph-总览)
  - [2.1 设计动机](#21-设计动机)
  - [2.2 核心概念：State / Node / Edge](#22-核心概念state--node--edge)
  - [2.3 工作原理：Pregel 风格执行模型](#23-工作原理pregel-风格执行模型)
  - [2.4 与 LangChain 的关系](#24-与-langchain-的关系)
  - [2.5 持久化、检查点、时间旅行](#25-持久化检查点时间旅行)
  - [2.6 多人协作与人机协同](#26-多人协作与人机协同)
  - [2.7 多 Agent 编排模式](#27-多-agent-编排模式)
- [3. 代码示例](#3-代码示例)
  - [3.1 LangChain：LCEL + RAG](#31-langchainlcel--rag)
  - [3.2 LangChain：Tools + ReAct Agent](#32-langchaintools--react-agent)
  - [3.3 LangGraph：可中断 + 检查点的人机协同](#33-langgraph可中断--检查点的人机协同)
  - [3.4 LangGraph：Supervisor 多 Agent](#34-langgraphsupervisor-多-agent)
- [4. 选型决策树](#4-选型决策树)
- [5. 性能、调试、可观测性](#5-性能调试可观测性)
- [6. 局限与常见反模式](#6-局限与常见反模式)
- [7. 概念对照表](#7-概念对照表)
- [8. 参考资源](#8-参考资源)

---

## 0. 写在前面：从"模型调用"到"应用层 Agent"

裸用 OpenAI / Anthropic SDK 写 `client.messages.create(...)` 解决的是"一次推理"。
真实应用要解决的是：

- **多步**：要做"先检索 → 再推理 → 再调用工具 → 再总结"。
- **可恢复**：要持久化中间状态，崩溃后能续跑。
- **可控**：要让人类在某些步骤打断、改写、批准。
- **可组合**：要让"翻译 + 摘要 + 分类"等节点可以自由拼接。
- **多角色**：要让多个子 Agent 分工对话。

LangChain 提供"组合层"，LangGraph 提供"状态机 + 持久化层"。两者结合，便覆盖了从原型到生产 Agent 的工程需求。

---

## 1. LangChain 总览

LangChain 是一个**面向 LLM 应用的组合框架**（Python / JavaScript 双实现）。它把大模型应用抽象成可拼接的"管道"，把每个组件统一到同一接口 (`Runnable`) 下，使"模型调用 + 数据接入 + 工具使用 + 记忆 + 解析"可以像函数组合一样被编排。

### 1.1 设计哲学：组合优于继承

LangChain 不主张"造一个大类把一切囊括进去"，而是：

> 一切皆 `Runnable`，一切可 `pipe` ( `|` )。

只要一个对象实现了 `Runnable`，它就能与另一个 `Runnable` 用同一套 API 串起来。这种"协议先于实现"的设计带来三个好处：

1. **可插拔**：换模型、换检索器、换工具不需要改业务代码。
2. **可观察**：每一步都暴露中间结果、流式事件、回调钩子。
3. **可测试**：每个 Runnable 都能独立 mock。

### 1.2 核心抽象：Runnable 协议

`Runnable` 是一个具有以下能力的协议类（Python 中为 `runnable.Runnable` / `langchain_core.runnables`）：

| 能力 | 说明 |
| --- | --- |
| `invoke(input)` | 同步单次调用 |
| `ainvoke(input)` | 异步单次调用 |
| `stream(input)` | 同步流式输出 |
| `astream(input)` | 异步流式输出 |
| `batch(inputs)` | 批处理 |
| `abatch(inputs)` | 异步批处理 |
| `astream_events(...)` | 细粒度事件流 |
| `|`（pipe） | 与其他 Runnable 组合 |

关键点：**输入 / 输出沿管道流动**。一个简单组合 `prompt | model | parser` 中，`prompt` 把 dict 转为 PromptValue，`model` 把 PromptValue 转为 `AIMessage`，`parser` 把 `AIMessage` 转为 str。LangChain 内部用 *type-selector* 把上游输出类型映射到下游期望类型，让开发者不用手动解包。

### 1.3 关键组件

| 组件 | 作用 | 关键类 |
| --- | --- | --- |
| **Models** | 统一接入多家 LLM / Chat | `ChatOpenAI`, `ChatAnthropic`, `ChatVertexAI`, `HuggingFacePipeline` |
| **Prompts** | 模板、few-shot、消息模板 | `ChatPromptTemplate`, `MessagesPlaceholder`, `FewShotChatMessagePromptTemplate` |
| **Output Parsers** | 把模型输出解析成结构化字段 | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` |
| **Document Loaders** | 把外部数据加载为 `Document` 列表 | `PyPDFLoader`, `WebBaseLoader`, `NotionDBLoader`, `CSVLoader` |
| **Text Splitters** | 切片 | `RecursiveCharacterTextSplitter`, `MarkdownTextSplitter` |
| **Embeddings** | 文本 → 向量 | `OpenAIEmbeddings`, `CohereEmbeddings`, `HuggingFaceEmbeddings` |
| **Vector Stores** | 向量检索后端 | `FAISS`, `Chroma`, `pgvector`, `Pinecone`, `Weaviate` |
| **Retrievers** | 把向量库 / BM25 / 混合检索抽象成统一接口 | `VectorStoreRetriever`, `BM25Retriever`, `EnsembleRetriever`, `MultiQueryRetriever` |
| **Tools** | 让 LLM 可调用的"工具"声明 | `BaseTool`（`name` / `description` / `args_schema` / `_run` / `_arun`） |
| **Memory / ChatHistory** | 会话级记忆 | `ChatMessageHistory`, `RedisChatMessageHistory`, `PostgresChatMessageHistory` |
| **Agents** | 把"ReAct / Function Calling"打包成 `Runnable` | `create_openai_tools_agent`, `create_react_agent` |
| **Callbacks** | 全链路观测 | `StdOutCallbackHandler`, `LangSmith`, `Langfuse` |
| **Caching** | 同输入复用结果 | `InMemoryCache`, `SQLiteCache`, `RedisCache` |

> 几乎所有组件最终都实现 `Runnable`，因此你写的"提示词 + 模型 + 解析器"和"加载器 + 切分器 + 嵌入 + 检索器 + 模型 + 解析器"是**同一类对象**。

### 1.4 LCEL：LangChain Expression Language

LCEL 是 LangChain 在 0.1+ 推出的声明式编排语法。它本质上就是 `Runnable` 的"糖衣"。

#### 1.4.1 基本组合

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位善于把复杂概念讲清楚的中文老师。"),
    ("human", "请用一段话解释：{topic}")
])
model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3)
parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"topic": "LCEL 的 pipe 操作"}))
```

- `prompt` 接受 `{"topic": ...}` 输出 `PromptValue`
- `model` 接受 `PromptValue` 输出 `AIMessage`
- `parser` 接受 `AIMessage` 输出 `str`
- `|` 表示"上一个的输出 -> 下一个的输入"

#### 1.4.2 并行 (RunnableParallel)

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "summary":  summary_chain,
    "keywords": keywords_chain,
})
result = parallel.invoke({"text": long_doc})
# {'summary': '...', 'keywords': [...]}
```

#### 1.4.3 回退 (RunnableWithFallbacks)

```python
safe_chain = primary_chain.with_fallbacks([fallback_chain])
```

#### 1.4.4 重试与超时

```python
chain = (
    prompt
    | model.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    | model.with_fallbacks([other_model])
    | parser
)
```

#### 1.4.5 透传自定义函数 (RunnableLambda)

```python
from langchain_core.runnables import RunnableLambda

clean = RunnableLambda(lambda x: x["text"].strip())
chain = retriever | clean | prompt | model | parser
```

### 1.5 工作原理：流式 / 批 / 异步 / 重试

#### 1.5.1 流式

```python
for chunk in chain.stream({"topic": "流式输出"}):
    print(chunk, end="", flush=True)
```

每个 `Runnable` 都暴露 `stream` / `astream`，从而可以做到"边生成边吐字符"。更细粒度：

```python
async for event in chain.astream_events({"topic": "..."}, version="v2"):
    print(event["event"], event["name"], event.get("data"))
```

事件包括 `on_chain_start` / `on_chat_model_stream` / `on_tool_start` 等，可以分别展示到 UI / 接入 trace 系统。

#### 1.5.2 批

```python
results = chain.batch([{"topic": "A"}, {"topic": "B"}, {"topic": "C"}])
```

`RunnableConfig` 里的 `max_concurrency` 控制并发度。

#### 1.5.3 异步

```python
result = await chain.ainvoke({"topic": "..."})
```

#### 1.5.4 可配置

```python
prompt_factory = prompt.configurable_fields(
    model=ConfigurableField(id="model", name="Model")
)
chain = prompt_factory | model | parser

chain.invoke({"topic": "..."},
             config={"configurable": {"model": "claude-opus-4-1"}})
```

让"运行时选模型"在不改代码的前提下完成。

### 1.6 集成生态

- **`langchain`** / **`langchain-core`**：高层 API + Runnable 协议。
- **`langchain-community`**：社区贡献的大量 Loader / VectorStore / Tool。
- **`langchain-anthropic` / `langchain-openai` / `langchain-google-genai` ...**：模型 adapter。
- **`langgraph`**：见下章。
- **`langsmith`**：官方 trace / 评测平台。
- **`langserve`**：把 `Runnable` 一键用 FastAPI 暴露成 REST。

---

## 2. LangGraph 总览

LangGraph 是 LangChain 团队推出的**面向有状态、多 Actor、长期运行 Agent 应用**的编排框架。它把"Agent 流程"建模为**有向图**：节点是函数，边是状态转移。

适合场景：

- 多步、有显式状态的 Agent（如 Long-running workflow）
- 多人协同 / 人机协同（Human-in-the-loop）
- 多 Agent 协作（Supervisor、Subgraph）
- 需要**中断 + 恢复**的工作流（审批、回填）

### 2.1 设计动机

ReAct / Function Calling 的循环是"无状态"的：

```
LLM ↔ Tool ↔ Observation ↔ LLM ↔ ...
```

应用到真实业务，会撞到几个问题：

1. **状态在哪？** 多步数据（如用户偏好、检索缓存、已尝试的工具列表）需要显式存放。
2. **谁能打断？** 支付前要审批，不能 LLM 一句话就过去。
3. **谁能续跑？** 服务重启后如何从上次断点恢复？
4. **谁能回看？** 调试 / 重放 / 审计。
5. **多 Agent 怎么对话？** 一个 Agent 调度多个子 Agent。

LangGraph 的回答是：**用图（Graph）+ 检查点（Checkpoint）+ 显式状态（State）。**

### 2.2 核心概念：State / Node / Edge

| 概念 | 含义 | 一句话定义 |
| --- | --- | --- |
| **StateGraph** | 状态图 | 用一个 reducer 描述的图 |
| **State** | 全局状态 | TypedDict / Pydantic + reducer |
| **Node** | 节点 | 一个 Python 函数，接收 state，输出 state 的更新 |
| **Edge** | 边 | 节点之间的转移（默认静态） |
| **Conditional Edge** | 条件边 | 由"路由函数"决定的动态转移 |
| **Entry Point** | 入口 | 图开始的节点 |
| **Checkpoint** | 检查点 | 每一步执行后保存的状态快照 |
| **Thread** | 会话 | 同一 thread 共享同一 checkpointer |
| **Interrupt** | 中断 | 在某节点前动态暂停，等待外部输入 |
| **Subgraph** | 子图 | 把另一个 `StateGraph` 当节点调用 |

#### 2.2.1 最简单的例子

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    text: str
    length: int

def get_length(state: State) -> dict:
    return {"length": len(state["text"])}

builder = StateGraph(State)
builder.add_node("get_length", get_length)
builder.add_edge(START, "get_length")
builder.add_edge("get_length", END)

graph = builder.compile(checkpointer=MemorySaver())
graph.invoke({"text": "hello"}, config={"configurable": {"thread_id": "1"}})
# {'text': 'hello', 'length': 5}
```

### 2.3 工作原理：Pregel 风格执行模型

LangGraph 的执行借鉴了 Google 的 Pregel（BSP，Bulk Synchronous Parallel）：

1. **每一步（super-step）执行一组被激活的节点**。
2. 节点读取 state，写入自己负责的字段（**partial update**）。
3. 图引擎把 partial update 用 reducer 合并到 state。
4. 再由条件边函数读新 state，决定下一步激活哪些节点。
5. 反复直到没有任何活跃节点。

这种"节点只 update 自己的字段，state 集中维护"的模型类似 Redux 的 reducer 思路，对多人协作、可观测、回放都友好。

### 2.4 与 LangChain 的关系

- LangGraph **依赖 LangChain** 的 `langchain-core`（消息类型、Runnables、Tools）。
- 在图节点里你可以自由使用 LangChain 的 chain / retriever / agent。
- 反过来，图本身也可以被当成一个 `Runnable`，从而能 `.stream()` / `.batch()`。

### 2.5 持久化、检查点、时间旅行

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(DB_URL) as cp:
    graph = builder.compile(checkpointer=cp)
```

每次图执行的中间状态都被存到 `cp`。给定 `thread_id`，你可以：

- `get_state(config)`：读取当前状态。
- `get_state_history(config)`：读取所有历史快照。
- `update_state(config, values)`：手动修改状态（"时间旅行"）。
- 把图从某个历史快照"重新跑"用于复现。

> 这是 RAG / Agent 调试、A/B、回放、审计的工程基石。

### 2.6 多人协作与人机协同

LangGraph 在节点前调用 `interrupt()` 可以**暂停**图执行，把当前状态冻结。客户端再以 `Command(resume=...)` 注入继续。

典型模式：

```python
from langgraph.types import interrupt, Command

def ask_human(state):
    answer = interrupt({"prompt": "是否继续？"})
    return {"approved": answer == "yes"}

builder.add_node("ask_human", ask_human)
```

调用：

```python
cfg = {"configurable": {"thread_id": "user-42"}}

# 第一次 invoke 会在 interrupt 处暂停，返回 __interrupt__
result = graph.invoke({"x": 1}, config=cfg)
print(result["__interrupt__"])

# 用户确认后，再喂回去
graph.invoke(Command(resume="yes"), config=cfg)
```

再配合 `Stream` / `WebSocket`，可以做"边跑边确认"的真实审批流。

### 2.7 多 Agent 编排模式

- **Supervisor**：一个 supervisor 节点把任务分给 worker 节点。
- **Subgraph**：把每个 worker 单独建模为子图，单独编译并 invoke。
- **Router**：根据条件边把消息路由到不同专家 Agent。
- **Handoff**：使用 `Command(goto=...)` 把当前线程"交接"给另一角色。

详见 `examples/代码片段.md` 中示例 3.4。

---

## 3. 代码示例

> 以下示例均可独立运行；展示了组合 / 检查点 / Interrupt / 多 Agent 的核心思路。

### 3.1 LangChain：LCEL + RAG

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

docs = WebBaseLoader("https://example.com/article").load()
splits = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)
vs = FAISS.from_documents(splits, OpenAIEmbeddings())
retriever = vs.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "只根据上下文回答，不要编造。\n\n<ctx>{context}</ctx>"),
    ("human", "{question}")
])
model = ChatAnthropic(model="claude-sonnet-4-5")
parser = StrOutputParser()

def join_docs(docs): return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | RunnableLambda(join_docs),
     "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

print(chain.invoke("文中最主要的观点是什么？"))
```

要点：
- `retriever | RunnableLambda(join_docs)` 把检索结果压平成字符串。
- `RunnablePassthrough()` 透传 `question`。
- 整条管道仍然是 `Runnable`，所以能直接 `.stream()`。

### 3.2 LangChain：Tools + ReAct Agent

```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_anthropic import ChatAnthropic

@tool
def get_weather(city: str) -> str:
    """获取某城市当前天气"""
    return f"{city}: 27°C, 晴"

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(ChatAnthropic(model="claude-sonnet-4-5"),
                           tools=[get_weather], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[get_weather], verbose=True)

print(executor.invoke({"input": "杭州今天多少度？"})["output"])
```

### 3.3 LangGraph：可中断 + 检查点的人机协同

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    draft: str
    approved: bool

def draft_email(s: State) -> dict:
    return {"draft": f"亲爱的用户，您好。这是自动生成的草稿。"}

def review(s: State) -> dict:
    decision = interrupt({"ask": "approve?", "draft": s["draft"]})
    return {"approved": decision == "yes"}

def send(s: State) -> dict:
    # 实际调用发送服务
    return {}

builder = StateGraph(State)
builder.add_node("draft", draft_email)
builder.add_node("review", review)
builder.add_node("send", send)
builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_conditional_edges(
    "review",
    lambda s: "send" if s["approved"] else END,
    {"send": "send", END: END},
)
builder.add_edge("send", END)

graph = builder.compile(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}

# 1) 跑到 review 后暂停
print(graph.invoke({}, config=cfg))
# 2) 模拟人工批准
print(graph.invoke(Command(resume="yes"), config=cfg))
```

### 3.4 LangGraph：Supervisor 多 Agent

```python
class State(TypedDict):
    question: str
    draft_a: str
    draft_b: str
    final: str

def researcher(state): return {"draft_a": "事实 A1, A2"}
def writer(state): return {"draft_b": "草稿 W 基于 A1, A2"}
def supervisor(state) -> str:
    return "finalize" if state["draft_a"] and state["draft_b"] else "researcher"

def finalize(state):
    return {"final": f"综合：{state['draft_a']} + {state['draft_b']}"}

builder = StateGraph(State)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("finalize", finalize)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_conditional_edges("writer", supervisor,
    {"finalize": "finalize", "researcher": "researcher"})
builder.add_edge("finalize", END)
```

更复杂的生产实现会把每个角色单独编译为子图，并用 `Command(goto=...)` 做"handoff"。

---

## 4. 选型决策树

```
开始
 ├─ 只是单次"提示 + 模型"？ ──→ 直接 SDK
 ├─ 需要"多步 + 工具 + 记忆"？
 │    ├─ 简单的循环式 / Function Calling？ ──→ LangChain Agent
 │    └─ 需要中断 / 状态恢复 / 多角色？  ──→ LangGraph
 └─ 只是检索问答？ ──→ LangChain LCEL（RAG）
```

经验法则：

- **不要重**：如果只跑一次模型调用，启动 LangChain 是过度工程。
- **默认从 LCEL 起手**：能 stream / async / batch 已是 80% 场景。
- **状态复杂再上 LangGraph**：一旦需要审批、调试、断点续跑、多 Agent，立即换。

---

## 5. 性能、调试、可观测性

| 维度 | LangChain | LangGraph |
| --- | --- | --- |
| 流式 | `chain.stream` / `astream_events` | `graph.stream({"node_state": ...})` |
| 批 | `chain.batch` | `graph.batch` |
| 异步 | 全 `a` 前缀 | 同上 |
| 观测 | `LangSmith` / `Callbacks` | `LangSmith` / `astream_events` / `get_state_history` |
| 缓存 | `set_llm_cache` | 节点级 `@lru_cache` 或自实现 |
| 重试 | `with_retry` / `with_fallbacks` | 在节点内 try/except 或用 retry policy |

调试技巧：

- 给每个 `Runnable` 起名：`chain.with_config({"run_name": "..."})`。
- 单独跑子步骤：`prompt.invoke({})`、`model.invoke(prompt.invoke({}))`。
- LangGraph：`graph.get_graph().draw_mermaid_png()` 把图渲染出来。

---

## 6. 局限与常见反模式

- **抽象税**：每包一层 Runnable 就多一层间接，理解栈会变深。
- **过度组合**：把所有能 `|` 都 `|` 上，让代码像管道工图纸一样不可读。
- **状态膨胀**：LangGraph 的 state 字段一多，调试时 diff 一片红，需要自觉划分节点职责。
- **Tool 描述含糊**：让 LLM 反复选错工具。描述要单职责，给正反例。
- **把 Prompt 写在 Node 里**：放在 LangChain 模板里更易于 A/B。
- **状态和 IO 混放**：容易把外部副作用与 state 写入逻辑揉在一起，建议 Node 只做"输入 → 状态更新"。

---

## 7. 概念对照表

| 概念 | LangChain | LangGraph |
| --- | --- | --- |
| 流程模型 | 线性 / LCEL 链 | 有向图 (DAG / 含环) |
| 状态 | 无显式（变量在闭包） | 显式 `State` + reducer |
| 持久化 | 外部自行实现 | 内置 `checkpointer` |
| 中断 | 不直接支持 | `interrupt()` |
| 时间旅行 | 不直接支持 | `get_state_history()` + `update_state()` |
| 多 Actor | 通过 agent 包 | 节点即 Actor，子图 |
| 可观察单元 | Runnable | Node |
| 适用阶段 | 原型 → 中等复杂度 | 中等到长期运行的复杂 Agent |

---

## 8. 参考资源

- LangChain 官方文档：[https://python.langchain.com](https://python.langchain.com)
- LangGraph 官方文档：[https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- LangSmith：[https://docs.smith.langchain.com](https://docs.smith.langchain.com)
- *LangChain for LLM Application Development*（DeepLearning.AI 短期课）
- *AI Agents in LangGraph*（DeepLearning.AI 短期课）
- 论文/文章：
  - LangChain 团队 "LangGraph: Multi-Agent Orchestration"
  - Google "Pregel: A System for Large-Scale Graph Processing"
  - Anthropic "Building Effective Agents"

---

> 与正文一致性：本文档为 `docs/06-...`，是 README 中"AI Skill"系列笔记的延伸专题。
