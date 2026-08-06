# AI Skill 详细介绍

> 一份系统梳理"AI 技能（AI Skill）"概念的笔记。涵盖定义、分类、能力图谱、Agent / 工具使用、提示工程、评估与提升路径，并附示例代码与参考资源。

## 目录

- [0. 为什么关注 AI Skill](#0-为什么关注-ai-skill)
- [1. 什么是 AI Skill](#1-什么是-ai-skill)
  - [1.1 定义](#11-定义)
  - [1.2 与人类技能的对比](#12-与人类技能的对比)
  - [1.3 一个 Skill 的最小构成](#13-一个-skill-的最小构成)
- [2. AI Skill 的分类体系](#2-ai-skill-的分类体系)
  - [2.1 按认知层级](#21-按认知层级)
  - [2.2 按模态](#22-按模态)
  - [2.3 按任务类型](#23-按任务类型)
  - [2.4 按专业领域](#24-按专业领域)
- [3. 核心技能详解](#3-核心技能详解)
  - [3.1 语言技能 (NLP)](#31-语言技能-nlp)
  - [3.2 推理与逻辑](#32-推理与逻辑)
  - [3.3 知识与记忆](#33-知识与记忆)
  - [3.4 工具使用与 Agent](#34-工具使用与-agent)
  - [3.5 多模态技能](#35-多模态技能)
  - [3.6 代码与编程](#36-代码与编程)
  - [3.7 数学与定量分析](#37-数学与定量分析)
- [4. Agent 技能专题](#4-agent-技能专题)
  - [4.1 ReAct 循环](#41-react-循环)
  - [4.2 Function Calling / Tool Use](#42-function-calling--tool-use)
  - [4.3 规划与分解](#43-规划与分解)
  - [4.4 记忆系统](#44-记忆系统)
  - [4.5 自反思与纠错](#45-自反思与纠错)
- [5. 提示工程 (Prompt Engineering)](#5-提示工程-prompt-engineering)
  - [5.1 基础技巧](#51-基础技巧)
  - [5.2 进阶模式](#52-进阶模式)
  - [5.3 系统提示设计](#53-系统提示设计)
- [6. 技能评估与基准](#6-技能评估与基准)
- [7. 技能提升路径](#7-技能提升路径)
- [8. 局限与风险](#8-局限与风险)
- [9. 未来趋势](#9-未来趋势)
- [深入专题：LangChain 与 LangGraph 原理](docs/06-langchain-langgraph.md)
- [深入专题：RAG 与 Agent 串联 Demo](docs/07-rag.md)
- [10. 参考资源](#10-参考资源)
- [附录 A：最小 Agent 示例代码](#附录-a最小-agent-示例代码)
- [附录 B：术语表](#附录-b术语表)

---

## 0. 为什么关注 AI Skill

随着大模型（LLM / VLM / MM-LLM）和智能体（Agent）技术成熟，"AI 能做什么"这个问题的答案已经从单一指标扩展为一张**能力图谱**。理解这张图谱——每一项技能如何分类、如何评估、如何扩展——是构建可靠 AI 应用、研究能力边界、设计人机协作流程的起点。

本笔记从工程与研究两侧视角，给出 AI Skill 的结构化介绍。

---

## 1. 什么是 AI Skill

### 1.1 定义

> **AI Skill**：AI 系统在特定输入下完成特定任务、并能在分布变化下保持稳定表现的能力单元。

它具备四个特征：

1. **可观察**：能通过输入-输出或行为轨迹打分、复现。
2. **可泛化**：在未见过的同分布数据上仍能完成。
3. **可拆解**：可以独立测试、组合调用。
4. **可度量**：有可比的指标（准确率、成功率、回合数等）。

### 1.2 与人类技能的对比

| 维度 | 人类技能 | AI Skill |
| --- | --- | --- |
| 学习方式 | 持续、终身、迁移学习 | 预训练 + 微调 + 上下文 |
| 上下文利用 | 工作记忆 + 经验 | 上下文窗口 + 检索增强 |
| 可扩展性 | 受时间精力限制 | 同模型可复制上千实例 |
| 一致性 | 因人而异、有疲劳 | 同温度下高度稳定 |
| 可解释性 | 中等（可询问） | 弱（黑盒或事后归因） |
| 失败模式 | 偏科、偶发失误 | 幻觉、长上下文遗忘、工具错用 |

### 1.3 一个 Skill 的最小构成

```
Skill = (Capability) × (Knowledge) × (Tool) × (Evaluator)
       能力         知识       工具     评估
```

- **Capability**：模型本身具有的能力（权重里的归纳偏置）。
- **Knowledge**：模型内化的事实与世界知识。
- **Tool**：技能调用时的外部工具（搜索、代码执行、SQL…）。
- **Evaluator**：判断技能发挥得如何的反馈信号（自身打分、规则、人工、奖励模型）。

---

## 2. AI Skill 的分类体系

### 2.1 按认知层级

按 Anderson / Bloom 改编的层级：

1. **记忆 (Recall)**：复述事实、术语、格式。
2. **理解 (Comprehension)**：解释、转述、类比。
3. **应用 (Application)**：在给定场景下使用学到的步骤。
4. **分析 (Analysis)**：分解问题、识别关系、定位错误。
5. **评价 (Evaluation)**：对方案/产物做评判、对比、批评。
6. **创造 (Creation)**：综合多源信息产生新方案、新结构。

> 现代大模型在 1–4 层已经非常强，第 5–6 层的稳定表现仍依赖外部评审与工具。

### 2.2 按模态

| 模态 | 输入 | 输出 | 代表能力 |
| --- | --- | --- | --- |
| 文本 | 文本 | 文本 | 写作、问答、总结、对话 |
| 视觉 | 图像 / 视频 | 文本 | OCR、视觉问答、定位、分割描述 |
| 语音 | 音频 | 文本 / 音频 | 语音识别、合成、翻译 |
| 多模态 | 混合 | 混合 | 图文问答、文档理解、视频分析 |

### 2.3 按任务类型

- **生成 (Generation)**：写文、画图、作曲、写代码。
- **理解 (Comprehension)**：分类、抽取、问答、摘要。
- **推理 (Reasoning)**：常识、演绎、数学、规划。
- **决策 (Decision-making)**：多步策略、资源分配。
- **交互 (Interaction)**：对话、协作、协商、教学。
- **工具使用 (Tool Use)**：调用 API、读写文件、执行代码。

### 2.4 按专业领域

- 编程 / 软件工程
- 数学 / 定量分析
- 法律 / 合规
- 医疗 / 生命科学
- 教育 / 辅导
- 金融 / 风险
- 设计 / 内容创作
- 客服 / 运营

> 同一基模型在 Prompt 模板、检索增强、工具、领域微调后，会呈现"领域级 Skill"。

---

## 3. 核心技能详解

### 3.1 语言技能 (NLP)

底层能力：

- **词法 / 句法**：分词、词性、依存、句法。
- **语义**：词义消歧、语义角色、蕴含 (NLI)。
- **语用**：意图、情感、立场、潜台词。
- **篇章**：摘要、指代、连贯、修辞。

可观测子任务示例：

| 子任务 | 典型输入 | 期望输出 | 评估 |
| --- | --- | --- | --- |
| 文本分类 | 一段评论 | 标签 (正/负/中) | Accuracy / F1 |
| 命名实体识别 | 句子 | (实体, 类型) 列表 | F1 / 精确匹配 |
| 抽取式问答 | 段落 + 问题 | 答案 span | EM / F1 |
| 摘要 | 文档 | 摘要 | ROUGE / BERTScore / LLM-judge |
| 翻译 | 双语对 | 译文 | BLEU / COMET / 人工 |
| 改写 | 原文 | 改写后 | 人工 / 相似度 |

### 3.2 推理与逻辑

推理类型：

- **演绎 (Deduction)**：规则 → 结论（"若 A→B 且 A，则 B"）。
- **归纳 (Induction)**：样本 → 规律（"100 只天鹅都是白的" → "天鹅是白的"）。
- **溯因 (Abduction)**：事实 → 最佳解释。
- **类比 (Analogy)**：A:B :: C:?
- **多步 / 链式 (Chain-of-Thought)**：把推理显式展开。

实践要点：

- 显式写"思考步骤"通常比隐式跳跃更稳。
- 让模型把答案包在 `\\boxed{}` 或结构化标签里，便于校验。
- 长链推理易算错，可用"自检 / 自评"循环提高鲁棒性。

### 3.3 知识与记忆

- **参数化知识**：权重里学到的世界事实。
- **上下文知识**：本轮提示窗里给到的内容。
- **外部知识**：检索增强生成 (RAG) 召回的文档。
- **长期记忆**：跨会话存储用户偏好、已完成任务。

常见模式：

```
Query
  ↓
Retriever (BM25 / Dense / Hybrid)
  ↓
Top-k 文档块
  ↓
Prompt = System + History + Query + 文档
  ↓
LLM → 答案（带引用）
```

### 3.4 工具使用与 Agent

**Tool Use = LLM 选择 + 参数化 + 调用外部能力**。

最小循环：

1. 收到用户目标。
2. 模型决定是否需要工具、需要哪一个、参数是什么。
3. 执行工具，得到观察 (observation)。
4. 把观察放回上下文中，再让模型决定下一步。
5. 直到给出最终答案。

工具例子：web_search、code_exec、sql_query、file_read、calendar、http_get、send_email……

### 3.5 多模态技能

- **VQA (Visual QA)**：对图提问。
- **文档理解**：表格 / 票据 / 扫描件的结构化抽取。
- **图像定位 / Grounding**：根据描述框出区域。
- **视频理解**：动作识别、视频问答、长视频摘要。
- **跨模态生成**：图生文、文生图、图生图、文生视频。

### 3.6 代码与编程

子技能：

- 阅读与解释代码
- 补全与生成代码
- 重构与优化
- 单元测试生成
- Bug 定位与修复
- Code Review
- 跨语言翻译 / 移植

可靠做法：

- **执行反馈**：跑测试、用 linter、TypeScript compiler。
- **小步提交**：每步可独立验证。
- **明确约束**：指定语言、版本、风格、依赖。
- **可观测**：保留日志、报错堆栈。

### 3.7 数学与定量分析

- 算术 / 代数 / 几何。
- 概率与统计。
- 数值方法与优化。
- 不确定下的决策。

> 直接让模型做大数运算常常不稳定。应配套工具（Python REPL / SymPy / Wolfram）执行。

---

## 4. Agent 技能专题

Agent = 在环境里持续行动以达成目标的系统。

### 4.1 ReAct 循环

```
Thought → Action → Observation → Thought → ... → Final Answer
```

每一轮：

1. **Thought**：内部推理（下一步该做什么）。
2. **Action**：选工具 + 参数。
3. **Observation**：工具返回。

### 4.2 Function Calling / Tool Use

模型以结构化 JSON 声明调用：

```json
{
  "name": "get_weather",
  "arguments": { "city": "Hangzhou", "unit": "celsius" }
}
```

宿主解析 → 执行 → 把结果作为 Observation 喂回。

关键设计：

- 工具描述要**清晰、单一、参数化**。
- 错误处理：失败要可重试或回退。
- 安全性：白名单、参数校验、防 prompt 注入。

### 4.3 规划与分解

- **Plan-and-Execute**：先产完整计划，再逐步执行。
- **Tree of Thoughts**：多分支并行评估。
- **Reflexion / Self-Refine**：执行→反思→改进。

### 4.4 记忆系统

- **Short-term**：本会话上下文。
- **Long-term**：向量库 / KV。
- **Episodic**：事件化（"昨天发生过……"）。
- **Procedural**：程序化记忆（"按这个模板做"）。

### 4.5 自反思与纠错

模型在被要求"先列出你的推理，再给最终答案，并打分"时通常更稳。常见 prompt：

```
请先一步步推理，再给最终答案。
请对自己的答案做 3 点评判，至少指出 1 个可能错处。
```

---

> 想要深入工程实现？阅读 [深入专题：LangChain 与 LangGraph 原理](docs/06-langchain-langgraph.md) — 覆盖 Runnable 协议、LCEL、StateGraph、检查点 / Interrupt / 多 Agent 编排。

## 5. 提示工程 (Prompt Engineering)

### 5.1 基础技巧

- **明确角色与目标**：`你是一名资深 SRE，` 比 "请帮忙" 强很多。
- **给出输入-输出示例 (Few-shot)**：用 1–5 个示例对齐格式。
- **结构化输出**：要求 JSON / 表格 / Markdown，便于后处理。
- **限制条件**：字数、不要的项目、必须包含的字段。
- **分步思考**：要求"先列步骤再回答"。

### 5.2 进阶模式

- **Chain-of-Thought (CoT)**。
- **Self-Consistency**：采样多解，投票。
- **ReAct**。
- **Least-to-Most**：把问题拆成子问题再逐级求解。
- **Generated Knowledge**：先让模型写背景知识，再回答。

### 5.3 系统提示设计

系统提示应包含：

1. **身份与边界**：能做什么、不能做什么。
2. **行为风格**：语气、长度、格式。
3. **可用工具**：明列工具与何时调用。
4. **失败处理**：无法回答时怎么做。
5. **护栏**：拒绝特定请求的规则。

---

## 6. 技能评估与基准

### 通用基准

- **MMLU / MMLU-Pro**：多学科知识。
- **GSM8K / MATH**：数学推理。
- **HumanEval / MBPP / SWE-bench**：编程。
- **TruthfulQA / HaluEval**：事实与幻觉。
- **BBH / ARC-AGI**：广度推理。
- **MTEB**：Embedding。
- **MMMU / MMBench**：多模态。

### Agent 基准

- **GAIA**：通用助手。
- **WebArena / VisualWebArena**：网页任务。
- **SWE-bench / SWE-bench Verified**：软件工程。
- **τ-bench / ToolBench**：工具使用。

### 评估方法

- **自动指标**：精确匹配、F1、BLEU、ROUGE、Pass@k。
- **模型裁判 (LLM-as-judge)**：用强模型对答案打分。
- **A/B 偏好对比**：人评 vs. 自动。
- **红队与对抗测试**：安全与边界。

---

## 7. 技能提升路径

### 训练阶段

1. **更大 / 更好的基模型**：能力地板由基座决定。
2. **继续预训练 (Continual Pre-training)**：补领域语料。
3. **指令微调 (SFT)**：让模型"听话"。
4. **偏好对齐 (RLHF / DPO / RLAIF)**：让模型"合用"。
5. **工具与 Agent 训练**：ReAct、Function Calling 数据。

### 部署阶段

- **提示工程**：用更稳的 prompt 模板。
- **检索增强 (RAG)**：补足事实。
- **工具编排**：把外部能力模块化。
- **记忆与个性化**：跨会话偏好。
- **自反思与多轮投票**：提高稳定性。
- **领域适配**：微调小模型 + 路由。

### 评估驱动的迭代

```
提出假设 → 改 prompt / 模型 / 数据 → 在评测集跑分 → 误差分析 → 循环
```

---

## 8. 局限与风险

- **幻觉**：模型自信地说错。
- **长上下文衰减**：越靠后越易被忽视。
- **工具误用**：参数错、调用错对象。
- **prompt 注入**：让 Agent 执行恶意操作。
- **偏见与公平**：训练数据带有的偏差。
- **过度信任**：把模型当事实源。

缓解手段：人机协同、检索引用、规则校验、对抗测试、限制高风险动作。

---

## 9. 未来趋势

- **更长上下文 + 更好记忆**：跨会话能力。
- **原生多模态**：单模型统一处理文本/图/音/视频。
- **可验证推理 (verifier RL)**：以结果正确性驱动训练。
- **Agent 操作系统**：工具 / 浏览器 / 计算机 / IDE 标准化。
- **个人化智能体**：用户专属记忆、偏好、流程。
- **本地化 + 端侧**：隐私、低延迟。
- **协作式多 Agent**：专家化分工、互相校验。

---

## 10. 参考资源

### 论文

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Few-Shot Learners (Brown et al., 2020)
- Chain-of-Thought Prompting (Wei et al., 2022)
- ReAct (Yao et al., 2022)
- Toolformer (Schick et al., 2023)
- Self-Refine (Madaan et al., 2023)
- Reflexion (Shinn et al., 2023)
- Tree of Thoughts (Yao et al., 2023)

### 站点

- Hugging Face Leaderboards
- Papers With Code
- Anthropic / OpenAI / Google DeepMind 官方博客
- LangChain / LlamaIndex / Anthropic MCP 文档

### 开源

- Hugging Face Transformers / Datasets / TRL
- LangChain / LlamaIndex
- AutoGen / CrewAI / LangGraph
- vLLM / TGI / SGLang
- Inspect AI / lm-evaluation-harness

---

## 附录 A：最小 Agent 示例代码

伪代码示意（不是直接可运行版本）：

```python
def agent_loop(llm, tools, user_goal, max_steps=10):
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user",   "content": user_goal}]

    for step in range(max_steps):
        reply = llm.chat(history, tools=tools)
        history.append(reply)

        if reply.has_tool_call():
            for call in reply.tool_calls:
                try:
                    obs = tools[call.name].run(**call.args)
                except Exception as e:
                    obs = f"ERROR: {e}"
                history.append({
                    "role": "tool",
                    "name": call.name,
                    "content": str(obs),
                })
        else:
            return reply.text

    return "MAX_STEPS_REACHED"
```

更具体的 Claude / OpenAI SDK 实现见 `examples/代码片段.md`。

---

## 附录 B：术语表

| 术语 | 释义 |
| --- | --- |
| LLM | 大语言模型 |
| VLM | 视觉语言模型 |
| Agent | 在环境中自主调用工具、达成目标的系统 |
| Tool Use | 模型调用外部工具的能力 |
| RAG | 检索增强生成 |
| CoT | 思维链 |
| ReAct | Reason + Act 循环 |
| Function Calling | 模型产出可解析的工具调用 JSON |
| RLHF | 人类反馈强化学习 |
| DPO | 直接偏好优化 |
| MCP | Model Context Protocol，工具与数据源协议 |
| Embedding | 把文本映射为向量的表示 |
| Vector DB | 用于相似度检索的向量数据库 |
| Hallucination | 模型自信地编造 |
| Prompt Injection | 通过输入劫持模型行为的攻击 |

---

> 许可：MIT。 维护：本仓库欢迎 PR 增补/勘误。
