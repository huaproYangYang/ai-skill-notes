# 03. AI 能力地图 (Capability Map)

下面是常见 AI Skill 的能力地图 (示意，非穷举)。

| Skill | 模态 | 类型 | 代表基准 | 典型工具 |
| --- | --- | --- | --- | --- |
| 开放问答 | text | 理解 | NaturalQuestions | web_search |
| 摘要 | text | 生成 | CNN/DM, XSum | — |
| 翻译 | text | 生成 | WMT, FLORES | dictionary |
| 数学解题 | text | 推理 | GSM8K, MATH | python |
| 代码生成 | text | 推理 + 工具 | HumanEval, SWE-bench | shell, fs |
| VQA | vision | 理解 | VQA-v2, OK-VQA | OCR |
| 文档抽取 | vision | 抽取 | DocVQA | OCR + parser |
| 视频理解 | video | 理解 | EgoSchema, LongVideoBench | ffmpeg |
| ASR | audio | 理解 | LibriSpeech, GigaSpeech | — |
| TTS | audio | 生成 | LJSpeech | vocoder |
| 多模态 Agent | multi | 工具 | GAIA, WebArena | 浏览器 |
| 长上下文阅读 | text | 理解 | LongBench, RULER | RAG |
| 推理规划 | text | 决策 | PlanBench | planner |

> 同一行可能在不同模型上从 0% 跨到 90%，这正是 AI Skill 时代
> 让"机会"和"风险"都同步放大的原因。
