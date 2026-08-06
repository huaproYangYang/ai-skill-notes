# 05. 实践与提示工程

## 1. 一个稳定 Prompt 的结构

```text
# Role
你是 {{role}}，{{background}}。

# Goals
- 完成 {{goal1}}
- 同时 {{goal2}}

# Constraints
- 必须用中文
- 字数 <= 200
- 不要 {{forbidden}}

# Output Format
- JSON: { "answer": str, "citations": [str] }

# Examples
[示例 1]
[示例 2]

# Steps
1. 先 ...
2. 再 ...
3. 最后 ...

# Now do it
Input: {{user_input}}
```

## 2. 常见错误与对策

| 症状 | 原因 | 对策 |
| --- | --- | --- |
| 答非所问 | 指令含糊 | 给出任务边界 + 例子 |
| 不遵循格式 | 输出无 schema | JSON / 正则约束 |
| 答案总是"不知道" | 安全过度 | 加 context + 反例 |
| 长上下文失忆 | 注意力分散 | 摘要 + 复用关键句 |
| 幻觉数字 | 缺乏工具 | 接 Python / Wolfram |
| 工具总用错 | 描述模糊 | 给参数 + 错误示例 |

## 3. 自评与多解投票

```
让模型给出 3 个候选 + 自评理由 + 最终选择。
```

能显著提高在主观任务上的稳定性。
