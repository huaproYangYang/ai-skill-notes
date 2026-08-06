# 04. Agent 与工具调用

## 1. 为什么需要 Agent

单一 prompt 解决 70% 的事；剩下 30% 需要：

- 多步决策
- 工具副作用
- 状态记忆
- 自我纠错

这就是 Agent 模式的目标。

## 2. 经典模式

### 2.1 ReAct

```
循环：
  t = think(state)
  a = choose_tool(t)
  o = execute(a)
  state = append(state, t, a, o)
直到满足终态。
```

### 2.2 Plan-and-Execute

```
plan = plan(goal)
for step in plan:
  out = execute(step)
  if out.fail: plan = replan(plan, out)
```

### 2.3 Reflexion

```
attempt = solve(goal)
critique = reflect(attempt)
attempt2 = solve(goal, critique)
```

## 3. 工具设计三原则

1. **单一职责**：一个工具做一件事。
2. **幂等**：可重入不破坏状态。
3. **可观测**：返回 JSON 字符串，包含 ok / data / error。

## 4. 安全护栏

- 工具白名单。
- 高风险动作（rm, drop, 发送邮件）必须二次确认。
- 输入消毒 (anti prompt-injection)。
- 输出审计日志。
- 速率限制与重试上限。
