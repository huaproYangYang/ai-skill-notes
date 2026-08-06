# 示例片段 (Examples)

## 1. Anthropic API: Tool Use 最小示例

```python
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "获取某城市当前天气",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}]

def get_weather(city: str, unit: str = "celsius") -> dict:
    # 真实实现里接第三方 API
    return {"city": city, "unit": unit, "temp": 27, "cond": "sunny"}

messages = [{"role": "user", "content": "杭州今天多少度？"}]

resp = client.messages.create(
    model="claude-sonnet",
    max_tokens=256,
    tools=tools,
    messages=messages,
)

# 简化：实际跑要处理 tool_use block → 执行 → tool_result block
print(resp)
```

## 2. OpenAI-Compatible: 结构化输出

```python
from openai import OpenAI
import json, pydantic

class Plan(pydantic.BaseModel):
    steps: list[str]
    risks: list[str]

client = OpenAI()
resp = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一名产品经理，给出上线计划。"},
        {"role": "user", "content": "我们要上线一个推荐系统。"},
    ],
    response_format=Plan,
)
print(resp.choices[0].message.parsed)
```

## 3. 简易 ReAct 伪实现

```python
def react_step(llm, history):
    prompt = history + ["\nThought:", "Action:", "Observation:"]
    text = llm(prompt)
    thought, action, *_ = text.split("Action:")
    return {"thought": thought, "action": action.strip()}
```

> 真实工程实现请直接使用成熟框架：LangGraph、AutoGen、CrewAI。
