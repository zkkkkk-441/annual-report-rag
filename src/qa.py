"""模块 3：生成回答（DeepSeek，OpenAI 兼容接口）+ 加分项 Agent（计算工具调用）。

减少幻觉的四道防线：
1. System prompt 规定只能依据给定原文回答，原文没有就答"年报未披露"；
2. 每段原文带编号和"文件+页码"，要求模型在 sources 里引用编号；
3. 用 JSON 模式输出 answer / sources / confidence，便于程序校验；
4. 程序侧校验：模型引用的编号必须是真实检索到的段落，否则丢弃，引用全被丢弃时置信度降为 low。

计算类问题（同比增长率等）不让模型心算，而是通过 tool calling 调用 Python 计算。

运行：python -m src.qa "比亚迪2025年营业收入同比增长多少？"
"""
import json
import sys

from openai import OpenAI

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from src.index import retrieve

NOT_DISCLOSED = "年报未披露"

SYSTEM_PROMPT = f"""你是上市公司年报分析助手。规则：
1. 只能依据用户给出的【原文段落】回答，禁止使用你自己的知识或推测。
2. 如果原文段落中没有能回答问题的信息，answer 必须是"{NOT_DISCLOSED}"，sources 为空数组，confidence 为 "low"。
3. 数字必须与原文一致，并写明单位（原文是"千元""百万元"就照写，不要自行换算单位）。
4. 涉及增长率、占比等计算时，必须调用 calculate 工具，不要心算。
5. 最终只输出一个 JSON 对象：
{{"answer": "简洁的回答", "sources": [段落编号, ...], "confidence": "high" | "medium" | "low"}}
sources 填你实际依据的段落编号（整数）。"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "计算增长率或比例。growth: (current-previous)/previous；ratio: part/total。返回百分比，保留两位小数。",
        "parameters": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["growth", "ratio"]},
                "a": {"type": "number", "description": "growth 时为本期值 current；ratio 时为部分值 part"},
                "b": {"type": "number", "description": "growth 时为上期值 previous；ratio 时为总值 total"},
            },
            "required": ["op", "a", "b"],
        },
    },
}]


def calculate(op: str, a: float, b: float) -> str:
    if b == 0:
        return "错误：除数为 0"
    value = (a - b) / b if op == "growth" else a / b
    return f"{value * 100:.2f}%"


def _client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise SystemExit("未找到 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入你的 Key")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _format_context(passages: list[dict]) -> str:
    return "\n\n".join(f"[{i}] 来源：{p['doc']} 第{p['page']}页\n{p['text']}" for i, p in enumerate(passages, 1))


def answer(question: str, chunk_size: int = DEFAULT_CHUNK_SIZE, top_k: int = DEFAULT_TOP_K,
           use_tools: bool = True) -> dict:
    passages = retrieve(question, chunk_size, top_k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【原文段落】\n{_format_context(passages)}\n\n【问题】{question}"},
    ]
    client, tool_log = _client(), []

    for _ in range(4):  # 最多 3 轮工具调用 + 1 轮最终回答
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL, messages=messages, temperature=0,
            response_format={"type": "json_object"},
            **({"tools": TOOLS} if use_tools else {}),
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = calculate(**args)
            tool_log.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    try:
        out = json.loads(msg.content or "{}")
    except json.JSONDecodeError:
        out = {"answer": msg.content, "sources": [], "confidence": "low"}

    # 程序侧校验引用：只保留真实存在的段落编号，换算成 文件 + 页码
    cited = []
    for s in out.get("sources") or []:
        try:
            idx = int(s)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(passages):
            p = passages[idx - 1]
            src = {"doc": p["doc"], "page": p["page"]}
            if src not in cited:
                cited.append(src)
    answer_text = str(out.get("answer", "")).strip() or NOT_DISCLOSED
    confidence = out.get("confidence", "low")
    if answer_text != NOT_DISCLOSED and not cited:
        confidence = "low"  # 有答案却给不出有效出处，视为不可信

    return {
        "question": question,
        "answer": answer_text,
        "sources": cited,
        "confidence": confidence,
        "tool_calls": tool_log,
        "retrieved": [{"doc": p["doc"], "page": p["page"], "score": p["score"]} for p in passages],
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    q = sys.argv[1] if len(sys.argv) > 1 else "宁德时代2025年营业收入是多少？"
    print(json.dumps(answer(q), ensure_ascii=False, indent=2))
