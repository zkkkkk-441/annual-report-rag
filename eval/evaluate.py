"""模块 4：评估 + 对比实验。

指标：
- 检索命中率 Hit@k：标准答案所在页（gold_pages）是否出现在 top-k 检索结果的页码中（只统计有答案的题）
- 回答准确率：答案中是否包含每组关键要素（keywords 是"组"的列表，每组任一写法出现即算命中，所有组都命中才算对）。
  "年报未披露"题要求回答包含"未披露"。用规则判分而不是让大模型当裁判，结果可复现、可人工复核。
- 引用准确率：回答引用的页码中是否包含 gold page（只统计有答案且答对的题）

实验设置：chunk 300 / 800 × top-k 3 / 5，共 4 组。

运行：
    python -m eval.evaluate --retrieval-only   # 只测检索，不需要 API Key，不花钱
    python -m eval.evaluate                    # 完整评估（会调用 DeepSeek，约 80 次请求）
结果写入 eval/results/。
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src.config import CHUNK_SIZES

EVAL_DIR = Path(__file__).resolve().parent
RESULT_DIR = EVAL_DIR / "results"
TOP_KS = [3, 5]


def _norm(s: str) -> str:
    return "".join(s.split()).replace(",", "").replace("，", "")


def is_correct(answer: str, keywords: list[list[str]]) -> bool:
    a = _norm(answer)
    return all(any(_norm(k) in a for k in group) for group in keywords)


def run(retrieval_only: bool):
    from src.index import retrieve  # 延迟导入：加载 embedding 模型较慢
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    answerable = [q for q in questions if q["gold_pages"]]
    RESULT_DIR.mkdir(exist_ok=True)
    summary = []

    for chunk_size in CHUNK_SIZES:
        for top_k in TOP_KS:
            setting = f"chunk{chunk_size}_top{top_k}"
            rows, hits = [], 0
            for q in questions:
                if retrieval_only:
                    retrieved = [{"doc": r["doc"], "page": r["page"], "score": r["score"]}
                                 for r in retrieve(q["question"], chunk_size, top_k)]
                    row = {"id": q["id"], "question": q["question"], "retrieved": retrieved}
                else:
                    from src.qa import answer
                    for attempt in range(3):  # 网络偶发错误重试
                        try:
                            row = answer(q["question"], chunk_size, top_k)
                            break
                        except Exception as e:  # noqa: BLE001
                            if attempt == 2:
                                raise
                            print(f"  重试 {q['id']}：{e}")
                            time.sleep(3)
                    row["id"] = q["id"]
                    row["correct"] = is_correct(row["answer"], q["keywords"])
                    cited = {s["page"] for s in row["sources"]}
                    row["cite_ok"] = bool(cited & set(q["gold_pages"])) if q["gold_pages"] else None
                retrieved_pages = {r["page"] for r in row["retrieved"]}
                row["hit"] = bool(retrieved_pages & set(q["gold_pages"])) if q["gold_pages"] else None
                hits += bool(row["hit"])
                rows.append(row)

            result = {"setting": setting, "chunk_size": chunk_size, "top_k": top_k,
                      "hit_rate": hits / len(answerable)}
            if not retrieval_only:
                correct_answerable = [r for r in rows if r["correct"] and r["cite_ok"] is not None]
                result["accuracy"] = sum(r["correct"] for r in rows) / len(rows)
                result["cite_accuracy"] = (sum(r["cite_ok"] for r in correct_answerable) / len(correct_answerable)
                                           if correct_answerable else 0.0)
                result["refusal_ok"] = sum(r["correct"] for r in rows if r["hit"] is None)
                result["refusal_total"] = len(questions) - len(answerable)
            summary.append(result)
            (RESULT_DIR / f"{setting}{'_retrieval' if retrieval_only else ''}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False))

    name = "summary_retrieval" if retrieval_only else "summary"
    (RESULT_DIR / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(summary, retrieval_only, len(answerable), len(questions))


def print_table(summary, retrieval_only, n_answerable, n_total):
    if retrieval_only:
        print(f"\n| 设置 | 检索命中率（{n_answerable} 题） |\n|---|---|")
        for r in summary:
            print(f"| chunk={r['chunk_size']}, top-k={r['top_k']} | {r['hit_rate']:.0%} |")
    else:
        print(f"\n| 设置 | 检索命中率（{n_answerable} 题） | 回答准确率（{n_total} 题） | 引用页码正确率 | 未披露题拒答 |")
        print("|---|---|---|---|---|")
        for r in summary:
            print(f"| chunk={r['chunk_size']}, top-k={r['top_k']} | {r['hit_rate']:.0%} | {r['accuracy']:.0%} "
                  f"| {r['cite_accuracy']:.0%} | {r['refusal_ok']}/{r['refusal_total']} |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="只评估检索，不调用大模型")
    run(parser.parse_args().retrieval_only)
