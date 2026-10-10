"""模块 2a：切块。

按页切块，chunk 不跨页，所以每个 chunk 的页码是确定的。
在页内按"行"累加到目标字数再切，表格的一行不会被从中间切断；
相邻 chunk 有约 15% 的重叠，避免一句话刚好被切在边界上。

运行：python -m src.chunking  （打印各切块大小下的 chunk 数量和样例）
"""
import json
import sys

from src.config import CHUNK_OVERLAP_RATIO, CHUNK_SIZES, PROCESSED_DIR, company_of


def split_page(text: str, size: int) -> list[str]:
    overlap = int(size * CHUNK_OVERLAP_RATIO)
    lines = []
    for line in text.splitlines():  # 超长行（极少见）先硬切
        lines += [line[i:i + size] for i in range(0, len(line), size)] or [""]

    chunks, cur = [], []
    for line in lines:
        if cur and sum(len(l) + 1 for l in cur) + len(line) > size:
            chunks.append("\n".join(cur))
            # 从末尾往回取若干行作为下一个 chunk 的开头（重叠部分）
            tail, n = [], 0
            for l in reversed(cur):
                if n + len(l) > overlap:
                    break
                tail.insert(0, l)
                n += len(l) + 1
            cur = tail
        cur.append(line)
    if cur and "".join(cur).strip():
        chunks.append("\n".join(cur))
    return chunks


def load_chunks(size: int) -> list[dict]:
    """返回 [{"text", "doc", "company", "page", "chunk_id"}]。"""
    out = []
    for path in sorted(PROCESSED_DIR.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            page = json.loads(line)
            for i, text in enumerate(split_page(page["text"], size)):
                if len(text.strip()) < 20:  # 过短的碎片（如孤立标题）没有检索价值
                    continue
                out.append({
                    "text": text,
                    "doc": page["doc"],
                    "company": company_of(page["doc"]),
                    "page": page["page"],
                    "chunk_id": f"{path.stem}-p{page['page']}-c{i}",
                })
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for size in CHUNK_SIZES:
        chunks = load_chunks(size)
        avg = sum(len(c["text"]) for c in chunks) / len(chunks)
        print(f"chunk_size={size}: {len(chunks):,} 个 chunk，平均 {avg:.0f} 字")
    print("样例：", json.dumps(load_chunks(300)[50], ensure_ascii=False, indent=2))
