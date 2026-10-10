"""模块 1：年报 PDF 解析。

读取 data/raw/*.pdf，逐页输出到 data/processed/<文件名>.jsonl，每行一个页面：
    {"doc": "宁德时代2025年年度报告.pdf", "page": 12, "text": "..."}

处理要点：
- 页眉页脚：统计各页顶部/底部区域的文本行，在 30% 以上页面重复出现的视为页眉页脚并删除。
  比较前先把数字替换成 #，这样 "第 12 页" 和 "第 13 页" 会被当成同一行。
- 表格：用 PyMuPDF 的 find_tables 识别，转成 "| a | b |" 形式的行，
  并把落在表格区域内的普通文本去掉，避免重复。
- 页码：使用 PDF 物理页码（从 1 开始），与 PDF 阅读器显示的页码一致。

运行：python -m src.parse_pdf
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

EDGE_RATIO = 0.08  # 页面顶部/底部 8% 的区域视为页眉页脚候选区
REPEAT_RATIO = 0.3  # 在 30% 以上页面出现的行判定为页眉页脚
ROW_TOL = 3.0  # 纵向中心相差 3pt 以内视为同一视觉行
CELL_GAP = 12.0  # 水平间距超过 12pt 视为不同单元格


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def _inside(rect, boxes: list[pymupdf.Rect]) -> bool:
    r = pymupdf.Rect(rect[:4])
    return any(b.contains(r) or (r & b).get_area() > 0.5 * r.get_area() for b in boxes)


def _visual_rows(page: pymupdf.Page, boxes: list[pymupdf.Rect]) -> list[tuple[float, float, str]]:
    """把页面上的文字按 y 坐标重组成"视觉行"，返回 [(y0, y1, 文本)]。

    无框线表格里，同一行的标签和数字是分散的多个文本行；按纵向中心对齐把它们拼回一行，
    水平间距较大的片段之间用 " | " 分隔，这样 "营业收入 | 838,270 | 821,803" 不会被拆散。
    """
    spans = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            x0, y0, x1, y1 = line["bbox"]
            # 个别 PDF 含有"挤成一条竖线"的隐藏文字层（如工行年报把封面文字叠在审计报告页上），
            # 特征是整串文字的宽度远小于 字数×字号，直接丢弃
            if len(text) >= 4 and (x1 - x0) < 0.2 * len(text) * (y1 - y0):
                continue
            if text and not _inside(line["bbox"], boxes):
                spans.append((line["bbox"], text))
    spans.sort(key=lambda s: (s[0][1] + s[0][3]) / 2)

    rows = []  # 每行: [y0, y1, [(x0, x1, text), ...]]
    for (x0, y0, x1, y1), text in spans:
        yc = (y0 + y1) / 2
        if rows:
            r = rows[-1]
            if abs(yc - (r[0] + r[1]) / 2) < ROW_TOL:
                r[0], r[1] = min(r[0], y0), max(r[1], y1)
                r[2].append((x0, x1, text))
                continue
        rows.append([y0, y1, [(x0, x1, text)]])

    out = []
    for y0, y1, pieces in rows:
        pieces.sort()
        line, last_x1 = "", None
        for x0, x1, text in pieces:
            if last_x1 is not None:
                if x0 - last_x1 > CELL_GAP:
                    line += " | "
                elif line[-1:].isascii() and text[:1].isascii():
                    line += " "
            line += text
            last_x1 = x1
        out.append((y0, y1, line))
    return out


def _is_edge(y0: float, y1: float, h: float) -> bool:
    return y1 < h * EDGE_RATIO or y0 > h * (1 - EDGE_RATIO)


def find_header_footer(doc: pymupdf.Document) -> set[str]:
    counter = Counter()
    for page in doc:
        h = page.rect.height
        counter.update({_norm(t) for y0, y1, t in _visual_rows(page, []) if _is_edge(y0, y1, h)})
    threshold = max(3, int(len(doc) * REPEAT_RATIO))
    return {line for line, n in counter.items() if n >= threshold}


def _table_to_text(table) -> str:
    """有框线表格 -> "| a | b |" 行。去掉空单元格，并把单元格内换行造成的续行并回上一行。"""
    rows = []  # [(第一个非空列号, [非空单元格])]
    for raw in table.extract():
        cells = [(i, re.sub(r"\s+", "", c or "")) for i, c in enumerate(raw)]
        cells = [(i, c) for i, c in cells if c]
        if not cells:
            continue
        if (rows and len(cells) == 1 and len(rows[-1][1]) >= 2 and cells[0][0] == rows[-1][0]
                and len(cells[0][1]) <= 12 and len(rows[-1][1][0]) >= 8):
            rows[-1][1][0] += cells[0][1]  # 如 "归属于上市公司股东" + "的净利润"
            continue
        rows.append((cells[0][0], [c for _, c in cells]))
    return "\n".join("| " + " | ".join(cells) + " |" for _, cells in rows)


def parse_page(page: pymupdf.Page, noise: set[str]) -> str:
    h = page.rect.height
    try:
        tables = list(page.find_tables())
    except Exception:  # 个别页面表格识别会报错，退化成纯文本
        tables = []
    boxes = [pymupdf.Rect(t.bbox) for t in tables]

    # (y 坐标, 文本) 列表，最后按 y 排序，保持表格与正文的阅读顺序
    items = [(t.bbox[1], _table_to_text(t)) for t in tables]
    for y0, y1, line in _visual_rows(page, boxes):
        if _is_edge(y0, y1, h) and (_norm(line) in noise or re.fullmatch(r"[\d\s/\-第页共]+", line)):
            continue
        items.append((y0, line))

    items.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in items if text).strip()


def parse_pdf(path: Path) -> list[dict]:
    doc = pymupdf.open(path)
    noise = find_header_footer(doc)
    pages = []
    for i, page in enumerate(doc):
        text = parse_page(page, noise)
        if text:
            pages.append({"doc": path.name, "page": i + 1, "text": text})
    return pages


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"没有找到 PDF，请把年报放进 {RAW_DIR}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        pages = parse_pdf(pdf)
        out = OUT_DIR / f"{pdf.stem}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for p in pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        chars = sum(len(p["text"]) for p in pages)
        n_table_lines = sum(p["text"].count("\n| ") for p in pages)
        print(f"{pdf.name}: {len(pages)} 页有文本，共 {chars:,} 字，表格行约 {n_table_lines:,} -> {out.name}")


if __name__ == "__main__":
    main()
