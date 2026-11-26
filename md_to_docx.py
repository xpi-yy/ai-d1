"""把面试问题的 Markdown 转成 Word（.docx）。

处理：# / ## / ### 标题、**加粗**、`行内代码`、有序/无序列表、表格、引用。
"""
import re

from docx import Document
from docx.shared import Pt, Inches
from docx.oxml.ns import qn


EAST_ASIAN_FONT = "宋体"          # 中文字体
LATIN_FONT = "Times New Roman"    # 英文字体
CODE_FONT = "Consolas"            # 行内代码字体


def set_font(run, font_name=LATIN_FONT):
    """给一个 run 同时设置中英文字体（中文必须额外设 w:eastAsia）。"""
    run.font.name = font_name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), EAST_ASIAN_FONT)


def add_inline(paragraph, text):
    """把 **加粗** 和 `代码` 拆成不同格式的 run。"""
    parts = re.split(r"(\*\*.+?\*\*|`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
            set_font(run)
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = CODE_FONT
            set_font(run, CODE_FONT)
        else:
            run = paragraph.add_run(part)
            set_font(run)


def parse_table(lines):
    """把连续的 markdown 表格行解析成二维列表，跳过 |---| 分隔行。"""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def add_table(doc, rows):
    """把二维列表写成 Word 表格。"""
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=0, cols=ncols)
    table.style = "Table Grid"
    for row in rows:
        cells = table.add_row().cells
        for i in range(ncols):
            val = row[i] if i < len(row) else ""
            p = cells[i].paragraphs[0]
            add_inline(p, val)


def convert(md_path, docx_path):
    """读取 markdown，生成 docx。"""
    lines = open(md_path, encoding="utf-8").read().split("\n")
    doc = Document()

    # 默认正文字体
    normal = doc.styles["Normal"]
    normal.font.name = LATIN_FONT
    normal.font.size = Pt(11)
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), EAST_ASIAN_FONT)

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:                       # 空行
            i += 1
            continue
        if stripped == "---":                   # 水平线
            i += 1
            continue

        if stripped.startswith("|"):            # 表格
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, parse_table(block))
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)   # 标题
        if m:
            level = len(m.group(1))
            p = doc.add_heading(level=0 if level == 1 else level - 1)
            run = p.add_run(m.group(2))
            set_font(run)
            i += 1
            continue

        if stripped.startswith(">"):            # 引用
            p = doc.add_paragraph(style="Quote")
            add_inline(p, stripped[1:].strip())
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):      # 有序列表（保留原编号）
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            add_inline(p, stripped)
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):   # 无序列表
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            add_inline(p, "• " + stripped[2:])
            i += 1
            continue

        p = doc.add_paragraph()                  # 普通段落
        add_inline(p, stripped)
        i += 1

    doc.save(docx_path)
    print("已生成:", docx_path)


if __name__ == "__main__":
    convert("面试问题_问题版.md", "面试问题_问题版.docx")
    convert("面试问题_答案版.md", "面试问题_答案版.docx")
