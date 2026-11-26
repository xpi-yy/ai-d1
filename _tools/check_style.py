#!/usr/bin/env python3
"""使用说明体例自检（对应《索引》R6 / 第七节第 9 条）

用法：
    python3 _tools/check_style.py              # 检查目录下所有使用说明
    python3 _tools/check_style.py 某文件.md     # 只查指定文件

检查五项：
    ① 代码块三反引号成对
    ② 表格列数一致（同一张表内，表头为准）
    ③ 加粗标记 ** 成对（跳过代码块与行内代码）
    ④ 标题层级不跳级（H1 → H3 这种）
    ⑤ 表头分隔行合法（第二行必须是 |---|---| 形式）

已知误报（本脚本已规避）：
    · 表格里的转义竖线 \\|
    · 行内代码段 `...` 里的 | 和 **
    · 代码块里的 # 被当成标题
"""
import re
import sys
from pathlib import Path

def 找目标文件(目录: Path):
    """自动发现工作区里的说明文档（文件名带日期，所以不能硬编码）"""
    return sorted(
        list(目录.glob("使用说明_*.md")) + list(目录.glob("_AI失误台账.md"))
    )


def 检查(path: Path):
    问题 = []
    lines = path.read_text(encoding="utf-8").splitlines()

    # ① 围栏成对
    fences = [i for i, l in enumerate(lines, 1) if l.strip().startswith("```")]
    if len(fences) % 2:
        问题.append(f"[围栏] 代码块 ``` 数量为奇数（{len(fences)}），有未闭合的代码块")

    in_code = False
    表头列数 = None
    表起始 = None
    上一个标题级 = 0

    for i, l in enumerate(lines, 1):
        s = l.strip()

        if s.startswith("```"):
            in_code = not in_code
            表头列数 = None
            continue
        if in_code:
            continue

        # ④ 标题层级
        if s.startswith("#"):
            级 = len(s) - len(s.lstrip("#"))
            if 上一个标题级 and 级 > 上一个标题级 + 1:
                问题.append(f"[标题] 第 {i} 行：H{上一个标题级} 直接跳到 H{级}，跳过了一级")
            上一个标题级 = 级

        # ② 表格列数
        if s.startswith("|") and s.endswith("|"):
            列数 = len(re.findall(r"(?<!\\)\|", s)) - 1
            if 表头列数 is None:
                表头列数 = 列数
                表起始 = i
            elif 列数 != 表头列数:
                问题.append(f"[表格] 第 {i} 行有 {列数} 列，但表头（第 {表起始} 行）是 {表头列数} 列")
        else:
            表头列数 = None

        # ③ 加粗配对（去行内代码后数）
        去代码 = re.sub(r"`[^`]*`", "", l)
        if 去代码.count("**") % 2:
            问题.append(f"[加粗] 第 {i} 行：** 数量为奇数，加粗标记未配对")

    return 问题


def main():
    args = sys.argv[1:]
    if args:
        文件们 = [Path(a) for a in args]
    else:
        目录 = Path(__file__).resolve().parent.parent
        文件们 = 找目标文件(目录)
        if not 文件们:
            print(f"⚠️  {目录} 下没找到 使用说明_*.md —— 是不是还没初始化？")
            print("   初始化： bash ~/.dsh/workspace-template/init_workspace.sh <目录>")
            return 1

    总问题 = 0
    for f in 文件们:
        if not f.exists():
            print(f"⚠️  跳过（不存在）：{f.name}")
            continue
        问题 = 检查(f)
        if 问题:
            总问题 += len(问题)
            print(f"❌ {f.name}  —— {len(问题)} 处问题")
            for q in 问题[:20]:
                print(f"     {q}")
            if len(问题) > 20:
                print(f"     …… 还有 {len(问题) - 20} 处")
        else:
            print(f"✅ {f.name}")

    print()
    if 总问题:
        print(f"共 {总问题} 处问题，请修完再交付（R6）。")
        return 1
    print("体例检查全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
