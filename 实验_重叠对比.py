"""实验：重叠 = 50 vs 重叠 = 0，亲眼看「边界」有什么区别。

直接双击运行，看输出里两个块的对比。
"""
from pathlib import Path
from d07_load_split import 按段落切, 读TXT   # 复用 D7 里写好的函数

text = 读TXT(Path(__file__).parent / "sample" / "RAG入门.txt")

for overlap in (50, 0):
    chunks = 按段落切(text, 每段字数=500, 重叠=overlap)

    # 第 0 段的「结尾」、第 1 段的「开头」，各取 50 字来比
    tail0 = chunks[0][-50:]
    head1 = chunks[1][:50]

    print("=" * 56)
    print(f"重叠 = {overlap}      切出 {len(chunks)} 段")
    print("=" * 56)
    print("第 0 段结尾（最后 50 字）：")
    print("  «" + tail0.replace("\n", " ") + "»")
    print("第 1 段开头（前 50 字）：")
    print("  «" + head1.replace("\n", " ") + "»")

    if tail0 == head1:
        print("  ↑ 结论：第1段开头 = 第0段结尾，同一句被抄进两段 → 重叠生效")
    else:
        print("  ↑ 结论：第1段开头是全新内容 → 边界没有重复")
    print()

print("=" * 56)
print("对比总结")
print("=" * 56)
print("  · 段数：两种情况都是 11 段 —— 重叠不改变段数")
print("  · 区别只在「边界」：重叠=50 时相邻两段有一段重复内容")
print("  · 这样切在边界上的那句话，至少有一段是完整的")
