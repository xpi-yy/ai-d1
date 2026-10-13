"""PyCharm 配置验证脚本

用法：在 PyCharm 里右键这个文件 → Run 'hello_pycharm'
四部分全部显示 ✅，说明解释器、工作目录、依赖都配好了。
"""
import os
import sys

print("=" * 60)
print("1. 解释器检查")
print("   Python 版本 :", sys.version.split()[0])
print("   Python 路径 :", sys.executable)
if ".venv" in sys.executable:
    print("   ✅ 用的是项目虚拟环境")
else:
    print("   ⚠️ 用的不是项目虚拟环境！")
    print("      去 PyCharm → Settings → Project: ai-d1 → Python Interpreter")
    print("      选择 Add Interpreter → Existing → 填：")
    print("      ~/ai-d1/.venv/bin/python")

print()
print("2. 工作目录检查")
print("   当前目录 :", os.getcwd())
if os.path.exists("sample/技术笔记.txt"):
    print("   ✅ 能找到 sample/技术笔记.txt")
else:
    print("   ⚠️ 找不到 sample/技术笔记.txt —— 工作目录不对")
    print("      在 PyCharm 里应该打开 ai-d1 这个文件夹作为项目根目录")

print()
print("3. 依赖检查")
try:
    import torch

    print("   PyTorch  :", torch.__version__)
    print("   MPS 可用 :", torch.backends.mps.is_available())
    print("   ✅ 依赖正常")
except ImportError:
    print("   ⚠️ 没找到 torch —— 基本可以确定是解释器选错了")

print()
print("4. 读文件（D3 的内容）")
try:
    with open("sample/技术笔记.txt", "r", encoding="utf-8") as f:
        content = f.read()
    print(f"   ✅ 读到 {len(content)} 个字")
except FileNotFoundError:
    print("   ⚠️ 读不到文件，见上面第 2 条")

print("=" * 60)


# ---------- 下面是你 D2-6 那道题的正确写法，可以直接在 PyCharm 里跑 ----------
def count_keywords(text, keywords):
    counts = {}
    for kw in keywords:
        counts[kw] = text.count(kw)  # 注意：方括号存取，圆括号调用方法
    return counts


if __name__ == "__main__":
    print()
    print("D2-6 的正确版本，跑一下看看：")
    文本 = "检索很重要，检索做得好，幻觉就少"
    词表 = ["检索", "幻觉", "向量"]
    print("   结果 =", count_keywords(文本, 词表))
    print("   期望 = {'检索': 2, '幻觉': 1, '向量': 0}")
