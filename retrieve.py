"""D8：建库与检索 —— 把 D6 的 FAISS 和 D7 的切分接起来。

用法（在终端里）：
    ./.venv/bin/python retrieve.py "RAG 是什么？"     # 问一个问题
    ./.venv/bin/python retrieve.py                   # 不给问题 → 自动跑 5 个验收测试

它做的事（就是 RAG 的前半条链路）：
    ① 读文档 → ② 切成 chunk → ③ 每段算向量 → ④ 存进 FAISS → ⑤ 给问题找最像的 3 段
"""
from pathlib import Path      # 导入 Path：用来看文件、拼路径，Mac/Windows 都能用
import re                     # 导入 re：正则表达式，用来识别「一、」「二、」这种小节标题
import sys                    # 导入 sys：用来读取命令行传给程序的参数

from embed_helper import embed   # 从我们自己的工具文件里，导入 D6 的 embed（句子 → 向量）


# ══════════════════════════════════════════════════════════
# 第 1 步：读文档
# ══════════════════════════════════════════════════════════
def load_text(path):                              # 定义函数 load_text，参数是文件路径
    """把整个文件读成一个大字符串。"""
    return Path(path).read_text(encoding="utf-8")   # 按 UTF-8 编码读取文件，中文才不会乱码

# ══════════════════════════════════════════════════════════
# 第 2 步：切分（按小节标题切，比按字数硬切准）
# ══════════════════════════════════════════════════════════
def split_into_chunks(text):                        # 定义函数 split_into_chunks，参数是整篇文字
    """按小节切：遇到「一、」「二、」这种标题，就开新的一段。

    为什么按小节切？这篇文档本身就有 23 个小节标题，按标题切，
    每一段正好是一个完整意思，检索时不会跟别的小节混在一起。

    （如果文档没有标题，就用 D7 那种按字数切 + 重叠的方式。）
    """
    chunks = []        # 空的列表，用来装切好的每一段
    current = []       # 空的列表，用来装「正在拼的那一段」的每一行

    for line in text.split("\n"):                   # 把全文按换行符拆成一行一行，逐行循环
        s = line.strip()                            # 去掉这一行开头和结尾的空白字符（空格、换行）
        if not s or set(s) <= {"="}:                # 如果这行是空的，或者整行只由 = 号组成（分隔线）
            continue                                # 那么跳过这一行，处理下一行
        is_heading = re.match(r"^[一二三四五六七八九十]+、", s) is not None   # True 表示这行是小节标题
        if is_heading:                              # 如果这一行是新小节的开头
            if current:                             # 而且前面已经攒了一段内容
                chunks.append("\n".join(current))   # 就把前面攒的那段封口，放进 chunks
            current = [s]                           # 新的一段从这一行标题开始装
        elif current:                               # 如果这行不是标题、而且已经在攒一段了
            current.append(s)                       # 就把这一行接到当前段的末尾

    if current:                                     # 循环结束后，最后一段可能还没封口
        chunks.append("\n".join(current))           # 把最后这段也封口放进 chunks

    return chunks                                   # 把切好的所有段交出去


# ══════════════════════════════════════════════════════════
# 第 3 步：建索引
# ══════════════════════════════════════════════════════════
def build_index(chunks):                            # 定义函数 build_index，参数是所有文本段
    """每段变成向量 → 存进 FAISS。返回 (索引, 向量矩阵)。"""
    import faiss                                    # 导入 faiss 向量检索库（放在函数里，不用时不加载）

    vectors = embed(chunks).numpy().astype("float32")   # 把每段都变成向量，得到矩阵 (段数, 512)
    index = faiss.IndexFlatL2(vectors.shape[1])         # 建一个 FAISS 索引，维度是 512（对应每段向量长度）
    index.add(vectors)                                  # 把所有向量装进索引，等着被搜索
    return index, vectors                           # 把索引和向量矩阵都交出去


# ══════════════════════════════════════════════════════════
# 第 4 步：检索
# ══════════════════════════════════════════════════════════
def search_chunks(query, chunks, index, k=3):        # 定义函数 search_chunks（问题、所有段、索引、要几段）
    """给一个问题，返回最像的 k 段：[(chunk 原文, 距离), ...]。"""
    query_vector = embed([query]).numpy().astype("float32")   # 把问题变成一个向量 (1, 512)
    distances, indices = index.search(query_vector, k)        # 让 FAISS 找出距离最近的 k 个

    results = []                                     # 空列表，用来装 (原文, 距离)
    for i in range(k):                               # 从第 1 名到第 k 名，逐个处理
        idx = indices[0][i]                          # 第 i 名的「下标」：是 23 段里的第几段
        results.append((chunks[idx], float(distances[0][i])))   # 用下标换回原文，和距离一起装进结果
    return results                                   # 把结果交出去


def print_results(query, results):                   # 定义函数 print_results，把结果打印得好看
    """把检索结果打印得好看点。"""
    print(f"\n问题：{query}")                         # 先空一行，再打印问题
    print("最相关的 3 段（距离越小越像）：")            # 打印一行提示语
    for n, (chunk, dist) in enumerate(results, 1):   # 从 1 开始编号，逐个取出（原文, 距离）
        preview = chunk.replace("\n", " ")[:70]      # 把原文里的换行换成空格，取前 70 字作预览
        print(f"  {n}. [距离 {dist:.3f}] {preview}…") # 打印：第几名、距离（保留 3 位）、预览


# ══════════════════════════════════════════════════════════
# 验收测试：5 个问题，至少命中 4 个
# ══════════════════════════════════════════════════════════
TESTS = [                                            # 测试题目列表，每项是 (问题, 期望关键词)
    ("RAG 是什么？", "Retrieval-Augmented"),          # 第 1 题
    ("为什么要让相邻两段重叠？", "重叠"),              # 第 2 题
    ("怎么减少幻觉？", "幻觉"),                        # 第 3 题
    ("模型怎么调用函数？", "Function Calling"),        # 第 4 题（用中文问，英文直接问检索不准）
    ("微调和 RAG 怎么选？", "微调"),                   # 第 5 题
]                                                    # 测试题目列表结束


def run_acceptance_test(chunks, index):              # 定义函数 run_acceptance_test：跑 5 题验收
    print("=" * 62)                                  # 打印一条分隔线（62 个 = 号）
    print("验收测试：5 个问题，看检索命中没")            # 打印标题
    print("=" * 62)                                  # 再打印一条分隔线
    passed = 0                                       # 命中的题数，先设为 0
    for query, keyword in TESTS:                     # 一题一题地测
        results = search_chunks(query, chunks, index, k=3)   # 取前 3 段结果
        hit = any(keyword in chunk for chunk, _ in results)  # 前 3 段里有期望关键词 → 算命中
        mark = "✅" if hit else "❌"                   # 命中打勾，没命中打叉
        top = results[0][0].replace("\n", " ")[:55]  # 第一名的开头 55 字，用于展示
        print(f"{mark} {query}")                     # 打印这一题的勾/叉和题目
        print(f"    期望包含「{keyword}」；第一名：{top}…")   # 打印期望关键词和实际第一名
        if hit:                                      # 如果这一题命中了
            passed += 1                              # 命中数加 1

    print()                                          # 空一行
    print(f"结果：{passed}/5 命中（验收线 4/5）")       # 打印总成绩
    assert passed >= 4, f"只命中 {passed}/5，低于验收线"  # 断言：少于 4 个就报错停下
    print("D8 验收通过 ✅")                            # 全部达标，打印通过


# ══════════════════════════════════════════════════════════
# 主程序（只有直接运行这个文件时才执行）
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":                           # 判断：是不是「直接运行」这个文件
    text = load_text(Path(__file__).parent / "sample" / "RAG入门.txt")   # 读入样例文档
    chunks = split_into_chunks(text)                 # 把文档切成 23 段
    print(f"读入 {len(text)} 字 → 切成 {len(chunks)} 段")   # 打印切分结果，方便核对

    index, _ = build_index(chunks)                   # 建库（每段变向量 + 存进索引）

    if len(sys.argv) > 1:                            # 如果命令行给了问题参数
        query = sys.argv[1]                          # 取出用户的问题
        print_results(query, search_chunks(query, chunks, index, 3))   # 检索并打印结果
    else:                                            # 如果没给问题参数
        run_acceptance_test(chunks, index)           # 就自动跑 5 题验收测试
