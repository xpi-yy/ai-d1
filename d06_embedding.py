"""D6：Embedding + 向量检索

在 PyCharm 里怎么跑：
    右键这个文件 → Run 'd06_embedding'
    （或者点右上角的绿色三角）

和 Jupyter 的区别：
    · 整个文件一次性跑完，没有「一格一格执行」
    · 想看某个函数单独的结果 → 把 __main__ 里其他调用注释掉
    · 想中途看变量 → 用 PyCharm 的 Debug（断点 + 变量面板），比 Jupyter 强
"""

# ══════════════════════════════════════════════════════════
# 环境准备
# ══════════════════════════════════════════════════════════
# 注意：KMP_DUPLICATE_LIB_OK 和 scipy 桩模块都已经在 embed_helper.py 里处理了。
# 所以这里只要正常 import 就行，不用管那些环境问题。
from embed_helper import embed, cosine_similarity   # 导入 D6 的工具函数（句子→向量、余弦相似度）

import math                                         # 导入 math：下面要用 math.sqrt 开根号


# ══════════════════════════════════════════════════════════
# 第一部分：手写余弦相似度（用 2 维向量理解公式）
# ══════════════════════════════════════════════════════════

def dot_product(a, b):                              # 定义函数：两个向量的点积
    """两个向量对应位置相乘，再把所有乘积加起来。

    例子：[1, 2] 和 [3, 4] → 1*3 + 2*4 = 11
    """
    result = 0                       # 累加器：从 0 开始，装总和
    for x, y in zip(a, b):           # zip 把 a 和 b 并排拉在一起，一次取一对
        result = result + x * y      # 这一对相乘，加进累加器
    return result                     # 循环结束，交出总和


def length(vec):                                    # 定义函数：向量长度
    """向量长度 = 各分量平方和再开根号（勾股定理的推广）。

    例子：[3, 4] → √(9 + 16) = √25 = 5
    """
    sum_of_squares = 0                # 累加器：装平方和
    for x in vec:                     # 遍历每个分量
        sum_of_squares = sum_of_squares + x * x   # 平方后加进去
    return math.sqrt(sum_of_squares)  # 开根号


def cosine_similarity_manual(a, b):                 # 定义函数：手写版余弦相似度
    """余弦相似度 = 点积 ÷ (长度a × 长度b)。

    结果范围：
        1  = 方向相同（完全一样）
        0  = 垂直（毫无关系）
        -1 = 方向相反（意思相反）
    """
    return dot_product(a, b) / (length(a) * length(b))   # 点积 ÷ 两个长度的乘积


def demo1_handwritten_cosine():                     # 演示 1：手写余弦相似度
    """用 2 维向量跑四种情况，看清公式在干嘛。"""
    print("=" * 62)                                # 打印分隔线
    print("第一部分：手写余弦相似度（2 维向量）")      # 打印标题
    print("=" * 62)                                # 打印分隔线

    print("情况 1：方向相同  a=[1,0]  b=[2,0]")      # 打印情况说明
    print("        相似度 =", round(cosine_similarity_manual([1, 0], [2, 0]), 4), "  ← 1.0")   # 算出并打印（保留 4 位）
    print()                                        # 空一行

    print("情况 2：互相垂直  a=[1,0]  b=[0,3]")      # 打印情况说明
    print("        相似度 =", round(cosine_similarity_manual([1, 0], [0, 3]), 4), "  ← 0")     # 算出并打印
    print()                                        # 空一行

    print("情况 3：方向相反  a=[1,2]  b=[-1,-2]")    # 打印情况说明
    print("        相似度 =", round(cosine_similarity_manual([1, 2], [-1, -2]), 4), "  ← -1")  # 算出并打印
    print()                                        # 空一行

    print("情况 4：偏一点    a=[1,0]  b=[1,1]")      # 打印情况说明
    print("        相似度 =", round(cosine_similarity_manual([1, 0], [1, 1]), 4), "  ← 0.707") # 算出并打印
    print()                                        # 空一行


# ══════════════════════════════════════════════════════════
# 第二部分：真实的 Embedding（512 维）
# ══════════════════════════════════════════════════════════

sentences = [                                     # 准备 4 句测试句子
    "今天天气很好",                                # 第 1 句
    "今儿个天气不错",                              # 第 2 句（和第 1 句意思一样）
    "我喜欢吃火锅",                                # 第 3 句（无关）
    "检索增强生成能减少幻觉",                       # 第 4 句（无关）
]                                                 # 4 句测试句子列表结束


def demo2_real_vectors():                         # 演示 2：真实向量
    """把 4 句话变成 512 维向量，并两两算相似度。"""
    print("=" * 62)                                # 打印分隔线
    print("第二部分：真实 Embedding（512 维）")        # 打印标题
    print("=" * 62)                                # 打印分隔线

    # embed 返回一个 (句子数, 512) 的矩阵
    vectors = embed(sentences)                    # 4 句话 → 4 个向量

    print("每句话 →", len(vectors[0]), "个数字")     # 打印每句有多少个数字
    for s, v in zip(sentences, vectors):           # 句子和它的向量并排遍历
        # v[:5] 只取前 5 个数字；round(...,3) 保留 3 位小数
        first_five = [round(float(x), 3) for x in v[:5]]   # 取前 5 个数、保留 3 位小数
        print(f"  {s:12s} {first_five}")            # 打印句子和它的前 5 个数
    print()                                        # 空一行

    print("两两相似度（1 = 完全一样，0 = 毫无关系）：")   # 打印标题
    for i in range(len(sentences)):                # 外层循环：第 i 句
        # 内层从 i+1 开始：每对只算一次，也不会自己跟自己比
        for j in range(i + 1, len(sentences)):     # 内层循环：第 j 句
            s = cosine_similarity(vectors[i], vectors[j])   # 算第 i 句和第 j 句的相似度
            print(f"  {sentences[i]:12s} ↔ {sentences[j]:12s}  {s:.4f}")   # 打印这一对的结果
    print()                                        # 空一行
    print("关键：『今天天气很好』↔『今儿个天气不错』字面几乎不同，但相似度 0.84")   # 打印关键说明
    print("      说明模型比的是【意思】，不是字面。")    # 打印结论
    print()                                        # 空一行


# ══════════════════════════════════════════════════════════
# 第三部分：FAISS 向量检索
# ══════════════════════════════════════════════════════════

documents = [                                     # 准备 8 篇文档
    "今天天气很好，适合出门散步",                    # 第 0 篇
    "RAG 的核心是先检索再生成",                     # 第 1 篇
    "火锅是四川的特色美食",                          # 第 2 篇
    "检索增强生成可以减少模型幻觉",                   # 第 3 篇
    "明天可能下雨，记得带伞",                        # 第 4 篇
    "向量数据库用来做相似度搜索",                     # 第 5 篇
    "张三是公司的销售经理",                          # 第 6 篇
    "FAISS 是 Facebook 开源的向量检索库",            # 第 7 篇
]                                                 # 8 篇文档列表结束


def demo3_faiss_search():                         # 演示 3：FAISS 检索
    """用 FAISS 建索引，输入问题返回最相关的 3 段。"""
    import faiss   # 放在函数里 import，这样不跑这个演示时不加载

    print("=" * 62)                                # 打印分隔线
    print("第三部分：FAISS 向量检索")                 # 打印标题
    print("=" * 62)                                # 打印分隔线

    # ① 所有文档变向量
    #    .numpy()            torch 张量 → numpy 数组
    #    .astype("float32")  FAISS 要求 float32
    doc_vectors = embed(documents).numpy().astype("float32")   # 8 篇文档 → 8 个向量
    print("文档向量形状:", doc_vectors.shape, " 读作（文档数, 维度）")   # 打印形状

    # ② 建索引。IndexFlatL2 = 暴力逐一比较，小数据够用
    index = faiss.IndexFlatL2(doc_vectors.shape[1])   # 参数是向量维度 512
    index.add(doc_vectors)                            # 把 8 个向量加进去
    print("索引里的向量数量:", index.ntotal)           # 打印索引里有多少个向量

    # ③ 搜索
    query = "怎么减少大模型的幻觉？"                    # 要问的问题
    # 注意：外面套 [query]，因为 FAISS 要 (查询数, 维度) 的二维数组
    query_vector = embed([query]).numpy().astype("float32")   # 问题 → 1 个向量

    # search 返回两个数组：距离（越小越像）和编号（对应文档库的下标）
    distances, indices = index.search(query_vector, 3)   # 找最近的 3 个

    print(f"\n问题：{query}")                          # 打印问题
    print("最相关的 3 段：")                            # 打印标题
    for k in range(3):                                 # 逐个打印 3 个结果
        # indices[0][k]：[0] 是第 0 个查询（只有 1 个），[k] 是第 k 条结果
        idx = indices[0][k]                            # 第 k 条是第几篇
        print(f"  {k + 1}. {documents[idx]}")          # 打印编号和原文
    print()                                            # 空一行


# ══════════════════════════════════════════════════════════
# 第四部分：自己写检索函数（不依赖 FAISS）
# ══════════════════════════════════════════════════════════

def search(query, docs, k=3):                      # 定义检索函数（问题、所有文档、返回几条）
    """暴力版检索：给一个问题，返回最相关的 k 篇文档正文。

    参数：
        query —— 一句提问
        docs  —— 一堆候选文本
        k     —— 返回几条（默认 3）

    返回：
        最相关的 k 篇文档正文，组成一个列表
    """
    # ① 问题 → 一个 512 维向量
    #    embed([query]) 返回 (1, 512) 的矩阵，[0] 取出第一行
    query_vector = embed([query])[0]               # 问题 → 单个向量

    # ② 所有文档 → (篇数, 512) 的矩阵
    #    注意：这里不加 [i]，因为要的是【全部】文档
    doc_vectors = embed(docs)                      # 所有文档 → 向量矩阵

    # ③ 给每一篇打分，存成 (分数, 下标) 的小包裹
    scores = []                                    # 空口袋，准备装 (分数, 下标)
    for i in range(len(docs)):                     # i 从 0 到「篇数-1」
        # doc_vectors[i] 是第 i 篇的向量（矩阵的第 i 行）
        score = cosine_similarity(query_vector, doc_vectors[i])   # 抽第 i 排来比，得分数
        scores.append((score, i))                  # 打包：分数 + 下标
        # ↑ 必须存下标，否则排完序就不知道第 1 名是哪篇了

    # ④ 按分数从大到小排序
    #    注意：sorted 返回【新列表】，所以左边要接住
    #          这和 append 相反 —— append 是原地修改，不能接
    scores = sorted(scores, key=lambda pair: pair[0], reverse=True)   # 按分数从高到低排
    #                                  ↑ 按包裹的第 0 个元素（分数）比大小

    # ⑤ 取前 k 个，用下标换回正文
    results = []                                   # 空列表，准备装结果
    for pair in scores[:k]:                        # [:k] 取前 k 个包裹
        results.append(docs[pair[1]])              # pair[1] 是下标 → 取正文

    return results                                 # 交出结果


def demo4_custom_search():                         # 演示 4：自己写的检索
    """调用自己写的 search，和 FAISS 的结果对比。"""
    print("=" * 62)                                # 打印分隔线
    print("第四部分：自己写的暴力检索")                 # 打印标题
    print("=" * 62)                                # 打印分隔线

    query = "怎么减少大模型的幻觉？"                    # 要问的问题
    results = search(query, documents, 3)          # 调自己写的 search

    print(f"问题：{query}")                          # 打印问题
    for i, doc in enumerate(results, 1):           # enumerate 同时给序号（从 1 开始）和内容
        print(f"  {i}. {doc}")                     # 打印序号和原文
    print()                                        # 空一行

    # 断言：结果里必须有一条跟幻觉相关，且正好 3 条
    assert any("幻觉" in doc for doc in results), "最相关的里面应该出现幻觉相关的文档"   # 检查结果里有幻觉
    assert len(results) == 3                        # 检查正好 3 条
    print("D6 任务通过 ✅")                          # 全部通过
    print()                                         # 空一行


# ══════════════════════════════════════════════════════════
# 主入口：运行这个文件时，按顺序执行所有演示
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":                          # 只有直接运行这个文件时才执行
    demo1_handwritten_cosine()                     # 跑演示 1：手写余弦
    demo2_real_vectors()                           # 跑演示 2：真实向量
    demo3_faiss_search()                           # 跑演示 3：FAISS 检索
    demo4_custom_search()                          # 跑演示 4：自己写检索

    print("=" * 62)                                # 打印分隔线
    print("全部跑完 ✅")                             # 打印完成
    print("=" * 62)                                # 打印分隔线
    print("下一步建议：")                            # 打印提示标题
    print("  · 把 demo2 里的 4 句话换成你自己想的句子，看相似度怎么变")   # 建议 1
    print("  · 在 search() 里加个 print，看看 scores 的排序前后长什么样")  # 建议 2
    print("  · 试试点 PyCharm 的 Debug（代码左边点一下设断点），一步步看变量")   # 建议 3
