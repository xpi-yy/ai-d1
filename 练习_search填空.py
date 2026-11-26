"""练习第 1 关：把 search() 里的 6 个空填上。

每个空对应你刚学的一个点，答案都在下面的「单词表」里。
填完运行，看到「填空练习通过 ✅」就过关。
"""
from embed_helper import embed, cosine_similarity

documents = [
    "今天天气很好，适合出门散步",
    "RAG 的核心是先检索再生成",
    "火锅是四川的特色美食",
    "检索增强生成可以减少模型幻觉",
    "明天可能下雨，记得带伞",
    "向量数据库用来做相似度搜索",
    "张三是公司的销售经理",
    "FAISS 是 Facebook 开源的向量检索库",
]


def search(query, docs, k=3):
    # 空 1：把问题变成向量，函数叫什么名字？
    query_vector = ____([query])[0]

    # 空 2：把所有文档变成矩阵，和空 1 是同一个函数
    doc_vectors = ____(docs)

    # 空 3：循环 i 从 0 到「篇数-1」，用什么关键字？
    scores = []
    for i in ____(len(docs)):
        # 空 4：这里要比「问题向量」和「第 i 篇」，两个空分别填哪个变量？
        score = cosine_similarity(____, ____[i])
        scores.append((score, i))

    # 空 5：从大到小排，reverse 填 True 还是 False？
    scores = sorted(scores, key=lambda pair: pair[0], reverse=____)

    # 空 6：用下标换回正文，pair 里第几个是下标？（0 还是 1？）
    results = []
    for pair in scores[:k]:
        results.append(docs[pair[____]])

    return results


if __name__ == "__main__":
    result = search("怎么减少大模型的幻觉？", documents, 3)
    print(result)
    assert any("幻觉" in d for d in result), "没找到幻觉相关的文档，检查空 1~4"
    assert len(result) == 3, "返回数量不对，检查空 5~6"
    print("填空练习通过 ✅")
