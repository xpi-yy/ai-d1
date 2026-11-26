"""练习：RAG 全流程填空（D9）

填完 5 个空，运行，看到「练习通过 ✅」就过关。

不用你写的部分（已经写好了，直接 import）：
    retrieve.py 里的 load_text / split_into_chunks / build_index / search_chunks
    llm.py 里的 chat
"""
from pathlib import Path                      # 拼路径

from llm import chat                          # 调大模型
from retrieve import load_text, split_into_chunks, build_index, search_chunks   # D8 的检索


def build_prompt(question, results):          # 定义函数：拼提示词
    """把问题和检索到的资料拼成一段提示词。"""
    context = ""                              # 用来拼所有资料
    for i, (chunk, _dist) in enumerate(results, 1):   # 第 1 段、第 2 段……
        # 空 1：给第 i 段资料编上号，格式是  [资料i] 原文 + 两个换行
        context += f"[资料{i}] {chunk}\n\n"                    # 提示：f"[资料{i}] {chunk}\n\n"
    return (                                  # 返回拼好的提示词
        "你是知识库问答助手，只回答资料里有的内容。\n"   # 角色
        "规则：资料里没有的内容，直接回答「资料中没有提到」，不要编造。\n"   # 边界（不用改）
        f"\n=== 资料开始 ===\n{context}=== 资料结束 ===\n"   # 把资料包起来
        f"\n用户问题：{question}\n"                        # 最后放问题
    )


def ask(question, chunks, index, k=3):        # 定义函数：完整 RAG
    """检索 → 拼提示词 → 调模型。"""
    # 空 2：先检索，要几个结果就传几个（提示：就是参数 k）
    results = search_chunks(question, chunks, index, k)
    # 空 3：把问题和检索结果拼成提示词（调哪个函数？）
    prompt = build_prompt(question, results)
    # 空 4：把提示词交给模型（调哪个函数？注意 chat 要一个消息列表）
    answer = chat([{"role": "user", "content": prompt}])
    return answer, results                    # 返回答案和检索结果


if __name__ == "__main__":                    # 直接运行这个文件时
    text = load_text(Path(__file__).parent / "sample" / "RAG入门.txt")   # 读文档
    chunks = split_into_chunks(text)          # 切成 23 段
    index, _ = build_index(chunks)            # 建库

    # 空 5：问一个「资料里有的」问题（提示：资料里讲过 RAG 是什么）
    answer, results = ask("RAG是什么", chunks, index)
    print("答案：", answer)                    # 打印答案
    assert answer and ("检索" in answer or "RAG" in answer), "没答出来，检查空 2~5"   # 必须答出来
    print("练习通过 ✅")                        # 过关
