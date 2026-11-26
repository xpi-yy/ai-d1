"""D9 + D10：RAG 全链路 + 引用溯源

D9 做了什么：检索 → 拼提示词 → 生成
D10 加了什么：程序解析答案里的 [来源N]，替换成「文件名 + 第几段 + 原文」，并检查有没有编造来源

用法：
    ./.venv/bin/python rag.py "RAG 是什么？"     # 问一个问题
    ./.venv/bin/python rag.py                    # 不给问题 → 跑 10 题验收
"""
import re                                    # 正则：找 [来源N]
import sys                                   # 拿命令行参数
from pathlib import Path                     # 拼路径

from llm import chat                         # 调模型
from retrieve import load_text, split_into_chunks, build_index, search_chunks   # 复用 D8 的检索

SOURCE_NAME = "RAG入门.txt"                  # 资料来源文件名（溯源要写出来）


def build_prompt(question, results):         # 定义函数：拼提示词
    """把问题和检索到的资料拼成提示词。"""
    context = ""                             # 用来拼所有资料
    for i, (chunk, _dist) in enumerate(results, 1):   # 第 1 段、第 2 段……
        context += f"[资料{i}] {chunk}\n\n"   # 给每段编上号
    return (                                 # 返回拼好的提示词
        "你是知识库问答助手，只回答资料里有的内容。\n"   # 角色设定
        "规则：\n"                                       # 规则标题
        "1. 只依据资料回答；资料里没有的内容，直接回答「资料中没有提到」，不要编造。\n"   # 规则 1：不许编
        "2. 回答末尾用 [来源1][来源2] 的格式标注你依据了哪几段资料。\n"                  # 规则 2：标来源
        f"\n=== 资料开始 ===\n{context}=== 资料结束 ===\n"   # 把资料包起来
        f"\n用户问题：{question}\n"                          # 最后放问题
    )


def ask(question, chunks, index, k=3):       # 定义函数：完整 RAG
    """检索 → 拼提示词 → 调模型。"""
    results = search_chunks(question, chunks, index, k)   # ① 检索：找最像的 3 段
    prompt = build_prompt(question, results)               # ② 拼提示词
    answer = chat([{"role": "user", "content": prompt}])   # ③ 调模型
    return answer, results                                 # 返回答案和检索结果


def format_source(n, chunk, source_name):    # 定义函数：格式化一条来源
    """把第 n 段格式化成「[来源n]《文件名》第 n 段：原文…」。"""
    preview = chunk.replace("\n", " ")[:50]   # 换行变空格，取前 50 字
    return f"[来源{n}]《{source_name}》第 {n} 段：{preview}…"   # 拼成一行


def resolve_citations(answer, results, source_name):   # 定义函数：溯源核心
    """解析答案里的 [来源N]，返回 (引用编号列表, 无效编号, 参考资料列表)。"""
    cited = re.findall(r"\[来源(\d+)\]", answer)      # 找出答案里所有 [来源N]
    cited = [int(n) for n in cited]                    # 转成数字
    invalid = [n for n in cited if not (1 <= n <= len(results))]   # 引用了不存在的来源
    refs = [format_source(i, chunk, source_name)      # 逐段生成参考资料
            for i, (chunk, _dist) in enumerate(results, 1)]   # 从 1 开始编号
    return cited, invalid, refs                        # 返回三样东西


def print_answer(question, answer, results):  # 定义函数：打印结果
    """打印：问题、答案、解析出的引用、真实参考资料。"""
    print(f"\n问题：{question}")                    # 打印问题
    print(f"答案：{answer}")                        # 打印模型答案
    cited, invalid, refs = resolve_citations(answer, results, SOURCE_NAME)   # 解析引用
    print(f"\n引用到的来源：{[f'来源{n}' for n in cited] or '（没有标）'}")   # 打印引用了哪些
    print("真实参考资料：")                          # 打印标题
    for ref in refs:                                 # 逐条打印
        print(f"  {ref}")                            # 打印一条参考资料
    if invalid:                                      # 如果引用了不存在的来源
        print(f"  ⚠️ 发现编造来源：{[f'来源{n}' for n in invalid]}")   # 警告


TESTS = [                                    # 10 个测试问题
    ("RAG 是什么？", "in"),                   # 第 1 题：资料内
    ("为什么要让相邻两段重叠？", "in"),        # 第 2 题：资料内
    ("怎么减少幻觉？", "in"),                  # 第 3 题：资料内
    ("模型怎么调用函数？", "in"),              # 第 4 题：资料内
    ("微调和 RAG 怎么选？", "in"),             # 第 5 题：资料内
    ("Transformer 的注意力机制在做什么？", "in"),   # 第 6 题：资料内
    ("为什么要多头注意力？", "in"),             # 第 7 题：资料内
    ("量化是用什么换什么？", "in"),             # 第 8 题：资料内
    ("今天天气怎么样？", "out"),               # 第 9 题：资料外，应拒答
    ("皮政凯是谁？", "out"),                   # 第 10 题：资料外，应拒答
]


def run_acceptance_test(chunks, index):      # 定义函数：跑 10 题验收
    print("=" * 62)                          # 打印分隔线
    print("D10 验收：10 个问题（8 个资料内 + 2 个资料外）")   # 打印标题
    print("=" * 62)                          # 打印分隔线
    passed = 0                               # 通过数，从 0 开始
    for question, kind in TESTS:             # 一题一题地测
        answer, results = ask(question, chunks, index)   # 跑一次 RAG
        if answer is None:                   # 如果 API 失败
            print(f"❌ {question}：API 失败")            # 标记失败
            continue                          # 跳过这题
        cited, invalid, _refs = resolve_citations(answer, results, SOURCE_NAME)   # 解析引用
        if kind == "in":                     # 资料内的问题
            ok = bool(cited) and not invalid   # 必须有引用、且不能编造来源
            tag = "✅ 有引用" if ok else ("❌ 没标来源" if not cited else "❌ 编造来源")
        else:                                # 资料外的问题
            refuse_words = ["没有", "不知道", "未提及", "无法", "不包含", "不相关", "资料中未"]   # 拒答特征词
            ok = any(w in answer for w in refuse_words)   # 出现任一特征词就判拒答
            tag = "✅ 正确拒答" if ok else "❌ 瞎编了"
        if ok:                               # 如果这题通过
            passed += 1                      # 通过数加 1
        print(f"{tag}  {question}")          # 打印判定
        print(f"    答案：{answer[:50]}")     # 打印答案前 50 字
        print(f"    引用：{[f'来源{n}' for n in cited] or '无'}")   # 打印引用
        print()                              # 空一行
    print(f"结果：{passed}/{len(TESTS)} 通过")          # 打印总成绩（用 len 自动数，不写死）
    assert passed >= len(TESTS), f"只通过 {passed}/{len(TESTS)}"   # 要求全部通过
    print("D10 验收通过 ✅")                   # 打印通过


if __name__ == "__main__":                   # 只有直接运行时才执行
    text = load_text(Path(__file__).parent / "sample" / "RAG入门.txt")   # 读入文档
    chunks = split_into_chunks(text)         # 切成 23 段
    print(f"读入 {len(text)} 字 → 切成 {len(chunks)} 段")   # 打印切分结果
    index, _ = build_index(chunks)           # 建库
    if len(sys.argv) > 1:                    # 如果命令行给了问题
        question = sys.argv[1]               # 取出问题
        answer, results = ask(question, chunks, index)   # 跑一次 RAG
        print_answer(question, answer, results)          # 打印结果
    else:                                    # 如果没给问题
        run_acceptance_test(chunks, index)   # 跑 10 题验收
