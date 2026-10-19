"""Embedding 工具函数（D6 用）

用法（notebook 里就一行）：
    from embed_helper import 转向量, 余弦相似度

────────────────────────────────────────────────────────────
为什么文件开头有一段"桩模块"代码？
────────────────────────────────────────────────────────────
2026/10/16 左右系统升级到 macOS 27.0.1 之后，scipy 1.15.3 的二进制
（_spropack.cpython-310-darwin.so）在 macOS 27 上加载失败，报：

    dlopen(...): section '__DATA/__thread_bss' has a zero-fill section type,
    but offset field is not zero

而能修好这个问题的 scipy 1.16+ 要求 Python >= 3.11，本项目是 Python 3.10
—— 所以「升级 scipy」这条路走不通。

但问题是：transformers 5.20 在 import 时会 **间接** 依赖 scipy 和 sklearn：
    transformers → modeling_utils → loss → loss_for_object_detection → scipy.optimize
    transformers → generation → candidate_generator → sklearn.metrics → scipy.sparse

我们只做 Embedding，一个 scipy 函数都用不到。
所以这里在 import transformers 之前，先往 sys.modules 里塞几个空的桩模块，
把这条无关的依赖链骗过去。真正的计算（分词、BERT 前向、归一化）全部由
transformers + torch 完成，不受影响。

等以后升级到 Python 3.11+ 并装上正常的 scipy，就可以删掉这段桩代码。
"""
import importlib.machinery
import os
import sys
import types

# ────────────────────────────────────────────────────────────
# OpenMP 冲突修复（必须在 import torch / faiss 之前设置）
# ────────────────────────────────────────────────────────────
# 症状：程序直接崩掉，报
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
# 原因：torch 和 faiss 各自链接了一份 OpenMP 运行时，同时加载就会撞车。
#       macOS 上这是经典问题（Linux 上也偶发）。
# 解法：允许重复加载。对「只做推理、不追求极限性能」的场景完全安全。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _装桩模块():
    """在 import transformers 之前，把损坏的 scipy / 用不到的 sklearn 桩掉。"""

    def 造(name, 属性=None, 是包=False):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=是包)
        m.__file__ = os.path.join(os.getcwd(), "_stub_" + name.replace(".", "_") + ".py")
        m.__path__ = [] if 是包 else None
        for k, v in (属性 or {}).items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    def _未实现(*a, **k):
        raise NotImplementedError("这是 macOS 27 + scipy 兼容问题的桩函数，本项目不使用")

    造("scipy", 是包=True)
    造("scipy.optimize", {"linear_sum_assignment": _未实现})
    造("scipy.sparse", 是包=True)
    造("sklearn", 是包=True)
    造("sklearn.metrics", {"roc_curve": _未实现, "auc": _未实现})


_装桩模块()

# 桩装好之后，才能 import transformers
import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_分词器 = None
_模型 = None


def _加载():
    global _分词器, _模型
    if _模型 is None:
        _分词器 = AutoTokenizer.from_pretrained(_MODEL_NAME, local_files_only=True)
        _模型 = AutoModel.from_pretrained(_MODEL_NAME, local_files_only=True)
        _模型.eval()
    return _分词器, _模型


def 转向量(句子列表):
    """把一组句子变成向量。

    参数：
        句子列表 —— 一个字符串列表，比如 ["你好", "再见"]
                    也可以直接传单个字符串（内部会自动包成列表）

    返回：
        torch.Tensor，形状 (句子数, 512)
        第二维固定是 512 —— 这是 bge-small-zh 模型的向量维度
    """
    if isinstance(句子列表, str):
        句子列表 = [句子列表]

    分词器, 模型 = _加载()

    输入 = 分词器(
        句子列表,
        padding=True,          # 长短不一时自动补 [PAD] 对齐
        truncation=True,       # 超过模型最大长度就截断
        return_tensors="pt",   # 返回 PyTorch 张量
    )

    with torch.no_grad():      # 只是推理，不需要算梯度，省内存
        输出 = 模型(**输入)

    # BGE 模型的标准做法：
    #   last_hidden_state 形状是 (句子数, token 数, 512)
    #   [:, 0] 取每个句子第 0 个 token（也就是 [CLS]）的向量
    向量 = 输出.last_hidden_state[:, 0]
    # L2 归一化：把每个向量缩放到长度为 1
    # 归一化之后，两个向量的「点积」就等于「余弦相似度」
    return torch.nn.functional.normalize(向量, p=2, dim=1)


def 余弦相似度(a, b):
    """两个已归一化的向量，点积就是余弦相似度。

    返回值范围 -1 ~ 1：
        1  = 方向完全相同（意思一样）
        0  = 垂直（没关系）
        -1 = 方向相反（意思相反）
    """
    return float(torch.dot(a, b))
