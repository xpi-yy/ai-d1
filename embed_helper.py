"""Embedding 工具函数（D6 用）

用法（notebook 里就一行）：
    from embed_helper import embed, cosine_similarity

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
import importlib.machinery            # 用来手工造一个「假模块」的规格（__spec__）
import os                             # 用来读环境变量、拼路径
import sys                            # 用来往 sys.modules 里塞假模块
import types                          # 用来创建模块对象 ModuleType

# ────────────────────────────────────────────────────────────
# OpenMP 冲突修复（必须在 import torch / faiss 之前设置）
# ────────────────────────────────────────────────────────────
# 症状：程序直接崩掉，报
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
# 原因：torch 和 faiss 各自链接了一份 OpenMP 运行时，同时加载就会撞车。
#       macOS 上这是经典问题（Linux 上也偶发）。
# 解法：允许重复加载。对「只做推理、不追求极限性能」的场景完全安全。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 允许重复加载 libomp


def _install_stubs():                       # 私有函数：名字前带 _ 表示"内部用，别直接调"
    """在 import transformers 之前，把损坏的 scipy / 用不到的 sklearn 桩掉。"""

    def make(name, attrs=None, is_package=False):   # 内部小函数：造一个假模块
        if name in sys.modules:                     # 如果这个名字已经存在
            return sys.modules[name]                # 直接拿去用，不重复造
        m = types.ModuleType(name)                  # 新建一个空模块
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)   # 给它一个模块规格
        m.__file__ = os.path.join(os.getcwd(), "_stub_" + name.replace(".", "_") + ".py")       # 给它一个假文件路径
        m.__path__ = [] if is_package else None     # 如果是包，给空路径列表
        for k, v in (attrs or {}).items():          # 把准备好的属性挨个塞进去
            setattr(m, k, v)                        # 设置模块属性
        sys.modules[name] = m                       # 注册到 sys.modules，以后 import 就拿到它
        return m                                    # 返回这个假模块

    def _not_implemented(*args, **kwargs):          # 桩函数：一旦被调用就报错
        raise NotImplementedError("这是 macOS 27 + scipy 兼容问题的桩函数，本项目不使用")   # 明确告诉调用者这里没实现

    make("scipy", is_package=True)                  # 造空 scipy
    make("scipy.optimize", {"linear_sum_assignment": _not_implemented})   # 造 scipy.optimize
    make("scipy.sparse", is_package=True)           # 造空 scipy.sparse
    make("sklearn", is_package=True)                # 造空 sklearn
    make("sklearn.metrics", {"roc_curve": _not_implemented, "auc": _not_implemented})   # 造 sklearn.metrics


_install_stubs()                                    # 调用上面的函数，真正把桩装进去

# 桩装好之后，才能 import transformers
import torch                                                   # PyTorch：张量计算
from transformers import AutoModel, AutoTokenizer               # 加载模型和分词器

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"      # 模型名（中文小模型，512 维）
_tokenizer = None      # 分词器：把句子切成一个个 token（词块）
_model = None          # 模型：真正的 BERT，负责把词块算成向量


def _load():           # 私有函数：只加载一次模型（懒加载，省内存）
    global _tokenizer, _model                  # 声明要用模块级变量
    if _model is None:                         # 如果还没加载
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, local_files_only=True)   # 加载分词器（只用本地文件）
        _model = AutoModel.from_pretrained(_MODEL_NAME, local_files_only=True)           # 加载模型（只用本地文件）
        _model.eval()                          # 切到推理模式（关掉训练相关行为）
    return _tokenizer, _model                  # 返回分词器和模型


def embed(sentences):       # embed = 把句子变成向量（业界叫 embedding / 嵌入）
    """把一组句子变成向量。

    参数：
        sentences —— 一个字符串列表，比如 ["你好", "再见"]
                     也可以直接传单个字符串（内部会自动包成列表）

    返回：
        torch.Tensor，形状 (句子数, 512)
        第二维固定是 512 —— 这是 bge-small-zh 模型的向量维度
    """
    if isinstance(sentences, str):        # 如果传进来的是单个字符串
        sentences = [sentences]           # 就把它包成列表，统一处理

    tokenizer, model = _load()            # 拿到分词器和模型

    inputs = tokenizer(                   # 把句子切成模型要的输入格式
        sentences,                        # 要处理的句子列表
        padding=True,                     # 长短不一时自动补 [PAD] 对齐
        truncation=True,                  # 超过模型最大长度就截断
        return_tensors="pt",              # 返回 PyTorch 张量
    )                                     # 参数写完，调用结束

    with torch.no_grad():                 # 只是推理，不需要算梯度，省内存
        outputs = model(**inputs)         # 把输入喂给模型，得到输出

    # BGE 模型的标准做法：
    #   last_hidden_state 形状是 (句子数, token 数, 512)
    #   [:, 0] 取每个句子第 0 个 token（也就是 [CLS]）的向量
    vectors = outputs.last_hidden_state[:, 0]        # 取每个句子的 [CLS] 向量
    # L2 归一化：把每个向量缩放到长度为 1
    # 归一化之后，两个向量的「点积」就等于「余弦相似度」
    return torch.nn.functional.normalize(vectors, p=2, dim=1)   # 归一化后返回


def cosine_similarity(a, b):          # 余弦相似度：比两个向量有多像
    """两个已归一化的向量，点积就是余弦相似度。

    返回值范围 -1 ~ 1：
        1  = 方向完全相同（意思一样）
        0  = 垂直（没关系）
        -1 = 方向相反（意思相反）
    """
    return float(torch.dot(a, b))     # 点积：两个向量对应相乘再相加，转成普通小数


# ────────────────────────────────────────────────────────────
# 旧名字保留（向后兼容）
# ────────────────────────────────────────────────────────────
# 你之前的 notebook（D6_Embedding与向量检索.ipynb）里还写着
#   from embed_helper import 转向量, 余弦相似度
# 为了让那个 notebook 不改也能继续跑，这里把旧名字也指到新函数上。
# 等你把 notebook 也换成英文后，可以删掉下面两行。
转向量 = embed                        # 旧名 → 新函数
余弦相似度 = cosine_similarity        # 旧名 → 新函数
