"""D1 环境自检：确认 Python / PyTorch / MPS 都正常。

运行：  ./\.venv/bin/python 00_verify_env.py
"""
import platform
import sys

import torch

print("=" * 52)
print("Python      :", sys.version.split()[0])
print("可执行文件  :", sys.executable)
print("机器架构    :", platform.machine())
print("PyTorch     :", torch.__version__)
print("=" * 52)

mps_ok = torch.backends.mps.is_available()
print("MPS 可用    :", mps_ok)
print("MPS 已构建  :", torch.backends.mps.is_built())

if mps_ok:
    device = torch.device("mps")
    print("使用设备    : mps (Apple 芯片 GPU)")
else:
    device = torch.device("cpu")
    print("使用设备    : cpu")
    print("提示：若你用的是 Python 3.9，请重建 venv —— MPS 需要 Python 3.9+ 且 arm64 原生")

# 真实地在目标设备上建张量并做一次运算，而不是只信 is_available()
a = torch.randn(3, 3, device=device)
b = torch.randn(3, 3, device=device)
c = (a @ b).sum()
print("设备上矩阵乘法 OK, 结果 =", float(c))

# 检验反向传播能否在 MPS 上跑通（D1 后面所有训练都靠它）
x = torch.randn(5, 1, device=device)
w = torch.randn(1, 1, device=device, requires_grad=True)
loss = ((x @ w) ** 2).mean()
loss.backward()
print("MPS 反向传播 OK, w.grad 形状 =", tuple(w.grad.shape))

print("=" * 52)
print("环境自检通过 ✅  可以开始 D1 训练任务")
