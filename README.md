# ai-d1 · 大模型转型 19 天计划的第 1 天

> D1 主题：环境与 PyTorch 手感。目标是**跑通**，不是理解全部理论。

## 环境

- 机器：Apple 芯片（arm64），macOS 26.5.2
- Python：3.10.11（arm64 原生，非 Rosetta）→ **MPS 可用**
- 环境：`./.venv`（本项目独立虚拟环境，不污染系统 Python）

## 三个脚本，按顺序跑

```bash
cd ~/ai-d1
./.venv/bin/python 00_verify_env.py        # 1. 自检：确认 MPS 能用、反向传播能跑
./.venv/bin/python 01_linear_regression.py # 2. 线性回归：看清训练循环那 6 行
./.venv/bin/python 02_mnist.py             # 3. MNIST：第一个真正的神经网络
```

第一次跑 `02_mnist.py` 会自动下载 MNIST 数据集（约 11 MB）到 `./data`。

## 产出（D1 的交付物）

| 文件 | 说明 |
|---|---|
| `loss_linear.png` | 线性回归的 loss 曲线 + 拟合结果 |
| `loss_mnist.png` | MNIST 训练 loss 与测试准确率曲线 |
| `samples.png` | 8 张测试图的预测结果对照 |

## 想用 Jupyter 的话

```bash
./.venv/bin/jupyter lab
```

浏览器打开后新建 notebook，把上面脚本的内容分 cell 敲一遍——**手敲比复制粘贴有效**，
敲的过程里你才会注意到 `zero_grad()` 的位置这种东西。

## 验收四问（能脱稿讲出来就算过关）

1. `forward` 在干什么？—— 前向传播，算出预测值
2. `loss.backward()` 在干什么？—— 反向传播，算出每个参数的梯度
3. `optimizer.step()` 在干什么？—— 用梯度更新参数
4. `optimizer.zero_grad()` 为什么必须调？—— 梯度默认会累加，不清零参数就更新错了

## 出问题时的降级

一个报错卡满 2 小时，直接改用 Google Colab 免费 GPU 跑，本地环境留到周末修，不要耗在配置上。
