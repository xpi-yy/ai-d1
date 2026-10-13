"""D1 任务 1：线性回归 —— 亲手看清「训练循环」到底在干什么。

对应验收四问，代码里逐条标了注释：
    forward  → 算出预测值
    backward → 算出每个参数的梯度
    step     → 用梯度更新参数
    zero_grad→ 清空梯度（不清零会累加，参数就更新错了）

运行：  ./\.venv/bin/python 01_linear_regression.py
产出：  loss_linear.png（loss 从大变小的曲线，D1 的交付物）
"""
import matplotlib

matplotlib.use("Agg")  # 无窗口环境也要能存图
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Mac 上 matplotlib 索引不到 "PingFang SC"，用下面这组可用的中文字体栈
plt.rcParams["font.sans-serif"] = ["PingFang HK", "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC", "STHeiti"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用设备:", device)

# ---------- 1. 造假数据：真实规律是 y = 2.5x + 1.2 ----------
TRUE_W, TRUE_B = 2.5, 1.2
torch.manual_seed(42)
N = 200
x = torch.randn(N, 1, device=device)
y = TRUE_W * x + TRUE_B + 0.3 * torch.randn(N, 1, device=device)  # 加点噪声

# ---------- 2. 模型、损失、优化器 ----------
model = nn.Linear(1, 1).to(device)  # 等同于 y = wx + b
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print(f"初始参数   w={model.weight.item():.4f}  b={model.bias.item():.4f}  (真值 2.5 / 1.2)")

# ---------- 3. 训练循环：D1 最核心的 6 行 ----------
EPOCHS = 100
history = []

for epoch in range(EPOCHS):
    pred = model(x)                    # forward：算预测值
    loss = loss_fn(pred, y)            # 算损失
    optimizer.zero_grad()              # 清空上一步的梯度（不写这行会累加！）
    loss.backward()                    # backward：算梯度
    optimizer.step()                   # step：更新参数 w、b

    history.append(loss.item())
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f} | w={model.weight.item():.3f} b={model.bias.item():.3f}")

print(f"训练后参数 w={model.weight.item():.4f}  b={model.bias.item():.4f}  ← 应接近 2.5 / 1.2")

# ---------- 4. 存 loss 曲线（今天的交付物） ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(history, color="#2563eb", linewidth=2)
axes[0].set_title("损失曲线 loss vs epoch")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("MSE loss")
axes[0].grid(alpha=0.3)

with torch.no_grad():
    axes[1].scatter(x.cpu(), y.cpu(), s=12, alpha=0.4, label="真实数据（含噪声）")
    axes[1].plot(x.cpu(), model(x).cpu(), color="#dc2626", linewidth=2, label="模型学到的直线")
axes[1].set_title("拟合结果")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("loss_linear.png", dpi=140)
print("已保存 loss_linear.png ✅")
