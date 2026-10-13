"""D1 任务 2：MNIST 手写数字 —— 第一个真正意义上的神经网络训练。

比线性回归多了三件事，刻意都写出来给你看：
    1. Dataset / DataLoader  —— 数据怎么一批一批喂进去
    2. nn.Module 自定义模型   —— 自己定义网络结构
    3. 训练 + 验证双循环       —— 训练集算梯度，验证集只看不学

运行：  ./\.venv/bin/python 02_mnist.py
产出：  loss_mnist.png（loss 下降曲线）、samples.png（预测样例）
注意：  第一次运行会自动下载 MNIST 数据集（约 11 MB）到 ./data
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Mac 上 matplotlib 索引不到 "PingFang SC"，用下面这组可用的中文字体栈
plt.rcParams["font.sans-serif"] = ["PingFang HK", "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC", "STHeiti"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用设备:", device)

# ---------- 1. 数据：Dataset + DataLoader ----------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Mac 上 num_workers 用 0，多进程在 MPS 下容易出怪问题
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
print(f"训练集 {len(train_set)} 张 | 测试集 {len(test_set)} 张")


# ---------- 2. 模型：nn.Module 自定义 ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 个数字类别
        )

    def forward(self, x):
        return self.net(x)


model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- 3. 训练 + 验证双循环 ----------
EPOCHS = 3
train_losses, test_accs = [], []

for epoch in range(1, EPOCHS + 1):
    # 训练：会更新参数
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_set)

    # 验证：不更新参数，所以包 no_grad
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
    acc = correct / len(test_set)

    train_losses.append(train_loss)
    test_accs.append(acc)
    print(f"epoch {epoch} | train loss {train_loss:.4f} | test acc {acc * 100:.2f}%")

# ---------- 4. 存图（今天的交付物） ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

axes[0].plot(range(1, EPOCHS + 1), train_losses, marker="o", color="#2563eb", linewidth=2)
axes[0].set_title("训练损失 loss vs epoch")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("CrossEntropy loss")
axes[0].grid(alpha=0.3)

axes[1].plot(range(1, EPOCHS + 1), [a * 100 for a in test_accs], marker="s", color="#16a34a", linewidth=2)
axes[1].set_title("测试准确率 acc vs epoch")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy (%)")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("loss_mnist.png", dpi=140)

# 抽 8 张测试图看预测对不对
model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    preds = model(xb.to(device)).argmax(1).cpu()

fig2, ax2 = plt.subplots(2, 4, figsize=(9, 4.6))
for i, ax in enumerate(ax2.ravel()):
    ax.imshow(xb[i].squeeze(), cmap="gray")
    ok = preds[i].item() == yb[i].item()
    ax.set_title(f"预测 {preds[i].item()} / 真值 {yb[i].item()}", color="#16a34a" if ok else "#dc2626", fontsize=10)
    ax.axis("off")
fig2.suptitle("MNIST 预测样例", fontsize=13)
fig2.tight_layout()
fig2.savefig("samples.png", dpi=140)

print("已保存 loss_mnist.png ✅  samples.png ✅")
