# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/21 21:31
Create User : 19410
Desc : xxx
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# https://docs.wandb.ai/
# import wandb
# https://swanlab.cn/
# pip install swanlab
import swanlab
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ===================== 1. 自定义数据集 =====================
class CustomDataset(Dataset):
    """自定义二分类数据集：生成带噪声的二维高斯分布数据"""

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # 生成两类数据：类别0（中心(1,1)）、类别1（中心(-1,-1)）
        np.random.seed(42)  # 固定随机种子，保证结果可复现
        # 类别0数据
        class0 = np.random.normal(loc=[1, 1], scale=0.5, size=(num_samples // 2, 2))
        class0_label = np.zeros((num_samples // 2, 1))
        # 类别1数据
        class1 = np.random.normal(loc=[-1, -1], scale=0.5, size=(num_samples // 2, 2))
        class1_label = np.ones((num_samples // 2, 1))
        # 合并数据并打乱
        self.data = np.vstack([class0, class1]).astype(np.float32)
        self.labels = np.vstack([class0_label, class1_label]).astype(np.float32)
        # 打乱数据
        shuffle_idx = np.random.permutation(num_samples)
        self.data = self.data[shuffle_idx]
        self.labels = self.labels[shuffle_idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])


# ===================== 2. 定义神经网络模型 =====================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 输入层：2个特征 → 隐藏层16个神经元
        self.fc2 = nn.Linear(16, 8)  # 隐藏层
        self.fc3 = nn.Linear(8, 1)  # 输出层：二分类输出1个值（sigmoid后判断）
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 输出0-1之间的概率
        return x


# ===================== 3. 训练配置与初始化 =====================
lr = 0.01
num_epochs = 30

# swanlab登录 todo: 更改成自己的api_key
swanlab.login(api_key="LmjXXEKV9LJ3sXhKvE0Dp")

# 初始化一个SwanLab项目
swanlab.init(
    project="my-pytorch-demo",
    config={
        "lr": lr,
        "num_epochs": num_epochs
    }
)

# 设备配置：优先使用GPU，没有则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化数据集和数据加载器
train_dataset = CustomDataset(num_samples=2000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数、优化器
model = SimpleNN().to(device)
criterion = nn.BCELoss()  # 二分类交叉熵损失（BCELoss）

# optimizer = optim.Adam(model.parameters(), lr=lr)  # 初始学习率0.01
optimizer = optim.AdamW(model.parameters(), lr=lr)  # 初始学习率0.01
# 学习率调度器：StepLR - 每step_size个epoch，学习率乘以gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# 初始化TensorBoard日志记录器
log_dir = f"runs/simple_nn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard日志保存路径: {log_dir}")

# ===================== 4. 训练主循环 =====================
total_steps = len(train_loader)

for epoch in range(num_epochs):
    model.train()  # 模型进入训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        # 将数据移到指定设备（GPU/CPU）
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        # 统计损失和准确率
        running_loss += loss.item()
        # 二分类准确率计算：输出>0.5为类别1，否则为类别0
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 每个epoch结束后更新学习率
    scheduler.step()

    # 计算当前epoch的平均损失和准确率
    epoch_loss = running_loss / total_steps
    epoch_acc = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

    # 打印训练信息
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, LR: {current_lr:.6f}')

    # 写入TensorBoard日志
    writer.add_scalar('Training/Loss', epoch_loss, epoch)  # 损失曲线
    writer.add_scalar('Training/Accuracy', epoch_acc, epoch)  # 准确率曲线
    writer.add_scalar('Training/Learning_Rate', current_lr, epoch)  # 学习率变化
    swanlab.log({
        "Training/epoch_Loss": epoch_loss,
        "Training/epoch_accuracy": epoch_acc,
        "Training/epoch_lr": current_lr,
    }, step=epoch)

# ===================== 5. 收尾工作 =====================
writer.close()  # 关闭TensorBoard写入器
swanlab.finish() # 完成训练
torch.save(model.state_dict(), 'simple_nn_model.pth')  # 保存模型
print("训练完成！模型已保存为 simple_nn_model.pth")

# 可视化数据集（可选）
plt.scatter(train_dataset.data[:, 0], train_dataset.data[:, 1], c=train_dataset.labels[:, 0], cmap='bwr', alpha=0.5)
plt.title('Custom Binary Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('dataset_visualization.png')
plt.show()




