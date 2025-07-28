import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import cast
import matplotlib.pyplot as plt

"""
本代码使用了SMOTE处理不平衡数据集 -- The Pima Indian Diabetes
并使用几个全连接层组成的线性网络训练，最后的准确率在将近80%
"""

df = pd.read_csv("./data/The Pima Indian Diabetes.csv")

# 数据预处理，将为0的值使用均值填充
zero_of_nan = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[zero_of_nan] = df[zero_of_nan].replace(0, np.nan)
imputer = SimpleImputer(strategy='mean') # imputer是一个填充器，用于处理缺失值
df[zero_of_nan] = imputer.fit_transform(df[zero_of_nan]) # Fill NaN with mean

scaler = StandardScaler() # 标准化
df[zero_of_nan] = scaler.fit_transform(df[zero_of_nan])

X = df.drop('Outcome', axis=1)  # drop函数用于删除参数给的列，返回剩余结果
y = df['Outcome']  # Target variable

# 使用SMOTE过采样处理不平衡的数据集
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y) # type: ignore

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 定义数据集的类
class DiabetesDataset(Dataset):
    def __init__(self, features: Tensor, labels: Tensor):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]

# 创建数据加载器
train_dataset = DiabetesDataset(X_train, y_train)
test_dataset = DiabetesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义神经网络类型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32), # 用于1维的批归一化
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

criterion = nn.BCELoss() # 二分类交叉熵损失，它计算批次的平均损失，返回一个标量
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 40

# 训练
train_loss, train_acc = [], []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features) # 这里outputs的形状通常为(batch_size, 1)
        loss = criterion(outputs.squeeze(), labels) # squeeze函数可以把所有为1的维度去除
        loss = cast(Tensor, loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss.append(running_loss / len(train_loader))
    
    # 计算本轮的准确率
    model.eval()
    with torch.no_grad():
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predicted = (outputs.squeeze() > 0.5).long()  # 将输出转换为0或1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    train_acc.append(acc)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss[-1]:.4f}, Accuracy: {acc:.4f}")
    
# 模型评估
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        predicted = torch.round(outputs).squeeze().long()
        correct += (predicted == y_test).sum().item()
        total += y_test.size(0)

print(f"Test Accuracy: {correct / total:.4f}")

# 数据可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()
plt.tight_layout()
plt.show()
