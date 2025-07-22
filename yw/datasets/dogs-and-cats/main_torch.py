import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL.Image import Image as IMG
from torch import nn, optim
from typing import cast
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
    这段代码使用pytorch实现了一个卷积神经网络的实例，数据集使用dog-vs-cat。
    torch的代码量要比keras多，但是总是要转到torch写代码的，keras能用作过渡。
    前面的代码如果有时间的话会补上torch的版本。
"""

torch.manual_seed(42)
data_dir = "G:\\datasets\\dogs_and_cat_dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")
test_dir = os.path.join(data_dir, "test")

# transform模块用于数据预处理和数据增强
# 训练集的处理
train_transform = transforms.Compose([
    transforms.Resize((150, 150)), # 将图像大小缩放为150
    transforms.RandomHorizontalFlip(), # 以50%的概率水平翻转图像，数据增强
    transforms.RandomRotation(15), # 随机旋转图像，在-15°到+15°之间
    transforms.ToTensor(), # 将图片转换为张量，注意它会自动把数值调整到[0.0, 1.0]，并且把维度顺序调整为通道优先
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 第一个是平均值，第二个是标准差，将数据标准化，这里传入的参数会把范围映射到[-1.0,1.0]
    # 推导如下： 原先的范围：[0.0, 1.0]  减去平均值(0.5)除以标准差(0.5) -> [-1.0, 1.0]
])

# 测试集和验证集的处理，不需要数据增强
val_test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# 自定义数据集的类
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["cat", "dog"]
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 数据集
train_dataset = CatDogDataset(train_dir, train_transform)
val_dataset = CatDogDataset(val_dir, val_test_transform)
test_dataset = CatDogDataset(test_dir, val_test_transform)

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size)
test_loader = DataLoader(test_dataset, batch_size = batch_size)

class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding = 1), # 输出32*150*150
            nn.ReLU(), # 记得加上激活函数
            nn.MaxPool2d(kernel_size=2, stride = 2), # 最大池化层，输出32*75*75
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 输出64*75*75
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出64*37*37
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 输出128*37*37
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 输出128*18*18
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 输出256*18*18
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 输出256*9*9
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*9*9, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # 正则化，防止过拟合
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
model = CatDogCNN().to("cuda")

loss = nn.BCELoss() # 即二元交叉熵损失， binary cross entropy loss
optimizer = optim.Adam(model.parameters(), lr = 0.001) # adam是rmsprop引入动量的版本

# 训练函数
def train_model(model: CatDogCNN, loss: nn.BCELoss, optimizer: optim.Adam, num_epochs = 15):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train() # 开启训练模式
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # 这里处理输入和标签，将它们放到显存
            inputs = cast(torch.Tensor, inputs)
            labels = cast(torch.Tensor, labels)
            inputs = inputs.to("cuda")
            labels = labels.float().unsqueeze(1).to("cuda") # unsqueeze的作用是在指定dim上增加一个维度，这里把标签转为列向量
            
            # 将原先的梯度清零，计算当前模型下输入内容得到的输出，计算和标签值之间的损失，调整模型参数
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels) # type: torch.Tensor
            l.backward()
            optimizer.step()
            
            running_loss += l.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # 验证
        model.eval() # 开始评估模式
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = cast(torch.Tensor, inputs)
                labels = cast(torch.Tensor, labels)
                
                inputs = inputs.to("cuda")
                labels = labels.float().unsqueeze(1).to("cuda")
                
                outputs = model(inputs) # 这里计算损失后直接记录即可
                l = loss(outputs, labels)
                val_loss += l.item() * inputs.size(0)
                
                preds = (outputs > 0.5).float() # type: torch.Tensor
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)
        val_acc = accuracy_score(val_true, val_preds)
        val_accuracies.append(val_acc)
        
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss: {epoch_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f} | "
        )
    
    return train_losses, val_losses, val_accuracies

print("start training...")
train_losses, val_losses, val_accuracies = train_model(model, loss, optimizer, num_epochs= 40)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label = "train_loss")
plt.plot(val_losses, label="val_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label = "val acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# 模型评估
def evaluate(model: nn.Module, test_loader):
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to("cuda")
            labels = labels.float().unsqueeze(1).to("cuda")
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
        
    test_acc = accuracy_score(test_true, test_preds)
    print(f"test accuracy: {test_acc:.4f}")

print("\nevaluating on the test set...")
evaluate(model, test_loader)

# 保存模型
torch.save(model.state_dict(), "cat_and_dog.pth")
print("model saved to cat_and_dog.pth")
