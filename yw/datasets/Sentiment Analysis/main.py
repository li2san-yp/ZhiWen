# coding: utf-8
import os
import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn

"""
    本代码用于训练模型 sentiment_model
"""

df = pd.read_csv("./Sentiment_Analysis.csv")

df.drop("Unnamed: 0",axis=1,inplace=True)
df.dropna(axis=0, inplace=True)

le = LabelEncoder() # sklearn.preprocessing
df["status"] = le.fit_transform(df["status"]) # LabelEncoder可以完成标签的编码，fit_transform是编码并转换标签的函数

# 计算每个类别的权重
# 对于标签的不同类别，占总样本的比例越小，其权重就会越大
class_weights = compute_class_weight( # sklearn.utils.class_weight
    class_weight="balanced",
    classes=np.unique(df["status"]),
    y=df["status"]
)

# 使用正则匹配网址等信息，并剔除
def preprocess(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["statement"] = df["statement"].apply(preprocess)
# stratify参数的作用为：按照传入参数的比例分层抽样，保证测试集和训练集中不同类别所占的比例相同，通常传入和标签一致
X_train, X_test, y_train, y_test = train_test_split(df["statement"], df["status"], test_size = 0.2, stratify=df["status"], random_state=42)

class SentimentDataset(Dataset):
    def __init__(self, labels, tokens):
        super().__init__()
        self.labels = labels
        self.tokens = tokens
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

"""
bert-base-uncased:
    transfromer提供的用于预处理英文数据的模型
    uncased  --  分词前将所有单词转为小写，忽略大小写差异
"""
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

training_tokens = tokenizer(
    X_train.to_list(), # 需要转换的数据
    truncation=True, # 句子过长时截断到max_length
    padding="max_length", # 所有输入填充到max_length长度
    max_length=256, # 设置max_length
    return_tensors="pt" # 返回pytorch的格式
)

# 下面定义训练集和测试集的加载器
train_dataset = SentimentDataset(y_train.astype(int).to_numpy(), training_tokens)

test_tokens = tokenizer(
    X_test.to_list(),
    truncation=True,
    padding="max_length",
    max_length=256,
    return_tensors="pt"
)
test_dataset = SentimentDataset(y_test.astype(int).to_numpy(), test_tokens)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    pin_memory=True, # 将数据保存在锁页内存中，提高使用GPU时的效率
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    pin_memory=True
)

# 定义神经网络
class Net(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased",num_classes=7, dropout=0.3, hidden_dims=128):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size = self.bert.config.hidden_size,
            hidden_size=hidden_dims,
            bidirectional=True,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dims*2, num_classes)
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state
        
        intermediate_hidden_outs, (final_hissen_state, call_state) = self.lstm(embedding)
        hidden = torch.cat((final_hissen_state[-2], final_hissen_state[-1]), dim=1)
        
        out = self.dropout(hidden)
        
        logits = self.linear(out)
        return logits

class tmpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 32),
            nn.ReLU(),
            
            nn.Linear(32, 7),
            nn.Softmax()
            
        )
        
    def forward(self, x, attention_mask):
        return self.layers(x)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

epochs = 5
model = Net().to(device) # 这里使用临时的线性神经网络，后面更换

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:

            features = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].long().to(device)

            optimizer.zero_grad()
            outputs = model(features, attention_mask)
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"epoch: [{epoch}/{epochs}], loss: {avg_loss}")

def save():
    torch.save(model.state_dict() ,"./sentiment_mdoel.pth")
    print("model saved.")

if __name__ == "__main__":
    train()
    save()
