import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import re
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

"""
    本代码用于评估训练好的模型，多数代码和main.py中一致，关注两个函数：evaluate和func
"""

df = pd.read_csv("./Sentiment_Analysis.csv")

df.drop("Unnamed: 0", axis=1,inplace=True)
df.dropna(axis=0,inplace=True)

le = LabelEncoder()
df["status"] = le.fit_transform(df["status"])

def preprocess(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["statement"] = df["statement"].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df["statement"], df["status"], test_size=0.2, stratify=df["status"], random_state=42)

from main import SentimentDataset
from main import Net

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
test_tokens = tokenizer(
    X_test.to_list(),
    truncation=True,
    padding="max_length",
    max_length=256,
    return_tensors="pt"
)
test_dataset = SentimentDataset(y_test.astype(int).to_numpy(), test_tokens)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, class_weights):
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights)

    
    correct = 0
    val_loss = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device) 

            logits = model(input_ids,attention_mask)
            loss = criterion(logits,labels)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total
    print(f'avg_val_loss : {avg_val_loss}, accuracy : {accuracy}')

def func(model):
    from sklearn.metrics import classification_report

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=[
        "Normal", "Depression", "Suicidal", "Anxiety", "Bipolar", "Stress", "Personality Disorder"
    ]))

if __name__ == "__main__":
    model = Net().to(device)
    model.load_state_dict(torch.load("./sentiment_mdoel.pth"))
    model.eval()
    
    class_weights = compute_class_weight( # sklearn.utils.class_weight
        class_weight="balanced",
        classes=np.unique(df["status"]),
        y=df["status"]
    )
    evaluate(model, class_weights)
    func(model)