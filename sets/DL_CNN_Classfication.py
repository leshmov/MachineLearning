import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np

# 데이터 로드 및 전처리
data = pd.read_csv("")

print("before")
print("shape:", data.shape)
print(" '@@' 분포 :")
print(data['@@'].value_counts())
print("전체 결측값 수:\n", data.isnull().sum())

# 문자열 -> 정수수
label_encoders = {}
for col in data.columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

print("after")
print("shape:", data.shape)
print("'@@' 분포 :")
print(data['@@'].value_counts())
print("\X feature shape:", data.drop('@@', axis=1).shape)
print("y label shape:", data['@@'].shape)

# Pandas -> numpy
X = data.drop('@@', axis=1).values
y = data['@@'].values.astype(np.float32) 

scaler = StandardScaler()
X = scaler.fit_transform(X)

# xtrain, xtest = float32, ytrain, ytest = long
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_test])
y_train, y_test = map(lambda y: torch.tensor(y, dtype=torch.long), [y_train, y_test])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        # 레이어 늘리기 
        # self.fc3 = nn.Linear(32, 64)
        # self.fc4 = nn.Linear(64, 128)
        # self.fc5 = nn.Linear(128, 64)
        # self.fc6 = nn.Linear(64, 32)
        # 출력층도 fc 7 로 바꿔야됨됨
        self.fc3 = nn.Linear(32, output_dim)  # output_dim = 클래스 수

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # 늘린만큼 늘려주기 
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        # x = torch.relu(self.fc6(x))
        return self.fc3(x)  # CrossEntropyLoss 쓸 거라 softmax X

# 인자 (특징)

output_dim = len(torch.unique(y_train))  # y_train의 고유값 개수
input_dim = X_train.shape[1]  
model = MLPClassifier(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 100
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    # 평가
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.numpy())
            labels.extend(yb.numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    print(f"Epoch {epoch+1}, Acc: {acc:.4f}, F1: {f1:.4f}")
