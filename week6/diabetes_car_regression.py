import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np

# 데이터 로드 
data = pd.read_csv("C:/4-1/ML/week6/diabetes.csv")

print("shape:", data.shape)
print("전체 결측값 수:\n", data.isnull().sum())

X = data.drop(columns=["BMI", "Outcome"]).values
y = data["BMI"].values.astype(np.float32).reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)


def split_sequences(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences) - n_steps + 1):
        seq_x = sequences[i:i+n_steps, :-1]
        seq_y = sequences[i+n_steps-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

data_array = np.hstack((X, y)) 
X_seq, y_seq = split_sequences(data_array, n_steps=5)

print(X_seq.shape) 
# 예: (1724, 5, 8) # 입력층에 넣을 특징수확인인
# CNN 은 seq_len 필요
# 그래서 X.shape = (샘플 수, 피처 수) -> (batch_size, channels, seq_len)
# 변경 필요 
# input_channels = X_seq.shape[2]  # 8 (특성 수)
# seq_len = X_seq.shape[1]         # 5 (시퀀스 길이)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


class CNNRegressor(nn.Module):
    def __init__(self, input_channels, seq_len):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * seq_len, 64)
        self.fc2 = nn.Linear(64, 1)  # 회귀는 1개 출력

    def forward(self, x):  # (batch, channels, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    

# 인자 추출후 model에 넣어주기 
input_channels = X_train.shape[1]
seq_len = X_train.shape[2]
output_dim = len(torch.unique(y_train))

model = CNNRegressor(input_channels, seq_len)
# 하나의 숫차를 예측하는것이므로 output_dim 필요없음 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
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
            out = model(xb).squeeze()
            preds.extend(out.numpy())
            labels.extend(yb.squeeze().numpy())
    mse = np.mean((np.array(labels) - np.array(preds)) ** 2)
    print(f"Epoch {epoch+1}, MSE: {mse:.4f}")
