import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np

# 데이터 로드 및 전처리
data = pd.read_csv("")

print("shape:", data.shape)
print("전체 결측값 수:\n", data.isnull().sum())

# Outcome을 제외한 BMI를 타겟으로 사용
X = data.drop(['@@'], axis=1).values  # 'Outcome'과 'BMI' 제외
y = data['@@'].values.astype(np.float32)  # 'BMI'를 회귀 타겟으로 설정

scaler = StandardScaler()
X = scaler.fit_transform(X)

# float 32 @@@@@@@@@@
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_test])
y_train, y_test = map(lambda y: torch.tensor(y, dtype=torch.float32), [y_train, y_test])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 회귀는 1개 출력

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 인자 (특징)
input_dim = X_train.shape[1]  
model = MLPRegressor(input_dim)
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
