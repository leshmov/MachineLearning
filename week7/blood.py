import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import pandas as pd
import numpy as np

# 데이터 로드, 확인 
data = pd.read_csv("C:/4-1/ML/week7/BP_data.csv")

print("before")
print("shape:", data.shape)
print("BP_abnormal 분포 :")
print(data['Blood_Pressure_Abnormality'].value_counts())
print("전체 결측값 수:\n", data.isnull().sum())


# 결측치 채우기 
data['Genetic_Pedigree_Coefficient'] = data['Genetic_Pedigree_Coefficient'].fillna(data['Genetic_Pedigree_Coefficient'].mean())
data['alcohol_consumption_per_day'] = data['alcohol_consumption_per_day'].fillna(data['alcohol_consumption_per_day'].mean())
data['Pregnancy'] = data['Pregnancy'].fillna(0)


label_encoders = {}
for col in data.columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

print("after")
print("shape:", data.shape)
print("'' 분포 :")
print(data['Blood_Pressure_Abnormality'].value_counts())
print("\X feature shape:", data.drop('Blood_Pressure_Abnormality', axis=1).shape)
print("y label shape:", data['Blood_Pressure_Abnormality'].shape)

X = data.drop('Blood_Pressure_Abnormality', axis=1).values
y = data['Blood_Pressure_Abnormality'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_test])
y_train, y_test = map(lambda y: torch.tensor(y, dtype=torch.int64), [y_train, y_test])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

def split_sequences(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences) - n_steps + 1):
        seq_x = sequences[i:i+n_steps, :-1]
        seq_y = sequences[i+n_steps-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

data_array = np.hstack((X, y.reshape(-1, 1)))
X_seq, y_seq = split_sequences(data_array, n_steps=5)

print(X_seq.shape)  
# 예: (1724, 5, 8) # 입력층에 넣을 특징수확인인
# CNN 은 seq_len 필요
# 그래서 X.shape = (샘플 수, 피처 수) -> (batch_size, channels, seq_len)
# 변경 필요 
# input_channels = X_seq.shape[2]  # 8 (특성 수)
# seq_len = X_seq.shape[1]         # 5 (시퀀스 길이)

# 텐서변환 ( 분류면 정수 int 64 사용 )
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # (batch, channels, seq_len)
y_train = torch.tensor(y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test = torch.tensor(y_test, dtype=torch.int64)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

class CNNClassifier(nn.Module):
    def __init__(self, input_channels, seq_len, output_dim):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        # 레이어 늘려주기 
        # self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        # 레이어 늘려준만큼 fc 값도 변경 
        self.fc1 = nn.Linear(64 * seq_len, 32)
        self.fc2 = nn.Linear(32, output_dim)  # 클래스 수

    def forward(self, x):  # (batch, channels, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
         # 레이어 늘리면서 x = torch.relu(self.conv2(x)) 도 3,4,5 .. 늘려주기 
        # x = torch.relu(self.conv4(x))
        # x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 인자 추출후 model에 넣어주기 
input_channels = X_train.shape[1]
seq_len = X_train.shape[2]
output_dim = len(torch.unique(y_train))

model = CNNClassifier(input_channels, seq_len, output_dim) 

# 평가함수 
criterion = nn.CrossEntropyLoss() 

optimizer = optim.Adam(model.parameters(), lr=0.001)

best_acc = 0.0
patience = 10
counter = 0

# 학습 
epochs = 200
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
    cm = confusion_matrix(labels, preds) 
    print(f"Epoch {epoch+1}, Acc: {acc:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm) 
    
    if acc > best_acc:
        best_acc = acc
        counter = 0
        print("✅ Accuracy improved.")
    else:
        counter += 1
        print(f"⏳ No improvement. Patience: {counter}/{patience}")
        if counter >= patience:
            print("⛔ Early stopping: accuracy not improving.")
            break
        
# corr = data.corr()

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 부호 깨짐 방지

# plt.figure(figsize=(8, 6))
# sns.boxplot(data=data, x='Sex', y='Age', hue='Blood_Pressure_Abnormality')
# plt.title('성별에 따른 나이 분포 (혈압 이상 유무별)')
# plt.xlabel('성별 (0: 남성, 1: 여성)')
# plt.ylabel('나이')
# plt.legend(title='혈압 이상 (0: 정상, 1: 이상)')
# plt.tight_layout()
# plt.show()

# # 히트맵 그리기
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
# plt.title('변수 간 상관관계 히트맵')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 5))
# sns.histplot(data=data, x='BMI', bins=20, kde=True, hue='Blood_Pressure_Abnormality', multiple='stack')
# plt.title('혈압 이상 여부에 따른 BMI 분포')
# plt.xlabel('BMI')
# plt.ylabel('빈도')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 5))
# sns.histplot(data=data, x='Age', bins=20, kde=True, hue='Blood_Pressure_Abnormality', multiple='stack')
# plt.title('혈압 이상 여부에 따른 Age 분포')
# plt.xlabel('Age')
# plt.ylabel('빈도')
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='BMI', y='Smoking', hue='Blood_Pressure_Abnormality', palette='Set1')
plt.title('BMI와 흡연 여부에 따른 혈압 이상 분포')
plt.xlabel('BMI')
plt.ylabel('흡연 여부 (0: 비흡연, 1: 흡연)')
plt.legend(title='혈압 이상 (0: 정상, 1: 이상)')
plt.tight_layout()
plt.show()
