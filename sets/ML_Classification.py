import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.ensemble import RandomForestClassifier #Randomforest
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #LR
from sklearn.metrics import confusion_matrix, accuracy_score #평가
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용
from sklearn.metrics import confusion_matrix #컨퓨전 매트릭스스

local = ""  
local_df = pd.read_csv(local)

# # 헤더 없을경우 
# local_df = pd.read_csv(local, header=None)
# # none으로 불러와서
# local_df.columns =["buying","maintain","doors","person","lug","safety","class"]
# #헤더 새로 추가해주기



print(" [전체 결측치 확인]")
print(local_df.isnull().sum())
print("\n📌 [데이터 크기]:", local_df.shape)

# for column in local_df.columns:
#     print(local_df[column].value_counts())

encoder = LabelEncoder()
for column in local_df.columns:
    local_df[column] = encoder.fit_transform(local_df[column])
    
# 인코딩 후 확인
print("\n [인코딩 후 데이터]")
print(local_df.head())

# for column in local_df.columns:
#     print(local_df[column].value_counts())


#타겟
tg=''
#필요 특성(up 2025-03-23)
# ft=['buying','maintain','doors','person','lug','safety']
ft = [col for col in local_df.columns if col != tg] 

x=local_df[ft]
y=local_df[tg]

#나눠주기
x_train, x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#모델정의
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVC": SVC()
}

accuracy_results = {}

# 모델 학습습
for name, model in models.items():
    model.fit(x_train, y_train)  # 학습
    y_pred = model.predict(x_test)  # 예측
    acc = accuracy_score(y_test, y_pred)  # 정확도 계산
    accuracy_results[name] = acc
    print(f" {name} 정확도: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("confusion \n",cm)

