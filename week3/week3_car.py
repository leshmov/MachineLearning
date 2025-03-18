import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.ensemble import RandomForestClassifier #Randomforest
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #LR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #평가
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용

local = "C:/4-1/ML/week3/car_evaluation.csv"  
local_df = pd.read_csv(local)

local_df.columns =["buying","maintain","doors","person","lug","safety","class"]

encoder = LabelEncoder()
for column in local_df.columns:
    local_df[column] = encoder.fit_transform(local_df[column])
# #라벨 인코딩 전
# print(local_df['Embarked'].value_counts())
# print(local_df['Sex'].value_counts())

# #embarked 칼럼 -) 라벨 인코딩
# encoder = LabelEncoder()
# local_df['Embarked'] = encoder.fit_transform(local_df['Embarked'])
# local_df['Sex'] = encoder.fit_transform(local_df['Sex'])

# #인코딩 된거 확인
# print(local_df[['Embarked']].head())
# print(local_df[['Sex']].head())

# #라벨 인코딩 후
# print(local_df['Embarked'].value_counts())
# print(local_df['Sex'].value_counts())
#필요 특성
ft=['buying','maintain','doors','person','lug','safety']
#타겟
tg='class'

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
    print(f"✅ {name} 정확도: {acc:.4f}")

