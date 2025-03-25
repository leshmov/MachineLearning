import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.ensemble import RandomForestClassifier #Randomforest
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #LR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #평가
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용


local = "C:/4-1/ML/week4/wine.csv"  
local_df = pd.read_csv(local)

print(local_df.head())  # 첫 행 출력
print(local_df.columns)  # 컬럼 이름 출력

#결측치 채우기 이전----------
print(local_df.isnull().mean())  # 각 열의 결측치 비율

# drop 으로 필요없는 데이터 지우기
# local_df.drop(columns=['Cabin'], inplace=True)

# fillna() 로 결측치 매꾸기
# local_df['Age'].fillna(local_df['Age'].mean(), inplace=True)
# local_df['Embarked'].fillna(local_df['Embarked'].mode()[0], inplace=True)


encoder = LabelEncoder()
for column in local_df.columns:
    local_df[column] = encoder.fit_transform(local_df[column])
    
# for column in local_df.columns:
#     print(local_df[column].value_counts())

#타겟
tg='Wine'
#필요 특성(up 2025-03-23)
# ft=['buying','maintain','doors','person','lug','safety']
ft = [col for col in local_df.columns if col != tg] 

x=local_df[ft]
y=local_df[tg]

#나눠주기
x_train, x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

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
    cm = confusion_matrix(y_test, y_pred)
    accuracy_results[name] = acc
    print(f"✅ {name} 정확도: {acc:.4f}")
    print(f"📊 {name} Confusion Matrix:\n{cm}")
