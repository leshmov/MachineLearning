import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.ensemble import RandomForestClassifier #Randomforest
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #LR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #평가
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용

local = "C:/4-1/ML/titanic.csv"  
local_df = pd.read_csv(local)

print(local_df.head())  # 첫 행 출력
print(local_df.columns)  # 컬럼 이름 출력

#결측치 채우기 이전----------
print(local_df.isnull().mean())  # 각 열의 결측치 비율

# drop 으로 필요없는 데이터 지우기
local_df.drop(columns=['Cabin'], inplace=True)

# fillna() 로 결측치 매꾸기
local_df['Age'].fillna(local_df['Age'].mean(), inplace=True)
local_df['Embarked'].fillna(local_df['Embarked'].mode()[0], inplace=True)

#결측치 매꾼후--------------
print("---------------")
print(local_df.isnull().mean())  # 각 열의 결측치 비율

#라벨 인코딩 전
print(local_df['Embarked'].value_counts())
print(local_df['Sex'].value_counts())

#embarked 칼럼 -) 라벨 인코딩
encoder = LabelEncoder()
local_df['Embarked'] = encoder.fit_transform(local_df['Embarked'])
local_df['Sex'] = encoder.fit_transform(local_df['Sex'])

#인코딩 된거 확인
print(local_df[['Embarked']].head())
print(local_df[['Sex']].head())

#라벨 인코딩 후
print(local_df['Embarked'].value_counts())
print(local_df['Sex'].value_counts())

#필요 특성
ft=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
#타겟
tg='Survived'

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

# 모델별 정확도 비교
print("\n📊 모델별 정확도 비교:")
for model, acc in accuracy_results.items():
    print(f"{model}: {acc:.4f}")
