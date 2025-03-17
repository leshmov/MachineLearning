import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.ensemble import RandomForestClassifier #Randomforest
from sklearn.svm import SVC #SVC
from sklearn.linear_model import LogisticRegression #LR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #평가
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용

# url= "https://raw.github.com/MyungKyuYi/AI-class/blob/main/titanic.csv"
url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/main/titanic.csv"
# ㄴ 마찬가지로 raw를 붙여서 html이 아니라 csv 파일을 반환하게 함.
# ㄴ raw.githubusercontent.com 는 실제 데이터를 받을수있음.

url_df = pd.read_csv(url)

print(url_df.head())  # 첫 행 출력
print(url_df.columns)  # 컬럼 이름 출력


#결측치 채우기 이전----------
print(url_df.isnull().mean())  # 각 열의 결측치 비율

# drop 으로 필요없는 데이터 지우기
url_df.drop(columns=['Cabin'], inplace=True)

# fillna() 로 결측치 매꾸기
url_df['Age'].fillna(url_df['Age'].mean(), inplace=True)
url_df['Embarked'].fillna(url_df['Embarked'].mode()[0], inplace=True)

#결측치 매꾼후--------------
print("---------------")
print(url_df.isnull().mean())  # 각 열의 결측치 비율

#라벨 인코딩 전
print(url_df['Embarked'].value_counts())
print(url_df['Sex'].value_counts())

#embarked 칼럼 -) 라벨 인코딩
encoder = LabelEncoder()
url_df['Embarked'] = encoder.fit_transform(url_df['Embarked'])
url_df['Sex'] = encoder.fit_transform(url_df['Sex'])

#인코딩 된거 확인
print(url_df[['Embarked']].head())
print(url_df[['Sex']].head())

#라벨 인코딩 후
print(url_df['Embarked'].value_counts())
print(url_df['Sex'].value_counts())

#필요 특성
ft=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
#타겟
tg='Survived'

x=url_df[ft]
y=url_df[tg]

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
