import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor #DTree
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #Randomforest
from sklearn.svm import SVC,SVR #SVC
from sklearn.linear_model import LogisticRegression,LinearRegression #LR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error #평가
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용


local = "C:/4-1/ML/week3/abalone.csv"  
local_df = pd.read_csv(local)

#결측치 채우기 0

# drop 으로 필요없는 데이터 지우기

# # fillna() 로 결측치 매꾸기
# local_df['Age'].fillna(local_df['Age'].mean(), inplace=True)
# local_df['Embarked'].fillna(local_df['Embarked'].mode()[0], inplace=True)

# #결측치 매꾼후--------------
# print("---------------")
# print(local_df.isnull().mean())  # 각 열의 결측치 비율

# #라벨 인코딩 전
# print(local_df['Embarked'].value_counts())
# print(local_df['Sex'].value_counts())

#인코딩
encoder = LabelEncoder()
for column in local_df.columns:
    local_df[column] = encoder.fit_transform(local_df[column])
# #인코딩 된거 확인
# print(local_df[['Embarked']].head())
# print(local_df[['Sex']].head())

# #라벨 인코딩 후
# print(local_df['Embarked'].value_counts())
# print(local_df['Sex'].value_counts())

#필요 특성
ft=['id','Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight']
#타겟
tg='Rings'

x=local_df[ft]
y=local_df[tg]

#나눠주기
x_train, x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#모델정의
models = {
    "Decision R": DecisionTreeRegressor(),
    "Random F R": RandomForestRegressor(),
    "Logistic Regression": LinearRegression(),
    # "SVR": SVR(kernel='linear')
}



accuracy_results = {}

# 모델 학습습
for name, model in models.items():
    model.fit(x_train, y_train)  # 학습
    y_pred = model.predict(x_test)  # 예측
    mse = mean_squared_error(y_test, y_pred) # 정확도 계산
    # accuracy_results[name] = acc
    # print(f"✅ {name} 정확도: {acc:.4f}")
    print(f"✅ {name} - MSE: {mse:.4f}")


