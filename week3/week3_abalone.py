#REG

import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeRegressor #DTreeR
from sklearn.ensemble import RandomForestRegressor #RandomforestR
from sklearn.svm import SVR #SVR
from sklearn.linear_model import LinearRegression #LinearR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error #평가
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용


local = "C:/4-1/ML/week3/abalone.csv"  
local_df = pd.read_csv(local)

#인코딩
encoder = LabelEncoder()
for column in local_df.columns:
    local_df[column] = encoder.fit_transform(local_df[column])

#타겟
tg='Rings'
#필요 특성(특성 자동 계산 ud 2025-03-23)
ft=[col for col in local_df.columns if col != tg]
# ft=['id','Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight']

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

# 모델 학습
for name, model in models.items():
    model.fit(x_train, y_train)  # 학습
    y_pred = model.predict(x_test)  # 예측
    mse = mean_squared_error(y_test, y_pred) # 정확도 함수
    print(f"✅ {name} - MSE: {mse:.4f}")


