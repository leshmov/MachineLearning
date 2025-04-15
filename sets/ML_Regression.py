import pandas as pd
from sklearn.model_selection import train_test_split #학습용 테스트용 나눌때 사용용
from sklearn.tree import DecisionTreeRegressor #DTreeR
from sklearn.ensemble import RandomForestRegressor #RandomforestR
from sklearn.svm import SVR #SVR
from sklearn.linear_model import LinearRegression #LinearR
from sklearn.metrics import mean_squared_error #평가
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #라벨 인코딩용

# 파일입력
local = ""  
local_df = pd.read_csv(local)

# # 헤더 없을경우 
# local_df = pd.read_csv(local, header=None)
# # none으로 불러와서
# local_df.columns =["buying","maintain","doors","person","lug","safety","class"]
# #헤더 새로 추가해주기

# 🔍 결측치 확인
print("📌 [전체 결측치 확인]")
print(local_df.isnull().sum())
print("\n📌 [데이터 크기]:", local_df.shape)

# 🔍 인코딩 전 주요 컬럼 value_counts 확인 (예: 'Sex', 'Class' 등)
print("\n📌 [인코딩 전 데이터 분포]")
for column in local_df.columns:
    if local_df[column].dtype == 'object':
        print(f"\n{column} 분포:")
        print(local_df[column].value_counts())

#인코딩
encoder = LabelEncoder()
for column in local_df.columns:
    local_df[column] = encoder.fit_transform(local_df[column])
    
# 🔍 인코딩 후 확인
print("\n📌 [인코딩 후 데이터]")
print(local_df.head())


#타겟
tg=''
#필요 특성(특성 자동 계산 ud 2025-03-23)
ft=[col for col in local_df.columns if col != tg]
# ft=['id','Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight']

x=local_df[ft]
y=local_df[tg]

#나눠주기
x_train, x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#모델정의(SVR은 시간이 오래걸림)
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
