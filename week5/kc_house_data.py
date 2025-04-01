
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error
from sklearn.model_selection import train_test_split


url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/main/kc_house_data.csv"
url_df = pd.read_csv(url)

print(url_df.head())  # 첫 행 출력
print(url_df.columns)  # 컬럼 이름 출력

url_df=url_df.drop(['id','date'],axis=1)
#date 문자열이라 삭제

# 특성분리 
X=url_df.drop('price',axis=1).values
# pandas -> numpy
y=url_df['price']


# # 원핫 인코딩딩
# Y = pd.get_dummies(y).values
# 회귀에선 필요없음 

print("after one hot")
print(y[:5])        # .head() 대신 슬라이싱하기
print("Y.shape:", y.shape)
print("X.shape:", X.shape)

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)
X_train,X_trainval, y_train,y_trainval = train_test_split(X_train,y_train, test_size=0.2, random_state=40)
# X_trainval, y_trainval -> validation 용용

model = Sequential() # 모델 시퀀셜

model.add(Dense(16,input_shape=(18,),activation='relu')) 

model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))

model.add(Dense(1)) 

model.compile(optimizer=Adam(learning_rate=0.001),loss='mse',metrics=['mse'])

model.summary()

model_history=model.fit(x=X_train, y=y_train, epochs=100, batch_size=128,validation_data= (X_trainval,y_trainval))
# # epoch : 몇번 반복해서 학습할지 
# # batch size : 데이터를 한번에 몇개씩 가져올건지 (보통은 16,32,64,128.. 로 실험)
# # epoch : 10000 하면 잘나오긴함


y_pred = model.predict(X_test).flatten()
# 연속값이면 flatten 으로 만들어주기 

# y_test_class = np.argmax(y_test,axis=1) 
# y_pred_class = np.argmax(y_pred,axis=1)
# 회귀에선 필요없음

loss =model_history.history['loss']
val_loss =model_history.history['val_loss']
epochs = range(1, len(loss) + 1)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  
plt.show()
# 그래프 해석 
# 대각점선에 가까울수록 정확한 예측측

# mse 해석
# 고가주택존재, 제곱오차, 단위가 달러 라서 10-20만 정도 오차 발생. 