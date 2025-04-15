
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical # dummies 안될때 쓰기

from sklearn.metrics import classification_report,confusion_matrix #분류류
from sklearn.metrics import mean_squared_error #회귀
from sklearn.model_selection import train_test_split


url = "" #파일 불러오기
url_df = pd.read_csv(url)

# 가져온 파일 확인인
print(url_df.head())  # 첫 행 출력
print(url_df.columns)  # 컬럼 이름 출력

# 결측치 확인 
print(url_df.isnull().values.any()) # 결측치 하나라도 있는지 확인인
print(url_df.isnull().sum()) # 결측치 갯수 확인
url_df=url_df.drop([],axis=1) #쓸모없는 데이터 드롭롭




# 특성분리 
X=url_df.drop('',axis=1).values # (values) pandas -> numpy
# .to_numpy() 도 사용가능 

y=url_df['']

print("before one hot")
print("Y.value", y.value_counts())
print("Y.shape:", y.shape)
print("X.shape:", X.shape)

# 원핫 인코딩 ( 분류에서만 사용 )
# Y = pd.get_dummies(y).values

# 회귀에선 사용 잘안함 


# print("after one hot")
# print("Y.value",Y.sum())
# print("Y.shape:", y.shape)
# print("X.shape:", X.shape)

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)
X_train,X_trainval, y_train,y_trainval = train_test_split(X_train,y_train, test_size=0.2, random_state=40)
# X_trainval, y_trainval -> validation 용용

model = Sequential() # 모델 시퀀셜

# 입력층
model.add(Dense(16,input_shape=(18,),activation='relu')) # shape 로확인 한후수정하기 

model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))

#출력층 
model.add(Dense(1)) 

# 손실함수는 lossfunction.jpg 참고 

model.compile(optimizer=Adam(learning_rate=0.001),loss='mse',metrics=['mse'])

model.summary()

# 훈련데이터 validation 으로 나눴으면 여기에 집어넣기 
model_history=model.fit(x=X_train, y=y_train, epochs=100, batch_size=128,validation_data= (X_trainval,y_trainval))
# # epoch : 몇번 반복해서 학습할지 
# # batch size : 데이터를 한번에 몇개씩 가져올건지 (보통은 16,32,64,128.. 로 실험)
# # epoch : 10000 하면 잘나오긴함

# 분류
# y_pred = model.predict(X_test)
# y_test_class = np.argmax(y_test,axis=1) 
# y_pred_class = np.argmax(y_pred,axis=1)

# 회귀 
y_pred = model.predict(X_test).flatten()
# 연속값이면 flatten 으로 만들어주기 

# flatten 분류에서도 차원맞춰줄려면 가끔 사용
# argmax 회귀에선 필요없음

loss =model_history.history['loss']
val_loss =model_history.history['val_loss']
epochs = range(1, len(loss) + 1)

# 모델평가 (회귀) 

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  
plt.show()

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# 그래프 해석 
# 대각점선에 가까울수록 정확한 예측 

# 모델평가 (분류)

# acc =model_history.history['accuracy']
# val_acc =model_history.history['val_accuracy']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# print(classification_report(y_test_class,y_pred_class))
# print(confusion_matrix(y_test_class,y_pred_class))
