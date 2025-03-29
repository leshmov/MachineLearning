
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

local = "C:/4-1/ML/week4/wine.csv"  
local_df = pd.read_csv(local)

print(local_df.head())  # 첫 행 출력
print(local_df.columns)  # 컬럼 이름 출력

X=local_df.drop('Wine',axis=1)
X.head() #미리보기기

y=local_df['Wine']
y.value_counts() #빈도수
y.head() #미리보기기

# 원핫 인코딩 전 데이터 (라벨 분포 보기)
print("before one hot")
print(y.head())
print("y.shape:", y.shape)
print("클래스 분포:\n", y.value_counts())

# 원핫 인코딩딩
Y = pd.get_dummies(y).values
X = X.values

print("after one hot")
print(Y[:5])        # .head() 대신 슬라이싱하기
print("Y.shape:", Y.shape)

X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

model = Sequential() # 모델 시퀀셜

model.add(Dense(10,input_shape=(13,),activation='relu')) # 특징이 13개 이므로 input_shape 에 13개 넣기 

model.add(Dense(50,activation='relu'))

model.add(Dense(100,activation='relu'))

# model.add(Dense(200,activation='relu'))

# model.add(Dense(200,activation='relu'))
# 200 까지 가니까 overfitting 발생 

model.add(Dense(100,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(20,activation='relu'))

model.add(Dense(3,activation='softmax')) # 출력층 

model.compile(Adam(learning_rate=0.04),'categorical_crossentropy',metrics=['accuracy'])

model.summary()

model_history=model.fit(x=X_train, y=y_train, epochs=50, batch_size=32,validation_data= (X_test,y_test))
# epoch : 몇번 반복해서 학습할지 
# batch size : 데이터를 한번에 몇개씩 가져올건지 (보통은 16,32,64,128.. 로 실험)

y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test,axis=1) 
y_pred_class = np.argmax(y_pred,axis=1)
loss =model_history.history['loss']
val_loss =model_history.history['val_loss']
epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

acc =model_history.history['accuracy']
val_acc =model_history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))
