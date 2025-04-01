import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/main/heart.csv"
url_df = pd.read_csv(url)

print(url_df.head())  
print(url_df.columns)  
print()
print(url_df['target'].value_counts())

X = url_df.drop('target', axis=1).values
y = url_df['target']

print("before")
print("y.shape",y.shape)
print("y.value_count",y.value_counts())

Y = pd.get_dummies(y).values

print("after")
print(y[:5])
print("Y.shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=40)

model = Sequential()

model.add(Dense(64, input_shape=(13,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
 
model.add(Dense(1, activation='sigmoid'))   

model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

model_history = model.fit(X_val, y_val, epochs=100, batch_size=16,validation_split=0.1)
#y train 원핫 된거임

y_pred = model.predict(X_test)
# argmax 원핫 인코딩 된것을 원상태로 돌려서 평가
y_test_class =y_test
y_pred_class = (y_pred > 0.5).astype(int)  # 확률을 0/1로 변환

# 그래프
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))
