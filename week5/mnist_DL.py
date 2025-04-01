import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical  

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

digits = datasets.load_digits()
images = digits.images
target = digits.target

# images.shape = (1797, 8, 8)
# 8x8 -> 64 이므로 input_shape 64

# 숫자 표시시
plt.imshow(images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"Label: {target[0]}")
plt.show()

# X 변환 ( input_shape 를 위함 )
n_samples = len(images)
X = images.reshape((n_samples, -1)) 

# y 변환 ( 원 핫 인코딩 )
# before: 
# y = target 
# Y = pd.get_dummies(y).values
# after:
y = to_categorical(target, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = Sequential()
model.add(Dense(64, input_shape=(64,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))      
model.add(Dense(10, activation='softmax'))   

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model_history = model.fit(X_train, y_train, epochs=100, batch_size=16,validation_split=0.1)
#y train 원핫 된거임

y_pred = model.predict(X_test)
# argmax 원핫 인코딩 된것을 원상태로 돌려서 평가
y_test_class = np.argmax(y_test, axis=1) 
y_pred_class = np.argmax(y_pred, axis=1)

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
