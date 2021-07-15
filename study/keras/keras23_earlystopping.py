from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#1. 데이터
boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=66) 

#1-1. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input = Input(shape=(13, ))
x = Dense(128, activation='relu')(input)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
num_epochs = 100

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=num_epochs, 
                            batch_size=8, callbacks=[es], validation_split=0.05)

# print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
# print('=========loss=========')
# print(hist.history['loss']) # [252.2976837158203, 30.154781341552734]
# print('=========val_loss=========')
# print(hist.history['val_loss']) # [23.07321548461914, 13.349472999572754]

import matplotlib.pyplot as plt
plt.plot(hist.history['loss']) # x: epoch, y: hist.history['loss']
plt.plot(hist.history['val_loss'])

import matplotlib.font_manager as fm
font_path = r'C:\Users\bit\AppData\Local\Microsoft\Windows\Fonts\Cookierun Regular.ttf'
fontprop = fm.FontProperties(fname=font_path, size=18)

plt.title('로스, 발로스', fontproperties=fontprop)
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['loss', 'val_loss']) # plt.plot 해준 순서대로 받음 ex) 'loss', 'val_loss'
plt.show()

'''
#4. 평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('epochs = ', num_epochs)
print('loss = ', loss)
print('r2 score = ', r2)


epochs =  100
loss =  6.122464179992676
r2 score =  0.9360175200769787
'''
