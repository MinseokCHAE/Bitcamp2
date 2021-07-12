from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터
diabets = load_diabetes()
x = diabets.data
y = diabets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

'''
print(diabets)
print(diabets.keys())
print(x.shape)
print(y.shape)
print(diabets.feature_names)
print(diabets.DESCR)
'''

#2. 모델링
model = Sequential()
model.add(Dense(21, input_dim=10, activation='relu')) # 활성화함수: default값 있음, relu가 통상적으로 best, 마지막 layer에는 안씀
model.add(Dense(18, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=222, batch_size=21, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss = ', loss)
print('r2 score = ', r2)

'''
epochs = 101
loss =  2966.928955078125
r2 score =  0.5294805373312435
'''
