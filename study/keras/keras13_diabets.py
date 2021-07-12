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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)

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
model.add(Dense(21, input_dim=10, activation='relu'))
model.add(Dense(18, activation='relu')) 
model.add(Dense(60, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=123, batch_size=31, validation_split=0.03)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss = ', loss)
print('r2 score = ', r2)

'''
random_state = 9, epochs = 123, batch_size = 21
loss =  2114.74560546875
r2 score =  0.617328805075021
'''
