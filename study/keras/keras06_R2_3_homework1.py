'''
0 <= R2 <= 0.5
data no touch
layer >=6 (include input, output layer)
batch_size = 1
epochs >= 100
10 <= hidden layer node <= 100
train 70%
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array(range(100)) # 0~99
y = np.array(range(1,101)) # 1~100

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train[-1:], y_train[-1:], epochs=101, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
result = model.predict([99])
y_predict = model.predict(x_test)

print('loss = ', loss)
print('result = ', result)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score = ', r2)

'''
epochs = 101
loss =  0.20065012574195862
result =  [[100.792854]]
r2 score =  0.9997214980704084
'''
