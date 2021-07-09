from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array(range(100)) # 0~99
y = np.array(range(1,101)) # 1~100

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66) 

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=1)

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
epochs = 50
loss =  1.8300019291928038e-05
result =  [[100.001945]]
r2 score = 0.9999999790709166
'''
