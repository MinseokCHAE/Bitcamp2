from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time

#1.  데이터
x1 = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) #(3, 10)
x2 = np.transpose(x1) #reshaping (10,3)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) #(10, ) <-> (10,1)
x_pred = np.array([[10, 1.3, 1]]) #(1, 3)

#2.모델링
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x2, y, epochs=123, batch_size=2, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x2, y)

print('loss = ', loss)

'''
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
=> loss값에 metrics 에 해당하는 지표가 추가된다.
'''
