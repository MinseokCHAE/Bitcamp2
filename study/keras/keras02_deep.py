from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

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
model.fit(x, y, epochs=4572, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x, y)
result = model.predict([6])
print('loss : ', loss)
print('result : ', result)

"""
epochs = 1234
loss :  0.380005419254303
result :  [[5.695348]]
"""
