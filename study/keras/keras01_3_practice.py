from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([5,10,15,20,25])
x_pred = [10]

#2.  모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=4572, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([10])
print('loss : ', loss)
print('result : ', result)

"""
epochs = 4572
loss :  2.597321291375465e-08
result :  [[49.99935]]
"""
