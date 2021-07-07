from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.python.keras.utils.layer_utils import print_summary

#  완성후 출력결과스크린샷

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=4572, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x, y)
result = model.predict([6])
print('loss : ', loss)
print('6의 예측값 : ', result)

