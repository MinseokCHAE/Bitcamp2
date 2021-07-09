from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

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
model.fit(x, y, epochs=99, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x, y)
result = model.predict([6])
y_predict = model.predict(x)

print('loss : ', loss)
print('result : ', result)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2 score = ', r2)

'''
loss :  0.3906857669353485
result :  [[5.8679695]]
r2 score =  0.804657116180475
'''
