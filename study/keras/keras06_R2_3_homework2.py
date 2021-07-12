'''
R2 >= 0.9
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(21, activation='selu'))
model.add(Dense(39, activation='selu'))
model.add(Dense(60, activation='selu'))
model.add(Dense(18, activation='selu'))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=765, batch_size=15)

#4. 평가,예측
loss = model.evaluate(x, y)
result = model.predict([6])
y_predict = model.predict(x)

print('loss = ', loss)
print('result = ', result)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2 score = ', r2)

'''
epochs = 765
loss =  0.19223839044570923
result =  [[5.2654734]]
r2 score =  0.9038808008068596
'''
