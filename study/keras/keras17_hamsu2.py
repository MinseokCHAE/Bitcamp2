from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

#2. 모델구성
input1 = Input(shape=(1, ))
dense1 = Dense(1)(input1)
dense2 = Dense(21)(dense1)
dense3 = Dense(39)(dense2)
dense4 = Dense(60)(dense3)
dense5 = Dense(18)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1) 
model.summary()

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
