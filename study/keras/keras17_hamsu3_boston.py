from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#1. 데이터
boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9) 

'''
print(boston)
print(boston.keys())
print(x.shape)
print(y.shape)
print(boston.feature_names)
print(boston.DESCR)
'''

#2. 모델구성
input1 = Input(shape=(13, ))
dense1 = Dense(11)(input1)
dense2 = Dense(13)(dense1)
dense3 = Dense(15)(dense2)
dense4 = Dense(17)(dense3)
dense5 = Dense(19)(dense4)
dense6 = Dense(21)(dense5)
output1 = Dense(1)(dense6)

model = Model(Inputs=input1, Outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=111, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss = ', loss)
print('r2 score = ', r2)

'''
epochs = 111
loss =  18.75067710876465
r2 score =  0.77304122195697
'''
