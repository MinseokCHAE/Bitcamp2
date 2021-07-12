from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5) 

'''
print(boston)
print(boston.keys())
print(x.shape)
print(y.shape)
print(boston.feature_names)
print(boston.DESCR)
'''

#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=13))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(19))
model.add(Dense(21))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=222, batch_size=1, validation_split=0.3)

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
