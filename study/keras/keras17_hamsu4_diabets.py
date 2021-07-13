from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd

#1. 데이터
diabets = load_diabetes()
x = diabets.data
y = diabets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)

'''
print(diabets)
print(diabets.keys())
print(x.shape)
print(y.shape)
print(diabets.feature_names)
print(diabets.DESCR)
'''

#2. 모델링
input1 = Input(shape=(10, ))
dense1 = Dense(4, activation='relu')(input1)
dense2 = Dense(8, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(16, activation='relu')(dense4)
dense6 = Dense(8, activation='relu')(dense5)
dense7 = Dense(4, activation='relu')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)
model.summary()

# model = Sequential()
# model.add(Dense(4, input_dim=10, activation='relu'))
# model.add(Dense(8, activation='relu')) 
# model.add(Dense(16, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.05)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss = ', loss)
print('r2 score = ', r2)

'''
epochs = 100, batch_size = 16
loss =  2064.8046875
r2 score =  0.6263657834047769
'''
