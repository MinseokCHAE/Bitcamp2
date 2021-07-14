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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66) 
# 먼저 test, train 을 구분해야 x 데이터 전체가 한꺼번에 scaling 되지않음

#1-1. 데이터 전처리 min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # x_train을 기준으로 scaling
# (주의) fit 은 scaling 비율만 구하고 저장안함, tranform 으로 return,적용 시켜줘야함

x_train = scaler.transform(x_train) # 위 기준으로 x_train transforming
x_test = scaler.transform(x_test) # 위 기준으로 x_test transforming

#2. 모델구성
input = Input(shape=(13, ))
x = Dense(128, activation='relu')(input)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
num_epochs = 100
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=num_epochs, batch_size=8)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('epochs = ', num_epochs)
print('loss = ', loss)
print('r2 score = ', r2)

'''
epochs =  100
loss =  5.559561252593994
r2 score =  0.9327068880469123
'''
