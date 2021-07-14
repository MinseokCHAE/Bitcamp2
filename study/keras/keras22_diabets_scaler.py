from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd

#1. 데이터
diabets = load_diabetes()
x = diabets.data
y = diabets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델링
input = Input(shape=(10, ))
x = Dense(16, activation='relu')(input)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(4, activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
num_epochs = 100
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=num_epochs, batch_size=8)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('epochs = ', num_epochs)
print('loss = ', loss)
print('r2 score = ', r2)

'''
MaxAbsScaler
epochs =  100
loss =  3195.79052734375
r2 score =  0.4870624806895163

RobustScaler
epochs =  100
loss =  3935.587646484375
r2 score =  0.36832202820082105

QuantileTransformer
epochs =  100
loss =  3114.07421875
r2 score =  0.5001783144633074

PowerTransformer
epochs =  100
loss =  3622.23779296875
r2 score =  0.44187743910803245
'''
