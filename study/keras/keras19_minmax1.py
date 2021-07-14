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

#1-1. 데이터 전처리 min max scaler
# print(np.min(x), np.max(x)) # 0.0 711.0
# x = x /711
x = (x - np.min(x)) / (np.max(x) - np.min(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66) 


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
loss =  14.556221008300781
r2 score =  0.8238110372768978
'''
