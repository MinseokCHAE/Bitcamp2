from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        test_size=0.4, random_state=66) # x -> train, test 로 분할
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, 
        test_size=0.5, random_state=66) #test -> test, val 로 분할

print(x_train)
print(x_test)
print(x_val)

'''
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1234, batch_size=1, 
                    validation_data=(x_val, y_val))

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print('loss = ', loss)
print('result = ', result)
'''
epochs = 1234
loss =  5.153803165486304e-12
result =  [[10.999999]]
'''
'''