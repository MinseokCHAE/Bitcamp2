from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1.  데이터
x1 = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) #(3, 10)
x2 = np.transpose(x1) #reshaping (10,3)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) #(10, ) <-> (10,1)
x_pred = np.array([[10, 1.3, 1]]) #(1, 3)

#2.모델링
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x2, y, epochs=2121, batch_size=1)

#4. 평가, 예측
'''
loss = model.evaluate(x2, y)
result = model.predict([[10, 1.3, 1]])
print('loss = ', loss)
print('result = ', result)

epochs = 2121
loss =  4.552236987365177e-06
result =  [[20.000088]]
'''
y_predict = model.predict(x2)

plt.scatter(x2[:,:1], y)
plt.plot(x2, y_predict, color='red')
plt.show()
