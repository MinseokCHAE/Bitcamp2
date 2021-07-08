from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1.  데이터
x1 = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]) #(2, 10)
x2 = np.array([[1, 1], [2, 1.1], [3, 1.2], [4, 1.3], [5, 1.4], 
                        [6, 1.5], [7, 1.6], [8, 1.5], [9, 1.4], [10, 1.3]]) #(10, 2)

# x1 -> x2 : numpy.transpose, numpy.swapaxes, T 
x3 = np.transpose(x1)
x4 = np.swapaxes(x1, 0, 1)
x5 = x1.T

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) #(10, ) <-> (10,1)
x_pred = np.array([[10, 1.3]]) #(1, 2)

#2.모델링
model = Sequential()
model.add(Dense(1, input_dim=2))
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
model.fit(x3, y, epochs=2222, batch_size=1)

#4. 평가, 예측
'''
loss = model.evaluate(x3, y)
result = model.predict([[10, 1.3]])
print('loss : ', loss)
print('result : ', result)

epochs : 2222
loss :  0.0006747512961737812
result :  [[19.988302]]
'''
y_predict = model.predict(x3)

plt.scatter(x3[:,:1], y)
plt.plot(x3, y_predict, color='red')
plt.show()
