from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1.  데이터
x1 = np.array([range(10), range(21, 31), range(201,211)]) #(3, 10)
x2 = np.transpose(x1) #reshaping (10, 3)
y1 = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) #(3, 10)
y2 = np.transpose(y1) #reshaping (10, 3)

x_pred = np.array([[0, 21, 201]]) #(1, 3)

#2.모델링
model = Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x2, y2, epochs=2222, batch_size=1)

#4. 평가, 예측
'''
loss = model.evaluate(x2, y2)
result = model.predict([[0, 21, 201]])
print('loss = ', loss)
print('result = ', result)

epochs = 2222
loss =  0.006706085987389088
result =  [[ 0.8887407  1.116973  10.029439 ]]
'''
y_predict = model.predict(x2)

plt.scatter(x2[:,:1],y2[:,:1])
plt.plot(x2,y_predict, color='red')
plt.show()
