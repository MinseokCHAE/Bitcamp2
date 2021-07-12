from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time

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

start = time.time()
model.fit(x2, y, epochs=333, batch_size=2, verbose=2)
end = time.time() - start
print('time : ', end)

'''
batch_size default = 32

Epoch 332/333
'5/5' - 0s - loss: 0.0379
=> 훈련시키는 데이터 행의 수 / batch_size = 5/5

verbose = 0 : 훈련내용X, epochX, progress barX
	1 : 훈련내용O, epochO, progress barO
	2 : 훈련내용O, epochO, progress barX
	3 : 훈련내용X, epochO, progress barX
	4~ : 3과동일

verbose = 0 / time :  8.22016429901123
verbose = 1  / time :  10.786490201950073
verbose = 2 / time :  8.986061334609985
verbose = 3 / time :  8.924289464950562

batch_size의 경우 값이 클수록 빠르다.

'''