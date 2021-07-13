import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)]) # (5, 100)
x2 = np.transpose(x1) # (100, 5)
y1 = np.array([range(711, 811), range(101, 201)]) # (2, 100)
y2 = np.transpose(y1) #  (100, 2)

#2. 모델링
model = Sequential()
model.add(Dense(3, input_dim=5))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param # 
=================================================================
dense (Dense)                (None, 3)                 18 = 5 * 3 + 3
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 16 = 3 * 4 + 4
_________________________________________________________________
dense_2 (Dense)              (None, 10)                50 = 4 * 10 + 10
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22 = 10 * 2 + 2 
=================================================================
Total params: 106 => paran# 총합(18+16+50+22)
Trainable params: 106
Non-trainable params: 0
'''

#3. 컴파일, 훈련

#4. 평가, 예측
