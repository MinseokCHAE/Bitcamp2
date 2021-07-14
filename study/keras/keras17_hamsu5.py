import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)]) # (5, 100)
x2 = np.transpose(x1) # (100, 5)
y1 = np.array([range(711, 811), range(101, 201)]) # (2, 100)
y2 = np.transpose(y1) #  (100, 2)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

input = Input(shape=(5, ))
x = Dense(3)(input)
x = Dense(4)(x)
x = Dense(10)(x)
output = Dense(2)(x)
# 함수형 모델이 단순 순차적 구성일 경우 각 layer 명을 동일하게 설정가능

model = Model(inputs=input, outputs=output) 
model.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
dense (Dense)                (None, 3)                 18
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 10)                50
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22
=================================================================
Total params: 106
Trainable params: 106
Non-trainable params: 0
'''

#3. 컴파일, 훈련

#4. 평가, 예측
