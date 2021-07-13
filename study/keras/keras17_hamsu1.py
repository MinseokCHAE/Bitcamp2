import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)]) # (5, 100)
x2 = np.transpose(x1) # (100, 5)
y1 = np.array([range(711, 811), range(101, 201)]) # (2, 100)
y2 = np.transpose(y1) #  (100, 2)

#2. 모델링
# model = Sequential()
# model.add(Dense(3, input_dim=5))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

input1 = Input(shape=(5, ))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1) 
# 함수형 모델은 마지막에 model 정의(시작지점, 끝지점 명시) <=> Sequential 모델은 순차적으로 축적
# 함수형 모델을 쓰는이유 : 모델내에서 layer 점프가능, 여러모델간 상호작용 가능, 형태적으로 가공에 용이

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
