import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
y1 = np.array([range(101, 201), range(411, 511), range(100, 200)])
z1 = np.array([range(1001, 1101)])

x2 = np.transpose(x1) # (100, 3)
y2 = np.transpose(y1) # (100, 3)
z2 = np.transpose(z1) # (100, )

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x2, y2, z2, 
                                                                                        test_size=0.3, random_state=9)

#2. 모델링
#2-1. 모델1
input1 = Input(shape=(3, ), name='input1')
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3, ), name='input2')
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)

#2-3. output1 & 2 통합 및  추가 layer
from tensorflow.keras.layers import Concatenate, concatenate
merge1 = concatenate([output1, output2], name='merge1')
# merge1 = Concatenate()([output1, output2])
merge2 = Dense(10, name='merge2')(merge1)
merge3 = Dense(5, activation='relu', name='merge3')(merge2)
last_output = Dense(1, name='last_output')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()
'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to

==================================================================================================
input2 (InputLayer)             [(None, 3)]          0

__________________________________________________________________________________________________
input1 (InputLayer)             [(None, 3)]          0

__________________________________________________________________________________________________
dense11 (Dense)                 (None, 10)           40          input2[0][0]

__________________________________________________________________________________________________
dense1 (Dense)                  (None, 10)           40          input1[0][0]

__________________________________________________________________________________________________
dense12 (Dense)                 (None, 10)           110         dense11[0][0]

__________________________________________________________________________________________________
dense2 (Dense)                  (None, 7)            77          dense1[0][0]

__________________________________________________________________________________________________
dense13 (Dense)                 (None, 10)           110         dense12[0][0]

__________________________________________________________________________________________________
dense3 (Dense)                  (None, 5)            40          dense2[0][0]

__________________________________________________________________________________________________
dense14 (Dense)                 (None, 10)           110         dense13[0][0]

__________________________________________________________________________________________________
output1 (Dense)                 (None, 11)           66          dense3[0][0]

__________________________________________________________________________________________________
output2 (Dense)                 (None, 12)           132         dense14[0][0]

__________________________________________________________________________________________________
merge1 (Concatenate)            (None, 23)           0           output1[0][0]

                                                                                            output2[0][0]

__________________________________________________________________________________________________
merge2 (Dense)                  (None, 10)           240         merge1[0][0]

__________________________________________________________________________________________________
merge3 (Dense)                  (None, 5)            55          merge2[0][0]

__________________________________________________________________________________________________
last_output (Dense)             (None, 1)            6           merge3[0][0]

==================================================================================================
Total params: 1,026
Trainable params: 1,026
Non-trainable params: 0
'''

#3. 컴파일, 훈련
num_epochs = 100
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x_train, y_train], z_train, epochs=num_epochs, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate([x_test, y_test], z_test)
z_predict = model.predict([x_test, y_test])
r2 = r2_score(z_test, z_predict)

print('epochs =', num_epochs)
print('loss = ', results[0])
print('r2 score =', r2)

'''
epochs = 100
loss =  0.028695017099380493
r2 score = 0.9999540189796811
'''
