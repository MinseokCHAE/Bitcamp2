import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, Dropout, LSTM, GRU

#1. Data Preprocessing
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9],
                            [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], [60,70,80], [70,80,90],
                            [80,90,100], [90,100,110], [100,110,120], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_pred = np.array([55,65,75])
x2_pred = np.array([65,75,85])

x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
x1_pred = x1_pred.reshape(1,3,1)
x2_pred = x2_pred.reshape(1,3,1)

#2. Modeling
input1 = Input(shape=(3, 1), name='input1')
dense1 = LSTM(10, activation='relu', name='dense1')(input1)
output1 = Dense(11, name='output1')(dense1)

input2 = Input(shape=(3, 1), name='input2')
dense2 = LSTM(10, activation='relu', name='dense2')(input2)
output2 = Dense(12, name='output2')(dense2)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='merge1')
last_output = Dense(1, name='last_output')(merge1)

model = Model(inputs=[input1, input2], outputs=last_output)

#3. Compiling, Training
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y, epochs=64, batch_size=1)

#. Evaluating, Prediction
loss = model.evaluate([x1, x2], y)
result = model.predict([x1_pred, x2_pred])

print('loss : ', loss)
print('pred : ', result)

'''
loss :  0.530460000038147
pred :  [[79.67656]]
'''
