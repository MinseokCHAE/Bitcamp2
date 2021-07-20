import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, Dropout, LSTM, GRU


#1. Data Preprocessing
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9],
                            [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]]) 
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]).reshape(1,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1) 

#2. Modeling
input = Input(shape=(3, 1))
s = GRU(32, activation='relu')(input)
s = Dense(32, activation='relu')(s)
s = Dense(16,  activation='relu')(s)
s = Dense(16, activation='relu')(s)
s = Dense(8, activation='relu')(s)
output = Dense(1, activation='relu')(s)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=128, batch_size=1)

#. Evaluating, Prediction
loss = model.evaluate(x, y)
result = model.predict(x_pred)

print('loss : ', loss)
print('pred : ', result)

'''
loss :  0.0493345633149147
pred :  [[80.60736]]
'''
