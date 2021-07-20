import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, Dropout


#1. Data Preprocessing
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3)
y = np.array([4,5,6,7]) # (4, )
x = x.reshape(4, 3, 1) # rnn input_dim =3, [batch_size, timesteps, feature]
# batch 행 timesteps 열 feature 자르는단위

#2. Modeling
input = Input(shape=(3, 1))
s = SimpleRNN(16, activation='relu')(input)
s = Dropout(0.1)(s)
s = Dense(16, activation='relu')(s)
s = Dropout(0.1)(s)
s = Dense(8, activation='relu')(s)
s = Dropout(0.1)(s)
output = Dense(1, activation='relu')(s)

model = Model(inputs=input, outputs=output)
# model.summary()
'''
Model: "model"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
input_1 (InputLayer)         [(None, 3, 1)]            0
_________________________________________________________________        
simple_rnn (SimpleRNN)       (None, 10)                120 = 10 * ( 10 + 1 + 1 )
_________________________________________________________________        
dense (Dense)                (None, 10)                110
_________________________________________________________________        
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________        
dense_2 (Dense)              (None, 1)                 11
=================================================================        
Total params: 351
Trainable params: 351
Non-trainable params: 0
'''

#3. Compiling, Training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=111, batch_size=1)

#. Evaluating, Prediction
loss = model.evaluate(x, y)
result = model.predict(np.array([5,6,7]).reshape(1,3,1))

print('loss : ', loss)
print('pred : ', result)

'''
loss :  0.03136453032493591
pred :  [[8.050576]]
'''
