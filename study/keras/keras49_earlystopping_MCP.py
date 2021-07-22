import numpy as np
from sklearn.metrics import r2_score
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. Data Preprocessing
x1 = np.array([range(100), range(301, 401), range(1, 101)])
y1 = np.array([range(101, 201), range(411, 511), range(100, 200)])
z1 = np.array([range(1001, 1101)])

x2 = np.transpose(x1) # (100, 3)
y2 = np.transpose(y1) # (100, 3)
z2 = np.transpose(z1) # (100, )

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x2, y2, z2, 
                                                                                        test_size=0.3, random_state=9)

#2. Modeling
input1 = Input(shape=(3, ), name='input1')
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3)

input2 = Input(shape=(3, ), name='input2')
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='merge1')
merge2 = Dense(10, name='merge2')(merge1)
merge3 = Dense(5, activation='relu', name='merge3')(merge2)
last_output = Dense(1, name='last_output')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

#3. Compiling, Traning
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/MCP/'
info = '{epoch:04d}_{val_loss:.4f}'
filepath = ''.join([path, 'keras49_es_MCP', '_', date_time, '_', info, '.hdf5'])

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1,
                                        filepath=filepath)
es = EarlyStopping(monitor='val_loss', patience=64, mode='auto', verbose=1,
                                restore_best_weights=False)

start_time = time.time()
model.fit([x_train, y_train], z_train, epochs=128, batch_size=2, verbose=1, 
                validation_split=0.001, callbacks=[es, cp])
end_time = time.time() - start_time

model.save('./_save/MODEL/keras49_es_MCP.h5')

#4. Evaluating, Prediction
'''
print('=====None=====')
loss = model.evaluate([x_test, y_test], z_test)
z_predict = model.predict([x_test, y_test])
r2 = r2_score(z_test, z_predict)
print('mse = ', loss[0])
print('mae = ', loss[1])
print('r2 score =', r2)
print('time taken(s) : ', end_time)

print('=====MODEL=====')
model2 = load_model('./_save/MODEL/keras49_es_MCP.h5')
loss = model2.evaluate([x_test, y_test], z_test)
z_predict = model2.predict([x_test, y_test])
r2 = r2_score(z_test, z_predict)
print('mse = ', loss[0])
print('mae = ', loss[1])
print('r2 score =', r2)

print('=====MCP=====')
model3 = load_model('./_save/MCP/keras49_es_MCP.hdf5')
loss = model3.evaluate([x_test, y_test], z_test)
z_predict = model3.predict([x_test, y_test])
r2 = r2_score(z_test, z_predict)
print('mse = ', loss[0])
print('mae = ', loss[1])
print('r2 score =', r2)



=====None=====
1/1 [==============================] - ETA: 0s - loss: 6.4213e-05 - mae: 
1/1 [==============================] - 0s 14ms/step - loss: 6.4213e-05 - 
mae: 0.0070
mse =  6.421344733098522e-05
mae =  0.0070170084945857525
r2 score = 0.9999998971040908
time taken(s) :  27.535815715789795
=====MODEL=====
1/1 [==============================] - ETA: 0s - loss: 6.4213e-05 - mae: 
1/1 [==============================] - 0s 93ms/step - loss: 6.4213e-05 - 
mae: 0.0070
mse =  6.421344733098522e-05
mae =  0.0070170084945857525
r2 score = 0.9999998971040908
=====MCP=====
1/1 [==============================] - ETA: 0s - loss: 0.0099 - mae: 0.081/1 [==============================] - 0s 91ms/step - loss: 0.0099 - mae: 0.0876
mse =  0.009907972067594528
mae =  0.08756306767463684
r2 score = 0.9999841234229687
'''
