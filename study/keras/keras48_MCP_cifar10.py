import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


#1. data preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape((50000, 32 * 32 * 3))
x_test = x_test.reshape((10000, 32 * 32 * 3))
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape((50000, 32*32, 3))
x_test = x_test.reshape((10000, 32*32, 3))

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# onehot_encoder = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# y_train = onehot_encoder.fit_transform(y_train).toarray()
# y_test = onehot_encoder.fit_transform(y_test).toarray()

# #2. modeling
# input = Input(shape=(32 * 32 , 3))
# x = Conv1D(64, (2,), activation='relu')(input)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = MaxPooling1D()(x)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = Conv1D(256, (2,), activation='relu')(x)
# x= Dropout(0.2)(x)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = Conv1D(32, (2,), activation='relu')(x)
# x = GlobalAveragePooling1D()(x)
# output = Dense(10, activation='softmax')(x)

# model = Model(inputs=input, outputs=output)

#3. compiling, training
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                                         filepath='./_save/MCP/keras48_MCP_cifar10.hdf5')
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
# model.compile(loss='categorical_crossentropy', optimizer='adam', 
#                         metrics=['acc'])
start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=1, batch_size=1024, 
#                                 validation_split=0.001, callbacks=[es, cp])
end_time = time.time() - start_time

model = load_model('./_save/MCP/keras48_MCP_cifar10.hdf5')

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
print('time taken(s) : ', end_time)

'''
loss =  2.0624825954437256
accuracy =  0.21160000562667847
time taken(s) :  15.468240022659302

load_MCP
loss =  2.0115280151367188
accuracy =  0.2433999925851822
'''


