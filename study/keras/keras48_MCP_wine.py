import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
datasets = load_wine()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66) 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# print(x_train.shape, x_test.shape) # (142, 13, 1) (36, 13, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (142, 3) (36, 3)

# onehot_encoder = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# y_train = onehot_encoder.fit_transform(y_train).toarray()
# y_test = onehot_encoder.fit_transform(y_test).toarray()

# #2. modeling
# input = Input(shape=(13, 1))
# x = Conv1D(64, (2,), activation='relu')(input)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = MaxPooling1D()(x)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = Conv1D(256, (2,), activation='relu')(x)
# x= Dropout(0.2)(x)
# x = Conv1D(128, (2,), activation='relu')(x)
# x = Conv1D(32, (2,), activation='relu')(x)
# x = GlobalAveragePooling1D()(x)
# output = Dense(3, activation='softmax')(x)

# model = Model(inputs=input, outputs=output)

#3. compiling, training
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                                         filepath='./_save/MCP/keras48_MCP_wine.hdf5')
# es = EarlyStopping(monitor='val_loss', patience=32, mode='min', verbose=1)
# model.compile(loss='categorical_crossentropy', optimizer='adam', 
#                         metrics=['acc'])
start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=128, batch_size=16, 
#                             validation_split=0.05, callbacks=[es, cp])
end_time = time.time() - start_time

model = load_model('./_save/MCP/keras48_MCP_wine.hdf5')

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test, batch_size=64)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
print('time taken(s) : ', end_time)

'''
loss =  0.12640994787216187
accuracy =  0.9722222089767456
time taken(s) :  11.268682479858398

load_MCP
loss =  0.04692402109503746
accuracy =  0.9722222089767456
'''

