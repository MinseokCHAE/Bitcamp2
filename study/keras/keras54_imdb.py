import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


#1. Data Preprocessing
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# x
max_len = max(len(i) for i in x_train)
avg_len = sum(map(len, x_train)) / len(x_train)
# print(max_len) # 2494
# print(avg_len) # 238.71364
x_train = pad_sequences(x_train, padding='pre', maxlen=256)
x_test = pad_sequences(x_test, padding='pre', maxlen=256)
# print(x_train.shape, x_test.shape) # (25000, 256) (25000, 256)
# print(np.unique(x_train)) # 0~9999

# y
# print(np.unique(y_train)) # 0, 1
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(np.unique(y_train)) # 0, 1

#2. Modeling
input = Input((256, ))
e = Embedding(10000, 8)(input)
l = LSTM(8, activation='relu')(e)
output = Dense(1, activation='sigmoid')(l)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/_mcp/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'keras54_imdb', '_', date_time, '_', info, '.hdf5'])
es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=4)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_images=True)
start_time = time.time()
model.fit(x_train, y_train, epochs=4, batch_size=512, verbose=1, validation_split=0.01, callbacks=[es, cp, tb])
end_time = time.time() - start_time

#4. Evaluating, Prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('acc = ', loss[1])
print('time taken(s) = ', end_time)

'''
loss =  0.5047570466995239
acc =  0.7961199879646301
time taken(s) =  372.5650370121002
'''
