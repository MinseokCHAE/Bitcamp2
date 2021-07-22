import time
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. Data Preprocessing
datasets_samsung = pd.read_csv('../_data/삼성전자 주가 20210721.csv', header=0, usecols=[1,2,3,4,10], nrows=2602, encoding='EUC-KR')
datasets_sk = pd.read_csv('../_data/SK주가 20210721.csv', header=0, usecols=[1,2,3,4,10], nrows=2602, encoding='EUC-KR')

datasets_samsung = datasets_samsung.dropna(axis=0)
datasets_sk = datasets_sk.dropna(axis=0)
# print(datasets_samsung.head())
# print(datasets_sk.head())

x1 = datasets_samsung.iloc[:, [0,1,2,4]]
y1 = datasets_samsung.iloc[:, 3]
# print(x1.head(), y1.head())
# print(x1.shape, y1.shape) #(2602, 4) (2602,)
x2 = datasets_sk.iloc[:, [0,1,2,4]]
y2 = datasets_sk.iloc[:, 3]
# print(x2.head(), y2.head())
# print(x2.shape, y2.shape) #(2602, 4) (2602,)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, test_size=0.1, random_state=9)
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

#2. Modeling
input1 = Input(shape=(4, ))
s1 = Dense(10, activation='relu')(input1)
s1 = Dense(10, activation='relu')(s1)

input2 = Input(shape=(4, ))
s2 = Dense(10, activation='relu')(input2)
s2 = Dense(10, activation='relu')(s2)

from tensorflow.keras.layers import concatenate
merge = concatenate([s1, s2])
m = Dense(10, activation='relu')(merge)
m = Dense(10, activation='relu')(m)
output = Dense(1, activation='relu')(m)

model = Model(inputs=[input1, input2], outputs=output)

#3. Compiling, Training
es = EarlyStopping(monitor='val_loss', patience=32, mode='min', verbose=1)
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=8, verbose=1)

#4. Evaluating, Prediction
loss = model.evaluate([x1_test, x2_test], y1_test)
y1_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y1_test, y1_predict)

print('loss = ', loss)
print('r2 score = ', r2)

#5. Plt Visualization
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()
'''
