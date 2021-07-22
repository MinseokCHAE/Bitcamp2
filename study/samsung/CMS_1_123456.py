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

#가격과 거래량 수치차이가 크기때문에 따로 MinMaxScaling
data1_ss = datasets_samsung.iloc[:, :-1]
data2_ss = datasets_samsung.iloc[:, -1:]
# print(data1_ss.head(), data2_ss.head())
scaler = MinMaxScaler()
scaler.fit(data1_ss)
data1_ss = scaler.transform(data1_ss)
scaler.fit(data2_ss)
data2_ss = scaler.transform(data2_ss)
# print(data1_ss, data2_ss)
data_ss = np.concatenate((data1_ss, data2_ss), axis=1)
# print(data_ss.shape) # (2602, 5)

data1_sk = datasets_sk.iloc[:, :-1]
data2_sk = datasets_sk.iloc[:, -1:]
# print(data1_sk.head(), data2_sk.head())
scaler = MinMaxScaler()
scaler.fit(data1_sk)
data1_sk = scaler.transform(data1_sk)
scaler.fit(data2_sk)
data2_sk = scaler.transform(data2_sk)
# print(data1_sk, data2_sk)
data_sk = np.concatenate((data1_sk, data2_sk), axis=1)
# print(data_sk.shape) # (2602, 5)
'''
************************Change Here************************
'''
target = data_ss[:, [-2]] # 첫날 target =  '삼성' + '종가' = data_ss + 뒤에서 2번째 열 
# print(target.shape) # (2602, 1)
'''
************************Change Here************************
'''

x1 = []
x2 = []
y = []
size = 50 # 몇일단위로 자를것인지 설정
for i in range(len(target) - (size + 1)):
    x1.append(data_ss[i: i + (size + 1)])
    y.append(target[i + (size + 1)])
for i in range(len(target) - (size + 1)):
    x2.append(data_sk[i: i + (size + 1)])
x1_pred = [data_ss[len(data_ss) - ( size + 1): ]]
x2_pred = [data_sk[len(data_ss) - ( size + 1): ]]

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
x1_pred = np.array(x1_pred)
x2_pred = np.array(x2_pred)
# print(x1.shape, x2.shape, y.shape, x1_pred.shape, x2_pred.shape) # (2551, 51, 5) (2551, 51, 5) (2551, 1) (1, 51, 5) (1, 51, 5)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, random_state=9)

#2. Modeling
input1 = Input(shape=(51, 5))
d1 = LSTM(16, activation='relu')(input1)

input2 = Input(shape=(51, 5))
d2 = LSTM(16, activation='relu')(input2)

from tensorflow.keras.layers import concatenate
m = concatenate([d1, d2])
output = Dense(1, activation='relu')(m)

model = Model(inputs=[input1, input2], outputs=output)

#3. Compiling, Training
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/MCP/'
info = '{epoch:04d}_{val_loss:.4f}'
filepath = ''.join([path, 'test', '_', date_time, '_', info, '.hdf5'])
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)
es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=4)

start_time = time.time()
model.fit([x1_train, x2_train], y_train, epochs=8, batch_size=16, verbose=1, validation_split=0.001, callbacks=[es, cp])
end_time = time.time() - start_time

#4. Evaluating, Prediction
loss = model.evaluate([x1_test, x2_test], y_test)
y_pred = model.predict([x1_pred, x2_pred])
y_pred = scaler.inverse_transform(y_pred)

print('loss = ', loss)
print("Tomorrow's stock price = ", y_pred)
print('time taken(s) : ', end_time)

'''

'''

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
