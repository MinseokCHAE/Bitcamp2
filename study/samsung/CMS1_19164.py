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
# read_csv
datasets_sk = pd.read_csv('../_data/SK주가 20210721.csv', header=0, usecols=[1,2,3,4,10], nrows=2602, encoding='EUC-KR')
datasets_samsung = pd.read_csv('../_data/삼성전자 주가 20210721.csv', header=0, usecols=[1,2,3,4,10], nrows=2602, encoding='EUC-KR')

# null값 제거
datasets_sk = datasets_sk.dropna(axis=0)
datasets_samsung = datasets_samsung.dropna(axis=0)
# print(datasets_sk.head())
# print(datasets_samsung.head())

# 주가4종류 (시가,고가,저가,종가) 와 거래량의 수치차이가 크기때문에 따로 MinMaxScaling - samsung
data1_ss = datasets_samsung.iloc[:, :-1] # 주가4종 추출
data2_ss = datasets_samsung.iloc[:, -1:] # 거래량 추출 
# print(data1_ss.head(), data2_ss.head())
scaler = MinMaxScaler()
scaler.fit(data1_ss)
data1_ss_scaled = scaler.transform(data1_ss) # scaled_ratio, bias를 구하기위해 naming 분리 (data1_ss 원본필요)
scaler.fit(data2_ss)
data2_ss = scaler.transform(data2_ss)
# print(data1_ss_scaled, data2_ss)
data_ss = np.concatenate((data1_ss_scaled, data2_ss), axis=1) # 병합 (주가4종 오른쪽 열에 거래량 추가)
# print(data_ss.shape) # (2602, 5)
scaled_ratio = np.max(data1_ss) - np.min(data1_ss) 
# print(scaled_ratio[3], np.min(data1_ss) [3]) # 종가에 해당하는 값(4번째 값)

# 상동 - sk
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
target = data_ss[:, [-2]] # 첫번째 target =  '삼성' + '종가' = data_ss + 뒤에서 2번째 열 
# print(target.shape) # (2602, 1)
'''
************************Change Here************************
'''

# LSTM 처리를 위한 data_split (= split_x)
x1 = [] # 삼성
x2 = [] # SK
y = [] # target
size = 50 # 데이터 slice 단위일수 설정 (단위일수만큼 끊어서 저장) -> 단위일수만큼의 데이터를 가지고 그 다음날의 주가 예측
for i in range(len(target) - size + 1):
    x1.append(data_ss[i: (i + size) ])
    x2.append(data_sk[i: (i + size) ])
    y.append(target[i + (size - 1)]) 
# 설정한 단위일수 만큼 최근 데이터 slice -> y_predict 를 위한 x1_pred, x2_pred 생성
x1_pred = [data_ss[len(data_ss) - size : ]]
x2_pred = [data_sk[len(data_ss) - size : ]]

# numpy 배열화
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
x1_pred = np.array(x1_pred)
x2_pred = np.array(x2_pred)
print(x1.shape, x2.shape, y.shape, x1_pred.shape, x2_pred.shape) # (2553, 50, 5) (2553, 50, 5) (2553, 1) (1, 50, 5) (1, 50, 5)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, random_state=9)

#2. Modeling
input1 = Input(shape=(50, 5))
d1 = LSTM(16, activation='relu')(input1)

input2 = Input(shape=(50, 5))
d2 = LSTM(16, activation='relu')(input2)

from tensorflow.keras.layers import concatenate
m = concatenate([d1, d2])
d3 = Dense(4, activation='relu')(m)
output = Dense(1, activation='relu')(d3)

model = Model(inputs=[input1, input2], outputs=output)

#3. Compiling, Training
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/MCP/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'CMS1_12345', '_', date_time, '_', info, '.hdf5'])
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)
es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=4)

start_time = time.time()
model.fit([x1_train, x2_train], y_train, epochs=16, batch_size=8, verbose=1, validation_split=0.001, callbacks=[es, cp])
end_time = time.time() - start_time

#4. Evaluating, Prediction
loss = model.evaluate([x1_test, x2_test], y_test)
y_pred = model.predict([x1_pred, x2_pred])
# y_pred = scaler.inverse_transform(y_pred) # 가격(원) 확인을 위한 inverse scaling
y_pred = y_pred * scaled_ratio[3] + np.min(data1_ss)[3]
'''
scaler.inverse_transform을 할때 data1_ss 의 scaling 정보가 필요
위에서 scaling을 여러번함 & 특정 scaler.fit 정보만 뽑아 적용하는 방법 모름
무지성 수식계산으로 땜빵
-> 사후활용을 위해 scaler를 저장하는 기능 있다함 
-> import joblib // joblib.dump(scaler, 'scaler.save') // scaler = joblib.load('scaler.save')
배운것만 쓰라고 해서 안씀
'''

print('loss = ', loss)
print("Tomorrow's closing price = ", y_pred)
print('time taken(s) : ', end_time)

'''
loss =  9.271289309253916e-05
Tomorrow's closing price =  [[19187.96054423]]
time taken(s) :  267.97529006004333
'''
