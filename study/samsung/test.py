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

# 역순정렬 및 null값 제거
datasets_sk = datasets_sk[::-1]
datasets_samsung = datasets_samsung[::-1]
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
# print(scaled_ratio[0], np.min(data1_ss)[0] ) # 시가에 해당하는 값(1번째 값)

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
target = data_ss[:, [-5]] # 두번째 target =  '삼성' + '시가' = data_ss + 뒤에서 5번째 열 
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
    y.append(target[i + (size - 2 )]) 
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
