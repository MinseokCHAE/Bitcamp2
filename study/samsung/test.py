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
datasets_sk = datasets_sk[::-1]
datasets_samsung = datasets_samsung[::-1]

datasets_sk = datasets_sk.dropna(axis=0)
datasets_samsung = datasets_samsung.dropna(axis=0)
print(datasets_sk.head())
print(datasets_samsung.head())