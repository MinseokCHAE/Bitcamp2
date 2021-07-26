import time
import datetime
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. Data Preprocessing
#read_csv 
'''
encoding 해결: 한자 영어 etc...
'''
datasets_train = pd.read_csv('../_data/train_data.csv', header=0)
datasets_test = pd.read_csv('../_data/test_data.csv', header=0)

# null값 제거
datasets_train = datasets_train.dropna(axis=0)
datasets_test = datasets_test.dropna(axis=0)
# print(datasets_train.shape, datasets_test.shape)    # (45654, 3) (9131, 2)

# x, y, x_pred 분류
x = datasets_train.iloc[:, -2]
y = datasets_train.iloc[:, -1]
# print(np.unique(y)) # 0~6

y = to_categorical(y)
# print(np.unique(y)) # 0, 1

y = np.argmax(y, axis=1)
# print(np.unique(y)) # 0~6
