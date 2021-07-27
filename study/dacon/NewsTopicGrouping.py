import time
import datetime
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. Data Preprocessing
#read_csv 
'''
encoding 해결: 한자 영어 etc...
'''
datasets_train = pd.read_csv('../_data/train_data.csv', header=0)
datasets_test = pd.read_csv('../_data/test_data.csv', header=0)
submission = pd.read_csv('../_data/sample_submission.csv', header=0)

# null값 제거
datasets_train = datasets_train.dropna(axis=0)
datasets_test = datasets_test.dropna(axis=0)
# print(datasets_train.shape, datasets_test.shape)    # (45654, 3) (9131, 2)

# x, y, x_pred 분류
x = datasets_train.iloc[:, -2]
y = datasets_train.iloc[:, -1]
x_pred = datasets_test.iloc[:, -1]
# print(x.head(), y.head(), x_pred.head())

# x, x_pred 토큰화 및 sequence화
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x_pred = token.texts_to_sequences(x_pred)
# print(x, x_pred)

# x, x_pred padding
max_len1 = max(len(i) for i in x)
avg_len1 = sum(map(len, x)) / len(x)
max_len2 = max(len(i) for i in x_pred)
avg_len2 = sum(map(len, x_pred)) / len(x_pred)
# print(max_len1, max_len2) # 13 11
# print(avg_len1, avg_len2) # 6.623954089455469 5.127696856861242
x = pad_sequences(x, padding='pre', maxlen=14)
x_pred = pad_sequences(x_pred, padding='post', maxlen=14)
# print(x.shape, x_pred.shape) # (45654, 14) (9131, 14)
# print(np.unique(x), np.unique(x_pred)) # 0~101081 // 동일기준(x)으로 fit,sequence 했기 때문에 같음

# # x, x_pred scaling -> 효과X
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_pred = scaler.transform(x_pred)
# print(x.shape, x_pred.shape) # (45654, 14) (9131, 14)

# y to categorical -> model.fit 에서 적용
# print(np.unique(y)) # 0~6
# y = to_categorical(y)
# print(np.unique(y)) # 0, 1

# 전처리 데이터 npy저장
# np.save('./_save/_npy/NTG_x.npy', arr=x)
# np.save('./_save/_npy/NTG_y.npy', arr=y)
# np.save('./_save/_npy/NTG_x_pred.npy', arr=x_pred)

# x, y train_test_split -> Stratified KFold로 대체
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

#2. Modeling
input = Input((14, ))
e = Embedding(101082, 256)(input)
b1 =Bidirectional(LSTM(64, activation='relu', return_sequences=True))(e)
o1 = Dropout(0.7)(b1)
b2 =Bidirectional(LSTM(64, activation='relu', return_sequences=True))(o1)
o2 = Dropout(0.7)(b2)
b3 =Bidirectional(LSTM(64, activation='relu', return_sequences=False))(o2)
o3 = Dropout(0.7)(b3)
output = Dense(7, activation='softmax')(o3)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/_mcp/'
info = '{epoch:02d}_{val_acc:.4f}'
filepath = ''.join([path, 'a_test', '_', date_time, '_', info, '.hdf5'])
es = EarlyStopping(monitor='val_loss', restore_best_weights=True, mode='auto', verbose=1, patience=2)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)

start_time = time.time()
# model.fit(x_train, y_train, epochs=8, batch_size=512, verbose=1, validation_split=0.1, callbacks=[es, cp])
#Stratified KFlod 적용
new_tech = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
for i, (i_train, i_val) in enumerate(new_tech.split(x, y), 1):
    print(f'training model for CV #{i}')
    model.fit(x[i_train], to_categorical(y[i_train]), epochs=4, batch_size=512, verbose=1, validation_data=(x[i_val], to_categorical(y[i_val])), callbacks=[es, cp])
end_time = time.time() - start_time

# best weight model 적용 부분
# model = load_model('./_save/MCP/test_0.7591.hdf5')

#4. Evaluating -> Stratified KFold 로 대체 
# loss = model.evaluate(x_test, y_test)

# print('loss = ', loss[0])
# print('acc = ', loss[1])
print('time taken(s) = ', end_time)

'''
Epoch 6/8
72/72 [==============================] - 24s 332ms/step - loss: 0.0056 - acc: 0.9989 - val_loss: 
0.0095 - val_acc: 0.9981

Epoch 00006: val_loss did not improve from 0.00303
Epoch 00006: early stopping
time taken(s) =  744.042829990387
'''

#5. Prediction
prediction = np.zeros((x_pred.shape[0], 7)) # predict 값 저장할 곳 생성
prediction += model.predict(x_pred) / 5 # 검증횟수(n_splits)로 나누기
topic_idx = []
for i in range(len(prediction)):
    topic_idx.append(np.argmax(prediction[i]))  # reverse to_categorical 적용후 리스트에 저장

# 제출파일형식 맞추기
submission['topic_idx'] = topic_idx
submission.to_csv('../_data/a_test.csv', index=False)

# index = np.array([range(45654, 54785)])
# index = np.transpose(index)
# index = index.reshape(9131, )
# file = np.column_stack([index, prediction])
# file = pd.DataFrame(file)
# file.to_csv('../_data/a_test.csv', header=['index', 'topic_idx'], index=False)
