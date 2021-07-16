import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                                            index_col=None, header=0)
# ./ : 현재폴더
# ../ : 상위폴더
'''
print(datasets.shape) # (4898, 12)
print(datasets.info())
print(datasets.describe())

1) pandas -> numpy
2) separate x & y
3) check # of y labels : np.unique(y)
'''
x = datasets.iloc[:, 0:11]
y = datasets.iloc[:, [11]]
# print(np.unique(y)) # [3 4 5 6 7 8 9]
# y = to_categorical(y) # 0 1 2 자동채움 -> label : 10
# print(y.shape) # (4898, 10)
onehot_encoder = OneHotEncoder() # 0 1 2 자동채움X
onehot_encoder.fit(y)
y = onehot_encoder.transform(y).toarray() 
# print(y.shape) # (4898, 7)

rs = 77
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=rs)
scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
input = Input(shape=(11, ))
x = Dense(256, activation='relu')(input)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(7, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

#3. compiling, training
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=64, 
                                validation_split=0.001, callbacks=[es])

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('scaler = ', scaler)
print('rs = ', rs)
print('loss = ', loss[0])
print('mse = ', loss[1])
print('accuracy = ', loss[2])

'''
scaler =  RobustScaler()
rs = 24
loss =  2.893921375274658
mse =  0.08134270459413528
accuracy =  0.6816326379776001

scaler =  RobustScaler()
rs =  47
loss =  2.288007974624634
mse =  0.05812676623463631
accuracy =  0.7755101919174194

scaler =  QuantileTransformer()
rs =  47
loss =  1.7584642171859741
mse =  0.05944771692156792
accuracy =  0.7346938848495483

scaler =  QuantileTransformer()
rs =  77
loss =  1.699642300605774
mse =  0.06264209002256393
accuracy =  0.7551020383834839
'''
