import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model, load_model, load_weights
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66) 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
model = load_model('./_save/keras46_save_model_1.h5')
model = load_weights('./_save/keras46_save_weights_1.h5')
'''
input = Input(shape=(10, ))
x = Dense(16, activation='relu')(input)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
output = Dense(1, activation='relu')(x)

model = Model(inputs=input, outputs=output)

# model.save('./_save/keras46_save_model_1.h5')
# model.save_weights('./_save/keras46_save_weights_1.h5')

#3. compiling, training
es = EarlyStopping(monitor='val_loss', patience=64, mode='min', verbose=1)
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=128, batch_size=8, 
                            validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

model = load_model('./_save/keras46_save_model_2.h5')
model = load_weights('./_save/keras46_save_weights_2.h5')

# model.save('./_save/keras46_save_model_2.h5')
# model.save_weights('./_save/keras46_save_weights_2.h5')

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

print('loss = ', loss)
print('R2 score = ', r2)
print('time taken(s) : ', end_time)


loss =  3245.237060546875
R2 score =  0.49996661198648285
time taken(s) :  9.564690113067627
'''
