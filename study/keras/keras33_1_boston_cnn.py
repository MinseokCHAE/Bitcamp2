import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66) 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# print(x_train.shape, x_test.shape) # ((404, 13, 1) (102, 13, 1)
# print(y_train.shape, y_test.shape) # (404,) (102,)

#2. modeling
input = Input(shape=(13, 1))
x = Conv1D(64, (1,), activation='relu')(input)
x = Conv1D(128, (1,), activation='relu')(x)
x = MaxPooling1D()(x)
x = Conv1D(128, (1,), activation='relu')(x)
x = Conv1D(256, (1,), activation='relu')(x)
x= Dropout(0.2)(x)
x = Conv1D(128, (2,), activation='relu')(x)
x = Conv1D(32, (2,), activation='relu')(x)
x = GlobalAveragePooling1D()(x)
output = Dense(1, activation='relu')(x)

model = Model(inputs=input, outputs=output)

#3. compiling, training
es = EarlyStopping(monitor='val_loss', patience=32, mode='min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=128, batch_size=16, 
                            validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test, batch_size=64)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
print('time taken(s) : ', end_time)

#5. plt visualization
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
loss =  nan
accuracy =  0.0
time taken(s) :  7.862398386001587
'''
