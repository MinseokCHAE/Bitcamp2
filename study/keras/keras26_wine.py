import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
datasets = load_wine()
x = datasets.data
y = datasets.target
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
input = Input(shape=(13, ))
x = Dense(128, activation='relu')(input)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

#3. compiling, training
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
num_epochs = 100
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=8, 
                    validation_split=0.05, callbacks=[es])

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test[:5])

print('epochs = ', num_epochs)
print('loss = ', loss[0])
print('mse = ', loss[1])
print('accuracy = ', loss[2])

print(y_test[:5])
print(y_predict)

'''
epochs =  100
loss =  0.1287219375371933
mse =  0.017529642209410667
accuracy =  0.9722222089767456

[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
[[3.0066875e-07 3.3070253e-05 9.9996662e-01]
 [1.0857791e-12 1.0000000e+00 8.0992981e-12]
 [1.3832664e-15 1.0000000e+00 1.5172525e-14]
 [9.9999261e-01 1.7212353e-06 5.6957720e-06]
 [1.0063572e-05 9.9998999e-01 4.7432557e-08]]
'''
