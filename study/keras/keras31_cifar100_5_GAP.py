import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train, x_test = x_train.reshape(50000, 32*32*3) / 255, x_test.reshape(10000, 32*32*3) / 255
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train, x_test = x_train.reshape(50000, 32, 32, 3), x_test.reshape(10000, 32, 32, 3)

# print(np.unique(y_train)) # [0 1 ... 98 99]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# onehot_encoder = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# y_train = onehot_encoder.fit_transform(y_train).toarray()
# y_test = onehot_encoder.fit_transform(y_test).toarray()

#2. modeling
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
input = Input(shape=(32, 32, 3))
x = Conv2D(32, (3,3), activation='relu')(input)
x = Dropout(0.2)(x)
x = MaxPooling2D()(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = Dropout(0.2)(x)
x = MaxPooling2D()(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = Dropout(0.2)(x)
x = MaxPooling2D()(x)

x = GlobalAveragePooling2D()(x)
output = Dense(100, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

#3. compiling, training
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=64, batch_size=256, 
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
loss =  2.2278549671173096
accuracy =  0.4449999928474426
time taken(s) :  108.50786089897156
'''
