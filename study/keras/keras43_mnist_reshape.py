import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, GlobalAvgPool2D, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


#1. data preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28 *28))
x_test = x_test.reshape((10000, 28* 28))
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape((60000, 28, 28))
x_test = x_test.reshape((10000, 28, 28))

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# onehot_encoder = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# y_train = onehot_encoder.fit_transform(y_train).toarray()
# y_test = onehot_encoder.fit_transform(y_test).toarray()

#2. modeling
from tensorflow.keras.layers import Reshape # 모델링과정에서도 reshape가능
input = Input(shape=(28, 28))
x = Dense(10, activation='relu')(input) # (N, 28, 10)
x = Flatten()(x)    # (N, 280)
x = Dense(784)(x)   # (N, 784)
x = Reshape((28, 28, 1))(x) # (N, 28, 28, 1)
x = Conv2D(64, (2,2))(x)    # (None, 27, 27, 64)
x = MaxPooling2D()(x)   # (None, 13, 13, 64)
x = Flatten()(x)    # (None, 10816) 
output = Dense(10, activation='softmax')(x)     # (None, 10)

model = Model(inputs=input, outputs=output)
model.summary()

'''
#3. compiling, training
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=4, batch_size=128, 
                                validation_split=0.001, callbacks=[es])
end_time = time.time() - start_time

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

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


loss =  0.715210497379303
accuracy =  0.7595000267028809
time taken(s) :  118.75079011917114
'''

