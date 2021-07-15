import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. data preprocessing
datasets = load_iris()
x = datasets.data
y = datasets.target

'''
print(x.shape) # (150, 4)
print(y.shape) # (150, )
print(y[:20]) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print(np.unique(y)) # [0 1 2]
'''

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y[:5]) # [[1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.]]
print(y.shape) # (150, 3)

'''
y_data one-hot-encoding 
0 -> [ 1, 0, 0 ]
1 -> [ 0, 1, 0 ]
2 -> [ 0, 0, 1 ]
y.shpae = (150, ) -> (150, 3) 라벨의 수 (여기선 0,1,2 총 3개) 만큼 열 생성
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
input = Input(shape=(4, ))
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
loss =  0.08512041717767715
mse =  0.016447804868221283
accuracy =  0.9666666388511658

[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
[[9.30347771e-04 9.96776760e-01 2.29283888e-03]
 [3.46159068e-04 9.87790823e-01 1.18630165e-02]
 [4.19381598e-04 9.72085595e-01 2.74949633e-02]
 [9.99824703e-01 1.75249239e-04 9.49099288e-08]
 [5.59110194e-04 9.96452332e-01 2.98855011e-03]]
'''
