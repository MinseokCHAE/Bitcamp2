import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_breast_cancer()
'''
print(datasets)
print(datasets.keys())
print(datasets.feature_names)
print(datasets.DESCR)
'''
x = datasets.data
y = datasets.target
'''
print(x.shape) # (569, 30)
print(y.shape) # (569, )
print(y[:20]) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
print(np.unique(y)) # [0 1]
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델링
input = Input(shape=(30, ))
x = Dense(128, activation='relu')(input)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
num_epochs = 100
model.compile(loss='binary_crossentropy', optimizer='adam', 
                        metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=8, 
                    validation_split=0.1, callbacks=[es])

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('epochs = ', num_epochs)
print('loss = ', loss[0])
print('mse = ', loss[1])
print('accuracy = ', loss[2])
print('r2 score = ', r2)

'''
epochs =  100
loss =  0.3202535808086395
mse =  0.038858525454998016
accuracy =  0.9561403393745422
r2 score =  0.8312711562332203
'''
