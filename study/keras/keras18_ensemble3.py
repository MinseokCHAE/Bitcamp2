import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y1 = np.array([range(1001, 1101)])
y2 = np.array([range(1901, 2001)])

x = np.transpose(x) # (100, 3)
y1 = np.transpose(y1) # (100, )
y2 = np.transpose(y2) # (100, )

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x, y1, y2, 
                                                                                        test_size=0.3, random_state=9)

#2. 모델링
#2-1. 모델1
input = Input(shape=(3, ), name='input')
hidden1 = Dense(10, activation='relu', name='hidden1')(input)
hidden2 = Dense(7, activation='relu', name='hidden2')(hidden1)
hidden3 = Dense(5, activation='relu', name='hidden3')(hidden2)
mid_output = Dense(11, name='mid_output')(hidden3)

#2-2. 분배
mid_input1 = Dense(7, name='mid_input1')(mid_output)
hidden1_1 = Dense(7, name='hidden1_1')(mid_input1)
last_output1 = Dense(1, name='last_output1')(hidden1_1)

mid_input2 = Dense(8, name='mid_input2')(mid_output)
hidden2_1 = Dense(7, name='hidden2_1')(mid_input2)
last_output2 = Dense(1, name='last_output2')(hidden2_1)

model = Model(inputs=input, outputs=[last_output1, last_output2])
model.summary()
'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to       

==================================================================================================
input (InputLayer)              [(None, 3)]          0

__________________________________________________________________________________________________
hidden1 (Dense)                 (None, 10)           40          input[0][0]        

__________________________________________________________________________________________________
hidden2 (Dense)                 (None, 7)            77          hidden1[0][0]      

__________________________________________________________________________________________________
hidden3 (Dense)                 (None, 5)            40          hidden2[0][0]      

__________________________________________________________________________________________________
mid_output (Dense)              (None, 11)           66          hidden3[0][0]      

__________________________________________________________________________________________________
mid_input1 (Dense)              (None, 7)            84          mid_output[0][0]   

__________________________________________________________________________________________________
mid_input2 (Dense)              (None, 8)            96          mid_output[0][0]   

__________________________________________________________________________________________________
hidden1_1 (Dense)               (None, 7)            56          mid_input1[0][0]   

__________________________________________________________________________________________________
hidden2_1 (Dense)               (None, 7)            63          mid_input2[0][0]   

__________________________________________________________________________________________________
last_output1 (Dense)            (None, 1)            8           hidden1_1[0][0]    

__________________________________________________________________________________________________
last_output2 (Dense)            (None, 1)            8           hidden2_1[0][0]    

==================================================================================================
Total params: 538
Trainable params: 538
Non-trainable params: 0
'''

#3. 컴파일, 훈련
num_epochs = 100
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, [y1_train, y2_train], epochs=num_epochs, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate(x_test, [y1_test, y2_test])
#y_predict = model.predict([x1_test, x2_test])
#r2 = r2_score([y1_test, y2_test], y_predict)

print('epochs = ', num_epochs)
print('loss = ', results[0])
#print('r2 score =', r2)

'''
epochs = 100
loss =  112.97917175292969
'''
