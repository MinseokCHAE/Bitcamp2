from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D
from tensorflow.keras.callbacks import EarlyStopping

input = Input(Conv2D(10, kernel_size=(2,2), shape=(28, 28, 1))) 
'''
10 : output
kernel_size : 자를 단위
shape : 투입
'''

model = Model(inputs=input, outputs=output)
