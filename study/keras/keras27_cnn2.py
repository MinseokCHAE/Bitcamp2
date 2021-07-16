from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

input = Input(shape=(10, 10, 1)) # (N, 10, 10, 1)
x = Conv2D(10, (2,2), padding='same')(input) # (N, 10, 10, 10)
x = Conv2D(20, (2,2), padding='valid', activation='relu')(x) # (N, 9, 9, 20)
'''
padding='valid'가 default -> kernal_size로 filter 자름 filter_width,height = 10-2+1 = 9
padding='same' -> kernel_size상관없이 filter 크기 유지
'''
x = Conv2D(30, (2,2))(x) # (N, 8, 8, 30)
x = MaxPooling2D()(x) # (N, 4, 4, 30) maxpooling : filter 크기 반띵
x = Conv2D(15, (2,2))(x) # (N, 3, 3, 15)
x = Flatten()(x) # (N, 480)
x = Dense(64, activation='relu')(x)
x = Dense(32)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)
model.summary()
'''
Model: "model"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
input_1 (InputLayer)         [(None, 10, 10, 1)]       0
_________________________________________________________________        
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________        
conv2d_1 (Conv2D)            (None, 9, 9, 20)          820
_________________________________________________________________        
conv2d_2 (Conv2D)            (None, 8, 8, 30)          2430
_________________________________________________________________        
max_pooling2d (MaxPooling2D) (None, 4, 4, 30)          0
_________________________________________________________________        
conv2d_3 (Conv2D)            (None, 3, 3, 15)          1815
_________________________________________________________________        
flatten (Flatten)            (None, 135)               0
_________________________________________________________________        
dense (Dense)                (None, 64)                8704
_________________________________________________________________        
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________        
dense_2 (Dense)              (None, 1)                 33
=================================================================        
Total params: 15,932
Trainable params: 15,932
Non-trainable params: 0
'''
