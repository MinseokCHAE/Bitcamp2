from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

model1 = Sequential()
model1.add(Conv2D(10, kernel_size=(2,2), input_shape=(5,5,1))) # (N, 4, 4, 10)
# 10: output node, kernel_size: 자를 단위(생략가능), shape: 원본size
model1.add(Flatten()) #(N, 160)
# model1.summary()
'''
Model: "sequential"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________        
flatten (Flatten)            (None, 160)               0
=================================================================        
Total params: 50
Trainable params: 50
Non-trainable params: 0
'''
model2 = Sequential()
model2.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # (N, 4, 4, 10)
model2.add(Conv2D(20, (2,2))) # (N, 3, 3, 20)
model2.add(Flatten()) #(N, 180)
# model2.summary()
'''
Model: "sequential"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
conv2d_1 (Conv2D)            (None, 4, 4, 10)          50
_________________________________________________________________        
conv2d_2 (Conv2D)            (None, 3, 3, 20)          820
_________________________________________________________________        
flatten_1 (Flatten)          (None, 180)               0
=================================================================        
Total params: 870
Trainable params: 870
Non-trainable params: 0
_________________________
'''
model3 = Sequential()
model3.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # (N, 4, 4, 10)
model3.add(Conv2D(20, (2,2), activation='relu')) # (N, 3, 3, 20)
model3.add(Flatten()) #(N, 180)
model3.add(Dense(64, activation='relu'))
model3.add(Dense(32))
model3.add(Dense(1, activation='sigmoid'))
# model3.summary()
'''
Model: "sequential_2"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
conv2d_3 (Conv2D)            (None, 4, 4, 10)          50
_________________________________________________________________        
conv2d_4 (Conv2D)            (None, 3, 3, 20)          820
_________________________________________________________________        
flatten_2 (Flatten)          (None, 180)               0
_________________________________________________________________        
dense (Dense)                (None, 64)                11584
_________________________________________________________________        
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________        
dense_2 (Dense)              (None, 1)                 33
=================================================================        
Total params: 14,567
Trainable params: 14,567
Non-trainable params: 0
'''

input = Input(shape=(5, 5, 1))
x = Conv2D(10, (2,2))(input)
x = Conv2D(20, (2,2), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(32)(x)
output = Dense(1, activation='sigmoid')(x)

model4 = Model(inputs=input, outputs=output)
model4.summary()
'''
Model: "model"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
input_1 (InputLayer)         [(None, 5, 5, 1)]         0
_________________________________________________________________        
conv2d_5 (Conv2D)            (None, 4, 4, 10)          50 = ((2 * 2 * 1) + 1) * 10
_________________________________________________________________        
conv2d_6 (Conv2D)            (None, 3, 3, 20)          820 = ((2 * 2 * 10) + 1) * 20
_________________________________________________________________        
flatten_3 (Flatten)          (None, 180)               0
_________________________________________________________________        
dense_3 (Dense)              (None, 64)                11584
_________________________________________________________________        
dense_4 (Dense)              (None, 32)                2080
_________________________________________________________________        
dense_5 (Dense)              (None, 1)                 33
=================================================================        
Total params: 14,567
Trainable params: 14,567
Non-trainable params: 0
'''
