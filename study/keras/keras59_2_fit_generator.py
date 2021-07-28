import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_IDG = ImageDataGenerator(rescale=1./255, 
                                                        horizontal_flip=True, 
                                                        vertical_flip=True, 
                                                        width_shift_range=0.1, 
                                                        height_shift_range=0.1,
                                                        rotation_range=5, 
                                                        zoom_range=1.2, 
                                                        shear_range=0.7, 
                                                        fill_mode='nearest')
test_IDG = ImageDataGenerator(rescale=1./255) # scale은 train, test 동일

xy_train = train_IDG.flow_from_directory('../_data/brain/train', 
                                                                        target_size=(150, 150),
                                                                        batch_size=5,
                                                                        class_mode='binary')
xy_test = test_IDG.flow_from_directory('../_data/brain/test', 
                                                                        target_size=(150, 150),
                                                                        batch_size=5,
                                                                        class_mode='binary')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten

input = Input((150, 150, 3))
c = Conv2D(32, (2,2))(input)
f = Flatten()(c)
output = Dense(1, activation='sigmoid')(f)

model = Model(inputs=input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, # 160 / 5 =32
                                        validation_data=xy_test)
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('acc =', acc[-1])
print('val_acc = ', val_acc[:-1])

