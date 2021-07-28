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

# Found 160 images belonging to 2 classes.

# print(xy_train) 
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001FB35B48550>

# print(xy_train[0][0].shape, xy_train[0][1].shape)  # (5, 150, 150, 3) (5,)
# [0][0]:x, [0][1]:y // 5=batch_size // 총 160장 /5 = 32 -> xy_train[0]~[31] // 3=color(RGB)
# if batch_size=11 (이미지 갯수의 약수가 아닐때) , 마지막에 남는 이미지 갯수만큼(xy_train[14][0].shape = (6,150,150,3)
