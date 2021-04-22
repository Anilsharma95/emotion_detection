from tensorflow.keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

train_dir='data/train'
test_dir='data/test'
train_data_gen=ImageDataGenerator(rescale=1./255)
tes_data_gen=ImageDataGenerator(rescale=1./255)
train_gen=train_data_gen.flow_from_directory(train_dir,target_size=(48,48),
                                             batch_size=64,color_mode='grayscale',
                                             class_mode='categorical')
test_gen=tes_data_gen.flow_from_directory(test_dir,target_size=(48,48),
                                             batch_size=64,color_mode='grayscale',
                                             class_mode='categorical')
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
model_info=model.fit_generator(
    train_gen,steps_per_epoch=28709//64,
    epochs=5,
    validation_data=test_gen,
    validation_steps=7178//64
)
model.save_weights('model.h5')
