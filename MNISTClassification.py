

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import os


# hyperparameters
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



input = Input(shape=(28,28,1))
conv1 = Conv2D(32, kernel_size=3,padding='same', activation='relu')(input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
dropout1 = Dropout(0.5)(pool1)
conv2_1 = Conv2D(64, kernel_size=3, padding='same',activation='relu')(dropout1)
conv2_2 = Conv2D(64, kernel_size=3,padding='same', activation='relu')(dropout1)
pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
pool2_2= MaxPooling2D(pool_size=(2, 2))(conv2_2)

conv3_1 = Conv2D(512, kernel_size=3, padding='same',activation='relu')(pool2_1)
conv3_2 = Conv2D(512, kernel_size=3,padding='same', activation='relu')(pool2_2)
added = keras.layers.Add()([conv3_1, conv3_2])
flattened= Flatten()(added)
fc1 = Dense(1000, activation='relu')(flattened)
fc2 = Dense(500, activation='relu')(fc1)
output = Dense(10, activation='sigmoid')(fc2)


model = Model(inputs=input, outputs=output)
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
# saving the model
model.save("keras_mnist.h5")

print('Test loss:', score[0])
print('Test accuracy:', score[1])
