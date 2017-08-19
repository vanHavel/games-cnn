
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, ZeroPadding2D, MaxPooling2D, Dropout
from keras import backend as K

from bp_mll import bp_mll_loss

import numpy as np

def create_vggnet_small(classes=14):
    model = Sequential()

    model.add(ZeroPadding2D(padding=(1,1), input_shape=(224, 224, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='sigmoid', kernel_initializer='glorot_uniform'))

    return model

def create_vggnet(classes=14):
    model = Sequential()

    model.add(ZeroPadding2D(padding=(1,1), input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='sigmoid', kernel_initializer='glorot_uniform'))

    return model

def compile_model(model, loss_function='binary_crossentropy', optimizer='adagrad', metrics=[]):
    if loss_function == 'bp-mll':
        loss_function = bp_mll_loss
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

model = create_vggnet_small()
compile_model(model)
model.summary()
model.save('model/model.mod')
