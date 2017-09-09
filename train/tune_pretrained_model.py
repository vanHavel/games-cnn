import numpy as np
import os
import sys

import tensorflow as tf

from keras import applications
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adagrad
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

from bp_mll import bp_mll_loss

# model: vgg16
def tune_pretrained_model(model_path='model.mod',
    model_type='xception',
    hidden_layers=2,
    hidden_neurons=1024,
    loss_function='binary_crossentropy',
    epochs=100,
    batch_size=32,
    optimizer=SGD(lr = .0001, momentum =.9),
    patience=5,
    validation_split=.2,
    dropout=.5):

    # set gpu usage
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    
    # load training data
    train_X = np.load(os.path.join('training_data', 'train_X.npy'))
    train_Y = np.load(os.path.join('training_data', 'train_Y.npy'))
    classes = len(train_Y[0])
    checkpoint_path = os.path.join('checkpoints', model_type+'tuning'+'{epoch:02d}-{val_loss:.3f}.mod')

    # load model
    model = load_model(model_path)

    # don't train base layers
    for layer in model.layers:      
        layer.trainable = False

    # fine tune last convolutional layers
    if model_type == 'vgg16':
        for layer in model.layers[15:]:
            layer.trainable = True
        model.summary()
    if model_type == 'xception':
        for layer in model.layers[106:]:
            layer.trainable = True
        model.summary()

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[])
    model.fit(train_X, train_Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience),
        ModelCheckpoint(checkpoint_path, period=10)]
    )

    # save final model
    save_path = os.path.join('checkpoints', model_type + '_tuned')
    model.save(save_path)

tune_pretrained_model(model_path='checkpoints/xception_trained',
    loss_function='binary_crossentropy',
    epochs=200,
    batch_size=4,
    optimizer=SGD(lr=.001, momentum =.9),
    patience=5,
    model_type='xception',
    validation_split=0.1,
    dropout=.2)
