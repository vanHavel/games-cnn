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

from bp_mll import bp_mll_loss

# train a model pretrained on ImageNet
# model_name: type of model. One of 'vgg16', 'xception', 'inception'
# hidden_layers: number of hidden dense layers on top of CNN
# hidden_neurons: number of neurons per hidden layer
# loss_function: loss function to use
# top_epochs: epochs to train
# batch_size: mini batch size for training
# optimizer: optimization algorithm to use
# patience: epochs to wait for decrease in validation loss before early stopping
# validation_split: fraction of training data to be used for validation
# dropout: reset probability for dropout layers
def train_pretrained_model(model_name='xception',
    hidden_layers=2,
    hidden_neurons=512,
    loss_function='binary_crossentropy',
    top_epochs=100,
    batch_size=16,
    optimizer=Adagrad(lr=.0001),
    patience=3,
    validation_split=.1,
    dropout=.2):

    # set gpu usage limits. You can remove this if you have enough memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    
    # load training data
    train_X = np.load(os.path.join('training_data', 'train_X.npy'))
    train_Y = np.load(os.path.join('training_data', 'train_Y.npy'))
    classes = len(train_Y[0])
    checkpoint_path = os.path.join('checkpoints', model_name+'{epoch:02d}-{val_loss:.3f}.mod')

    # vgg16
    if model_name == 'vgg16':
        base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
    # xception
    elif model_name == 'xception':
        base_model = applications.Xception(include_top=False, weights='imagenet', input_shape=(299,299,3))
    # inception
    elif model_name == 'inception':
        base_model = applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3))


    # don't train base layers
    for layer in base_model.layers:
        layer.trainable = False

    # add new top model
    x = base_model.output
    x = Flatten()(x)
    for i in range(0, hidden_layers):
        x = Dense(hidden_neurons, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(dropout)(x)
    predictions = Dense(classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)
    model.summary()

    # train new top layers
    if loss_function == 'bp_mll':
        loss_function = bp_mll_loss
    model.compile(optimizer=optimizer, metrics=[], loss=loss_function)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience),
                 ModelCheckpoint(checkpoint_path, period=5)]
    model.fit(train_X, train_Y,
        epochs=top_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks
    )
    save_path = os.path.join('checkpoints', model_name + '_trained')
    model.save(save_path)

train_pretrained_model()
