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

# model: vgg16
def tune_pretrained_model(model_name='vgg16',
    hidden_layers=2,
    hidden_neurons=1024,
    loss_function='mean_squared_error',
    top_epochs=100,
    batch_size=32,
    optimizer='adagrad',
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
    checkpoint_path1 = os.path.join('checkpoints', model_name+'{epoch:02d}-{val_loss:.3f}.mod')
    checkpoint_path2 = os.path.join('checkpoints', model_name+'tuning'+'{epoch:02d}-{val_loss:.3f}.mod')

    # vgg16
    if model_name == 'vgg16':
        base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
        # for layer in base_model.layers:
        #    print(layer.get_config()['name'] + '\n')
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
    if validation_split == 0:
        print('no validation')
        callbacks = [ModelCheckpoint(checkpoint_path1, period=5)]
    else:
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience),
                     ModelCheckpoint(checkpoint_path1, period=5)]
    model.fit(train_X, train_Y,
        epochs=top_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks
    )
    save_path1 = os.path.join('checkpoints', model_name + '_trained')
    model.save(save_path1)

train_pretrained_model(hidden_layers=3,
    hidden_neurons=512,
    loss_function='binary_crossentropy',
    top_epochs=400,
    batch_size=16,
    optimizer=Adagrad(lr=.0001),
    patience=5,
    model_name='xception',
    validation_split=0.1,
    dropout=.2)
