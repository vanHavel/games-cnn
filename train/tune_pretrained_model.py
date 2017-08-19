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
    full_epochs=100,
    batch_size=32,
    optimizer='adagrad',
    patience=5,
    validation_split=.2):

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
        # get convolutional output from vgg
        # conv_X = base_model.predict(train_X, batch_size=1024)

    # don't train base layers
    for layer in base_model.layers:
        layer.trainable = False

    # add new top model
    x = base_model.output
    x = Flatten()(x)
    for i in range(0, hidden_layers):
        x = Dense(hidden_neurons, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.5)(x)
    predictions = Dense(classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)
    model.summary()

    # train new top layers
    if loss_function == 'bp_mll':
        loss_function = bp_mll_loss
    model.compile(optimizer=optimizer, metrics=[], loss=loss_function)
    model.fit(train_X, train_Y,
        epochs=top_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience),
                   ModelCheckpoint(checkpoint_path1, period=10)]
    )
    save_path1 = os.path.join('checkpoints', model_name + '_trained')
    model.save(save_path1)

    # fine tune last convolutional layers
    if model_name == 'vgg16':
        for layer in model.layers[15:]:
            layer.trainable = True
    model.summary()

    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=[])
    model.fit(train_X, train_Y,
        epochs=full_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience),
        ModelCheckpoint(checkpoint_path2, period=10)]
    )

    # save final model
    save_path2 = os.path.join('checkpoints', model_name + '_tuned')
    model.save(save_path2)

tune_pretrained_model(hidden_layers=3,
    hidden_neurons=1024,
    loss_function='binary_crossentropy',
    top_epochs=400,
    full_epochs=20,
    batch_size=32,
    optimizer=Adagrad(lr=.0001),
    patience=30)
