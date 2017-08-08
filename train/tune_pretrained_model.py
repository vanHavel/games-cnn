import numpy as np

from keras import applications
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

from bp_mll import bp_mll_loss

# model: vgg16
def tune_pretrained_model(model_name='vgg16',
    hidden_layers=2,
    hidden_neurons=1024,
    loss_function='binary_crossentropy',
    top_epochs=10,
    full_epochs=5,
    batch_size=32):
    # load training data
    train_X = np.load('training_data/train_X.npy')
    train_Y = np.load('training_data/train_Y.npy')
    classes = len(train_Y[0])

    # vgg16
    if model_name == 'vgg16':
        # get convolutional output from vgg
        base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
        # for layer in base_model.layers:
        #    print(layer.get_config()['name'] + '\n')
        # conv_X = base_model.predict(train_X, batch_size=1024)

    # don't train base layers
    for layer in base_model.layers:
        layer.trainable = False

    # add new top model
    x = base_model.output
    x = Flatten()(x)
    for i in range(0, hidden_layers):
        x = Dense(hidden_neurons, kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.5)(x)
    predictions = Dense(classes, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)

    # train new top layers
    if loss_function == 'bp_mll':
        loss_function = bp_mll_loss
    model.compile(optimizer='adagrad', metrics=[], loss=loss_function)
    model.fit(train_X, train_Y,
        epochs=top_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)]
    )

    # fine tune last convolutional layers
    if model_name = 'vgg16':
        for layer in model.layers[15:]:
            layer.trainable = True

    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=[])
    model.fit(train_X, train_Y,
        epochs=full_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)]
    )

    # save final model
    save_path = os.path.join('checkpoints', model_name + '_tuned')
    model.save(save_path)

tune_pretrained_model()
