import numpy as np
import os
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def train_own_model(model_name='model',
                    epochs=10,
                    batch_size=32,
                    initial_epoch=0):
    # load training data
    train_X = np.load('training_data/train_X.npy').astype('float32')
    train_Y = np.load('training_data/train_Y.npy')

    # maybe preprocess (mean subtraction)
    preprocess_path = os.path.join('training_data', 'preprocess.txt')
    with open(preprocess_path, 'r') as preprocess_file:
        dictionary = ast.literal_eval(preprocess_file.read())
        preprocess_method = dictionary['preprocess']
    if preprocess_method in ['mean_image', 'mean_pixel']:
        mean_path = os.path.join('training_data', 'means.npy')
        mean = np.load(mean_path)
        train_X -= mean

    #load model
    model_file = os.path.join('model', model_name + '.mod')
    model = load_model(model_file)

    # fit model
    checkpoint_path = os.path.join('checkpoints', model_name+'{epoch:02d}-{val_loss:.2f}.mod')
    model.fit(train_X, train_Y,
        epochs=epochs,
        initial_epoch=initial_epoch,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[ModelCheckpoint(checkpoint_path, period=10),
                   EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)]
    )

train_own_model(epochs=20, batch_size=18)
