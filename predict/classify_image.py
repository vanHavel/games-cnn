from keras.models import load_model
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input as preprocess_xception
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import ast

# classify some images
# image_paths: lists of paths to jpg images
# model_path: path to keras model
# cutoff_file: path to threshold file
def classify_image(image_paths=['img.jpg'],
    model_path=os.path.join('checkpoints', 'xception_trained'),
    cutoff_file='cutoffs.npy'):
    # load model
    model = load_model(model_path)

    # read genre file 
    genre_file_path = os.path.join('training_data', 'genres.txt')
    with open(genre_file_path, 'r') as handler:
        genres = handler.readlines()

    # determine preprocess method
    preprocess_path = os.path.join('training_data', 'preprocess.txt')
    with open(preprocess_path, 'r') as preprocess_file:
        dictionary = ast.literal_eval(preprocess_file.read())
        preprocess_method = dictionary['preprocess']
    if preprocess_method == 'xception':
        preprocess = preprocess_xception
    elif preprocess_method == 'vgg':
        preprocess = imagenet_utils.preprocess_input
    elif preprocess_method == 'none':
        preprocess = lambda x:x

    # preprocess images
    input_shape = model.layers[0].input_shape
    dimension = (input_shape[1], input_shape[2])
    screenshots = [process_screen(image_path, dimension, preprocess) for image_path in image_paths]

    # load cutoffs
    cutoffs = np.load(os.path.join('cutoffs', cutoff_file))

    # predict classes
    predictions = model.predict(np.array(screenshots))
    for prediction in predictions:
        print(prediction)
        classes = [i for i in range(0, len(prediction)) if prediction[i] >= cutoffs[i]]
        print('Predicted genres:')
        for c in classes:
            print(genres[c][:-1])

# preprocess a single screen
def process_screen(screen_file, dimension, preprocess):
    screenshot = load_img(screen_file, target_size=dimension)
    screenshot = img_to_array(screenshot)
    screenshot = np.expand_dims(screenshot, axis=0)
    screenshot = preprocess(screenshot)
    screenshot = screenshot[0]
    return screenshot

classify_image()
