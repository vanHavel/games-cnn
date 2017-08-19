from keras.models import load_model
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input as preprocess_xception
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import ast

def classify_image(image_path='img.jpg',
    model_path=os.path.join('model', 'model.mod')):
    # load model
    model = load_model(model_path)

    # determine preprocess method
    preprocess_path = os.path.join('training_data', 'preprocess.txt')
    with open(preprocess_path, 'r') as preprocess_file:
        dictionary = ast.literal_eval(preprocess_file.read())
        preprocess_method = dictionary['preprocess']
    if preprocess_method == 'xception':
        preprocess = preprocess_xception
    elif preprocess_method == 'vgg':
        preprocess = imagenet_utils.preprocess_input
    elif preprocess_method in ['mean_image', 'mean_pixel']:
        mean_path = os.path.join('training_data', 'means.npy')
        mean = np.load(mean_path)
        preprocess = lambda x: x - mean
    elif preprocess_method == 'none':
        preprocess = lambda x:x

    # preprocess image
    input_shape = model.layers[0].input_shape
    dimension = (input_shape[1], input_shape[2])
    screenshot = process_screen(image_path, dimension, preprocess)

    # predict classes
    prediction = model.predict(np.array([screenshot]))[0]
    print(prediction)
    classes = [i for i in range(0, len(prediction)) if prediction[i] >= 0.5]

    # read genre file and output genres
    genre_file_path = os.path.join('training_data', 'genres.txt')
    with open(genre_file_path, 'r') as handler:
        genres = handler.readlines()
    print('Predicted genres:')
    for c in classes:
        print(genres[c][:-1])

def process_screen(screen_file, dimension, preprocess):
    screenshot = load_img(screen_file, target_size=dimension)
    screenshot = img_to_array(screenshot)
    screenshot = np.expand_dims(screenshot, axis=0)
    screenshot = preprocess(screenshot)
    screenshot = screenshot[0]
    return screenshot

classify_image(image_path='raw_data/10/2.jpg', model_path='checkpoints/vgg1609-0.414.mod')
classify_image(image_path='raw_data/30/2.jpg', model_path='checkpoints/vgg1609-0.414.mod')
classify_image(image_path='raw_data/70/2.jpg', model_path='checkpoints/vgg1609-0.414.mod')
classify_image(image_path='raw_data/110/2.jpg', model_path='checkpoints/vgg1609-0.414.mod')
