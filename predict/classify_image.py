from keras.models import load_model
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input as preprocess_xception
from keras.preprocessing.image import load_img, img_to_array
from create_training_datsa import process_screen

def classify_image(image_path='img.jpg', model_path='model.mod'):
    # load model
    model = load_model(model_path)

    # determine preprocess method
    preprocess_path = os.path.join('training_data', 'preprocess.txt')
    with open(preprocess_path, 'r') as handler:
        preprocess_method = handler.read()
    if preprocess_method == 'xception':
        preprocess = preprocess_xception
    elif preprocess_method == 'vgg':
        preprocess = imagenet_utils.preprocess_input
    elif preprocess_method == 'none':
        preprocess = lambda x:x

    # preprocess image
    dimension = (input_shape[0], input_shape[1])
    screenshot = process_screen(image_path, dimension, preprocess)

    # predict classes
    prediction = model.predict([screenshot])[0]
    classes = [i for i in range(0, len(prediction)) if prediction[i] >= 0]

    # read genre file and output genres
    genre_file_path = os.path.join('training_data', '')
    with open(genre_file_path, 'r') as handler:
        genres = handler.readlines()
    print('Predicted genres:\n')
    for c in classes:
        print(genres[c])
