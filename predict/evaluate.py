from keras.models import load_model
import numpy as np
import os

import measures

def evaluate(model_path=os.path.join('model', 'mod'),
             extra_data=True,
             cutoff_file='cutoffs.npy'):

    # load test data
    test_X = np.load(os.path.join('training_data','test_X.npy'))
    test_Y = np.load(os.path.join('training_data', 'test_Y.npy'))
    if extra_data:
        extra_X = np.load(os.path.join('training_data', 'extra_X.npy'))
        extra_Y = np.load(os.path.join('training_data', 'extra_Y.npy'))

    # load model
    model = model = load_model(model_path)
    
    # load cutoffs
    cutoffs = np.load(os.path.join('cutoffs', cutoff_file))

    # get predictions
    test_predictions = model.predict(test_X, batch_size=16)
    if extra_data:
        extra_predictions = model.predict(extra_X, batch_size=16)

    # get measures
    test_measures = measures.get_measures(test_predictions, test_Y, cutoffs)
    if extra_data:
        extra_measures = measures.get_measures(extra_predictions, extra_Y, cutoffs)
       
    # read genres
    genre_file_path = os.path.join('training_data', 'genres.txt')
    with open(genre_file_path, 'r') as handler:
        genres = handler.readlines()
    genres = [genre[:-1] for genre in genres]
    
    # print measures 
    print("Statistics on test data:")
    measures.print_measures(test_measures, genres)
    if extra_data:
        print("Statistics on extra data:")
        measures.print_measures(extra_measures, genres)
    

evaluate(model_path='checkpoints/attempt5(best)/xception_trained',
         extra_data=False)
