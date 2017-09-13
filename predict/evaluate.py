from keras.models import load_model
import numpy as np
import os

import measures

# evaluate a given model on test data
# model_path: path to keras model
# cutoff_file: path to threshold file
def evaluate(model_path=os.path.join('model', 'mod'),
             cutoff_file='cutoffs.npy'):

    # load test data
    test_X = np.load(os.path.join('training_data','test_X.npy'))
    test_Y = np.load(os.path.join('training_data', 'test_Y.npy'))

    # load model
    model = model = load_model(model_path)
    
    # load cutoffs
    cutoffs = np.load(os.path.join('cutoffs', cutoff_file))

    # get predictions
    test_predictions = model.predict(test_X, batch_size=16)

    # get measures
    test_measures = measures.get_measures(test_predictions, test_Y, cutoffs)
       
    # read genres
    genre_file_path = os.path.join('training_data', 'genres.txt')
    with open(genre_file_path, 'r') as handler:
        genres = handler.readlines()
    genres = [genre[:-1] for genre in genres]
    
    # print measures 
    print("Statistics on test data:")
    measures.print_measures(test_measures, genres)    

evaluate()
