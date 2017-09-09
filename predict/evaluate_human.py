from keras.models import load_model
import numpy as np
import os

import measures

def evaluate_human(evals_path='human_evals.npy'):

    # load predictions
    human_predictions = np.load(evals_path)
    n = np.shape(human_predictions)[0]
    test_Y = np.load(os.path.join('training_data', 'test_Y.npy'))[0:n]

    # read genres
    genre_file_path = os.path.join('training_data', 'genres.txt')
    with open(genre_file_path, 'r') as handler:
        genres = handler.readlines()
    genres = [genre[:-1] for genre in genres]

    # get measures
    cutoffs = [0.5 for genre in genres]
    human_measures = measures.get_measures(human_predictions, test_Y, cutoffs)
    
    # print measures 
    print("Statistics on test data:")
    measures.print_measures(human_measures, genres)
    

evaluate_human()
