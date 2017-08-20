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
        extra_Y = np.load(os.path.join('training_data', 'extra_Y.npy'))

    # load model
    model = model = load_model(model_path)
    
    # load cutoffs
    cutoffs = np.load(os.path.join('cutoffs', cutoff_file))

    # get predictions
    test_predictions = model.predict(test_X)
    if extra_data:
        extra_prediction = model.predict(extra_X)

    # get measures
    test_measures = measures.get_measures(test_predictions)
    if extra_data:
        extra_measures = measures.get_measures(extra_predictions)
       
    # read genres
    genre_file_path = os.path.join('training_data', 'genres.txt')
    with open(genre_file_path, 'r') as handler:
        genres = handler.readlines()
    genres = [genre[:-1] for genre in genres]
    
    # print measures 
    print("Statistics on test data:")
    print_measures(test_measures, genres)
    if extra_data:
        print("Statistics on extra data:")
        print_measures(extra_measures, genres)
    
def print_measures(measures, genres):
    print("Label cardinality: " + str(measures['label_cardinality']))
    print("Label density: " + str(measures['label_density']))
    print("")
    print("Zero-one error: " + str(measures['zero_one_error']))
    print("Global precision: " + str(measures['global_precision']))
    print("Global recall: " + str(measures['global_recall']))
    print("Global F1 score: " + str(measures['global_f1_score']))
    print("")
    print("Average precision: " + str(measures['average_precision']))
    print("Average recall: " + str(measures['average_recall']))
    print("Average F1 score: " + str(measures['average_f1_score']))
    print("")
    print("Precision per genre:")
    for i in range(len(genres)):
        print(genres[i] + ": " + str(measures['precision'][i]))
    print("")
    print("Recall per genre:")
    for i in range(len(genres)):
        print(genres[i] + ": " + str(measures['recall'][i]))
    print("")
    print("F1 score per genre:")
    for i in range(len(genres)):
        print(genres[i] + ": " + str(measures['f1_score'][i]))
    

evaluate(model_path='checkpoints/vgg1619-0.379.mod')
