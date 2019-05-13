import numpy as np
import os
from keras.models import load_model

import measures

# get optimal thresholds for each label on the training data, maximizing F1 score
# model_path: path to keras model
def get_cutoffs(model_path=os.path.join('checkpoints', 'xception_trained')):
    
    # load training data
    train_X = np.load(os.path.join('training_data','train_X.npy'))
    y_true = np.load(os.path.join('training_data', 'train_Y.npy'))

    # load model
    model = load_model(model_path)
    
    # get predictions
    y_pred = model.predict(train_X, batch_size=16)
    
    # calculate optimal cutoff point for each label separately
    k = np.shape(y_true)[1]
    cutoffs = np.zeros(k)
    for label_id in range(k):
        print("label " + str(label_id))
        # get candidates
        possible_cutoffs = y_pred[:,label_id]
        best_cutoff = -1
        best_f1 = -1
        # try each candidate, comparing f1 scores
        possible_cutoffs = np.sort(possible_cutoffs)
        for i in range(0, len(possible_cutoffs)):
            candidate = possible_cutoffs[i]
            f1_score = measures.get_f1_score_for_label(label_id, y_pred, y_true, cutoff=candidate)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_cutoff = candidate
        cutoffs[label_id] = best_cutoff
        print(best_cutoff)
        
    # write cutoffs
    np.save(os.path.join('cutoffs', 'cutoffs.npy'), cutoffs)
    
get_cutoffs()
