import numpy as np

import measures

def get_cutoffs(model_path=os.path.join('model', 'mod')):
    
    # load training data
    train_X = np.load(os.path.join('training_data','train_X.npy'))
    train_Y = np.load(os.path.join('training_data', 'train_Y.npy'))

    # load model
    model = model = load_model(model_path)
    
    # get predictions
    predictions = model.predict(train_X)
    
    # calculate optimal cutoff point for each label separately
    k = np.shape(y_true)[1]
    cutoffs = np.zeros(k)
    for label_id in range(k):
        # get candidates
        possible_cutoffs = y_pred[:,label_id]
        best_cutoff = -1
        best_f1 = -1
        # try each candidate, comparing f1 scores
        for candidate in possible_cutoffs:
            f1_score = measures.get_f1_score_for_label(label_id, y_pred, y_true, cutoff=candidate)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_cutoff = candidate
        cutoffs[label_id] = best_cutoff
        
    # write cutoffs
    np.save(os.path.join('cutoffs', 'cutoffs.npy'), cutoffs)
    
get_cutoffs()