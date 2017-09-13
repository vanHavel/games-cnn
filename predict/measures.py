import numpy as np

# calculate the f1_score for a specific label component
def get_f1_score_for_label(label_id, y_pred, y_true, cutoff=0.5):
    # project to component
    y_pred = y_pred[:,label_id]
    y_true = y_true[:,label_id]
    
    # transform predictions to binary matrix
    fun = lambda x: 1 if x >= cutoff else 0
    vfunc = np.vectorize(fun)
    y_pred = vfunc(y_pred)
    
    # calculate f1 score
    true_and_predicted = np.sum(y_true * y_pred)
    true = np.sum(y_true)
    predicted = np.sum(y_pred)
    if predicted == 0 or true == 0:
        return -1
    recall = true_and_predicted / true
    precision = true_and_predicted / predicted
    f1_score = 2 * (recall * precision) / (recall + precision)
    
    return f1_score
    
# get all kinds of accuracy measures for predicted labels
def get_measures(y_pred, y_true, cutoffs):
    # transform predictions to binary matrix
    y_pred = transform_to_binary(y_pred, cutoffs)
    
    # get input sizes
    n = np.shape(y_pred)[0]
    k = np.shape(y_pred)[1]
    
    # create result data structure
    measures = dict()
    
    # basic measures
    measures['label_cardinality'] = np.sum(y_true) / n
    measures['label_density'] = measures['label_cardinality'] / k
    
    # elementary statistics
    zero_one_error = sum([(1 if (y_true[i] == y_pred[i]).all() else 0) for i in range(np.shape(y_true)[0])]) / n
    true_and_predicted = np.sum(y_true * y_pred, axis=0)
    true = np.sum(y_true, axis=0)
    predicted = np.sum(y_pred, axis=0)
    recall = true_and_predicted / true
    precision = true_and_predicted / predicted
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    measures['zero_one_error'] = zero_one_error
    measures['recall'] = recall
    measures['precision'] = precision
    measures['f1_score'] = f1_score
    
    # global statistics
    global_recall = np.sum(true_and_predicted) / np.sum(true)
    global_precision = np.sum(true_and_predicted) / np.sum(predicted)
    global_f1_score = 2 * (global_precision * global_recall) / (global_precision + global_recall)
    
    measures['global_recall'] = global_recall
    measures['global_precision'] = global_precision
    measures['global_f1_score'] = global_f1_score
    
    # average statistics
    measures['average_recall'] = np.mean(recall)
    measures['average_precision'] = np.mean(precision)
    measures['average_f1_score'] = np.mean(f1_score)
    
    return measures

# transform predictions to binary matrix, given cutoff
def transform_to_binary(y_pred, cutoffs):
    # get input sizes
    n = np.shape(y_pred)[0]
    k = np.shape(y_pred)[1]
    
    # transform to binary according to cutoffs
    for i in range(n):
        for j in range(k):
            y_pred[i][j] = 1 if y_pred[i][j] >= cutoffs[j] else 0
    
    return y_pred

# print measures given genre names   
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

# save precision, recall and F1 score
def save_measures(measures):
    np.save('precision.npy', measures['precision'])
    np.save('recall.npy', measures['recall'])
    np.save('f1.npy', measures['f1_score'])

