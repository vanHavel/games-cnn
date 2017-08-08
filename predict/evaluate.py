from keras.models import load_model
import numpy as np
import os

def evaluate(model_path=os.path.join('model', 'mod'),
             test_data_type="test"):

    #load test data
    test_X = np.load(os.path.join('training_data","test_X.npy')).astype('float32')
    if test_data_type == "test":
        test_Y = np.load(os.path.join('training_data", "test_Y.npy'))
    elif test_data_type == "extra":
        test_Y = np.load(os.path.join('training_data", "extra_Y.npy'))

    #load model
    model = model = load_model(model_path)

    # get predictions
    predictions = model.predict(test_X)
    vfunc = np.vectorize(lambda x: 1 if x > 0  else -1)
    predictions = vfunc(predictions)

    # print some statistics
    print("Label cardinality: " + label_cardinality(test_Y))
    print("Label density: " + label_density(test_Y))

    # evaluate predictions using several measures
    print("0-1 error: " + zero_one_error(test_Y, predictions))
    print("Hamming Loss: " + hamming_loss(test_Y, predictions))
    print("Jaccard Index: " + jaccard_index(test_Y, predictions))
    print("Precision: " + precision(test_Y, predictions))
    print("Recall: " + recall(test_Y, predictions))
    print("F1 Score: " + f1(test_Y, predictions))

def label_cardinality(y_true):
    vfunc = np.vectorize(lambda x: 1 if x == 1 else 0)
    n = np.shape(y_true)[0]
    return np.sum(vfunc(y_true)) / n

def label_density(y_true):
    k = np.shape(y_true)[1]
    return label_cardinality(y_true) / k

def zero_one_error(y_true, y_pred):
    errors = 0
    n = np.shape(y_true)[0]
    for i in range(n):
        if not y_true[i] == y_pred[i]:
            errors += 1
    return errors / n

def hamming_loss(y_true, y_pred):
    errors = 0
    n = np.shape(y_true)[0]
    k = np.shape(y_true)[1]
    for i in range(n):
        for j in range(k):
            if not y_true[i][j] == y_pred[i][j]:
                errors += 1
    return errors / (n * k)

def jaccard_index(y_true, y_pred):
    res = 0
    n = np.shape(y_true)[0]
    k = np.shape(y_true)[1]
    for i in range(n):
        true_and_predicted = len([j for j in range(k) if y_true[i][j] == 1 and y_pred[i][j] == 1])
        true_or_predicted = len([j for j in range(k) if y_true[i][j] == 1 or y_pred[i][j] == 1])
        res += (true_and_predicted / true_or_predicted)
    return res / n

def precision(y_true, y_pred):
    res = 0
    n = np.shape(y_true)[0]
    k = np.shape(y_true)[1]
    for i in range(n):
        true_and_predicted = len([j for j in range(k) if y_true[i][j] == 1 and y_pred[i][j] == 1])
        predicted = len([j for j in range(k) if y_pred[i][j] == 1])
        res += (true_and_predicted / predicted)
    return res / n

def recall(y_true, y_pred):
    res = 0
    n = np.shape(y_true)[0]
    k = np.shape(y_true)[1]
    for i in range(n):
        true_and_predicted = len([j for j in range(k) if y_true[i][j] == 1 and y_pred[i][j] == 1])
        true = len([j for j in range(k) if y_true[i][j] == 1])
        res += (true_and_predicted / true)
    return res / n

def f1(y_true, y_pred):
    res = 0
    n = np.shape(y_true)[0]
    k = np.shape(y_true)[1]
    for i in range(n):
        true_and_predicted = len([j for j in range(k) if y_true[i][j] == 1 and y_pred[i][j] == 1])
        true = len([j for j in range(k) if y_true[i][j] == 1])
        predicted = len([j for j in range(k) if y_pred[i][j] == 1])
        recall = (true_and_predicted / true)
        precision = (true_and_predicted / predicted)
        res += 2 / (1 / recall + 1 / precision)
    return res / n

evaluate()
