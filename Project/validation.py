import numpy as np
import data_utils as du
from sklearn.utils import shuffle

def calc_accuracy(labels, predicted):
    confronted = (labels == predicted)
    TP = 0
    
    for i in confronted:
        if(i == True):
            TP = TP + 1
    
    return TP/len(predicted)

def k_fold(learners,x,labels,k):
    error_rates = []
    for learner in learners:
        X, Y = shuffle(x.T, labels)
        X = np.array_split(X, k)
        y = np.array_split(Y, k)
        concat_predicted = []
        for i in range(k): #for each fold
            X_train = np.concatenate(np.delete(X, i, axis=0), axis=0).T
            y_train = np.concatenate(np.delete(y, i, axis=0), axis=0)
            X_val = X[i].T
            m,C = learner.fit(X_train, y_train)
            predicted = learner.trasform(X_val,m,C)
            concat_predicted.extend((predicted.tolist()))
        error_rates.append((1-calc_accuracy(Y, np.array(concat_predicted)))*100)
    return error_rates