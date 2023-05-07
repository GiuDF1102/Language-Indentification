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
        prova = du.split_k(X, Y, k)
        # print(prova)
        concat_predicted = []
        errors = []             
        for i in range(k): #for each fold
            X_train = X[prova[0]]
            y_train = Y[prova[1]]
            X_val = X[i]
            y_val = Y[i]
            m,C = learner.fit(X_train.T, y_train)
            predicted = learner.trasform(X_val.T,m,C)
            concat_predicted.append((predicted.tolist().pop()))
        error_rates.append((1-calc_accuracy(y.T[0], np.array(concat_predicted)))*100)
    return error_rates