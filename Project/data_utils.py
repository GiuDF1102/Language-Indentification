import numpy as np
import math_utils as mu

def load(file_name):
    file_path = file_name

    label_list = []
    features_list = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                splitted_line = line.removesuffix("\n").split(",")
                label_list.append(splitted_line.pop().lower())
                features_list.append([float(elem) for elem in splitted_line])
            except:
                pass                            

    return (np.array(label_list, dtype=int), np.array(features_list, dtype=float).T)

def split_db(D, L,pTrain,pTest,seed=0):#versicolor=1, virginica=0
    # 2/3 dei dati per il training----->100 per training, 50 per test
    nTrain = int(D.shape[1]*pTrain/pTest)# split in percentuale
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)  # DTR= data training, LTR= Label training
    # DTE= Data test, LTE= label testing

def split_k(X, y, k):
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Divide the data into k folds
    folds = []
    fold_size = len(X) // k
    for i in range(k):
        start, end = i * fold_size, (i+1) * fold_size
        val_indices = list(range(start, end))
        train_indices = list(range(0, start)) + list(range(end, len(X)))
        print(train_indices, val_indices)
        folds.append((train_indices, val_indices))
    return folds

def features_expansion(Dataset):
    expansion = []
    for i in range(Dataset.shape[1]):
        vec = np.reshape(np.dot(mu.FromRowToColumn(Dataset[:, i]), mu.FromRowToColumn(Dataset[:, i]).T), (-1, 1), order='F')
        expansion.append(vec)
    return np.vstack((np.hstack(expansion), Dataset))

def modifyLabel(trainLabels):
    return np.where(trainLabels==0,-1,1)
