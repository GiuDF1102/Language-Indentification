import numpy as np

def dataset_cov_mat(D,mu):
    DC = D - mu.reshape((mu.size, 1))
    return (1/D.shape[1])*np.dot(DC,DC.T)

def FromRowToColumn(v):
    return v.reshape((v.size, 1))

def calcmean(D):
    return D.mean(1) #ritorno media sulle colonne

def calcmean_classes(Data, labels):
    classes_list = []
    classes_means = []
    num_classes = len(np.unique(labels))
    num_feats = len(Data)

    for i in range(num_classes):
        classes_list.append(np.array(Data[:, (labels == i)]))
    
    for i in range(num_classes):
        means = []
        for j in range(num_feats):
            means.append(classes_list[i][j].mean())
        classes_means.append(means)

    return classes_means

def calcmean_variance(Data, labels):
    classes_list = []
    classes_variance = []
    num_classes = len(np.unique(labels))
    num_feats = len(Data)

    for i in range(num_classes):
        classes_list.append(np.array(Data[:, (labels == i)]))
    
    for i in range(num_classes):
        variance_class = []
        for j in range(num_feats):
            variance_class.append(classes_list[i][j].var())
        classes_variance.append(variance_class)

    return classes_variance


    