import numpy as np
import sklearn.datasets as sk
import scipy.optimize as opt
from itertools import repeat

def split_db_2to1(D, L, seed=0):#versicolor=1, virginica=0
    # 2/3 dei dati per il training----->100 per training, 50 per test
    nTrain = int(D.shape[1]*2.0/3.0)
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

def load_iris_without_setosa():
    D, L = sk.load_iris()['data'].T, sk.load_iris()['target']
    D = D[:, L != 0] #non prendo iris setosa
    L = L[L!=0]     #label virginica=0
    L[L==2] = 0    
    return D,L

def expandMatrix(K,Data):
    row_to_add=np.ones(Data.shape[1])*K
    return np.vstack([Data,row_to_add])

def calcG(expandedData):
    return np.dot(expandedData.T,expandedData)

def modifyLabel(trainLabels):#from 0,1 to 1,-1
    return np.where(trainLabels==0,-1,1)

def calcH(data,labels,K):
    Z=modifyLabel(labels)
    D=expandMatrix(K,data)
    G=calcG(D)
    H = np.zeros(G.shape)
    for i in range(D.shape[1]):
        for j in range(D.shape[1]):
            H[i][j] = Z[i]*Z[j]*G[i][j]

    return H

def J(alpha,H):
    grad=np.dot(H,alpha)-np.ones(H.shape[1])
    return (0.5*np.dot(alpha.T,np.dot(H,alpha))-np.dot(alpha,np.ones(H.shape[1])),grad)

if __name__ == "__main__":
    data,labels=load_iris_without_setosa()
    (DTR, LTR), (DTE, LTE)=split_db_2to1(data,labels)
    alpha=np.zeros(DTR.shape[1])#stessa dim del numero si sample

    H = calcH(DTR,LTR,1)
    C=0.1
    bounds = list(repeat((0, C), DTR.shape[1]))
    (x, f, data)=opt.fmin_l_bfgs_b(J, alpha, args=(H,),bounds=bounds, factr=1.0)
