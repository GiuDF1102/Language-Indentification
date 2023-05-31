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

""" def primalSolution(alpha,C,DTR,LTR,K):
    return C*np.sum(np.maximum(0,1-modifyLabel(LTR)*np.dot(w.T,expandMatrix(K, DTR)))) """

def primalSolution(w, D, C, LTR,K, f):
    normTerm = (1/2)*(np.linalg.norm(w)**2)
    m = np.zeros(modifyLabel(LTR.size))
    for i in range(modifyLabel(LTR).size):
        vett = [0, 1-modifyLabel(LTR)[i]*(np.dot(w, expandMatrix(K, D)[:, i]))]
        m[i] = vett[np.argmax(vett)]
    pl = normTerm + C*np.sum(m)
    dg = pl-f
    return  dg

def primalObjective(w, D, C, LTR, f):
    normTerm = (1/2)*(np.linalg.norm(w)**2)
    m = np.zeros(LTR.size)
    for i in range(LTR.size):
        vett = [0, 1-modifyLabel(LTR)[i]*(np.dot(w.T, D[:, i]))]
        m[i] = vett[np.argmax(vett)]
    pl = normTerm + C*np.sum(m)
    dl = -f
    dg = pl-dl
    return pl, dl, dg

if __name__ == "__main__":
    data,labels=load_iris_without_setosa()
    (DTR, LTR), (DTE, LTE)=split_db_2to1(data,labels)
    alpha=np.zeros(DTR.shape[1])#stessa dim del numero si sample

    H = calcH(DTR,LTR,1)
    C=0.1
    bounds = list(repeat((0, C), DTR.shape[1]))
    (alpha, f, data)=opt.fmin_l_bfgs_b(J, alpha, args=(H,),bounds=bounds, factr=1.0)
    w = np.sum((alpha*modifyLabel(LTR)).reshape(1, DTR.shape[1])*expandMatrix(1, DTR), axis=1)
    
    scores = np.dot(w.T, expandMatrix(1, DTE))
    
    LP = 1*(scores > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = np.array(LP == modifyLabel(LTE)).sum()
    accuracy = numberOfCorrectPredictions/modifyLabel(LTE).size*100
    errorRate = 100-accuracy
    print(errorRate)

    D = expandMatrix(1, DTR)
    print(primalObjective(w, D, C, LTR, f))