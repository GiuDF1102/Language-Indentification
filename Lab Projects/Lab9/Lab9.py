import numpy as np
import sklearn.datasets as sk
import scipy.optimize as opt
from itertools import repeat

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
    Zi=modifyLabel(labels)
    Zj=modifyLabel(labels)
    ZiZj=np.dot(Zi.T,Zj)
    D=expandMatrix(K,data)
    G=calcG(D)
    """ H = np.zeros(G.shape) """
    """ for i in range(D.shape[1]):
        for j in range(D.shape[1]):
            H[i][j] = Z[i]*Z[j]*G[i][j] """
    

    return ZiZj*G

def J(alpha,data,labels,K):
    H=calcH(data,labels,K)
    grad=np.dot(H,alpha)-np.ones(H.shape[1])
    return (0.5*np.dot(alpha.T,np.dot(H,alpha))-np.dot(alpha,np.ones(H.shape[1])),grad)

if __name__ == "__main__":
    data,labels=load_iris_without_setosa()
    alpha=np.zeros(data.shape[1])#stessa dim del numero si sample


    C=0.1
    bounds = list(repeat((0, C), data.shape[1]))
    (x, f, data)=opt.fmin_l_bfgs_b(J,alpha,args=(data,labels,1),bounds=bounds)
    print(x)