import numpy as np
import sklearn.datasets as sk
import scipy.optimize as opt
from itertools import repeat


class SVM_linear:
    def _init_(self,DTR,LTR,K,C,c,d,eps):
        self.DTR=DTR
        self.LTR=LTR
        self.K=K
        self.C=C 
        self.c=c 
        self.eps=eps

    def expandMatrix(self,K,Data):
    row_to_add=np.ones(Data.shape[1])*K
        return np.vstack([Data,row_to_add])

    def calcG(self,expandedData):
        return np.dot(expandedData.T,expandedData)

    def modifyLabel(self,trainLabels):#from 0,1 to 1,-1
        return np.where(trainLabels==0,-1,1)

    def calcH(self,data,labels,K):
    Z=modifyLabel(labels)
    D=expandMatrix(K,data)
    G=calcG(D)
    H = np.zeros(G.shape)
    for i in range(D.shape[1]):
        for j in range(D.shape[1]):
            H[i][j] = Z[i]*Z[j]*G[i][j]

    return H

    def J(self,alpha,H):
        grad=np.dot(H,alpha)-np.ones(H.shape[1])
        return (0.5*np.dot(alpha.T,np.dot(H,alpha))-np.dot(alpha,np.ones(H.shape[1])),grad)

    def primalSolution(self,w, D, C, LTR,K, f):
        normTerm = (1/2)*(np.linalg.norm(w)**2)
        m = np.zeros(modifyLabel(LTR.size))
        for i in range(modifyLabel(LTR).size):
            vett = [0, 1-modifyLabel(LTR)[i]*(np.dot(w, expandMatrix(K, D)[:, i]))]
            m[i] = vett[np.argmax(vett)]
        pl = normTerm + C*np.sum(m)
        dg = pl-f
        return  dg

    def primalObjective(self,w, D, C, LTR, f):
        normTerm = (1/2)*(np.linalg.norm(w)**2)
        m = np.zeros(LTR.size)
        for i in range(LTR.size):
            vett = [0, 1-modifyLabel(LTR)[i]*(np.dot(w.T, D[:, i]))]
            m[i] = vett[np.argmax(vett)]
        pl = normTerm + C*np.sum(m)
        dl = -f
        dg = pl-dl
        return pl, dl, dg