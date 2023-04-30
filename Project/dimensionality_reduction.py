from math_utils import *
import numpy as np
import scipy as sci

def PCA(D,m): #D=matrice con dati organizzati per colonne, m=dimensionalita' finale
     mu=D.mean(1) 
     
     mu=FromRowToColumn(mu)
    
     DC=D-mu

     C=np.dot(DC,DC.T)
     C=C/float(DC.shape[1]) 

     s,U=np.linalg.eigh(C)
     P=U[:,::-1][:,0:m]
    
     DP=np.dot(P.T, D)
     return DP

def LDA(Dataset,Labels,m):
    SB=0
    SW=0
    mu=FromRowToColumn(calcmean(Dataset))
    for i in range(2):
        DC1s=Dataset[:,Labels==i]
        muC1s=FromRowToColumn(DC1s.mean(1))
        SW+=np.dot(DC1s-muC1s,(DC1s-muC1s).T)
        SB+=DC1s.shape[1]*np.dot(muC1s-mu,(muC1s-mu).T)
    SW/=Dataset.shape[1]
    SB/=Dataset.shape[1]
    # print(SW)
    # print(SB)

    # risolvo problema generale agli autovalori
    s,U=sci.eigh(SB,SW)
    W=U[:,::-1][:,0:m]
    DP=np.dot(W.T,Dataset)
    return DP