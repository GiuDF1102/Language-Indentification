import numpy as np
import scipy as sci
import json

def FromRowToColumn(v):
    return v.reshape((v.size, 1))

def FromColumnToRow(v):
    return v.reshape((1, v.size))

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    k_1 = (M*0.5)*np.log(2*np.pi)
    _,log_C = np.linalg.slogdet(C)
    k_2 = 0.5*log_C
    C_inv = np.linalg.inv(C)
    x_m = x - mu
    k_3 = 0.5*(x_m*np.dot(C_inv,x_m))
    
    return -k_1-k_2-k_3.sum(0)

def logpdf_GMM(X,gmm,nGaussian):
    S = [] #(M,N)
    for i in range(nGaussian):
        wPrior = gmm[i][0]
        mu = gmm[i][1]
        covMatrix = gmm[i][2]
        S.append(logpdf_GAU_ND(X,mu,covMatrix) + np.log(wPrior))
    S = np.array(S)
    logdens = sci.special.logsumexp(S, axis=0)
    return logdens

def EStep(gmmStart,nGaussian,X):
    logSJoint = logpdf_GMM(X,gmm,nGaussian)
    logSMarginal = FromColumnToRow(sci.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    return logSPost

def MStep(resposibilties,X,nGaussian):
    #remember that posteriors are made of a posteriors for each class
    zList = []

    for g in range(nGaussian):
        Zg = resposibilties[g].sum()
        Fg = np.dot(resposibilties[g],np.dot(X,X.T))
    # for i in range(nGaussian):
    #     #Statistics
    #     F=np.dot(resposibilties,X).sum()
    #     Z=resposibilties.sum() #
    #     S=np.dot(resposibilties,np.dot(X,X.T)).sum()

    #     #New 
    #     muNew=F/Z
    #     covNew=(S/Z)-np.dot(muNew,muNew.T)
    #     zList.append(Z)

    #     wNew = 

if __name__ == '__main__':
    xMain = np.load("./GMM-data/GMM_data_4D.npy")
    gmmMain = load_gmm("./GMM-data/GMM_4D_3G_init.json")
    solMain = np.load("./GMM-data/GMM_4D_3G_init_ll.npy")
    print(logpdf_GMM(xMain, gmmMain,3) == solMain)