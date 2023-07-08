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

def EStep(gmmStart,X):
    G = len(gmmStart)
    N = X.shape[1]
    SJ = np.zeros((G, N))
    for g in range(G):
        SJ[g,:] = logpdf_GAU_ND(X, gmmStart[g][1], gmmStart[g][2]) + np.log(gmmStart[g][0])
    SM = sci.special.logsumexp(SJ, axis=0)
    llNew = SM.sum()/N
    P = np.exp(SJ - SM)
    print(llNew)
    return P, llNew

def MStep(P, X, nGaussian):
    gmmNew = []
    N = X.shape[1]
    for g in range(nGaussian):
        gamma = P[g, :]
        Z = gamma.sum()
        F = (FromColumnToRow(gamma)*X).sum(1)
        S = np.dot(X, (FromColumnToRow(gamma)*X).T)
        w = Z/N
        mu = FromColumnToRow(F/Z)
        Sigma = S/Z - np.dot(mu, mu.T)
        gmmNew.append((w, FromRowToColumn(mu), Sigma))
    return gmmNew

def EM_GMM(X, nGaussian, gmm):
    llNew = None
    llOld = None
    nIter = 0
    while llOld is None or np.abs(llNew - llOld) > 1e-6:
        llOld = llNew
        nIter += 1
        respons, llNew = EStep(gmm, X)
        gmmNew = MStep(respons, X, nGaussian)
        print("Iteration: ", nIter)
        gmm = gmmNew
    return gmm

if __name__ == '__main__':
    xMain = np.load("./GMM-data/GMM_data_4D.npy")
    gmmMain = load_gmm("./GMM-data/GMM_4D_3G_init.json")
    solMain = np.load("./GMM-data/GMM_4D_3G_init_ll.npy")
    
    print(EM_GMM(xMain, 3, gmmMain))
