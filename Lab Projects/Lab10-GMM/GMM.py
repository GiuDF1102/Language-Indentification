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

def logpdf_GAU_ND_Opt(X, mu, C):
    P = np.linalg.inv(C)
    const = -0.5 * X.shape[0] * np.log(2*np.pi)
    const += -0.5 * np.linalg.slogdet(C)[1]
    
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * np.dot((x-mu).T, np.dot(P, (x-mu)))
        Y.append(res)
    
    return np.array(Y).ravel()

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
        SJ[g,:] = logpdf_GAU_ND_Opt(X, gmmStart[g][1], gmmStart[g][2]) + np.log(gmmStart[g][0])
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

def GMM_EM_nostro(X, nGaussian, gmm):
    llNew = None
    llOld = None
    nIter = 0
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        nIter += 1
        respons, llNew = EStep(gmm, X)
        gmmNew = MStep(respons, X, nGaussian)
        print("Iteration: ", nIter)
        gmm = gmmNew
    return gmm


def GMM_EM(X, gmm):
    ll_new = None
    ll_old = None
    G = len(gmm)
    N = X.shape[1]
    
    while ll_old is None or ll_new-ll_old>1e-6:
        ll_old = ll_new
        SJ = np.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        SM = sci.special.logsumexp(SJ, axis=0)
        ll_new = SM.sum() / N
        P = np.exp(SJ - SM)
        
        gmm_new = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (FromColumnToRow(gamma)*X).sum(1)
            S = np.dot(X, (FromColumnToRow(gamma)*X).T)
            w = Z/N
            mu = FromRowToColumn(F/Z)
            sigma = S/Z - np.dot(mu, mu.T)
            
            gmm_new.append((w, mu, sigma))
        gmm = gmm_new
        #print(ll_new)
    #print(ll_new-ll_old)
    return gmm
    
if __name__ == '__main__':
    xMain = np.load("./GMM-data/GMM_data_4D.npy")
    gmmMain = load_gmm("./GMM-data/GMM_4D_3G_init.json")
    gmmFinal= load_gmm("./GMM-data/GMM_4D_3G_EM.json")
    solMain = np.load("./GMM-data/GMM_4D_3G_init_ll.npy")
    
    print(GMM_EM_nostro(xMain, 3, gmmMain))
    print(GMM_EM(xMain, gmmMain))
    print(gmmFinal)