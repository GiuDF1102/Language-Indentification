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

def EStep(gmmStart,X,nGaussian):
    S = [] #(M,N)
    for i in range(nGaussian):
        wPrior = gmmStart[i][0]
        mu = gmmStart[i][1]
        covMatrix = gmmStart[i][2]
        S.append(logpdf_GAU_ND(X,mu,covMatrix) + np.log(wPrior))
    S = np.array(S)    #joint    
    logdens = sci.special.logsumexp(S, axis=0) #marginal

    responsibilities = S - logdens
    return np.exp(responsibilities)

"""     logSJoint = np.array(listLogJoint)
    logSMarginal = mut.vrow(sci.special.logsumexp(selflogSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    return logSPost """

def MStep(resposibilties,X,nGaussian):
    #remember that posteriors are made of a posteriors for each class
    Zg_list = []
    Fg_list = []
    Sg_list = []
    Zg_list = []
    Fg_list = []
    Sg_list = []
    gmmParamaters = []
    for g in range(nGaussian):
        Zg_list.append(resposibilties[g].sum())
        Fg_list.append(np.dot(resposibilties[g],X.T))
        Sg_list.append(np.dot(resposibilties[g],np.dot(X.T,X)).sum())

        muG.append(Fg_list[g]/Zg_list[g])
        covG.append(Sg_list[g]/Zg_list[g]-np.dot(muG,muG.T))
        wG.append((Zg_list[g]/np.array(Zg_list).sum())
    gmmParamaters.append((muG, covG, wG))
    return gmmParamaters

if __name__ == '__main__':
    xMain = np.load("./GMM-data/GMM_data_4D.npy")
    gmmMain = load_gmm("./GMM-data/GMM_4D_3G_init.json")
    solMain = np.load("./GMM-data/GMM_4D_3G_init_ll.npy")

    #logDensMain = logpdf_GMM(xMain,gmmMain,3)
    
    resposibilitiesMain = EStep(gmmMain, xMain, 3)
    print(MStep(resposibilitiesMain, xMain, 3))
