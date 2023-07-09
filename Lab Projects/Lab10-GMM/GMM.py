import numpy as np
import scipy as sci
import json
import sklearn.datasets as sk

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


def GMM_ll_per_sample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    
    for g in range(G):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return sci.special.logsumexp(S, axis=0)

def EStep(gmmStart,X):
    nGaussian = len(gmmStart)
    N = X.shape[1]
    SJ = np.zeros((nGaussian, N))
    for g in range(nGaussian):
        SJ[g, :] = logpdf_GAU_ND(X, gmmStart[g][1], gmmStart[g][2]) + np.log(gmmStart[g][0])
    SM = sci.special.logsumexp(SJ, axis=0)
    llNew = SM.sum() / N
    P = np.exp(SJ - SM)

    return P, llNew

def MStep(P, X, nGaussian):
    gmmNew = []
    N = X.shape[1]
    psi = 0.01
    for g in range(nGaussian):
        gamma = P[g, :]
        Z = gamma.sum()
        F = (FromColumnToRow(gamma)*X).sum(1)
        S = np.dot(X, (FromColumnToRow(gamma)*X).T)
        w = Z/N
        mu = FromRowToColumn(F/Z)
        sigma = S/Z - np.dot(mu, mu.T)

        U, s, _ = np.linalg.svd(sigma)
        s[s<psi] = psi
        sigma = np.dot(U, FromRowToColumn(s)*U.T)

        gmmNew.append((w, mu, sigma))
    return gmmNew

def MStepDiagonal(P, X, nGaussian):
    gmmNew = []
    N = X.shape[1]
    psi = 0.01
    for g in range(nGaussian):
        gamma = P[g, :]
        Z = gamma.sum()
        F = (FromColumnToRow(gamma)*X).sum(1)
        S = np.dot(X, (FromColumnToRow(gamma)*X).T)
        w = Z/N
        mu = FromRowToColumn(F/Z)
        sigma = S/Z - np.dot(mu, mu.T)

        sigma=sigma*np.eye(sigma.shape[0])
        U, s, _ = np.linalg.svd(sigma)
        s[s<psi] = psi
        sigma = np.dot(U, FromRowToColumn(s)*U.T)

        gmmNew.append((w, mu, sigma))
    return gmmNew

def MStepTied(P, X, nGaussian):
    gmmNew = []
    N = X.shape[1]
    psi = 0.01
    covMatrixList=[]
    Zlist=[]
    sigmaNew = np.zeros((X.shape[0], X.shape[0]))

    for g in range(nGaussian):
        gamma = P[g, :]
        Z = gamma.sum()
        Zlist.append(Z)
        F = (FromColumnToRow(gamma)*X).sum(1)
        S = np.dot(X, (FromColumnToRow(gamma)*X).T)
        w = Z/N
        mu = FromRowToColumn(F/Z)
        sigma = S/Z - np.dot(mu, mu.T)
        gmmNew.append((w, mu, sigma))

    for i,gmm in enumerate(gmmNew):
        sigmaNew += np.dot(Zlist[i],gmm[2])

    sigmaNew=sigmaNew/N
    
    U, s, _ = np.linalg.svd(sigmaNew)
    s[s<psi] = psi
    sigmaNew = np.dot(U, FromRowToColumn(s)*U.T)
    
    for g in range(nGaussian):
        gmmNew[g] = (gmmNew[g][0], gmmNew[g][1], sigmaNew)
    
    return gmmNew

def MStepTiedDiagonal(P, X, nGaussian):
    print("MStep tied Diag")
    gmmNew = []
    N = X.shape[1]
    psi = 0.01
    covMatrixList=[]
    Zlist=[]
    sigmaNew = np.zeros((X.shape[0], X.shape[0]))

    for g in range(nGaussian):
        gamma = P[g, :]
        Z = gamma.sum()
        Zlist.append(Z)
        F = (FromColumnToRow(gamma)*X).sum(1)
        S = np.dot(X, (FromColumnToRow(gamma)*X).T)
        w = Z/N
        mu = FromRowToColumn(F/Z)
        sigma = S/Z - np.dot(mu, mu.T)
        gmmNew.append((w, mu, sigma))

    for i,gmm in enumerate(gmmNew):
        sigmaNew += np.dot(Zlist[i],gmm[2])

    sigmaNew=sigmaNew/N
    sigmaNew=sigmaNew*np.eye(sigmaNew.shape[0])
    
    U, s, _ = np.linalg.svd(sigmaNew)
    s[s<psi] = psi
    sigmaNew = np.dot(U, FromRowToColumn(s)*U.T)
    
    for g in range(nGaussian):
        gmmNew[g] = (gmmNew[g][0], gmmNew[g][1], sigmaNew)
    
    return gmmNew

def GMM_EM(X, nGaussian, gmmStart,Mtype):
    llNew = None
    llOld = None
    nIter = 0
    N = X.shape[1]
    
    while llOld is None or llNew-llOld>1e-6:
        llOld = llNew
        nIter += 1
        respons, llNew = EStep(gmmStart, X)
        gmmNew = None
        if(Mtype=="tied"):
            gmmNew = MStepTied(respons,X,nGaussian)
        elif(Mtype=="diagonal"):
            gmmNew = MStepDiagonal(respons,X,nGaussian)
        elif(Mtype=="tied diagonal"):
            gmmNew = MStepTiedDiagonal(respons,X,nGaussian)
        else:
            gmmNew = MStep(respons, X, nGaussian)
        gmmStart = gmmNew
        if llOld is not None:
            if llNew < llOld:
                print("Error: Log likelihood decreased")
                print("llOld: {} llNew: {} iter: {}".format(llOld, llNew, nIter))
                return 
    return gmmStart

def LBG(X, iterations,Mtype, alpha = 0.1):
    mu = FromRowToColumn(X.mean(1))
    C=np.cov(X)
    psi = 0.01
    GMM_i = [(1.0, mu, C)]
    if(C.shape==()):
        C = C.reshape((C.size, 1))

    if(Mtype == "diagonal" or Mtype=="tied diagonal"):
        C = C*np.eye(C.shape[0])
    U, s, _ = np.linalg.svd(C)
    s[s<psi] = psi
    sigma = np.dot(U, FromRowToColumn(s)*U.T)
    GMM_i = [(1.0, mu, sigma)]
    for i in range(iterations):
        GMM_list=[]
        for g in GMM_i:
            Wg = g[0]
            Ug = g[1]
            Sg = g[2]
            if(Sg.shape==()):
                Sg = Sg.reshape((Sg.size, 1))
            U, s, Vh = np.linalg.svd(Sg)
            Dg = U[:, 0:1] * s[0]**0.5 * alpha
            c1 = (Wg/2,Ug+Dg,Sg)
            c2 = (Wg/2,Ug-Dg,Sg)
            GMM_list.append(c1)
            GMM_list.append(c2)
            if(Mtype=="tied"):
                GMM_i = GMM_EM(X,len(GMM_list),GMM_list,"tied")
            elif(Mtype == "diagonal"):
                GMM_i = GMM_EM(X,len(GMM_list),GMM_list,"diagonal")
            elif(Mtype=="tied diagonal"):    
                GMM_i = GMM_EM(X,len(GMM_list),GMM_list,"tied diagonal")
            else:
                GMM_i = GMM_EM(X,len(GMM_list),GMM_list,"mvg")
    return GMM_i

def GMM_ll_per_sample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = np.zeros((G, N))
    
    for g in range(G):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    return sci.special.logsumexp(S, axis=0)

def train(xTrain, xLabels, numComponents,Mtype):
    numClasses = len(np.unique(xLabels))
    GMM_list = []
    for i in range(numClasses):
        xClassI = xTrain[:, xLabels == i]
        k = int(np.ceil(np.log2(numComponents)))
        GMMI = LBG(xClassI, k, Mtype)
        GMM_list.append(GMMI)
    return GMM_list

def trasform(xTest, xLabels, GMM_list, numComponents):
    numClasses = len(np.unique(xLabels))
    ll_list = []
    for i in range(numClasses):
        ll_list.append(GMM_ll_per_sample(xTest, GMM_list[i]))
    ll_list = np.array(ll_list)
    return np.argmax(ll_list, axis=0)

def load_iris():
    D, L = sk.load_iris()['data'].T, sk.load_iris()['target']  
    return D,L

def split_db_2to1(D, L, seed=0):
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

if __name__ == '__main__':
    xMain = np.load("./GMM-data/GMM_data_4D.npy")
    gmmMain = load_gmm("./GMM-data/GMM_4D_3G_init.json")
    gmmFinal= load_gmm("./GMM-data/GMM_4D_3G_EM.json")
    solMain = np.load("./GMM-data/GMM_4D_3G_init_ll.npy")
    solLBG= load_gmm("./GMM-data/GMM_1D_4G_EM_LBG.json")
    
    #IRIS
    xTest, xLabels = load_iris()
    (dataTrain, labelsTrain), (dataTest, labelsTest) = split_db_2to1(xTest, xLabels)

    GMM_list_main = train(dataTrain, labelsTrain, 4,"tied diagonal")#numero di gaussiani non da elevare alla seconda
    predicted = trasform(dataTest, labelsTest, GMM_list_main, 4)#numero di gaussiani non da elevare alla seconda
    print("Error: {}".format((1-np.sum(predicted == labelsTest)/len(labelsTest))*100))
    
    
