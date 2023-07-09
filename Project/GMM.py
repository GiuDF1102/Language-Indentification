import numpy as np
import scipy as sci
import math_utils as mut  


class GMM:

    ll=None
    GMM_list=None
    predicted=None
    nTarget=None
    nNonTarget=None
    Mtype=None

    def __init__(self,nTarget,nNonTarget,Mtype):
        self.nTarget=nTarget
        self.nNonTarget=nNonTarget
        self.Mtype=Mtype

    def _logpdf_GAU_ND(self,x, mu, C):
        M = x.shape[0]
        k_1 = (M*0.5)*np.log(2*np.pi)
        _,log_C = np.linalg.slogdet(C)
        k_2 = 0.5*log_C
        C_inv = np.linalg.inv(C)
        x_m = x - mu
        k_3 = 0.5*(x_m*np.dot(C_inv,x_m))
        
        return -k_1-k_2-k_3.sum(0)

    def _GMM_ll_per_sample(self,X, gmm):
        G = len(gmm)
        N = X.shape[1]
        S = np.zeros((G, N))
        
        for g in range(G):
            S[g, :] = self._logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        return sci.special.logsumexp(S, axis=0)

    def _EStep(self,gmmStart,X):
        gmmStartlength=len(gmmStart)
        N = X.shape[1]
        SJ = np.zeros((gmmStartlength, N))
        for g in range(gmmStartlength):
            SJ[g, :] = self._logpdf_GAU_ND(X, gmmStart[g][1], gmmStart[g][2]) + np.log(gmmStart[g][0])
        SM = sci.special.logsumexp(SJ, axis=0)
        llNew = SM.sum() / N
        P = np.exp(SJ - SM)

        return P, llNew

    def _MStep(self,P, X,lenGMMStart):
        gmmNew = []
        N = X.shape[1]
        psi = 0.01
        for g in range(lenGMMStart):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mut.FromColumnToRow(gamma)*X).sum(1)
            S = np.dot(X, (mut.FromColumnToRow(gamma)*X).T)
            w = Z/N
            mu = mut.FromRowToColumn(F/Z)
            sigma = S/Z - np.dot(mu, mu.T)

            U, s, _ = np.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = np.dot(U, mut.FromRowToColumn(s)*U.T)

            gmmNew.append((w, mu, sigma))
        return gmmNew

    def _MStepDiagonal(self,P, X,nGaussian):
        gmmNew = []
        N = X.shape[1]
        psi = 0.01
        for g in range(nGaussian):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mut.FromColumnToRow(gamma)*X).sum(1)
            S = np.dot(X, (mut.FromColumnToRow(gamma)*X).T)
            w = Z/N
            mu = mut.FromRowToColumn(F/Z)
            sigma = S/Z - np.dot(mu, mu.T)

            sigma=sigma*np.eye(sigma.shape[0])
            U, s, _ = np.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = np.dot(U, mut.FromRowToColumn(s)*U.T)

            gmmNew.append((w, mu, sigma))
        return gmmNew

    def _MStepTied(self,P, X,nGaussian):
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
            F = (mut.FromColumnToRow(gamma)*X).sum(1)
            S = np.dot(X, (mut.FromColumnToRow(gamma)*X).T)
            w = Z/N
            mu = mut.FromRowToColumn(F/Z)
            sigma = S/Z - np.dot(mu, mu.T)
            gmmNew.append((w, mu, sigma))

        for i,gmm in enumerate(gmmNew):
            sigmaNew += np.dot(Zlist[i],gmm[2])

        sigmaNew=sigmaNew/N
        
        U, s, _ = np.linalg.svd(sigmaNew)
        s[s<psi] = psi
        sigmaNew = np.dot(U, mut.FromRowToColumn(s)*U.T)
        
        for g in range(nGaussian):
            gmmNew[g] = (gmmNew[g][0], gmmNew[g][1], sigmaNew)
        
        return gmmNew

    def _MStepTiedDiagonal(self,P, X,nGaussian):
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
            F = (mut.FromColumnToRow(gamma)*X).sum(1)
            S = np.dot(X, (mut.FromColumnToRow(gamma)*X).T)
            w = Z/N
            mu = mut.FromRowToColumn(F/Z)
            sigma = S/Z - np.dot(mu, mu.T)
            gmmNew.append((w, mu, sigma))

        for i,gmm in enumerate(gmmNew):
            sigmaNew += np.dot(Zlist[i],gmm[2])

        sigmaNew=sigmaNew/N
        sigmaNew=sigmaNew*np.eye(sigmaNew.shape[0])
        
        U, s, _ = np.linalg.svd(sigmaNew)
        s[s<psi] = psi
        sigmaNew = np.dot(U, mut.FromRowToColumn(s)*U.T)
        
        for g in range(nGaussian):
            gmmNew[g] = (gmmNew[g][0], gmmNew[g][1], sigmaNew)
        
        return gmmNew

    def _GMM_EM(self,X, gmmStart):
        llNew = None
        llOld = None
        nIter = 0
        N = X.shape[1]
        
        while llOld is None or llNew-llOld>1e-6:
            llOld = llNew
            nIter += 1
            respons, llNew = self._EStep(gmmStart, X)
            gmmNew = None
            if(self.Mtype=="tied"):
                gmmNew = self._MStepTied(respons,X,len(gmmStart))
            elif(self.Mtype=="diagonal"):
                gmmNew = self._MStepDiagonal(respons,X,len(gmmStart))
            elif(self.Mtype=="tied diagonal"):
                gmmNew = self._MStepTiedDiagonal(respons,X,len(gmmStart))
            else:
                gmmNew =self._MStep(respons, X,len(gmmStart))
            gmmStart = gmmNew
            if llOld is not None:
                if llNew < llOld:
                    print("Error: Log likelihood decreased")
                    print("llOld: {} llNew: {} iter: {}".format(llOld, llNew, nIter))
                    return 
        return gmmStart

    def _LBG(self,X, iterations,nGaussian, alpha = 0.1):
        mu = mut.FromRowToColumn(X.mean(1))
        C=np.cov(X)
        psi = 0.01
        GMM_i = [(1.0, mu, C)]
        if(C.shape==()):
            C = C.reshape((C.size, 1))

        if(self.Mtype == "diagonal" or self.Mtype=="tied diagonal"):
            C = C*np.eye(C.shape[0])
        U, s, _ = np.linalg.svd(C)
        s[s<psi] = psi
        sigma = np.dot(U, mut.FromRowToColumn(s)*U.T)
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
                if(self.Mtype=="tied"):
                    GMM_i = self._GMM_EM(X, GMM_list)
                elif(self.Mtype == "diagonal"):
                    GMM_i = self._GMM_EM(X, GMM_list)
                elif(self.Mtype=="tied diagonal"):    
                    GMM_i = self._GMM_EM(X, GMM_list)
                else:
                    GMM_i = self._GMM_EM(X, GMM_list)
        return GMM_i

    def train(self,xTrain, xLabels):
        self.GMM_list = []
        D0 = xTrain[:, xLabels == 0]
        D1= xTrain[:,xLabels==1]
        k0= int(np.ceil(np.log2(self.nNonTarget)))
        k1= int(np.ceil(np.log2(self.nTarget)))
        GMM0 = self._LBG(D0,k0,self.nNonTarget)
        GMM1= self._LBG(D1,k1,self.nTarget)
        self.GMM_list.append(GMM0)
        self.GMM_list.append(GMM1)

    def trasform(self,xTest, xLabels):
        numClasses = len(np.unique(xLabels))
        self.ll=[]
        for i in range(numClasses):
            self.ll.append(self._GMM_ll_per_sample(xTest, self.GMM_list[i]))
        self.ll = np.array(self.ll)
        self.predicted=np.argmax(self.ll, axis=0)

    def get_scores(self):
        return self.ll

    def get_predicted(self):
        return self.predicted

    
    
