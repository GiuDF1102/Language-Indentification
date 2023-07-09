import numpy as np 
import multivariateGaussianGMM
import multivariateGaussian

def mcol(v):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return v.reshape((v.size, 1))
    
def constrainSigma(sigma, psi = 0.01):

    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, mcol(s)*U.T)
    return sigma

def Estep(logdens, S):
    # E-step: compute the POSTERIOR PROBABILITY (=responsibilities) for each component of the GMM
    # for each sample, using the previous estimate of the model parameters.
    return np.exp(S-logdens.reshape(1, logdens.size))


def Mstep(X, S, posterior):
    Zg = np.sum(posterior, axis=1)  # 3
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * np.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    mu = Fg / Zg
    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    for g in range(S.shape[0]):        
        cov[g] = constrainSigma(cov[g])
    w = Zg/np.sum(Zg)
    return (w, mu, cov)

def EMalgorithm(X, gmm, delta=10**(-6)):
    flag = True
    while(flag):
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        S = multivariateGaussianGMM.joint_log_density_GMM(multivariateGaussianGMM.logpdf_GMM(X, gmm), gmm)
        logmarg= multivariateGaussianGMM.marginal_density_GMM(multivariateGaussianGMM.joint_log_density_GMM(multivariateGaussianGMM.logpdf_GMM(X, gmm), gmm) )                                    #AGGIUSTARE
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = multivariateGaussianGMM.log_likelihood_GMM(logmarg, X)
        # ------ E-step ----------
        posterior = Estep(logmarg, S)
        # ------ M-step ----------
        (w, mu, cov) = Mstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        logmarg= multivariateGaussianGMM.marginal_density_GMM(multivariateGaussianGMM.joint_log_density_GMM(multivariateGaussianGMM.logpdf_GMM(X, gmm), gmm) )                                                                            #aggiustare
        loglikelihood2 = multivariateGaussianGMM.log_likelihood_GMM(logmarg, X)
        if (loglikelihood2-loglikelihood1 < delta):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    return gmm

def split(GMM, alpha = 0.1):
    
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = np.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    return splittedGMM

def LBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm(X, GMM)
    return GMM

def constrainSigma(sigma, psi = 0.01):

    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, mcol(s)*U.T)
    return sigma
        

if __name__ == "__main__":
    D0 = np.load("./GMM-data/GMM_data_4D.npy")
    GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), np.cov(D0).reshape((D0.shape[0], D0.shape[0])))]
    GMM0 = LBGalgorithm(GMM0_init, D0, 2)

    print("GMM0: ", GMM0)
