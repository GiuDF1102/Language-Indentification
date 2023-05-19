import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import scipy as sci
import math_utils as mut
import data_utils as du
import dimensionality_reduction as dr
import data_visualization as dv

class multivariate_cl:
    priors = []
    logSPost = []
    
    def __init__(self, priors=None):
        if priors is not None:
            self.priors = priors
    
    def fit(self,x,labels):
        m_estimates = []
        C_estimates = []
        loop_len = len(np.unique(labels))
        array_classes = np.empty(loop_len, dtype=float)

        for i in range(loop_len):
            class_count = len(x[:, labels==i][0])
            array_classes[i] = class_count
            m_ML = mut.calcmean(x[:,labels==i])
            C_ML = mut.cov_mat(x[:,labels==i], m_ML)
            m_estimates.append(m_ML.reshape(m_ML.shape[0], 1))
            C_estimates.append(C_ML)
            

        if len(self.priors) == 0:
            self.priors = array_classes/len(x[0])
            
        return m_estimates, C_estimates
    
    def trasform(self,x, mu, C):
        listLogJoint = []
        for i in range(len(mu)):
            listLogJoint.append(mut.log_gaussian_multivariate(x, mu[i], C[i])+np.log(self.priors[i]))
        logSJoint = np.array(listLogJoint)
        logSMarginal = mut.vrow(sci.special.logsumexp(logSJoint, axis=0))
        self.logSPost = logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)

    def get_posteriors(self):
        return self.logSPost

class tied_multivariate_cl:
    priors = []
    logSPost = []
    logSJoint = []
        
    def __init__(self, priors=None):
        if priors is not None:
            self.priors = priors
        
    def fit(self,x,labels):
        m_estimates = []
        C_estimates = []
        loop_len = len(np.unique(labels))
        array_classes = np.empty(loop_len, dtype=float)
        C = 0
        
        for i in range(loop_len):
            class_count = len(x[:, labels==i][0])
            array_classes[i] = class_count
            m_ML = mut.calcmean(x[:,labels==i])
            C_ML = mut.cov_mat(x[:,labels==i], m_ML)
            m_estimates.append(m_ML.reshape(m_ML.shape[0], 1))
            C_estimates.append(C_ML)
         
        for i in range(len(m_estimates)):
            C = C + x[:,labels==i].shape[1]*C_estimates[i]
            
        C_tied = (1/x.shape[1])*C
        
        if len(self.priors) == 0:
            self.priors = array_classes/len(x[0])
        
        return m_estimates, C_tied
        
    def trasform(self,x, mu, C):
        listLogJoint = []
    
        for i in range(len(mu)):
            listLogJoint.append(mut.log_gaussian_multivariate(x, mu[i], C)+np.log(self.priors[i]))
        self.logSJoint = np.array(listLogJoint)
        logSMarginal = mut.vrow(sci.special.logsumexp(self.logSJoint, axis=0))
        self.logSPost = self.logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)

    def get_posteriors(self):
        return self.logSPost
    
    def get_joint(self):
        return self.logSJoint
    

if __name__ == "__main__":
    labels, features = du.load("..\PROJECTS\Language_detection\Train.txt")

    labels_dict = {
        "Not-Italian": 0,
        "Italian": 1
    }

    features_dict_PCA = {
        "PC-0": 0,
        "PC-1": 1
    }


    """     
    DP = dr.PCA(features,1)
    mu, C = tied_multivariate_cl().fit(DP, labels)
    print(mu)
    print(C)
    #plot guassians with matplotlib
    X_0 = np.linspace(-100,100,1000)
    Y_0 = np.exp(mut.log_gaussian_univariate(X_0, C, mu[0]))
    Y_1 = np.exp(mut.log_gaussian_univariate(X_0, C, mu[1]))
    plt.plot(np.linspace(-10,10,1000), Y_0[0])
    plt.plot(np.linspace(-10,10,1000), Y_1[0])
    plt.show()
    """
 
    DP = dr.PCA(features,2)
    mu, C = multivariate_cl().fit(DP, labels)
    
    #plot guassians with matplotlib
    #contour plot examples
    feature_x = np.arange(-100,100,2)
    feature_y = np.arange(-100,100,2)
    [X,Y] = np.meshgrid(feature_x, feature_y)
    Z = mut.log_gaussian_multivariate(np.array([X.flatten(), Y.flatten()]), mu[0], C[0])
    print(Z)
    Z = X**2 + Y**2
    print(Z)
    plt.contour(X,Y,Z)
    plt.show()
    plt.close()    
    """
    plt.plot(np.linspace(-10,10,1000), Y_0[0])
    plt.plot(np.linspace(-10,10,1000), Y_1[0])
    plt.show() """