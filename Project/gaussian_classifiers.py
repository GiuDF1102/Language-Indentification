import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import scipy as sci
import math_utils as mut

class multivariate_cl:
    priors = []
    logSPost = []
    logSJoint = None
    name = "Multivariate"
    m = None
    C = None
    
    def __init__(self, priors):
        self.priors = priors
    
    def train(self,x,labels):
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
                    
        self.m =  m_estimates
        self.C = C_estimates
        return
    
    def transform(self,x):
        listLogJoint = []
        for i in range(len(self.m)):
            listLogJoint.append(mut.log_gaussian_multivariate(x, self.m[i], self.C[i])+np.log(self.priors[i]))
        self.logSJoint = np.array(listLogJoint)
        logSMarginal = mut.vrow(sci.special.logsumexp(self.logSJoint, axis=0))
        self.logSPost = self.logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)

    def get_scores(self):
        #only if 2 classes
        return self.logSPost[1]-self.logSPost[0]
    
class naive_multivariate_cl:
    priors = []
    logSPost = []
    name  = "Naive Multivariate"
    mu= None
    C=None

    def __init__(self, priors):
        self.priors = priors
        
    def train(self,x,labels):
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
            
        self.mu= m_estimates
        self.C= C_estimates
        return

    def transform(self,x):
        listLogJoint = []
        for i in range(len(self.mu)):
            listLogJoint.append(mut.log_gaussian_multivariate(x, self.mu[i], self.C[i]*np.eye(self.C[i].shape[0]))+np.log(self.priors[i]))
        logSJoint = np.array(listLogJoint)
        logSMarginal = mut.vrow(sci.special.logsumexp(logSJoint, axis=0))
        self.logSPost = logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)

    def get_scores(self):
        #only if 2 classes
        return self.logSPost[1]-self.logSPost[0]

class tied_multivariate_cl:
    priors = []
    logSPost = []
    logSJoint = []
    name = "Tied Multivariate"
    mu=None
    C=None

    def __init__(self, priors):
        self.priors = priors

    def train(self,x,labels):
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
        
        self.mu=m_estimates
        self.C=C_tied
        return
        
    def transform(self,x):
        listLogJoint = []
    
        for i in range(len(self.mu)):
            listLogJoint.append(mut.log_gaussian_multivariate(x, self.mu[i], self.C)+np.log(self.priors[i]))
        self.logSJoint = np.array(listLogJoint)
        logSMarginal = mut.vrow(sci.special.logsumexp(self.logSJoint, axis=0))
        self.logSPost = self.logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)

    def get_scores(self):
        #only if 2 classes
        return self.logSPost[1]-self.logSPost[0]

class tied_naive_multivariate_cl:
    priors = []
    logSPost = []
    num_classes = 0
    logSJoint = None
    name = "Tied Naive Multivariate"
    m = None
    C = None
        
    def __init__(self, priors):
        self.priors = priors
        
    def train(self,x,labels):
        loop_len = len(np.unique(labels))
        m_estimates = []
        C_estimates = []
        array_classes = np.empty(loop_len, dtype=float)
        
        for i in range(loop_len):
            class_count = len(x[:, labels==i][0])
            array_classes[i] = class_count
            m_ML = mut.calcmean(x[:,labels==i])
            C_ML = mut.cov_mat(x[:,labels==i], m_ML)
            mu = m_ML.reshape(m_ML.shape[0], 1)
            m_estimates.append(mu)
            C_estimates.append(C_ML)

        self.C = np.zeros(C_estimates[0].shape)
        for i in range(len(m_estimates)):
            self.C = self.C + x[:,labels==i].shape[1]*C_estimates[i]
            
        C_tied = (1/x.shape[1])*self.C
        
        self.C = C_tied
        self.m = m_estimates
        return 
        
    def transform(self,x):
        listLogJoint = []
        for i in range(len(self.m)):
            listLogJoint.append(mut.log_gaussian_multivariate(x, self.m[i], self.C*np.eye(self.C.shape[0]))+np.log(self.priors[i]))
        self.logSJoint = np.array(listLogJoint)
        logSMarginal = mut.vrow(sci.special.logsumexp(self.logSJoint, axis=0))
        self.logSPost = self.logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)

    def get_scores(self):
        #only if 2 classes
        return self.logSPost[1]-self.logSPost[0]
