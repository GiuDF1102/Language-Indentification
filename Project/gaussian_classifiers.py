# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:37:59 2023

@author: giuli

About gaussian distributions and classifiers
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import scipy as sci
import math_utils as mut

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
    
class naive_multivariate_cl:
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
            listLogJoint.append(mut.log_gaussian_multivariate(x, mu[i], C[i]*np.eye(C[i].shape[0]))+np.log(self.priors[i]))
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

class tied_naive_multivariate_cl:
    priors = []
    logSPost = []
        
    def __init__(self, priors=None):
        if priors is not None:
            self.priors = priors
        
    def fit(self,x,labels):
        m_estimates = []
        C_estimates = []
        loop_len = len(np.unique(labels[0]))
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
            listLogJoint.append(mut.log_gaussian_multivariate(x, mu[i], C*np.eye(C.shape[0]))+np.log(self.priors[i]))
        logSJoint = np.array(listLogJoint)
        logSMarginal = mut.vrow(sci.special.logsumexp(logSJoint, axis=0))
        self.logSPost = logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)
    
    def get_posteriors(self):
        return self.logSPost

    
