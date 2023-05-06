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

def vrow(x):
    return x.reshape((1,x.shape[0]))

def dataset_mean(D):
    return D.mean(1)

def dataset_cov_mat(D,mu):
    DC = D - mu.reshape((mu.size, 1))
    return (1/D.shape[1])*np.dot(DC,DC.T)

def exp_gaussian_univariate(x, sigma, mu):
    """
    This function takes a value x and returns the value of the distribution.
    """
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-mu)**2/(2*sigma**2)))

def log_gaussian_univariate(x, sigma, mu):
    """
    This function takes a value x and returns the value of the distribution
    using the log multivariate formula. To avoid numerical issues due to 
    exponentiation of large numbers, in many practical cases it’s more 
    convenient to work with the logarithm of the density.
    """
    k_1 = -0.5*np.log(2*np.pi*sigma)
    k_2 = 0.5*np.log(1/sigma)
    k_3 = -(x-mu)**2/(2*(sigma**2))
    return np.exp(k_1+k_2+k_3)

def log_gaussian_multivariate(x, mu, C):
    M = x.shape[0]
    k_1 = (M*0.5)*np.log(2*np.pi)

    _,log_C = np.linalg.slogdet(C)
    k_2 = 0.5*log_C
    
    C_inv = np.linalg.inv(C)
    x_m = x - mu
    k_3 = 0.5*(x_m*np.dot(C_inv,x_m))
    
    return -k_1-k_2-k_3.sum(0)

def log_likelihood(x,m_ML,C_ML):
    return np.sum(log_gaussian_multivariate(x,m_ML,C_ML))

def trasform_multivariate_exp_gaussian_cl(x, mu, C, priors):
    listJoint = []
    for i in range(len(mu)):
        listJoint.append(np.exp(log_gaussian_multivariate(x, mu[i], C[i]))*priors[i])
    SJoint = np.array(listJoint)
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint/SMarginal
    return np.argmax(SPost, 0)

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
            m_ML = dataset_mean(DTR[:,LTR==i])
            C_ML = dataset_cov_mat(DTR[:,LTR==i], m_ML)
            m_estimates.append(m_ML.reshape(m_ML.shape[0], 1))
            C_estimates.append(C_ML)

        if len(self.priors) == 0:
            self.priors = array_classes/len(x[0])
            
        return m_estimates, C_estimates
    
    def trasform(self,x, mu, C):
        listLogJoint = []
        for i in range(len(mu)):
            listLogJoint.append(log_gaussian_multivariate(x, mu[i], C[i])+np.log(self.priors[i]))
        logSJoint = np.array(listLogJoint)
        logSMarginal = vrow(sci.special.logsumexp(logSJoint, axis=0))
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
            m_ML = dataset_mean(DTR[:,LTR==i])
            C_ML = dataset_cov_mat(DTR[:,LTR==i], m_ML)
            m_estimates.append(m_ML.reshape(m_ML.shape[0], 1))
            C_estimates.append(C_ML)

        if len(self.priors) == 0:
            self.priors = array_classes/len(x[0])
            
        return m_estimates, C_estimates

    def trasform(self,x, mu, C):
        listLogJoint = []
        for i in range(len(mu)):
            listLogJoint.append(log_gaussian_multivariate(x, mu[i], C[i]*np.eye(C[i].shape[0]))+np.log(self.priors[i]))
        logSJoint = np.array(listLogJoint)
        logSMarginal = vrow(sci.special.logsumexp(logSJoint, axis=0))
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
            m_ML = dataset_mean(x[:,labels==i])
            C_ML = dataset_cov_mat(x[:,labels==i], m_ML)
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
            listLogJoint.append(log_gaussian_multivariate(x, mu[i], C)+np.log(self.priors[i]))
        self.logSJoint = np.array(listLogJoint)
        logSMarginal = vrow(sci.special.logsumexp(self.logSJoint, axis=0))
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
        loop_len = len(np.unique(labels))
        array_classes = np.empty(loop_len, dtype=float)
        C = 0
        
        for i in range(loop_len):
            class_count = len(x[:, labels==i][0])
            array_classes[i] = class_count
            m_ML = dataset_mean(x[:,labels==i])
            C_ML = dataset_cov_mat(x[:,labels==i], m_ML)
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
            listLogJoint.append(log_gaussian_multivariate(x, mu[i], C*np.eye(C.shape[0]))+np.log(self.priors[i]))
        logSJoint = np.array(listLogJoint)
        logSMarginal = vrow(sci.special.logsumexp(logSJoint, axis=0))
        self.logSPost = logSJoint - logSMarginal
        return np.argmax(self.logSPost, axis=0)
    
    def get_posteriors(self):
        return self.logSPost

    
