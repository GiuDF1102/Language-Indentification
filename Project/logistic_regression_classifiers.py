import data_utils as du
import math_utils as mu
import numpy as np
import scipy.optimize as opt

class logReg():
    __b = None
    __w = None

    def __init__(self,l):
        self.__l=l

    def __logreg_obj(self,v):
        w= mu.FromRowToColumn(v[0:self.__DTR.shape[0]])
        b=v[-1]
        scores=np.dot(w.T,self.__DTR)+b
        loss_per_sample=np.logaddexp(0,-self.__ZTR*scores)
        loss=loss_per_sample.mean()+0.5*self.__l*np.linalg.norm(w)**2
        return loss
    
    def train(self, data, labels):
        self.__DTR=data
        self.__ZTR=labels*2.0-1.0
        x0=np.zeros(self.__DTR.shape[0]+1)
        xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.__logreg_obj,x0=x0,approx_grad=True)
        self.__w = xOpt[0:self.__DTR.shape[0]]
        self.__b = xOpt[-1]    

    def transform(self, DTE):
        self.scores = np.dot(self.__w.T,DTE)+self.__b
        labels = np.where(self.scores>0, 1,0)
        return labels

    def get_scores(self):
        return self.scores
