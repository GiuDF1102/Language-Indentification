import data_utils as du
import math_utils as mu
import numpy as np
import scipy.optimize as opt

class logReg():
    def __init__(self,D,L,l):
        self.DTR=D
        self.ZTR=L*2.0-1.0
        self.l=l
        self.dim=D.shape[0]

    def logreg_obj(self,v):
        w= mu.FromRowToColumn(v[0:self.DTR.shape[0]])
        b=v[-1]
        scores=np.dot(w.T,self.DTR)+b
        loss_per_sample=np.logaddexp(0,-self.ZTR*scores)
        loss=loss_per_sample.mean()+0.5*self.l*np.linalg.norm(w)**2
        return loss
    
    def train(self):
        x0=np.zeros(self.DTR.shape[0]+1)
        xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.logreg_obj,x0=x0,approx_grad=True)
        return xOpt[0:self.DTR.shape[0]], xOpt[-1]

def transform(DTE, w, b):
    posteriors = np.dot(w.T,DTE)+b
    posteriors[posteriors>0] = 1
    posteriors[posteriors<=0] = 0
    return posteriors
