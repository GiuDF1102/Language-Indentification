import data_utils as du
import math_utils as mu
import numpy as np
import scipy.optimize as opt

class logReg():
    __b = None
    __w = None

    def __init__(self,l, pi, balanced):
        self.__l=l
        self.__pi=pi
        self.__balanced=balanced

    def __logreg_obj(self,v):
        w= mu.FromRowToColumn(v[0:self.__DTR.shape[0]])
        b=v[-1]
        scores=np.dot(w.T,self.__DTR)+b
        loss_per_sample=np.logaddexp(0,-self.__ZTR*scores)
        loss=loss_per_sample.mean()+0.5*self.__l*np.linalg.norm(w)**2
        return loss

    def __logreg_obj_balanced(self,v):
        w= mu.FromRowToColumn(v[0:self.__DTR.shape[0]])
        b=v[-1]
        norm_elem = 0.5*self.__l*np.linalg.norm(w)**2
        DTR_0 = self.__DTR.T[self.__ZTR==-1]
        DTR_1 = self.__DTR.T[self.__ZTR==1]
        ZTR_0 = self.__ZTR[self.__ZTR==-1]
        ZTR_1=self.__ZTR[self.__ZTR==1]
        scores_0=np.dot(w.T,DTR_0.T)+b
        scores_1=np.dot(w.T,DTR_1.T)+b
        loss_per_sample_0=np.logaddexp(0,-ZTR_0*scores_0)
        loss_per_sample_1=np.logaddexp(0,-ZTR_1*scores_1)
        loss_0 = (self.__pi/DTR_0.shape[1])*loss_per_sample_0.sum()
        loss_1 = ((1-self.__pi)/DTR_1.shape[1])*loss_per_sample_1.sum()
        loss=loss_0+loss_1+norm_elem
        return loss    

    def __logreg_obj_calibration(self, v):
        alpha = v[0]
        gamma = v[1]
        exp = - np.dot(self.zi,(alpha*self.DTR.T + gamma + np.log(self.pi/(1-self.pi))))
        wi = np.where(self.zi == 1, self.pi/self.nt, (1-self.pi)/self.nf)
        loss = np.dot(wi,np.logaddexp(0, exp).T).sum(axis=0)
        return loss
    
    def train(self, data, labels):
        if self.__balanced == False:
            self.__DTR=data
            self.__ZTR=labels*2.0-1.0
            x0=np.zeros(self.__DTR.shape[0]+1)
            xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.__logreg_obj,x0=x0,approx_grad=True)
            self.__w = xOpt[0:self.__DTR.shape[0]]
            self.__b = xOpt[-1]
        else:    
            self.__DTR=data
            self.__ZTR=labels*2.0-1.0
            x0=np.zeros(self.__DTR.shape[0]+1)
            xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.__logreg_obj_balanced,x0=x0,approx_grad=True)
            self.__w = xOpt[0:self.__DTR.shape[0]]
            self.__b = xOpt[-1]

    def transform(self, DTE):
        self.scores = np.dot(self.__w.T,DTE)+self.__b
        labels = np.where(self.scores>0, 1,0)
        return labels

    def get_scores(self):
        return self.scores
