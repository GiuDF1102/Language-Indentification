import numpy as np
import scipy.optimize as opt

class logRegCalibration():
    def __init__(self,D,L, pi):
        self.DTR=D
        self.zi = L*2.0-1.0
        self.dim=D.shape[0]
        self.nt = L[L==1].shape[0]
        self.nf = L[L==0].shape[0]
        self.pi = pi

    def logreg_obj(self, v):
        alpha = v[0]
        gamma = v[1]
        exp = - np.dot(self.zi,(alpha*self.DTR.T + gamma + np.log(self.pi/(1-self.pi))))
        wi = np.where(self.zi == 1, self.pi/self.nt, (1-self.pi)/self.nf)
        loss = np.dot(wi,np.logaddexp(0, exp).T).sum(axis=0)
        return loss
    
    def train(self):
        x0=np.zeros(2)
        xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.logreg_obj,x0=x0,approx_grad=True)
        return xOpt
    
def get_calibrated_scores(scores, alpha, gamma, pi):
    return alpha*scores + gamma
