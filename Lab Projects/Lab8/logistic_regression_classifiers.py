import data_utils as du
import math_utils as mu
import numpy as np
import scipy.optimize as opt

class logReg():
    __b = None
    __w = None

    def __init__(self,l, pi, mode):
        self.__l=l
        self.__pi=pi
        self.__mode=mode

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
        exp = - np.dot(self.zi,(alpha*self.__DTR.T + gamma + np.log(self.__pi/(1-self.__pi))))
        wi = np.where(self.zi == 1, self.__pi/self.nt, (1-self.__pi)/self.nf)
        loss = np.dot(wi,np.logaddexp(0, exp).T).sum(axis=0)
        return loss

    def logreg_obj_wrap(self,dtr, ltr, lambda_r, prior=-1):
        def logreg_obj(v):
            w, b = mu.FromRowToColumn(v[0:-1]), v[-1]
            return self.J(w, b, dtr, ltr, lambda_r, prior)
        return logreg_obj

    def J(self,w, b, DTR, LTR, lambda_r, prior):
        z = (LTR*2) - 1
        norm_term = 0.5 * lambda_r * (np.linalg.norm(w) ** 2)
        if prior >= 0:
            c1 = ((prior) / (LTR[LTR == 1].shape[0])) * np.logaddexp(0, -1*z[z==1]*(np.dot(w.T, DTR[:, LTR == 1])+b)).sum()
            c0 = ((1-prior) / (LTR[LTR == 0].shape[0])) * np.logaddexp(0, -1*z[z==-1]*(np.dot(w.T, DTR[:, LTR == 0])+b)).sum()
            return norm_term + c1 + c0
        else:
            c = (LTR.shape[0] ** -1) * np.logaddexp(0, -1*z*(np.dot(w.T, DTR)+b)).sum()
            return norm_term + c

    def weighted_logreg_obj_wrap(self,DTR, LTR, l, pi=0.5):
        M = DTR.shape[0]
        Z = LTR * 2.0 - 1.0

        def logreg_obj(v):
            w = mcol(v[0:M])
            b = v[-1]
            reg = 0.5 * l * numpy.linalg.norm(w) ** 2
            s = (numpy.dot(w.T, DTR) + b).ravel()
            nt = DTR[:, LTR == 0].shape[1]
            avg_risk_0 = (numpy.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
            avg_risk_1 = (numpy.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
            return reg + (pi / nt) * avg_risk_1 + (1-pi) / (DTR.shape[1]-nt) * avg_risk_0
        return logreg_obj

    def train(self, data, labels):
        if self.__mode == 'unbalanced':
            self.__DTR=data
            self.__ZTR=labels*2.0-1.0
            x0=np.zeros(self.__DTR.shape[0]+1)
            xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.__logreg_obj,x0=x0,approx_grad=True)
            self.__w = xOpt[0:self.__DTR.shape[0]]
            self.__b = xOpt[-1]
        elif self.__mode == 'calibration':
            self.__DTR=data
            self.__ZTR=labels*2.0-1.0
            self.nt = np.sum(self.__ZTR == 1)
            self.nf = np.sum(self.__ZTR == -1)
            self.zi = np.where(self.__ZTR == 1, 1, 0)
            x0 = np.zeros(2)
            xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.__logreg_obj_calibration,x0=x0,approx_grad=True)
            self.alpha = xOpt[0]
            self.gamma = xOpt[-1]
        elif self.__mode == 'balanced':
            self.__DTR=data
            self.__ZTR=labels*2.0-1.0
            x0=np.zeros(self.__DTR.shape[0]+1)
            logreg_obj = self.logreg_obj_wrap(data, labels, self.__l, self.__pi)
            xOpt,fOpt,d=opt.fmin_l_bfgs_b(logreg_obj,x0=x0,approx_grad=True)
            self.__w = xOpt[0:self.__DTR.shape[0]]
            self.__b = xOpt[-1]
        elif self.__mode == 'calibrated2':
            DTR = data
            LTR = labels
            logreg_obj = self.weighted_logreg_obj_wrap(numpy.array(DTR), LTR, l)
            _v, _J, _d = opt.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
            _w = _v[0:DTR.shape[0]]
            _b = _v[-1]
            calibration = 0 if pi is None else numpy.log(pi / (1 - pi))
            STE = numpy.dot(_w.T, DTE) + _b - calibration
            return STE, _w, _b
        else:
            print("---------------> error, mode:", self.__mode)

    def transform(self, DTE):
        self.scores = np.dot(self.__w.T,DTE)+self.__b
        labels = np.where(self.scores>0, 1,0)
        return labels

    def get_calibrated_scores(self, DTE):
        self.scores = np.dot(self.alpha.T,DTE)+self.gamma
        return self.scores
        
    def get_scores(self):
        return self.scores

    def get_params(self):
        return self.__w, self.__b

    def compute_scores(self, DTE):
        return np.dot(self.__w.T,DTE)+self.__b