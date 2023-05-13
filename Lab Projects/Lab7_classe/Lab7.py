import numpy as np
import scipy.optimize as opt
import sklearn as sk

def load_iris_without_setosa():
    D, L = sk.datasets.load_iris()['data'].T, sk.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 
    return D, L

def vcol(v):
    return v.reshape((v.size, 1))

def split_db_2to1(D, L, seed=0):#versicolor=1, virginica=0
    # 2/3 dei dati per il training----->100 per training, 50 per test
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)  # DTR= data training, LTR= Label training
    # DTE= Data test, LTE= label testing

class logReg():
    def __init__(self,D,L,l):
        self.DTR=D
        self.ZTR=L*2.0-1.0
        self.l=l
        self.dim=D.shape[0]

    def logreg_obj(self,v):
        w=vcol(v[0:self.dim])
        b=v[-1]
        scores=np.dot(w.T,self.DTR)+b
        loss_per_sample=np.logaddexp(0,-self.ZTR*scores)
        loss=loss_per_sample.mean()+0.5*self.l*np.linalg.norm(w)**2
        return loss
    
    def train(self):
        x0=np.zeros(self.DTR.shape[0]+1)
        xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.logreg_obj,x0=x0,approx_grad=True)
        w,b=vcol(xOpt[0:self.DTR.shape[0]]),xOpt[-1]
        return w,b

if __name__=="__main__":

    D,L=load_iris_without_setosa()
    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)
    logReg.__init__(DTR,LTR,1.0)
    


    