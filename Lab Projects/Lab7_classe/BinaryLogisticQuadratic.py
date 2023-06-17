import numpy as np
import sklearn.datasets as sk
import scipy.optimize as opt

def fromRowToColumn(v):
    return v.reshape((v.size, 1))

def calc_accuracy(labels, predicted):
    #Needs two lists
    confronted = (labels == predicted)
    TP = 0
    for i in confronted:
        if(i == True):
            TP = TP + 1
    
    return TP/len(predicted)

def load_iris_without_setosa():
    D, L = sk.load_iris()['data'].T, sk.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 
    return D, L

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

# def transform(DTE, w, b):
#     labels = np.zeros(DTE.shape[1])
#     posteriors = np.dot(w.T,DTE)+b
#     labels[posteriors>0] = 1
#     labels[posteriors<=0] = 0
#     return posteriors, labels

class logReg():
    __b = None
    __w = None

    def __init__(self,l):
        self.__l=l

    def __logreg_obj(self,v):
        w= fromRowToColumn(v[0:self.__DTR.shape[0]])
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
        return self.__w, self.__b    

    def get_scores(self):
        return self.scores

def transform(w, b, data):
    scores = np.dot(w.T,data)+b
    return scores

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

def features_expansion(Dataset):#restituisce phi
    expansion = []
    for i in range(Dataset.shape[1]):
        vec = np.reshape(np.dot(fromRowToColumn(Dataset[:, i]), fromRowToColumn(Dataset[:, i]).T), (-1, 1), order='F')
        expansion.append(vec)
    return np.vstack((np.hstack(expansion), Dataset))

if __name__ == '__main__':
    D, L = load_iris_without_setosa()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    print("features number:", DTR.shape[0])
    
    logreg = logReg(0.000001)
    w,b = logreg.train(DTR, LTR)
    scoresLogReg = transform(w,b,DTE)

    pi = 0.1
    logregCal = logRegCalibration(scoresLogReg, LTE, pi)
    alpha, gamma = logregCal.train()
    print(alpha, gamma)
    calibrated_scores = get_calibrated_scores(scoresLogReg, alpha, gamma, pi)
    print(scoresLogReg)
    print(calibrated_scores)
    # labels = list(map(int,labels.flatten()))
    # print(1-calc_accuracy(LTE, labels))

    # """ test_x2 = phi_x(DTE.T) """
    # regressor=logReg(DTR,LTR,0.001)
    # x=regressor.train()
    # test = transform(DTE, x[0:4], x[-1])
