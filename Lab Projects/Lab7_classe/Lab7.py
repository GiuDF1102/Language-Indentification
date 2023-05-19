import numpy as np
import scipy.optimize as opt
import sklearn as sk

def fromRowToColumn(v):
    return v.reshape((v.size, 1))

def fromColumnToRow(v):
    return v.reshape((1,v.size))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = fromRowToColumn(np.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

def load_iris_without_setosa():
    D, L = sk.datasets.load_iris()['data'].T, sk.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 
    return D, L



def FromVectorToMatrix(V,ncol):
    return V.reshape((V.size, ncol))

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
        

class MulticlassLogisticRegression:
    
    def __init__(self, num_classes, DTR, LTR,lamb):
        self.num_classes = num_classes
        """self.weights = []
        self.bias = [] """
        self.DTR = DTR
        self.LTR=LTR
        self.lamb = lamb

    def _softmax(self, Z):
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def _one_hot(self, y):
        one_hot_y = np.zeros((self.num_classes, y.shape[0]))
        for i in range(y.shape[0]):
            one_hot_y[y[i], i] = 1
        return one_hot_y

    def logreg_obj(self,v):
        W=fromColumnToRow(v[0:self.DTR.shape[0]])#traspongo gia il vettore non c'e bisognodi fare W.T in riga 105
        b=v[-1]
        S = np.dot(W, self.DTR) + b
        sum_log_exp = np.log(np.exp(S).sum(axis=0))
        log = S - sum_log_exp
        one_hot_y = np.zeros((self.num_classes, self.LTR.shape[0]))

        for i in range(self.LTR.shape[0]):
            one_hot_y[self.LTR[i], i] = 1

        secondo_termine = (1/self.DTR.shape[0])*np.dot(one_hot_y, log.T)# traspongo perchè altrimenti non funziona
        norm = (self.lamb/2)*np.linalg.norm(W)**2
        return norm - secondo_termine

    def train(self):
        x0=np.zeros(self.DTR.shape[0]+1)
        xOpt,fOpt,d=opt.fmin_l_bfgs_b(self.logreg_obj,x0=x0,approx_grad=True)
        w,b=fromColumnToRow(xOpt[0:self.DTR.shape[0]]),xOpt[-1]
        return w,b


    
    # def predict(self, X):
    #     Z = np.dot(self.weights, X.T) + self.bias
    #     A = self._softmax(Z)
    #     return np.argmax(A, axis=0)

        
    

if __name__=="__main__":

    """    D,L=load_iris_without_setosa()
    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)
    logReg.__init__(DTR,LTR,1.0)
    """
    D,L=load("iris.csv")
    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)


    MLR = MulticlassLogisticRegression(3,DTR,LTR,1)
    print(MLR.train())