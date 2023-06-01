import numpy as np
import sklearn.datasets as sk
import scipy.optimize as opt
from itertools import repeat

def calcErrorRate(lab, pre):
    confronted = lab == pre
    correct = np.sum(np.where(confronted == True, 1, 0))
    return (1-correct/len(lab))*100

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

def load_iris_without_setosa():
    D, L = sk.load_iris()['data'].T, sk.load_iris()['target']
    D = D[:, L != 0] #non prendo iris setosa
    L = L[L!=0]     #label virginica=0
    L[L==2] = 0    
    return D,L

def modifyLabel(trainLabels):
    return np.where(trainLabels==0,-1,1)

class SVM:

    __kernel = None
    __H = None
    scores = None
    __K = None
    __w = None
    __k12 = None
    __labels = None
    __data = None
    __alpha = None
    __az = None

    def __init__(self, kernel='linear'):
        self.__kernel = kernel

    def __modifyLabel(self, trainLabels):
        return np.where(trainLabels==0,-1,1)

    def __expandMatrix(self, K,Data):
        row_to_add=np.ones(Data.shape[1])*K
        return np.vstack([Data,row_to_add])

    def __calcG(self, expandedData):
        return np.dot(expandedData.T,expandedData)

    def __polinomialKernel(self, data1,data2,costant,degree,K,eps):
        return ((np.dot(data1.T,data2)+costant)**degree)+eps

    def __RBFKernel(self,data1,data2,gamma,eps):
        G = np.zeros((data1.shape[1],data2.shape[1]))
        for i in range(data1.shape[1]):
            for j in range(data2.shape[1]):
                G[i,j]=np.exp(-self.__gamma*(np.linalg.norm(data1[:, i]-data2[:, j])**2))+eps
        return G
        
    def __calcHLinear(self, data, labels):
        Z = self.__modifyLabel(labels)
        D = self.__expandMatrix(self.__K,data)
        G = self.__calcG(D)
        H = np.zeros(G.shape)
        for i in range(D.shape[1]):
            for j in range(D.shape[1]):
                H[i][j] = Z[i]*Z[j]*G[i][j]
        self.__H = H

    def __calcHWithQuadraticKernel(self,data1,data2,labels,costant,degree,K,eps):
        Z = self.__modifyLabel(labels)
        G = self.__polinomialKernel(data1,data2,costant,degree,K,eps)
        H = np.zeros(G.shape)
        for i in range(data1.shape[1]):
            for j in range(data2.shape[1]):
                H[i][j] = Z[i]*Z[j]*G[i][j]
        self.__H = H

    def __calcHWithRBFKernel(self,data1,data2,labels,gamma,eps):
        Z = self.__modifyLabel(labels)
        G = self.__RBFKernel(data1,data2,gamma,eps)
        H = np.zeros(G.shape)
        for i in range(data1.shape[1]):
            for j in range(data2.shape[1]):
                H[i][j] = Z[i]*Z[j]*G[i][j]
        self.__H = H

    def __J(self,alpha,H):
        grad=np.dot(H,alpha)-np.ones(H.shape[1])
        return (0.5*np.dot(alpha.T,np.dot(H,alpha))-np.dot(alpha,np.ones(H.shape[1])),grad)

    def __optGetWLinear(self, C, K, data, labels):
        alpha=np.zeros(data.shape[1])#stessa dim del numero si sample
        bounds = list(repeat((0, C), data.shape[1]))
        (alpha, f, dataopt)=opt.fmin_l_bfgs_b(self.__J, alpha, args=(self.__H,),bounds=bounds, factr=1.0)
        w = np.sum((alpha*self.__modifyLabel(labels)).reshape(self.__K, data.shape[1])*self.__expandMatrix(self.__K, data), axis=1)
        self.__w = w

    def __optGetWPolinomial(self, C, K, data, labels):
        alpha=np.zeros(data.shape[1])#stessa dim del numero si sample
        bounds = list(repeat((0, C), data.shape[1]))
        (alpha, f, dataopt)=opt.fmin_l_bfgs_b(self.__J, alpha, args=(self.__H,),bounds=bounds, factr=1.0)
        self.__az = (alpha*modifyLabel(labels)).reshape(1, data.shape[1])

    def __optGetWRBF(self, C, K, data, labels):
        alpha=np.zeros(data.shape[1])#stessa dim del numero si sample
        bounds = list(repeat((0, C), data.shape[1]))
        (alpha, f, dataopt)=opt.fmin_l_bfgs_b(self.__J, alpha, args=(self.__H,),bounds=bounds, factr=1.0)
        self.__az = (alpha*modifyLabel(labels)).reshape(1, data.shape[1])

    def train(self, data, labels, **kwargs):

        if len(kwargs) == 0:
            print("Please provide the arguments for {} SVM".format(self.__kernel))
            return

        if self.__kernel == 'linear':
            self.__K = kwargs['K']
            self.__calcHLinear(data, labels)
            self.__optGetWLinear(kwargs['C'], kwargs['K'], data, labels)
        
        elif self.__kernel == 'polinomial':
            self.__K = kwargs['K']
            self.__c = kwargs['c']
            self.__d = kwargs['d']
            self.__data = data
            eps = np.sqrt(self.__K)
            self.__calcHWithQuadraticKernel(data,data,labels,self.__c,self.__d,self.__K,eps)
            self.__optGetWPolinomial(kwargs['C'], self.__K, data, labels)

        elif self.__kernel == 'RBF':
            self.__K = kwargs['K']
            self.__gamma = kwargs['gamma']
            self.__data = data
            eps = np.sqrt(self.__K)
            self.__calcHWithRBFKernel(data,data,labels,self.__gamma,eps)
            self.__optGetWRBF(kwargs['C'], self.__K, data, labels)

    def trasform(self, dataTest):

        if self.__kernel == 'linear':
            self.scores = np.dot(self.__w.T, self.__expandMatrix(self.__K, dataTest))
            predicted = np.where(self.scores > 0, 1, -1)
            return predicted
        
        elif self.__kernel == 'polinomial':
            self.scores= np.sum(np.dot(self.__az,self.__polinomialKernel(self.__data,dataTest,self.__c,self.__d,self.__K, np.sqrt(self.__K))), axis=0)
            predicted = np.where(self.scores > 0, 1, -1)
            return predicted

        elif self.__kernel == 'RBF':
            self.scores= np.sum(np.dot(self.__az,self.__RBFKernel(self.__data,dataTest,self.__gamma, np.sqrt(self.__K))), axis=0)
            predicted = np.where(self.scores > 0, 1, -1)
            return predicted



if __name__ == '__main__':
    D, L = load_iris_without_setosa()
    (dataTrain, labelsTrain), (dataTest, labelsTest) = split_db_2to1(D, L)

    svmObjLinear = SVM('linear')
    svmObjLinear.train(dataTrain, labelsTrain, C=0.1, K=1)
    P = svmObjLinear.trasform(dataTest)
    errorRateLinear = calcErrorRate(modifyLabel(labelsTest), P)
    print("Linear error rate: {:.1f}".format(errorRateLinear))

    svmObjPolinomial = SVM('polinomial')
    svmObjPolinomial.train(dataTrain, labelsTrain, C=1, K=1, d=2, c=0)
    P = svmObjPolinomial.trasform(dataTest)
    errorRatePolinomial = calcErrorRate(modifyLabel(labelsTest), P)
    print("Polinomial error rate: {:.1f}".format(errorRatePolinomial))

    svmObjRBF = SVM('RBF')
    svmObjRBF.train(dataTrain, labelsTrain, C=1, K=0, gamma=10)
    P = svmObjRBF.trasform(dataTest)
    errorRateRBF = calcErrorRate(modifyLabel(labelsTest), P)
    print("RBF error rate: {:.1f}".format(errorRateRBF))

