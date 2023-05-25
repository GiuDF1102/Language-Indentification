import numpy
import scipy as sci
import scipy.optimize as sciopt
import sklearn.datasets as sk

def load_iris_without_setosa():
    D, L = sk.load_iris()['data'].T, sk.load_iris()['target']
    D = D[:, L != 0] #non prendo iris setosa
    L = L[L!=0]     #label virginica=0
    L[L==2] = 0    
    return D,L

def split_db_2to1(D, L, seed=0):#versicolor=1, virginica=0
    # 2/3 dei dati per il training----->100 per training, 50 per test
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)  # DTR= data training, LTR= Label training
    # DTE= Data test, LTE= label testing

def J(params,*args):#args={DTR,LTR,l}
    w=params[0]
    b=params[1]
    if(args[1]==0):
         z=-1
    else:
         z=1
    first_term=(args[2]/2)*(numpy.linalg.norm(w, ord=2)**2)
    for i in range(args[0].shape[1]):
         xi=DTR[:,i]#vettore con feature di ogni elemento
         second_term+=z*numpy.logaddexp(0,-z*(w.T*xi+b))
    second_term= second_term/args[0].shape[1]
    return first_term+second_term



def logreg_obj(v,DTR,LTR,l):#l=lambda,v=(D+1,)
    w,b=v[0:-1],v[-1]
    params=numpy.array([w,b])
    x,o,d=sciopt.fmin_l_bfgs_b(J(params),args={DTR,LTR,l},approx_grad=True)
    return x

if __name__=="__main__":
    #parte di calcolo numerico del gradiente
    coord=numpy.array([0,0]) #1-D numpy array
    """ print(f(coord))
    print(l_bfgs_with_approx_grad(coord))
    print(calc_gradient(coord)) """
    # inizio regressione logistica
    D,L=load_iris_without_setosa()
    (DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)
    logreg_obj(DTR,LTR,10**-6)

