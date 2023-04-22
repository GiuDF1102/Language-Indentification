import scipy
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as sci

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx1 in range(2): #m
        for dIdx2 in range(2): #m
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Setosa')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Versicolor')
            plt.scatter(D2[dIdx1, :], D2[dIdx2, :], label = 'Virginica')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()

def FromRowToColumn(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname,"r") as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = FromRowToColumn(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def calcmean(D):
    return D.mean(1) #ritorno media sulle colonne

def covariance(D):
    mu=calcmean(D)
     
    mu=FromRowToColumn(mu)
    
    DC=D-mu

    C=numpy.dot(DC,DC.T)
    C=C/float(DC.shape[1]) # media pesata su il numero di elementi per classe
    return C

def calcSB(D,muDataset):
    MUc=calcmean(D)
    MUc=FromRowToColumn(MUc)
    nc=D.shape[1]
    return nc*(numpy.dot((MUc-muDataset),(MUc-muDataset).T))

if __name__== "__main__":

    Dataset,Labels=load("iris.csv")
   
    SB=0
    SW=0
    mu=FromRowToColumn(calcmean(Dataset))
    for i in range(Labels.max()+1):
        DC1s=Dataset[:,Labels==i]
        muC1s=FromRowToColumn(DC1s.mean(1))
        SW+=numpy.dot(DC1s-muC1s,(DC1s-muC1s).T)
        SB+=DC1s.shape[1]*numpy.dot(muC1s-mu,(muC1s-mu).T)
    SW/=Dataset.shape[1]
    SB/=Dataset.shape[1]
    print(SW)
    print(SB)

    # risolvo problema generale agli autovalori
    s,U=sci.eigh(SB,SW)
    W=U[:,::-1][:,0:2]
    DP=numpy.dot(W.T,Dataset)
    plot_scatter(DP,Labels)