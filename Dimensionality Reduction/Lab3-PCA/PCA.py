import numpy
import matplotlib
import matplotlib.pyplot as plt

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

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = FromRowToColumn(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name] # assegno 0,1,2 in base al nome
                DList.append(attrs)
                labelsList.append(label) # mantengo ordine con gli indici tra attributi e label
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def plot_scatter(D, L):
    #filtro per separare le classi, e' un filtro sulle colonne
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx1 in range(2): # numero di grafi da tracciare,(??===m??)
        for dIdx2 in range(2): #numero di colonne dopo la riduzione di dimensionalita',===m
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

def PCA(D,m): #D=matrice con dati organizzati per colonne, m=dimensionalita' finale
     mu=D.mean(1) 
     
     mu=FromRowToColumn(mu)
    
     DC=D-mu

     C=numpy.dot(DC,DC.T)
     C=C/float(DC.shape[1]) 

     s,U=numpy.linalg.eigh(C)
     m=4
     P=U[:,::-1][:,0:m]
    
     DP=numpy.dot(P.T,Data)
     return DP

if __name__ == "__main__":

# struttura dati (vettore di vettori colonna)
#  EL1| EL2 | .... questa riga non conta
#  c1 | C1  | ..... dim:(4,150)
#  c2 | C2  | ....
#  C3 | C3  | .....
#  C4 | C4  | .....
     Data, Labels = load('iris.csv') #Attributi caricati in corrispondenza con l'indice al label associato(label espresso in 0,1,2) 
    #  print(Labels)
     mu=Data.mean(1) # media lungo le colonne, salvata come vettore riga
     # print(mu)
     mu=FromRowToColumn(mu)
     # print(mu)
     # centro il dataset, tolgo il valore medio ad ogni colonna(ogni componente di ogni elemento)
     DC=Data-mu
     # calcolo matrice delle covarianze
     C=numpy.dot(DC,DC.T)
     C=C/float(DC.shape[1]) #divido per il numero di campioni(punti) presenti nel dataset(ovvero ilnumero di colonne)
     #print(mu)
     #print(C)
     # calcolo autovalori(contenuti in s) e autovettori(colonne di U) di C
     s,U=numpy.linalg.eigh(C)#questa funzione vale solo per matrici quadrate e simmetrice, per matrici quadrate 
                             # e non simmetrice si usa numpy.linalg.eig() ma non ordina in ordine crescente autovalori e autovettori
                             # al contrario di numpy.linalg.eigh()
     #riduzione della dimensionalita', estraggo solo i primi m autovettori corrispondenti agli autovalori ordinati in oridne crescente
     m=2
     P=U[:,::-1][:,0:m]
     #proietto la matrice dei dati originari sugli autovettori estratti dal dataset e raccolti in P
     DP=numpy.dot(P.T,Data)
     plot_scatter(DP,Labels)
