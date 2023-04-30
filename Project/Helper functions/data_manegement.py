import numpy as np
import scipy.linalg as sci
import matplotlib.pyplot as plt
import os 
import shutil

def load(file_name):
    file_path = file_name

    label_list = []
    features_list = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                splitted_line = line.removesuffix("\n").split(",")
                label_list.append(splitted_line.pop().lower())
                features_list.append([float(elem) for elem in splitted_line])
            except:
                pass                            

    return (np.array(label_list, dtype=int), np.array(features_list, dtype=float).T)


def get_hist(data, labels, map_classes, map_features):

    shutil.rmtree("histograms")
    os.makedirs("histograms")

    length = len(np.unique(labels))
    classes_list = []
    inv_map_class = {v: k for k, v in map_classes.items()}
    inv_map_feats = {v: k for k, v in map_features.items()}

    for i in range(length):
        classes_list.append(data[:, (labels == i)])

    for i in range(len(map_features)):
        plt.figure()
        plt.xlabel(inv_map_feats[i])
        for j in range(length):
            plt.hist(classes_list[j][i], bins = 10, density = True, alpha = 0.4, label = inv_map_class[j])
            plt.legend()
            plt.tight_layout()

        plt.savefig("histograms/Histogram {}.svg".format(inv_map_feats[i]))


def get_scatter(data, labels, map_classes, map_features):
    
    shutil.rmtree("scatter_plots")
    os.makedirs("scatter_plots")

    length_feat = len(map_features)
    length_class = len(map_classes)

    classes_list = []
    inv_map_class = {v: k for k, v in map_classes.items()}
    inv_map_feats = {v: k for k, v in map_features.items()}

    for i in range(length_feat):
        classes_list.append(data[:, (labels == i)])


    for i in range(length_feat):
        for j in range(length_feat):
            plt.figure()
            for k in range(length_class):
                if i != j:
                    plt.xlabel(inv_map_feats[i])
                    plt.ylabel(inv_map_feats[j])
                    plt.scatter(classes_list[k][i], classes_list[k][j], label = inv_map_class[k])
            if i != j:
                plt.legend()
                plt.tight_layout()
                plt.savefig("scatter_plots/Scatter Plot {} x {}.svg".format(inv_map_feats[i], inv_map_feats[j]))

def get_scatter_3d(Data,n_classes):
    classes_list = []
    for i in range(2):
        classes_list.append(Data[:, (labels == i)])
    #[classe][feature]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for i in range(n_classes):
         ax.scatter3D(classes_list[i][0], classes_list[i][1], classes_list[i][2], s=20)

    plt.show()

def FromRowToColumn(v):
    return v.reshape((v.size, 1))

def calcmean(D):
    return D.mean(1) #ritorno media sulle colonne

def PCA(D,m): #D=matrice con dati organizzati per colonne, m=dimensionalita' finale
     mu=D.mean(1) 
     
     mu=FromRowToColumn(mu)
    
     DC=D-mu

     C=np.dot(DC,DC.T)
     C=C/float(DC.shape[1]) 

     s,U=np.linalg.eigh(C)
     P=U[:,::-1][:,0:m]
    
     DP=np.dot(P.T, D)
     return DP

def LDA(Dataset,Labels,m):
    SB=0
    SW=0
    mu=FromRowToColumn(calcmean(Dataset))
    for i in range(2):
        DC1s=Dataset[:,Labels==i]
        muC1s=FromRowToColumn(DC1s.mean(1))
        SW+=np.dot(DC1s-muC1s,(DC1s-muC1s).T)
        SB+=DC1s.shape[1]*np.dot(muC1s-mu,(muC1s-mu).T)
    SW/=Dataset.shape[1]
    SB/=Dataset.shape[1]
    # print(SW)
    # print(SB)

    # risolvo problema generale agli autovalori
    s,U=sci.eigh(SB,SW)
    W=U[:,::-1][:,0:m]
    DP=np.dot(W.T,Dataset)
    return DP

if __name__=="__main__":
    labels, features = load("Train.txt")

    labels_dict = {
        "Not-Italian": 0,
        "Italian": 1
    }
    features_dict = {
        "elem 0": 0,
        "elem 1": 1,
        "elem 2": 2,
        "elem 3": 3,
        "elem 4": 4,
        "elem 5": 5        
    }
    features_dict_PCA = {
        "PCA elem 0": 0,
        "PCA elem 1": 1   
    }

    #get_hist(features,labels,labels_dict, features_dict)
    #get_scatter(features,labels,labels_dict, features_dict)
    DP = PCA(features,3)
    #DP = LDA(features,labels,3)
    get_scatter_3d(DP,2)
    get_scatter(DP,labels,labels_dict, features_dict_PCA)
    get_hist(DP,labels,labels_dict, features_dict_PCA)