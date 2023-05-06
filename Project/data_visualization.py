import matplotlib.pyplot as plt
import os 
import shutil
import numpy as np
import math_utils as mu
import seaborn as sns

def calc_correlation_matrix(D, name): #TODO:ottimizzare
    # if(os.path.exists("correlation_martices")):
    #     shutil.rmtree("correlation_martices")
    # os.makedirs("correlation_martices")

    mean = mu.calcmean(D)
    cov_matr = mu.dataset_cov_mat(D,mean)
    variance=cov_matr.diagonal()
    corr_matrix = np.zeros(cov_matr.shape)
    for i in range(cov_matr.shape[0]):
            for j in range(cov_matr.shape[1]):
                    corr_matrix[i][j] = cov_matr[i][j]/np.sqrt(variance[i]*variance[j])

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig("correlation_martices/correlation_martices {}.png".format(name))
    plt.close()
    return corr_matrix

def get_hist(data, labels, map_classes, map_features):

    if(os.path.exists("histograms")):
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
            plt.hist(classes_list[j][i], bins = 25, density = True, alpha = 0.4, label = inv_map_class[j])
            plt.legend()
            plt.tight_layout()

        plt.savefig("histograms/Histogram {}.png".format(inv_map_feats[i]))

def get_scatter(data, labels, map_classes, map_features):
    
    if(os.path.exists("scatter_plots")):
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
            
            plt.savefig("scatter_plots/Scatter Plot {} x {}.png".format(inv_map_feats[i], inv_map_feats[j]))

def get_scatter_3d(Data, n_classes, labels):
    classes_list = []
    for i in range(2):
        classes_list.append(Data[:, (labels == i)])
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for i in range(n_classes):
         ax.scatter3D(classes_list[i][0], classes_list[i][1], classes_list[i][2], s=20) #[classe][feature]

    plt.show()