import matplotlib.pyplot as plt
import os 
import shutil
import numpy as np
import math_utils as mu
import seaborn as sns

def calc_correlation_matrix(D, name):
    mean = mu.calcmean(D)
    cov_matr = mu.cov_mat(D,mean)
    variance=cov_matr.diagonal()
    corr_matrix = np.zeros(cov_matr.shape)
    for i in range(cov_matr.shape[0]):
            for j in range(cov_matr.shape[1]):
                    corr_matrix[i][j] = cov_matr[i][j]/np.sqrt(variance[i]*variance[j])

    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='Greys', cbar=False, square=True)
    plt.title(f"{name} correlation matrix")
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("correlation_matrice {}.svg".format(name), bbox_inches='tight')
    plt.close()
    return corr_matrix

def get_hist(data, labels, map_classes, map_features):

    shutil.rmtree("histograms")
    os.makedirs("histograms")

    num_classes = len(np.unique(labels))
    num_histograms = len(map_features)
    classes_list = []
    inv_map_class = {v: k for k, v in map_classes.items()}
    inv_map_feats = {v: k for k, v in map_features.items()}

    for i in range(num_classes):
        classes_list.append(data[:, (labels == i)])

    for i in range(num_histograms):
        plt.figure()
        plt.xlabel(inv_map_feats[i])
        for j in range(num_classes):
            plt.hist(classes_list[j][i], bins = 10, density = True, alpha = 0.4, label = inv_map_class[j])
            plt.legend()
            plt.tight_layout()

        plt.savefig("histograms/Histogram {}.svg".format(inv_map_feats[i]))
    plt.close()

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
                plt.savefig("scatter_plots/Scatter Plot {} x {}.svg".format(inv_map_feats[i], inv_map_feats[j]))
            plt.close()

def get_scatter_total(data, labels, map_classes, map_features):
    #plots all scatter plots in one figure
    if(os.path.exists("scatter_plots")):
        shutil.rmtree("scatter_plots")
    os.makedirs("scatter_plots")

    length_feat = len(map_features)
    length_class = len(map_classes)

    classes_list = []
    inv_map_class = {v: k for k, v in map_classes.items()}
    inv_map_feats = {v: k for k, v in map_features.items()}
    fig, axs = plt.subplots(length_feat, length_feat, figsize=(30,30))
    for i in range(length_feat):
        classes_list.append(data[:, (labels == i)])

    for i in range(length_feat):
        for j in range(length_feat):
            for k in range(length_class):
                if i != j:
                    axs[i,j].scatter(classes_list[k][i], classes_list[k][j], label = inv_map_class[k], rasterized=True)
                    axs[i,j].set_xlabel(inv_map_feats[i])
                    axs[i,j].set_ylabel(inv_map_feats[j])
                else:
                    axs[i,j].hist(classes_list[k][i], bins = 10, density = True, alpha = 0.4, label = inv_map_class[k], rasterized=True)
                    axs[i,j].set_xlabel(inv_map_feats[i])
                    axs[i,j].set_ylabel(inv_map_feats[j])
    plt.tight_layout()
    plt.savefig("scatter_plots/Scatter Plot Total.svg", dpi=80)
    plt.close()


def get_scatter_3d(Data, n_classes, labels):
    classes_list = []
    for i in range(2):
        classes_list.append(Data[:, (labels == i)])
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for i in range(n_classes):
         ax.scatter3D(classes_list[i][0], classes_list[i][1], classes_list[i][2], s=20) #[classe][feature]

    plt.show()
    plt.close()

def plotCPrim(x, y, labels, xlabel, name):
    plt.figure()
    for index,track in enumerate(y):
        plt.plot(x, track, label=labels[index])
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(labels)
    plt.xlabel(xlabel)
    plt.ylabel("min Cprim")
    plt.tight_layout()
    plt.savefig("{}.svg".format(name))
    return