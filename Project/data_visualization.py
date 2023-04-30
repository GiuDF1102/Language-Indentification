import matplotlib.pyplot as plt
import os 
import shutil
import numpy as np

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